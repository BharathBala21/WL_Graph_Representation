"""Graph neural network baselines (GCN, GIN, GAT) on the same leaf feature graphs as WL.

Every leaf is the same 38-node graph that the WL kernel uses (nodes = the 38 measurements, edges from
the chosen graph construction, fitted on the training fold only). The nodes only differ between leaves
in their values, exactly as in WL, so the comparison isolates "WL kernel + SVM" vs "learned message
passing".

Node input (per leaf, per node)
  * the measurement, standardised on the training fold (clipped to +-5)
  * its Low / Medium / High quantile bin (one-hot), the same bins WL uses
  * a learned embedding of WHICH measurement the node is (WL keeps this identity with `feature::level`)

Architectures (dense implementation; the graph is tiny, so no sparse library is needed)
  * GCN  (Kipf & Welling, 2017):   H' = ReLU( D^-1/2 (A+I) D^-1/2  H W )
  * GIN  (Xu et al., 2019):        H' = MLP( (1+eps) H + A H )      -- provably as expressive as the WL test
  * GAT  (Velickovic et al., 2018): attention over each node's neighbours (and itself), 4 heads
Readout: mean and max over the nodes after every layer (jumping knowledge), then a small MLP.

Training: class-balanced cross-entropy (like class_weight='balanced' everywhere else), Adam, gradient clipping,
up to `epochs` epochs (at least `min_epochs`) with early stopping on the macro-F1 of a stratified 10% split
of the TRAINING fold; the best epoch's weights are kept.
One CPU thread, like every other model in the benchmark, or a GPU when one is available (Colab).
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler

from app.wl import GRAPHS, N

torch.set_num_threads(1)
# GPU when available (e.g. Colab); GNN_DEVICE=cpu forces the CPU. Other models always run on the CPU.
DEVICE = 'cuda' if torch.cuda.is_available() and os.environ.get('GNN_DEVICE', 'auto') != 'cpu' else 'cpu'

ARCHS = ('gcn', 'gin', 'gat')
ARCH_NAMES = {'gcn': 'GCN', 'gin': 'GIN', 'gat': 'GAT'}


def adjacency(graph: str, X: np.ndarray) -> np.ndarray:
    """Boolean N x N adjacency without self-loops; graph='none' gives no edges (a set of nodes)."""
    if graph == 'none':
        return np.zeros((N, N), bool)
    A = np.asarray(GRAPHS[graph](X), bool).copy()
    np.fill_diagonal(A, False)
    return A | A.T


# ---------------------------------------------------------------- layers
class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out, A):
        super().__init__()
        At = torch.tensor(A, dtype=torch.float32) + torch.eye(N)
        dinv = At.sum(1).pow(-0.5)
        self.register_buffer('A_hat', dinv[:, None] * At * dinv[None, :])
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, H):
        return self.lin(self.A_hat @ H)


class GINLayer(nn.Module):
    def __init__(self, d_in, d_out, A):
        super().__init__()
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU(), nn.Linear(d_out, d_out))

    def forward(self, H):
        return self.mlp((1 + self.eps) * H + self.A @ H)


class GATLayer(nn.Module):
    def __init__(self, d_in, d_out, A, heads=4):
        super().__init__()
        assert d_out % heads == 0
        self.heads, self.dh = heads, d_out // heads
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.register_buffer('mask', torch.tensor(A, dtype=torch.bool) | torch.eye(N, dtype=torch.bool))
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, H):                                   # H: B x N x d_in
        B = H.shape[0]
        Wh = self.W(H).view(B, N, self.heads, self.dh)      # B x N x h x dh
        s = (Wh * self.a_src).sum(-1)                       # B x N x h   (as source j)
        t = (Wh * self.a_dst).sum(-1)                       # B x N x h   (as target i)
        e = F.leaky_relu(t.unsqueeze(2) + s.unsqueeze(1), 0.2)  # B x N(i) x N(j) x h
        e = e.masked_fill(~self.mask[None, :, :, None], float('-inf'))
        att = torch.softmax(e, dim=2)
        out = torch.einsum('bijh,bjhd->bihd', att, Wh).reshape(B, N, -1)
        return out + self.bias


class GraphNet(nn.Module):
    def __init__(self, arch, A, d_in, hidden=64, layers=3, dropout=0.2, id_dim=8, n_classes=2):
        super().__init__()
        self.node_id = nn.Embedding(N, id_dim)
        self.inp = nn.Linear(d_in + id_dim, hidden)
        make = {'gcn': GCNLayer, 'gin': GINLayer, 'gat': GATLayer}[arch]
        self.convs = nn.ModuleList([make(hidden, hidden, A) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden * (layers + 1), hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x):                                   # x: B x N x d_in
        ids = self.node_id.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        H = F.relu(self.inp(torch.cat([x, ids], -1)))
        reads = [torch.cat([H.mean(1), H.amax(1)], -1)]
        for conv, norm in zip(self.convs, self.norms):
            H = self.drop(F.relu(norm(conv(H)))) + H          # residual keeps deep layers trainable
            reads.append(torch.cat([H.mean(1), H.amax(1)], -1))
        return self.head(torch.cat(reads, -1))


# ---------------------------------------------------------------- scikit-learn estimator
class GNNClassifier(BaseEstimator, ClassifierMixin):
    """GCN / GIN / GAT graph classifier on the 38-node leaf graph; fit(X, y) with X = the 38 measurements."""

    def __init__(self, arch='gin', graph='hybrid', layers=3, hidden=32, dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 epochs=100, patience=20, min_epochs=20, batch_size=64, seed=42):
        self.arch, self.graph, self.layers, self.hidden, self.dropout = arch, graph, layers, hidden, dropout
        self.lr, self.weight_decay, self.epochs, self.patience = lr, weight_decay, epochs, patience
        self.min_epochs = min_epochs
        self.batch_size, self.seed = batch_size, seed

    # node inputs: [z-score, one-hot Low/Medium/High]
    def _nodes(self, X):
        z = np.clip(self.scaler_.transform(X), -5, 5)
        b = self.disc_.transform(X).astype(int)
        onehot = np.eye(3, dtype=np.float32)[b]             # n x N x 3
        return torch.tensor(np.concatenate([z[..., None], onehot], -1), dtype=torch.float32)

    def fit(self, X, y):
        X = np.asarray(X, float)
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        self.enc_ = LabelEncoder().fit(y)
        self.classes_ = self.enc_.classes_
        yi = self.enc_.transform(y)
        self.scaler_ = StandardScaler().fit(X)
        self.disc_ = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', subsample=None).fit(X)
        self.A_ = adjacency(self.graph, X)

        tr, va = train_test_split(np.arange(len(X)), test_size=0.1, stratify=yi, random_state=self.seed)
        self.device_ = DEVICE
        Xn, yt = self._nodes(X).to(DEVICE), torch.tensor(yi, dtype=torch.long, device=DEVICE)
        counts = np.bincount(yi[tr], minlength=len(self.classes_)).astype(float)
        w = torch.tensor(len(tr) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)

        self.net_ = GraphNet(self.arch, self.A_, Xn.shape[-1], self.hidden, self.layers, self.dropout,
                             n_classes=len(self.classes_)).to(DEVICE)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait = -1.0, None, 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.net_.train()
            perm = tr[rng.permutation(len(tr))]
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                idx = torch.as_tensor(idx, device=DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(self.net_(Xn[idx]), yt[idx], weight=w)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                pv = self.net_(Xn[torch.as_tensor(va, device=DEVICE)]).argmax(1).cpu().numpy()
            f1 = f1_score(yi[va], pv, average='macro')
            self.history_.append(f1)
            if f1 > best + 1e-4:
                best, best_state, wait = f1, copy.deepcopy(self.net_.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience and epoch + 1 >= self.min_epochs:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.epochs_run_, self.best_val_f1_ = len(self.history_), best
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            x = self._nodes(np.asarray(X, float)).to(self.device_)
            return torch.softmax(self.net_(x), 1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def n_parameters(self):
        return sum(p.numel() for p in self.net_.parameters())
