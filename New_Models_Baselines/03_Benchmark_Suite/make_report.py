"""Builds the comparison tables, statistical tests and figures from the results/ CSV files.

Usage (from this folder):  python make_report.py
Writes results/summary_*.csv, results/statistics_*.csv, results/REPORT_TABLES.md and figures/*.png.
Missing experiments are skipped, so it can be run after any of the run_*.py scripts.
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import bench  # noqa: E402

R, F = bench.RESULTS, bench.FIGURES
os.makedirs(F, exist_ok=True)
WL = 'WL + SVM'
# validated palette (dataviz reference, light mode); baselines stay neutral so "ours vs the rest" reads at once
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
GREY, INK, INK2, GRID, SURFACE = '#b4b2a9', '#0b0b0b', '#52514e', '#e7e6e1', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'savefig.dpi': 200})
md = []


def load(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def fam(df):
    return df.assign(model=df.model_family)


def ms(series, digits=3):
    return f'{series.mean():.{digits}f} ± {series.std(ddof=1):.{digits}f}'


def table(df, title, note=None):
    md.append(f'\n### {title}\n')
    if note:
        md.append(note + '\n')
    cells = [[str(c) for c in df.columns]] + [['' if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)]
    md.append('| ' + ' | '.join(cells[0]) + ' |')
    md.append('|' + '---|' * len(cells[0]))
    md.extend('| ' + ' | '.join(r) + ' |' for r in cells[1:])
    md.append('')


WL_TUNED = 'WL + SVM (tuned)'


def is_ours(m):
    return m in (WL, WL_TUNED)


def is_cnn(m):
    return m.startswith('CNN ')


def colour(m):
    return BLUE if is_ours(m) else ORANGE if is_cnn(m) else GREY


def label(m):
    return f'{m}  (ours)' if is_ours(m) else m


def end_to_end_ms():
    """Typical time from photo file to answer, before the model: feature models read the full photo and
    segment + measure it; CNNs read it at 1/4 size and resize it (the CNN itself is in the model time)."""
    ext, cnn = load('feature_extraction_timing.csv'), load('cnn_end_to_end_timing.csv')
    base = float((ext.read_ms + ext.segment_and_measure_ms).median()) if ext is not None else np.nan
    per = {} if cnn is None else dict(zip(cnn.model, cnn.read_resize_ms))
    return lambda m: per.get(m, base) if is_cnn(m) else base


def save(fig, name):
    fig.savefig(os.path.join(F, name), bbox_inches='tight')
    plt.close(fig)
    print('figure', name)


# ---------------------------------------------------------------- reference: always answer "stressed"
ALWAYS = 'Always says "stressed" (reference)'


def always_stressed_scores(task):
    """Scores of a model that ignores the leaf and always answers "stressed". Accuracy = share of stressed
    leaves; macro-F1 = p / (1 + p) (F1 of the stressed class is 2p/(1+p), of the healthy class 0)."""
    d = bench.load()
    groups = [g for _, g in d.groupby('crop')] if task == 'level2' else [d]
    p = np.array([g.stressed.mean() for g in groups])
    return {'accuracy': float(p.mean()), 'macro_f1': float((p / (1 + p)).mean())}


def always_stressed(task, f1col):
    sc = always_stressed_scores(task)
    row = {'Model': ALWAYS, f1col: sc['macro_f1'], 'Balanced acc.': 0.5,
           'Specificity (healthy found)': 0.0, 'Sensitivity (stressed found)': 1.0}
    if task != 'level2':
        row['Accuracy'] = sc['accuracy']
    else:
        row['Accuracy (mean of crops)'] = sc['accuracy']
    return row


# ---------------------------------------------------------------- 1. quality, speed, size
def summarise(task, df):
    df = fam(df)
    binary = task != '57-class'
    rows = []
    pre = end_to_end_ms()
    for m, g in df.groupby('model', sort=False):
        if task == 'level2':  # average per crop first, then across the 8 crops
            per_crop = g.groupby('crop').mean(numeric_only=True)
            row = {'Model': m, 'Macro-F1 (mean of crops)': per_crop.macro_f1.mean(),
                   'Balanced acc.': per_crop.balanced_accuracy.mean(), 'Accuracy (mean of crops)': per_crop.accuracy.mean(),
                   'ROC-AUC': per_crop.roc_auc.mean(),
                   'Specificity (healthy found)': per_crop.specificity_healthy.mean(),
                   'Train s (all 8 crops)': per_crop.train_s.sum(),
                   'Leaf latency ms': per_crop.single_leaf_ms.mean(),
                   'End-to-end ms/photo': pre(m) + per_crop.single_leaf_ms.mean(),
                   # CNN: the 8 crop classifiers share ONE frozen network, so count it once
                   'Size KB (all 8 crops)': (per_crop.head_kb.sum() + (per_crop.model_kb - per_crop.head_kb).iloc[0])
                   if is_cnn(m) and 'head_kb' in g else per_crop.model_kb.sum(),
                   'Peak train MB': g.train_peak_mb.max()}
        else:
            row = {'Model': m, 'Macro-F1': g.macro_f1.mean(), 'Macro-F1 SD': g.macro_f1.std(ddof=1),
                   'Balanced acc.': g.balanced_accuracy.mean(), 'Accuracy': g.accuracy.mean()}
            if binary:
                if 'roc_auc' in g and g.roc_auc.notna().any():
                    row['ROC-AUC'] = g.roc_auc.mean()
                row.update({'Specificity (healthy found)': g.specificity_healthy.mean(),
                            'Sensitivity (stressed found)': g.sensitivity_stressed.mean()})
            if task == 'level3':
                row.update({'Crop identified correctly': g.routing_accuracy.mean(),
                            'Macro-F1 if crop were known': g.oracle_macro_f1.mean()})
            row.update({'Train s': g.train_s.mean(), 'Leaf latency ms': g.single_leaf_ms.mean(),
                        'End-to-end ms/photo': pre(m) + g.single_leaf_ms.mean(),
                        'Size KB': g.model_kb.mean()})
            if 'batch_ms_per_leaf' in g:
                row['Batch ms/leaf'] = g.batch_ms_per_leaf.mean()
            if 'train_peak_mb' in g:
                row['Peak train MB'] = g.train_peak_mb.max()
        rows.append(row)
    if binary:
        rows.append(always_stressed(task, list(rows[0])[1]))
    out = pd.DataFrame(rows).sort_values(list(rows[0])[1], ascending=False)  # by macro-F1
    out.round(4).to_csv(os.path.join(R, f'summary_{task}.csv'), index=False)
    table(out.round(3), f'Task: {task}')
    return out


def level2_details(df):
    df = fam(df)
    crop_tab = df.groupby(['crop', 'model']).macro_f1.mean().unstack()
    best = crop_tab.idxmax(axis=1)
    crop_tab['Best model'] = best
    crop_tab.round(3).reset_index().to_csv(os.path.join(R, 'summary_level2_per_crop.csv'), index=False)
    table(crop_tab.round(3).reset_index(), 'Level 2: macro-F1 per crop')

    # Friedman test over crops + average rank (1 = best), as in Demsar (2006)
    means = df.groupby(['crop', 'model']).macro_f1.mean().unstack()
    ranks = means.rank(axis=1, ascending=False).mean().sort_values()
    chi2, p = stats.friedmanchisquare(*[means[c] for c in means])
    # paired Wilcoxon, WL vs each other model, over the 8 crops x 5 folds = 40 paired fold scores
    piv = df.pivot_table(index=['crop', 'fold'], columns='model', values='macro_f1')
    rank_df = ranks.round(2).reset_index().rename(columns={'model': 'Model', 0: 'Average rank (1 = best)'})
    rank_df.to_csv(os.path.join(R, 'statistics_level2_ranks.csv'), index=False)
    table(rank_df, 'Level 2: average rank over the 8 crops',
          f'Friedman test: χ² = {chi2:.2f}, p = {p:.4f} (do the models differ at all?)')
    for ours, fname, title in [(WL, 'statistics_level2.csv', 'WL + SVM (best research setting)'),
                               (WL_TUNED, 'statistics_level2_tuned.csv', 'WL + SVM (tuned by nested CV)')]:
        if ours not in piv:
            continue
        rows = []
        for m in piv.columns.drop([c for c in (WL, WL_TUNED) if c in piv]):
            d = piv[ours] - piv[m]
            w = stats.wilcoxon(piv[ours], piv[m])
            rows.append({'Baseline': m, 'WL better on crops': f'{int((means[ours] > means[m]).sum())}/8',
                         'Mean ΔF1 (WL − baseline)': round(d.mean(), 4),
                         'Wilcoxon p (40 paired folds)': round(w.pvalue, 4)})
        res = pd.DataFrame(rows).sort_values('Wilcoxon p (40 paired folds)')
        k = len(res)  # Holm-Bonferroni correction
        sig, still = [], True
        for i, p_ in enumerate(res['Wilcoxon p (40 paired folds)']):
            still = still and p_ <= 0.05 / (k - i)
            sig.append('yes' if still else 'no')
        res['Significant (Holm, α=0.05)'] = sig
        res.to_csv(os.path.join(R, fname), index=False)
        table(res, f'Level 2: {title} vs each other model')

    # figure: mean macro-F1 across crops, with each crop as a dot
    order = means.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(order) + 1.4))
    y = np.arange(len(order))
    ax.barh(y, means.mean()[order], color=[colour(m) for m in order], height=0.62)
    for i, m in enumerate(order):
        ax.scatter(means[m], np.full(len(means), i), s=12, color=INK2, zorder=3, linewidths=0)
        ax.text(means.mean()[m] + 0.005, i + 0.28, f'{means.mean()[m]:.3f}', va='center', fontsize=8.5, color=INK)
    ax.set_yticks(y, [label(m) for m in order])
    ax.set_xlabel('Macro-F1, healthy vs stressed (bar = mean of 8 crops, dots = each crop)')
    ax.set_xlim(0.3, 1.0)
    ax.grid(axis='y', visible=False)
    ax.set_title('Level 2 — crop-specific stress detection', loc='left', fontsize=12, fontweight='bold')
    save(fig, 'fig1_level2_macro_f1.png')


def bars(sum_df, col, title, xlabel, fname, xmin=0.3):
    """Horizontal bars per model; the always-"stressed" reference is drawn as a dashed line, not a bar."""
    d = sum_df[sum_df.Model != ALWAYS].sort_values(col)
    ref = sum_df.loc[sum_df.Model == ALWAYS, col]
    fig, ax = plt.subplots(figsize=(7.8, 0.34 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.barh(y, d[col], color=[colour(m) for m in d.Model], height=0.62)
    for i, v in enumerate(d[col]):
        ax.text(v + 0.004, i, f'{v:.3f}', va='center', fontsize=8.5, color=INK)
    if len(ref) and not pd.isna(ref.iloc[0]):
        ax.axvline(ref.iloc[0], color=INK2, ls='--', lw=1.2)
        ax.text(ref.iloc[0], len(d) - 0.35, f'  always says "stressed": {ref.iloc[0]:.3f}', fontsize=8.5,
                color=INK2, va='bottom')
    ax.set_yticks(y, [label(m) for m in d.Model])
    ax.set_xlim(xmin, 1.0)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    save(fig, fname)


def tradeoff(sum_df, task, xcol, xlabel, fname, logx=True):
    sum_df = sum_df[~sum_df.Model.str.contains('tuned') & (sum_df.Model != ALWAYS)]  # tuned variants: same speed/size as the defaults
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    f1col = [c for c in sum_df if c.startswith('Macro-F1') and 'SD' not in c][0]
    for _, r in sum_df.iterrows():
        ax.scatter(r[xcol], r[f1col], s=70 if is_ours(r.Model) else 46, color=colour(r.Model), zorder=3,
                   edgecolor=SURFACE, linewidth=2)
    if logx:
        ax.set_xscale('log')
    ax.margins(x=0.12, y=0.1)
    fig.canvas.draw()
    placed = []  # label boxes in pixels; try a few positions and take the first free one
    for _, r in sum_df.sort_values(f1col, ascending=False).iterrows():
        ours = is_ours(r.Model)
        text = r.Model + (' (ours)' if ours else '')
        px, py = ax.transData.transform((r[xcol], r[f1col]))
        w, h = 5.4 * len(text) * fig.dpi / 100, 11 * fig.dpi / 100
        for dx, dy, ha in [(7, 3, 'left'), (7, -13, 'left'), (-7, 3, 'right'), (-7, -13, 'right'), (7, 14, 'left')]:
            x0 = px + dx * fig.dpi / 72 - (w if ha == 'right' else 0)
            y0 = py + dy * fig.dpi / 72
            box = (x0, y0, x0 + w, y0 + h)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                break
        placed.append(box)
        ax.annotate(text, (r[xcol], r[f1col]), xytext=(dx, dy), textcoords='offset points', ha=ha,
                    fontsize=8.5, color=INK if ours else INK2, fontweight='bold' if ours else 'normal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f1col)
    ax.set_title(f'{task}: quality vs {xlabel.split(" (")[0].lower()}', loc='left', fontsize=12, fontweight='bold')
    save(fig, fname)


# ---------------------------------------------------------------- 2. learning curves
def learning(task, df):
    df = fam(df)
    g = df.groupby(['model', 'crop', 'fraction']).macro_f1.mean().groupby(['model', 'fraction']).mean().unstack()
    full = g[1.0]
    out = g.copy()
    out.columns = [f'{int(c * 100)}% data' for c in out.columns]
    out['F1 kept at 20% data'] = g[0.2] / full
    out['Area under curve'] = [np.trapz(g.loc[m].values, g.columns.values) / (g.columns[-1] - g.columns[0]) for m in g.index]
    out = out.sort_values('20% data', ascending=False).reset_index().rename(columns={'model': 'Model'})
    out.round(4).to_csv(os.path.join(R, f'summary_learning_curve_{task}.csv'), index=False)
    table(out.round(3), f'Data efficiency: {task} (macro-F1 by share of training data)')

    top = [m for m in g[1.0].sort_values(ascending=False).index if m != WL][:3]
    styles = {WL: (BLUE, '-', 2.4), top[0]: (ORANGE, '--', 2), top[1]: (AQUA, '-.', 2), top[2]: (YELLOW, ':', 2.2)}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = g.columns.values * 100
    for m in g.index:
        if m not in styles:
            ax.plot(x, g.loc[m], color=GREY, lw=1, alpha=0.8)
    for m, (c, ls, lw) in styles.items():
        ax.plot(x, g.loc[m], color=c, ls=ls, lw=lw, marker='o', ms=4)
    # end labels, pushed apart vertically so close finishes do not overlap
    ends = sorted(((g.loc[m].iloc[-1], m) for m in styles), reverse=True)
    lo_, hi_ = g.values.min(), g.values.max()
    gap = 0.045 * (hi_ - lo_)
    ypos = []
    for y_, m in ends:
        ypos.append(min(y_, ypos[-1] - gap) if ypos else y_)
    for (y_, m), yl in zip(ends, ypos):
        ax.annotate(m + (' (ours)' if m == WL else ''), (x[-1], y_), xytext=(x[-1] + 3, yl), textcoords='data',
                    va='center', fontsize=8.5, color=INK, fontweight='bold' if m == WL else 'normal',
                    arrowprops=dict(arrowstyle='-', color=GRID, lw=0.8) if abs(yl - y_) > 1e-9 else None)
    ax.plot([], [], color=GREY, lw=1, label='other baselines')
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    ax.set_xlabel('Share of the training data used (%)')
    ax.set_ylabel('Macro-F1 on the full test fold')
    ax.set_xlim(5, 140)
    ax.set_xticks([10, 20, 35, 50, 75, 100])
    nice = {'level1': 'Level 1', 'level2': 'Level 2 (mean of 8 crops)'}[task]
    ax.set_title(f'{nice}: how much data does each model need?', loc='left', fontsize=12, fontweight='bold')
    save(fig, f'fig4_learning_curve_{task}.png')


# ---------------------------------------------------------------- 3. robustness
def _boot_f1(y, p, W):
    """Binary macro-F1 for every bootstrap weight row of W (reps x n)."""
    out = []
    for pos in (0, 1):
        tp = W @ ((y == pos) & (p == pos)).astype(float)
        fp = W @ ((y != pos) & (p == pos)).astype(float)
        fn = W @ ((y == pos) & (p != pos)).astype(float)
        out.append(np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0))
    return (out[0] + out[1]) / 2


def robustness(df, preds, reps=2000):
    df = fam(df)
    flips = df.pivot_table(index='model', columns='perturbation', values='flip_rate').drop(columns='clean')
    f1 = df.pivot_table(index='model', columns='perturbation', values='macro_f1')
    change = f1.drop(columns='clean').sub(f1['clean'], axis=0)

    # paired bootstrap over photos: resample the same photos for every model and change
    images = sorted(preds.image_path.unique())
    idx = {p: i for i, p in enumerate(images)}
    rng = np.random.default_rng(bench.SEED)
    W = rng.multinomial(len(images), np.full(len(images), 1 / len(images)), size=reps).astype(float)
    kinds = [k for k in f1.columns if k != 'clean']
    boot = {}
    for m, g in preds.groupby('model_family'):
        per = {}
        for k, gk in g.groupby('perturbation'):
            cols = gk.image_path.map(idx).values
            y = np.full(len(images), -1)
            p = np.full(len(images), -1)
            y[cols], p[cols] = gk.y_true.values, gk.y_pred.values  # rejected photos stay -1 (never counted)
            per[k] = _boot_f1(y, p, W)
        boot[m] = np.mean([per[k] - per['clean'] for k in kinds], axis=0)  # mean F1 change per bootstrap
    rows = []
    for m in f1.index:
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        d = boot[WL] - boot[m]
        rows.append({'Model': m, 'Clean macro-F1': f1.loc[m, 'clean'], 'Mean macro-F1 under changes': f1.loc[m, kinds].mean(),
                     'Mean F1 change': change.loc[m].mean(), '95% CI': f'[{lo:.3f}, {hi:.3f}]',
                     'Worst change': change.loc[m].min(), 'Mean flip rate': flips.loc[m].mean(),
                     'WL loses less? (95% CI of difference)': '' if m == WL else
                     f'{"yes" if np.percentile(d, 2.5) > 0 else "not sure"} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]'})
    out = pd.DataFrame(rows).sort_values('Mean F1 change', ascending=False)
    out.round(4).to_csv(os.path.join(R, 'summary_robustness_level1.csv'), index=False)
    table(out.round(3), 'Robustness (Level 1, 240 test photos changed in 7 ways)',
          'Mean F1 change = average over the 7 changes of (macro-F1 on changed photos − macro-F1 on the originals); '
          'closer to 0 is better. CIs: paired bootstrap over photos (2,000 resamples). Flip rate = share of leaves '
          'whose answer changes. Caution: a model that nearly always answers "stressed" flips rarely without being '
          'robust, so read flip rate together with the F1 change.')
    table(change.round(3).reset_index(), 'Macro-F1 change per photo change (negative = worse)')
    table((flips * 100).round(1).reset_index(), 'Flip rate (%) per photo change')
    rej = df[df.model == df.model.iloc[0]][['perturbation', 'rejected_by_qc', 'n_photos']]
    table(rej, 'Photos rejected by the segmentation quality check')

    order = out.Model.tolist()
    labels = [f'{m}  (ours)' if m == WL else m for m in order]
    cols = [c.replace('_', ' ') for c in change.columns]
    for data, fname, title, cmap, vmin, vmax, fmt in [
            (change.loc[order], 'fig5_robustness_f1_change.png',
             'Change in macro-F1 when the photo changes (0 = unaffected)', 'RdBu', -0.3, 0.3, '{:+.2f}'),
            (flips.loc[order] * 100, 'fig5b_robustness_flip_rate.png',
             'Answers that flip when the photo changes (read with the F1 change)', 'Blues', 0, 50, '{:.0f}%')]:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(data.shape[1]), cols, rotation=25, ha='right')
        ax.set_yticks(range(len(order)), labels)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.values[i, j]
                strong = abs(v - (0 if vmin < 0 else vmin)) > 0.6 * (vmax if vmin < 0 else vmax - vmin)
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8, color='white' if strong else INK)
        ax.grid(False)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        save(fig, fname)


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    md.append('# Multi-dimensional baseline comparison: result tables\n\nGenerated by make_report.py.\n')
    sums = {}
    for task in ('level2', 'level1', 'level3'):
        d = load(f'benchmark_{task}_folds.csv')
        if d is not None:
            sums[task] = summarise(task, d)
            if task == 'level2':
                level2_details(d)
    if 'level3' in sums:
        bars(sums['level3'], 'Macro-F1', 'Level 3 — full pipeline (identify crop, then crop stress model)',
             'Macro-F1, healthy vs stressed, all test leaves', 'fig6_level3_macro_f1.png')
    if 'level1' in sums:
        bars(sums['level1'], 'Accuracy', 'Level 1 — accuracy is misleading on this data',
             'Accuracy, healthy vs stressed (84% of leaves are stressed)', 'fig7_level1_accuracy_trap.png', xmin=0.5)
        bars(sums['level1'], 'Balanced acc.', 'Level 1 — balanced accuracy (healthy and stressed count equally)',
             'Balanced accuracy = mean of (stressed found, healthy found)', 'fig7b_level1_balanced_accuracy.png', xmin=0.45)
    if 'level1' in sums:
        tradeoff(sums['level1'], 'Level 1', 'End-to-end ms/photo',
                 'Time from photo file to answer (ms, one leaf, log scale)', 'fig2_tradeoff_latency_level1.png')
        tradeoff(sums['level1'], 'Level 1', 'Train s', 'Training time (s, one CPU core, log scale)',
                 'fig3_tradeoff_training_level1.png')
    if 'level2' in sums:
        tradeoff(sums['level2'], 'Level 2', 'Size KB (all 8 crops)', 'Model size on disk (KB, all 8 crop models, log scale)',
                 'fig3b_tradeoff_size_level2.png')
    ext = load('feature_extraction_timing.csv')
    if ext is not None:
        table(ext.describe().round(1).reset_index(), 'Photo → 38 measurements (same for every feature-based model)')
    for task in ('level2', 'level1'):
        d = load(f'learning_curve_{task}.csv')
        if d is not None:
            learning(task, d)
    d, p = load('robustness_level1.csv'), load('robustness_level1_predictions.csv')
    if d is not None and p is not None:
        robustness(d, p)
    d = load('benchmark_57-class_folds.csv')
    if d is not None:
        md.append('\n## Appendix: 57-class (crop x condition), set aside as future work\n')
        summarise('57-class', d)
    with open(os.path.join(R, 'REPORT_TABLES.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('wrote results/REPORT_TABLES.md')
