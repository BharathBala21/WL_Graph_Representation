"""Generates the two result charts used in presentation.tex.

Values are copied from the project's recorded results:
  * Level-1 WL-height sweep: NB21 (21_Level1_Binary_Stress_Detection) summary table.
  * Level-2 per-crop Macro-F1: Task_Direction_Audit/task_comparison_table.csv.
Run:  python make_figures.py   (writes figs/*.png next to this script)
"""
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figs')
os.makedirs(OUT, exist_ok=True)

MAROON, INK, MUTED, GRID = '#A4123F', '#1f1f1f', '#5f5e5a', '#e6e5e0'
GRAYS = ['#6f6e69', '#9a9993', '#bdbcb6']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': '#c9c8c2',
    'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ---------------------------------------------------------------- 1. WL height sweep (Level 1)
heights = [0, 1, 2, 3]
sweep = {
    'Hybrid':         [0.6007, 0.6591, 0.6874, 0.6969],
    'Feature-Family': [0.6007, 0.6557, 0.6817, 0.6914],
    'kNN':            [0.6007, 0.6374, 0.6647, 0.6737],
    'Correlation':    [0.6007, 0.6117, 0.6170, 0.6216],
    'Spatial':        [0.6007, 0.6175, 0.6216, 0.6212],
}
style = {'Hybrid': (MAROON, 2.6), 'Feature-Family': ('#d0607f', 2.0),
         'kNN': (GRAYS[0], 1.8), 'Correlation': (GRAYS[1], 1.8), 'Spatial': (GRAYS[2], 1.8)}

fig, ax = plt.subplots(figsize=(6.4, 4.3))
for name, ys in sweep.items():
    c, lw = style[name]
    ax.plot(heights, ys, color=c, lw=lw, marker='o', ms=6, mec='white', mew=1.5, zorder=3)
# direct labels at the right end, nudged apart where lines converge
label_y = {'Hybrid': 0.700, 'Feature-Family': 0.689, 'kNN': 0.6737,
           'Correlation': 0.6165, 'Spatial': 0.6255}
for name, y in label_y.items():
    ax.text(3.08, y, name, va='center', fontsize=10, color=style[name][0] if name in ('Hybrid', 'Feature-Family') else MUTED)
for val, lab in [(0.6642, 'RBF-SVM'), (0.6522, 'Random Forest')]:
    ax.axhline(val, color=INK, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.08, val + 0.0025, lab, fontsize=9, color=INK, ha='left', va='bottom')
ax.set_xticks(heights, ['h=0\n(no graph)', 'h=1', 'h=2', 'h=3'])
ax.set_xlim(-0.15, 3.9)
ax.set_ylim(0.59, 0.71)
ax.set_ylabel('Macro-F1')
ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title('Healthy vs stressed: score by WL depth', loc='left', color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'wl_height_sweep.png'), dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- 2. Per-crop Level 2
crops = [  # crop, best topology, WL, RF, RBF
    ('Cucumber', 'Family', 0.852, 0.804, 0.681),
    ('Snake gourd', 'Hybrid', 0.778, 0.718, 0.692),
    ('Tomato', 'Correlation', 0.776, 0.794, 0.786),
    ('Eggplant', 'Hybrid', 0.750, 0.695, 0.694),
    ('Bottle gourd', 'Hybrid', 0.719, 0.636, 0.711),
    ('Bitter gourd', 'kNN', 0.662, 0.644, 0.667),
    ('Ash gourd', 'Family', 0.646, 0.550, 0.644),
    ('Ridge gourd', 'Hybrid', 0.548, 0.458, 0.556),
]
crops = crops[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.3))
for i, (c, topo, wl, rf, rbf) in enumerate(crops):
    lo, hi = min(wl, rf, rbf), max(wl, rf, rbf)
    ax.plot([lo, hi], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle='round')
    ax.scatter(rf, i, s=55, color=GRAYS[0], edgecolor='white', lw=1.5, zorder=3)
    ax.scatter(rbf, i, s=55, color=GRAYS[2], edgecolor='white', lw=1.5, zorder=3)
    ax.scatter(wl, i, s=75, color=MAROON, edgecolor='white', lw=1.5, zorder=4)
ax.set_yticks(range(len(crops)), [f'{c}  ({t})' for c, _, t, *_ in [(c, None, t) for c, t, *_ in crops]])
ax.set_xlabel('Macro-F1 (healthy vs stressed)')
ax.set_xlim(0.43, 0.88)
ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.scatter([], [], s=75, color=MAROON, label='WL (best graph in brackets)')
ax.scatter([], [], s=55, color=GRAYS[0], label='Random Forest')
ax.scatter([], [], s=55, color=GRAYS[2], label='RBF-SVM')
ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.32, -0.17),
          ncol=3, handletextpad=0.2, columnspacing=1.0)
ax.set_title('Per crop: which graph works best', loc='left', color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'per_crop_l2.png'), dpi=220)
plt.close(fig)
print('wrote', os.listdir(OUT))
