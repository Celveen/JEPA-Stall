"""Plot final CD on every target and success versus the physical push budget."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'jepa-stall-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
summary = json.loads((args.report / 'summary.json').read_text())
assert summary['status'] == 'complete' and summary['samples_per_condition'] == 20
arms = [('native5', 'Native 5/5', '#444444'), ('fixed3', 'Fixed S 3/3', '#1976b9'),
        ('dynamic3', 'Dynamic S 3/3', '#cf6b19'), ('fixed3_nowarm', 'Fixed S, no warmup', '#7c509e')]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
values = []
for j, (arm, label, color) in enumerate(arms):
    cds, successes, steps = [], [], []
    for i in range(20):
        with np.load(args.root / arm / f'sample_{i:02d}/rope_metrics.npz') as m:
            cds.append(float(m['final_cd'][0]))
            successes.append(bool(m['successes'][0]))
            steps.append(int(m['actual_pushes'][0]))
    values.append(cds)
    axes[0].scatter(np.full(20, j) + np.linspace(-.12, .12, 20), cds, color=color, s=20, alpha=.8)
    axes[0].scatter(j, np.mean(cds), color=color, s=95, marker='D', edgecolor='white', zorder=4)
    curve = [100 * np.mean(np.asarray(successes) & (np.asarray(steps) <= k)) for k in range(21)]
    axes[1].step(range(21), curve, where='post', color=color, label=label, linewidth=1.8)
axes[0].axhline(.3, color='#777777', linestyle=':', linewidth=1)
axes[0].set(xticks=range(4), xticklabels=[x[1] for x in arms], ylabel='Final Chamfer distance',
            title='All 20 targets; diamonds show means', ylim=(0, None))
axes[0].tick_params(axis='x', labelsize=8)
axes[1].set(xlabel='Completed pushes', ylabel='Success (%)', title='Success at CD < 0.3',
            xticks=range(0,21,5), yticks=range(0,101,20), ylim=(0,100))
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Rope: 20 fixed targets per condition, per-push observation, 20-push limit', fontsize=11)
for suffix in ('png','pdf'):
    fig.savefig(args.report / f'results.{suffix}', dpi=200)
plt.close(fig)
