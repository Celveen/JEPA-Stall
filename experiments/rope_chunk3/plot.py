"""Plot the completed paired Rope results without hiding unsuccessful tasks."""

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'jepa-stall-matplotlib'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    with (args.report / 'per_sample.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    arms = [('native3', 'Native-3', '#333333'),
            ('fixed3', 'Fixed S', '#1976b9'),
            ('dynamic3', 'Dynamic S', '#cf6b19'),
            ('native5', 'Native-5', '#8c6bb1')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for index, (arm, label, color) in enumerate(arms):
        selected = sorted((row for row in rows if row['arm'] == arm),
                          key=lambda row: int(row['sample']))
        if len(selected) != 10:
            raise ValueError('Plot only after all ten predetermined tasks finish.')
        axes[0].scatter(np.arange(10) + (index - 1.5) * .12,
                        [float(row['final_cd']) for row in selected],
                        s=30, color=color, label=label)
        with np.load(args.root / arm / 'sample_09/rope_metrics.npz') as metrics:
            cd = np.r_[metrics['initial_cd'][0], metrics['cd_by_replan'][0]]
        axes[1].plot(np.arange(len(cd)), cd, color=color, label=label,
                     linewidth=1.7, linestyle='--' if arm == 'dynamic3' else '-')
    for ax in axes:
        ax.axhline(.3, color='#777777', linestyle=':', linewidth=1)
        ax.set_ylabel('Particle Chamfer distance (lower is better)')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(title='All 10 tasks: final CD', xlabel='Task index', xticks=range(10))
    axes[1].set(title='Task 9: closed-loop simulation variation', xlabel='Completed pushes',
                xticks=range(0, 21, 5))
    axes[1].axvline(13, color='#999999', linewidth=.8, alpha=.6)
    axes[1].text(13.2, .42, 'Fixed/dynamic CD diverges\nunder identical controls', fontsize=8)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle('Rope: 20-push limit, success at CD < 0.3; fixed/dynamic strengths both saturated at 2',
                 fontsize=11)
    for extension in ('png', 'pdf'):
        fig.savefig(args.report / f'results.{extension}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
