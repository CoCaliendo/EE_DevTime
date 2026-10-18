#!/usr/bin/env python3
"""
Step 7b: Plot sparse PCA trajectories through PC space.

One point per replicate x generation (28 total), connected in generation
order (Gen0 -> Gen6) within each replicate, so each line shows how that
replicate's genome-wide allele-frequency profile moved through PC space
over the course of the experiment. Start (Gen0) and end (Gen6) are marked
distinctly so direction of travel is clear without needing arrows.

USAGE:
    python3 plot_sparse_pca.py --scores sparse_pca_scores.csv
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import plot_style as ps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores', required=True, help='sparse_pca_scores.csv from step7_sparse_pca.py')
    args = ap.parse_args()

    import os
    os.makedirs('figures', exist_ok=True)
    ps.apply_style()

    df = pd.read_csv(args.scores)

    fig, ax = plt.subplots(figsize=(7, 6))

    for replicate, grp in df.groupby('replicate'):
        grp = grp.sort_values('generation')
        color = ps.REPLICATE_COLORS.get(replicate, '#333333')

        ax.plot(grp['PC1'], grp['PC2'], color=color, linewidth=1.8, zorder=2, alpha=0.85)

        # intermediate generations: small dots
        mid = grp.iloc[1:-1]
        ax.scatter(mid['PC1'], mid['PC2'], color=color, s=30, zorder=3)

        # Gen0 (start): open circle
        start = grp.iloc[0]
        ax.scatter(start['PC1'], start['PC2'], facecolors='white', edgecolors=color,
                   linewidths=2, s=90, zorder=4)

        # Gen6 (end): filled square
        end = grp.iloc[-1]
        ax.scatter(end['PC1'], end['PC2'], color=color, marker='s', s=90, zorder=4,
                   label=replicate)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.axhline(0, color='grey', linewidth=0.6, zorder=1)
    ax.axvline(0, color='grey', linewidth=0.6, zorder=1)
    ax.legend(frameon=False, title='replicate\n(○=Gen0, ■=Gen6)', loc='best', fontsize=ps.TICK_FONTSIZE)

    fig.tight_layout()
    saved = ps.save_fig(fig, 'figures/sparse_pca_trajectories',
                         "sparse PCA trajectories through generations, all replicates")
    print(f"Saved {saved}")


if __name__ == '__main__':
    main()
