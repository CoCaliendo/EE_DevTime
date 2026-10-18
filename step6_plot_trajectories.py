#!/usr/bin/env python3
"""
Step 6c: Allele frequency trajectory plot, colored by Type0/Type1
classification (see classify_trajectories.py).

No single dashed "drift limit" line is drawn - that concept doesn't apply
anymore, since each SNP is now compared against its OWN properly-scaled
drift envelope (conditioned on its own starting frequency), not one shared
threshold. The Type0 (dark) vs Type1 (light) color coding IS how the
classification is shown, replacing the line.

USAGE:
    python3 plot_trajectories.py \
        --classified Red=Red_classified_trajectories.csv Blue=Blue_classified_trajectories.csv Gold=Gold_classified_trajectories.csv Green=Green_classified_trajectories.csv
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_style as ps

GEN_COLS = list(range(7))


def parse_kv(pairs):
    return {p.split('=', 1)[0]: p.split('=', 1)[1] for p in pairs}


def plot_replicate(ax, df, label, max_lines_per_type=40, seed=0):
    base_color = ps.REPLICATE_COLORS.get(label, '#333333')
    dark = base_color
    light = ps.lighten(base_color, amount=0.6)       # for thin individual Type1 lines - pale is fine there
    mean1_color = ps.lighten(base_color, amount=0.4)  # less pale, specifically for the Type1 MEAN line so it stays visible against white

    type0 = df[df['type'] == 'Type0_early']
    type1 = df[df['type'] != 'Type0_early']  # Type1_late + Never_crossed (should be ~0 rows normally)

    def subsample(sub_df):
        if len(sub_df) <= max_lines_per_type:
            return sub_df
        ordered = sub_df.sort_values(0)
        pick_idx = np.linspace(0, len(ordered) - 1, max_lines_per_type).round().astype(int)
        return ordered.iloc[pick_idx]

    for _, row in subsample(type1).iterrows():
        ax.plot(GEN_COLS, row[GEN_COLS].values, color=light, linewidth=0.7, alpha=0.5, zorder=1)
    for _, row in subsample(type0).iterrows():
        ax.plot(GEN_COLS, row[GEN_COLS].values, color=dark, linewidth=0.7, alpha=0.4, zorder=2)

    if len(type0) > 0:
        mean0 = type0[GEN_COLS].mean().values
        ax.plot(GEN_COLS, mean0, color='white', linewidth=4.5, zorder=3)  # halo for contrast
        ax.plot(GEN_COLS, mean0, color=dark, linewidth=3, zorder=4)
    if len(type1) > 0:
        mean1 = type1[GEN_COLS].mean().values
        ax.plot(GEN_COLS, mean1, color='white', linewidth=5.5, zorder=3)   # wider white halo
        ax.plot(GEN_COLS, mean1, color=dark, linewidth=4, zorder=3.5, alpha=0.9)  # solid dark border, full width
        ax.plot(GEN_COLS, mean1, color=mean1_color, linewidth=2.4, zorder=4)      # visible fill on top

    n_total = len(df)
    n0, n1 = len(type0), len(type1)
    pct0 = 100 * n0 / n_total if n_total else float('nan')

    # Find the corner with the least actual data nearby (using the FULL
    # trajectory data, all types), rather than assuming a fixed corner -
    # this handles cases like Blue, where both left-side corners are partly
    # occupied by different subgroups (Type0 crowds upper-left, Type1
    # crowds lower-left), so the truly clear corner is elsewhere.
    all_gens = np.tile(GEN_COLS, len(df))
    all_vals = df[GEN_COLS].values.flatten()
    corners = {
        'upper-left':  (all_gens <= 2.1) & (all_vals >= 0.65),
        'lower-left':  (all_gens <= 2.1) & (all_vals <= 0.35),
        'upper-right': (all_gens >= 3.9) & (all_vals >= 0.65),
        'lower-right': (all_gens >= 3.9) & (all_vals <= 0.35),
    }
    best_corner = min(corners, key=lambda k: corners[k].sum())

    positions = {
        'upper-left':  dict(x=0.02, ys=[0.97, 0.87, 0.80], va='top',    ha='left'),
        'lower-left':  dict(x=0.02, ys=[0.16, 0.09, 0.02], va='bottom', ha='left'),
        'upper-right': dict(x=0.98, ys=[0.97, 0.87, 0.80], va='top',    ha='right'),
        'lower-right': dict(x=0.98, ys=[0.16, 0.09, 0.02], va='bottom', ha='right'),
    }
    pos = positions[best_corner]

    ax.text(pos['x'], pos['ys'][0], label, transform=ax.transAxes, color=dark,
             fontweight='bold', fontsize=ps.AXIS_FONTSIZE, va=pos['va'], ha=pos['ha'])
    ax.text(pos['x'], pos['ys'][1], f"Type0 (early): {pct0:.0f}% (n={n0})", transform=ax.transAxes, color=dark,
             fontsize=ps.TICK_FONTSIZE, va=pos['va'], ha=pos['ha'])
    ax.text(pos['x'], pos['ys'][2], f"Type1 (late): {100-pct0:.0f}% (n={n1})", transform=ax.transAxes,
             color=mean1_color, fontsize=ps.TICK_FONTSIZE, va=pos['va'], ha=pos['ha'])

    ax.set_xlabel('generation')
    ax.set_ylabel('allele frequency\n(polarized)')
    ax.set_ylim(0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--classified', nargs='+', required=True, help='label=path pairs (classify_trajectories.py output)')
    args = ap.parse_args()

    import os
    os.makedirs('figures', exist_ok=True)
    ps.apply_style()

    paths = parse_kv(args.classified)
    labels = list(paths.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, label in zip(axes, labels):
        df = pd.read_csv(paths[label])
        df.columns = [int(c) if c.lstrip('-').isdigit() else c for c in df.columns]
        plot_replicate(ax, df, label)

    fig.tight_layout()
    saved = ps.save_fig(fig, 'figures/trajectory_types',
                         "allele frequency trajectories by type - early vs late drift-envelope crossing")
    print(f"Saved {saved}")


if __name__ == '__main__':
    main()
