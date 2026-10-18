#!/usr/bin/env python3
"""
Step 4: Bayesian GLMM partitioning genome-wide theta loss into drift vs.
selection, per replicate.


USAGE:
    python3 step4_theta_glmm.py \
        --gen1 Red=a1_Gen_Rot.1kb.count3.theta Blue=a1_Gen_Blau.1kb.count3.theta Gold=a1_Gen_Gold.1kb.count3.theta Green=a1_Gen_Grun.1kb.count3.theta \
        --gen7 Red=a7_Gen_Rot.1kb.count3.theta Blue=a7_Gen_Blau.1kb.count3.theta Gold=a7_Gen_Gold.1kb.count3.theta Green=a7_Gen_Grun.1kb.count3.theta \
        --blocks Red=Red_haplotypeblocks.csv Blue=Blue_haplotypeblocks.csv Gold=Gold_haplotypeblocks.csv Green=Green_haplotypeblocks.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd

WINDOW_SIZE = 1000
HDI_PROB = 0.95   # credible-interval width; 0.95 to match the brms analysis (ArviZ default would be 0.94)
HDI_PCT = int(round(HDI_PROB * 100))
COLUMN_NAMES = ['scaff', 'window_mid', 'n_snps', 'covered_fraction', 'theta']


def parse_kv(pairs):
    return {p.split('=', 1)[0]: p.split('=', 1)[1] for p in pairs}


def load_theta_file(path):
    df = pd.read_csv(path, sep='\t', header=None, names=COLUMN_NAMES)
    n_rows = len(df)
    for col in ['n_snps', 'covered_fraction', 'theta']:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        after_na = df[col].isna().sum()
        n_non_numeric = after_na - before_na
        if n_non_numeric > 0:
            print(f"  NOTE: {n_non_numeric}/{n_rows} rows in {path} had a non-numeric '{col}' value "
                  f"(likely 'na' from PoPoolation for windows failing the coverage filter) - "
                  f"coerced to NaN, will be dropped.")
    half = WINDOW_SIZE // 2
    df['window_start'] = df['window_mid'] - half + 1
    df['window_end'] = df['window_mid'] + half
    return df


def build_windows(gen1_path, gen7_path):
    g1 = load_theta_file(gen1_path)
    g7 = load_theta_file(gen7_path)
    merged = g1.merge(
        g7, on=['scaff', 'window_mid', 'window_start', 'window_end'],
        suffixes=('_gen1', '_gen7'), how='inner'
    )
    n_g1_only = len(g1) - len(merged)
    n_g7_only = len(g7) - len(merged)
    if n_g1_only or n_g7_only:
        print(f"  NOTE: {n_g1_only} Gen1-only windows, {n_g7_only} Gen7-only windows dropped in the join "
              f"(window definitions didn't match between the two files)")

    n_before = len(merged)
    merged = merged[(merged['n_snps_gen1'] > 0) | (merged['n_snps_gen7'] > 0)].copy()
    n_dropped = n_before - len(merged)
    print(f"  {len(merged)} windows retained, {n_dropped} dropped "
          f"(zero SNPs at both Gen1 and Gen7 - uninformative)")

    n_before_na = len(merged)
    merged = merged.dropna(subset=['theta_gen1', 'theta_gen7'])
    n_dropped_na = n_before_na - len(merged)
    if n_dropped_na > 0:
        print(f"  {n_dropped_na} further windows dropped (non-numeric/missing theta at Gen1 or Gen7, "
              f"e.g. 'na' from PoPoolation's coverage filter)")

    merged['delta_theta'] = merged['theta_gen1'] - merged['theta_gen7']
    return merged


def tag_selected(windows_df, blocks_path):
    blocks = pd.read_csv(blocks_path)
    for col in ('scaff', 'pos', 'haplotypeblock_nr'):
        if col not in blocks.columns:
            sys.exit(f"ERROR: expected column '{col}' in {blocks_path}")
    extents = blocks.groupby(['scaff', 'haplotypeblock_nr']).agg(
        block_start=('pos', 'min'), block_end=('pos', 'max')
    ).reset_index()

    windows_df = windows_df.copy()
    windows_df['selected'] = False
    for scaff, grp in extents.groupby('scaff'):
        mask_scaff = windows_df['scaff'] == scaff
        if not mask_scaff.any():
            continue
        w = windows_df.loc[mask_scaff]
        overlap_any = np.zeros(len(w), dtype=bool)
        for _, blk in grp.iterrows():
            overlap_any |= (w['window_start'] <= blk['block_end']) & (w['window_end'] >= blk['block_start'])
        windows_df.loc[mask_scaff, 'selected'] = overlap_any

    n_sel = windows_df['selected'].sum()
    print(f"  {n_sel} windows tagged as selected (overlap this replicate's own haplotype blocks) "
          f"out of {len(windows_df)}")
    return windows_df


def fit_glmm(windows_df, label, draws=2000, tune=2000, chains=4, seed=42):
    import pymc as pm
    import arviz as az

    delta = windows_df['delta_theta'].values
    selected = windows_df['selected'].values.astype(int)

    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=0, sigma=0.01)          # background (drift) theta loss
        beta_selected = pm.Normal('beta_selected', mu=0, sigma=0.01)  # ADDITIONAL loss in selected windows
        sigma = pm.HalfNormal('sigma', sigma=0.01)
        mu = intercept + beta_selected * selected
        pm.Normal('delta_theta_obs', mu=mu, sigma=sigma, observed=delta)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                           progressbar=False, target_accept=0.9)

    hdi_results = {}
    means = {}
    for var in ['intercept', 'beta_selected', 'sigma']:
        try:
            var_hdi = az.hdi(trace, var_names=[var], prob=HDI_PROB)
        except TypeError:
            var_hdi = az.hdi(trace, var_names=[var], hdi_prob=HDI_PROB)
        hdi_results[var] = var_hdi[var].values
        means[var] = float(trace.posterior[var].values.mean())

    print(f"\n{label} GLMM posterior summary ({HDI_PCT}% HDI):")
    for var in ['intercept', 'beta_selected', 'sigma']:
        lo, hi = float(hdi_results[var][0]), float(hdi_results[var][1])
        print(f"  {var}: mean={means[var]:.6g}, {HDI_PCT}% HDI=[{lo:.6g}, {hi:.6g}]")

    beta_draws = trace.posterior['beta_selected'].values.flatten()
    prob_extra_loss = float((beta_draws > 0).mean())
    print(f"  Posterior probability that selection adds EXTRA theta loss beyond drift "
          f"(P(beta_selected > 0)): {prob_extra_loss:.3f}")

    summary = {
        'intercept_mean': means['intercept'], 'intercept_hdi': hdi_results['intercept'].tolist(),
        'beta_selected_mean': means['beta_selected'], 'beta_selected_hdi': hdi_results['beta_selected'].tolist(),
    }
    return trace, summary, prob_extra_loss


def plot_posteriors(traces_by_label, save_path):
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply_style()

    labels = list(traces_by_label.keys())
    fig, axes = plt.subplots(len(labels), 2, figsize=(11, 2.2 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = axes.reshape(1, 2)

    intercept_draws = {l: traces_by_label[l].posterior['intercept'].values.flatten() for l in labels}
    beta_draws = {l: traces_by_label[l].posterior['beta_selected'].values.flatten() for l in labels}

    # ONE shared x-range across every panel (both columns, all replicates), so
    # drift-only loss and additional selection-effect loss are directly
    # comparable in magnitude, both within and across replicates.
    all_vals = list(intercept_draws.values()) + list(beta_draws.values())
    xmin = min(d.min() for d in all_vals)
    xmax = max(d.max() for d in all_vals)

    intercept_stats = {}
    for l in labels:
        d = intercept_draws[l]
        intercept_stats[l] = (float(d.mean()), float(np.percentile(d, 3)), float(np.percentile(d, 97)))

    for row, label in enumerate(labels):
        color = ps.REPLICATE_COLORS.get(label, '#333333')

        ax_drift = axes[row][0]
        ax_drift.hist(intercept_draws[label], bins=60, color=color, alpha=0.4)
        ax_drift.axvline(0, color='black', linewidth=1)
        ax_drift.set_xlim(xmin, xmax)
        ax_drift.text(0.02, 0.85, label, transform=ax_drift.transAxes, color=color,
                      fontweight='bold', fontsize=ps.AXIS_FONTSIZE, va='top', ha='left')
        mean_i, lo_i, hi_i = intercept_stats[label]
        ax_drift.text(0.98, 0.85, f"mean={mean_i:.4g}\n{HDI_PCT}% HDI=[{lo_i:.3g}, {hi_i:.3g}]",
                      transform=ax_drift.transAxes, color=color, fontsize=ps.TICK_FONTSIZE,
                      va='top', ha='right')
        ax_drift.set_ylabel('posterior\ndensity')
        ax_drift.set_yticks([])

        ax_sel = axes[row][1]
        ax_sel.hist(beta_draws[label], bins=60, color=color, alpha=0.9)
        ax_sel.axvline(0, color='black', linewidth=1)
        ax_sel.set_xlim(xmin, xmax)
        ax_sel.set_yticks([])

    axes[0][0].text(0.5, 1.08, 'drift-only loss (intercept)', transform=axes[0][0].transAxes,
                    fontsize=ps.AXIS_FONTSIZE, fontweight='bold', ha='center', va='bottom')
    axes[0][1].text(0.5, 1.08, 'additional loss from selection (beta_selected)', transform=axes[0][1].transAxes,
                    fontsize=ps.AXIS_FONTSIZE, fontweight='bold', ha='center', va='bottom')
    axes[-1][0].set_xlabel('theta loss')
    axes[-1][1].set_xlabel('theta loss')
    fig.tight_layout()
    saved = ps.save_fig(fig, save_path,
                         "theta loss partitioned into drift and selection effects, shared scale across replicates")
    print(f"Saved {saved}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen1', nargs='+', required=True, help='label=path pairs')
    ap.add_argument('--gen7', nargs='+', required=True, help='label=path pairs')
    ap.add_argument('--blocks', nargs='+', required=True, help='label=path pairs (Step 1 output)')
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--tune', type=int, default=2000)
    args = ap.parse_args()

    import os
    os.makedirs('figures', exist_ok=True)

    gen1_paths = parse_kv(args.gen1)
    gen7_paths = parse_kv(args.gen7)
    block_paths = parse_kv(args.blocks)

    traces = {}
    summary_rows = []
    for label in gen1_paths:
        print(f"\n=== {label} ===")
        windows = build_windows(gen1_paths[label], gen7_paths[label])
        windows = tag_selected(windows, block_paths[label])
        windows.to_csv(f"{label}_theta_windows_tagged.csv", index=False)

        trace, summary, prob_extra_loss = fit_glmm(windows, label, draws=args.draws, tune=args.tune)
        traces[label] = trace
        summary_rows.append({
            'replicate': label,
            'n_windows_total': len(windows),
            'n_windows_selected': int(windows['selected'].sum()),
            'intercept_mean': summary['intercept_mean'],
            'beta_selected_mean': summary['beta_selected_mean'],
            'beta_selected_hdi_lower': summary['beta_selected_hdi'][0],
            'beta_selected_hdi_upper': summary['beta_selected_hdi'][1],
            'prob_selection_adds_extra_loss': prob_extra_loss,
        })

    overview = pd.DataFrame(summary_rows)
    overview.to_csv('theta_glmm_overview.csv', index=False)
    print("\nSaved theta_glmm_overview.csv")
    print(overview.to_string(index=False))

    plot_posteriors(traces, 'figures/theta_glmm_posterior')


if __name__ == '__main__':
    main()
