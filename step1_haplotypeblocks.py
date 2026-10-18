import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Knee/elbow detection - finds the "variant desert" gap size from the
#    data itself instead of a hand-picked fixed window (e.g. 125bp/550bp).
# ---------------------------------------------------------------------------
def _find_knee(y_sorted_asc):
    """Index of maximum perpendicular distance from the chord connecting the
    first and last point of a sorted curve. This is the same trick used to
    pick eps in a DBSCAN k-distance plot."""
    y = np.asarray(y_sorted_asc, dtype=float)
    n = len(y)
    if n < 3:
        return 0
    x = np.arange(n, dtype=float)
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len
    px, py = x - x[0], y - y[0]
    proj_len = px * line_unit[0] + py * line_unit[1]
    proj_x, proj_y = proj_len * line_unit[0], proj_len * line_unit[1]
    dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return int(np.argmax(dist))


def _gauss_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_1d_gmm_2comp(x, n_iter=300, tol=1e-8, seed=0):
    """Minimal EM for a 2-component 1D Gaussian mixture (no sklearn needed).
    Returns (mu, sigma, pi) sorted so component 0 is the lower mean."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    lo, hi = x[x <= med], x[x > med]
    mu = np.array([lo.mean() if len(lo) else med - 1, hi.mean() if len(hi) else med + 1])
    sigma = np.array([lo.std() + 1e-6 if len(lo) else 1.0, hi.std() + 1e-6 if len(hi) else 1.0])
    pi = np.array([0.5, 0.5])

    prev_ll = -np.inf
    for _ in range(n_iter):
        resp = np.stack([pi[k] * _gauss_pdf(x, mu[k], sigma[k]) for k in range(2)], axis=1)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum <= 0] = 1e-300
        resp = resp / resp_sum

        Nk = resp.sum(axis=0)
        Nk[Nk <= 0] = 1e-300
        mu = (resp * x[:, None]).sum(axis=0) / Nk
        sigma = np.sqrt((resp * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk) + 1e-6
        pi = Nk / len(x)

        ll = np.sum(np.log(resp_sum.flatten()))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    order = np.argsort(mu)
    return mu[order], sigma[order], pi[order]


def estimate_desert_threshold_gmm(df, scaff_col='scaff', pos_col='pos', sensitivity=1.0):
    """Fits a 2-component Gaussian mixture to log10(gap) between consecutive
    significant positions per scaffold, and returns the valley (minimum
    density point) between the two component peaks as the desert threshold.
    More robust than the knee/chord method when the transition between
    'intra-cluster spacing' and 'background spacing' is a broad bend rather
    than a sharp elbow."""
    gaps = []
    for _, grp in df.groupby(scaff_col):
        pos = np.sort(grp[pos_col].values)
        if len(pos) > 1:
            gaps.extend(np.diff(pos))
    gaps = np.sort(np.array(gaps))
    gaps = gaps[gaps > 0]
    if len(gaps) < 10:
        raise ValueError("Not enough within-scaffold neighbour pairs to fit a mixture.")

    log_gaps = np.log10(gaps)
    mu, sigma, pi = _fit_1d_gmm_2comp(log_gaps)

    grid = np.linspace(mu[0], mu[1], 4000)
    density = pi[0] * _gauss_pdf(grid, mu[0], sigma[0]) + pi[1] * _gauss_pdf(grid, mu[1], sigma[1])
    valley_idx = int(np.argmin(density))
    log_threshold = grid[valley_idx] + np.log10(sensitivity)
    threshold = 10 ** log_threshold
    gmm_info = {'mu': mu, 'sigma': sigma, 'pi': pi, 'grid': grid, 'density': density}
    return threshold, gaps, gmm_info


def estimate_desert_threshold(df, scaff_col='scaff', pos_col='pos', sensitivity=1.0):
    """Pools all within-scaffold nearest-neighbour distances between
    significant positions and finds the elbow: gaps below it are 'normal'
    spacing inside an accumulation, gaps above it are 'variant deserts'.

    Genomic gap distributions are heavily right-skewed (most gaps tiny, a
    long tail out to hundreds of kb / Mb). Finding the knee on the raw,
    linear gap values lets that tail dominate the chord geometry and drags
    the detected knee far to the right, producing implausibly large blocks.
    We therefore find the knee on log10(gap) instead, which is the standard
    way to handle this kind of skewed distance distribution, then convert
    back to bp.

    sensitivity < 1.0  -> stricter (smaller) threshold, more/smaller blocks
    sensitivity > 1.0  -> looser (larger) threshold, fewer/bigger blocks
    """
    gaps = []
    for _, grp in df.groupby(scaff_col):
        pos = np.sort(grp[pos_col].values)
        if len(pos) > 1:
            gaps.extend(np.diff(pos))
    gaps = np.sort(np.array(gaps))
    gaps = gaps[gaps > 0]  # log undefined at 0; ties at the same position aren't a gap anyway
    if len(gaps) < 3:
        raise ValueError("Not enough within-scaffold neighbour pairs to estimate a threshold.")
    log_gaps = np.log10(gaps)
    knee_idx = _find_knee(log_gaps)
    log_threshold = log_gaps[knee_idx] + np.log10(sensitivity)
    threshold = 10 ** log_threshold
    return threshold, gaps, knee_idx


# ---------------------------------------------------------------------------
# 2. Variable-size haplotype blocks via 1D single-linkage clustering per
#    scaffold: a new block starts whenever the gap to the previous
#    significant position exceeds the desert threshold. Fully vectorized,
#    no nested loops needed.
# ---------------------------------------------------------------------------
def sweep_thresholds(df, scaff_col='scaff', pos_col='pos', thresholds=None):
    """Diagnostic sweep: for a range of candidate desert thresholds, report
    how many blocks result and how concentrated the signal is. Plot the
    output (n_blocks vs threshold, log-log) and look for the point where
    the curve stops dropping steeply and flattens into a plateau -- that
    plateau is the natural scale at which you've merged genuine intra-block
    noise but haven't yet started fusing separate accumulations together.
    This is more robust than fitting a single global distribution shape
    (knee or GMM) when the gap distribution is multi-scale rather than
    cleanly bimodal.
    """
    df = df.sort_values([scaff_col, pos_col]).reset_index(drop=True)
    if thresholds is None:
        gaps = []
        for _, grp in df.groupby(scaff_col):
            pos = np.sort(grp[pos_col].values)
            if len(pos) > 1:
                gaps.extend(np.diff(pos))
        gaps = np.array(gaps)
        gaps = gaps[gaps > 0]
        lo, hi = np.percentile(gaps, [1, 99])
        thresholds = np.unique(np.round(np.logspace(np.log10(max(lo, 1)), np.log10(hi), 30)).astype(int))

    rows = []
    for t in thresholds:
        clustered, _ = cluster_by_gap(df, scaff_col, pos_col, gap_threshold=t)
        n_blocks = clustered['haplotypeblock_nr'].nunique()
        block_sizes = clustered.groupby('haplotypeblock_nr').size()
        rows.append({
            'threshold_bp': t,
            'n_blocks': n_blocks,
            'median_block_size': block_sizes.median(),
            'max_block_size': block_sizes.max(),
        })
    return pd.DataFrame(rows)


def cluster_by_gap(df, scaff_col='scaff', pos_col='pos', gap_threshold=None, sensitivity=1.0, method='gmm'):
    df = df.sort_values([scaff_col, pos_col]).reset_index(drop=True)

    if gap_threshold is None:
        if method == 'gmm':
            gap_threshold, gaps, gmm_info = estimate_desert_threshold_gmm(
                df, scaff_col, pos_col, sensitivity=sensitivity
            )
            print(f"  Auto-detected variant-desert threshold (GMM valley): {gap_threshold:,.0f} bp "
                  f"(component means at {10**gmm_info['mu'][0]:,.0f} bp / {10**gmm_info['mu'][1]:,.0f} bp)")
        else:
            gap_threshold, gaps, knee_idx = estimate_desert_threshold(
                df, scaff_col, pos_col, sensitivity=sensitivity
            )
            print(f"  Auto-detected variant-desert threshold (knee): {gap_threshold:,.0f} bp "
                  f"(knee at rank {knee_idx}/{len(gaps)} of pooled nearest-neighbour gaps)")

    new_scaff = df[scaff_col].ne(df[scaff_col].shift(1))
    gap_to_prev = df[pos_col] - df[pos_col].shift(1)
    new_block = new_scaff | (gap_to_prev > gap_threshold)

    df = df.copy()
    df['haplotypeblock_nr'] = new_block.cumsum()
    df['position'] = df[scaff_col].astype(str) + '.' + df[pos_col].astype(str)
    return df, gap_threshold


# ---------------------------------------------------------------------------
# 3. Summarise blocks and score how "concentrated" the signal is
# ---------------------------------------------------------------------------
def summarize_blocks(df_clustered, pval_col='pval', block_col='haplotypeblock_nr', pos_col='pos'):
    summary = df_clustered.groupby(block_col).agg(
        n_snps=(pos_col, 'size'),
        start=(pos_col, 'min'),
        end=(pos_col, 'max'),
        mean_sig=(pval_col, 'mean'),
        max_sig=(pval_col, 'max'),
    )
    summary['length_bp'] = summary['end'] - summary['start'] + 1
    summary['density'] = summary['n_snps'] / summary['length_bp']
    return summary.sort_values('n_snps', ascending=False)


def concentration_score(summary, top_fraction=0.10):
    """Fraction of all significant positions captured by the densest
    top_fraction of blocks. High -> a few dominant clusters (dense mode
    justified). Low -> signal is spread thin (fall back to top-N)."""
    n_top = max(1, int(np.ceil(len(summary) * top_fraction)))
    return summary['n_snps'].iloc[:n_top].sum() / summary['n_snps'].sum()


# ---------------------------------------------------------------------------
# 5. Supplementary materials helpers: sensitivity table + sweep-curve figure.
#    These report block statistics only (no dense/fallback strategy label,
#    since that decision is described in prose rather than in the table).
# ---------------------------------------------------------------------------
def build_sensitivity_table(datasets, thresholds=(75, 150, 300), pval_col='pval',
                             scaff_col='scaff', pos_col='pos'):
    """datasets: dict of {label: (svm_df, top_n)}. For each replicate and
    threshold, clusters the top_n most significant positions and reports
    block-level summary stats, without any strategy/mode label."""
    rows = []
    for label, (svm_df, top_n) in datasets.items():
        top_positions = svm_df.nlargest(n=top_n, columns=pval_col)
        for t in thresholds:
            clustered, _ = cluster_by_gap(top_positions, scaff_col, pos_col, gap_threshold=t)
            summary = summarize_blocks(clustered, pval_col, pos_col=pos_col)
            n_isolated = int((summary['n_snps'] == 1).sum())
            n_multi = int((summary['n_snps'] > 1).sum())
            n_in_clusters = int(summary.loc[summary['n_snps'] > 1, 'n_snps'].sum())
            total = int(summary['n_snps'].sum())
            rows.append({
                'replicate': label,
                'threshold_bp': t,
                'n_positions': total,
                'n_blocks': int(summary.shape[0]),
                'median_block_size': summary['n_snps'].median(),
                'max_block_size': int(summary['n_snps'].max()),
                'n_isolated_singletons': n_isolated,
                'n_multi_snp_blocks': n_multi,
                'pct_positions_clustered': round(100 * n_in_clusters / total, 1) if total else float('nan'),
            })
    table = pd.DataFrame(rows)
    return table.sort_values(['replicate', 'threshold_bp']).reset_index(drop=True)


def plot_gap_sweep_grid(datasets, scaff_col='scaff', pos_col='pos', save_path=None):
    """4-panel (or NxM) figure: n_blocks vs desert threshold (log-log) per
    replicate, using the full significant-position pool for each. Used to
    show the absence of a plateau/knee -- i.e. that the gap distribution is
    continuous/scale-free rather than cleanly bimodal, which is why the
    threshold was anchored to a published value instead of detected
    automatically. No strategy/decision labels are included.

    Styling: replicate color per panel, 14pt axis labels, no in-plot titles
    (panel identity is shown via the replicate-colored line + axis alone;
    the descriptive title belongs in the saved filename, via save_fig()).
    """
    import matplotlib.pyplot as plt
    import plot_style as ps

    ps.apply_style()
    labels = list(datasets.keys())
    n = len(labels)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    flat_axes = []
    for i, label in enumerate(labels):
        svm_df, _ = datasets[label]
        sweep = sweep_thresholds(svm_df, scaff_col=scaff_col, pos_col=pos_col)
        ax = axes[i // ncols][i % ncols]
        color = ps.REPLICATE_COLORS.get(label, '#333333')
        ax.plot(sweep['threshold_bp'], sweep['n_blocks'], marker='o', markersize=3, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axvline(150, color='grey', linestyle='--', linewidth=1)
        ax.set_xlabel('desert threshold (bp)')
        ax.set_ylabel('n_blocks')
        # replicate identity goes in the panel via a small colored label,
        # not a title, per the no-title convention
        ax.text(0.03, 0.03, label, transform=ax.transAxes, color=color,
                 fontweight='bold', fontsize=ps.AXIS_FONTSIZE, va='bottom', ha='left')
        flat_axes.append(ax)

    ps.shared_ylim(flat_axes)

    # hide any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.tight_layout()
    if save_path:
        out = ps.save_fig(fig, save_path, "variant desert threshold sweep - n blocks vs threshold per replicate")
        return fig, out
    return fig, None


# ---------------------------------------------------------------------------
# 4. Orchestration: run once per colour/replicate, decide strategy automatically
# ---------------------------------------------------------------------------
def build_variable_haplotypeblocks(svm_df, top_n_fallback, pval_col='pval',
                                    scaff_col='scaff', pos_col='pos',
                                    significance_threshold=None,
                                    concentration_cutoff=0.5, top_fraction=0.10,
                                    min_snps_per_block=3, gap_sensitivity=1.0,
                                    gap_method='gmm', gap_threshold=None, label=''):
    """
    significance_threshold : if your svm_df is not already pre-filtered to
        significant hits only, set this (e.g. pval_col >= your FDR cutoff).
        If None, the whole svm_df is treated as the significant pool.

    gap_threshold : if given (e.g. from a published LD-decay estimate), this
        fixed value is used directly and the knee/GMM auto-detection is
        skipped entirely. Recommended once you have a literature-anchored
        number rather than trusting distributional fitting on data that may
        not be cleanly bimodal.

    concentration_cutoff : if the densest `top_fraction` of blocks contain
        at least this fraction of all significant positions, we're in
        "dense cluster" mode -> keep only real accumulations.
        Otherwise the signal is diffuse -> fall back to the top_n_fallback
        most significant positions (still clustered adaptively, not with a
        fixed window).
    """
    print(f"\n=== {label or 'dataset'}: clustering all significant positions ===")
    sig_df = svm_df if significance_threshold is None else svm_df[svm_df[pval_col] >= significance_threshold]

    clustered, used_threshold = cluster_by_gap(sig_df, scaff_col, pos_col, gap_threshold=gap_threshold,
                                                sensitivity=gap_sensitivity, method=gap_method)
    summary = summarize_blocks(clustered, pval_col, pos_col=pos_col)
    score = concentration_score(summary, top_fraction=top_fraction)
    print(f"  {len(summary)} candidate blocks, concentration score "
          f"(top {int(top_fraction * 100)}% of blocks) = {score:.2f}")

    if score >= concentration_cutoff:
        strategy = 'dense_cluster'
        keep_ids = summary[summary['n_snps'] >= min_snps_per_block].index
        final = clustered[clustered['haplotypeblock_nr'].isin(keep_ids)].copy()
        print(f"  -> DENSE mode: kept {len(keep_ids)} block(s) with >= {min_snps_per_block} SNPs each")
    else:
        strategy = 'top_n_fallback'
        top_positions = svm_df.nlargest(n=top_n_fallback, columns=pval_col)
        final, _ = cluster_by_gap(top_positions, scaff_col, pos_col, gap_threshold=used_threshold, method=gap_method)
        print(f"  -> FALLBACK mode: used top {top_n_fallback} positions, "
              f"still clustered with the {used_threshold:,.0f} bp desert threshold")

    final_summary = summarize_blocks(final, pval_col, pos_col=pos_col)
    return final, final_summary, strategy


# ---------------------------------------------------------------------------
# For candidates that have ALREADY been through final significance selection
# (via the OCSVM + FET grid-search threshold combo in select_candidates.py) -
# no top-N re-selection and no dense-vs-fallback decision needed here, since
# that decision was already made at the candidate-selection stage. This just
# clusters every row into haplotype blocks.
# ---------------------------------------------------------------------------
def build_haplotypeblocks_from_final_candidates(candidates_df, pval_col='neg_log_p_corrected',
                                                  scaff_col='scaff', pos_col='pos',
                                                  gap_threshold=None, gap_method='gmm',
                                                  gap_sensitivity=1.0, label=''):
    print(f"\n=== {label or 'dataset'}: clustering {len(candidates_df)} final candidate positions ===")
    clustered, used_threshold = cluster_by_gap(candidates_df, scaff_col, pos_col, gap_threshold=gap_threshold,
                                                sensitivity=gap_sensitivity, method=gap_method)
    summary = summarize_blocks(clustered, pval_col, pos_col=pos_col)
    return clustered, summary, used_threshold


# ---------------------------------------------------------------------------
# Final pipeline: LD-decay-anchored threshold (150bp, upper bound from
# Pfenninger & Foucault 2022, upper bound of LD decay in C. riparius),
# applied to the final, already-selected candidate list per replicate
# (from select_candidates.py's --emit-ocsvm-pct/--emit-fet-pct output).
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    GAP_THRESHOLD_BP = 150  # published LD-decay upper bound (Pfenninger & Foucault 2022)

    def load_candidates(path):
        df = pd.read_csv(path)
        # 'coordinate' looks like "Chr1.2007691" - split into scaff + pos
        split = df['coordinate'].str.split('.', n=1, expand=True)
        df['scaff'] = split[0]
        df['pos'] = split[1].astype(int)
        return df

    # Step 0 output files are still named with the old German working-labels
    # on disk; load from those, but key everything from here on by the
    # current English display names (Red/Blue/Gold/Green). All NEW output
    # files below are written using the English names.
    import plot_style as ps
    datasets = {}
    for file_label, display_label in ps.LEGACY_LABEL_MAP.items():
        datasets[display_label] = load_candidates(f"{file_label}_candidates_final.csv")

    results = {}
    overview_rows = []
    for label, cand_df in datasets.items():
        blocks, summary, used_threshold = build_haplotypeblocks_from_final_candidates(
            cand_df, gap_threshold=GAP_THRESHOLD_BP, label=label
        )
        results[label] = {'blocks': blocks, 'summary': summary}

        blocks.to_csv(f'{label}_haplotypeblocks.csv', index=False)
        summary.to_csv(f'{label}_haplotypeblock_summary.csv')

        n_isolated = int((summary['n_snps'] == 1).sum())
        n_clustered_blocks = int((summary['n_snps'] > 1).sum())
        n_snps_in_clusters = int(summary.loc[summary['n_snps'] > 1, 'n_snps'].sum())
        total_snps = int(summary['n_snps'].sum())
        clustered_fraction = n_snps_in_clusters / total_snps if total_snps else float('nan')

        print(f"{label}: {blocks['haplotypeblock_nr'].nunique()} blocks, "
              f"{len(blocks)} positions | {n_isolated} isolated singletons, "
              f"{n_clustered_blocks} multi-SNP blocks "
              f"({n_snps_in_clusters}/{total_snps} = {clustered_fraction:.1%} of positions "
              f"sit in an accumulation rather than alone)")

        overview_rows.append({
            'colour': label,
            'gap_threshold_bp': GAP_THRESHOLD_BP,
            'n_input_candidates': len(cand_df),
            'n_blocks': blocks['haplotypeblock_nr'].nunique(),
            'n_positions': len(blocks),
            'n_isolated_singletons': n_isolated,
            'n_multi_snp_blocks': n_clustered_blocks,
            'n_positions_in_clusters': n_snps_in_clusters,
            'fraction_positions_clustered': clustered_fraction,
        })

    overview = pd.DataFrame(overview_rows)
    overview.to_csv('haplotypeblock.csv', index=False)
    print("\nSaved per-colour blocks + summaries, and haplotypeblock_overview_all_colours.csv")

    # --- Supplementary materials: sensitivity table + sweep figure ---
    # (reuse build_sensitivity_table / plot_gap_sweep_grid, which expect
    # {label: (df, top_n)} - passing top_n=len(df) means "use everything",
    # matching the fact these are already final candidates, not a pool to
    # sub-select from)
    sensitivity_datasets = {label: (df, len(df)) for label, df in datasets.items()}
    sensitivity_table = build_sensitivity_table(
        sensitivity_datasets, thresholds=(75, 150, 300), pval_col='neg_log_p_corrected'
    )
    sensitivity_table.to_csv('supplementary_sensitivity_table.csv', index=False)
    print("Saved supplementary_sensitivity_table.csv")

    _, saved_path = plot_gap_sweep_grid(sensitivity_datasets, save_path='figures/haplotypeblock_sweep')
    print(f"Saved sweep figure: {saved_path}")

