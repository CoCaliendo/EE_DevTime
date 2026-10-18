#!/usr/bin/env python3
"""
Step 5: Selection coefficient over generations (Taus et al. 2017), per replicate.

s_t = ln[ (p_t * (1 - p_1)) / (p_1 * (1 - p_t)) ]

Baseline is Gen0 (biological numbering). Note: file names on disk use
gen1..gen7 (file gen1 = biological Gen0, the pre-selection baseline, per
user's long-standing pipeline convention) - this script loads from those
file names but reports/plots using biological Gen0-Gen6 throughout.

Computed over the candidate SNP set (Step 0 output), matching the SNPs
used for haplotype blocks. Frequencies are clipped to [1/(2N), 1-1/(2N)]
for a pool of N=100 individuals (200 chromosomes -> minimum representable
allele count = 1/200 = 0.005), to avoid log(0)/division-by-zero at fixed
or lost sites - this is a real resolution limit of the data, not an
arbitrary small-epsilon hack.

INPUT (R write.table default export, one file per replicate x generation):
    "x"
    "Chr1.358"      0.025
    "Chr1.554"      0.5
    ...

USAGE:
    python3 step5_selection_coefficient.py \
        --af-dir background_data \
        --candidates Red=Red_candidates_final.csv Blue=Blue_candidates_final.csv Gold=Gold_candidates_final.csv Green=Green_candidates_final.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd

POOL_SIZE = 100
MIN_FREQ = 1 / (2 * POOL_SIZE)   # 0.005
MAX_FREQ = 1 - MIN_FREQ

# maps the current English display labels to the German working-labels used
# in the background_data filenames (af_blau_gen1.txt etc)
FILE_LABEL_MAP = {'Red': 'rot', 'Blue': 'blau', 'Gold': 'gold', 'Green': 'grun'}


def parse_kv(pairs):
    return {p.split('=', 1)[0]: p.split('=', 1)[1] for p in pairs}


def load_af_file(path):
    """R write.table default export: quoted rowname (coordinate) + one
    unnamed value column, header line is just '"x"'."""
    df = pd.read_csv(path, sep=r'\s+', header=0, index_col=0, quotechar='"', engine='python')
    df.columns = ['af']
    df.index.name = 'coordinate'
    df = df.reset_index()
    df['coordinate'] = df['coordinate'].astype(str).str.strip('"')  # quotechar isn't reliably
    return df                                                        # stripped by pandas here


def compute_selection_coefficients(af_dir, file_label, candidate_coords, file_generations=range(1, 8),
                                    polarize=True):
    """Returns a DataFrame: coordinate x generation -> s.

    file_generations are the FILE suffixes (gen1..gen7 on disk). Output uses
    BIOLOGICAL generation numbering (Gen0-Gen6): file gen1 = biological Gen0
    (the pre-selection baseline, per user), so biological_gen = file_gen - 1.
    s=0 by definition at the baseline (file gen1 / biological Gen0).

    polarize: the raw af_*.txt files track a fixed allele (e.g. whichever is
    the reference at each site), not necessarily the one selection favors -
    so roughly half of candidate SNPs can show a net DECREASE even under
    strong selection, just because the tracked allele happens to be the one
    being swept away. If True (default, matches the polarization convention
    already used elsewhere in this project for trajectory classification),
    each SNP's entire s trajectory is sign-flipped if its net change from
    Gen0 to the final generation is negative, so every SNP's s reflects
    strength of directional change in whichever direction it actually moved,
    not an arbitrary reference-allele label. This is mathematically exact:
    flipping a SNP's frequency (1-p at every generation) is equivalent to
    negating its whole s trajectory, so no allele frequencies need touching.
    """
    af_by_gen = {}
    for g in file_generations:
        path = f"{af_dir}/af_{file_label}_gen{g}.txt"
        df = load_af_file(path)
        df = df[df['coordinate'].isin(candidate_coords)]
        af_by_gen[g] = df.set_index('coordinate')['af']

    baseline = af_by_gen[1].clip(MIN_FREQ, MAX_FREQ)

    s_by_gen = {}
    for g in file_generations:
        af_g = af_by_gen[g].reindex(baseline.index).clip(MIN_FREQ, MAX_FREQ)
        if g == 1:
            s_by_gen[g] = pd.Series(0.0, index=baseline.index)
        else:
            s_by_gen[g] = np.log((af_g * (1 - baseline)) / (baseline * (1 - af_g)))

    if polarize:
        final_gen = max(file_generations)
        sign = np.sign(s_by_gen[final_gen]).replace(0, 1)  # net-zero SNPs left unflipped
        for g in file_generations:
            s_by_gen[g] = s_by_gen[g] * sign

    rows = []
    for g in file_generations:
        biological_gen = g - 1
        for coord, val in s_by_gen[g].items():
            rows.append({'coordinate': coord, 'generation': biological_gen, 's': val})
    return pd.DataFrame(rows)


def summarize_and_plot(results_by_label, save_path):
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply_style()

    fig, ax = plt.subplots(figsize=(7, 5))
    summary_rows = []
    for label, df in results_by_label.items():
        color = ps.REPLICATE_COLORS.get(label, '#333333')
        agg = df.groupby('generation')['s'].agg(['mean', 'sem', 'count']).reset_index()
        summary_rows.append(agg.assign(replicate=label))
        ax.errorbar(agg['generation'], agg['mean'], yerr=agg['sem'],
                    color=color, marker='o', markersize=4, capsize=3, label=label)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('generation (Gen0 = pre-selection baseline)')
    ax.set_ylabel('selection coefficient (s), mean \u00b1 SEM')
    ax.legend(frameon=False)
    fig.tight_layout()
    saved = ps.save_fig(fig, save_path,
                         "mean selection coefficient over generations relative to Gen0, per replicate")
    print(f"Saved {saved}")

    summary = pd.concat(summary_rows, ignore_index=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--af-dir', required=True)
    ap.add_argument('--candidates', nargs='+', required=True, help='label=path pairs (Step 0 output)')
    ap.add_argument('--no-polarize', action='store_true',
                     help='Skip per-SNP polarization (default: polarize, so each SNP\'s s reflects '
                          'the direction it actually moved, not an arbitrary reference-allele label).')
    args = ap.parse_args()

    import os
    os.makedirs('figures', exist_ok=True)

    candidate_paths = parse_kv(args.candidates)
    results_by_label = {}
    for label, path in candidate_paths.items():
        cand = pd.read_csv(path)
        candidate_coords = set(cand['coordinate'])
        file_label = FILE_LABEL_MAP[label]
        print(f"{label}: computing s for {len(candidate_coords)} candidate SNPs (file label '{file_label}')")
        df = compute_selection_coefficients(args.af_dir, file_label, candidate_coords,
                                             polarize=not args.no_polarize)
        df.to_csv(f"{label}_selection_coefficients.csv", index=False)
        results_by_label[label] = df

    summary = summarize_and_plot(results_by_label, 'figures/selection_coefficient')
    summary.to_csv('selection_coefficient_summary.csv', index=False)
    print("\nSaved selection_coefficient_summary.csv")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
