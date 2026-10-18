#!/usr/bin/env python3
"""
Step 3: Replicate overlap analysis, three levels.

1. SNP level    - exact coordinate match across replicates' final candidates.
2. Block level  - genomic INTERVAL overlap between haplotype blocks across
                   replicates (blocks never share exact coordinates, so this
                   merges overlapping [start,end] spans across all four
                   replicates into consolidated loci, then tags each locus
                   by which replicate(s) contributed a block to it).
3. Gene level   - exact gene_id match across replicates' assigned genes
                   (default: any assigned gene, overlap or nearest; pass
                   --genes-overlap-only to restrict to direct hits only).

USAGE:
    python3 step3_overlap_analysis.py \
        --candidates Red=Red_candidates_final.csv Blue=Blue_candidates_final.csv Gold=Gold_candidates_final.csv Green=Green_candidates_final.csv \
        --blocks Red=Red_haplotypeblocks.csv Blue=Blue_haplotypeblocks.csv Gold=Gold_haplotypeblocks.csv Green=Green_haplotypeblocks.csv \
        --genes Red=Red_haplotypeblock_genes.csv Blue=Blue_haplotypeblock_genes.csv Gold=Gold_haplotypeblock_genes.csv Green=Green_haplotypeblock_genes.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np


def parse_kv(pairs):
    return {p.split('=', 1)[0]: p.split('=', 1)[1] for p in pairs}


# ---------------------------------------------------------------------------
# 1. SNP-level overlap: exact coordinate match
# ---------------------------------------------------------------------------
def snp_level_overlap(candidate_paths):
    sets = {}
    for label, path in candidate_paths.items():
        df = pd.read_csv(path)
        sets[label] = set(df['coordinate'])
        print(f"{label}: {len(sets[label])} candidate SNPs")
    return sets


# ---------------------------------------------------------------------------
# 2. Block-level overlap: merge overlapping intervals across replicates,
#    tag each merged locus by contributing replicate(s)
# ---------------------------------------------------------------------------
def block_level_overlap(block_paths):
    all_blocks = []
    for label, path in block_paths.items():
        df = pd.read_csv(path)
        extents = df.groupby('haplotypeblock_nr').agg(
            scaff=('scaff', 'first'), start=('pos', 'min'), end=('pos', 'max')
        ).reset_index()
        extents['replicate'] = label
        all_blocks.append(extents)
        print(f"{label}: {len(extents)} haplotype blocks")
    combined = pd.concat(all_blocks, ignore_index=True)
    combined = combined.sort_values(['scaff', 'start']).reset_index(drop=True)

    # merge overlapping/touching intervals genome-wide (across ALL replicates
    # at once), tracking which replicate(s) contributed to each merged locus
    loci = []
    current = None
    for _, row in combined.iterrows():
        if current is None:
            current = {'scaff': row['scaff'], 'start': row['start'], 'end': row['end'],
                       'replicates': {row['replicate']}}
        elif row['scaff'] == current['scaff'] and row['start'] <= current['end']:
            current['end'] = max(current['end'], row['end'])
            current['replicates'].add(row['replicate'])
        else:
            loci.append(current)
            current = {'scaff': row['scaff'], 'start': row['start'], 'end': row['end'],
                       'replicates': {row['replicate']}}
    if current is not None:
        loci.append(current)

    loci_df = pd.DataFrame(loci)
    loci_df['n_replicates'] = loci_df['replicates'].apply(len)
    loci_df['replicates_str'] = loci_df['replicates'].apply(lambda s: ';'.join(sorted(s)))

    # per-replicate sets, for the venn: which merged loci did each replicate contribute to
    sets = {}
    for label in block_paths:
        sets[label] = set(loci_df.index[loci_df['replicates'].apply(lambda s: label in s)])
    return sets, loci_df


# ---------------------------------------------------------------------------
# 3. Gene-level overlap: exact gene_id match
# ---------------------------------------------------------------------------
def gene_level_overlap(gene_paths, overlap_only=False):
    sets = {}
    for label, path in gene_paths.items():
        df = pd.read_csv(path)
        df = df[df['gene_id'].notna()]
        if overlap_only:
            df = df[df['relationship'] == 'overlap']
        sets[label] = set(df['gene_id'])
        print(f"{label}: {len(sets[label])} {'directly-overlapped' if overlap_only else 'assigned'} genes")
    return sets


# ---------------------------------------------------------------------------
# Reporting: counts table + 4-way venn, for any of the three levels
# ---------------------------------------------------------------------------
def summarize_and_plot(sets_dict, level_name, out_prefix):
    import plot_style as ps
    ps.apply_style()
    from venn import venn as venn_plot

    labels = list(sets_dict.keys())
    total = len(set.union(*sets_dict.values()))
    private_counts = {}
    for label in labels:
        others = set.union(*[s for l, s in sets_dict.items() if l != label])
        private_counts[label] = len(sets_dict[label] - others)
    shared_all = len(set.intersection(*sets_dict.values())) if len(labels) > 1 else None

    print(f"\n--- {level_name} overlap summary ---")
    print(f"Total distinct items across all replicates: {total}")
    for label in labels:
        pct = 100 * private_counts[label] / len(sets_dict[label]) if len(sets_dict[label]) else float('nan')
        print(f"  {label}: {len(sets_dict[label])} total, {private_counts[label]} private "
              f"({pct:.1f}% of {label}'s own items are unique to it)")
    if shared_all is not None:
        print(f"  Shared across ALL {len(labels)} replicates: {shared_all}")

    summary_rows = [{'replicate': l, 'n_total': len(sets_dict[l]), 'n_private': private_counts[l]}
                     for l in labels]
    summary_rows.append({'replicate': 'ALL (intersection)', 'n_total': shared_all, 'n_private': None})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f"{out_prefix}_overlap_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    fig = venn_plot({l: sets_dict[l] for l in labels},
                     fmt="{size}",
                     cmap=[ps.REPLICATE_COLORS.get(l, '#333333') for l in labels],
                     fontsize=ps.AXIS_FONTSIZE)
    fig_obj = fig.get_figure() if hasattr(fig, 'get_figure') else fig.figure
    saved = ps.save_fig(fig_obj, f"figures/{out_prefix}_venn", f"{level_name} overlap across replicates")
    print(f"Saved {saved}")
    return summary_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidates', nargs='+', required=True, help='label=path pairs (Step 0 output)')
    ap.add_argument('--blocks', nargs='+', required=True, help='label=path pairs (Step 1 output)')
    ap.add_argument('--genes', nargs='+', required=True, help='label=path pairs (Step 2 output)')
    ap.add_argument('--genes-overlap-only', action='store_true',
                     help='Restrict gene-level overlap to relationship==overlap rows only (default: all assigned genes)')
    args = ap.parse_args()

    import os
    os.makedirs('figures', exist_ok=True)

    print("=== SNP-level ===")
    snp_sets = snp_level_overlap(parse_kv(args.candidates))
    summarize_and_plot(snp_sets, "SNP-level", "snp")

    print("\n=== Block-level ===")
    block_sets, loci_df = block_level_overlap(parse_kv(args.blocks))
    loci_df.to_csv("haplotypeblock_merged_loci.csv", index=False)
    summarize_and_plot(block_sets, "Haplotype-block-level", "block")

    print("\n=== Gene-level ===")
    gene_sets = gene_level_overlap(parse_kv(args.genes), overlap_only=args.genes_overlap_only)
    summarize_and_plot(gene_sets, "Gene-level", "gene")


if __name__ == '__main__':
    main()
