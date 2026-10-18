#!/usr/bin/env python3
"""
Step 2: Nearest-gene assignment for haplotype blocks.

For each haplotype block (computed directly from {colour}_haplotypeblocks.csv,
not the summary file - see note in chat about a scaff-column gap there),
finds the nearest gene feature in a BRAKER3-style GFF3: either the
overlapping gene (distance 0, relationship='overlap'), or the closest gene
upstream/downstream on the same scaffold.

USAGE:
    python3 step2_gene_annotation.py --gff braker3_2024_proteins.gff3 \
        --blocks Red_haplotypeblocks.csv --label Red
    (repeat per replicate, or use --blocks-glob "*_haplotypeblocks.csv" to do all at once)
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd


def load_genes_from_gff(gff_path):
    """Parse only 'gene' feature rows. Returns scaff, gene_start, gene_end,
    strand, gene_id (from the ID= attribute)."""
    rows = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            chrom, start, end, strand, attrs = fields[0], int(fields[3]), int(fields[4]), fields[6], fields[8]
            m = re.search(r'ID=([^;]+)', attrs)
            gene_id = m.group(1) if m else None
            rows.append({'scaff': chrom, 'gene_start': start, 'gene_end': end,
                         'strand': strand, 'gene_id': gene_id})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        sys.exit(f"ERROR: no 'gene' feature rows found in {gff_path} - check column 3 values with "
                  f"`cut -f3 {gff_path} | sort | uniq -c`")
    print(f"Loaded {len(df)} genes from {gff_path} across {df['scaff'].nunique()} scaffolds")
    return df


def block_extents_from_positions(blocks_path):
    """Compute each haplotype block's [start, end] and scaffold directly
    from the per-position blocks CSV (has scaff, pos, haplotypeblock_nr
    columns), since the summary CSV from step 1 is missing 'scaff'."""
    df = pd.read_csv(blocks_path)
    for col in ('scaff', 'pos', 'haplotypeblock_nr'):
        if col not in df.columns:
            sys.exit(f"ERROR: expected column '{col}' in {blocks_path}, found: {list(df.columns)}")
    extents = df.groupby('haplotypeblock_nr').agg(
        scaff=('scaff', 'first'),
        start=('pos', 'min'),
        end=('pos', 'max'),
        n_snps=('pos', 'size'),
    ).reset_index()
    return extents


def nearest_gene_for_block(block_scaff, block_start, block_end, genes_df):
    cand = genes_df[genes_df['scaff'] == block_scaff].copy()
    if len(cand) == 0:
        return {'gene_id': None, 'strand': None, 'distance_bp': np.nan, 'relationship': 'no_genes_on_scaffold'}

    is_upstream = cand['gene_end'] < block_start      # gene entirely before the block
    is_downstream = cand['gene_start'] > block_end     # gene entirely after the block
    # anything not upstream and not downstream overlaps the block

    cand['distance_bp'] = 0
    cand.loc[is_upstream, 'distance_bp'] = block_start - cand.loc[is_upstream, 'gene_end']
    cand.loc[is_downstream, 'distance_bp'] = cand.loc[is_downstream, 'gene_start'] - block_end

    best = cand.loc[cand['distance_bp'].idxmin()]
    if best['distance_bp'] == 0:
        relationship = 'overlap'
    elif is_upstream[best.name]:
        relationship = 'upstream'
    else:
        relationship = 'downstream'

    return {'gene_id': best['gene_id'], 'strand': best['strand'],
            'distance_bp': int(best['distance_bp']), 'relationship': relationship}


def annotate_blocks(blocks_path, genes_df, label='', far_gene_warning_bp=50_000):
    extents = block_extents_from_positions(blocks_path)
    results = []
    for _, row in extents.iterrows():
        gene_info = nearest_gene_for_block(row['scaff'], row['start'], row['end'], genes_df)
        results.append({
            'haplotypeblock_nr': row['haplotypeblock_nr'],
            'scaff': row['scaff'], 'block_start': row['start'], 'block_end': row['end'],
            'n_snps': row['n_snps'], **gene_info,
        })
    out = pd.DataFrame(results)

    n_no_genes = (out['relationship'] == 'no_genes_on_scaffold').sum()
    n_far = (out['distance_bp'] > far_gene_warning_bp).sum()
    print(f"{label}: {len(out)} blocks annotated. "
          f"{(out['relationship'] == 'overlap').sum()} overlap a gene directly, "
          f"{n_far} have their nearest gene >{far_gene_warning_bp:,} bp away, "
          f"{n_no_genes} are on a scaffold with no annotated genes at all.")
    if n_no_genes > 0:
        print(f"  -> scaffolds with no genes: {sorted(out.loc[out['relationship']=='no_genes_on_scaffold', 'scaff'].unique())}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gff', required=True)
    ap.add_argument('--blocks', help='One {colour}_haplotypeblocks.csv file')
    ap.add_argument('--label', help='Replicate label (required with --blocks)')
    ap.add_argument('--all', action='store_true',
                     help='Run for Red/Blue/Gold/Green at once, expecting '
                          '{label}_haplotypeblocks.csv for each in the current directory.')
    ap.add_argument('--far-gene-warning-bp', type=int, default=50_000)
    args = ap.parse_args()

    genes_df = load_genes_from_gff(args.gff)

    if args.all:
        for label in ['Red', 'Blue', 'Gold', 'Green']:
            path = f"{label}_haplotypeblocks.csv"
            out = annotate_blocks(path, genes_df, label=label,
                                   far_gene_warning_bp=args.far_gene_warning_bp)
            out.to_csv(f"{label}_haplotypeblock_genes.csv", index=False)
            print(f"Saved {label}_haplotypeblock_genes.csv\n")
        return

    if not args.blocks:
        sys.exit("Provide --blocks <path> --label <name> (single replicate), or --all (all four at once).")

    out = annotate_blocks(args.blocks, genes_df, label=args.label,
                           far_gene_warning_bp=args.far_gene_warning_bp)
    out_path = f"{args.label}_haplotypeblock_genes.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
