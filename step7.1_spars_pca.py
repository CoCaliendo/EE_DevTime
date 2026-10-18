#!/usr/bin/env python3
"""
Sparse PCA 

Method: arcsine-square-root transform allele frequency, then sparse PCA, on
a matrix of 28 rows (4 replicates x 7 generations, Gen0-Gen6) x SNPs (a
random subsample of SNPs present AND variable across all 28 samples).

USAGE:
    python3 sparse_pca.py --af-dir background_data --frac 0.05 --seed 42
"""
import argparse
import numpy as np
import pandas as pd

FILE_LABELS = ['rot', 'blau', 'gold', 'grun']
DISPLAY_LABELS = {'rot': 'Red', 'blau': 'Blue', 'gold': 'Gold', 'grun': 'Green'}


def load_af(path):
    """File format: header line '"x"', then '"coordinate"\\tvalue' rows -
    R's own write.table default output. Fast C-engine read (skip the header
    line entirely rather than trying to parse it, since row/data column
    counts differ) - important at genome-wide scale (millions of rows)."""
    df = pd.read_csv(path, sep='\t', header=None, skiprows=1,
                      names=['coordinate', 'af'], engine='c',
                      dtype={'af': 'float32'})
    df['coordinate'] = df['coordinate'].str.strip('"')
    return df


def build_matrix(af_dir):
    """Memory-efficient approach for genome-wide scale. Two things matter here:

    1. Coordinates are packed into a single int64 each (chromosome id *
       multiplier + position) and held as plain numpy arrays instead of
       Python string sets/dicts - int64 arrays cost only 8 bytes per entry
       with no per-object overhead, versus 60-100+ bytes per Python string.
    2. Parsing is done with PLAIN PYTHON file iteration, not pandas -
       pd.read_csv + .str accessor operations (rsplit, strip) each create a
       full temporary copy of the string column internally, and at
       millions of rows those temporary copies were themselves enough to
       exhaust memory even though the FINAL packed-int representation is
       tiny. Reading line-by-line and converting directly to packed ints
       avoids ever materializing a full pandas string column at all.

    Pass 1: find the SNPs present in all 28 files, working entirely in
    packed-int64 space via np.intersect1d.
    Pass 2: re-read each file, keep only rows whose packed coordinate is in
    the intersection (via np.searchsorted on a sorted array), write
    directly into a preallocated matrix column.
    """
    file_keys = [f"{fl}_gen{g}" for fl in FILE_LABELS for g in range(1, 8)]
    paths = {key: f"{af_dir}/af_{key.split('_gen')[0]}_gen{key.split('_gen')[1]}.txt" for key in file_keys}

    chrom_to_id = {}
    MULTIPLIER = 2_000_000_000  # generous headroom above any real chromosome position

    def parse_coord(coord_str):
        """'\"Chr1.12345\"' -> packed int64. Adds new chromosome names to
        chrom_to_id on first encounter (tiny dict, ~handful of entries)."""
        coord_str = coord_str.strip().strip('"')
        chrom, pos = coord_str.rsplit('.', 1)
        cid = chrom_to_id.get(chrom)
        if cid is None:
            cid = len(chrom_to_id)
            chrom_to_id[chrom] = cid
        return cid * MULTIPLIER + int(pos)

    def load_coords_packed(path):
        """Coordinates only, packed to int64, via plain Python line
        iteration (no pandas) to avoid its internal temporary string-array
        copies at genome-wide scale."""
        def gen():
            with open(path) as f:
                next(f)  # header line: "x"
                for line in f:
                    coord_str = line.split('\t', 1)[0]
                    yield parse_coord(coord_str)
        packed = np.fromiter(gen(), dtype=np.int64)
        return np.unique(packed)  # sorts and dedupes

    def load_coords_and_values_packed(path):
        packed_list = []
        values_list = []
        with open(path) as f:
            next(f)
            for line in f:
                coord_str, val_str = line.rstrip('\n').split('\t', 1)
                packed_list.append(parse_coord(coord_str))
                values_list.append(float(val_str))
        packed = np.array(packed_list, dtype=np.int64)
        values = np.array(values_list, dtype=np.float32)
        del packed_list, values_list
        order = np.argsort(packed, kind='stable')
        packed, values = packed[order], values[order]
        # dedupe on packed id, keep first occurrence (matches earlier duplicate-coordinate handling)
        keep = np.ones(len(packed), dtype=bool)
        keep[1:] = packed[1:] != packed[:-1]
        return packed[keep], values[keep]

    print("Pass 1/2: finding SNPs present in all 28 files (coordinates only, packed as int64)...")
    common = None
    for key in file_keys:
        packed = load_coords_packed(paths[key])
        common = packed if common is None else np.intersect1d(common, packed, assume_unique=True)
        print(f"  {key}: running intersection size = {len(common)}")
        del packed

    print(f"  {len(common)} SNPs present in all 28 samples")
    n_common = len(common)  # already sorted (np.unique/np.intersect1d both sort)

    print("Pass 2/2: loading values for the shared SNPs only...")
    X_full = np.full((n_common, len(file_keys)), np.nan, dtype=np.float32)
    for i, key in enumerate(file_keys):
        packed, values = load_coords_and_values_packed(paths[key])
        pos_in_common = np.searchsorted(common, packed)
        valid = (pos_in_common < n_common) & (common[np.clip(pos_in_common, 0, n_common - 1)] == packed)
        X_full[pos_in_common[valid], i] = values[valid]
        print(f"  Loaded {key}")
        del packed, values, pos_in_common, valid

    id_to_chrom = {v: k for k, v in chrom_to_id.items()}
    chrom_ids = common // MULTIPLIER
    positions = common % MULTIPLIER
    coord_strings = [f"{id_to_chrom[c]}.{p}" for c, p in zip(chrom_ids, positions)]

    merged = pd.DataFrame(X_full, index=pd.Index(coord_strings), columns=file_keys)
    print(f"  Final merged matrix: {merged.shape[0]} SNPs x {merged.shape[1]} samples")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--af-dir', required=True)
    ap.add_argument('--frac', type=float, default=0.05, help='Fraction of VARIABLE SNPs to subsample')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-components', type=int, default=2)
    ap.add_argument('--alpha', type=float, default=1.0, help='Sparsity penalty - if 0 loadings are nonzero, try SMALLER; if nearly ALL are nonzero (not actually sparse), try LARGER')
    ap.add_argument('--rebuild', action='store_true', help='Force rebuilding the input matrix from the genome-wide files even if a cache exists (needed if --frac or --seed changed, or the underlying data changed)')
    args = ap.parse_args()

    np.random.seed(args.seed)

    import os
    cache_path = f'sparse_pca_input_cache_frac{args.frac}_seed{args.seed}.npz'
    if os.path.exists(cache_path) and not args.rebuild:
        print(f"Found cached input matrix ({cache_path}) - loading it instead of re-merging "
              f"2.5M+ genome-wide SNPs from scratch. Use --rebuild to force regenerating it "
              f"(needed if you change --frac or --seed).")
        cached = np.load(cache_path, allow_pickle=True)
        X = cached['X']
        sample_names = cached['sample_names'].tolist()
        subsample_index = cached['subsample_index']
    else:
        merged = build_matrix(args.af_dir)

        print("Filtering to variable SNPs (nonzero variance across all 28 samples)...")
        variances = merged.var(axis=1)
        variable = merged.loc[variances > 0]
        print(f"  {len(variable)} of {len(merged)} SNPs are variable")

        n_sub = int(round(args.frac * len(variable)))
        sub_idx = np.random.choice(variable.index, size=n_sub, replace=False)
        subsample = variable.loc[sub_idx]
        print(f"  Subsampled {n_sub} SNPs ({args.frac:.0%} of the variable set)")

        print("Applying arcsine-square-root transform...")
        transformed = 2 * np.arcsin(np.sqrt(subsample.clip(0, 1)))

        X = transformed.T.values
        sample_names = transformed.columns.tolist()
        subsample_index = subsample.index.values

        np.savez(cache_path, X=X, sample_names=np.array(sample_names, dtype=object),
                 subsample_index=subsample_index)
        print(f"Saved input matrix cache to {cache_path} - re-runs with the same --frac/--seed "
              f"will reuse this instead of re-loading the genome-wide files, so alpha can be "
              f"tuned quickly from here.")

    print("Running sparse PCA (sklearn SparsePCA)...")
    from sklearn.decomposition import SparsePCA
    model = SparsePCA(n_components=args.n_components, alpha=args.alpha, random_state=args.seed)
    scores = model.fit_transform(X)
    n_nonzero = int((model.components_ != 0).sum())
    pct_nonzero = 100 * n_nonzero / model.components_.size
    print(f"  {n_nonzero} of {model.components_.size} loadings are nonzero ({pct_nonzero:.1f}%), alpha={args.alpha}")
    if pct_nonzero == 0:
        print("  -> ALL ZERO: alpha is too high (too much sparsity penalty) - try a SMALLER value (e.g. /10).")
    elif pct_nonzero > 50:
        print("  -> Not actually sparse (most SNPs contribute to every PC) - try a LARGER alpha (e.g. x10) "
              "for a genuinely interpretable, small set of driving SNPs.")

    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(args.n_components)])
    scores_df['sample'] = sample_names
    scores_df['replicate'] = [DISPLAY_LABELS[s.split('_gen')[0]] for s in sample_names]
    scores_df['generation'] = [int(s.split('_gen')[1]) - 1 for s in sample_names]  # file gen1 = biological Gen0

    out_path = 'sparse_pca_scores.csv'
    scores_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    loadings_df = pd.DataFrame(model.components_.T, columns=[f"PC{i+1}_loading" for i in range(args.n_components)])
    loadings_df['coordinate'] = subsample_index
    loadings_df.to_csv('sparse_pca_loadings.csv', index=False)
    print("Saved sparse_pca_loadings.csv")

    print(f"\nFinal reconstruction error: {model.error_[-1] if hasattr(model, 'error_') and len(model.error_) else 'n/a'}")
    print("(SparsePCA components aren't necessarily orthogonal, so a standard 'variance explained' "
          "doesn't cleanly apply here - reconstruction error is the closest fit diagnostic available.)")


if __name__ == '__main__':
    main()
