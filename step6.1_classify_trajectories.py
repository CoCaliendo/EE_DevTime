#!/usr/bin/env python3
"""
Step 6b: Classify polarized peak-SNP trajectories into Type0 (early) vs
Type1 (late) vs Never_crossed, based on the generation each trajectory
first exceeds ITS OWN properly-conditioned neutral-drift envelope.

IMPORTANT: the envelope is built PER SNP, simulated from that SNP's own
Gen0 frequency - not one aggregate curve for the whole genome. Neutral
drift variance depends on starting frequency (~p(1-p)/(2Ne) per
generation), so a rare SNP (p0=0.02) and a common SNP (p0=0.5) have very
different drift ceilings; comparing every SNP in a replicate against one
shared genome-wide envelope silently miscalibrates the comparison for any
SNP whose starting frequency differs from the genome-wide bulk. This
matches how the original R analysis conditioned wf.traj() on each SNP's
own p0, rather than an earlier (incorrect) whole-genome-aggregate version
of this script.

Type0: crosses its own envelope by Gen3 (inclusive)
Type1: crosses its own envelope after Gen3
Never_crossed: does not cross within Gen0-Gen6

Also computes five descriptive features (for the secondary validation
figure) but they do NOT drive the classification itself:
    early_rise      : trajectory[2] - trajectory[0]  (rise by Gen2)
    gen_exceed       : first generation the SNP's own envelope is crossed
    max_jump         : largest single-generation increase
    gen_of_max_jump  : generation at which that max jump occurs
    slope_ratio      : (rise from Gen0-Gen3) / (rise from Gen3-Gen6)
"""
import argparse
import numpy as np
import pandas as pd
from wf_drift import wf_traj_neutral

GEN_COLS = list(range(7))
EARLY_LATE_CUTOFF = 3
NE_VALUES = {'Red': 17.46576, 'Blue': 16.70802, 'Gold': 22.51174, 'Green': 17.10349}


def build_per_snp_envelope(p0_array, Ne, quantile=0.95, generations=GEN_COLS, n_sims_per_snp=1000, seed=42):
    """For each SNP's own starting frequency, simulate n_sims_per_snp
    independent neutral-drift replicates and take the quantile AT EACH
    GENERATION, per SNP. Returns array shape (len(p0_array), len(generations))."""
    n_snps = len(p0_array)
    p0_expanded = np.repeat(p0_array, n_sims_per_snp)
    sim = wf_traj_neutral(p0_expanded, Ne, generations, seed=seed)

    envelope = np.zeros((n_snps, len(generations)))
    for gi, g in enumerate(generations):
        vals = sim[g].reshape(n_snps, n_sims_per_snp)
        envelope[:, gi] = np.nanquantile(vals, quantile, axis=1)
    return envelope


def classify_and_featurize(traj_df, Ne, quantile=0.95, n_sims_per_snp=1000, seed=42):
    """traj_df: output of extract_peak_trajectories.py (one row per peak SNP,
    columns 0..6 = polarized frequency per generation)."""
    df = traj_df.copy()
    traj_arr = df[GEN_COLS].values
    p0_array = traj_arr[:, 0]  # each SNP's own Gen0 (post-polarization) frequency

    env_arr = build_per_snp_envelope(p0_array, Ne, quantile=quantile,
                                      generations=GEN_COLS, n_sims_per_snp=n_sims_per_snp, seed=seed)

    exceeds = traj_arr > env_arr
    gen_exceed = np.full(len(df), np.nan)
    for i in range(len(df)):
        crossed = np.where(exceeds[i])[0]
        if len(crossed) > 0:
            gen_exceed[i] = crossed[0]
    df['gen_exceed'] = gen_exceed
    for gi, g in enumerate(GEN_COLS):
        df[f'envelope_gen{g}'] = env_arr[:, gi]

    df['type'] = np.select(
        [
            (df['gen_exceed'].notna()) & (df['gen_exceed'] <= EARLY_LATE_CUTOFF),
            (df['gen_exceed'].notna()) & (df['gen_exceed'] > EARLY_LATE_CUTOFF),
        ],
        ['Type0_early', 'Type1_late'],
        default='Never_crossed'
    )

    df['early_rise'] = df[2] - df[0]
    diffs = np.diff(traj_arr, axis=1)
    df['max_jump'] = diffs.max(axis=1)
    df['gen_of_max_jump'] = diffs.argmax(axis=1)
    rise_early = df[3] - df[0]
    rise_late = df[6] - df[3]
    with np.errstate(divide='ignore', invalid='ignore'):
        df['slope_ratio'] = np.where(rise_late != 0, rise_early / rise_late, np.nan)

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trajectories', required=True, help='{label}_peak_trajectories.csv from Step 6a')
    ap.add_argument('--label', required=True, help='Replicate label (Red/Blue/Gold/Green) - used to look up Ne')
    ap.add_argument('--quantile', type=float, default=0.95)
    ap.add_argument('--n-sims-per-snp', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    traj_df = pd.read_csv(args.trajectories)
    traj_df.columns = [int(c) if c.lstrip('-').isdigit() else c for c in traj_df.columns]
    Ne = NE_VALUES[args.label]

    classified = classify_and_featurize(traj_df, Ne, quantile=args.quantile,
                                         n_sims_per_snp=args.n_sims_per_snp, seed=args.seed)
    n_type0 = (classified['type'] == 'Type0_early').sum()
    n_type1 = (classified['type'] == 'Type1_late').sum()
    n_never = (classified['type'] == 'Never_crossed').sum()
    n_total = len(classified)
    print(f"{args.label} (Ne={Ne}): {n_type0}/{n_total} ({n_type0/n_total:.1%}) Type0 (early, crosses by Gen{EARLY_LATE_CUTOFF}); "
          f"{n_type1}/{n_total} ({n_type1/n_total:.1%}) Type1 (crosses after Gen{EARLY_LATE_CUTOFF}); "
          f"{n_never}/{n_total} ({n_never/n_total:.1%}) never crosses within Gen0-Gen6")

    # Diagnostic: spot-check a low-p0 SNP and a high-p0 SNP's own envelope,
    # so it's visible that the envelope now genuinely differs by starting frequency.
    sorted_by_p0 = classified.sort_values(0)
    for label, row in [('lowest-p0 SNP', sorted_by_p0.iloc[0]), ('highest-p0 SNP', sorted_by_p0.iloc[-1])]:
        env_vals = [row[f'envelope_gen{g}'] for g in GEN_COLS]
        traj_vals = [row[g] for g in GEN_COLS]
        print(f"  {label} (p0={row[0]:.4f}): own envelope={[f'{v:.3f}' for v in env_vals]}, "
              f"own trajectory={[f'{v:.3f}' for v in traj_vals]}, gen_exceed={row['gen_exceed']}")

    out_path = f"{args.label}_classified_trajectories.csv"
    classified.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
