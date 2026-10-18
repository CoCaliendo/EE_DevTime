#!/usr/bin/env python3
"""
Step 9: GO-term enrichment analysis, per replicate, using goatools.

Background/population: all genes with at least one GO annotation in
gene_to_GO.txt (the natural universe for this test - genes with zero GO
annotations can never be "enriched" for anything and their inclusion vs.
exclusion from the population is a common source of debate; this script
uses the annotated-only convention, the most common default in goatools'
own examples - state this explicitly in methods).

Study/foreground set, per replicate: the genes assigned to that replicate's
haplotype blocks (Step 2 output, {colour}_haplotypeblock_genes.csv,
gene_id column). By default uses ALL assigned genes (both direct-overlap
and nearest-gene rows), matching the convention already used for Step 3's
gene-level overlap analysis; pass --overlap-only to restrict to direct
relationship=='overlap' hits only.

Method: Fisher's exact test per GO term (goatools GOEnrichmentStudyNS,
namespace-aware: BP/CC/MF tested and reported separately), with GO-hierarchy
propagation (a gene annotated to a specific term also counts toward all of
that term's ancestors) and Benjamini-Hochberg FDR correction across all
terms tested. Only "enriched" (over-represented, not "purified"/depleted)
terms at FDR < 0.05 are kept in the output.

USAGE:
    python3 step9_go_enrichment.py \
        --gene-to-go gene_to_GO.txt \
        --obo go-basic.obo \
        --genes Red=Red_haplotypeblock_genes.csv Blue=Blue_haplotypeblock_genes.csv Gold=Gold_haplotypeblock_genes.csv Green=Green_haplotypeblock_genes.csv
"""
import argparse
import pandas as pd
from collections import defaultdict

# Compatibility shim: goatools 1.6.5 imports multipletests from the old
# statsmodels.sandbox.stats.multicomp location, which no longer has it in
# current statsmodels (moved to statsmodels.stats.multitest). Patch it in
# rather than downgrade statsmodels.
try:
    import statsmodels.sandbox.stats.multicomp as _legacy_multicomp
    if not hasattr(_legacy_multicomp, 'multipletests'):
        from statsmodels.stats.multitest import multipletests as _multipletests
        _legacy_multicomp.multipletests = _multipletests
except ImportError:
    pass


def parse_kv(pairs):
    return {p.split('=', 1)[0]: p.split('=', 1)[1] for p in pairs}


def load_associations(gene_to_go_path):
    """Returns dict {gene_id: set(GO_ids)}."""
    assoc = defaultdict(set)
    with open(gene_to_go_path) as f:
        for line in f:
            gene, go = line.rstrip('\n').split('\t')
            assoc[gene].add(go)
    return dict(assoc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gene-to-go', required=True, help='gene_to_GO.txt (2-column: gene_id, GO term)')
    ap.add_argument('--obo', required=True, help='go-basic.obo path')
    ap.add_argument('--genes', nargs='+', required=True, help='label=path pairs ({colour}_haplotypeblock_genes.csv, Step 2 output)')
    ap.add_argument('--overlap-only', action='store_true',
                     help='Restrict candidate genes to relationship==overlap rows only (default: all assigned genes)')
    ap.add_argument('--fdr', type=float, default=0.05)
    ap.add_argument('--diagnostic-top-n', type=int, default=20,
                     help='For each replicate, save this many top terms by uncorrected p-value regardless '
                          'of significance, to a SEPARATE diagnostic file - useful for checking whether a '
                          'replicate with 0 FDR-significant hits has a weak trend or genuinely nothing.')
    args = ap.parse_args()

    from goatools.obo_parser import GODag
    from goatools.go_enrichment import GOEnrichmentStudy

    print(f"Loading ontology from {args.obo}...")
    obodag = GODag(args.obo)

    print(f"Loading gene-to-GO associations from {args.gene_to_go}...")
    assoc = load_associations(args.gene_to_go)
    population = set(assoc.keys())
    print(f"  {len(population)} genes with at least one GO annotation (background population)")

    study_obj = GOEnrichmentStudy(
        population, assoc, obodag,
        propagate_counts=True,
        alpha=0.05,
        methods=['fdr_bh'],
    )

    gene_paths = parse_kv(args.genes)
    all_results = []
    all_diagnostic = []
    for label, path in gene_paths.items():
        df = pd.read_csv(path)
        if args.overlap_only:
            df = df[df['relationship'] == 'overlap']
        candidate_genes = set(df['gene_id'].dropna().unique())
        study_genes = candidate_genes & population
        n_dropped = len(candidate_genes) - len(study_genes)
        print(f"\n{label}: {len(candidate_genes)} candidate genes, {len(study_genes)} have a GO annotation "
              f"({n_dropped} dropped - no GO annotation, cannot contribute to this test)")

        results = study_obj.run_study(study_genes)
        sig_enriched = [r for r in results if r.enrichment == 'e' and r.p_fdr_bh < args.fdr]
        print(f"  {len(sig_enriched)} significantly enriched GO terms (FDR < {args.fdr})")

        for r in sig_enriched:
            all_results.append({
                'replicate': label,
                'GO_id': r.GO,
                'namespace': r.NS,
                'name': r.name,
                'p_uncorrected': r.p_uncorrected,
                'p_fdr_bh': r.p_fdr_bh,
                'ratio_in_study': f"{r.ratio_in_study[0]}/{r.ratio_in_study[1]}",
                'ratio_in_pop': f"{r.ratio_in_pop[0]}/{r.ratio_in_pop[1]}",
                'study_genes': ','.join(sorted(r.study_items)) if hasattr(r, 'study_items') else '',
            })

        # Diagnostic only (NOT significant results): top-N terms by uncorrected
        # p-value regardless of FDR outcome, so a replicate with 0 significant
        # hits can still be checked for "a weak trend that didn't survive
        # correction" vs. "genuinely nothing at all" - these are different
        # findings and this file must never be reported as if it were the
        # real (FDR-corrected) result.
        enriched_only = [r for r in results if r.enrichment == 'e']
        top_n = sorted(enriched_only, key=lambda r: r.p_uncorrected)[:args.diagnostic_top_n]
        for rank, r in enumerate(top_n, 1):
            all_diagnostic.append({
                'replicate': label, 'rank': rank, 'GO_id': r.GO, 'namespace': r.NS, 'name': r.name,
                'p_uncorrected': r.p_uncorrected, 'p_fdr_bh': r.p_fdr_bh,
                'passed_fdr': r.p_fdr_bh < args.fdr,
                'ratio_in_study': f"{r.ratio_in_study[0]}/{r.ratio_in_study[1]}",
                'ratio_in_pop': f"{r.ratio_in_pop[0]}/{r.ratio_in_pop[1]}",
            })

    out_df = pd.DataFrame(all_results)
    out_path = 'go_enrichment_results.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(out_df)} total significant term x replicate rows)")

    diag_df = pd.DataFrame(all_diagnostic)
    diag_path = 'go_enrichment_DIAGNOSTIC_top_terms.csv'
    diag_df.to_csv(diag_path, index=False)
    print(f"Saved {diag_path} - top {args.diagnostic_top_n} terms per replicate by uncorrected p-value, "
          f"REGARDLESS of significance ('passed_fdr' column shows which actually cleared FDR<{args.fdr}). "
          f"This is a diagnostic file, not a results file - do not report rows with passed_fdr=False as findings.")


if __name__ == '__main__':
    main()
