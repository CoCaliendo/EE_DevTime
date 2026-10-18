#!/usr/bin/env python3
"""
Step 11a: GO enrichment dot plot, one panel per replicate (from step9 go_enrichment_results.csv).

x = -log10(FDR), SAME scale in every panel (so replicates are comparable); dot size = number of
candidate genes annotated to the term; dot colour = replicate; black outline = term also
significant in at least one other replicate. Terms are sorted by FDR, and terms with an
IDENTICAL candidate-gene set are collapsed into one row (GO hierarchy propagation otherwise
fills the plot with near-duplicates of the same signal); the label shows how many were merged.
A replicate with no significant terms gets an empty panel saying so.

USAGE:
    python3 step11_plot_go_enrichment.py --results go_enrichment_results.csv
    python3 step11_plot_go_enrichment.py --results go_enrichment_results.csv --namespaces BP,CC --top-n 12
"""
import argparse, os, textwrap
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_style as ps

ORDER = ['Red', 'Blue', 'Gold', 'Green']

def prepare(df, namespaces, top_n, collapse=True):
    d = df[df['namespace'].isin(namespaces)].copy()
    d['n_study'] = d['ratio_in_study'].str.split('/').str[0].astype(int)
    d['n_pop'] = d['ratio_in_pop'].str.split('/').str[0].astype(int)
    d = d.sort_values(['p_fdr_bh', 'n_pop'])
    if collapse:
        d['study_genes'] = d['study_genes'].fillna('')
        kept, seen = [], {}
        for _, r in d.iterrows():
            key = r['study_genes'] if r['study_genes'] else r['GO_id']
            if key in seen:
                kept[seen[key]]['n_merged'] += 1
            else:
                seen[key] = len(kept); r = r.copy(); r['n_merged'] = 0; kept.append(r)
        d = pd.DataFrame(kept)
    else:
        d['n_merged'] = 0
    return d.head(top_n)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--namespaces', default='BP', help="comma list of BP,CC,MF (default BP)")
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--no-collapse', action='store_true')
    args = ap.parse_args()
    os.makedirs('figures', exist_ok=True); ps.apply_style()

    res = pd.read_csv(args.results)
    ns = [x.strip() for x in args.namespaces.split(',')]
    reps = [r for r in ORDER if r in set(res['replicate']) or True]
    # terms significant in >=2 replicates (any namespace)
    shared = {g for g, c in res.groupby('GO_id')['replicate'].nunique().items() if c >= 2}

    panels = {r: prepare(res[res.replicate == r], ns, args.top_n, not args.no_collapse) for r in reps}
    heights = [max(1.0, 0.45 * len(panels[r]) + 0.5) for r in reps]
    fig, axes = plt.subplots(len(reps), 1, figsize=(9.5, sum(heights) + 1.2), sharex=True,
                             gridspec_kw={'height_ratios': heights})
    xmax = max([(-np.log10(d['p_fdr_bh'].astype(float).clip(lower=1e-300))).max() for d in panels.values() if len(d)] + [1]) * 1.08

    for ax, r in zip(axes, reps):
        d = panels[r]; color = ps.REPLICATE_COLORS[r]
        if len(d) == 0:
            ax.text(0.5, 0.5, 'no significant terms', transform=ax.transAxes, ha='center', va='center',
                    fontsize=ps.TICK_FONTSIZE, color='grey'); ax.set_yticks([])
        else:
            y = np.arange(len(d))[::-1]
            x = (-np.log10(d['p_fdr_bh'].astype(float).clip(lower=1e-300))).values
            ax.scatter(x, y, s=30 + 14 * d['n_study'].values, color=color, alpha=0.85,
                       edgecolors=['black' if g in shared else 'none' for g in d['GO_id']], linewidths=1.8, zorder=3)
            labels = [textwrap.fill(n, 46) + (f'  (+{m})' if m else '') for n, m in zip(d['name'], d['n_merged'])]
            ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=ps.TICK_FONTSIZE - 1)
            ax.set_ylim(-0.7, len(d) - 0.3)
        ax.grid(axis='x', color='#e6e6e6', zorder=0); ax.set_xlim(0, xmax)
        ax.text(0.99, 0.06, r, transform=ax.transAxes, ha='right', va='bottom', color=color,
                fontweight='bold', fontsize=ps.AXIS_FONTSIZE)
    axes[-1].set_xlabel('-log10(FDR)')

    allc = pd.concat([d for d in panels.values() if len(d)])
    sizes = sorted({int(allc['n_study'].min()), int(np.median(allc['n_study'])), int(allc['n_study'].max())})
    handles = [Line2D([], [], marker='o', ls='', color='grey', markersize=np.sqrt(30 + 14 * s), label=f'{s} genes') for s in sizes]
    handles.append(Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor='black',
                          markeredgewidth=1.8, markersize=9, label='also significant in another replicate'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=ps.TICK_FONTSIZE - 1)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    saved = ps.save_fig(fig, 'figures/go_enrichment_dotplot', f"GO enrichment top terms per replicate ({'+'.join(ns)})")
    print('Saved', saved)

if __name__ == '__main__':
    main()
