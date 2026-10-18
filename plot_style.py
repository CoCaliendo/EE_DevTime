"""
Shared plotting style for the EE rebuild (haplotype blocks, theta, selection
coefficients, trajectories, PCA, GO enrichment, etc).

Import this in every script and use REPLICATE_COLORS / apply_style() /
save_fig() so every figure in the rebuild looks consistent, without
having to repeat these settings in each notebook cell.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Fixed replicate color mapping - use this dict everywhere a replicate is
# colored (lines, bars, points, violins, etc), rather than letting matplotlib
# assign colors automatically, so the same replicate is always the same color
# across every figure in the whole rebuild.
# ---------------------------------------------------------------------------
REPLICATE_COLORS = {
    'Red':   '#D62728',
    'Blue':  '#1F77B4',
    'Gold':  '#D4AC0D',   # not pure yellow - stays legible on white
    'Green': '#2CA02C',
}

# Earlier steps (candidate selection) were run with German working-labels and
# their output files are still named that way on disk (rot_candidates_final.csv
# etc). Use this once, when loading those existing files, to key everything
# downstream by the current English display names instead. Do NOT use German
# labels for any new output going forward.
LEGACY_LABEL_MAP = {
    'rot': 'Red',
    'blau': 'Blue',
    'gold': 'Gold',
    'grün': 'Green',
}

# Non-replicate categorical colors used across multiple plot types (e.g.
# trajectory type, selected-vs-background) - kept separate from replicate
# colors so the two color schemes are never visually confused.
CATEGORY_COLORS = {
    'early_gradual': '#4B0082',   # dark indigo - Type 0 / early-rise trajectories
    'late_abrupt':   '#B19CD9',   # light purple - Type 1 / late-rise trajectories
    'selected':      '#D62728',
    'background':    '#7F7F7F',
}

AXIS_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12


def lighten(hex_color, amount=0.55):
    """Blend a hex color toward white by `amount` (0 = unchanged, 1 = white).
    Used to derive a Type1 shade from a replicate's own base color, so a
    plot can distinguish two sub-groups within one replicate's color."""
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(hex_color)
    return tuple(c + (1 - c) * amount for c in rgb)

def apply_style():
    """Call once at the top of a notebook/script. Sets global rcParams so
    every subsequent plt figure gets the right font sizes by default, and
    turns off automatic titles so nobody accidentally leaves one on."""
    mpl.rcParams.update({
        'axes.labelsize': AXIS_FONTSIZE,
        'xtick.labelsize': TICK_FONTSIZE,
        'ytick.labelsize': TICK_FONTSIZE,
        'legend.fontsize': LEGEND_FONTSIZE,
        'axes.titlesize': 0,       # discourage in-plot titles by default
        'figure.dpi': 150,
    })

def strip_title(ax):
    """Explicitly remove any title that got set on an axis - call this right
    before saving if a plotting library (e.g. seaborn) auto-adds one."""
    ax.set_title("")

def save_fig(fig, out_path, descriptive_title):
    """Save a figure with NO title drawn on the plot itself; the descriptive
    title goes into the filename instead (sanitized), per the agreed
    convention. `out_path` should be the target path WITHOUT the descriptive
    suffix - it gets appended automatically.

    Example:
        save_fig(fig, "figures/theta_genome_wide", "Genome-wide Watterson theta Gen0 vs Gen6")
        -> saves as figures/theta_genome_wide__genome-wide-watterson-theta-gen0-vs-gen6.png
    """
    import re
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', descriptive_title.strip().lower()).strip('-')
    full_path = f"{out_path}__{slug}.png"
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    # Also write a vector PDF next to the PNG (for publication). fonttype 42 embeds
    # editable TrueType text so Illustrator/Inkscape and journal systems handle it.
    import matplotlib
    with matplotlib.rc_context({'pdf.fonttype': 42}):
        fig.savefig(f"{out_path}__{slug}.pdf", bbox_inches='tight')
    return full_path

def shared_ylim(axes_list, pad_fraction=0.05):
    """Force a list of matplotlib Axes to share the same y-axis range - use
    this whenever plotting multiple replicates/panels of the SAME plot type
    that should be visually comparable (e.g. 4 theta violin plots, 4 PCA
    panels). Computes the min/max across all axes' current data and applies
    it to every axis in the list. Handles both linear and log-scaled axes
    (padding is additive on linear axes, multiplicative on log axes, so a
    log-scale plot never gets a non-positive limit)."""
    ymins, ymaxs = [], []
    is_log = axes_list[0].get_yscale() == 'log'
    for ax in axes_list:
        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)
    lo, hi = min(ymins), max(ymaxs)
    if is_log:
        factor = (hi / lo) ** pad_fraction if lo > 0 else 1.0
        lo_padded, hi_padded = lo / factor, hi * factor
    else:
        pad = (hi - lo) * pad_fraction
        lo_padded, hi_padded = lo - pad, hi + pad
    for ax in axes_list:
        ax.set_ylim(lo_padded, hi_padded)
