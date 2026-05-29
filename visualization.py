"""
visualization.py
================
All plotting functions used in the paper.

Figures produced
----------------
Figure 1   plot_incidence_grid_dense   — 5-panel time series of quarterly
                                         deprivation incidence (2019–2025)
Figure 2   plot_transition_master      — heatmap of the 32×32 transition matrix
Figure 3   draw_panel (×2)             — layered flow graphs (recovery + worsening)

Additional diagnostics (not in paper)
--------------------------------------
plot_convergence   — bootstrap convergence curve
draw_legend        — standalone legend for flow graphs

Dependencies: matplotlib, seaborn, numpy, pandas.
"""

from __future__ import annotations

import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

from config import INDICATORS, LABEL_MAP

# Use non-interactive backend when no display is available (e.g. Colab, server)
try:
    matplotlib.use('Agg')
except Exception:
    pass


# ---------------------------------------------------------------------------
# Figure 1: quarterly incidence time series
# ---------------------------------------------------------------------------

def plot_incidence_grid_dense(incidence_df: pd.DataFrame) -> plt.Figure:
    """Plot a 5-panel grid of quarterly deprivation incidence (Fig. 1).

    Each panel shows one indicator's incidence (%) over time, with shaded
    political-period backgrounds and COVID boundary lines.

    Parameters
    ----------
    incidence_df : pd.DataFrame
        Output of calculate_quarterly_incidence(): must contain 'Period'
        and one column per indicator in INDICATORS.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 12), sharex=True)
    sns.set_style('white')

    start_bolsonaro = pd.Timestamp('2019-01-01')
    end_bolsonaro   = pd.Timestamp('2022-12-31')
    start_lula      = pd.Timestamp('2023-01-01')
    end_data        = incidence_df['Period'].max()
    covid_start     = pd.Timestamp('2020-04-01')
    covid_end       = pd.Timestamp('2022-04-01')

    dates_to_annotate = [
        (start_bolsonaro, 'center'),
        (start_lula, 'center'),
        (end_data, 'left'),
    ]

    for i, col in enumerate(INDICATORS):
        ax = axes[i]
        sns.lineplot(
            data=incidence_df, x='Period', y=col, ax=ax,
            marker='o', markersize=5, linewidth=2.2,
            color=sns.color_palette('viridis', 5)[i], zorder=5,
        )

        ax.axvspan(start_bolsonaro, end_bolsonaro, facecolor='#e6f2ff', alpha=0.6, zorder=0)
        ax.axvspan(start_lula, end_data + pd.DateOffset(months=2), facecolor='#ffe6e6', alpha=0.6, zorder=0)
        ax.axvline(covid_start, color='#dc3545', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4)
        ax.axvline(covid_end,   color='#198754', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4)

        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.grid(True, which='minor', axis='x', color='#dddddd', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, which='major', axis='x', color='#cccccc', linestyle='-', linewidth=0.8)
        ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which='major', axis='y', color='#cccccc', linestyle='-', linewidth=0.8)
        ax.grid(True, which='minor', axis='y', color='#eeeeee', linestyle='-', linewidth=0.5)

        current_ymax = incidence_df[col].max()
        ax.set_ylim(0, 33)
        ax.set_title(LABEL_MAP[col], loc='left', fontsize=11, fontweight='bold', pad=12)
        ax.set_ylabel('Deprivation incidence', fontsize=11)
        ax.tick_params(axis='y', labelsize=9)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

        for date_target, align in dates_to_annotate:
            row = incidence_df[incidence_df['Period'] == date_target]
            if not row.empty:
                val = row[col].values[0]
                y_pos = val + (current_ymax * 0.02)
                ax.text(
                    x=date_target, y=y_pos, s=f'{val:.1f}%',
                    color='black', fontweight='bold', fontsize=10,
                    ha=align, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='#cccccc', lw=0.5, alpha=0.8),
                    zorder=10,
                )
                ax.scatter(date_target, val, color='black', s=25, zorder=10)

    top_ax = axes[0]
    mid_bolsonaro = start_bolsonaro + (end_bolsonaro - start_bolsonaro) / 2
    mid_lula = start_lula + (end_data - start_lula) / 2
    top_ax.text(mid_bolsonaro, top_ax.get_ylim()[1] * 0.97, 'Bolsonaro Gov.',
                ha='center', va='bottom', fontsize=9, color='#000000')
    top_ax.text(mid_lula, top_ax.get_ylim()[1] * 0.97, 'Lula Gov.',
                ha='center', va='bottom', fontsize=9, color='#000000')
    top_ax.text(covid_start, top_ax.get_ylim()[1] * 0.90, 'Covid ',
                color='#dc3545', ha='right', fontsize=9)
    top_ax.text(covid_end, top_ax.get_ylim()[1] * 0.90, ' End of National Emergency',
                color='#198754', ha='left', fontsize=9)

    axes[-1].set_xlabel('Quarters', fontsize=11)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].tick_params(axis='x', labelsize=10, rotation=0)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: transition matrix heatmap
# ---------------------------------------------------------------------------

def plot_transition_master(
    P_df: pd.DataFrame,
    title: str = 'Markov Transition Matrix',
    mode: str = 'full',
    highlights: list | None = None,
    annotations: list | None = None,
) -> plt.Figure:
    """Plot the 32×32 transition matrix as a heatmap (Fig. 2).

    Parameters
    ----------
    P_df : pd.DataFrame
        Transition probability matrix (32×32), indexed by profile strings.
    title : str
        Plot title.
    mode : str
        'full' — detailed view with axis labels and cell values.
        'mini' — compact view without labels (for multi-panel layouts).
    highlights : list of (origin, destination) tuples or None
        If provided, highlighted cells are shown in colour while the rest
        are shown as greyed-out background.
    annotations : list of dicts or None
        Each dict: {'pos': (origin_lbl, dest_lbl), 'text': str,
                    'offset': (dx, dy)} for arrow annotations.
    """
    from transition_matrix import sort_matrix_by_severity
    P_sorted = sort_matrix_by_severity(P_df)
    profiles = list(P_sorted.index)

    if mode == 'mini':
        figsize = (6, 6)
        show_cbar = show_values = False
        x_label, y_label = 'to state', 'from state'
        title_size = label_size = 12 if mode == 'mini' else 16
        grid_lw = 0.0
    else:
        figsize = (20, 18)
        show_cbar = show_values = True
        x_label = 'Destination Profile ($t+1$)'
        y_label = 'Origin Profile ($t$)'
        title_size, label_size = 20, 16
        grid_lw = 0.01

    fig, ax = plt.subplots(figsize=figsize)
    cbar_kws = {
        'label': 'Transition Probability', 'orientation': 'horizontal',
        'shrink': 0.5, 'pad': 0.08, 'aspect': 30,
    } if show_cbar else None
    cmap_viridis = copy.copy(plt.cm.viridis)

    if highlights:
        sns.heatmap(P_sorted, cmap='Greys',
                    norm=PowerNorm(gamma=0.2, vmin=-0.5, vmax=0.5),
                    cbar=False, annot=False,
                    linewidths=grid_lw, linecolor='#cecece', square=True, ax=ax)
        mask = pd.DataFrame(True, index=P_sorted.index, columns=P_sorted.columns)
        for origin, dest in highlights:
            if origin in P_sorted.index and dest in P_sorted.columns:
                mask.loc[origin, dest] = False
        sns.heatmap(P_sorted, mask=mask, cmap=cmap_viridis,
                    norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=0.8),
                    annot=False, linewidths=grid_lw, linecolor='#111111',
                    square=True, cbar=show_cbar, cbar_kws=cbar_kws, ax=ax)
    else:
        sns.heatmap(P_sorted, cmap=cmap_viridis,
                    norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=0.8),
                    annot=False, linewidths=grid_lw, linecolor='black',
                    square=True, cbar=show_cbar, cbar_kws=cbar_kws, ax=ax)

    if show_values:
        for y in range(P_sorted.shape[0]):
            for x in range(P_sorted.shape[1]):
                val = P_sorted.iloc[y, x]
                should_print = False
                if highlights:
                    py, px = P_sorted.index[y], P_sorted.columns[x]
                    if not mask.loc[py, px] and val >= 0.10:
                        should_print = True
                elif val >= 0.01:
                    should_print = True
                if should_print:
                    text_color = 'black' if val > 0.4 else 'white'
                    ax.text(x + 0.5, y + 0.5, f'{val:.2f}',
                            ha='center', va='center',
                            color=text_color, fontsize=9, fontweight='normal')

    # Severity-level separator lines
    severity_counts: dict = {}
    for i, profile in enumerate(profiles):
        sev = profile.count('1')
        if sev not in severity_counts:
            severity_counts[sev] = []
        severity_counts[sev].append(i)
    line_color = '#999999' if highlights else 'white'
    for sev, indices in severity_counts.items():
        end = max(indices) + 1
        if end < len(profiles):
            ax.hlines(end, *ax.get_xlim(), colors=line_color, linestyles='--', linewidth=1.0, alpha=0.6)
            ax.vlines(end, *ax.get_ylim(), colors=line_color, linestyles='--', linewidth=1.0, alpha=0.6)

    if annotations and highlights:
        y_indices = {label: i for i, label in enumerate(P_sorted.index)}
        x_indices = {label: i for i, label in enumerate(P_sorted.columns)}
        for note in annotations:
            try:
                origin_lbl, dest_lbl = note['pos']
                dx, dy = note.get('offset', (-4, -2))
                y_coord = y_indices[origin_lbl] + 0.5
                x_coord = x_indices[dest_lbl] + 0.5
                ax.annotate(
                    note['text'],
                    xy=(x_coord, y_coord),
                    xytext=(x_coord + dx, y_coord + dy),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    bbox=dict(boxstyle='square,pad=0.8', fc='white', ec='black', lw=1, alpha=0.9),
                    fontsize=12, color='#454545', ha='center', va='center',
                )
            except KeyError:
                pass

    ax.set_xlabel(x_label, fontsize=label_size, labelpad=15)
    ax.set_ylabel(y_label, fontsize=label_size, labelpad=15)
    ax.set_title(title, fontsize=title_size, pad=20, fontweight='bold')

    if mode == 'mini':
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ticks = np.arange(len(profiles)) + 0.5
        ax.set_xticks(ticks)
        ax.set_xticklabels(profiles, fontsize=9, rotation=90, family='monospace')
        ax.set_yticks(ticks)
        ax.set_yticklabels(profiles, fontsize=9, rotation=0, family='monospace')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diagnostic: bootstrap convergence
# ---------------------------------------------------------------------------

def plot_convergence(result: dict, log_x: bool = False, ax=None):
    """Plot the bootstrap convergence curve (diagnostic, not in paper).

    Parameters
    ----------
    result : dict
        Output of bootstrap_convergence().
    log_x : bool
        Use log scale on the x-axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    curve = result['curve']
    eps = result['epsilon']
    n_conv = result['converged_at']

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = curve['n_families'].to_numpy()
    ax.fill_between(x, curve['q05'], curve['q95'], alpha=0.15, color='#34495e', label='5–95% band')
    ax.fill_between(x, curve['q25'], curve['q75'], alpha=0.30, color='#34495e', label='IQR')
    ax.plot(x, curve['median'], marker='o', color='#2c3e50', linewidth=2, label='Median TVD')
    ax.axhline(eps, color='#c0392b', linestyle='--', alpha=0.7, label=f'ε = {eps}')
    if n_conv is not None:
        ax.axvline(n_conv, color='#27ae60', linestyle=':', alpha=0.7, label=f'converged at N = {n_conv:,}')

    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel('Number of sampled households' + (' (log scale)' if log_x else ''))
    ax.set_ylabel('Median row-wise TVD vs. P_full')
    ax.set_title('Bootstrap convergence of the transition matrix')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    return ax


# ---------------------------------------------------------------------------
# Figure 3: layered flow graph
# ---------------------------------------------------------------------------

# Layout parameters
_YPOS = {0: 27.5, 1: 22.3, 2: 17.1, 3: 11.9, 4: 6.9, 5: 2.0}
_XL, _XR = -7.0, 12.0
_PURPLE = '#6a3d9a'
_C_RECOVERY = '#1b9e77'
_C_WORSENING = '#d95f02'


def _load_flow_data(P_df: pd.DataFrame, occu: pd.DataFrame) -> tuple:
    """Prepare arrays for the flow graph from a transition matrix and occupancy table."""
    states = list(P_df.columns)
    Pv = P_df.values.astype(float)
    Pv = Pv / Pv.sum(axis=1, keepdims=True)
    n = Pv.shape[0]

    occ = occu.copy()
    occ['b'] = occ['binary'].astype(str).str.zfill(5)
    share_map = occ.set_index('b')['avg_share'].to_dict()
    sh = np.array([share_map[s] for s in states])
    ham = np.array([bin(i).count('1') for i in range(n)])
    return states, Pv, sh, ham, n


def _select_edges(
    Pv: np.ndarray,
    sh: np.ndarray,
    ham: np.ndarray,
    n: int,
    direction: int,
    floor_occ: float = 0.01,
    floor_p: float = 0.05,
    k: int = 2,
) -> list:
    """Select the top-k transitions per origin in the given direction (±1).

    direction = −1 for recovery (fewer deprivations), +1 for worsening.
    floor_occ: minimum occupancy share to draw edges from a state.
    floor_p: minimum transition probability to include an edge.
    k: maximum number of edges per origin.
    """
    off = Pv.copy()
    np.fill_diagonal(off, 0.0)
    edges = []
    for i in range(n):
        if sh[i] < floor_occ:
            continue
        cand = [
            (j, off[i, j]) for j in range(n)
            if off[i, j] >= floor_p and np.sign(ham[j] - ham[i]) == direction
        ]
        cand = sorted(cand, key=lambda t: -t[1])[:k]
        for rank, (j, p) in enumerate(cand):
            edges.append((i, j, p, rank))
    return edges


def _compute_positions(ham: np.ndarray, n: int) -> dict:
    """Compute (x, y) positions for each node based on Hamming weight."""
    pos = {}
    for wt in range(6):
        idx = sorted([i for i in range(n) if ham[i] == wt])
        m = len(idx)
        xs = np.linspace(_XL, _XR, m) if m > 1 else np.array([(_XL + _XR) / 2])
        for i, x in zip(idx, xs):
            pos[i] = (x, _YPOS[wt])
    return pos


def draw_panel(
    states: list,
    Pv: np.ndarray,
    sh: np.ndarray,
    ham: np.ndarray,
    n: int,
    edges: list,
    color: str,
    fname: str,
    label_below: bool,
    pmax: float,
    out_dir: str = '.',
    exts: list | None = None,
    dpi: int = 190,
) -> None:
    """Draw one flow graph panel (recovery or worsening).

    Parameters
    ----------
    label_below : bool
        True for the recovery panel (arrows go up, labels below nodes);
        False for the worsening panel (arrows go down, labels above nodes).
    fname : str
        Output filename stem (without extension).
    out_dir : str
        Output directory.
    exts : list of str
        File extensions to save (default: ['png', 'pdf', 'svg']).
    dpi : int
        Resolution for raster output.
    """
    if exts is None:
        exts = ['png', 'pdf', 'svg']

    pos = _compute_positions(ham, n)
    shmax = sh.max()

    fig, ax = plt.subplots(figsize=(15.5, 9.6))

    # Layer backgrounds
    for wt in range(6):
        y = _YPOS[wt]
        ax.axhspan(y - 2.6, y + 2.6, color=('#f6f6f4' if wt % 2 else '#ededeb'), zorder=0)
        ax.text(-8.5, y, f'{wt}', fontsize=17, fontweight='bold',
                va='center', ha='center', color='#999')

    # Nodes
    circ = {}
    for i in range(n):
        x, y = pos[i]
        r = 0.30 + 0.95 * np.sqrt(sh[i] / shmax)
        c = Circle((x, y), r, facecolor='#fbfafc', edgecolor=_PURPLE,
                   lw=0.6 + 3.6 * Pv[i, i], zorder=3)
        ax.add_patch(c)
        circ[i] = (c, r)
        yy = (y - r - 0.16) if label_below else (y + r + 0.16)
        va = 'top' if label_below else 'bottom'
        ax.text(x, yy, states[i], fontsize=(8.0 if sh[i] > 0.02 else 6.6),
                ha='center', va=va, zorder=4, fontfamily='monospace',
                fontweight=('bold' if sh[i] > 0.02 else 'normal'), color='#222')

    # Edges
    for i, j, p, rank in edges:
        s = np.sqrt(p / pmax)
        rad = 0.17 * (1 if i < j else -1)
        ax.add_patch(FancyArrowPatch(
            pos[i], pos[j], connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>', mutation_scale=8 + 11 * s, lw=0.6 + 5.2 * s,
            color=color, alpha=0.6, zorder=2,
            patchA=circ[i][0], patchB=circ[j][0], shrinkA=1, shrinkB=2,
        ))
        xi, yi = pos[i]
        xj, yj = pos[j]
        dx, dy = xj - xi, yj - yi
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        d = circ[i][1] + 0.30
        ax.text(xi + ux * d, yi + uy * d, f'{p:.2f}'.lstrip('0'),
                fontsize=6.4, ha='center', va='center', zorder=5,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.9))

    ax.set_xlim(-9.4, 13.7)
    ax.set_ylim(-1.4, 30.4)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    import os
    for ext in exts:
        plt.savefig(os.path.join(out_dir, f'{fname}.{ext}'), dpi=dpi,
                    bbox_inches='tight', facecolor='white')
    plt.close()


def draw_legend(
    color: str,
    line_label: str,
    fname: str,
    out_dir: str = '.',
    exts: list | None = None,
    dpi: int = 190,
) -> None:
    """Draw a standalone legend for the flow graph panels.

    Parameters
    ----------
    color : str
        Arrow colour used in the corresponding panel.
    line_label : str
        Label for the arrow (e.g. 'recovery (sheds a deprivation)').
    fname : str
        Output filename stem.
    """
    if exts is None:
        exts = ['png', 'pdf', 'svg']

    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    ax.axis('off')
    handles = [
        Line2D([0], [0], color=color, lw=3.6, label=line_label),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fbfafc',
               markeredgecolor=_PURPLE, markeredgewidth=3.8, markersize=14,
               label='ring thickness ∝ persistence, P(i|i)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fbfafc',
               markeredgecolor=_PURPLE, markeredgewidth=1, markersize=18,
               label='node size ∝ observed occupancy'),
    ]
    lg = ax.legend(
        handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.0),
        fontsize=10, frameon=False,
        title='arrow width ∝ P(j|i)  ·  label at base = P (two largest per origin)',
        title_fontsize=10, handletextpad=0.8, labelspacing=0.7,
    )
    lg.get_title().set_fontweight('bold')
    lg._legend_box.align = 'left'
    fig.text(
        0.02, 0.06,
        "layers = number of deprivations (0 = none … 5 = all)   ·   "
        "bits: illiteracy · adult schooling · precarious work · "
        "social security · informality",
        fontsize=7.6, color='#777', ha='left', va='bottom',
    )

    import os
    for ext in exts:
        plt.savefig(os.path.join(out_dir, f'{fname}.{ext}'), dpi=dpi,
                    bbox_inches='tight', facecolor='white')
    plt.close()


def draw_flow_graphs(
    P_df: pd.DataFrame,
    occu: pd.DataFrame,
    out_dir: str = '.',
    floor_occ: float = 0.01,
    floor_p: float = 0.05,
    k: int = 2,
    exts: list | None = None,
    dpi: int = 190,
) -> None:
    """Generate both recovery and worsening flow graphs plus their legends.

    Parameters
    ----------
    P_df : pd.DataFrame
        32×32 transition matrix (typically P_2025).
    occu : pd.DataFrame
        Output of state_occupancy_year() for the same year.
    out_dir : str
        Directory where figures are saved.
    """
    if exts is None:
        exts = ['png', 'pdf', 'svg']

    states, Pv, sh, ham, n = _load_flow_data(P_df, occu)
    E_rec = _select_edges(Pv, sh, ham, n, direction=-1, floor_occ=floor_occ, floor_p=floor_p, k=k)
    E_wor = _select_edges(Pv, sh, ham, n, direction=+1, floor_occ=floor_occ, floor_p=floor_p, k=k)

    # Shared scale for arrow widths (comparability between panels)
    pmax = max(p for E in (E_rec, E_wor) for _, _, p, _ in E)

    draw_panel(states, Pv, sh, ham, n, E_rec, _C_RECOVERY,
               'flow_recovery_P2025', label_below=True, pmax=pmax,
               out_dir=out_dir, exts=exts, dpi=dpi)
    draw_panel(states, Pv, sh, ham, n, E_wor, _C_WORSENING,
               'flow_worsening_P2025', label_below=False, pmax=pmax,
               out_dir=out_dir, exts=exts, dpi=dpi)
    draw_legend(_C_RECOVERY, 'recovery (sheds a deprivation)', 'legend_recovery',
                out_dir=out_dir, exts=exts, dpi=dpi)
    draw_legend(_C_WORSENING, 'worsening (gains a deprivation)', 'legend_worsening',
                out_dir=out_dir, exts=exts, dpi=dpi)
    print(f"Flow graphs saved to: {out_dir}/")
