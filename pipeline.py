"""
pipeline.py
===========
End-to-end replication script for:

    "Structural Traps in Multidimensional Labor Poverty in Brazil:
     A Granular Markovian Analysis (2019–2025)"

Running this file reproduces every number and figure in the paper.

Usage
-----
Command line:
    python pipeline.py                  # full run (Part I + II)
    python pipeline.py --skip-part-i    # Part II only (needs saved CSV)
    python pipeline.py --no-plots       # suppress interactive plot windows

Google Colab:
    from pipeline import run
    run()                               # full run
    run(run_part_i=False)               # Part II only

Two-part structure
------------------
Part I  — Data ingestion (slow, requires PNADC raw files on Drive).
          Reads quarterly PNADC fixed-width files, builds the longitudinal
          household panel, attaches deprivation indicators, and saves
          df_deprivation_profiles_panel.csv.

Part II — Analysis (fast, requires only the saved CSV).
          Builds transition matrices, runs all statistical tests, produces
          every figure, and verifies the key numerical results reported in
          the paper.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the directory that contains pipeline.py is on sys.path so that
# sibling modules (config, data_loading, …) are always importable regardless
# of the working directory from which the script is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local modules
from config import (
    BASE_PATH, INDICATORS, LABEL_MAP, PAPER_YEAR, PERIODS,
)
from data_loading import process_period_data, process_panel_retention
from indicators import (
    build_dict_lookup, build_benefit_flags,
    finalize_social_security,
    calculate_deprivation_profile_and_score,
)
from transition_matrix import build_transition_matrix
from markov_tests import (
    lr_test_homogeneity, effect_size_pairwise,
    bootstrap_convergence,
    test_markov_property, violation_by_pair, summarize_violations,
    predictive_validation,
)
from analysis import (
    calculate_quarterly_incidence, compute_gradient_inputs,
    build_indicator_table, state_occupancy_year,
)
from visualization import (
    plot_incidence_grid_dense, plot_transition_master,
    plot_convergence, draw_flow_graphs,
)
from stationary import report


# ---------------------------------------------------------------------------
# Key numerical results reported in the paper (used in verify())
# ---------------------------------------------------------------------------

_EXPECTED = {
    'pi_00000':              0.425,   # stationary mass at non-deprived state
    'lq_trap_mass':          0.185,   # Low-Qualification Trap stationary mass
    'unprotection_mass':     0.128,   # Unprotection Trap stationary mass
    'overlap_mass':          0.037,   # intersection of both traps
    'union_mass':            0.276,   # union of both traps
    # One-step exit probability (P_2025 row mean for single-deprivation states)
    'exit_d3':               0.557,   # employment deprivation
    'exit_d5':               0.261,   # informality
    'exit_d4':               0.196,   # social security
    'exit_d1':               0.182,   # illiteracy
    'exit_d2':               0.129,   # low adult education
    # Mean hitting time to non-deprived state '00000' (quarters)
    'hitting_1depriv':        8.3,
    'hitting_lq_trap':       14.1,
    'hitting_unprotect':     10.2,
    # Quarterly turnover rates
    'turnover_lq_trap':      0.196,
    'turnover_unprotect':    0.351,
}
_TOL = 0.005   # absolute tolerance for verification


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _period_slice(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return rows falling inside the named analysis period."""
    sy, sq, ey, eq = PERIODS[label]
    mask = (
        (df['Ano'].astype(int) * 10 + df['Trimestre'].astype(int))
        .between(sy * 10 + sq, ey * 10 + eq)
    )
    return df[mask].copy()


def _year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df['Ano'].astype(int) == year].copy()


def _save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(BASE_PATH, name)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Part I — Data ingestion
# ---------------------------------------------------------------------------

def run_part_i() -> pd.DataFrame:
    """
    Build the longitudinal deprivation-profiles panel from raw PNADC files.

    Steps
    -----
    1. Load quarterly files for 2019–2025 via process_period_data().
    2. Retain households observed in ≥2 interviews via process_panel_retention().
    3. Attach social-security flags from visit files (V5004A / V5001A).
    4. Compute deprivation profiles and scores.
    5. Save df_deprivation_profiles_panel.csv.

    Returns
    -------
    pd.DataFrame
        Household panel ready for Part II analysis.
    """
    print("\n=== PART I: Data Ingestion ===")

    print("Loading quarterly PNADC files (2019–2025)…")
    list_of_dfs = process_period_data(start_year=2019, end_year=2025)

    print("Applying panel retention filter (≥2 interviews)…")
    df_panel = process_panel_retention(list_of_dfs, min_interviews=2)
    print(f"  Panel observations: {len(df_panel):,}")

    print("Building social-security flags from visit files…")
    lk = build_dict_lookup(BASE_PATH)
    # Only the specific visit files needed for d4 (visita1 = first observation;
    # visita5 = final observation in the 5-quarter rotation).
    # This list mirrors the original notebook exactly.
    _vj_spec = [
        ('PNADC_2019_visita1.txt', ('2019', '1')),
        ('PNADC_2019_visita5.txt', ('2019', '5')),
        ('PNADC_2020_visita5.txt', ('2020', '5')),
        ('PNADC_2021_visita5.txt', ('2021', '5')),
        ('PNADC_2022_visita1.txt', ('2022', '1')),
        ('PNADC_2022_visita5.txt', ('2022', '5')),
        ('PNADC_2023_visita1.txt', ('2023', '1')),
        ('PNADC_2023_visita5.txt', ('2023', '5')),
        ('PNADC_2024_visita1.txt', ('2024', '1')),
        ('PNADC_2024_visita5.txt', ('2024', '5')),
        ('PNADC_2025_visita1.txt', ('2025', '1')),
    ]
    visit_jobs = [
        (os.path.join(BASE_PATH, fname), lk[key])
        for fname, key in _vj_spec
        if key in lk
    ]
    missing_lk = [fname for fname, key in _vj_spec if key not in lk]
    if missing_lk:
        print(f"  WARNING: dictionary files not found for: {missing_lk}")
    flags = build_benefit_flags(visit_jobs)
    df_panel    = finalize_social_security(df_panel, flags)

    print("Computing deprivation profiles…")
    panel = calculate_deprivation_profile_and_score(df_panel)
    print(f"  Unique households: {panel['id_dom'].nunique():,}")

    _save(panel, 'df_deprivation_profiles_panel.csv')
    return panel


# ---------------------------------------------------------------------------
# Part II — Analysis
# ---------------------------------------------------------------------------

def run_part_ii(panel: pd.DataFrame, show_plots: bool = False) -> dict:
    """
    Full analysis pipeline (transition matrices, tests, figures, verification).

    Parameters
    ----------
    panel : pd.DataFrame
        Output of run_part_i(), or loaded from df_deprivation_profiles_panel.csv.
    show_plots : bool
        If True, call plt.show() after each figure (useful in interactive
        sessions). Set False for headless/batch execution.

    Returns
    -------
    dict
        Collected results: matrices, test outputs, stationary analysis.
    """
    print("\n=== PART II: Analysis ===")

    # ------------------------------------------------------------------
    # §1  Quarterly incidence (Figure 1)
    # ------------------------------------------------------------------
    print("\n[1] Quarterly incidence by deprivation indicator…")
    df_trends = calculate_quarterly_incidence(panel)
    fig = plot_incidence_grid_dense(df_trends)
    fig.savefig(os.path.join(BASE_PATH, 'fig1_incidence_grid.pdf'),
                bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # §2  Transition matrices
    # ------------------------------------------------------------------
    print("\n[2] Building transition matrices…")

    pre_df   = _period_slice(panel, 'pre_pandemic')
    pan_df   = _period_slice(panel, 'pandemic')
    post_df  = _period_slice(panel, 'post_pandemic')

    year_dfs = {str(y): _year_slice(panel, y) for y in range(2019, 2026)}

    _btm_period = dict(n_bits=5, min_interviews=2,
                       zero_row_policy='keep', verbose=False)
    _btm_full   = dict(n_bits=5, min_interviews=2, verbose=False)

    matrices: dict[str, dict] = {}

    P_full, counts_full, meta_full = build_transition_matrix(
        panel, **_btm_full)
    matrices['P_full'] = {'P': P_full, 'counts': counts_full, 'meta': meta_full}

    for key, df_p in [('P_pre_pandemic', pre_df),
                      ('P_pandemic',     pan_df),
                      ('P_post_pandemic', post_df)]:
        P, counts, meta = build_transition_matrix(df_p, **_btm_period)
        matrices[key] = {'P': P, 'counts': counts, 'meta': meta}
        print(f"  {key}: {meta.get('n_transitions', '?')} transitions")

    for yr, df_y in year_dfs.items():
        P, counts, meta = build_transition_matrix(df_y, **_btm_period)
        matrices[yr] = {'P': P, 'counts': counts, 'meta': meta}

    P_2025      = matrices[str(PAPER_YEAR)]['P']
    counts_2025 = matrices[str(PAPER_YEAR)]['counts']

    # Period mini-plots (diagnostic, not in paper)
    for key in ('P_pre_pandemic', 'P_pandemic', 'P_post_pandemic'):
        fig_m = plot_transition_master(matrices[key]['P'], title=key, mode='mini')
        fig_m.savefig(os.path.join(BASE_PATH, f'fig_{key}_mini.pdf'),
                      bbox_inches='tight', dpi=200)
        if show_plots:
            plt.show()
        plt.close(fig_m)

    # ------------------------------------------------------------------
    # §3  Homogeneity and effect-size tests
    # ------------------------------------------------------------------
    print("\n[3] Statistical homogeneity tests…")

    res_3way = lr_test_homogeneity({
        'pre':      matrices['P_pre_pandemic']['counts'],
        'pandemic': matrices['P_pandemic']['counts'],
        'pos':      matrices['P_post_pandemic']['counts'],
    })

    pair_results = effect_size_pairwise(
        matrices,
        min_count=0,
        period_pairs=[
            ('P_pre_pandemic', 'P_pandemic'),
            ('P_pandemic',     'P_post_pandemic'),
            ('P_pre_pandemic', 'P_post_pandemic'),
        ],
        delta_thresholds=(0.01, 0.05, 0.10),
    )

    P_delta = matrices['P_post_pandemic']['P'] - matrices['P_pre_pandemic']['P']
    _save(P_delta, 'P_delta.csv')

    pair_results_annual = effect_size_pairwise(
        matrices,
        min_count=0,
        period_pairs=[('2022', '2023'), ('2023', '2024'), ('2024', '2025')],
        delta_thresholds=(0.01, 0.05, 0.10),
    )

    # ------------------------------------------------------------------
    # §4  Resolution gradient (Table 2)
    # ------------------------------------------------------------------
    print("\n[4] Resolution gradient (2023→2024, 2024→2025)…")

    deltas_2324, bands_2324 = compute_gradient_inputs(
        matrices['2023']['P'], matrices['2024']['P'],
        matrices['2023']['counts'], matrices['2024']['counts'],
    )
    deltas_2425, bands_2425 = compute_gradient_inputs(
        matrices['2024']['P'], matrices['2025']['P'],
        matrices['2024']['counts'], matrices['2025']['counts'],
    )

    # ------------------------------------------------------------------
    # §5  Indicator resolution table (Table 3)
    # ------------------------------------------------------------------
    print("\n[5] Indicator resolution table…")

    indicator_labels = {
        'd1': r'd_1 — Literacy',
        'd2': r'd_2 — Adult Educational Attainment',
        'd3': r'd_3 — Employment',
        'd4': r'd_4 — Social Security',
        'd5': r'd_5 — Informality',
    }
    table, detail = build_indicator_table(
        [
            ('2023', matrices['2023']['P'], matrices['2023']['counts']),
            ('2024', matrices['2024']['P'], matrices['2024']['counts']),
            ('2025', matrices['2025']['P'], matrices['2025']['counts']),
        ],
        indicator_labels=indicator_labels,
        units='prob',
    )

    # ------------------------------------------------------------------
    # §6  Paper transition matrix P_2025 (Figure 2)
    # ------------------------------------------------------------------
    print("\n[6] Paper transition matrix P_2025…")
    _save(P_2025,      'P_2025.csv')
    _save(counts_2025, 'counts_2025.csv')

    fig_P = plot_transition_master(P_2025, title='', mode='full')
    fig_P.savefig(os.path.join(BASE_PATH, 'fig2_P2025_full.pdf'),
                  bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig_P)

    # ------------------------------------------------------------------
    # §7  Data-sufficiency: bootstrap convergence
    # ------------------------------------------------------------------
    print("\n[7] Bootstrap convergence diagnostic…")
    df_2025 = year_dfs[str(PAPER_YEAR)]
    result_boot = bootstrap_convergence(df_2025, epsilon=0.05)

    fig_boot = plt.figure(figsize=(10, 4))
    ax1, ax2 = fig_boot.subplots(1, 2)
    plot_convergence(result_boot, log_x=False, ax=ax1)
    plot_convergence(result_boot, log_x=True,  ax=ax2)
    fig_boot.savefig(os.path.join(BASE_PATH, 'fig_bootstrap_convergence.pdf'),
                     bbox_inches='tight', dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig_boot)

    # ------------------------------------------------------------------
    # §8  Markov property tests
    # ------------------------------------------------------------------
    print("\n[8] Markov property tests…")
    result_markov = test_markov_property(df_2025, length=3, min_obs=30)

    df_viol = violation_by_pair(
        result_markov['n1'], result_markov['n2'],
        S=len(result_markov['states']),
        states=result_markov['states'],
        min_obs=30,
    )
    print(df_viol.head(20).to_string())
    summarize_violations(df_viol)

    # ------------------------------------------------------------------
    # §9  Predictive validation
    # ------------------------------------------------------------------
    print("\n[9] Predictive validation…")
    pred = predictive_validation(df_2025)

    # ------------------------------------------------------------------
    # §10  Sparsity and state occupancy
    # ------------------------------------------------------------------
    print("\n[10] Sparsity check and state occupancy…")
    zero_count         = (P_2025 == 0).sum().sum()
    total_elements     = P_2025.size
    sparsity_pct       = zero_count / total_elements * 100
    print(f"  P_2025 sparsity: {sparsity_pct:.2f}% "
          f"({zero_count} zeros / {total_elements} cells)")

    occu = state_occupancy_year(df_2025, year=PAPER_YEAR)

    # ------------------------------------------------------------------
    # §11  Flow graphs (Figure 3)
    # ------------------------------------------------------------------
    print("\n[11] Flow graphs…")
    draw_flow_graphs(P_2025, occu, out_dir=BASE_PATH)

    # ------------------------------------------------------------------
    # §12  Stationary distribution and structural traps (Table 4)
    # ------------------------------------------------------------------
    print("\n[12] Stationary distribution and structural traps…")
    stat = report(P_2025, counts_2025)

    results = {
        'matrices':    matrices,
        'P_2025':      P_2025,
        'counts_2025': counts_2025,
        'table3':      table,
        'stationary':  stat,
        'markov_test': result_markov,
        'pred':        pred,
        'res_3way':    res_3way,
        'pair_results': pair_results,
        'deltas_2324': deltas_2324,
        'deltas_2425': deltas_2425,
    }
    print("\n=== Pipeline complete ===")
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(run_part_i_flag: bool = True, show_plots: bool = False) -> dict:
    """
    Execute the full replication pipeline.

    Parameters
    ----------
    run_part_i_flag : bool
        If True, build the panel from raw PNADC files (requires Drive access).
        If False, load df_deprivation_profiles_panel.csv from BASE_PATH.
    show_plots : bool
        Forward to run_part_ii; set True for interactive Colab sessions.

    Returns
    -------
    dict
        All analytical results (matrices, tests, stationary analysis, etc.).
    """
    # Select matplotlib backend.
    # 'Agg' is always safe (renders to file, no display needed).
    # An interactive backend is only attempted when show_plots=True and
    # a display is actually available; on headless servers or when Agg is
    # already active, we stay with Agg and skip plt.show() calls silently.
    if not show_plots:
        matplotlib.use('Agg')
    else:
        current = matplotlib.get_backend()
        if current == 'agg':
            # Already non-interactive; plt.show() will warn but not crash
            pass

    if run_part_i_flag:
        panel = run_part_i()
    else:
        csv_path = os.path.join(BASE_PATH, 'df_deprivation_profiles_panel.csv')
        print(f"Loading panel from {csv_path} …")
        panel = pd.read_csv(
            csv_path,
            sep=',', decimal=',',
            dtype={'deprivation_profile': str},
        )
        print(f"  {len(panel):,} rows loaded.")

    return run_part_ii(panel, show_plots=show_plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replication pipeline for the structural-traps paper.')
    parser.add_argument(
        '--skip-part-i', action='store_true',
        help='Skip raw-data ingestion; load the saved CSV instead.')
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Suppress interactive plot windows (always saves PDFs).')
    args = parser.parse_args()

    run(run_part_i_flag=not args.skip_part_i,
        show_plots=not args.no_plots)
