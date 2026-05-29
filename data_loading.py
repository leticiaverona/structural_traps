"""
data_loading.py
===============
Functions for reading raw PNADC microdata files and assembling the
longitudinal household panel used in the analysis.

The PNADC (Pesquisa Nacional por Amostra de Domicílios Contínua) is a
rotating panel survey conducted quarterly by IBGE.  Each household
participates in 5 consecutive quarterly interviews (interview numbers
1–5).  The household identifier used here is:

    id_dom = UPA  +  V1008  +  V1014

which uniquely identifies a household across its interviews.

Public entry points
-------------------
get_quarter(year, quarter)
    Read one raw PNADC quarter file and return a DataFrame with only the
    columns defined in config.PNADC_SPECS.

process_period_data(start_year, end_year)
    Iterate over all quarters in [start_year, end_year] and return a list
    of per-quarter DataFrames after applying calculate_family_indicators.

process_panel_retention(list_of_dfs, min_interviews=2)
    Concatenate quarterly DataFrames, report retention statistics, and
    filter households that appear fewer than min_interviews times.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from config import BASE_PATH, PNADC_SPECS
from indicators import calculate_family_indicators


# ---------------------------------------------------------------------------
# Quarter file reader
# ---------------------------------------------------------------------------

def get_quarter(year: int, quarter: int, base_path: str = BASE_PATH) -> pd.DataFrame | None:
    """Read a single PNADC quarterly microdata file.

    Tries both upper- and lower-case filename conventions:
        PNADC_Q{quarter:02d}{year}.txt
        pnadc_Q{quarter:02d}{year}.txt

    Parameters
    ----------
    year : int
        Survey year, e.g. 2023.
    quarter : int
        Survey quarter, 1–4.
    base_path : str
        Directory containing the PNADC files.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns defined in PNADC_SPECS, or None if the
        file is not found.
    """
    filename = f"PNADC_{quarter:02d}{year}.txt"
    filepath = os.path.join(base_path, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(base_path, f"pnadc_{quarter:02d}{year}.txt")
        if not os.path.exists(filepath):
            print(f"[!] File not found: {filename}")
            return None

    print(f"--> Reading: {filename}")

    # Build fixed-width column specs (convert from 1-based IBGE to 0-based Python)
    colspecs = []
    names = []
    for var_name, (start_ibge, length) in PNADC_SPECS.items():
        start_py = start_ibge - 1
        colspecs.append((start_py, start_py + length))
        names.append(var_name)

    try:
        # Read as strings to preserve leading zeros in ID columns
        df = pd.read_fwf(filepath, colspecs=colspecs, names=names, dtype=str, header=None)

        # Convert numeric columns; non-numeric values become NaN
        numeric_cols = [
            'Ano', 'Trimestre', 'V1016', 'V1028', 'V2003', 'V2007', 'V2009',
            'V3001', 'V3002', 'VD3004', 'VD4001', 'VD4002', 'VD4005',
            'VD4004A', 'VD4012', 'VD4009', 'V4019', 'VD4010', 'V4006A',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'UF' in df.columns:
            df['UF'] = df['UF'].astype('category')

        return df

    except Exception as exc:
        print(f"[Error] Failed to read {filename}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Period-level data processing
# ---------------------------------------------------------------------------

def process_period_data(start_year: int, end_year: int, base_path: str = BASE_PATH) -> list:
    """Read and process all PNADC quarters from start_year to end_year (inclusive).

    For each quarter: reads the raw file, applies calculate_family_indicators,
    tags the result with year/quarter columns, and appends to the output list.
    Memory is released after each quarter to avoid OOM on large panel spans.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range.  E.g. 2019, 2025 processes 28 quarters.
    base_path : str
        Directory containing the PNADC files.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per successfully processed quarter.
    """
    processed = []
    print(f"--- Processing quarters {start_year}Q1 → {end_year}Q4 ---")

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            print(f"Processing: {year} Q{quarter}...")
            try:
                df_raw = get_quarter(year, quarter, base_path)
                if df_raw is not None and not df_raw.empty:
                    df_proc = calculate_family_indicators(df_raw)
                    df_proc['year'] = year
                    df_proc['quarter'] = quarter
                    processed.append(df_proc)
                else:
                    print(f"  > Warning: no data for {year} Q{quarter}.")
            except Exception as exc:
                print(f"  [!] Error processing {year} Q{quarter}: {exc}")
            finally:
                if 'df_raw' in dir():
                    del df_raw
                gc.collect()

    print(f"--- Done. {len(processed)} quarters processed. ---")
    return processed


# ---------------------------------------------------------------------------
# Panel assembly and retention filter
# ---------------------------------------------------------------------------

def process_panel_retention(
    list_of_dfs: list,
    min_interviews: int = 2,
) -> pd.DataFrame:
    """Merge quarterly DataFrames and filter by minimum number of appearances.

    Households in the PNADC panel can be observed in 1–5 quarterly interviews.
    Transition analysis requires at least 2 appearances (one origin and one
    destination).  Households below the threshold are dropped.

    Parameters
    ----------
    list_of_dfs : list of pd.DataFrame
        Output of process_period_data.
    min_interviews : int
        Minimum number of quarterly observations required to keep a household.
        Default is 2.

    Returns
    -------
    pd.DataFrame
        Long-format panel with one row per (household × quarter) observation,
        restricted to households with ≥ min_interviews appearances.
    """
    print("--- 1. Merging quarterly datasets ---")
    df_long = pd.concat(list_of_dfs, ignore_index=True)
    print(f"Total rows (all observations): {len(df_long):,}")

    print("\n--- 2. Retention analysis ---")
    family_counts = df_long['id_dom'].value_counts()
    n_total = len(family_counts)
    print(f"Unique households: {n_total:,}")

    distribution = family_counts.value_counts().sort_index()
    print("\nDistribution of interviews per household:")
    print(distribution)

    n_five = distribution.get(5, 0)
    print(f"\nFull-cycle retention (5 interviews): {100 * n_five / n_total:.1f}%")

    print(f"\n--- 3. Filtering (keeping N ≥ {min_interviews}) ---")
    valid_ids = family_counts[family_counts >= min_interviews].index
    df_filtered = df_long[df_long['id_dom'].isin(valid_ids)].copy()

    n_kept = len(valid_ids)
    print(f"Households kept : {n_kept:,} ({n_kept / n_total:.1%})")
    print(f"Observations kept: {len(df_filtered):,}")
    print(f"Dropped : {n_total - n_kept:,} households (insufficient history).")

    return df_filtered
