"""
indicators.py
=============
Computation of the five binary deprivation indicators and the household
deprivation profile used throughout the paper.

Deprivation definitions (household-level, worst-member aggregation)
--------------------------------------------------------------------
d1  Illiteracy              Any household member aged ≥15 cannot read or write.
d2  Low Adult Education     No adult (aged ≥18) has completed elementary school.
d3  Employment Deprivation  Any adult is unemployed, underemployed, or discouraged.
d4  Social Security         No member contributes to or receives social-security
                            benefits (including retirement/pension from visit files).
d5  Informality             At least one adult is employed but no member holds a
                            formal job (CLT card, CNPJ, public sector, military).

Social-security correction (d4)
--------------------------------
The quarterly microdata (VD4012) records only current-quarter contribution
status.  Retirement and disability pensions are captured in the PNADC
visit files (V5004A — retirement/pension; V5001A — BPC-LOAS disability
benefit) via a separate file layout.  finalize_social_security() merges
these visit-file flags into the panel before computing d4, matching the
coverage described in Section 3 of the paper.

Bolsa Família / Auxílio Brasil and unemployment insurance are deliberately
excluded from social-security protection, following the paper's definition.

Public entry points
-------------------
calculate_family_indicators(df)
    Compute all five deprivation flags at the household level from one raw
    PNADC quarter DataFrame.

calculate_deprivation_profile_and_score(df)
    Append 'deprivation_profile' (binary string, e.g. '01011') and
    'deprivation_score' (share of deprivations, 0.0–1.0) to the panel.

build_dict_lookup(dict_dir)
    Scan the PNADC data directory for visit-file dictionary (.xls) files
    and return a {(year_str, visit_str): path} mapping.

build_benefit_flags(visit_jobs)
    Read the listed (data_file, dictionary_file) pairs and return a
    per-household flag indicating whether any member receives retirement
    or BPC-LOAS benefits in any observed visit.

finalize_social_security(panel, benefit_flags)
    Recompute d4 by combining the in-quarter contribution status already
    computed by calculate_family_indicators with the visit-file benefit
    flags.  Must be called before calculate_deprivation_profile_and_score.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from config import BASE_PATH, INDICATORS


# ---------------------------------------------------------------------------
# Household-level deprivation indicators
# ---------------------------------------------------------------------------

def calculate_family_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all five binary deprivation flags, aggregated at the household level.

    The function works in five stages:
        0. Pre-clean: fill NaN in key input columns with −1 so that boolean
           comparisons work cleanly without NaN propagation.
        1. Individual-level binary flags for education and labour outcomes.
        2. Aggregation to household level (max = worst-member logic).
        3. Final household-level deprivation logic (some indicators require a
           negation of the aggregated flag, e.g. 'no adult has elementary').
        4. Rename and select final output columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PNADC quarter DataFrame as returned by get_quarter().

    Returns
    -------
    pd.DataFrame
        One row per household, with columns:
        id_dom, Ano, Trimestre, UF, interview_number,
        depriv_illiteracy, depriv_adult_edu, depriv_precarious_work,
        prot_labor (intermediate for d4), depriv_informality.
    """
    df = df.copy()

    # --- 0. Pre-clean: replace NaN in critical inputs with sentinel −1 ---
    inputs_to_clean = [
        'V2009', 'V3001', 'V3002', 'VD3004', 'VD4001',
        'VD4002', 'VD4005', 'VD4004A', 'VD4012', 'VD4009', 'V4019',
    ]
    for col in inputs_to_clean:
        if col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                if -1 not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([-1])
            df[col] = df[col].fillna(-1)

    # --- 1. Demographic scope flags ---
    df['is_adult'] = df['V2009'] >= 18
    df['expect_literacy'] = df['V2009'] >= 15     # literacy assessed at age ≥15
    df['is_school_age'] = df['V2009'].between(4, 17)

    # --- 2. Individual-level education flags ---

    # d1: individual is illiterate (age ≥15, V3001==2 means "cannot read/write")
    df['ind_illiterate'] = np.where(
        df['expect_literacy'] & (df['V3001'] == 2), 1, 0
    )

    # d2: adult has completed at least elementary school (VD3004 ≥ 3)
    # Negated at household level: deprived if NO adult has elementary.
    df['has_elementary'] = np.where(
        df['is_adult'] & (df['VD3004'] >= 3), 1, 0
    )

    # --- 3. Individual-level labour flags ---

    df['is_occupied'] = np.where(
        df['is_adult'] & (df['VD4002'] == 1), 1, 0
    )

    # d3: unemployed (VD4002==2) OR underemployed (VD4004A==1) OR
    #     discouraged worker (VD4005==1)
    df['ind_precarious_work'] = np.where(
        df['is_adult'] & (
            (df['VD4002'] == 2) |
            (df['VD4004A'] == 1) |
            (df['VD4005'] == 1)
        ),
        1, 0,
    )

    # d4 (partial): contributes to social security or is protected by statute.
    # Cases covered:
    #   VD4012==1  — explicitly contributes to social security
    #   VD4009∈{5,7} — military or statutory civil servant (automatically covered)
    #   VD4002==1 & VD4010==1 & VD4009∈{9,10} — agricultural family/autonomous
    #                                              worker (protected by special rule)
    df['contributes_social_security'] = np.where(
        (df['VD4012'] == 1) |
        (df['VD4009'].isin([5, 7])) |
        (
            (df['VD4002'] == 1) & (df['VD4010'] == 1) &
            (df['VD4009'].isin([9, 10]))
        ),
        1, 0,
    )

    # V4006A==2 or 3: on temporary leave with social-security benefit
    df['is_beneficiary'] = np.where(
        df['V4006A'].isin([2, 3]), 1, 0
    )

    # d5: formality of employment
    # Formal employee codes: 1=CLT private, 3=domestic w/ card,
    #                        5=public w/ card, 7=military/statutory
    formal_codes = [1, 3, 5, 7]
    entrepreneur_codes = [8, 9]

    df['formal_employee'] = np.where(
        df['is_adult'] & df['is_occupied'] & df['VD4009'].isin(formal_codes),
        1, 0,
    )
    df['has_cnpj'] = np.where(df['V4019'] == 1, 1, 0)
    df['formal_entrepreneur'] = np.where(
        df['is_occupied'] & df['has_cnpj'] & df['VD4009'].isin(entrepreneur_codes),
        1, 0,
    )
    df['has_formal_job'] = np.where(
        (df['formal_employee'] == 1) | (df['formal_entrepreneur'] == 1), 1, 0
    )

    # --- 4. Household ID and aggregation ---
    df['id_dom'] = df['UPA'].astype(str) + df['V1008'].astype(str) + df['V1014'].astype(str)

    df_fam = df.groupby(
        ['id_dom', 'Ano', 'Trimestre', 'UF', 'V1016'],
        observed=True,
    ).agg(
        ind_illiterate=('ind_illiterate', 'max'),
        has_elementary=('has_elementary', 'max'),
        ind_precarious_work=('ind_precarious_work', 'max'),
        contributes_social_security=('contributes_social_security', 'max'),
        has_formal_job=('has_formal_job', 'max'),
        is_beneficiary=('is_beneficiary', 'max'),
        V2003=('V2003', 'count'),    # household size
        is_occupied=('is_occupied', 'max'),
    ).reset_index()

    # --- 5. Final household-level deprivation logic ---

    # d2: deprived if NO adult has elementary (max==0 → nobody has it)
    df_fam['depriv_adult_edu'] = np.where(df_fam['has_elementary'] == 0, 1, 0)

    # d4 intermediate: protected if contributes OR is a current benefit recipient
    df_fam['prot_labor'] = np.where(
        (df_fam['contributes_social_security'] == 1) |
        (df_fam['is_beneficiary'] == 1),
        1, 0,
    ).astype('int8')

    # d5: deprived if someone is occupied but no one holds a formal job
    df_fam['depriv_informality'] = np.where(
        (df_fam['is_occupied'] == 1) & (df_fam['has_formal_job'] == 0),
        1, 0,
    )

    # --- 6. Rename and select ---
    df_fam.rename(columns={
        'ind_illiterate':      'depriv_illiteracy',
        'ind_precarious_work': 'depriv_precarious_work',
        'V1016':               'interview_number',
    }, inplace=True)

    final_cols = [
        'id_dom', 'Ano', 'Trimestre', 'UF', 'interview_number',
        'depriv_illiteracy', 'depriv_adult_edu', 'depriv_precarious_work',
        'prot_labor', 'depriv_informality',
    ]
    df_final = df_fam[final_cols].copy()

    for col in [c for c in final_cols if 'depriv_' in c]:
        df_final[col] = df_final[col].fillna(0).astype('int8')

    return df_final


# ---------------------------------------------------------------------------
# Deprivation profile and score
# ---------------------------------------------------------------------------

def calculate_deprivation_profile_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'deprivation_profile' and 'deprivation_score' columns to the panel.

    deprivation_profile : str
        Five-character binary string encoding the household's deprivation
        vector, e.g. '01011' means d2, d4, d5 are deprived (d1, d3 are not).
        Order matches INDICATORS in config.py (d1 d2 d3 d4 d5, MSB-first).
    deprivation_score : float
        Number of active deprivations divided by 5 (range 0.0–1.0).

    Parameters
    ----------
    df : pd.DataFrame
        Panel with the five binary deprivation columns from INDICATORS.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two new columns appended.
    """
    df['deprivation_score'] = df[INDICATORS].sum(axis=1) / 5.0

    # Build the binary profile string by concatenating each indicator column
    df['deprivation_profile'] = df[INDICATORS[0]].astype(str)
    for col in INDICATORS[1:]:
        df['deprivation_profile'] += df[col].astype(str)

    return df


# ---------------------------------------------------------------------------
# Social-security correction (d4) using visit-file data
# ---------------------------------------------------------------------------

def yv_from_name(filename: str) -> tuple[str | None, str | None]:
    """Extract (year, visit_number) from a PNADC dictionary filename.

    Expected pattern: dicionario_PNADC_{YYYY}_visita{V}.xls

    Returns (None, None) if the pattern is not matched.
    """
    match = re.search(r'_(\d{4})_visita(\d)', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def build_dict_lookup(dict_dir: str = BASE_PATH) -> dict:
    """Scan a directory for PNADC visit dictionary files and return a lookup.

    Parameters
    ----------
    dict_dir : str
        Directory to search (defaults to BASE_PATH from config).

    Returns
    -------
    dict
        Mapping {(year_str, visit_str): file_path}.  When multiple files
        match the same (year, visit) pair (e.g. a corrected edition), the
        last file in sorted order wins.
    """
    lookup: dict = {}
    pattern = os.path.join(dict_dir, 'dicionario_PNADC_*visita*.xls')
    for fpath in sorted(glob.glob(pattern)):
        year, visit = yv_from_name(os.path.basename(fpath))
        if year is not None:
            lookup[(year, visit)] = fpath
    return lookup


def parse_dictionary(xls_path: str) -> dict:
    """Parse a PNADC variable dictionary (.xls) and return column specifications.

    The dictionary file contains variable codes, start positions, and lengths.
    Column positions vary by edition, so they are parsed rather than hard-coded.

    Parameters
    ----------
    xls_path : str
        Path to the IBGE dictionary Excel file.

    Returns
    -------
    dict
        Mapping {variable_code: (start_position_ibge, length)}.
    """
    df = pd.read_excel(xls_path, engine='xlrd', sheet_name=0, header=None)
    specs: dict = {}
    for i in range(df.shape[0]):
        pos, size, code = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]
        if pd.notna(code) and pd.notna(pos):
            try:
                specs[str(code).strip()] = (int(float(pos)), int(float(size)))
            except (ValueError, TypeError):
                pass
    return specs


# Variables needed from visit files for the d4 social-security correction
_VISIT_KEYS = ['UPA', 'V1008', 'V1014']          # household identifier
_BENEFIT_VARS = ['V5004A', 'V5001A']              # retirement/pension; BPC-LOAS


def read_visit_benefits(data_path: str, dict_path: str) -> pd.DataFrame:
    """Read benefit receipt flags from one PNADC visit file.

    A household is flagged as receiving benefits (hh_receives_benefit=1) if
    any member receives retirement/pension income (V5004A==1) or BPC-LOAS
    disability benefit (V5001A==1).

    Bolsa Família / Auxílio Brasil and unemployment insurance are NOT
    included, consistent with the paper's social-security definition.

    Parameters
    ----------
    data_path : str
        Path to the PNADC visit microdata file (.txt, fixed-width).
    dict_path : str
        Path to the corresponding IBGE dictionary (.xls).

    Returns
    -------
    pd.DataFrame
        Columns: id_dom, hh_receives_benefit (0/1).
    """
    specs = parse_dictionary(dict_path)
    wanted = _VISIT_KEYS + _BENEFIT_VARS
    missing = [v for v in wanted if v not in specs]
    if missing:
        raise KeyError(f"{os.path.basename(dict_path)} is missing variables: {missing}")

    colspecs, names = [], []
    for v in wanted:
        start_ibge, length = specs[v]
        colspecs.append((start_ibge - 1, start_ibge - 1 + length))
        names.append(v)

    df = pd.read_fwf(data_path, colspecs=colspecs, names=names, dtype=str, header=None)

    for v in _BENEFIT_VARS:
        df[v] = pd.to_numeric(df[v], errors='coerce')

    df['id_dom'] = df['UPA'].astype(str) + df['V1008'].astype(str) + df['V1014'].astype(str)
    df['receives'] = np.where((df['V5004A'] == 1) | (df['V5001A'] == 1), 1, 0)

    hh = df.groupby('id_dom', observed=True)['receives'].max().reset_index()
    hh.rename(columns={'receives': 'hh_receives_benefit'}, inplace=True)
    return hh


def build_benefit_flags(visit_jobs: list) -> pd.DataFrame:
    """Aggregate benefit flags across all specified (data_file, dict_file) pairs.

    A household is flagged if any member received benefits in ANY of the
    listed visit files (union across visits).  This ensures that a household
    is not misclassified as unprotected simply because it was observed in a
    non-benefit quarter.

    Parameters
    ----------
    visit_jobs : list of (data_path, dict_path) tuples
        Each tuple points to one PNADC visit file and its dictionary.

    Returns
    -------
    pd.DataFrame
        Columns: id_dom, hh_receives_benefit (0/1, max across all visits).
    """
    parts = []
    for data_path, dict_path in visit_jobs:
        hh = read_visit_benefits(data_path, dict_path)
        parts.append(hh)
        print(
            f"  + {os.path.basename(data_path)}: "
            f"{len(hh):,} households, "
            f"{hh['hh_receives_benefit'].mean():.1%} with benefit"
        )

    all_hh = pd.concat(parts, ignore_index=True)
    flags = all_hh.groupby('id_dom', observed=True)['hh_receives_benefit'].max().reset_index()
    print(
        f"  = union: {len(flags):,} households, "
        f"{flags['hh_receives_benefit'].mean():.1%} with benefit"
    )
    return flags


def finalize_social_security(panel: pd.DataFrame, benefit_flags: pd.DataFrame) -> pd.DataFrame:
    """Recompute d4 (Social Security Deprivation) using visit-file benefit flags.

    A household is considered socially protected if:
        prot_labor == 1  (contributes to SS or is on statutory leave with benefit)
        OR
        hh_receives_benefit == 1  (receives retirement/pension or BPC-LOAS)

    depriv_social_security = 1 − protected

    This function must be called BEFORE calculate_deprivation_profile_and_score,
    since the profile string is built from the five final deprivation columns.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of process_panel_retention, containing the 'prot_labor' column
        produced by calculate_family_indicators.
    benefit_flags : pd.DataFrame
        Output of build_benefit_flags, with columns id_dom and hh_receives_benefit.

    Returns
    -------
    pd.DataFrame
        Panel with 'depriv_social_security' column added and 'prot_labor' /
        'hh_receives_benefit' intermediate columns removed.
    """
    out = panel.merge(benefit_flags, on='id_dom', how='left')
    out['hh_receives_benefit'] = out['hh_receives_benefit'].fillna(0).astype('int8')

    protected = (out['prot_labor'] == 1) | (out['hh_receives_benefit'] == 1)
    out['depriv_social_security'] = np.where(protected, 0, 1).astype('int8')

    return out.drop(columns=['prot_labor', 'hh_receives_benefit'])
