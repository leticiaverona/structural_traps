"""
config.py
=========
Central configuration for the replication code of:

    "Structural Traps in Multidimensional Labor Poverty in Brazil:
     A Granular Markovian Analysis (2019–2025)"

All constants, PNADC fixed-width specifications, variable mappings, and
the base file-path are defined here. Every other module imports from this
file rather than redeclaring these values.

Path configuration
------------------
Set BASE_PATH to the directory that holds your PNADC files. When running
on Google Colab with the data on Drive, mount the drive first and set
BASE_PATH = '/content/drive/MyDrive/PNADC'.  For local execution, set it
to the corresponding local directory.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# File path
# ---------------------------------------------------------------------------

# Auto-detect Colab vs local; override here if needed.
if os.path.isdir('/content/drive'):
    BASE_PATH: str = '/content/drive/MyDrive/PNADC'
else:
    BASE_PATH: str = os.path.join(os.path.dirname(__file__), 'data')

# ---------------------------------------------------------------------------
# PNADC fixed-width column specifications
# ---------------------------------------------------------------------------
# Structure: 'VARIABLE_NAME': (IBGE_Start_Position, Length)
# Position is 1-based as per IBGE documentation; conversion to 0-based
# Python slices is performed in get_quarter().

PNADC_SPECS: dict[str, tuple[int, int]] = {
    # Identification & Panel Keys
    'Ano':     (1, 4),       # Survey year
    'Trimestre': (5, 1),     # Survey quarter (1–4)
    'UF':      (6, 2),       # State code
    'UPA':     (12, 9),      # Primary sampling unit
    'V1008':   (28, 2),      # Household selection number
    'V1014':   (30, 2),      # Panel group
    'V1016':   (32, 1),      # Interview number within panel (1–5)
    'V1028':   (50, 15),     # Population projection weight

    # Person Demographics
    'V2003':   (91, 2),      # Person order number (household member count proxy)
    'V2007':   (95, 1),      # Sex (1=Male, 2=Female)
    'V2009':   (104, 3),     # Age in years

    # Education
    'V3001':   (108, 1),     # Literacy — can read and write? (1=Yes, 2=No)
    'V3002':   (109, 1),     # Currently attending school? (1=Yes, 2=No)
    'VD3004':  (405, 1),     # IBGE-derived education level (1–7 scale)

    # Labour Force (IBGE-derived variables)
    'VD4001':  (409, 1),     # In the labour force? (1=Yes, 2=No)
    'VD4002':  (410, 1),     # Employment status (1=Employed, 2=Unemployed)
    'VD4005':  (414, 1),     # Discouraged worker (1=Yes)
    'VD4004A': (413, 1),     # Time-related underemployment (1=Yes)
    'VD4009':  (417, 2),     # Employment position/category (1–10 codes)
    'VD4010':  (419, 2),     # Occupation sector (1=Agriculture, 2=Other)
    'VD4012':  (423, 1),     # Social-security contribution (1=Yes, 2=No)
    'V4006A':  (142, 1),     # Reason for temporary work absence (2/3=on leave with benefit)
    'V4019':   (186, 1),     # Has CNPJ (firm registration)? (1=Yes)
}

# ---------------------------------------------------------------------------
# Deprivation indicators (ordered; this order determines the bit position
# in the binary state profile d1 d2 d3 d4 d5, MSB-first)
# ---------------------------------------------------------------------------

INDICATORS: list[str] = [
    'depriv_illiteracy',       # d1 — bit 0 (MSB)
    'depriv_adult_edu',        # d2 — bit 1
    'depriv_precarious_work',  # d3 — bit 2
    'depriv_social_security',  # d4 — bit 3
    'depriv_informality',      # d5 — bit 4 (LSB)
]

INDICATOR_LABELS: dict[str, str] = {
    'depriv_illiteracy':      'Illiteracy',
    'depriv_adult_edu':       'Low Adult Educational Attainment',
    'depriv_precarious_work': 'Employment Deprivation',
    'depriv_social_security': 'Social Security Deprivation',
    'depriv_informality':     'Informality',
}

# Compact labels used in charts (LaTeX subscripts for the indicator symbol)
LABEL_MAP: dict[str, str] = {
    'depriv_illiteracy':      r'Illiteracy $(d_1)$',
    'depriv_adult_edu':       r'Low Adult Educational Attainment $(d_2)$',
    'depriv_precarious_work': r'Employment Deprivation $(d_3)$',
    'depriv_social_security': r'Social Security Deprivation $(d_4)$',
    'depriv_informality':     r'Informality $(d_5)$',
}

# ---------------------------------------------------------------------------
# Geographic mappings
# ---------------------------------------------------------------------------

UF_MAP: dict[int, str] = {
    11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL',
    28: 'SE', 29: 'BA',
    31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
    41: 'PR', 42: 'SC', 43: 'RS',
    50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF',
}

REGION_MAP: dict[int, str] = {
    1: 'North', 2: 'Northeast', 3: 'Southeast', 4: 'South', 5: 'Center-West',
}

# ---------------------------------------------------------------------------
# Analysis periods used in the paper
# ---------------------------------------------------------------------------
# Each period is (start_year, start_quarter, end_year, end_quarter), inclusive.

PERIODS: dict[str, tuple[int, int, int, int]] = {
    'pre_pandemic':  (2019, 1, 2020, 1),
    'pandemic':      (2020, 2, 2022, 1),
    'post_pandemic': (2022, 2, 2025, 1),  # adjust end when new data arrive
}

# The primary matrix used in the paper is estimated on 2025 data only.
PAPER_YEAR: int = 2025
