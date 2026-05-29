# Replication Package

**"Structural Traps in Multidimensional Labor Poverty in Brazil: A Granular Markovian Analysis (2019–2025)"**

---

## Overview

This repository contains the complete replication code for the paper.  
Running `pipeline.py` reproduces every table, figure, and numerical result from scratch.

The analysis is divided into two independent parts:

| Part | What it does | What it needs |
|------|-------------|---------------|
| **I** | Builds the longitudinal household panel from raw PNADC files | IBGE microdata on disk (~20 GB) |
| **II** | Runs all Markov-chain analysis and produces figures | `df_deprivation_profiles_panel.csv` (included) |

Researchers who only want to verify the results can **skip Part I** and run Part II directly using the pre-built panel file.

---

## Repository Structure

```
structural_traps/
│
├── pipeline.py            ← Entry point: orchestrates the full run
├── config.py              ← All constants, variable mappings, file path
├── data_loading.py        ← PNADC raw-file parsing (Part I)
├── indicators.py          ← Deprivation indicators + social-security flags (Part I)
├── transition_matrix.py   ← Markov transition matrix construction
├── markov_tests.py        ← Stationarity, homogeneity, Markov-property tests
├── analysis.py            ← Incidence, gradient, indicator table, occupancy
├── stationary.py          ← Stationary distribution, hitting times, traps
├── visualization.py       ← All figures
│
├── requirements.txt
└── README.md
```

---

## Deprivation Framework

Five binary household-level deprivations are combined into a 5-bit state string **d₁ d₂ d₃ d₄ d₅** (MSB-first):

| Bit | Label | Variable(s) | Deprived if… |
|-----|-------|-------------|-------------|
| d₁ | Illiteracy | `V3001` | Any adult household member is illiterate |
| d₂ | Low Adult Education | `VD3004` | No adult member reached secondary schooling (level ≥ 5) |
| d₃ | Employment Deprivation | `VD4001`–`VD4005`, `VD4004A` | No member is fully employed (includes unemployment, underemployment, discouragement) |
| d₄ | Social Security | `V5004A`, `V5001A` | No member contributes to INSS or receives retirement/BPC pension |
| d₅ | Informality | `VD4009`, `VD4012`, `V4019` | No member holds a formal job |

This yields **32 states** (`00000` through `11111`). The **paper matrix** P is estimated on 2025 data only.

---

## Structural Traps

Two aggregate traps are identified from the stationary distribution π:

| Trap | Definition | States | π mass |
|------|-----------|--------|--------|
| Low-Qualification Trap (LQT) | score ≥ 2 deprivations **and** (d₁=1 or d₂=1) | 22 profiles | 18.5% |
| Unprotection Trap (UT) | score ≥ 2 deprivations **and** d₄=1 **and** d₅=1 | 8 profiles | 12.8% |

Overlap (LQT ∩ UT) = 3.7% · Union = 27.6%

---

## Quick Start

### Option A — Part II only (verify numbers, no raw data needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place df_deprivation_profiles_panel.csv in the data/ folder
#    (or adjust BASE_PATH in config.py)

# 3. Run
python pipeline.py --skip-part-i --no-plots
```

Or from a notebook / Colab:

```python
from pipeline import run
results = run(run_part_i_flag=False, show_plots=True)
```

### Option B — Full run (Part I + II, requires raw PNADC files)

#### On Google Colab

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify data folder
import os
os.listdir('/content/drive/MyDrive/PNADC')

# Run
from pipeline import run
results = run(run_part_i_flag=True, show_plots=True)
```

#### Locally

1. Edit `BASE_PATH` in `config.py` to point to your local PNADC folder.
2. Run `python pipeline.py`.

---

## Data Requirements (Part I)

The raw IBGE microdata must be placed inside the directory specified by `BASE_PATH` in `config.py`. All files below are freely downloadable from [IBGE](https://www.ibge.gov.br/estatisticas/sociais/trabalho/9171-pesquisa-nacional-por-amostra-de-domicilios-continua-trimestral.html).

### Quarterly cross-section files

One file per quarter, naming pattern: `PNADC_QQ{YYYY}.txt`  
(e.g. `PNADC_012019.txt`, `PNADC_042025.txt`)

| Years needed |
|-------------|
| 2019 Q1 – 2025 Q1 (25 files) |

### Visit files (for social-security indicator d₄)

One file per visit × year, naming pattern: `PNADC_{YYYY}_visita{V}.txt`

| Files needed |
|-------------|
| `PNADC_{2019..2025}_visita{1..5}.txt` |

### Dictionary files (for visit-file parsing)

One Excel file per visit × year, naming pattern: `dicionario_PNADC_{YYYY}_visita{V}.xls`

| Files needed |
|-------------|
| `dicionario_PNADC_{2019..2025}_visita{1..5}.xls` |

> **Note:** Only the `.xls` format (not `.xlsx`) is currently supported, matching the IBGE release format.

---

## Analysis Periods

| Label | Quarters | Rationale |
|-------|----------|-----------|
| `pre_pandemic` | 2019 Q1 – 2020 Q1 | Pre-COVID baseline |
| `pandemic` | 2020 Q2 – 2022 Q1 | Acute COVID shock |
| `post_pandemic` | 2022 Q2 – 2025 Q1 | Recovery period |

---

## Key Numerical Results (Verification)

After running the pipeline, step **[13]** prints a verification table comparing computed values against the paper targets. All results should match within ±0.005.

| Quantity | Expected |
|----------|----------|
| π\[00000\] (non-deprived state) | 0.425 |
| LQ-Trap stationary mass | 18.5% |
| Unprotection Trap stationary mass | 12.8% |
| Trap overlap | 3.7% |
| Trap union | 27.6% |
| Mean hitting time from 1-deprivation states to 00000 | 8.3 quarters |
| Mean hitting time from LQ-Trap to 00000 | 14.1 quarters |
| Mean hitting time from Unprotection Trap to 00000 | 10.2 quarters |
| Quarterly turnover — LQ-Trap | 19.6% |
| Quarterly turnover — Unprotection Trap | 35.1% |
| P_2025 one-step exit rate: d₃ (Employment) | 55.7% |
| P_2025 one-step exit rate: d₅ (Informality) | 26.1% |
| P_2025 one-step exit rate: d₄ (Social Security) | 19.6% |
| P_2025 one-step exit rate: d₁ (Illiteracy) | 18.2% |
| P_2025 one-step exit rate: d₂ (Adult Education) | 12.9% |

---

## Adapting to Other Contexts

The code is designed to be modular. Common adaptations:

**Different country / survey:** Replace `data_loading.py` to parse your survey's fixed-width format and update `PNADC_SPECS` in `config.py`. The indicator logic in `indicators.py`, and all downstream modules, are independent of the data source.

**Different deprivation dimensions:** Modify `INDICATORS` in `config.py` and the corresponding logic in `indicators.py`. The Markov machinery in `transition_matrix.py`, `markov_tests.py`, and `stationary.py` operates on generic binary state strings and requires no changes.

**Different number of dimensions (≠5):** Pass `n_bits=N` to `build_transition_matrix()`. The state space grows as 2ᴺ; computational cost increases accordingly.

**Different periods:** Update `PERIODS` in `config.py`. The pipeline slices `panel` by (year, quarter) automatically.

**Adding a new deprivation:** (i) Add the column name to `INDICATORS`, (ii) compute the flag in `indicators.py`, (iii) rebuild the panel (Part I), (iv) re-run Part II with `n_bits=6`.

---

## License

This code is released under the MIT License. The underlying PNADC microdata are owned by IBGE and distributed under their [terms of use](https://www.ibge.gov.br/acesso-informacao/institutional/direitos-autorais.html).

---

## Citation

```
[author] (2025). Structural Traps in Multidimensional Labor Poverty in Brazil:
A Granular Markovian Analysis (2019–2025). Social Indicators Research.
```
