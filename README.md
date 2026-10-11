# eVTOL Battery Thermal-Runaway Risk Model

Public code release for the IEEE Access article:

**R. M. Furlan, G. L. G. R. Vieira, L. O. Carvalho, D. A. Pamplona, and M. X. Guterres, “Literature-Informed Stochastic Scenario Analysis of Thermal-Runaway Risk During eVTOL Battery Fast Charging,” IEEE Access, vol. 14, pp. 129532–129543, 2026.**

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2026.3725575-blue)](https://doi.org/10.1109/ACCESS.2026.3725575)
[![Tests](https://github.com/Orbolato/eVTOL-Battery-Thermal-Runaway-Risk/actions/workflows/tests.yml/badge.svg)](https://github.com/Orbolato/eVTOL-Battery-Thermal-Runaway-Risk/actions/workflows/tests.yml)

## Overview

This repository implements the Monte Carlo scenario-analysis framework used in the final published study of thermal-runaway **initiation susceptibility** during eVTOL battery fast charging. The model combines seven stochastic inputs in a semi-empirical multiplicative risk-factor architecture and evaluates uncertainty associated with input dependence, structural assumptions, thermal-onset extrapolation, sensitivity, and mitigation trade-offs.

The primary model output is the dimensionless risk multiplier **R**. The optional scaling `P_TR = P_base × R` is retained only as an illustrative reference and should **not** be interpreted as a validated real-world or regulator-grade probability of battery fire.

This is a **post-publication public code release**. The IEEE article stated that no supplementary code package accompanied the submission; this repository makes the documented implementation used for the final revised analysis publicly accessible after publication.

## Repository structure

```text
.
├── run_all.py                 # Single entry point for the complete analysis
├── requirements.txt           # Exact Python package versions used
├── src/
│   ├── distributions.py       # Seven input-variable samplers
│   ├── risk_functions.py      # Risk-amplification functions and ablations
│   ├── dependence.py          # Rank-dependence scenarios
│   ├── simulation.py          # Core Monte Carlo engine
│   ├── convergence.py         # Convergence and runtime analysis
│   ├── sensitivity.py         # Spearman and Sobol sensitivity analyses
│   ├── mitigation.py          # Mitigation scenarios and proxy trade-offs
│   ├── lifecycle.py           # Supporting sequential life-cycle utilities
│   └── plotting.py            # Figure generation
├── tests/                     # Automated verification tests
└── outputs/
    ├── tables/                # Reference numeric outputs
    ├── figures/               # Reference figures (PDF)
    ├── logs/                  # Reference run logs
    └── result_manifest.csv    # Maps headline results to pipeline outputs
```

## Environment

The reference run documented in the code package used:

- Python 3.13.9
- NumPy 2.3.5
- SciPy 1.16.3
- pandas 2.3.3
- Matplotlib 3.10.6
- pytest 8.4.2
- PyYAML 6.0.3

Install the exact dependencies with:

```bash
python -m pip install -r requirements.txt
```

A virtual environment is recommended.

## Run the complete analysis

From the repository root:

```bash
python run_all.py
```

The script regenerates the analysis outputs under `outputs/` and uses the fixed master seed:

```text
20260716
```

The full pipeline includes a baseline simulation with **N = 10,000,000** samples plus dependence, ablation, convergence, sensitivity, heatmap, mitigation, and runtime analyses. The supplied reference package reports a total wall time of roughly eight minutes on the documented reference hardware; runtime will vary by system.

## Run the tests

```bash
python -m pytest tests/ -q
```

The public package was checked before release and the supplied test suite passed **22/22 tests**.

## Reference outputs

The checked-in `outputs/` directory contains the tables, figures, logs, and result manifest produced by the supplied analysis package. These files provide reference values against which a new run can be compared.

Key output groups include:

- baseline risk-index summary;
- structural-ablation matrix;
- input-dependence scenarios;
- convergence and runtime benchmark;
- Spearman and Sobol sensitivity analyses;
- SOC × C-rate interaction heatmap;
- mitigation-strategy trade-offs.

## Data availability

No external empirical dataset is required. The analysis uses synthetic Monte Carlo draws from the distributions, equations, scenario definitions, sample sizes, and fixed random seed documented in the paper and implementation.

## Scope and interpretation

This repository should be used for scenario analysis, methodological inspection, and computational repeatability under the documented assumptions. In particular:

- the model addresses **thermal-runaway initiation susceptibility**, not propagation after initiation;
- the risk-amplification functions are literature-informed semi-empirical constructions, not eVTOL-calibrated regressions;
- the automotive baseline reference used for optional absolute scaling could not be independently reconstructed as a genuine per-charge probability;
- dependence structures and mitigation-cost indices are scenario/proxy assumptions rather than measured quantities;
- no regulatory compliance classification or external safety-threshold comparison is supported by this model;
- the absence of eVTOL-specific operational and incident data remains the main external-validity limitation.

These limitations are central to the interpretation of the published results.

## Citation

If you use this repository or its implementation in academic work, please cite the published article:

```bibtex
@article{Furlan2026eVTOLThermalRunaway,
  author  = {Furlan, Rodrigo Mollo and Vieira, Gabriel Luiz Goulart Rufino and Carvalho, Lucas Orbolato and Pamplona, Daniel Alberto and Guterres, Marcelo Xavier},
  title   = {Literature-Informed Stochastic Scenario Analysis of Thermal-Runaway Risk During eVTOL Battery Fast Charging},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {129532--129543},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3725575}
}
```

## License

No separate software license has currently been assigned to this repository. The Creative Commons license applying to the published IEEE article does not by itself grant a software license for this source code.
