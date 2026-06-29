# Bayesian Nonparametric Joint Modeling for Multi-Stage Reliability Growth

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**Manuscript**: *A Bayesian Nonparametric Joint Modeling Framework for Multi-Stage Reliability Growth with Degradation and Lifetime Data*  
**Journal**: Reliability Engineering & System Safety (RESS)  
**Authors**: Cenyu Hu\*, Tielu Gao, Yabin Wang, Zhonghua Cheng, Xianming Shi  
**Affiliation**: Shijiazhuang Campus, Army Engineering University, China  
**Contact**: 1339816023@qq.com  

---

## Overview / 概述

This repository contains the complete code, data, and results for the paper. We propose a fully nonparametric Bayesian framework for multi-stage reliability growth analysis:

- **Dependent Dirichlet Process (DDP)** mixtures replace parametric lifetime assumptions
- **Gaussian Process** degradation submodel with shared random effects couples degradation to lifetimes
- **Custom hybrid MCMC** algorithm (slice sampling + blocked Gibbs + Metropolis–Hastings) for posterior inference

**Key finding**: The degradation–lifetime coupling parameter ψ is not reliably identifiable at small per-stage sample sizes ($n_l \leq 10$). DDP-Only (without degradation coupling) is the recommended model for small samples; the full DDP-Joint model requires $n_l \gtrsim 30$.

---

## Repository Structure / 仓库结构

```
.
├── README.md                              # This file
├── manuscript_RESS_cassc.tex              # LaTeX source of the manuscript
├── manuscript_RESS_cassc.pdf              # Compiled manuscript (45 pages)
├── semiconductor_data.xlsx                # Complete data package (11 sheets, see below)
│
└── figures/
    ├── generate_figures.py                # Generate all 22 publication-quality figures
    ├── run_simulation_v3.py               # Run the MCMC simulation study
    ├── case_study_ddp_joint.py            # Run the semiconductor case study
    ├── verify_consistency.py              # Cross-check all numerical claims
    │
    ├── simulation_results_v3.json         # Final simulation results (RMSE, Coverage, WAIC)
    ├── case_study_results.json            # Case study MCMC posterior summaries
    ├── mcmc_diagnostics.json              # MCMC convergence diagnostics (R-hat, ACF)
    │
    ├── fig01_framework.pdf                # Figure 1: Framework architecture
    ├── fig02_stick_breaking.pdf           # Figure 2: DPGMM density estimation
    ├── fig03_dpgmm_concept.pdf            # Figure 3: DDP stick-breaking weights
    ├── fig04_density_comparison.pdf       # Figure 4: Density comparison (4 scenarios)
    ├── fig05_rmse.pdf                     # Figure 5: RMSE dot-plot
    ├── fig06_coverage.pdf                 # Figure 6: Coverage calibration curves
    ├── fig07_interval_width.pdf           # Figure 7: Interval width comparison
    ├── fig08_waic.pdf                     # Figure 8: WAIC model comparison
    ├── fig09_psi_recovery.pdf             # Figure 9: ψ recovery across scenarios
    ├── fig10_sample_size.pdf              # Figure 10: Sample size effect
    ├── fig11_growth_detection.pdf         # Figure 11: Growth detection posteriors
    ├── fig12_degradation.pdf              # Figure 12: Degradation trajectories
    ├── fig13_stage_densities.pdf          # Figure 13: Stage-wise density estimates
    ├── fig14_reliability.pdf              # Figure 14: Reliability curves
    ├── fig15_cluster_weights.pdf          # Figure 15: Cluster weight evolution
    ├── fig16_rho_diagnostics.pdf          # Figure 16: ρ posterior diagnostics
    ├── fig17_psi_diagnostic.pdf           # Figure 17: ψ posterior diagnostics
    ├── fig18_mcmc_traces.pdf              # Figure 18: MCMC trace plots
    ├── fig19_mcmc_traces_global.pdf       # Figure 19: Global MCMC traces
    ├── fig20_mcmc_diagnostics.pdf         # Figure 20: Comprehensive MCMC diagnostics
    ├── fig21_rmse_heatmap.pdf             # Figure 21: RMSE heatmap
    └── fig22_joint_posterior.pdf          # Figure 22: Joint posterior (ψ vs ρ₂)
```

---

## Data Package (`semiconductor_data.xlsx`) / 数据文件

The Excel file contains **11 sheets** (README + 10 data sheets) documenting the complete data pipeline:

| Sheet | Content | Source |
|:---|:---|:---|
| **README** | File overview, data flow, key findings, FAQ | — |
| 1. Failure Times | 15 semiconductor devices: failure/censoring times | Liu (2026a) |
| 2. Summary Statistics | Stage-wise summary: sample means, variances, μ_γ (truth & posterior), K\* | JSON output |
| 3. Degradation Model | GP hyperparameters, degradation rates, ψ, MCMC configuration | Simulation design |
| 4. Simulation Scenarios | 4 scenarios: distribution, components, cross-stage parameter evolution | Simulation design |
| 5. Sim Results (n=5) | RMSE (3 quantiles), Coverage, WAIC, lppd: 6 methods × 4 scenarios | `simulation_results_v3.json` |
| 6. Sim Results (n=10) | Same structure for n_l = 10 per stage | `simulation_results_v3.json` |
| 7. WAIC Decomposition | lppd + p_WAIC + verification check | `simulation_results_v3.json` |
| 8. Case Study MCMC | Posterior: ψ, ρ₂, ρ₃, μ_γ, K\* (mean, SD, HPD) | `case_study_results.json` |
| 9. MCMC Diagnostics | R-hat and ACF for 12 scalar parameters | `mcmc_diagnostics.json` |
| 10. Prior Specification | 14 prior distributions with hyperparameters | Manuscript Table |

---

## Reproducing the Results / 复现结果

### Requirements

```bash
pip install numpy scipy matplotlib scikit-learn openpyxl
```

Python 3.10+ recommended. No specialized probabilistic programming frameworks (Stan, PyMC) are required.

### Step 1: Run the MCMC Simulation Study

```bash
cd figures
python run_simulation_v3.py
```

This runs the full simulation: 6 methods × 4 scenarios × 2 sample sizes × 20 replications.  
**Expected runtime**: several hours on a standard workstation.  
**Output**: `simulation_results_v3.json`

> ⚠️ **Note**: Due to the computational cost (960 MCMC runs totaling ~19 million iterations), pre-computed results are provided in `simulation_results_v3.json`.

### Step 2: Run the Case Study

```bash
cd figures
python case_study_ddp_joint.py
```

Fits the DDP-Joint model to the semiconductor dataset.  
**Expected runtime**: ~79 seconds (Intel i7-13700, 32GB RAM).  
**Output**: `case_study_results.json`

### Step 3: Verify Numerical Consistency

```bash
cd figures
python verify_consistency.py
```

Cross-checks all 240+ data points between JSON output and LaTeX tables.

### Step 4: Generate Figures

```bash
cd figures
python generate_figures.py
```

Generates all 22 publication-quality PDF figures. Requires `simulation_results_v3.json`, `case_study_results.json`, and `mcmc_diagnostics.json`.

---

## Key Simulation Design / 仿真设计

| Scenario | Distribution | Components | Key Feature |
|:---|:---|:---:|:---|
| A | Gaussian Mixture | 3 | Correctly specified for DDP base measure |
| B | Skew-Normal Mixture | 2 | Asymmetric; |λ|=3 → 1 across stages |
| C | Student-t Mixture | 2 | Heavy-tailed; ν=3 df |
| D | Mixed Growth | 3 | One subpopulation stagnates during growth |

**Common parameters**: GP degradation with ψ=−2.0, μ_γ=(1.0, 0.7, 0.4), ~20% censoring per stage.  
**MCMC**: 20,000 iterations (10,000 burn-in), K=8 truncation, 3 chains, R=20 replications.  
**Competing methods**: DDP-Joint, DDP-Only, Separate, LN-Bayes, Wbl-Bayes, Ind-DPM.

---

## Key Findings / 核心发现

1. **DDP stage-dependence reduces RMSE by 37–46%** relative to independent DP mixtures (n_l=5, median quantile)
2. **DDP-Joint underperforms DDP-Only at n_l ≤ 10** — ψ is not reliably identifiable; the joint model adds complexity *and* worsens predictive fit
3. **Weibull model achieves zero empirical coverage** under all non-Gaussian scenarios (48/48 combinations)
4. **DDP-Only is recommended for n_l ≤ 10**; DDP-Joint requires n_l ≳ 30 (preliminary calibration)
5. **ψ posterior is wider than prior** (SD 1.45 vs 1.0) — data cannot constrain the association at n_l=5
6. **μ_γ posterior direction is opposite to truth** — further evidence of ψ non-identifiability

---

## Data Provenance / 数据来源

- **Failure time data**: Liu, H., et al. (2026). Bayesian estimation of reliability for semiconductor devices. *Computers & Industrial Engineering*, 211, 111629.
- **Degradation data**: Simulated from a Gaussian process model (transparently acknowledged; see manuscript §6.1). The original Liu (2026a) study did not collect unit-level degradation measurements.
- **All numerical results**: All values in tables and figures are verified against `simulation_results_v3.json`, `case_study_results.json`, and `mcmc_diagnostics.json` by `verify_consistency.py`.

---

## Citation / 引用

If you use this code or data, please cite:

```bibtex
@article{hu2026bayesian,
  title={A Bayesian Nonparametric Joint Modeling Framework for Multi-Stage
         Reliability Growth with Degradation and Lifetime Data},
  author={Hu, Cenyu and Gao, Tielu and Wang, Yabin and Cheng, Zhonghua and Shi, Xianming},
  journal={Reliability Engineering \& System Safety},
  year={2026}
}
```

Original failure time data:
```bibtex
@article{liu2026bayesian,
  title={Bayesian estimation of reliability for semiconductor devices following
         lognormal distribution in multi-stage small sample growth tests},
  author={Liu, H. and Chen, T. and Hu, T. and Zheng, L. and Li, M.},
  journal={Computers \& Industrial Engineering},
  volume={211},
  pages={111629},
  year={2026}
}
```

---

## License / 许可证

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
The semiconductor failure time data is from Liu et al. (2026) — cite accordingly if used.

---

## Contact / 联系方式

**Cenyu Hu** (corresponding author)  
📧 1339816023@qq.com  
🏫 Shijiazhuang Campus, Army Engineering University  
97 Heping West Road, Shijiazhuang 050004, Hebei, China
