# Unsupervised Clustering for Credit Card Fraud-Risk Screening

Code for the article:

> M. H. H. Himal, M. T. Tazwar, M. M. Hossain, S. H. Chowdhury, M. R. Khatun, and M. S. Iqbal,
> **"A Comprehensive Evaluation of Unsupervised Clustering Algorithms for Credit Card Fraud Detection in Large-Scale Transaction Datasets,"**
> *Complexity* (Wiley), 2026.

The study compares four clustering families on a merged cardholder–transaction dataset that has **no fraud labels**:

| Abbreviation | Algorithm | Family |
|---|---|---|
| KMC | K-Means Clustering | partition-based |
| AHC | Agglomerative Hierarchical Clustering (mini-batch k-means + Ward) | hierarchical |
| GMMC | Gaussian Mixture Model Clustering | probabilistic mixture |
| FCMC | Fuzzy C-Means Clustering | fuzzy membership |

Cluster quality is assessed with the Silhouette Score (SS), Calinski–Harabasz Index (CHI), and Davies–Bouldin Index (DBI) under a 10-fold resampling protocol, and fraud-risk candidates are scored with a per-cluster Tukey fence on the distance to the centroid.

---

## Repository structure

```
unsupervised-ccf-clustering/
├── README.md
├── LICENSE                         MIT licence
├── CITATION.cff                    citation metadata (GitHub "Cite this repository")
├── requirements.txt                package versions of the original Colab session
├── data/
│   └── README.md                   how to download the Kaggle files (not redistributed)
├── notebooks/
│   ├── 01_original_runs_with_outputs.ipynb   original Colab cells and outputs behind Tables 7 and 9
│   └── 02_exploratory_figures.ipynb          plotting code for Figures 5, 6, and 7
├── src/ccf_clustering/
│   ├── data.py                     loading, merging, label encoding, feature matrices
│   ├── models.py                   KMC, AHC, GMMC, FCMC with the settings of Table 2
│   ├── evaluation.py               10-fold protocol, SS/CHI/DBI, mean-rank rule
│   └── scoring.py                  distance scores and per-cluster Tukey fence
├── scripts/
│   ├── run_table7_comparison.py    Table 7 (four algorithms) and Table 9 (KMC per fold)
│   ├── run_table8_k_selection.py   Table 8 (KMC, K = 2 to 5)
│   └── run_anomaly_scoring.py      Section 3.4.5 (fraud-risk candidates)
└── results/
    ├── paper_values/               Tables 7, 8, and 9 as published (CSV)
    └── original_run_logs/          per-fold console output of the original runs
```

The code snippets in **Appendix A** of the paper correspond to `src/ccf_clustering/` as follows: Listing A1 → `data.py`, Listing A2 → `evaluation.py`, Listing A3 → `models.py`, Listing A4 → `scoring.py`.

---

## Data

Download **CreditCard-Fraud-Detection** from Kaggle
(<https://www.kaggle.com/datasets/iabhishekofficial/creditcard-fraud-detection>) and place
`cc_info.csv` and `transactions.csv.zip` in `data/`. See [`data/README.md`](data/README.md).

## Installation

```bash
git clone https://github.com/safiulhaquechowdhury/unsupervised-ccf-clustering.git
cd unsupervised-ccf-clustering
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The original experiments ran on Python 3.11 (Google Colab, CPU only).

## Reproducing the results

| Paper item | Command | Output (in `results/reproduced/`) |
|---|---|---|
| Table 7 and Table 9 | `python scripts/run_table7_comparison.py` | `table7_internal_indices.csv`, `kmc_10fold.csv`, … |
| Table 8 | `python scripts/run_table8_k_selection.py` | `table8_kmc_k_selection.csv`, `kmc_k{K}_10fold.csv` |
| Section 3.4.5 | `python scripts/run_anomaly_scoring.py` | `kmc_anomaly_summary.json`, `kmc_anomaly_scores.csv.gz`, `kmc_fraud_risk_candidates.csv` |
| Figures 5–7 | `notebooks/02_exploratory_figures.ipynb` | plots |

**Run time.** The exact Silhouette Score on about 265,000 transactions per training fold takes roughly 17 minutes per fold on a 2-core CPU. For a quick check, add `--ss-sample 20000`; CHI and DBI are then still exact, and SS is estimated from a random subsample.

```bash
python scripts/run_table7_comparison.py --ss-sample 20000 --models KMC GMMC
```

**Expected agreement with the paper.** CHI and DBI of KMC, GMMC, and FCMC reproduce the published fold values exactly (compare with `results/paper_values/` and `results/original_run_logs/`). The AHC values depend on the mini-batch k-means stage and can differ between scikit-learn versions, as reported in Section 3.1.6. With `--ss-sample`, SS differs slightly from the published exact values.

## Settings (Table 2)

| Algorithm | Feature space | Main settings |
|---|---|---|
| KMC | unscaled | K = 2, `init='k-means++'`, `n_init=10`, `tol=1e-4`, `max_iter=300` |
| AHC | unscaled | 50 mini-batch k-means centroids (batch size 1,000), Ward linkage, K = 2 |
| GMMC | standardized (z-scores) | 2 components, full covariance, k-means initialization, `n_init=1`, `tol=1e-3`, `max_iter=100`, `reg_covar=1e-6` |
| FCMC | unscaled | K = 2, m = 2.0, `error=1e-5`, `max_iter=150` |

All models use `random_state=42`; the 10-fold split is `KFold(n_splits=10, shuffle=True, random_state=42)`.

## Important notes for interpretation

- **No fraud labels.** SS, CHI, and DBI measure cluster geometry, not fraud-detection accuracy. Flagged transactions are leads for manual review, not confirmed fraud.
- **Feature spaces differ.** KMC, AHC, and FCMC were run on unscaled features, in which the `credit_card` identifier (about 10¹⁵ to 10¹⁶) dominates the Euclidean distance; GMMC was run on standardized features. The GMMC indices are therefore not directly comparable with the other three (Section 3.3).
- **State encoding.** In the reported experiments the `State` column was encoded from the city names and duplicates the city codes (Section 3.3). `data.encode()` keeps this behaviour so that the published tables can be reproduced; `data.encode_corrected()` encodes the state names instead.
- **Held-out folds are not used.** The 10-fold protocol refits on the nine training folds and scores those same transactions, so it measures stability, not generalization.

## Citation

If you use this code, please cite the article above (see also [`CITATION.cff`](CITATION.cff)).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The Kaggle dataset is subject to its own terms on Kaggle.

## Contact

Corresponding author: Safiul Haque Chowdhury, Department of Computer Science and Engineering, Jahangirnagar University, Savar, Dhaka, Bangladesh (ohinhaque0852@gmail.com).
