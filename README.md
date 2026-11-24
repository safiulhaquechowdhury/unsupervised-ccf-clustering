# Unsupervised Clustering for Credit Card Fraud-Risk Screening

Code for the article:

> M. H. H. Himal, M. T. Tazwar, M. M. Hossain, S. H. Chowdhury, M. R. Khatun, and M. S. Iqbal,
> **"A Comprehensive Evaluation of Unsupervised Clustering Algorithms for Credit Card Fraud Detection in Large-Scale Transaction Datasets,"**
> *Complexity* (Wiley), 2026.

The study compares four clustering families (KMC, AHC, GMMC, FCMC) for fraud-risk screening of unlabeled
card transactions and validates the resulting risk scores on a labeled benchmark.

## Revision of November 2026 (current version)

Following the second reviewer, all experiments were repeated on a corrected feature matrix:

* the `credit_card` key is used only to merge the two files and is **not** a clustering feature;
* `State` is label-encoded from the **state** names (it repeated the city codes in the earlier version);
* all seven features (`State`, `City`, `zipcode`, `credit_card_limit`, `transaction_dollar_amount`, `Long`, `Lat`)
  are **standardized**, and the same matrix is used by all four algorithms.

New analyses: K = 2 to 6 for every algorithm, hyperparameter, seed and feature-set sensitivity, fraud-risk
candidates on the corrected matrix, Friedman and Holm-corrected Wilcoxon tests, and validation on the labeled
ULB European credit card benchmark (precision, recall, F1, ROC-AUC, PR-AUC).

## Earlier version

The code, notebooks and logs of the earlier submission (superseded) remain available in the commit history,
at commit `c91f834`.

## Repository structure

```
src/ccf_clustering/        data.py, models.py, evaluation.py, scoring.py (Listings A1 to A4)
scripts/
  run_round4_kaggle.py     Tables 7 to 11 (stages: main k hyper seeds features scoring)
  run_round4_ulb.py        Tables 12 and 13 (labeled benchmark); arguments = values of K
  make_round4_figures_eda.py   Figures 5 to 7
  make_round4_figure8.py       Figure 8 (a) to (d)
  check_ss_subsample.py        exact vs subsampled Silhouette Score (Section 3.1.6)
notebooks/
  03_round4_revision_analysis.ipynb   executed notebook: all table values and statistical tests
  build_03_notebook.py                builds and executes the notebook above
results/round4/            per-fold CSV files, summary_round4.json, figures/
                           (scores_all_transactions.csv.gz, 16 MB, is written by run_round4_kaggle.py and not stored here)
logs/                      console output of every run
```

## Data

* Kaggle **CreditCard-Fraud-Detection** (Sharma, 2023): <https://www.kaggle.com/datasets/iabhishekofficial/creditcard-fraud-detection>,
  files `cc_info.csv` and `transactions.csv.zip` in `data/` (no fraud labels).
* ULB **Credit Card Fraud Detection** (Dal Pozzolo et al., 2015): <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>,
  file `creditcard.csv` in `data/` (284,807 transactions, 492 frauds; with or without header row).

The data files are not redistributed.

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/run_round4_kaggle.py main k hyper seeds features scoring   # about 1 h on 2 CPU cores
python scripts/run_round4_ulb.py 2          # Tables 12 and 13
python scripts/run_round4_ulb.py 3 4 5 6    # K sweep on the benchmark
python scripts/make_round4_figures_eda.py
python scripts/make_round4_figure8.py
python notebooks/build_03_notebook.py       # aggregates everything into the executed notebook
```

SS is computed on a fixed random subsample of 20,000 transactions per training fold (CHI and DBI are exact);
`scripts/check_ss_subsample.py` compares it with the exact value. Software of the reported runs: Python 3.13,
NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, SciPy 1.18.1, fuzzy-c-means 1.7.2, CPU only.

## Notes for interpretation

* SS, CHI and DBI measure cluster geometry, not fraud-detection accuracy; detection is measured on the labeled benchmark,
  whose labels are used only for evaluation.
* Flagged transactions are leads for manual review, not confirmed fraud.

## Citation and licence

See [`CITATION.cff`](CITATION.cff). Code: MIT ([`LICENSE`](LICENSE)). The datasets are subject to their own terms.

## Contact

Corresponding author: Safiul Haque Chowdhury, Department of Computer Science and Engineering, Jahangirnagar University,
Savar, Dhaka, Bangladesh (ohinhaque0852@gmail.com).
