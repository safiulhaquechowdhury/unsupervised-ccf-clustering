# Data

The experiments use the public Kaggle dataset **CreditCard-Fraud-Detection**
(Sharma, 2023): <https://www.kaggle.com/datasets/iabhishekofficial/creditcard-fraud-detection>.
The files are not redistributed in this repository. Download them from Kaggle and
place them here, unchanged:

```
data/
├── cc_info.csv            # 984 cards: credit_card, city, state, zipcode, credit_card_limit
└── transactions.csv.zip   # 294,588 transactions: credit_card, date, transaction_dollar_amount, Long, Lat
```

`transactions.csv.zip` is read directly by pandas; there is no need to unzip it.
The Kaggle dataset contains **no fraud labels**; detection is evaluated on the labeled benchmark below.

## Labeled benchmark (revision of November 2026)

`creditcard.csv` from the ULB Credit Card Fraud Detection dataset (Dal Pozzolo et al., 2015;
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>), 284,807 transactions with 492 frauds.
Place it here unchanged; it is used only by `scripts/run_round4_ulb.py`, and its labels only for evaluation.
