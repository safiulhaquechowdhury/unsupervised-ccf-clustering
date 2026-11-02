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
The dataset contains **no fraud labels**, so all results are internal cluster
validation indices, not detection accuracy.
