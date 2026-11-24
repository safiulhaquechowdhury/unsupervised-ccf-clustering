"""Data loading, merging, encoding, and feature matrices (Sections 3.2 and 3.3)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Feature matrix of the revised experiments: the credit_card key is excluded,
# State is encoded from the state names, and every column is standardized.
FEATURES = ['State', 'City', 'zipcode', 'credit_card_limit',
            'transaction_dollar_amount', 'Long', 'Lat']

# Columns of the previous submission (kept only to document the earlier error).
FEATURES_PREVIOUS = ['credit_card', 'State', 'City', 'zipcode', 'credit_card_limit',
                     'transaction_dollar_amount', 'Long', 'Lat']


def load_merged(data_dir='data'):
    """Read the two Kaggle files and merge them on the credit_card key."""
    data_dir = Path(data_dir)
    cc = pd.read_csv(data_dir / 'cc_info.csv')
    trans = pd.read_csv(data_dir / 'transactions.csv.zip')
    return pd.merge(cc, trans, on='credit_card', how='left')


def encode(df):
    """Label-encode City from the city names and State from the state names."""
    df = df.copy()
    df['City'] = LabelEncoder().fit_transform(df['city'])
    df['State'] = LabelEncoder().fit_transform(df['state'])
    return df


def encode_previous(df):
    """Encoding of the previous submission, in which State repeated the city codes."""
    df = df.copy()
    le = LabelEncoder()
    df['City'] = le.fit_transform(df['city'])
    df['State'] = le.fit_transform(df['city'])
    return df


def feature_matrix(df, features=FEATURES):
    """Standardized (z-score) matrix used by all four algorithms."""
    X = df[features].to_numpy(dtype=np.float64)
    return StandardScaler().fit_transform(X)


def load_ulb(path):
    """ULB European credit card benchmark (284,807 transactions, 492 frauds).

    Returns the 29 features V1..V28 and log(1 + Amount), and the fraud label.
    Time is excluded because it only counts seconds from the first transaction.
    """
    raw = pd.read_csv(path, header=None)
    if isinstance(raw.iloc[0, 0], str):          # file with a header row
        raw = pd.read_csv(path)
        raw.columns = range(raw.shape[1])
    X = raw.iloc[:, 1:29].to_numpy(dtype=np.float64)
    amount = np.log1p(raw.iloc[:, 29].to_numpy(dtype=np.float64))
    X = np.column_stack([X, amount])
    y = raw.iloc[:, 30].to_numpy(dtype=int)
    return X, y
