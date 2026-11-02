"""Data loading, merging, and encoding (Sections 3.2 and 3.3 of the paper)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = ['credit_card', 'State', 'City', 'zipcode', 'credit_card_limit',
            'transaction_dollar_amount', 'Long', 'Lat']


def load_merged(data_dir='data'):
    """Read the two Kaggle files and merge them on the credit_card key."""
    data_dir = Path(data_dir)
    cc = pd.read_csv(data_dir / 'cc_info.csv')
    trans = pd.read_csv(data_dir / 'transactions.csv.zip')
    return pd.merge(cc, trans, on='credit_card', how='left')


def encode(df):
    """Label-encode city and state exactly as in the reported experiments.

    The State column is encoded from the *city* names. This coding error is
    disclosed in Section 3.3 of the paper and is kept so that the published
    tables can be reproduced. Pass the result of ``encode_corrected`` instead
    to use the state names.
    """
    df = df.copy()
    le = LabelEncoder()
    df['City'] = le.fit_transform(df['city'])
    df['State'] = le.fit_transform(df['city'])
    return df


def encode_corrected(df):
    """Encode State from the state names (corrected variant, not used in the paper)."""
    df = df.copy()
    df['City'] = LabelEncoder().fit_transform(df['city'])
    df['State'] = LabelEncoder().fit_transform(df['state'])
    return df


def feature_matrices(df):
    """Return the unscaled matrix (KMC, AHC, FCMC) and the standardized matrix (GMMC)."""
    X = df[FEATURES].to_numpy(dtype=np.float64)
    X_std = StandardScaler().fit_transform(X)
    return X, X_std
