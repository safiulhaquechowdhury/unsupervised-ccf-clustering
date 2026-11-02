"""Ten-fold resampling evaluation with SS, CHI, and DBI (Sections 3.4 and 4)."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.model_selection import KFold

SEED = 42


def internal_indices(X, labels, ss_sample=None):
    """Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.

    ``ss_sample=None`` computes the exact SS on all points, as in the paper
    (about 17 minutes per training fold on a 2-core CPU). Pass an integer to
    estimate SS on a random subsample for a quick check.
    """
    ss = silhouette_score(X, labels, sample_size=ss_sample,
                          random_state=SEED if ss_sample else None)
    chi = calinski_harabasz_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    return ss, chi, dbi


def ten_fold(fit_predict, X, ss_sample=None, verbose=True):
    """Refit on the nine training folds of each split and score those transactions.

    The held-out fold is not used, so the procedure measures the stability of
    the clustering configuration, not generalization (Section 4).
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train_idx, _) in enumerate(kf.split(X), 1):
        t0 = time.time()
        X_train = X[train_idx]
        labels = fit_predict(X_train)
        ss, chi, dbi = internal_indices(X_train, labels, ss_sample)
        rows.append({'fold': fold, 'n_train': len(train_idx),
                     'SS': ss, 'CHI': chi, 'DBI': dbi})
        if verbose:
            print(f"Fold {fold:2d}: SS={ss:.4f}, CHI={chi:.2f}, DBI={dbi:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def summarize(per_fold):
    """Mean of the ten folds, as reported in Tables 7 and 8."""
    return per_fold[['SS', 'CHI', 'DBI']].mean()


def mean_rank(table):
    """Rank-aggregation rule of Section 3.4.4 (higher SS/CHI and lower DBI are better)."""
    ranks = pd.DataFrame({
        'SS': table['SS'].rank(ascending=False, method='average'),
        'CHI': table['CHI'].rank(ascending=False, method='average'),
        'DBI': table['DBI'].rank(ascending=True, method='average'),
    })
    return ranks.mean(axis=1)
