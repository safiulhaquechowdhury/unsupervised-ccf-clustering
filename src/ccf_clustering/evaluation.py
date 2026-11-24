"""Ten-fold resampling evaluation, internal indices, detection metrics, and tests."""
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import KFold

SEED = 42
SS_SAMPLE = 20000


def internal_indices(X, labels, ss_sample=SS_SAMPLE, seed=SEED):
    """SS (on a fixed random subsample of ``ss_sample`` points), exact CHI and DBI."""
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    ss = silhouette_score(X, labels, sample_size=ss_sample, random_state=seed)
    chi = calinski_harabasz_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    return ss, chi, dbi


def ten_fold(make_model, X, ref_labels=None, verbose=True, tag=''):
    """Refit on the nine training folds of each split and score those transactions.

    If ``ref_labels`` (labels of a model fitted on all transactions) is given, the
    adjusted Rand index between the fold model and that reference is computed on
    all transactions, as a measure of partition stability.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train_idx, _) in enumerate(kf.split(X), 1):
        t0 = time.time()
        X_train = X[train_idx]
        m = make_model().fit(X_train)
        ss, chi, dbi = internal_indices(X_train, m.labels_)
        row = {'fold': fold, 'n_train': len(train_idx), 'SS': ss, 'CHI': chi, 'DBI': dbi,
               'min_cluster_share': np.bincount(m.labels_).min() / len(m.labels_)}
        if ref_labels is not None:
            row['ARI'] = adjusted_rand_score(ref_labels, m.predict(X))
        rows.append(row)
        if verbose:
            print(f"{tag} fold {fold:2d}: SS={ss:.4f} CHI={chi:.2f} DBI={dbi:.4f} "
                  + (f"ARI={row['ARI']:.4f} " if 'ARI' in row else '')
                  + f"({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def mean_rank(table):
    """Rank-aggregation rule of Section 3.4.4 (higher SS/CHI and lower DBI are better)."""
    ranks = pd.DataFrame({
        'SS': table['SS'].rank(ascending=False, method='average'),
        'CHI': table['CHI'].rank(ascending=False, method='average'),
        'DBI': table['DBI'].rank(ascending=True, method='average'),
    })
    return ranks.mean(axis=1)


def detection_metrics(y_true, flags, scores):
    """Precision, recall, F1 of the flags; ROC-AUC and PR-AUC of the continuous scores."""
    return {
        'precision': precision_score(y_true, flags, zero_division=0),
        'recall': recall_score(y_true, flags, zero_division=0),
        'f1': f1_score(y_true, flags, zero_division=0),
        'roc_auc': roc_auc_score(y_true, scores),
        'pr_auc': average_precision_score(y_true, scores),
        'flag_rate': float(np.mean(flags)),
    }


def friedman_wilcoxon(per_fold, metric, models):
    """Friedman test across models and Holm-corrected pairwise Wilcoxon tests."""
    M = np.column_stack([per_fold[m][metric].to_numpy() for m in models])
    fr = stats.friedmanchisquare(*M.T)
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            p = stats.wilcoxon(M[:, i], M[:, j]).pvalue
            pairs.append((models[i], models[j], p))
    order = np.argsort([p for *_, p in pairs])
    adj = np.empty(len(pairs))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pairs) - rank) * pairs[idx][2]))
        adj[idx] = running
    out = pd.DataFrame(pairs, columns=['model_a', 'model_b', 'p_wilcoxon'])
    out['p_holm'] = adj
    return fr.statistic, fr.pvalue, out
