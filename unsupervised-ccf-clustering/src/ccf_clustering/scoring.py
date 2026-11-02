"""Fraud-risk candidate scoring with a per-cluster Tukey fence (Section 3.4.5)."""
import numpy as np
from sklearn.cluster import KMeans

SEED = 42


def kmc_distance_scores(X, k=2):
    """Fit KMC on all transactions and return labels and distance-to-centroid scores."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=SEED).fit(X)
    labels = km.labels_
    scores = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    return labels, scores


def tukey_flags(labels, scores, whisker=1.5):
    """Flag scores above Q3 + 1.5 IQR, with the fence computed separately per cluster."""
    flags = np.zeros(len(scores), dtype=bool)
    fences = {}
    for c in np.unique(labels):
        in_c = labels == c
        q1, q3 = np.percentile(scores[in_c], [25, 75])
        tau = q3 + whisker * (q3 - q1)
        fences[int(c)] = float(tau)
        flags[in_c] = scores[in_c] > tau
    return flags, fences
