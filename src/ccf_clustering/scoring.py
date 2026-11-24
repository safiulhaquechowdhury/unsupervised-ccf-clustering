"""Per-cluster Tukey fence for fraud-risk candidates (Section 3.4.5)."""
import numpy as np


def tukey_fences(labels, scores, whisker=1.5):
    """Upper Tukey fence Q3 + whisker * IQR of the scores in each cluster."""
    fences = {}
    for c in np.unique(labels):
        s = scores[labels == c]
        q1, q3 = np.percentile(s, [25, 75])
        fences[int(c)] = float(q3 + whisker * (q3 - q1))
    return fences


def apply_fences(labels, scores, fences):
    """Flag scores above the fence of their own cluster."""
    tau = np.array([fences.get(int(c), np.inf) for c in labels])
    return scores > tau


def tukey_flags(labels, scores, whisker=1.5):
    fences = tukey_fences(labels, scores, whisker)
    return apply_fences(labels, scores, fences), fences
