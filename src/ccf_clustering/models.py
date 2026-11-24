"""The four clustering algorithms (Table 2) with a common interface.

Each wrapper exposes ``fit(X)``, ``labels_`` (training labels), ``predict(X)``
(hard labels for new transactions), and ``score(X)`` (fraud-risk score of
Section 3.4.5; larger means more atypical).
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

SEED = 42


def _dist_to_assigned(X, centers, labels):
    return np.linalg.norm(X - centers[labels], axis=1)


class KMC:
    """K-Means: k-means++ initialization, 10 restarts, tol 1e-4, max_iter 300."""

    def __init__(self, k=2, init='k-means++', n_init=10, seed=SEED):
        self.model = KMeans(n_clusters=k, init=init, n_init=n_init,
                            random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.centers_ = self.model.cluster_centers_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        lab = self.predict(X)
        return _dist_to_assigned(X, self.centers_, lab)


class AHC:
    """Two-stage AHC: mini-batch k-means summary, then agglomerative linkage on the centroids."""

    def __init__(self, k=2, linkage='ward', n_pre=50, batch_size=1000, seed=SEED):
        self.k, self.linkage, self.n_pre = k, linkage, n_pre
        self.mbk = MiniBatchKMeans(n_clusters=n_pre, batch_size=batch_size,
                                   random_state=seed)

    def fit(self, X):
        init_lab = self.mbk.fit_predict(X)
        agg = AgglomerativeClustering(n_clusters=self.k, linkage=self.linkage)
        self.centroid_labels_ = agg.fit_predict(self.mbk.cluster_centers_)
        self.labels_ = self.centroid_labels_[init_lab]
        # cluster means of the transactions (used for the distance score)
        self.centers_ = np.vstack([X[self.labels_ == c].mean(axis=0)
                                   for c in range(self.k)])
        return self

    def predict(self, X):
        return self.centroid_labels_[self.mbk.predict(X)]

    def score(self, X):
        return _dist_to_assigned(X, self.centers_, self.predict(X))


class GMMC:
    """Gaussian mixture: k-means initialization, n_init 1, tol 1e-3, reg_covar 1e-6."""

    def __init__(self, k=2, covariance_type='full', seed=SEED):
        self.model = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                     random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return -self.model.score_samples(X)          # negative log-likelihood

    def bic(self, X):
        return self.model.bic(X)


class FCMC:
    """Fuzzy C-Means (fuzzy-c-means 1.7.2): hard label = largest membership."""

    def __init__(self, k=2, m=2.0, max_iter=150, seed=SEED):
        from fcmeans import FCM
        self.model = FCM(n_clusters=k, m=m, max_iter=max_iter, random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = np.argmax(self.model.u, axis=1)
        self.centers_ = self.model.centers
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return _dist_to_assigned(X, self.centers_, self.predict(X))

    def ambiguity(self, X):
        """1 - largest membership (boundary ambiguity, reported for comparison)."""
        return 1.0 - self.model.soft_predict(X).max(axis=1)


MODELS = {'KMC': KMC, 'AHC': AHC, 'GMMC': GMMC, 'FCMC': FCMC}
