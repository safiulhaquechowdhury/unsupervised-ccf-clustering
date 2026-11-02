"""The four clustering algorithms with the settings of Table 2."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

SEED = 42


def kmc(X, k=2):
    """K-Means: k-means++ initialization, 10 restarts, tol 1e-4, max_iter 300."""
    return KMeans(n_clusters=k, init='k-means++', n_init=10,
                  random_state=SEED).fit_predict(X)


def ahc(X, k=2, n_pre=50, batch_size=1000):
    """Two-stage AHC: mini-batch k-means summary, then Ward linkage on the centroids."""
    mbk = MiniBatchKMeans(n_clusters=n_pre, batch_size=batch_size, random_state=SEED)
    initial_labels = mbk.fit_predict(X)
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    centroid_labels = agg.fit_predict(mbk.cluster_centers_)
    return centroid_labels[initial_labels]


def gmmc(X, k=2):
    """Gaussian mixture: full covariance, k-means initialization, n_init 1, tol 1e-3."""
    return GaussianMixture(n_components=k, covariance_type='full',
                           random_state=SEED).fit_predict(X)


def fcmc(X, k=2, m=2.0, max_iter=150):
    """Fuzzy C-Means (fuzzy-c-means 1.7.2): hard label = largest membership."""
    from fcmeans import FCM  # imported lazily so the other models work without it
    fcm = FCM(n_clusters=k, m=m, max_iter=max_iter, random_state=SEED)
    fcm.fit(X)
    return np.argmax(fcm.u, axis=1)


MODELS = {'KMC': kmc, 'AHC': ahc, 'GMMC': gmmc, 'FCMC': fcmc}
