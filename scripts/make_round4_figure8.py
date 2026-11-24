"""Figure 8 (a)-(d): K = 2 partitions on the corrected, standardized feature matrix,
shown on its first two principal components."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ccf_clustering import data, models  # noqa: E402

FIG = ROOT / 'results' / 'round4' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
df = data.encode(data.load_merged(ROOT / 'data'))
X = data.feature_matrix(df)
pca = PCA(n_components=2, random_state=42).fit(X)
Z = pca.transform(X)
ev = pca.explained_variance_ratio_
print('explained variance', ev.round(4), 'loadings', dict(zip(data.FEATURES, pca.components_.round(2).tolist())))

titles = {'KMC': 'K-Means Clustering', 'AHC': 'Agglomerative Clustering',
          'GMMC': 'Gaussian Mixture Model (GMM) Clustering', 'FCMC': 'Fuzzy C-Means Clustering'}
for i, (name, Model) in enumerate(models.MODELS.items(), start=9):
    m = Model(k=2).fit(X)
    lab = m.labels_
    print(name, np.bincount(lab))
    fig, ax = plt.subplots(figsize=(5.8, 4.55), dpi=100)
    ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap='viridis', s=12, alpha=0.6, rasterized=True)
    centers = None
    if name == 'KMC':
        centers = m.centers_
    elif name == 'GMMC':
        centers = m.model.means_
    elif name == 'FCMC':
        centers = m.centers_
    if centers is not None:
        c2 = pca.transform(centers)
        ax.scatter(c2[:, 0], c2[:, 1], c='red', marker='X', s=200, label='Centroids')
        ax.legend(loc='lower right')
    ax.set_title(titles[name])
    ax.set_xlabel(f'Principal component 1 ({100 * ev[0]:.1f}% of variance)')
    ax.set_ylabel(f'Principal component 2 ({100 * ev[1]:.1f}% of variance)')
    fig.tight_layout()
    fig.savefig(FIG / f'figure8_{name}.png')
    plt.close(fig)
