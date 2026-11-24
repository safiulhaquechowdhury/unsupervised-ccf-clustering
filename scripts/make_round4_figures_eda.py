"""Figures 5 to 7 on the corrected feature set (State from state names, no credit_card key)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ccf_clustering import data  # noqa: E402

FIG = ROOT / 'results' / 'round4' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
df = data.encode(data.load_merged(ROOT / 'data'))
cols = data.FEATURES

# Figure 5: histograms with KDE (original style)
sns.set_style('ticks')
plt.figure(figsize=(15, 7))
for i, col in enumerate(cols, 1):
    ax = plt.subplot(2, 4, i)
    sns.histplot(df[col], bins=30, edgecolor='black', color='#9898f8', kde=False, ax=ax,
                 stat='density', alpha=0.5)
    sns.kdeplot(df[col], color='#4949ff', ax=ax, linewidth=1.5)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
plt.tight_layout()
plt.savefig(FIG / 'figure5_histograms.png', dpi=100)
plt.close()

# Figure 6: box plots
plt.figure(figsize=(15, 6))
for i, col in enumerate(cols, 1):
    ax = plt.subplot(2, 4, i)
    sns.boxplot(x=df[col], color='#9898f8', ax=ax)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=10)
    for lab in ax.get_xticklabels():
        lab.set_fontweight('bold')
plt.tight_layout()
plt.savefig(FIG / 'figure6_boxplots.png', dpi=100)
plt.close()

# Figure 7: Pearson correlation heatmap
plt.figure(figsize=(7.8, 6.2))
sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(FIG / 'figure7_heatmap.png', dpi=100)
plt.close()
print(df[cols].corr().round(2))
print(df[['state', 'State', 'city', 'City']].iloc[[0, 745, 995, 1407, 276725, 282096, 284307, 285283, 287284]])
