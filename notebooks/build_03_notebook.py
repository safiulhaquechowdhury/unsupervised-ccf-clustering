"""Builds and executes notebooks/03_round4_revision_analysis.ipynb (all outputs kept)."""
import nbformat
from nbclient import NotebookClient
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []
md = lambda s: cells.append(nbformat.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbformat.v4.new_code_cell(s))

md("""# Revision round 4: analysis of the rerun experiments

This notebook summarizes the experiments requested by Reviewer 2 (Complexity, manuscript 5411286).
The raw per-fold outputs were produced by `scripts/run_round4_kaggle.py` and `scripts/run_round4_ulb.py`
(console logs in `logs/`). Here they are aggregated into the values of Tables 7 to 13 and the statistical tests.

* Kaggle CreditCard-Fraud-Detection data: `credit_card` key excluded, `State` encoded from the state names,
  all seven columns standardized, one feature matrix for all four algorithms.
* ULB European benchmark (284,807 transactions, 492 frauds): labels used only for evaluation.""")

code("""import json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, sklearn, scipy
from scipy import stats
sys.path.insert(0, str(Path('..') / 'src'))
from ccf_clustering import evaluation as ev
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)
R = Path('..') / 'results' / 'round4'
print(platform.python_version(), 'numpy', np.__version__, 'pandas', pd.__version__,
      'scikit-learn', sklearn.__version__, 'scipy', scipy.__version__)
S = {}   # all numbers used in the manuscript
MODELS = ['KMC', 'AHC', 'GMMC', 'FCMC']""")

md("## Table 7: four algorithms, K = 2, same standardized feature matrix (10-fold)")
code("""main = pd.read_csv(R / 'main_K2_per_fold.csv')
agg = main.groupby('model')[['SS', 'CHI', 'DBI', 'ARI', 'min_cluster_share']].agg(['mean', 'std']).loc[MODELS]
t7 = pd.DataFrame({m: {k: agg.loc[m, (k, 'mean')] for k in ['SS', 'CHI', 'DBI', 'ARI', 'min_cluster_share']} for m in MODELS}).T
t7['mean_rank'] = ev.mean_rank(t7)
for m in MODELS:
    S[f't7_{m}'] = {k: float(agg.loc[m, (k, s)]) for k in ['SS', 'CHI', 'DBI', 'ARI', 'min_cluster_share'] for s in ['mean']}
    S[f't7_{m}'].update({k + '_sd': float(agg.loc[m, (k, 'std')]) for k in ['SS', 'CHI', 'DBI', 'ARI']})
    S[f't7_{m}']['mean_rank'] = float(t7.loc[m, 'mean_rank'])
    S[f't7_{m}']['SS_range'] = [float(main[main.model == m].SS.min()), float(main[main.model == m].SS.max())]
    S[f't7_{m}']['ARI_range'] = [float(main[main.model == m].ARI.min()), float(main[main.model == m].ARI.max())]
t7.round(4)""")

md("### Friedman test and Holm-corrected Wilcoxon signed-rank tests across the ten folds")
code("""pf = {m: main[main.model == m].sort_values('fold').reset_index(drop=True) for m in MODELS}
S['friedman_kaggle'] = {}
for metric in ['SS', 'CHI', 'DBI']:
    chi2, p, pairs = ev.friedman_wilcoxon(pf, metric, MODELS)
    S['friedman_kaggle'][metric] = {'chi2': float(chi2), 'p': float(p),
                                    'max_p_holm': float(pairs.p_holm.max())}
    print(metric, 'Friedman chi2 = %.2f, p = %.2e' % (chi2, p)); print(pairs.round(4))
# KMC vs AHC on each index, number of folds in which KMC is better
S['kmc_better_than_ahc_CHI_folds'] = int((pf['KMC'].CHI > pf['AHC'].CHI).sum())""")

md("### Table 9: KMC per fold")
code("""t9 = pf['KMC'][['fold', 'SS', 'CHI', 'DBI', 'ARI']]
S['t9'] = t9.to_dict('list')
t9.round(4)""")

md("### Accuracy of the subsampled Silhouette Score (training fold 1, KMC)")
code("""chk = json.load(open(R / 'ss_exact_check_fold1.json')); S['ss_check'] = chk['KMC']; chk""")

md("## Table 8: K = 2 to 6 for all four algorithms")
code("""ks = pd.read_csv(R / 'k_sweep_per_fold.csv')
t8 = ks.groupby(['model', 'K'])[['SS', 'CHI', 'DBI', 'ARI', 'min_cluster_share']].mean()
t8['rank_within_model'] = np.nan
for m in MODELS:
    sub = t8.loc[m]
    t8.loc[m, 'rank_within_model'] = ev.mean_rank(sub).values
t8 = t8.reindex(MODELS, level=0)
S['t8'] = {f'{m}_K{k}': {c: float(t8.loc[(m, k), c]) for c in t8.columns} for m, k in t8.index}
ib = pd.read_csv(R / 'k_sweep_inertia_bic.csv'); S['inertia_bic'] = ib.to_dict('list')
display(t8.round(4)); ib""")

md("## Table 10: hyperparameter sensitivity at K = 2, seed stability, and feature-set sensitivity")
code("""hy = pd.read_csv(R / 'hyper_K2_per_fold.csv')
t10 = hy.groupby(['model', 'config'], sort=False)[['SS', 'CHI', 'DBI', 'ARI', 'min_cluster_share']].mean()
S['t10'] = {c: {k: float(v) for k, v in row.items()} for (m, c), row in t10.iterrows()}
seeds = pd.read_csv(R / 'seed_stability.csv'); S['seeds'] = seeds.set_index('model').to_dict('index')
fs = pd.read_csv(R / 'features_continuous5_per_fold.csv')
t_fs = fs.groupby('model')[['SS', 'CHI', 'DBI', 'ARI', 'min_cluster_share']].mean().loc[MODELS]
t_fs['mean_rank'] = ev.mean_rank(t_fs)
S['features5'] = t_fs.to_dict('index')
display(t10.round(4)); display(seeds.round(4)); t_fs.round(4)""")

md("## Table 11: fraud-risk candidates on the Kaggle data (no labels)")
code("""sc = pd.read_csv(R / 'scoring_summary.csv'); S['scoring'] = sc.to_dict('records')
ov = pd.read_csv(R / 'scoring_overlap.csv'); S['overlap'] = ov.to_dict('records')
rho = pd.read_csv(R / 'scoring_spearman.csv', index_col=0); S['score_spearman'] = rho.to_dict()
display(sc.round(3)); display(ov.round(3)); rho.round(3)""")

md("## Table 12: detection performance on the labeled ULB benchmark (stratified 10-fold, K = 2)")
code("""u = pd.read_csv(R / 'ulb_per_fold_K2.csv')
DET = ['KMC', 'AHC', 'GMMC', 'FCMC', 'IF']
cols = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'flag_rate',
        'w3_precision', 'w3_recall', 'w3_f1', 'w3_flag_rate',
        'top1_precision', 'top1_recall', 'top1_f1', 'top1_flag_rate', 'top05_precision', 'top05_recall', 'top05_f1', 'top05_flag_rate']
t12m = u.groupby('model')[cols].mean().loc[DET]; t12s = u.groupby('model')[cols].std().loc[DET]
S['t12'] = {m: {**{c: float(t12m.loc[m, c]) for c in cols}, **{c + '_sd': float(t12s.loc[m, c]) for c in cols}} for m in DET}
S['ulb_prevalence'] = 492 / 284807
for m in MODELS:
    S['t12'][m]['SS'] = float(u[u.model == m].SS.mean())
    S['t12'][m]['ARI_labels'] = float(u[u.model == m].ARI_labels.mean())
S['fcmc_ambiguity_ulb'] = {'roc_auc': float(u[u.model == 'FCMC']['roc_auc_1-maxu'].mean()),
                           'pr_auc': float(u[u.model == 'FCMC']['pr_auc_1-maxu'].mean())}
display(t12m.round(4)); t12s.round(4)""")

code("""upf = {m: u[u.model == m].sort_values('fold').reset_index(drop=True) for m in DET}
S['friedman_ulb'] = {}
for metric in ['f1', 'pr_auc', 'roc_auc']:
    chi2, p, pairs = ev.friedman_wilcoxon(upf, metric, DET)
    S['friedman_ulb'][metric] = {'chi2': float(chi2), 'p': float(p), 'pairs': pairs.to_dict('records')}
    print(metric, 'Friedman chi2 = %.2f, p = %.3g' % (chi2, p)); print(pairs.round(4))""")

md("### Fraud share of the clusters (K = 2, training folds)")
code("""cf = pd.read_csv(R / 'ulb_cluster_fraud_rates_K2.csv')
cf['share'] = cf['mean']
S['cluster_fraud'] = {m: {'max_rate_per_fold': cf[cf.model == m].groupby('fold')['mean'].max().round(4).tolist()} for m in MODELS}
cf.pivot_table(index=['model', 'fold'], columns='cluster', values=['mean', 'size'])""")

md("## Do internal indices predict detection? (ULB, 4 algorithms x K = 2 to 6)")
code("""uk = pd.concat([u, pd.read_csv(R / 'ulb_per_fold_K3-4-5-6.csv')]); uk = uk[uk.model != 'IF']
cfg = uk.groupby(['model', 'K'])[['SS', 'CHI', 'DBI', 'f1', 'pr_auc', 'roc_auc', 'recall', 'precision']].mean()
S['ulb_k_configs'] = {f'{m}_K{k}': {c: float(v) for c, v in r.items()} for (m, k), r in cfg.iterrows()}
S['spearman_index_vs_detection'] = {}
for idx in ['SS', 'CHI', 'DBI']:
    for det in ['pr_auc', 'f1', 'roc_auc']:
        r_, p_ = stats.spearmanr(cfg[idx], cfg[det])
        S['spearman_index_vs_detection'][f'{idx}_{det}'] = {'rho': float(r_), 'p': float(p_)}
        print(f'{idx} vs {det}: rho = {r_:.3f}, p = {p_:.3f}')
S['best_SS_config'] = str(cfg['SS'].idxmax()); S['best_prauc_config'] = str(cfg['pr_auc'].idxmax())
S['best_f1_config'] = str(cfg['f1'].idxmax())
cfg.round(4)""")

code("""json.dump(S, open(R / 'summary_round4.json', 'w'), indent=1, default=float)
print('saved', R / 'summary_round4.json')""")

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = HERE / '03_round4_revision_analysis.ipynb'
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
nbformat.write(nb, out)
print('executed', out)
