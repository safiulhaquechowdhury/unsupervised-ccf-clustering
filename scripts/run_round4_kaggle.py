"""Revised experiments on the Kaggle CreditCard-Fraud-Detection data (round 4).

Corrected feature matrix: credit_card key excluded, State encoded from the state
names, all seven columns standardized, and the same matrix used by all four
algorithms. Produces the values of Tables 7 to 9 and 12 to 14 and Section 3.4.5.
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ccf_clustering import data, evaluation as ev, models, scoring  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

OUT = ROOT / 'results' / 'round4'
OUT.mkdir(parents=True, exist_ok=True)
STAGES = sys.argv[1:] or ['main', 'k', 'hyper', 'seeds', 'features', 'scoring']

df = data.encode(data.load_merged(ROOT / 'data'))
X = data.feature_matrix(df)
print('Feature matrix', X.shape, data.FEATURES, flush=True)
M = models.MODELS


def run(name, make, Xm, tag):
    ref = make().fit(Xm).labels_
    pf = ev.ten_fold(make, Xm, ref_labels=ref, tag=tag)
    pf.insert(0, 'config', tag)
    pf.insert(0, 'model', name)
    return pf


# ---------------------------------------------------------------- main (K = 2)
if 'main' in STAGES:
    frames = [run(n, lambda n=n: M[n](k=2), X, f'{n}_K2') for n in M]
    pf = pd.concat(frames)
    pf.to_csv(OUT / 'main_K2_per_fold.csv', index=False)
    print(pf.groupby('model')[['SS', 'CHI', 'DBI', 'ARI', 'min_cluster_share']].agg(['mean', 'std']), flush=True)

# ---------------------------------------------------------------- K = 2..6
if 'k' in STAGES:
    frames, extra = [], []
    for k in range(2, 7):
        for n in M:
            frames.append(run(n, lambda n=n, k=k: M[n](k=k), X, f'{n}_K{k}'))
            frames[-1]['K'] = k
        km = models.KMC(k=k).fit(X)
        gm = models.GMMC(k=k).fit(X)
        extra.append({'K': k, 'KMC_inertia': km.model.inertia_, 'GMMC_BIC': gm.bic(X)})
    gm1 = models.GMMC(k=1).fit(X)
    extra.insert(0, {'K': 1, 'KMC_inertia': float(((X - X.mean(0)) ** 2).sum()), 'GMMC_BIC': gm1.bic(X)})
    pd.concat(frames).to_csv(OUT / 'k_sweep_per_fold.csv', index=False)
    pd.DataFrame(extra).to_csv(OUT / 'k_sweep_inertia_bic.csv', index=False)

# ---------------------------------------------------------------- hyperparameters at K = 2
if 'hyper' in STAGES:
    configs = [
        ('KMC', 'init=random, n_init=10', lambda: models.KMC(k=2, init='random')),
        ('KMC', 'init=k-means++, n_init=1', lambda: models.KMC(k=2, n_init=1)),
        ('AHC', 'linkage=complete, 50 centroids', lambda: models.AHC(k=2, linkage='complete')),
        ('AHC', 'linkage=average, 50 centroids', lambda: models.AHC(k=2, linkage='average')),
        ('AHC', 'linkage=ward, 200 centroids', lambda: models.AHC(k=2, n_pre=200)),
        ('GMMC', 'covariance=tied', lambda: models.GMMC(k=2, covariance_type='tied')),
        ('GMMC', 'covariance=diag', lambda: models.GMMC(k=2, covariance_type='diag')),
        ('GMMC', 'covariance=spherical', lambda: models.GMMC(k=2, covariance_type='spherical')),
        ('FCMC', 'm=1.5', lambda: models.FCMC(k=2, m=1.5)),
        ('FCMC', 'm=2.5', lambda: models.FCMC(k=2, m=2.5)),
        ('FCMC', 'm=3.0', lambda: models.FCMC(k=2, m=3.0)),
    ]
    frames = [run(n, mk, X, f'{n} | {c}') for n, c, mk in configs]
    pd.concat(frames).to_csv(OUT / 'hyper_K2_per_fold.csv', index=False)

# ---------------------------------------------------------------- seed sensitivity
if 'seeds' in STAGES:
    rows = []
    seeds = [0, 1, 7, 42, 123]
    for n in M:
        labs = [M[n](k=2, seed=s).fit(X).labels_ for s in seeds]
        aris = [adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]
        rows.append({'model': n, 'ARI_mean': np.mean(aris), 'ARI_min': np.min(aris)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'seed_stability.csv', index=False)

# ---------------------------------------------------------------- feature-set sensitivity
if 'features' in STAGES:
    cont = ['zipcode', 'credit_card_limit', 'transaction_dollar_amount', 'Long', 'Lat']
    X2 = data.feature_matrix(df, cont)
    frames = [run(n, lambda n=n: M[n](k=2), X2, f'{n}_continuous5') for n in M]
    pd.concat(frames).to_csv(OUT / 'features_continuous5_per_fold.csv', index=False)

# ---------------------------------------------------------------- fraud-risk scoring
if 'scoring' in STAGES:
    # behavioural descriptors used only to characterise the flagged transactions
    med = df.groupby('credit_card')[['Long', 'Lat']].transform('median')
    lat1, lon1 = np.radians(df['Lat']), np.radians(df['Long'])
    lat2, lon2 = np.radians(med['Lat']), np.radians(med['Long'])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df['km_from_card_median'] = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    df['amount_to_limit'] = df['transaction_dollar_amount'] / df['credit_card_limit']
    card_med_amt = df.groupby('credit_card')['transaction_dollar_amount'].transform('median')
    df['amount_to_card_median'] = df['transaction_dollar_amount'] / card_med_amt

    summary, flags_all = [], {}
    for n in M:
        m = M[n](k=2).fit(X)
        sc = m.score(X)
        for w in (1.5, 3.0):
            fl, fences = scoring.tukey_flags(m.labels_, sc, w)
            flags_all[(n, w)] = fl
            sub, rest = df[fl], df[~fl]
            summary.append({
                'model': n, 'whisker': w, 'n_flagged': int(fl.sum()),
                'pct_flagged': 100 * fl.mean(),
                'cards_with_flags': int(sub['credit_card'].nunique()),
                'median_amount_flagged': sub['transaction_dollar_amount'].median(),
                'median_amount_other': rest['transaction_dollar_amount'].median(),
                'median_km_flagged': sub['km_from_card_median'].median(),
                'median_km_other': rest['km_from_card_median'].median(),
                'median_amt_to_cardmed_flagged': sub['amount_to_card_median'].median(),
                'median_amt_to_cardmed_other': rest['amount_to_card_median'].median(),
                'share_km_gt_100_flagged': (sub['km_from_card_median'] > 100).mean(),
                'share_km_gt_100_other': (rest['km_from_card_median'] > 100).mean(),
            })
            print(summary[-1], flush=True)
        if n == 'FCMC':
            amb = m.ambiguity(X)
            fl, _ = scoring.tukey_flags(m.labels_, amb, 1.5)
            print('FCMC ambiguity score (1 - max u) flags:', int(fl.sum()), flush=True)
            summary.append({'model': 'FCMC (1 - max u)', 'whisker': 1.5, 'n_flagged': int(fl.sum()),
                            'pct_flagged': 100 * fl.mean()})
        df[f'score_{n}'] = sc
    iso = IsolationForest(n_estimators=100, random_state=42).fit(X)
    isc = -iso.score_samples(X)
    fl_iso, _ = scoring.tukey_flags(np.zeros(len(X), int), isc, 1.5)
    flags_all[('IF', 1.5)] = fl_iso
    summary.append({'model': 'IsolationForest', 'whisker': 1.5, 'n_flagged': int(fl_iso.sum()),
                    'pct_flagged': 100 * fl_iso.mean()})
    pd.DataFrame(summary).to_csv(OUT / 'scoring_summary.csv', index=False)
    keys = [k for k in flags_all if k[1] == 1.5]
    ov = []
    for a_, b_ in combinations(keys, 2):
        fa, fb = flags_all[a_], flags_all[b_]
        ov.append({'a': a_[0], 'b': b_[0], 'jaccard': (fa & fb).sum() / max(1, (fa | fb).sum())})
    pd.DataFrame(ov).to_csv(OUT / 'scoring_overlap.csv', index=False)
    rho = df[[f'score_{n}' for n in M]].assign(IF=isc).corr(method='spearman')
    rho.to_csv(OUT / 'scoring_spearman.csv')
    print(pd.DataFrame(ov), rho, sep='\n', flush=True)
    keep = ['credit_card', 'transaction_dollar_amount', 'credit_card_limit', 'km_from_card_median'] + \
           [f'score_{n}' for n in M]
    out = df[keep].copy()
    out['score_IF'] = isc
    for n in M:
        out[f'flag_{n}'] = flags_all[(n, 1.5)]
    out.to_csv(OUT / 'scores_all_transactions.csv.gz', index=False)
print('done', time.ctime(), flush=True)
