"""Labeled benchmark validation on the ULB European credit card data (round 4).

The fraud labels are never used for fitting: each algorithm is fitted on the
unlabeled training folds of a stratified 10-fold split, the per-cluster Tukey
fences are computed from the training scores, and the held-out fold is scored
and compared with the labels (precision, recall, F1, ROC-AUC, PR-AUC).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ccf_clustering import data, evaluation as ev, models, scoring  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = ROOT / 'results' / 'round4'
OUT.mkdir(parents=True, exist_ok=True)
KS = [int(k) for k in sys.argv[1:]] or [2, 3, 4, 5, 6]

X_all, y = data.load_ulb(ROOT / 'data' / 'creditcard.csv')
print('ULB', X_all.shape, 'frauds', y.sum(), flush=True)
def extra_thresholds(r, y_te, lab_tr, s_tr, lab_te, s_te):
    """Stricter fence (Q3 + 3 IQR) and fixed alert budgets (top 1% and 0.5% of training scores)."""
    f3 = scoring.apply_fences(lab_te, s_te, scoring.tukey_fences(lab_tr, s_tr, 3.0))
    variants = {'w3': f3}
    for b, tag in ((0.01, 'top1'), (0.005, 'top05')):
        variants[tag] = s_te > np.quantile(s_tr, 1 - b)
    for tag, fl in variants.items():
        m = ev.detection_metrics(y_te, fl, s_te)
        for key in ('precision', 'recall', 'f1', 'flag_rate'):
            r[f'{tag}_{key}'] = m[key]


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rows, clus = [], []
for fold, (tr, te) in enumerate(skf.split(X_all, y), 1):
    sc = StandardScaler().fit(X_all[tr])
    Xtr, Xte = sc.transform(X_all[tr]), sc.transform(X_all[te])
    for k in KS:
        for name, Model in models.MODELS.items():
            t0 = time.time()
            m = Model(k=k).fit(Xtr)
            s_tr = m.score(Xtr)
            fences = scoring.tukey_fences(m.labels_, s_tr, 1.5)
            s_te = m.score(Xte)
            lab_te = m.predict(Xte)
            flags = scoring.apply_fences(lab_te, s_te, fences)
            r = {'fold': fold, 'model': name, 'K': k}
            r.update(ev.detection_metrics(y[te], flags, s_te))
            extra_thresholds(r, y[te], m.labels_, s_tr, lab_te, s_te)
            r['SS'], r['CHI'], r['DBI'] = ev.internal_indices(Xtr, m.labels_)
            r['ARI_labels'] = adjusted_rand_score(y[te], lab_te)
            # fraud share of the cluster that contains most frauds (training fold)
            fr = pd.Series(y[tr]).groupby(m.labels_).agg(['mean', 'sum', 'size'])
            r['max_cluster_fraud_rate'] = fr['mean'].max()
            if name == 'FCMC':
                amb = m.ambiguity(Xte)
                r['roc_auc_1-maxu'] = ev.roc_auc_score(y[te], amb)
                r['pr_auc_1-maxu'] = ev.average_precision_score(y[te], amb)
            rows.append(r)
            if k == 2:
                fr['model'], fr['fold'] = name, fold
                clus.append(fr.reset_index().rename(columns={'index': 'cluster'}))
            print(f"fold {fold} {name} K={k}: P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} ROC={r['roc_auc']:.3f} PR={r['pr_auc']:.3f} "
                  f"SS={r['SS']:.3f} flag={r['flag_rate']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    iso = IsolationForest(n_estimators=100, random_state=42).fit(Xtr)
    s_tr, s_te = -iso.score_samples(Xtr), -iso.score_samples(Xte)
    fences = scoring.tukey_fences(np.zeros(len(s_tr), int), s_tr, 1.5)
    flags = scoring.apply_fences(np.zeros(len(s_te), int), s_te, fences)
    r = {'fold': fold, 'model': 'IF', 'K': 0}
    r.update(ev.detection_metrics(y[te], flags, s_te))
    z_tr, z_te = np.zeros(len(s_tr), int), np.zeros(len(s_te), int)
    extra_thresholds(r, y[te], z_tr, s_tr, z_te, s_te)
    rows.append(r)
    print(f"fold {fold} IF: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
          f"ROC={r['roc_auc']:.3f} PR={r['pr_auc']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"ulb_per_fold_K{'-'.join(map(str, KS))}.csv", index=False)
    if clus:
        pd.concat(clus).to_csv(OUT / 'ulb_cluster_fraud_rates_K2.csv', index=False)
print('done', time.ctime(), flush=True)
