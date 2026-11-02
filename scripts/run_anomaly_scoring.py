"""Fraud-risk candidate scoring of Section 3.4.5 (KMC, K = 2, all transactions).

Writes every transaction's cluster and score, and the flagged candidates
(none in the reported feature space). Flagged transactions are leads for
manual review, not confirmed fraud: the dataset has no fraud labels.
"""
import json
from pathlib import Path

import numpy as np

from _common import base_parser
from ccf_clustering.data import FEATURES, encode, feature_matrices, load_merged
from ccf_clustering.scoring import kmc_distance_scores, tukey_flags


def main():
    args = base_parser(__doc__).parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = encode(load_merged(args.data_dir))
    X, _ = feature_matrices(df)
    labels, scores = kmc_distance_scores(X)
    flags, fences = tukey_flags(labels, scores)

    scored = df[FEATURES].assign(cluster=labels, score=scores, flagged=flags)
    scored = scored.sort_values(['cluster', 'score'], ascending=[True, False])
    scored.to_csv(out / 'kmc_anomaly_scores.csv.gz', index=False)
    scored[scored['flagged']].to_csv(out / 'kmc_fraud_risk_candidates.csv', index=False)

    report = {
        'n_transactions': int(len(X)),
        'cluster_sizes': {int(c): int((labels == c).sum()) for c in np.unique(labels)},
        'tukey_fence_per_cluster': fences,
        'max_score_per_cluster': {int(c): float(scores[labels == c].max()) for c in np.unique(labels)},
        'n_flagged': int(flags.sum()),
    }
    (out / 'kmc_anomaly_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
