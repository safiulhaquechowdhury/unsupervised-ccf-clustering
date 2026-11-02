"""Reproduce Table 7 (four algorithms) and Table 9 (KMC per-fold stability).

Usage:
    python scripts/run_table7_comparison.py                # exact, slow
    python scripts/run_table7_comparison.py --ss-sample 20000   # quick check
"""
from pathlib import Path

import pandas as pd

from _common import base_parser
from ccf_clustering.data import encode, feature_matrices, load_merged
from ccf_clustering.evaluation import mean_rank, summarize, ten_fold
from ccf_clustering.models import MODELS

FEATURE_SPACE = {'KMC': 'unscaled', 'AHC': 'unscaled', 'GMMC': 'standardized', 'FCMC': 'unscaled'}


def main():
    p = base_parser(__doc__)
    p.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    X, X_std = feature_matrices(encode(load_merged(args.data_dir)))
    print(f"Feature matrix: {X.shape[0]} transactions x {X.shape[1]} columns")

    summary = {}
    for name in args.models:
        print(f"\n=== {name} ({FEATURE_SPACE[name]} features) ===")
        data = X_std if FEATURE_SPACE[name] == 'standardized' else X
        per_fold = ten_fold(MODELS[name], data, ss_sample=args.ss_sample)
        per_fold.to_csv(out / f'{name.lower()}_10fold.csv', index=False)
        summary[name] = summarize(per_fold)

    table7 = pd.DataFrame(summary).T
    table7['feature_space'] = [FEATURE_SPACE[m] for m in table7.index]
    same_space = table7[table7['feature_space'] == 'unscaled']
    table7.loc[same_space.index, 'mean_rank'] = mean_rank(same_space)
    table7.to_csv(out / 'table7_internal_indices.csv', index_label='model')
    print('\nTable 7 (mean of 10 folds):')
    print(table7.round(4).to_string())
    print('\nGMMC is not ranked with the others: it was fitted on standardized features.')


if __name__ == '__main__':
    main()
