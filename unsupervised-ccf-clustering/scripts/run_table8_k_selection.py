"""Reproduce Table 8: KMC internal indices for K = 2, 3, 4, 5 under the 10-fold protocol."""
from pathlib import Path

import pandas as pd

from _common import base_parser
from ccf_clustering.data import encode, feature_matrices, load_merged
from ccf_clustering.evaluation import mean_rank, summarize, ten_fold
from ccf_clustering.models import kmc


def main():
    p = base_parser(__doc__)
    p.add_argument('--k', nargs='+', type=int, default=[2, 3, 4, 5])
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    X, _ = feature_matrices(encode(load_merged(args.data_dir)))
    rows = {}
    for k in args.k:
        print(f"\n=== KMC, K = {k} ===")
        per_fold = ten_fold(lambda Z, k=k: kmc(Z, k), X, ss_sample=args.ss_sample)
        per_fold.to_csv(out / f'kmc_k{k}_10fold.csv', index=False)
        rows[k] = summarize(per_fold)

    table8 = pd.DataFrame(rows).T
    table8['mean_rank'] = mean_rank(table8)
    table8.to_csv(out / 'table8_kmc_k_selection.csv', index_label='K')
    print('\nTable 8:')
    print(table8.round(4).to_string())


if __name__ == '__main__':
    main()
