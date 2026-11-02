import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data-dir', default=str(ROOT / 'data'),
                   help='folder with cc_info.csv and transactions.csv.zip')
    p.add_argument('--out-dir', default=str(ROOT / 'results' / 'reproduced'),
                   help='folder for the output CSV files')
    p.add_argument('--ss-sample', type=int, default=None,
                   help='estimate SS on a random subsample of this size '
                        '(default: exact SS on all transactions, as in the paper)')
    return p
