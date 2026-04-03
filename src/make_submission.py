from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from xgboost import XGBRanker


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--adapter_dirs', nargs='+', required=True)
    ap.add_argument('--selector_dir', required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    selector = XGBRanker()
    selector.load_model(str(Path(args.selector_dir) / 'selector.json'))

    # In practice, call generate_candidates.py for each adapter and concatenate.
    # This script assumes those candidate parquet files have already been created.
    candidate_files = sorted(Path(args.selector_dir).parent.glob('test_candidates_*.parquet'))
    if not candidate_files:
        raise FileNotFoundError('Expected pre-generated test_candidates_*.parquet files.')

    df = pd.concat([pd.read_parquet(f) for f in candidate_files], ignore_index=True)
    feats = df[['svg_len', 'num_paths', 'num_circles', 'num_rects', 'num_groups', 'is_valid', 'temperature', 'top_p']]
    df['pred'] = selector.predict(feats)
    best = df.sort_values(['id', 'pred'], ascending=[True, False]).groupby('id').head(1)
    sub = best[['id', 'candidate_svg']].rename(columns={'candidate_svg': 'svg'}).sort_values('id')
    sub.to_csv(args.out_csv, index=False)
    print('saved', args.out_csv)


if __name__ == '__main__':
    main()
