from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from xgboost import XGBRanker

from svg_constraints import extract_basic_features


def approximate_target_score(prompt: str, cand_svg: str, gt_svg: str) -> float:
    """
    Cheap proxy score for selector training.
    This is not the competition metric, but it is useful for ranking.
    """
    f1 = extract_basic_features(cand_svg)
    f2 = extract_basic_features(gt_svg)
    len_ratio = min(f1['svg_len'], f2['svg_len']) / max(f1['svg_len'], f2['svg_len'])
    path_ratio = min(f1['num_paths'] + 1, f2['num_paths'] + 1) / max(f1['num_paths'] + 1, f2['num_paths'] + 1)
    struct = 0.5 * len_ratio + 0.5 * path_ratio
    return 0.6 * f1['is_valid'] + 0.4 * struct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof_glob', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--train_csv', default='/content/drive/MyDrive/svg/processed/train_clean.csv')
    args = ap.parse_args()

    gt = pd.read_csv(args.train_csv)[['id', 'svg']].rename(columns={'svg': 'gt_svg'})
    files = glob.glob(args.oof_glob)
    if not files:
        raise FileNotFoundError('No OOF parquet files found.')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.merge(gt, on='id', how='left')
    df['target'] = [approximate_target_score(p, c, g) for p, c, g in zip(df.prompt, df.candidate_svg, df.gt_svg)]

    X = df[['svg_len', 'num_paths', 'num_circles', 'num_rects', 'num_groups', 'is_valid', 'temperature', 'top_p']].copy()
    y = df['target'].values
    group_sizes = df.groupby('id').size().tolist()

    model = XGBRanker(
        objective='rank:pairwise',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=17,
    )
    model.fit(X, y, group=group_sizes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / 'selector.json')
    print('saved selector to', out_dir)


if __name__ == '__main__':
    main()
