from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from svg_constraints import repair_svg, extract_basic_features

COLOR_WORDS = [
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple',
    'pink', 'brown', 'gray', 'grey', 'gold', 'silver', 'cyan'
]
SHAPE_WORDS = [
    'circle', 'square', 'triangle', 'star', 'heart', 'moon', 'sun', 'flower',
    'car', 'house', 'tree', 'person', 'animal', 'bird', 'fish', 'icon'
]


def normalize_prompt(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def prompt_bucket(prompt: str) -> str:
    p = prompt.lower()
    color = next((c for c in COLOR_WORDS if c in p), 'none')
    shape = next((s for s in SHAPE_WORDS if s in p), 'other')
    nwords = min(40, len(p.split()))
    return f'{shape}|{color}|{nwords//5}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_folds', type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)

    train['prompt'] = train['prompt'].map(normalize_prompt)
    test['prompt'] = test['prompt'].map(normalize_prompt)
    train['svg_raw'] = train['svg']
    train['svg'] = train['svg'].map(repair_svg)

    feats = pd.DataFrame(train['svg'].map(extract_basic_features).tolist())
    train = pd.concat([train, feats], axis=1)

    train['bucket'] = train['prompt'].map(prompt_bucket)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=17)
    train['fold'] = -1
    for fold, (_, va) in enumerate(skf.split(train, train['bucket'])):
        train.loc[va, 'fold'] = fold

    train[['id', 'prompt', 'svg', 'fold']].to_csv(out_dir / 'train_clean.csv', index=False)
    test[['id', 'prompt']].to_csv(out_dir / 'test_clean.csv', index=False)
    train[['id', 'fold', 'bucket', 'svg_len', 'num_paths', 'num_circles', 'num_rects', 'num_groups']].to_csv(out_dir / 'folds.csv', index=False)

    print('saved:', out_dir)
    print(train[['svg_len', 'num_paths']].describe())


if __name__ == '__main__':
    main()
