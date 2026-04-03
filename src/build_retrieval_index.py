from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model_name', default='sentence-transformers/all-MiniLM-L6-v2')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    model = SentenceTransformer(args.model_name)
    emb = model.encode(df['prompt'].tolist(), batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype='float32')

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out_dir / 'prompt.index'))
    df[['id', 'prompt', 'svg']].to_parquet(out_dir / 'train_bank.parquet', index=False)
    np.save(out_dir / 'prompt_emb.npy', emb)
    print('saved retrieval bank to', out_dir)


if __name__ == '__main__':
    main()
