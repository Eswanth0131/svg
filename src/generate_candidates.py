
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import faiss
import pandas as pd
import torch
import yaml
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from svg_constraints import repair_svg, extract_basic_features

SYSTEM_PROMPT = """You generate a compact, valid SVG for a 256x256 canvas.
Return only SVG XML.
Match the prompt semantically.
Use simple geometric shapes when possible.
Avoid generic icons unless the prompt clearly asks for them.
Prefer clear composition over unnecessary detail."""


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def retrieve(prompt: str, encoder, index, bank_df: pd.DataFrame, k: int = 1) -> List[dict]:
    if k <= 0:
        return []
    q = encoder.encode([prompt], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, k)
    rows = []
    for s, i in zip(scores[0], idxs[0]):
        r = bank_df.iloc[int(i)]
        rows.append({
            "score": float(s),
            "prompt": str(r["prompt"]),
            "svg": str(r["svg"])[:450],
        })
    return rows


def build_user_prompt(prompt: str, retrieved: List[dict]) -> str:
    parts = [f"Prompt: {prompt}"]
    if retrieved:
        r = retrieved[0]
        parts.append("Reference example:")
        parts.append(f"Example prompt: {r['prompt']}")
        parts.append(f"Example SVG snippet: {r['svg']}")
    parts.append("Return exactly one valid SVG.")
    return "\n".join(parts)


def generate_one(model, tok, user_prompt: str, temp: float, top_p: float, max_new_tokens: int) -> str:
    text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    ins = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **ins,
            max_new_tokens=max_new_tokens,
            do_sample=(temp > 0),
            temperature=max(temp, 1e-5),
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    gen = tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if "<svg" in gen:
        gen = gen[gen.find("<svg"):]
    end = gen.rfind("</svg>")
    if end != -1:
        gen = gen[:end+6]
    return repair_svg(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg["test_csv"])

    print("Loading tokenizer/model...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    retrieval_k = int(cfg.get("retrieval_k", 0))
    bank_df = None
    index = None
    encoder = None

    if retrieval_k > 0:
        print("Loading retrieval...", flush=True)
        retr_dir = Path(cfg["retrieval_dir"])
        bank_df = pd.read_parquet(retr_dir / "train_bank.parquet")
        index = faiss.read_index(str(retr_dir / "prompt.index"))
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        print("Retrieval disabled", flush=True)

    temps = list(cfg["candidate_temps"])
    top_ps = list(cfg["candidate_top_ps"])

    rows = []
    total = len(df) * len(temps) * len(top_ps)
    bar = tqdm(total=total, desc="Generating")

    for row_idx, row in enumerate(df.itertuples(index=False), start=1):
        retrieved = retrieve(row.prompt, encoder, index, bank_df, k=retrieval_k) if retrieval_k > 0 else []
        user_prompt = build_user_prompt(row.prompt, retrieved)

        for t in temps:
            for p in top_ps:
                svg = generate_one(model, tok, user_prompt, float(t), float(p), int(cfg["max_new_tokens"]))
                feats = extract_basic_features(svg)
                rows.append({
                    "id": row.id,
                    "prompt": row.prompt,
                    "candidate_svg": svg,
                    "temperature": float(t),
                    "top_p": float(p),
                    **feats,
                })
                bar.update(1)
                bar.set_postfix(prompt=f"{row_idx}/{len(df)}", saved=len(rows))
    bar.close()

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print("saved to", out_file, flush=True)


if __name__ == "__main__":
    main()
