from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

SYSTEM_PROMPT = (
    'You generate compact, valid SVG from a text description. '\
    'Return only SVG XML. Use a 256x256 canvas. Prefer simple clean geometry.'
)


def build_example(prompt: str, svg: str) -> str:
    return (
        f'<|system|>\n{SYSTEM_PROMPT}\n'
        f'<|user|>\nPrompt: {prompt}\nReturn only valid SVG.\n'
        f'<|assistant|>\n{svg}'
    )


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run_name', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.join(cfg['output_root'], args.run_name), exist_ok=True)

    df = pd.read_csv(cfg['train_csv'])
    train_df = df[df['fold'] != cfg['val_fold']].copy()
    val_df = df[df['fold'] == cfg['val_fold']].copy()

    train_ds = Dataset.from_pandas(pd.DataFrame({'text': [build_example(p, s) for p, s in zip(train_df.prompt, train_df.svg)]}))
    val_ds = Dataset.from_pandas(pd.DataFrame({'text': [build_example(p, s) for p, s in zip(val_df.prompt, val_df.svg)]}))

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg.get('load_in_4bit', True),
        bnb_4bit_compute_dtype='bfloat16',
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(cfg['base_model'], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg['base_model'],
        quantization_config=bnb_cfg,
        torch_dtype='auto',
        device_map='auto',
        trust_remote_code=True,
    )

    peft_cfg = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )

    train_args = TrainingArguments(
        output_dir=os.path.join(cfg['output_root'], args.run_name),
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=float(cfg['learning_rate']),
        weight_decay=float(cfg['weight_decay']),
        num_train_epochs=cfg['num_epochs'],
        warmup_ratio=float(cfg['warmup_ratio']),
        logging_steps=cfg['logging_steps'],
        eval_strategy='steps',
        eval_steps=cfg['eval_steps'],
        save_steps=cfg['save_steps'],
        save_total_limit=2,
        bf16=bool(cfg['bf16']),
        report_to='none',
        lr_scheduler_type='cosine',
        seed=int(cfg['seed']),
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(os.path.join(cfg['output_root'], args.run_name, 'final'))
    tok.save_pretrained(os.path.join(cfg['output_root'], args.run_name, 'final'))


if __name__ == '__main__':
    main()
