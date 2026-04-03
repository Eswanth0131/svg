# Text-to-SVG Generation with Fine-Tuned Qwen2.5-3B

Course project/midterm for NYU Tandon Deep Learning (CS-GY 9223) Spring 2026. Fine-tunes Qwen2.5-3B-Instruct with LoRA on 50,000 prompt-SVG pairs to generate valid SVG images from natural-language prompts. All outputs pass through an SVG repair pipeline at inference time to ensure 100% submission validity.

**Best public leaderboard score: 18.11**

## Model Weights

Pre-trained LoRA adapter weights are available on [Google Drive](https://drive.google.com/drive/folders/1LyNXfaoFn6FXf3-P20Gr4LCi__H5lIHr?usp=share_link) under `artifacts/models/seed11/`. The folder contains the final checkpoint (after 2 epochs) and intermediate checkpoints at steps 4500 and 5000.

## Setup

```bash
pip install -r requirements.txt
```

Requires a GPU with at least 16GB VRAM. The project was developed and run on Google Colab with an A100 GPU. All training and inference uses 4-bit NF4 quantization to fit within memory limits.

## Reproducing Results

There are two ways to run this project:

**Option 1 — Recreate the Google Drive folder structure exactly.**
The notebooks use hardcoded paths under `MyDrive/svg_llm_competition/`. If you replicate this folder structure in your own Drive and place the competition data and model weights in the correct locations, the notebooks will run as-is without any changes.

**Option 2 — Clone this repo and update the paths.**
Clone the repo, then update the `ROOT` / `PROJECT_ROOT` path variables at the top of each notebook to point to wherever you placed the project on your machine or Drive. The scripts in `src/` accept paths as command-line arguments and do not require any path changes.

### Running via Scripts

Place competition data in `data/raw/` (train.csv, test.csv, sample_submission.csv).

**1. Preprocess data** — cleans SVGs, extracts structural features, creates 5-fold splits
```bash
python src/data_prep.py --train_csv data/raw/train.csv --test_csv data/raw/test.csv --out_dir data/processed --n_folds 5
```

**2. Build retrieval index** — encodes training prompts with MiniLM and builds a FAISS index
```bash
python src/build_retrieval_index.py --train_csv data/processed/train_clean.csv --out_dir artifacts/retrieval
```

**3. Fine-tune** (skip if using pre-trained weights from Drive)
```bash
python src/train_sft.py --config configs/base.yaml --run_name seed11
```
Training takes approximately 14.4 hours on a single A100 with the default config (2 epochs, LoRA r=64, lr=2e-4).

**4. Generate predictions**
```bash
python src/generate_candidates.py --mode test --config configs/base.yaml --adapter_dir artifacts/models/seed11/final --out_file artifacts/predictions.csv
```

For the multi-adapter run with retrieval and heuristic candidate selection, use `notebooks/svg_v2.ipynb`.

## Results

| Run | Max Tokens | Retrieval | Public | Private |
|---|---|---|---|---|
| Baseline | 400 | off | 17.45 | 13.23 |
| Fast v2 (best) | 220 | off | 18.11 | 13.60 |
| Multi-adapter | 220 | k=1 | 17.69 | 13.48 |

Shorter generation budgets outperformed longer ones. Cutting max tokens from 400 to 220 improved the public score by 0.66 points with no other changes. Retrieval augmentation with heuristic candidate ranking did not improve over single-pass generation.
