# Confidence Regulation Neurons

Seminar project for **Trends in AI and LLMs** (HPI): reproducing and extending [Stolfo et al., NeurIPS 2024](https://arxiv.org/abs/2406.16254) — *Confidence Regulation Neurons in Language Models*.

## Naming

| Experiment | Meaning | Layout |
|---|---|---|
| **Exp 1** | Paper reproduction (plots & claims) | `exp1/phase1`, `exp1/phase2` |
| **Exp 2** | MLP writes vs later W_in / W_U, plus W_in read of U_dark(W_U) | `exp2/` (no phases) |
| **Exp 3** | Hallucination probes | `exp3/` (pipeline steps, no phases) |

**Exp 1 phases**

| Phase | What it does |
|---|---|
| `phase1` | Static entropy-neuron identification (ρ ranking) |
| `phase2` | Mean ablation (TE / DE_LN, ΔH, prediction change; joint.entropy + joint.random) |

## Layout

| Path | Description |
|---|---|
| `RESEARCH_ROADMAP.md` | Study and reproduction plan |
| `base_paper.pdf` | Local copy of the paper |
| `reference_code/` | Upstream reference ([GitHub](https://github.com/bpwu1/confidence-regulation-neurons)) |
| `src/common/` | Shared utilities (`hyperparameters`, `run_config`, plotting) |
| `src/experiments/exp1/` | Exp 1 package marker (runners under `src/models/*/exp1/`) |
| `src/experiments/exp2/` | Exp 2 shared library (skip pairs, every layer→W_U, U_dark read/visibility) |
| `src/experiments/exp3/` | Exp 3 five-step pipeline (prepare / gather / train / eval / ablate) |
| `data/exp3/` | Exp 3 collections (`<slug>/<dataset>/<split>/`) and item caches |
| `src/models/gpt2/exp1/phase1_identify.py` | Exp 1 Phase 1 |
| `src/models/gpt2/exp1/phase2_ablation.py` | Exp 1 Phase 2 |
| `src/models/gpt2/exp2/weight_projection.py` | Exp 2 |
| `src/models/llama/` | Llama 3.2 loaders + `1b_base/` / `instruct/` runners |
| `src/models/qwen/` | Qwen 3.5 loaders + size/variant runners |
| `src/models/pythia/` | Pythia loaders + `410m/` / `1.4b/` runners |
| `src/models/smollm/` | SmolLM2 loaders + `1.7b_base/` / `1.7b_instruct/` runners |
| `src/models/gemma/` | Gemma-2 loaders + `2b_base/` / `2b_it/` runners |
| `config/` | Per-model YAML run configs (Exp 1–3 hyperparameters) |

## Models & configs

Each model variant has a matching config under `config/` and entrypoint scripts under `src/models/<family>/<variant>/`. CLI flags override YAML values.

| Model variant | Config | Exp 1 | Exp 2 | Exp 3 |
|---|---|:---:|:---:|:---:|
| gpt2 | `config/gpt2.yaml` | ✓ | ✓ | ✓ |
| llama 1B base | `config/llama.yaml` | ✓ | ✓ | ✓ |
| llama 1B instruct | `config/llama_instruct.yaml` | ✓ | ✓ | ✓ |
| qwen 0.8B base / instruct | `config/qwen3_0.8b.yaml` | ✓ | ✓ | — |
| qwen 2B base / instruct | `config/qwen3_2b.yaml` / `qwen3_2b_instruct.yaml` | ✓ | ✓ | ✓ |
| pythia 410m | `config/pythia_410m.yaml` | ✓ | ✓ | ✓ |
| pythia 1.4b | `config/pythia_1.4b.yaml` | ✓ | ✓ | ✓ |
| smollm 1.7B base | `config/smollm2_1.7b.yaml` | ✓ | ✓ | ✓ |
| smollm 1.7B instruct | `config/smollm2_1.7b_instruct.yaml` | ✓ | ✓ | ✓ |
| gemma 2B base | `config/gemma2_2b.yaml` | ✓ | ✓ | ✓ |
| gemma 2B instruct | `config/gemma2_2b_it.yaml` | ✓ | ✓ | ✓ |

Example (pythia 410m, Exp 1):

```powershell
python src/models/pythia/410m/exp1/phase1_identify.py --device cuda
python src/models/pythia/410m/exp1/phase2_ablation.py --device cuda
python src/models/pythia/410m/exp2/weight_projection.py --device cuda
```

## Outputs layout

```
data/exp3/
  gpt2/qa_wikidata/{sanity,main}/
  llama_3.2_1B_instruct/lama_trex/{sanity,main}/
  …

outputs/
  gpt2/
    exp1/{phase1,phase2}/
    exp2/
    exp3/<dataset>/          # probes / ablation
    exp3_sanity/<dataset>/
  llama_3.2_1B_base/   …
  llama_3.2_1B_instruct/ …
  qwen_3.5_0.8B_base/  …
```

## Quick start

```powershell
# Windows — activate venv first
.\.venv\Scripts\Activate.ps1

# CUDA PyTorch (required before requirements.txt on Windows; cu121 index stops at 2.5.1)
pip install "torch>=2.6.0,<2.7.0" --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python src/models/gpt2/exp1/phase1_identify.py --device cuda
python src/models/gpt2/exp1/phase2_ablation.py --device cuda
```

Outputs: `outputs/gpt2/exp1/phase1/` → `phase2/`; extensions under `exp2/`, `exp3/`

Exp 2 (static): each MLP `W_out[ℓ]` vs later `W_in` dark subspaces and vs `W_U`. High ρ = unread; baseline `√(k/d_model)`. Also measures how much each `W_in[ℓ]` **reads** the shared unembedding null space `U_dark(W_U)` (`layer_udark_read.csv`, `read_rho`) and whether top-ρ→`W_U` neurons are unread by later MLP reads (`top_wu_visibility.csv`; high `rho_to_reader_dark` means a later MLP does not read that write). Summaries for every pair; per-neuron CSV for consecutive hops and every layer→`W_U`. Optional Exp 1 Phase 1 for last→`W_U` sanity. Gemma-2 `W_out` is `down_proj` (post-FF RMSNorm is not folded in).

```powershell
.\.venv\Scripts\Activate.ps1
python src/models/gpt2/exp2/weight_projection.py --check-functions
python src/models/gpt2/exp2/weight_projection.py --device cuda
python src/models/llama/1b_base/exp2/weight_projection.py --device cuda
python src/models/llama/instruct/exp2/weight_projection.py --device cuda
python src/models/qwen/0.8b_base/exp2/weight_projection.py --device cuda
python src/models/qwen/0.8b_instruct/exp2/weight_projection.py --device cuda
python src/models/qwen/2b_base/exp2/weight_projection.py --device cuda
python src/models/qwen/2b_instruct/exp2/weight_projection.py --device cuda
```

Exp 3 correctness probes (Wikidata / LAMA T-REx cloze). Five separate scripts; each loops both datasets unless `--dataset` is set. Default arms: last-MLP all, entropy, bottom-k, and 10 random last-MLP subsets (no residual arm).

1. `prepare_datasets.py` — sample and cache items (tokenizer only)
2. `gather_activations.py` — last-token residuals at every layer plus last-MLP posts
3. `train_arms.py` — logistic probes on feature arms
4. `evaluate.py` — held-out metrics and `matched_k_summary`
5. `ablate_entropy_features.py` — frozen last-MLP logistic, mean/zero high-ρ dims vs random controls

Item caches: `data/exp3/{qa_wikidata,lama_trex}_items_<slug>.json`. Collections: `data/exp3/<model_slug>/<dataset>/{sanity,main}/`. Probe artifacts: `outputs/<model_slug>/exp3/<dataset>/` (or `exp3_sanity/<dataset>/`).

```powershell
.\.venv\Scripts\Activate.ps1

# One model, both cloze datasets, one step at a time
python src/experiments/exp3/prepare_datasets.py --config config/llama_instruct.yaml --split main
python src/experiments/exp3/gather_activations.py --config config/llama_instruct.yaml --split main --device cuda
python src/experiments/exp3/train_arms.py --config config/llama_instruct.yaml --split main --device cuda
python src/experiments/exp3/evaluate.py --config config/llama_instruct.yaml --split main
python src/experiments/exp3/ablate_entropy_features.py --config config/llama_instruct.yaml --split main --device cuda

# Fast smoke (64 items, fp16 on CUDA): GPT-2 then Llama-instruct, both datasets
python src/experiments/exp3/run_grid.py --step all --split sanity --fast --model gpt2 --device cuda --force
python src/experiments/exp3/run_grid.py --step all --split sanity --fast --model llama_instruct --device cuda --force

# Five-model fast grid (gpt2, llama_instruct, qwen_2b_instruct, smollm_instruct, pythia_1.4b)
python src/experiments/exp3/run_grid.py --step prepare --split sanity --fast --device cuda --force
python src/experiments/exp3/run_grid.py --step gather --split sanity --fast --device cuda --force
python src/experiments/exp3/run_grid.py --step train --split sanity --fast --device cuda
python src/experiments/exp3/run_grid.py --step eval --split sanity --fast
python src/experiments/exp3/run_grid.py --step ablate --split sanity --fast --device cuda

# Full run (same 5 models x 2 datasets, default N). Sequential on one GPU.
python src/experiments/exp3/run_grid.py --step all --split main --device cuda
```

`--model` keys: `gpt2`, `llama_instruct`, `qwen_2b_instruct`, `smollm_instruct`, `pythia_1.4b`. `--dataset`: `qa_wikidata`, `lama_trex` (default `all`). `--fast` sets N=64 (or `--n`), fp16 on CUDA, and shrinks logistic/MLP/random/ablation knobs. Default train/eval arms are last-MLP all, entropy, bottom-k, and 10 random last-MLP subsets. `--include-resid` adds resid_all. `--headline-only` is all vs entropy only. `--n-random` overrides the 10-draw default. `--step collect` is an alias for prepare then gather.

Labels are stored at gather time (`top1`, `top5`, …). Cloze default is `top1`. Chat templates stay off so last-token scoring is next-token completion.

Step 5 writes `entropy_feature_ablation.json` under the last-MLP logistic dir. It mean-replaces (and zeros) high-ρ dims in stored last-MLP activations, scores the **frozen** probe, and compares to matched random controls. This is probe feature dependence, not a causal LM effect.


For gated models, set `HF_TOKEN` or add `reference_code/ablations/hf_token.txt`.

See `RESEARCH_ROADMAP.md` for full tracks and experiment log.

## License

See `LICENSE`.
