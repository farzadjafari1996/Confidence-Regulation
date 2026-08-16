# Confidence Regulation Neurons

Seminar project for **Trends in AI and LLMs** (HPI): reproducing and extending [Stolfo et al., NeurIPS 2024](https://arxiv.org/abs/2406.16254) — *Confidence Regulation Neurons in Language Models*.

## Naming

| Experiment | Meaning | Layout |
|---|---|---|
| **Exp 1** | Paper reproduction (plots & claims) | `exp1/phase1`, `exp1/phase2` |
| **Exp 2** | Layer-by-layer null-space evaluation | `exp2/` (no phases) |
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
| `src/experiments/exp2/` | Exp 2 shared library (weight projection) |
| `src/experiments/exp3/` | Exp 3 shared library (hallucination probe) |
| `data/exp3/` | Exp 3 item cache (BIG-bench `qa_wikidata`) |
| `src/models/gpt2/exp1/phase1_identify.py` | Exp 1 Phase 1 |
| `src/models/gpt2/exp1/phase2_ablation.py` | Exp 1 Phase 2 |
| `src/models/gpt2/exp2/weight_projection.py` | Exp 2 |
| `src/models/gpt2/exp3/` | Exp 3 collect / train / evaluate |
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
| gpt2 | `config/gpt2.yaml` | ✓ | ✓ | partial |
| llama 1B base | `config/llama.yaml` | ✓ | ✓ | ✓ |
| llama 1B instruct | `config/llama_instruct.yaml` | ✓ | ✓ | ✓ |
| qwen 0.8B base / instruct | `config/qwen3_0.8b.yaml` | ✓ | ✓ | — |
| qwen 2B base / instruct | `config/qwen3_2b.yaml` | ✓ | ✓ | — |
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
python src/models/pythia/410m/exp3/collect.py --device cuda --split sanity
```

## Outputs layout

```
outputs/
  gpt2/
    exp1/{phase1,phase2}/
    exp2/
    exp3/  exp3_sanity/
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

Exp 2 inter-layer weight projection (static; after optional Exp 1 Phase 1 for W_U sanity):

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

Exp 3 hallucination probe (BIG-bench `qa_wikidata` cloze → last-MLP acts → correct/wrong classifier):

```powershell
.\.venv\Scripts\Activate.ps1
# Step 1 — forward model once; saves full last-MLP acts + label fields
python src/models/gpt2/exp3/collect.py --device cuda --split sanity
python src/models/gpt2/exp3/collect.py --device cuda --split main
python src/models/llama/1b_base/exp3/collect.py --device cuda --split sanity
python src/models/llama/instruct/exp3/collect.py --device cuda --split sanity

# Step 2 — train probes on the saved dataset (no LLM)
python src/models/gpt2/exp3/train_logistic.py --split main --label top1
python src/models/gpt2/exp3/train_mlp.py --split main --label top1 --device cuda
python src/models/llama/1b_base/exp3/train_logistic.py --split main --label top1
python src/models/llama/instruct/exp3/train_logistic.py --split main --label top1

# Step 2b — retrain on entropy-only or rest-only dims (correct sufficiency test)
python src/models/llama/1b_base/exp3/train_logistic.py --split main --features entropy
python src/models/llama/1b_base/exp3/train_logistic.py --split main --features rest
python src/models/llama/1b_base/exp3/train_mlp.py --split main --features entropy --device cuda
python src/models/llama/1b_base/exp3/train_mlp.py --split main --features rest --device cuda

# Step 3 — held-out test + per-neuron importance
python src/models/gpt2/exp3/evaluate.py --model logistic --split main
python src/models/gpt2/exp3/evaluate.py --model mlp --split main
python src/models/llama/1b_base/exp3/evaluate.py --model logistic --split main
python src/models/llama/instruct/exp3/evaluate.py --model logistic --split main
python src/models/llama/1b_base/exp3/evaluate.py --model logistic --features entropy --split main --skip-importance
python src/models/llama/1b_base/exp3/evaluate.py --model logistic --features rest --split main --skip-importance

# Step 4 — frozen-probe dependence on Exp 1 Phase-1 entropy neurons (auto-runs Phase 1 if missing)
python src/models/llama/1b_base/exp3/ablate_entropy_features.py --model both --split main --device cuda
python src/models/llama/instruct/exp3/ablate_entropy_features.py --model both --split main --device cuda
```

Outputs: `outputs/<model_slug>/exp3/` (or `exp3_sanity/`). Change `--label` (`top1`, `top5`, `confident_error`, …) without re-collecting.

Step 4 writes `entropy_feature_ablation.json` under each probe dir. It mean-replaces (and zeros) high-ρ dims in stored activations, scores the **frozen** logistic/MLP probes, and compares to matched random controls (necessity + sufficiency). This measures probe feature dependence, not a causal LLM effect. Interpret ΔAUROC via the percentile vs random controls.

For gated models, set `HF_TOKEN` or add `reference_code/ablations/hf_token.txt`.

See `RESEARCH_ROADMAP.md` for full tracks and experiment log.

## License

See `LICENSE`.
