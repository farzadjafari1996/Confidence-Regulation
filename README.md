# Confidence Regulation Neurons

Seminar project for **Trends in AI and LLMs** (HPI): reproducing and extending [Stolfo et al., NeurIPS 2024](https://arxiv.org/abs/2406.16254) — *Confidence Regulation Neurons in Language Models*.

## Layout

| Path | Description |
|---|---|
| `RESEARCH_ROADMAP.md` | Study and reproduction plan |
| `base_paper.pdf` | Local copy of the paper |
| `reference_code/` | Upstream reference ([GitHub](https://github.com/bpwu1/confidence-regulation-neurons)) |
| `src/common/` | Shared utilities (`hyperparameters`, `run_config`, smoke tests) |
| `src/experiments/exp1/` | Exp 1 shared library (QA ablation) |
| `src/experiments/exp2/` | Exp 2 shared library (weight projection) |
| `src/experiments/exp3/` | Exp 3 shared library (hallucination probe) |
| `data/exp1/` | Exp 1 cached LAMA / TriviaQA items |
| `data/exp2/` | Exp 2 data (reserved) |
| `data/exp3/` | Exp 3 item cache (BIG-bench `qa_wikidata`) |
| `src/models/gpt2/core.py` | GPT-2 metrics, plots, ablation engine |
| `src/models/gpt2/phase1_identify.py` | Phase 1: static identification (Fig 2a, 2d) |
| `src/models/gpt2/phase2_ablation.py` | Phase 2: mean ablation (Fig 1, 2c, 2e) |
| `src/models/gpt2/phase3_case_study.py` | Phase 3: neuron activity & induction (Fig 4–5) |
| `src/models/gpt2/exp1/qa_ablation.py` | Exp 1: QA ablation on LAMA + TriviaQA |
| `src/models/gpt2/exp2/weight_projection.py` | Exp 2: inter-layer MLP→MLP null-space ρ |
| `src/models/gpt2/exp3/collect.py` | Exp 3 Step 1: collect last-MLP activations + LAMA labels |
| `src/models/gpt2/exp3/train_logistic.py` | Exp 3 Step 2a: logistic probe + random search |
| `src/models/gpt2/exp3/train_mlp.py` | Exp 3 Step 2b: MLP probe + random search |
| `src/models/gpt2/exp3/evaluate.py` | Exp 3 Step 3: test metrics + feature importance |
| `src/models/llama/` | Llama 3.2 loaders + `1b_base/` / `instruct/` runners (exp1–3) |
| `src/models/qwen/` | Qwen 3.5 loaders + size/variant runners |

## Outputs layout

```
outputs/
  gpt2/
    reproduce/{phase1,phase2,phase2b,phase3}/
    exp1/  exp1_sanity/
    exp2/
    exp3/  exp3_sanity/
  llama_3.2_1B_base/   …
  llama_3.2_1B_instruct/ …
  qwen_3.5_0.8B_base/  …
  comparison/          # cross-model aggregates
```

## Quick start

```powershell
# Windows — activate venv first
.\.venv\Scripts\Activate.ps1

# CUDA PyTorch (required before requirements.txt on Windows; cu121 index stops at 2.5.1)
pip install "torch>=2.6.0,<2.7.0" --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python src/models/gpt2/phase1_identify.py --device cuda
python src/models/gpt2/phase2_ablation.py --device cuda
python src/models/gpt2/phase3_case_study.py --device cuda
```

Exp 1 QA ablation (after phase 1):

```powershell
.\.venv\Scripts\Activate.ps1
python src/experiments/exp1/run_sanity_checks.py --model gpt2-small
python src/models/gpt2/exp1/qa_ablation.py --device cuda --split sanity
```

Outputs: `outputs/gpt2/reproduce/phase1/` → `phase2/` → `phase3/`; extensions under `exp1/`, `exp2/`

Exp 2 inter-layer weight projection (static; after optional Phase 1 for W_U sanity):

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

# Step 3 — held-out test + per-neuron importance
python src/models/gpt2/exp3/evaluate.py --model logistic --split main
python src/models/gpt2/exp3/evaluate.py --model mlp --split main
python src/models/llama/1b_base/exp3/evaluate.py --model logistic --split main
python src/models/llama/instruct/exp3/evaluate.py --model logistic --split main
```

Outputs: `outputs/<model_slug>/exp3/` (or `exp3_sanity/`). Change `--label` (`top1`, `top5`, `confident_error`, …) without re-collecting.

For gated models, set `HF_TOKEN` or add `reference_code/ablations/hf_token.txt`.

See `RESEARCH_ROADMAP.md` for full tracks and experiment log.

## License

See `LICENSE`.
