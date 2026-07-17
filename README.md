# Confidence Regulation Neurons

Seminar project for **Trends in AI and LLMs** (HPI): reproducing and extending [Stolfo et al., NeurIPS 2024](https://arxiv.org/abs/2406.16254) — *Confidence Regulation Neurons in Language Models*.

## Layout

| Path | Description |
|---|---|
| `RESEARCH_ROADMAP.md` | Study and reproduction plan |
| `base_paper.pdf` | Local copy of the paper |
| `reference_code/` | Upstream reference ([GitHub](https://github.com/bpwu1/confidence-regulation-neurons)) |
| `src/gpt2/core.py` | GPT-2 metrics, plots, ablation engine |
| `src/gpt2/phase1_identify.py` | Phase 1: static identification (Fig 2a, 2d) |
| `src/gpt2/phase2_ablation.py` | Phase 2: mean ablation (Fig 1, 2c, 2e) |
| `src/gpt2/phase3_case_study.py` | Phase 3: neuron activity & induction (Fig 4–5) |
| `src/gpt2/ext1_qa_ablation.py` | Ext 1: QA ablation on LAMA + TriviaQA |
| `src/gpt2/ext2_weight_projection.py` | Ext 2: inter-layer MLP→MLP null-space ρ |

## Quick start

```powershell
# Windows — activate venv first
.\.venv\Scripts\Activate.ps1

# CUDA PyTorch (required before requirements.txt on Windows; cu121 index stops at 2.5.1)
pip install "torch>=2.6.0,<2.7.0" --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python src/gpt2/phase1_identify.py --device cuda
python src/gpt2/phase2_ablation.py --device cuda
python src/gpt2/phase3_case_study.py --device cuda
```

Ext 1 QA ablation (after phase 1):

```powershell
.\.venv\Scripts\Activate.ps1
python src/qa/run_sanity_checks.py --model gpt2-small
python src/gpt2/ext1_qa_ablation.py --device cuda --split sanity
```

Outputs: `outputs/gpt2-small/phase1/` → `phase2/` → `phase3/`; extensions under `ext1/`, `ext2/`

Ext 2 inter-layer weight projection (static; after optional Phase 1 for W_U sanity):

```powershell
.\.venv\Scripts\Activate.ps1
python src/gpt2/ext2_weight_projection.py --check-functions
python src/gpt2/ext2_weight_projection.py --device cuda
python src/llama/ext2_weight_projection_1b.py --device cuda
python src/llama/ext2_weight_projection_instruct.py --device cuda
python src/qwen/0.8b_base/ext2_weight_projection.py --device cuda
python src/qwen/0.8b_instruct/ext2_weight_projection.py --device cuda
python src/qwen/2b_base/ext2_weight_projection.py --device cuda
python src/qwen/2b_instruct/ext2_weight_projection.py --device cuda
```

For gated models, set `HF_TOKEN` or add `reference_code/ablations/hf_token.txt`.

See `RESEARCH_ROADMAP.md` for full tracks and experiment log.

## License

See `LICENSE`.
