# Confidence Regulation Neurons

Seminar project for **Trends in AI and LLMs** (HPI): reproducing and extending [Stolfo et al., NeurIPS 2024](https://arxiv.org/abs/2406.16254) — *Confidence Regulation Neurons in Language Models*.

## Layout

| Path | Description |
|---|---|
| `RESEARCH_ROADMAP.md` | Study and reproduction plan |
| `base_paper.pdf` | Local copy of the paper |
| `reference_code/` | Upstream reference ([GitHub](https://github.com/bpwu1/confidence-regulation-neurons)) |
| `src/gpt2/core.py` | GPT-2 metrics, plots, ablation engine |
| `src/gpt2/load.py` | GPT-2 model loading |
| `src/gpt2/phase1_identify.py` | Phase 1: static neuron identification |
| `src/gpt2/phase2_ablation.py` | Phase 2: mean ablation (TE vs DE_LN) |

## Quick start

```bash
pip install -r requirements.txt
python src/gpt2/phase1_identify.py --device cuda
python src/gpt2/phase2_ablation.py --device cuda
```

Outputs: `outputs/gpt2-small/phase1/` then `outputs/gpt2-small/phase2/`

For gated models, set `HF_TOKEN` or add `reference_code/ablations/hf_token.txt`.

See `RESEARCH_ROADMAP.md` for full tracks and experiment log.

## License

See `LICENSE`.
