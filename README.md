# Confidence Regulation Neurons

Seminar project for **Trends in AI and LLMs** (HPI): reproducing and extending [Stolfo et al., NeurIPS 2024](https://arxiv.org/abs/2406.16254) — *Confidence Regulation Neurons in Language Models*.

## Contents

| Path | Description |
|---|---|
| `RESEARCH_ROADMAP.md` | Tier-based study and reproduction plan |
| `base_paper.pdf` | Local copy of the base paper |
| `reference_code/` | Reference implementation ([upstream](https://github.com/bpwu1/confidence-regulation-neurons)) |

## Quick start

```bash
cd reference_code
pip install -r requirements.txt
pip install git+https://github.com/neelnanda-io/neel-plotly.git
pip install git+https://github.com/neelnanda-io/neelutils.git
```

For gated models (e.g. LLaMA), add your HuggingFace token to `reference_code/ablations/hf_token.txt` (not tracked in git).

See `RESEARCH_ROADMAP.md` for reproduction steps and experiment tracks.

## License

See `LICENSE`.
