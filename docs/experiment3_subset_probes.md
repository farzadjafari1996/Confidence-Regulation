# Experiment 3 — Entropy vs Rest Feature-Subset Probes

Companion to [`experiment3.md`](experiment3.md). Tests whether Phase-1 **entropy neurons** alone carry the hallucination-detection signal, vs all other last-MLP dims.

## Setup

| Item | Detail |
|------|--------|
| Models | Llama-3.2-1B base, Llama-3.2-1B-Instruct |
| Label | `top1` (binary correct / wrong) |
| Entropy dims | Phase-1 top-ρ neurons, **k = 12**, layer 15 |
| Feature sets | `all` (8192), `entropy` (12), `rest` (8180) |
| Probes | Logistic + MLP (same hyperparameter search as full probe) |
| Eval | Held-out test, n = 1500 (no permutation importance) |

**Method:** train a **new** probe on each feature subset (`--features entropy` / `--features rest`). This replaces the earlier frozen-probe sufficiency ablation, which fed a full-dim probe only 12 activations.

Outputs: `outputs/<model>/exp3/{logistic,mlp}_{entropy,rest}/`

---

## Test results

### Llama 3.2 1B base

| Probe | Features | Acc | F1 | AUROC |
|-------|----------|-----|-----|-------|
| Logistic | all | 0.905 | 0.877 | **0.967** |
| Logistic | entropy | 0.800 | 0.749 | 0.868 |
| Logistic | rest | 0.905 | 0.877 | **0.967** |
| MLP | all | 0.893 | 0.866 | **0.963** |
| MLP | entropy | 0.851 | 0.828 | 0.941 |
| MLP | rest | 0.891 | 0.868 | **0.965** |

### Llama 3.2 1B Instruct

| Probe | Features | Acc | F1 | AUROC |
|-------|----------|-----|-----|-------|
| Logistic | all | 0.867 | 0.818 | **0.951** |
| Logistic | entropy | 0.703 | 0.548 | 0.762 |
| Logistic | rest | 0.867 | 0.819 | **0.951** |
| MLP | all | 0.865 | 0.827 | **0.952** |
| MLP | entropy | 0.787 | 0.749 | 0.887 |
| MLP | rest | 0.863 | 0.823 | **0.949** |

---

## Takeaways

1. **Rest-only ≈ full probe** on both models (AUROC within ~0.002–0.003 of `all`). The bulk of the signal lives outside the 12 entropy neurons.
2. **Entropy-only is informative but weaker**, especially for instruct logistic (AUROC 0.76). MLP on 12 dims does better (base 0.94, instruct 0.89).
3. **Base > instruct** on entropy-only probes; instruct entropy logistic is near chance on F1 (0.55).
4. Frozen-probe ablation (Step 4 in `experiment3.md`) overstated entropy sufficiency — that probe was never trained on 12-dim input.

---

## Reproduce

```powershell
.\.venv\Scripts\Activate.ps1

# Train subset probes (logistic = CPU; MLP = CUDA)
python src/models/llama/1b_base/exp3/train_logistic.py --split main --features entropy --skip-phase1
python src/models/llama/1b_base/exp3/train_logistic.py --split main --features rest --skip-phase1
python src/models/llama/1b_base/exp3/train_mlp.py --split main --features entropy --device cuda --skip-phase1
python src/models/llama/1b_base/exp3/train_mlp.py --split main --features rest --device cuda --skip-phase1

# Evaluate (skip importance for speed)
python src/models/llama/1b_base/exp3/evaluate.py --model logistic --split main --features entropy --skip-importance
python src/models/llama/1b_base/exp3/evaluate.py --model mlp --split main --features rest --skip-importance
```

Same commands under `src/models/llama/instruct/exp3/`.
