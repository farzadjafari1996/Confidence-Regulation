# Experiment 3 — Hallucination Probe (Llama 3.2 1B)

## What is learned on what

We train small **probes** (logistic regression and an MLP) on frozen LLM internal states. The LLM itself is not fine-tuned.

| Role | Content |
|------|---------|
| **Input (X)** | Last-layer MLP post-activation at the final prompt token (`d_mlp = 8192`) |
| **Target (y)** | Whether the LLM’s **top-1 next-token** answer matches the gold answer (`top1` label) |
| **LLM under study** | Llama-3.2-1B and Llama-3.2-1B-Instruct (forward pass only, to collect X and y) |
| **What the probe learns** | A mapping from those activations → “LM answer correct?” (hallucination / factual-error signal) |
| **What it does *not* learn** | The cloze answer itself; the probe does not generate tokens |

In short: **probe learns correctness of the LM’s prediction from the LM’s last-MLP activations.**

---

## Dataset

| Item | Detail |
|------|--------|
| Source | BIG-bench `qa_wikidata` (`tasksource/bigbench`) |
| Format | Cloze (e.g. `"Havana is capital of"` → `"Cuba"`) |
| Filter | Gold/aliases must map to ≥1 tokenizer ID |
| Sample | N = **10 000**, seed **42** |
| Split | **70 / 15 / 15** → 7000 / 1500 / 1500, stratified on `top1` |
| Features | Last-layer MLP post-activation at final prompt token |
| Dim / layer | **d_mlp = 8192**, layer **15** |
| Chat template | Off (wraps break completion) |

LM accuracy on this set: base **38.2%** top-1 / **57.6%** top-5; instruct **35.6%** / **56.7%**.

**Labels** (stored at collect time; switch without re-collecting):

| Label | Task | Definition |
|-------|------|------------|
| `top1` (used here) | binary | pred ∈ gold IDs |
| `top5` | binary | gold rank &lt; 5 |
| `confident_error` | binary | wrong and `top1_prob ≥ 0.5` |
| `multiclass` | 3-way | correct / wrong+confident / wrong+uncertain |
| `gold_logprob` | regression | log-prob of gold token |

---

## Pipeline

1. **Collect** — forward LM once; save `X_full` + label fields → `dataset.npz`
2. **Train** — logistic and MLP probes with random hyperparameter search (no LLM)
3. **Evaluate** — held-out test Acc / F1 / AUROC
4. **Ablate** — freeze probe; mean/zero-replace Phase-1 entropy dims; compare to random controls

---

## Probe architectures

**Logistic:** `StandardScaler` → L2 `LogisticRegression` (`lbfgs`, `max_iter=2000`).

**MLP (`ProbeMLP`):**
```
Linear(d_in → H) → ReLU → Dropout
→ Linear(H → H/2) → ReLU → Dropout
→ Linear(H/2 → 1)
```

Best selected configs (`top1`):

| Model | Logistic | MLP |
|-------|----------|-----|
| Base | C=0.001, L2, no class weight | **8192 → 256 → 128 → 1**, dropout 0.3, lr≈1.75e-4, wd≈9.6e-5 |
| Instruct | C=0.001, L2, no class weight | **8192 → 64 → 32 → 1**, dropout 0.0, lr≈2.05e-4, wd≈4.2e-6 |

---

## Loss and hyperparameters

| Setting | Value |
|---------|-------|
| Seed | 42 |
| Logistic loss | Log-loss (sklearn); CV scoring = AUROC |
| Logistic search | `n_iter=40`, `cv=3`; `C ∈ logspace(-3,3)` (40 vals); `class_weight ∈ {None, balanced}`; penalty=L2; solver=`lbfgs` |
| MLP loss (binary) | `BCEWithLogitsLoss` with `pos_weight = (1−p)/p` |
| MLP optimizer | AdamW |
| MLP search | `n_iter=20`; `H ∈ {64,128,256,512}`; dropout ∈ {0,0.1,0.2,0.3}; `lr ∼ 10^{U(-4,-2)}`; `wd ∼ 10^{U(-6,-2)}` |
| MLP train | batch 64, max 80 epochs, early stop patience 10 |
| Final fit | Refit on train+val after selection |
| Metrics | Accuracy, F1, AUROC |

---

## Test results (`top1`, n=1500)

| Model | Probe | Acc | F1 | AUROC |
|-------|-------|-----|-----|-------|
| Base | Logistic | 0.905 | 0.877 | **0.967** |
| Base | MLP | 0.893 | 0.866 | 0.963 |
| Instruct | Logistic | 0.867 | 0.818 | 0.951 |
| Instruct | MLP | 0.865 | 0.827 | 0.952 |

Probes discriminate LM correctness far above the LM’s own top-1 rate (~36–38%).

---

## Entropy-neuron feature ablation

Freeze the trained probe. Intervene on **stored activations only** (not a causal LLM ablation). Entropy dims = Phase-1 top-ρ neurons (**k=12**, writer layer 15). **20** matched random controls.

- **Necessity:** replace entropy dims with train-set mean (primary) or zero.
- **Sufficiency:** keep only entropy dims; replace all others.
- Report **ΔAUROC = baseline − ablated**; percentile vs controls (sufficiency: higher ⇒ smaller drop than random ⇒ more relatively sufficient).

### ΔAUROC summary

| Model / probe | Nec. mean | Nec. zero | Suff. mean (pct) | Suff. zero (pct) |
|---------------|-----------|-----------|------------------|------------------|
| Base logistic | ≈0 | ≈0 | 0.177 (**100**) | 0.177 (95) |
| Base MLP | ≈0 | ≈0 | 0.135 (**95**) | 0.194 (70) |
| Instruct logistic | ≈0 | ≈0 | 0.261 (**85**) | 0.261 (90) |
| Instruct MLP | ≈0 | ≈0 | 0.282 (**75**) | 0.307 (55) |

**Takeaway:** entropy dims are **not necessary** for probe performance (Δ≈0). Keeping only them still leaves a usable but much weaker signal, and that drop is **smaller than for most random 12-dim sets** (high sufficiency percentiles)—so they are relatively informative, but the rest of the representation matters more.

> **Correction (feature-subset probes):** intervening on a frozen full-dim probe is not a fair sufficiency test — that probe was trained to use the full activation vector. Train **new** probes with `--features entropy` (Phase-1 dims only) and `--features rest` (all other dims), then compare held-out metrics to the full probe. Outputs land in `logistic_entropy/`, `logistic_rest/`, `mlp_entropy/`, `mlp_rest/`.
