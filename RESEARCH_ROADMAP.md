# Research Roadmap: Confidence Regulation Neurons

**Course:** Trends in AI and LLMs (HPI seminar)  
**Base work:** Stolfo et al., NeurIPS 2024 — [arXiv:2406.16254](https://arxiv.org/abs/2406.16254) · [GitHub](https://github.com/bpwu1/confidence-regulation-neurons)  
**Local reference:** `reference_code/` · Paper: `base_paper.pdf`

---

## Project Goals

1. **Understand** the mechanism: entropy neurons, null-space writes, LayerNorm/RMSNorm mediation, induction hedging.
2. **Reproduce** core GPT-2 results step by step (not a blind re-run — each step should be explainable).
3. **Extend** the same analyses to modern small/base/instruct models (hardware-permitting).
4. **Achieve light novelty** for the seminar grade; keep a clear path toward publishable follow-up.
5. **Iterate** — update this document as runs succeed/fail and as novelty direction is chosen.

---



## Constraints & Resources


| Resource                          | Notes                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------ |
| **GPU (primary)**                 | RTX 4070 (~12 GB VRAM) — use `batch_size=1`, `chunk_size=20`, seq_len ≤ 256    |
| **GPU (optional)**                | H40/H100 — justify with a concrete batch (e.g. 7B base vs instruct comparison) |
| **Team**                          | Solo                                                                           |
| **Code policy**                   | Small speed optimizations only; do not rewrite architecture                    |
| **Future (out of scope for now)** | Semantic/epistemic uncertainty beyond token-level entropy                      |


---



## How to Use This Document

Tasks are grouped by **work/study load**, not calendar weeks. Tiers build on each other:

```
Study → Core Repro (GPT-2) → Extension (modern models) → Light Novelty → Future Depth
```

Mark each task: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` skipped/deferred

**Decision gates** (choose when you reach them, not upfront):

- **Gate A:** Which modern model to extend first (after GPT-2 core repro).
- **Gate B:** Light novelty direction (after extension teaser results exist).
- **Gate C:** Whether to request H40/H100 (only if Gate B needs ≥7B or large sweeps).

---



## Model Registry

Start with GPT-2 Small. Add models incrementally; verify TransformerLens + HuggingFace support before committing.


| Priority | Model          | Role                                  | VRAM (approx.) | TransformerLens check |
| -------- | -------------- | ------------------------------------- | -------------- | --------------------- |
| 0        | `gpt2-small`   | Full reproduction anchor              | Low            | Supported             |
| 1        | `pythia-410m`  | Token frequency neurons (§4)          | Low            | Supported             |
| 2        | *TBD @ Gate A* | First modern extension                | Medium         | Verify before use     |
| 3+       | *Add later*    | Additional universality / comparisons | Varies         | Verify before use     |


**Candidate pool (add/remove as needed):**

- Qwen 3.5 base + instruction-tuned (same family, different checkpoint)
- Llama 3.2 (1B / 3B — prefer smaller for 4070)
- Mistral (7B may need H100 or aggressive memory settings)
- Others: Gemma-2, SmolLM, Phi-3-mini, etc.

**Before adding a model**, run a 5-minute smoke test:

```text
1. load_model_from_tl_name(model_name, device='cuda')
2. Forward pass on 1 batch, seq_len=64
3. Read model.cfg: n_layers, d_mlp, normalization_type, d_vocab
4. Note udark_start (bottom-k singular vectors for ρ) — paper uses k≈1% of d_model
```

Record working names and `udark_start` values in the **Experiment Log** section at the bottom.

---



## Track 1 — Entropy Neurons (Primary)



### 1A · Study · *Light–moderate*

- [x] Read paper §3.1–3.3 (identification, LN mediation, null space)
- [x] Read paper Fig 1–2 captions and Appendix C (LN folding)
- [x] Read `reference_code/plot_generation/fig1.py` (Norm vs LogitVar)
- [x] Read `get_potential_entropy_neurons_udark()` in `reference_code/utils.py`
- [x] Read `reference_code/ablations/run_and_store_ablation_results.py` (TE vs DE_LN)

**Done when:** You can explain LogitVar, ρ, TE, DE_LN, and why direct logit attribution misses the effect.

---



### 1B · Core Repro (GPT-2) · *Moderate–heavy*


| Step          | Paper ref     | Script / function                                       | Output artifact                                      |
| ------------- | ------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| B1 Static ID  | Fig 2a        | `plot_generation/fig1.py`                               | Scatter: `‖w_out‖` vs LogitVar; mark entropy neurons |
| B2 Null space | Fig 2d–e      | `get_potential_entropy_neurons_udark()` + custom ρ plot | ρ vs `1 − DE_LN/TE` or LN-mediated fraction          |
| B3 Ablation   | Fig 2c, Fig 1 | `ablations/run_and_store_ablation_results.py`           | TE vs DE for entropy vs random neurons               |
| B4 Confidence | Fig 1         | From ablation feather + `load_results.py`               |                                                      |


**GPT-2 entropy neurons (paper):** `11.584`, `11.2378`, `11.2870`, `11.2123`, `11.1611`, `11.2910`  
**GPT-2 udark_start:** `-12` (bottom 12 singular vectors)

**Run notes (4070-friendly):**

```bash
cd reference_code/ablations
# Edit conf/config_ln_ablations.yaml: model=gpt2-small, k=10, batch_size=1
python run_and_store_ablation_results.py
```

- Use `neuron_range` in config to sweep subsets (e.g. `0-512`) if full MLP is slow.
- Store results under `ablations/results/`; visualize with `load_results.py`.

**Done when:** Your plots qualitatively match paper claims (entropy neurons = high ΔH, low argmax change; TE ≫ DE_LN for entropy neurons, not for random).

---



### 1C · Extension (Modern Model) · *Moderate*

Repeat **B1–B4** on the model chosen at **Gate A**.

Adjust per model:

- `udark_start`: use `k ≈ 0.01 × d_model` (paper: 40 for Llama-7B, 12 for GPT-2)
- `normalization_type`: RMSNorm models need frozen-scale branch in ablation code (already handled for RMSPre in `run_and_store_ablation_results.py`)
- Vocab padding: check `d_vocab` vs tokenizer size (Pythia-style padding issues)

**Done when:** Side-by-side table GPT-2 vs modern: top entropy candidates, mean ρ, mean LN-mediated effect, induction relevance (Track 2).

---



## Track 2 — Induction Case Study (High Presentation Value)



### 2A · Study · *Light–moderate*

- [ ] Read paper §5–6, Appendix H
- [ ] Skim Olsson et al. induction heads (prefix-matching score — concept only)
- [ ] Read `reference_code/case_studies/induction.py`
- [ ] Read `reference_code/case_studies/induction_with_bos_ablation.py`

**Done when:** You can explain duplicated-sequence setup and why entropy should rise on the second occurrence.

---



### 2B · Core Repro (GPT-2) · *Moderate*


| Step            | Paper ref | Script                                 | Output                                                   |
| --------------- | --------- | -------------------------------------- | -------------------------------------------------------- |
| B1 Activations  | Fig 5a    | `case_studies/induction.py`            | Avg entropy-neuron activation, entropy, loss vs position |
| B2 Ablation     | Fig 5b    | Same script (clip/mean ablate section) | Post-ablation entropy ratio on 2nd half                  |
| B3 BOS ablation | Fig 5c    | `induction_with_bos_ablation.py`       | Induction head ablation → entropy neuron activation drop |


**GPT-2 induction heads (paper):** L5H1, L5H5, L6H9

**Done when:** Entropy increases on 2nd sequence half; ablating 11.2378 (or top neuron) sharply lowers entropy; BOS ablation of ≥1 induction head reduces entropy-neuron activation.

---



### 2C · Extension (Modern Model) · *Moderate*

- Repeat 2B on Gate A model.
- Re-detect induction heads via `generate_induction_df()` (threshold ~0.7) — layers will differ from GPT-2.
- Compare hedging magnitude and dominant entropy neuron(s).

**Optional ( heavier ):**

- [ ] `case_studies/natural_induction_true.py` — real C4 induction (Appendix H.1)

**Done when:** One slide: “Induction hedging in [modern model] vs GPT-2.”

---



## Track 3 — Token Frequency Neurons (Optional / Secondary)



### 3A · Study · *Moderate*

- [ ] Read paper §4, Appendix G
- [ ] Read `run_and_store_unigram_results.py`, unigram helpers in `utils.py`



### 3B · Core Repro · *Heavy*

Best on **Pythia-410M** (paper default; unigram `.npy` in repo).


| Step         | Paper ref | Script                              |
| ------------ | --------- | ----------------------------------- |
| Unigram + KL | Fig 3a    | `generate_unigram_df` / pile counts |
| Freq neurons | Fig 3b–c  | `run_and_store_unigram_results.py`  |


**Defer if:** Tracks 1–2 are not solid yet. Seminar can succeed without this track.

---



## Track 4 — Gate B Novelty: Hallucination Forecasting from Entropy Neurons

**Gate B decision:** Hallucination forecasting on factual QA using entropy-neuron behavior  
Models: `gpt2-small`, `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-1B-Instruct`  
Data: **LAMA T-REx** + **TriviaQA**  
Plan: **Phase 3a (observational)** → **Phase 3b (QA ablation, separate)** → **Phase 3c (light probe, optional)**


---

### 4A · Phase 3a — Observational factual-QA study (run first)

Run observational analyses on correct vs wrong generations and **per-token** activation/entropy traces.

**Models / prompts**
- GPT-2: simple fill-in / cloze prompts
- Llama base: fill-in prompts
- Llama Instruct: chat/instruction prompts (keep format consistent)

**Datasets**
- LAMA: **T-REx** subset first (templated cloze facts)
- TriviaQA: free-form QA with official aliases + normalization

**Decoding**
- both `greedy` and `sampling`
- token limits: LAMA ≤ 32 generated new tokens; TriviaQA ≤ 64–128 generated new tokens

**Correct vs wrong labels**
- T-REx (cloze): evaluate at the masked answer span with standard LAMA-style ranking (`@1`, `@5`, MRR)
- TriviaQA (open): normalized EM + token-level F1

**Hallucination bins (H1–H4; inspect all H)**
- H1 wrong (gold/alias not supported)
- H2 wrong and confident (low entropy / high top-1 prob at answer onset)
- H3 wrong with a specific false entity (predicted entity differs from gold aliases)
- H4 early warning: compare token-position features at `t` vs whether the wrong span starts within the next `k` tokens (default `k=3`)

**Features to log**
- full-sequence aggregates and per-token traces: entropy, entropy-neuron activations, and confidence proxies (top-1 prob/logit; optionally rank)

**Sample size targets (start larger if runtime allows)**
- T-REx: ~`3,000–5,000` probes per model
- TriviaQA: ~`2,000` questions per model

---

### 4B · Phase 3b — QA ablation (separate; only after Phase 3a shows a signal)

Mean-ablate Phase-1 entropy-neurons on QA prompts and measure how hallucination bins change (especially H2/H4).

**Runtime guardrail:** keep ablation sample smaller than Phase 3a (e.g. `50–200` QA items).

**Llama prerequisite:** validate the frozen RMSNorm ablation path used for `DE_LN` before trusting any Llama QA-ablation causal claims.

---

### 4C · Phase 3c — Lightweight hallucination detector (optional but recommended)

Train a small classifier/probe using Phase 3a features to predict hallucination risk / early warning.

Default: interpretable probe on entropy + top entropy-neuron activations + confidence (no heavy LoRA yet). The detector can trigger abstention/safer phrasing at inference time.

---



## Track 5 — Future Depth (Post-Seminar / Publish Path)

Not required for course grade. Pursue if Gates A–B go well.

- [ ] **Comparative:** Full base vs instruct study across Qwen 3.5, Llama 3.2, Mistral
- [ ] **Applied:** Systematic confidence steering + benchmark (calibration metrics, ECE)
- [ ] **Mechanistic:** SAE on final MLP — do sparse features split entropy vs frequency roles?
- [ ] **Architectural:** MoE experts, larger scale (H100)
- [ ] **Semantic uncertainty:** QA / CoT / refusal — separate project phase

---



## Performance Tips (No Architecture Rewrite)

- `torch.set_grad_enabled(False)` everywhere (already in scripts)
- `batch_size=1`, `chunk_size=20` in ablation loops
- Cache `entropy_df` once; reuse for ablation passes
- Sweep neurons in `neuron_range` chunks; merge feather files
- Use `dry_run: true` in Hydra config to validate pipeline on 20 neurons
- Clear CUDA cache between heavy sequences: `torch.cuda.empty_cache()`

---



## Seminar Deliverables Checklist

Minimum viable presentation:

- [ ] Mechanism diagram (null space → norm → LN → entropy) in your own words
- [ ] GPT-2: Norm vs LogitVar + one TE/DE plot
- [ ] GPT-2: Induction Fig 5a–b (or 5c)
- [ ] ≥1 modern model: same analysis or explicit failure analysis
- [ ] Gate B novelty: Phase 3a observational QA (LAMA T-REx + TriviaQA) showing entropy + entropy-neuron differences across H1–H4 (early-warning focus)
- [~] (Optional) Phase 3c detector results: probe accuracy / early-warning forecasting
- [~] (Optional) Phase 3b ablation results after Phase 3a signal (for causal interpretation)
- [ ] Future work slide pointing to Track 5

---



## Experiment Log

*(Update as you go — model names, hyperparams, blockers)*

```text
Date       | Model              | Task              | Result / notes
-----------|--------------------|-------------------|------------------------------
           | gpt2-small         |                   |
           |                    |                   |
```

---



## References

- Paper: `base_paper.pdf`
- Code: `reference_code/` (upstream: [https://github.com/bpwu1/confidence-regulation-neurons](https://github.com/bpwu1/confidence-regulation-neurons))
- Key deps: `transformer-lens`, `transformers`, Hydra — see `reference_code/requirements.txt` (consider bumping versions only if load breaks)

---

*Last updated: project scoping session. Revise Gate A/B decisions and Experiment Log as work proceeds.*