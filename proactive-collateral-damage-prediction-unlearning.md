# Proactive Collateral Damage Prediction for Machine Unlearning in LLMs

**Authors:** Bo Su, Yueru Yan, Thai Le (Indiana University, Bloomington)  
**Status:** Draft (ACL submission format)  
**Working Title Variants:** "Protect Knowledge from Unlearning" / "Unlearning-induced Collateral Corruption in Machine Unlearning for LLMs"

---

## TL;DR

Current machine unlearning in LLMs is reactive: collateral corruption to unrelated knowledge is only detected *after* unlearning occurs. This work demonstrates that collateral corruption is **structured and geometrically predictable**, enabling a proactive pre-unlearning framework that identifies at-risk knowledge before any weights are modified.

---

## 1. Core Problem & Motivation

Machine unlearning in LLMs (forgetting specific training data for GDPR compliance, IP protection, safety) suffers from a fundamental but underappreciated failure mode: **collateral corruption**. Because transformers encode knowledge in overlapping, distributed representation subspaces (superposition hypothesis), removing one target concept inevitably perturbs shared subspaces that support *other*, unrelated capabilities.

The existing literature treats this as a post-hoc diagnostic problem — collateral damage is measured *after* unlearning via benchmarks (TOFU, WMDP, MUSE). This is analogous to "Monday-morning quarterbacking": issues are recognized only after the damage has occurred, perpetuating an expensive **unlearn → patch → unlearn** cycle.

**The central question this paper asks:** Can we *predict* which non-target knowledge will be collaterally corrupted, using only pre-unlearning model weights and representations?

---

## 2. What the Preliminary Experiments Demonstrate

The paper presents three interlocking empirical observations. Together they build the case that collateral corruption is not random noise but a **structured, geometrically identifiable phenomenon** that is detectable before unlearning.

### 2.1 Experiment 1 — Task Vector Orthogonality

**Setup:** Following the task-arithmetic formalism (Ilharco et al., 2023), an *unlearning vector* is defined as:

> `v_unlearn = θ_after_unlearning − θ_original`

Unlearning vectors from multiple algorithms (GradDiff, RMU, DPO) are computed on the same forget set (TOFU forget10) and their pairwise cosine similarities are measured.

**Finding:** Unlearning vectors from different algorithms on *identical forget data* are **nearly orthogonal** to each other (cosine similarity ≈ 0).

**What this demonstrates:**
- The geometric structure of an unlearning vector primarily encodes the *algorithm's intrinsic mechanism*, **not** the content of the forgotten data.
- Two algorithms that forget the same knowledge via different mechanisms produce orthogonal perturbations in parameter space.
- **Critical implication:** Each algorithm damages a *different, algorithm-specific neighborhood* of unrelated knowledge. There is no single "collateral damage profile" for a given forget set — the damage depends jointly on the forget content *and* the algorithm used.

**Why this matters for the paper's thesis:** This motivates *algorithm-adaptive* prediction. A proactive framework cannot use a generic corruption estimate; it must account for the specific unlearning algorithm's parameter-space behavior. It also explains why existing reactive evaluation (e.g., global perplexity on Wikitext) consistently underestimates damage — the evaluation set is not adapted to the algorithm's specific failure subspace.

---

### 2.2 Experiment 2 — Representation-Level Localization (PCA / CKA / Fisher)

**Setup:** Using a diagnostic toolkit based on Xu et al. (2025), RMU is applied to Llama-3.2-1B-Instruct on TOFU forget10. Four metrics are analyzed layer-by-layer across forget, retain, real-authors, and world-facts evaluation sets:

| Metric | What it measures |
|--------|-----------------|
| **PCA Similarity** | Alignment of dominant activation directions before vs. after unlearning |
| **PCA Shift** | Translational drift of activation center along PC1/PC2 |
| **CKA** | Subspace similarity via centered kernel alignment |
| **Fisher Information** | Parameter sensitivity / loss-landscape curvature |

**Findings:**

| Metric | Forget set | Retain / Unrelated sets |
|--------|-----------|------------------------|
| PCA Similarity | Drops noticeably only in **final transformer blocks** | Near 1.0 across all layers |
| PCA Shift | Moderate drift in **upper layers**; not propagated throughout the network | Minimal displacement |
| CKA | Partial decline in **final layer only** | Near 1.0 throughout |
| Fisher Information | **Redistributed** (reorganized) at Layer 13; no global flattening | — |

**Overall assessment:** RMU on TOFU forget10 induces **Reversible, Non-Catastrophic Forgetting** — targeted and localized representational modifications in upper layers, with global subspace structure and parameter sensitivity substantially preserved.

**What this demonstrates:**
- Collateral corruption is **geometrically confined**: it does not spread uniformly across all layers or all knowledge but is concentrated in the upper-layer subspaces associated with the forget set.
- The corruption is **low-dimensional and structured** (confined to a few principal directions in upper layers).
- This directly supports the claim that corruption is not random — it follows predictable geometric patterns rooted in how the forget set occupies the representation space.
- **Critical implication (RQ1b):** If corruption is localized to a low-dimensional subspace, then the "corruption frontier" can be geometrically characterized *before* unlearning by analyzing which non-target samples project substantially onto the forget-associated subspace.

---

### 2.3 Experiment 3 — Semantic Filtering and Side Effect Index

**Setup:** A semantic anchor `V_forget` is extracted from the forget set (e.g., mean embedding over TOFU forget10 samples). Wikitext-2-raw-v1 is then searched for samples with high cosine similarity to `V_forget`, forming a **vulnerable subset**. The Side Effect Index (SEI) is computed as:

> `SEI = Avg_PPL_unlearn / Avg_PPL_base`

SEI is computed over (a) the full Wikitext corpus and (b) the semantically filtered vulnerable subset.

**Finding:** The globally-averaged SEI on the full Wikitext corpus is close to 1.0 (minimal apparent degradation), but the SEI computed on the **semantic neighborhood** of the forget set reveals a **significant and detectable increase** in perplexity — a signal that is completely hidden in global averages.

**What this demonstrates:**
- Standard evaluation metrics (global perplexity on Wikitext-103) systematically **dilute** collateral damage because the damage is concentrated in a small, semantically proximate subset of the corpus.
- Collateral damage is **semantically localized**: samples near the forget set in embedding space are disproportionately affected.
- **Critical implication (RQ1a):** Semantic proximity to the forget set is a measurable, pre-unlearning predictor of corruption probability. By identifying the semantic neighborhood *before* unlearning, we can predict which external knowledge is at risk.

---

### 2.4 Summary: What the Three Experiments Jointly Support

The three experiments together establish three complementary claims:

| Claim | Supporting Experiment | Key Evidence |
|-------|-----------------------|-------------|
| **C1 — Algorithm specificity:** Collateral damage patterns are algorithm-dependent and structurally distinct | Task vector orthogonality | Near-orthogonal unlearning vectors across algorithms on identical forget data |
| **C2 — Geometric localization:** Corruption is confined to a low-dimensional subspace in upper layers | Representation-level analysis | PCA/CKA/Fisher localization to final transformer blocks |
| **C3 — Semantic predictability:** Semantic proximity to forget set is the primary predictor of vulnerability | Semantic filtering | SEI amplification in semantic neighborhood vs. global dilution |

These three claims together make the case that the **proactive prediction of collateral corruption is both necessary** (existing evaluation misses it) **and feasible** (corruption is structured enough to be predicted geometrically).

---

## 3. Research Questions

**RQ1a (Predictability):** Can feature-space overlap between a non-target sample and the forget-associated subspace serve as a pre-unlearning predictor of that sample's corruption probability under the unlearned model?

**RQ1b (Localization):** Is the representation drift induced by unlearning confined to a structured, low-dimensional subspace — specifically the one spanned by the forget-associated principal directions?

**Unified answer from preliminary results:** Yes to both. Corruption is low-dimensional (RQ1b) and semantically structured (RQ1a), making proactive prediction principled rather than heuristic.

---

## 4. Problem Formulation

Let `M` denote the target LLM, `D_forget ⊂ D` the data to be unlearned, and `M'` the resulting unlearned model. Prior to applying any unlearning algorithm, the goal is to construct a **vulnerable set** `D*` such that:

**Formulation I — Accuracy degradation coverage (Eq. 1–5):**

> `max_{D*} Acc(M', D*)`  
> subject to:  
> - `Acc(M, D*) ≥ α` (M still answers correctly — these are benign, retained-knowledge questions)  
> - `sim(D*, D_forget) ≤ δ` (D* is semantically distant from the forget set — truly collateral)  
> - `PPL_M(D*) ≤ τ` (linguistically valid, real-world knowledge)  
> - `|D*| ≥ k` (comprehensive coverage)  
> - `M'` is *not* accessed during construction; used only for evaluation

**Formulation II — Vulnerable set size maximization (Eq. 6–9):**

> `max_{D*} |D*|`  
> subject to the same irrelevance, linguistic quality, and utility constraints

**Key design principle:** `D*` must be semantically irrelevant to `D_forget` (truly non-target knowledge), yet must be knowledge that `M` handles correctly but `M'` degrades on. This makes `D*` a principled, algorithm-specific stress test for collateral corruption.

---

## 5. Proposed Framework

### 5.1 Overview

The framework operates *before* any unlearning algorithm is applied and consists of three stages:

```
D_forget  ──▶  [1] Semantic Anchor Construction  ──▶  V_forget
                        │
                        ▼
Large Corpus  ──▶  [2] Vulnerable Subset Identification  ──▶  D_vulnerable
                        │
                        ▼
                   [3] Side Effect Quantification  ──▶  SEI per sample / SEI(D*)
                        │
                        ▼
               [4] Corruption Prediction  ──▶  P(corrupt | x, D_forget, Algorithm)
```

### 5.2 Semantic Anchor Construction

Extract a semantic anchor vector `V_forget` from the forget set by computing the mean hidden-state representation (e.g., at a target layer) over all samples in `D_forget`. This vector represents the "center of gravity" of the target knowledge in representation space.

### 5.3 Vulnerable Subset Identification

Search a large external corpus (e.g., Wikitext-2-raw-v1) for samples whose representation is semantically proximate to `V_forget` via cosine similarity. Samples exceeding a threshold `δ` form the vulnerable subset candidate pool. These samples are semantically *different* from the forget content (they satisfy the irrelevance constraint) but share representational proximity that makes them at risk.

### 5.4 Side Effect Index (SEI)

To quantify collateral damage on any subset `S`:

> `SEI(S) = Avg_PPL_{M'}(S) / Avg_PPL_M(S)`

- `SEI ≈ 1.0` → minimal collateral impact on `S`  
- `SEI >> 1.0` → significant degradation in model fluency / knowledge on `S`

This metric requires access to `M'` only for evaluation; prediction happens prior to unlearning.

### 5.5 Corruption Prediction via Feature-Space Overlap

For each candidate sample `x`, compute its projection onto the forget-associated principal subspace:

> `overlap(x) = ||P_{forget} · h(x)||² / ||h(x)||²`

where `P_{forget}` is the projection matrix onto the top-k principal directions of forget-set activations, and `h(x)` is x's hidden state. The hypothesis (RQ1a) is that `overlap(x)` is positively correlated with the post-unlearning accuracy degradation on `x`.

---

## 6. Revised Paper Outline

### 1 Introduction
- Motivation: GDPR, privacy, IP protection → LLM unlearning is necessary
- Problem: Non-local nature of transformer knowledge → unlearning causes collateral corruption
- Gap: Existing approaches are entirely reactive; no principled pre-unlearning protection exists
- Key insight: Collateral corruption is *structured and geometrically predictable*
- Contribution: A proactive pre-unlearning framework for collateral damage prediction and protection
- Significance: Breaks the "unlearn–patch–unlearn" cycle; enables constraint-aware optimization

### 2 Related Work
- 2.1 Machine Unlearning Benchmarks (TOFU, WMDP, MUSE)
- 2.2 Knowledge Entanglement and Superposition in Transformers
- 2.3 Reactive Post-Unlearning Analysis (Ko et al. 2025 — knowledge holes; Shah & Le 2025 — SKEB)
- 2.4 Parameter-Space Geometry: Task Arithmetic (Ilharco et al. 2023), LoRA (Hu et al. 2022)
- 2.5 Representation-Level Diagnostics for Unlearning (Xu et al. 2025)

### 3 Motivating Analysis: Is Collateral Corruption Structured?
- 3.1 Task Vector Orthogonality Analysis
  - Setup and results (Figure 1: cosine similarity heatmap across algorithms)
  - Interpretation: Algorithm mechanism ≠ forget data content in parameter space
- 3.2 Representation-Level Localization
  - PCA Similarity / Shift, CKA, Fisher Information (Figures 2–5)
  - Interpretation: Corruption confined to upper-layer, low-dimensional subspace
- 3.3 Semantic Filtering Reveals Hidden Damage
  - SEI on global corpus vs. semantic neighborhood (Figure 6)
  - Interpretation: Standard evaluation dilutes localized damage
- 3.4 Synthesis: Three Structural Properties of Collateral Corruption
  - C1 (Algorithm-specificity), C2 (Geometric localization), C3 (Semantic predictability)

### 4 Problem Formulation
- 4.1 Formal setup: `M`, `D_forget`, `M'`
- 4.2 Goal: Construct pre-unlearning vulnerable set `D*`
- 4.3 Formulation I: Maximize `Acc(M', D*)` (Eqs. 1–5)
- 4.4 Formulation II: Maximize `|D*|` (Eqs. 6–9)
- 4.5 Research Questions: RQ1a (predictability) and RQ1b (localization)

### 5 Proposed Framework
- 5.1 Overview pipeline
- 5.2 Semantic Anchor Construction (`V_forget`)
- 5.3 Vulnerable Subset Identification (semantic similarity search over corpus)
- 5.4 Side Effect Index (SEI = PPL ratio)
- 5.5 Feature-Space Overlap as Corruption Predictor (`overlap(x)`)
- 5.6 *(Future/Optional)* Data Synthesis for Pre-Unlearning Protection

### 6 Experimental Setup
- 6.1 Models: Llama-3.2-1B-Instruct
- 6.2 Unlearning methods: GradDiff (40 hyperparameter variants), RMU, DPO
- 6.3 Datasets: TOFU forget10 (forget), Wikitext-2-raw-v1 (retain/evaluation)
- 6.4 Evaluation metrics: SEI, Model Utility, Forget Quality, MIA Accuracy
- 6.5 Pareto-optimal model selection from 40 GradDiff variants

### 7 Results and Analysis
- 7.1 RQ1b: Evidence for low-dimensional corruption localization (representation analysis)
- 7.2 RQ1a: Feature-space overlap as corruption probability predictor
- 7.3 Algorithm specificity: Different algorithms → different collateral footprints
- 7.4 Ablation: Semantic similarity threshold `δ`, anchor construction method, subspace rank `k`

### 8 Discussion
- 8.1 From reactive monitoring to proactive constraint: What structured corruption enables
- 8.2 Implications for algorithm-adaptive protection
- 8.3 Continual unlearning: Compounding corruption prediction over sequential deletions
- 8.4 Limitations: One-shot unlearning assumption; single architecture; TOFU is fictitious data

### 9 Conclusion

---

## 7. Key Connections to Related Work (Research Gap)

| Prior work | What it establishes | What this paper adds |
|-----------|--------------------|--------------------|
| Ko et al. 2025 (knowledge holes) | RL-based reactive discovery of failure prompts after unlearning | Proactive prediction *before* unlearning; systematic geometric coverage vs. RL sampling in failure space |
| Shah & Le 2025 (SKEB) | Prompt structural properties predict leakage risk at inference time | Model-weight-level subspace analysis predicts corruption risk at *training/unlearning* time |
| Ilharco et al. 2023 (Task Arithmetic) | Task vectors encode semantic similarity of tasks | Unlearning vectors encode *algorithm mechanism*, not data content; orthogonality reveals algorithm-specificity of damage |
| Xu et al. 2025 (Reversibility analysis) | Taxonomy of reversible vs. irreversible forgetting | Applied diagnostically to *predict* which knowledge is at risk, not just classify post-hoc forgetting regimes |
| Hu et al. 2022 (LoRA) | Meaningful adaptation lives in a low-rank subspace | Collateral corruption also lives in a low-rank subspace → same subspace structure that enables efficient adaptation also constrains where damage propagates |

---

## 8. Open Questions and Future Directions

1. **Prediction accuracy:** How well does `overlap(x)` actually predict corruption probability across different algorithms and architectures? Is a linear predictor sufficient or is nonlinear modeling needed?

2. **Algorithm-adaptive anchor construction:** Should `V_forget` be constructed differently for gradient-based methods (GradDiff) vs. representation-space methods (RMU)? The task vector orthogonality result suggests the answer is yes.

3. **Continual unlearning:** In sequential deletion settings, can cumulative subspace stress be monitored via SEI to detect impending capability collapse before it occurs?

4. **From prediction to protection:** Once `D*` is identified, how should the unlearning objective be modified? Options include: (a) adding samples from `D*` as hard retain constraints, (b) projecting updates away from `D*`-associated subspace directions, (c) data augmentation of the retain set with `D*`.

5. **Generalization beyond TOFU:** TOFU uses fictitious authors (low real-world entanglement). Does the semantic filtering approach scale to real-world unlearning targets where entanglement is higher?

---

## 9. Connections to Existing Notes in This Repo

- **`papers/pre-unlearning-data-augmentation-shortlist.md`** — ReLearn, Align-then-Unlearn, and Geometric Disentanglement Unlearning all provide complementary methods that this framework could leverage for the "data synthesis for pre-unlearning protection" component (Section 5.6).
- **`papers/llm-unlearning-research-gap-table.md`** — Gap G3 ("Missing bridge from latent geometry to data operations") is precisely what this paper's framework addresses: using subspace diagnostics to decide concrete data operations *before* unlearning.

---

*Last updated: 2026-03-12*
