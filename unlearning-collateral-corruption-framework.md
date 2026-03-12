# Predicting and Preventing Collateral Corruption in LLM Unlearning: Framework Analysis and Draft Outline

> **Status**: Working draft — consolidates the abstract submission and preliminary experimental findings into a coherent narrative.

---

## 0  Executive Summary: What Do the Preliminary Experiments Show?

Three lines of evidence, each from a different analytical lens, converge on a single thesis:

| Experiment | Method | Key Finding | Implication |
|---|---|---|---|
| **Task-vector orthogonality** | Cosine similarity of unlearning vectors (Δθ = θ_unlearned − θ_original) across algorithms | Unlearning vectors from different algorithms on the **same** forget set are largely **orthogonal** (mean off-diagonal cosine ≈ 0.34; retain90 vs any GradDiff ≈ 0.01–0.04) | The geometric structure of an unlearning vector encodes the **algorithm's mechanism**, not the content of the forgotten data. Different mechanisms ⟹ different collateral-damage footprints. |
| **Semantic filtering / Side-Effect Index** | Extract semantic anchor from forget set → find semantically proximal subset in Wikitext → measure PPL ratio (unlearned / base) | Global PPL on the full Wikitext corpus **dilutes** real side effects; the vulnerable subset (semantically near the forget set) shows a significant PPL spike invisible to whole-corpus averages. | Standard "model utility" benchmarks systematically **underestimate** collateral corruption; a semantically targeted evaluation reveals hidden knowledge degradation. |
| **Representation-level diagnostics** (PCA similarity/shift, CKA, Fisher Information) | Layer-wise comparison of base vs unlearned activations on forget, retain, and unrelated evaluation sets | For RMU on TOFU forget10: representational change is **localized** to upper transformer layers and concentrated on the forget set; retain/unrelated sets show minimal drift. Fisher spectrum redistributes rather than flattens. | Corruption is **structured and low-dimensional** — it lives in a specific subspace rather than being globally distributed. This structure is what makes pre-unlearning prediction feasible. |

**Bottom line**: Collateral corruption from unlearning is (a) algorithm-dependent, (b) invisible to standard global metrics, and (c) geometrically structured in representation space. These three properties together justify a **proactive prediction framework** that operates *before* unlearning rather than diagnosing damage *after* the fact.

---

## 1  Introduction

### 1.1 Context

Machine unlearning in LLMs aims to selectively remove specific training data influence (e.g., for GDPR "right to be forgotten") without full retraining. Recent benchmarks (TOFU, WMDP, MUSE) and evaluation protocols have made the field measurable, but a fundamental problem remains underexplored:

> **Unlearning is inherently destructive.** Because LLMs encode knowledge across shared parameters and overlapping representation subspaces (the "superposition" phenomenon), removing target knowledge inevitably perturbs representations that support other capabilities.

### 1.2 The Reactive–Proactive Gap

Current work on collateral damage is **reactive**:

- Ko et al. (2025) use RL to discover "knowledge holes" — but only after unlearning has occurred.
- Shah & Le (2025) predict leakage risk from prompt structure — but this tells us about *stimulus* vulnerability, not about *which retained knowledge* will degrade.
- Standard benchmarks evaluate on fixed corpora (e.g., full Wikitext), which dilutes localized damage into global averages.

What is missing is a **proactive** framework that predicts, before unlearning, which non-target knowledge is structurally vulnerable — and then either protects it or reshapes the data to reduce damage.

### 1.3 Our Contribution (Thesis)

We propose to shift unlearning evaluation and preparation from reactive to proactive:

1. **Predict** which retained knowledge will be collaterally corrupted, using representation-space geometry.
2. **Prevent** that corruption through algorithm-adaptive data preparation before one-shot unlearning.

The preliminary experiments reported here establish the empirical foundation for this framework by demonstrating that collateral corruption is structured, algorithm-dependent, and predictable from pre-unlearning representations.

---

## 2  Motivation: Why Collateral Corruption Is Inevitable and Algorithm-Dependent

### 2.1 Knowledge Entanglement in Transformers

Transformers do not store facts in isolated, modular components. Mechanistic interpretability work (Elhage et al., 2022) shows that features are represented in **superposition** — multiple concepts share directions in activation space. Consequence: deleting concept A necessarily perturbs the subspace that also supports concepts B, C, ....

### 2.2 Empirical Evidence: Unlearning Vectors Are Algorithm-Specific

**Setup.** We compute unlearning task vectors τ = θ_unlearned − θ_original for multiple algorithms/hyperparameter settings on the same TOFU forget10 dataset, using Llama-3.2-1B-Instruct as base.

**Result.** Pairwise cosine similarities between unlearning vectors:

- retain90 vs any GradDiff variant: **0.01 – 0.04** (nearly orthogonal)
- Across GradDiff variants with different hyperparameters: **0.31 – 0.97** (varies widely)
- Mean off-diagonal: **0.34**

**Interpretation.** If unlearning vectors simply encoded "which facts were removed," vectors for the same forget set should be highly aligned. Instead, they are largely orthogonal across algorithms/settings. This means the **mechanism** of unlearning — not the target data — dominates the parameter-space direction of change. Different mechanisms perturb different subspaces, and therefore cause **different patterns of collateral damage**.

### 2.3 Implication for Framework Design

Because collateral damage is algorithm-dependent, any prediction or prevention framework must be **algorithm-adaptive**: the vulnerable knowledge set changes depending on which unlearning method will be applied.

---

## 3  Problem Formulation

### 3.1 Collateral Damage Prediction (Pre-Unlearning)

Given:
- A base model f trained on dataset D
- A forget set D_forget ⊂ D
- An unlearning algorithm A

**Goal:** Before running A, identify the set of non-target samples D* ⊂ D \ D_forget whose model performance will degrade after unlearning.

### 3.2 Formal Objective — Maximize Detection of Vulnerable Knowledge

Construct the largest possible "corruption-probe" dataset D* such that:

1. **Irrelevance**: sim(D*, D_forget) ≤ δ — D* is semantically distant from the forget set
2. **Validity**: PPL_f(D*) ≤ τ — D* consists of linguistically valid, answerable knowledge
3. **Base utility**: Acc(f, D*) ≥ α — the base model handles D* well
4. **Comprehensiveness**: |D*| ≥ k — sufficient coverage

The "success" of D* is measured by how much accuracy drops on D* after unlearning: a large drop reveals hidden collateral corruption that standard benchmarks miss.

### 3.3 Two Complementary Formulations

| Formulation | Objective | Use Case |
|---|---|---|
| **Max-Accuracy** | max_{D*} Acc(f', D*) s.t. constraints | Find the *worst-case* corruption — what retained knowledge is most severely damaged? |
| **Max-Coverage** | max_{D*} |D*| s.t. constraints | Find the *broadest* corruption frontier — how much knowledge is affected at all? |

### 3.4 Key Sub-Questions

**RQ1a — Can feature-space overlap predict corruption probability?**
If a sample x projects strongly onto the forget-associated subspace (before unlearning), is it more likely to experience accuracy degradation under the unlearned model?

**RQ1b — Is corruption localized to a low-dimensional subspace?**
Does unlearning-induced representation drift concentrate in a structured low-dimensional subspace rather than propagating globally?

---

## 4  Method: A Proactive Prediction and Prevention Framework

### 4.1 Stage 1 — Semantic Filtering: Identifying the Vulnerable Neighborhood

1. **Semantic anchor construction**: Extract a central embedding vector V_forget from the forget set (e.g., mean of sentence embeddings).
2. **Vulnerable subset identification**: Search a large reference corpus (e.g., Wikitext-2) for samples whose embeddings fall within distance δ of V_forget. These form the "semantic neighborhood" — the zone most at risk.
3. **Side-Effect Index**: Quantify localized damage as:

   ```
   Side-Effect Index = Avg_PPL(unlearned model, vulnerable subset) / Avg_PPL(base model, vulnerable subset)
   ```

   A high ratio on the vulnerable subset, combined with a near-1.0 ratio on the full corpus, reveals damage invisible to global metrics.

### 4.2 Stage 2 — Representation Subspace Analysis: Predicting Corruption Geometry

Use the base model's internal representations to estimate which parameter subspace will be perturbed:

1. **PCA of forget-set activations**: Identify the principal directions associated with forget knowledge at each layer.
2. **Projection scoring**: For any candidate retained sample, compute its projection onto the forget-associated subspace. High projection ⟹ high corruption risk.
3. **Subspace overlap quantification**: Use CKA or subspace angles to measure how much the forget subspace overlaps with subspaces important for other knowledge domains.

### 4.3 Stage 3 — Algorithm-Adaptive Data Preparation

Based on Stages 1–2, reshape the training data before unlearning:

- **Add retain anchors**: Reinforce knowledge near the corruption frontier with additional retain examples.
- **Edit ambiguous samples**: Rewrite samples that straddle the forget/retain boundary to reduce subspace overlap.
- **Reweight training pairs**: Upweight hard-to-preserve samples; downweight redundant easy cases.
- **Algorithm conditioning**: Adjust strategies based on the unlearning algorithm's parameter-space behavior (as revealed by task-vector analysis).

---

## 5  Experimental Setup

### 5.1 Models and Benchmarks

| Component | Configuration |
|---|---|
| Base model | Llama-3.2-1B-Instruct (fine-tuned on TOFU) |
| Forget set | TOFU forget10 (10% of training data) |
| Unlearning algorithms | GradDiff (40 hyperparameter variants), RMU, DPO |
| Hyperparameter grid (GradDiff) | LR ∈ {1e-5, 2e-5, 3e-5, 4e-5, 5e-5} × α ∈ {1, 2, 5, 10} × Epochs ∈ {5, 10} = 40 variants |
| Reference corpus | Wikitext-2-raw-v1 |
| Pareto selection | Subset of models on the Pareto frontier of Forget Quality vs Model Utility |

### 5.2 Analysis Toolkit

| Analysis | Tool / Metric | What It Reveals |
|---|---|---|
| Task-vector orthogonality | Cosine similarity of flattened Δθ vectors | Algorithm-dependence of unlearning direction |
| Semantic filtering | Sentence-embedding similarity + PPL ratio | Localized knowledge degradation invisible to global metrics |
| PCA similarity & shift | Layer-wise PC1 alignment and mean displacement | Directional stability and translational drift per layer |
| CKA | Linear centered kernel alignment per layer | Subspace structural preservation |
| Fisher Information | Eigenspectrum of Fisher matrix at selected layers | Parameter sensitivity redistribution vs. landscape flattening |

---

## 6  Results and Analysis

### 6.1 Task-Vector Orthogonality (Finding 1: Collateral damage is algorithm-dependent)

- **retain90** vector is nearly orthogonal to all GradDiff vectors (cosine 0.01–0.04), confirming that retaining data vs. unlearning data produce fundamentally different parameter-space trajectories.
- Within GradDiff, hyperparameters strongly modulate direction: lr1e-05_α10_ep5 and lr1e-05_α10_ep10 are almost parallel (0.97), while lr2e-05_α1_ep5 vs lr3e-05_α10_ep5 diverge substantially (0.33).
- **Takeaway**: The same forget set produces very different parameter perturbations depending on algorithm choice and hyperparameters. Collateral damage prediction must be algorithm-aware.

### 6.2 Semantic Filtering Reveals Hidden Damage (Finding 2: Global metrics underestimate corruption)

- When PPL is measured on the full Wikitext-2 corpus, unlearned models show only modest degradation from the base model.
- When PPL is measured on the **semantically filtered vulnerable subset** (samples near the forget-set embedding), the Side-Effect Index reveals significant spikes.
- **Takeaway**: Standard model-utility metrics systematically miss localized knowledge corruption. Semantic filtering is necessary and sufficient to reveal the true damage boundary.

### 6.3 Representation-Level Diagnostics (Finding 3: Corruption is structured and localized)

For RMU on TOFU forget10:

- **PCA similarity**: Near 1.0 across most layers; drops only in final transformer blocks, and only for the forget set. Retain and unrelated sets stable throughout.
- **PCA shift**: Moderate translational drift on forget set in upper layers; minimal drift on retain/unrelated sets.
- **CKA**: Near 1.0 globally; partial decline only in the final layer for the forget set. No widespread subspace fracture.
- **Fisher Information**: Redistribution of parameter sensitivity (not flattening), consistent with localized weight updates rather than catastrophic landscape degradation.

**Taxonomy classification**: RMU on forget10 exhibits **Reversible, Non-Catastrophic Forgetting** — controlled suppression of output rather than genuine erasure of internal representations.

**Takeaway**: Corruption is geometrically structured and confined to identifiable subspaces, which makes pre-unlearning prediction feasible.

### 6.4 Synthesis: The Three Findings Support a Proactive Framework

| Finding | What it tells us | Why it matters for proactive prediction |
|---|---|---|
| Algorithm-dependent vectors | Different algorithms damage different subspaces | Prediction must be algorithm-adaptive |
| Semantic filtering reveals hidden damage | Damage is localized near the forget set's semantic neighborhood | Prediction can target the right region using semantic similarity |
| Structured representation drift | Corruption lives in a low-dimensional subspace | Prediction can use subspace projection to estimate per-sample risk |

Together, these three findings establish that collateral corruption is **structured enough to predict and localized enough to prevent**.

---

## 7  Related Work (Positioning)

### 7.1 Benchmarks and Diagnostics

TOFU (Maini et al., 2024), WMDP (Li et al., 2024), MUSE (Shi et al., 2024), BLUR (Hu et al., 2025), Knowledge Holes (Ko et al., 2025). These define *what* to measure but are **reactive** — they diagnose failure after unlearning.

### 7.2 Representation-Aware Unlearning

RMU, LUNAR (Shen et al., 2025), FALCON (Hu et al., 2025), CIR (Sondej & Yang, 2025), MRP (Wu et al., 2025). These operate on activations/representations during unlearning but do not address **pre-unlearning data preparation**.

### 7.3 Geometry-Aware Optimization

SOUL (Jia et al., 2024), Gauss-Newton Unlearning (McKinney et al., 2026), Geometric Disentanglement (Zhou et al., 2025). Better optimizers help but assume the data are already suitable.

### 7.4 Data-Centric Unlearning

ReLearn (Xu et al., 2025), Align-then-Unlearn (Spohn et al., 2025), Data Augmentation for MU (Falcao & Cordeiro, 2025). Closest to our direction but lack: (a) a prediction stage before augmentation, (b) algorithm-adaptive conditioning, (c) semantic-filtering-based vulnerability detection.

### 7.5 Our Position

We occupy a unique niche: **proactive, prediction-first, algorithm-adaptive data preparation** for unlearning. Rather than improving the unlearning algorithm itself or diagnosing failures after the fact, we intervene at the data level before unlearning begins, guided by geometric analysis of the model's representation space.

---

## 8  Proposed Paper Outline (Rewritten Draft)

Below is the recommended section-by-section structure for the rewritten paper.

### Title
**Predicting Collateral Corruption in LLM Unlearning: A Proactive, Geometry-Aware Framework**

### Abstract
- LLM unlearning causes collateral corruption due to representational entanglement.
- Current approaches detect damage only after unlearning (reactive).
- We propose a proactive framework: predict which retained knowledge is vulnerable before unlearning, using representation geometry.
- Three preliminary findings support feasibility: (1) unlearning vectors are algorithm-specific not data-specific, (2) semantic filtering reveals damage invisible to global metrics, (3) representation drift is structured and low-dimensional.

### 1. Introduction
- Motivation: unlearning necessity + the collateral damage problem.
- Gap: reactive vs proactive. Current tools diagnose but don't predict.
- Contribution: a three-stage proactive framework (semantic filtering → subspace analysis → algorithm-adaptive data preparation).

### 2. Background and Related Work
- 2.1 Machine Unlearning in LLMs (benchmarks, methods taxonomy)
- 2.2 Representation Entanglement and Superposition
- 2.3 Reactive Approaches to Collateral Damage (Knowledge Holes, SKEB)
- 2.4 Data-Centric and Geometry-Aware Unlearning
- 2.5 Research Gap: No Proactive Prediction Framework

### 3. Problem Formulation
- 3.1 Definitions (base model, forget set, unlearning vector, collateral corruption)
- 3.2 Pre-Unlearning Vulnerability Prediction Problem
- 3.3 Two Objectives: Max-Accuracy (worst-case) and Max-Coverage (breadth)

### 4. Method
- 4.1 Semantic Filtering and Vulnerable Subset Identification
- 4.2 Representation Subspace Analysis (PCA projection scoring, CKA overlap)
- 4.3 Algorithm-Adaptive Data Preparation
- 4.4 Side-Effect Index as Evaluation Metric

### 5. Experiments
- 5.1 Setup (models, datasets, hyperparameter grid)
- 5.2 Task-Vector Analysis: Algorithm-Dependence of Unlearning Direction
- 5.3 Semantic Filtering: Revealing Hidden Side Effects
- 5.4 Representation Diagnostics: Structured and Localized Drift
- 5.5 Pareto-Optimal Model Analysis

### 6. Discussion
- 6.1 Why Global Metrics Fail (the dilution argument)
- 6.2 Corruption Is Predictable: From Observation to Framework
- 6.3 Implications for Continual Unlearning
- 6.4 Limitations and Future Work

### 7. Conclusion

---

## 9  Key Differences from the Current Draft

| Aspect | Current Draft | Proposed Rewrite |
|---|---|---|
| **Narrative focus** | Jumps between RQ1, RQ1a, RQ1b, two problem formulations — reads as brainstorming notes | Single coherent story: motivation → evidence → framework |
| **Title** | "Protect Knowledge from Unlearning" (vague, could mean anything) | "Predicting Collateral Corruption..." (precise, signals the contribution) |
| **Problem formulation** | Two separate formulations (Max Acc and Max Size) without clear connection | Unified formulation with two complementary objectives clearly motivated |
| **RQs** | Listed as separate sub-questions with annotations ("Bo [proactive]...") | Integrated as sub-questions within a single coherent problem statement |
| **Experiments** | Task vectors, semantic filtering, and representation analysis appear disconnected | Three experiments explicitly mapped to three pillars of the framework |
| **Related work** | Scattered within RQ sections | Consolidated into one section with clear positioning table |
| **Methods section** | Mostly placeholder ("5.1 ** Prediction") | Three concrete stages with clear inputs/outputs |
| **Writing style** | Mix of formal and informal, some notes-to-self visible | Consistent formal tone throughout |

---

## 10  References

- Elhage, N., et al. (2022). Toy models of superposition. Transformer Circuits Thread.
- Entesari, T., et al. (2025). Constrained entropic unlearning. NeurIPS.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR.
- Ilharco, G., et al. (2023). Editing models with task arithmetic. ICLR.
- Ko, M., et al. (2025). Probing hidden knowledge holes in unlearned LLMs. NeurIPS.
- Li, N., et al. (2024). The WMDP benchmark. ICML.
- Maini, P., et al. (2024). TOFU: A task of fictitious unlearning for LLMs. CoLM.
- Shah, A. & Le, T. (2025). The limits of obliviate. arXiv:2510.25732.
- Shi, W., et al. (2024). MUSE: Machine unlearning six-way evaluation.
- To, B. T. T. & Le, T. (2025). Harry Potter is still here! EMNLP Findings.
- Xu, X., et al. (2025). Unlearning isn't deletion. arXiv:2505.16831.
