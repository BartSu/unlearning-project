# Protecting Model Utility from Unlearning: A Semantic Filtering Approach

**Authors:** Bo Su, Yueru Yan, Thai Le (Indiana University, Bloomington)  
**Status:** Draft v2 — streamlined backbone

---

## Core Idea (One Sentence)

Standard model utility evaluation after unlearning uses the full Wikitext corpus, which dilutes and hides localized damage; we show that semantic proximity to the forget set is a reliable predictor of which Wikitext samples will be hurt, and use this to build a targeted evaluation subset before unlearning happens.

---

## 1. Introduction

**Context.** Machine unlearning in LLMs — selectively removing the influence of specific training data — has become essential for privacy compliance (GDPR), IP protection, and safety. A well-designed unlearning algorithm must satisfy two competing objectives: (1) successfully forget the target data, and (2) preserve the model's general utility on retained, unrelated knowledge.

**Problem.** Because LLMs encode knowledge in distributed, overlapping representation subspaces, removing one target concept inevitably perturbs shared parameters that also support other capabilities. This causes **collateral corruption**: model utility on seemingly unrelated knowledge degrades after unlearning, without any warning.

**Gap in existing evaluation.** Current benchmarks measure model utility by computing perplexity on the full Wikitext-103 corpus. This global average systematically *hides* collateral damage — the damage is concentrated in a small, semantically proximate subset of the corpus, and averaging over the full corpus dilutes the signal to near zero. As a result, a model can appear to maintain utility globally while silently failing on specific knowledge domains.

**Our approach.** We propose a **semantic filtering framework** that identifies the vulnerable subset of Wikitext — the samples most likely to suffer collateral utility degradation — directly from the forget set, *before* unlearning is applied. This targeted subset provides a sensitive, principled evaluation signal that global metrics cannot.

**Contributions.**
1. We demonstrate that model utility degradation after unlearning is semantically localized: samples semantically proximate to the forget set are disproportionately affected.
2. We propose the **Side Effect Index (SEI)** as a sample-level metric to quantify collateral damage.
3. We propose a semantic filtering pipeline that identifies a vulnerable subset D\* ⊂ Wikitext from D_forget alone, without access to the unlearned model during construction.
4. We empirically show that SEI(D\*) is significantly higher than SEI on globally-sampled Wikitext, validating that semantic filtering isolates the right samples.

---

## 2. Background and Related Work

**Machine unlearning benchmarks.** TOFU (Maini et al., 2024) establishes the standard unlearning task: given a model fine-tuned on fictitious author biographies, unlearn a forget subset while retaining model utility. Model utility is measured via perplexity on Wikitext and accuracy on factual QA sets. WMDP (Li et al., 2024) and MUSE (Shi et al., 2024) extend this to safety-critical and copyright domains. A common thread: model utility is evaluated globally, not in targeted subsets.

**Collateral corruption is underdiagnosed.** Ko et al. (2025) show that unlearned models harbor "hidden knowledge holes" — capability gaps not captured by standard evaluation. Their approach uses reinforcement learning to discover failure prompts *after* unlearning. Shah and Le (2025) demonstrate that prompt-level structural properties can predict knowledge leakage at inference time. Both approaches are **reactive**: evaluation happens after unlearning, and identification of damage depends on access to the unlearned model.

**Why global perplexity fails.** Global Wikitext perplexity averages over tens of thousands of samples. If collateral damage is concentrated in, say, 200 semantically related samples out of 30,000, the signal is suppressed by a factor of ~150. This motivates targeted evaluation over a semantically filtered subset.

---

## 3. Problem Formulation

Let M denote the original LLM and D_forget ⊂ D_train the data designated for removal. After applying an unlearning algorithm, M' denotes the unlearned model. The standard utility evaluation computes:

> `PPL_M'(Wikitext)` vs. `PPL_M(Wikitext)`

We argue this is insufficient. Instead, we seek to construct a **vulnerable subset** D\* ⊂ Wikitext satisfying:

| Constraint | Meaning |
|-----------|---------|
| `sim(D*, D_forget) ≤ δ` | D\* is semantically *distinct* from the forget content — genuine collateral damage, not leakage |
| `PPL_M(D*) ≤ τ` | M handles D\* well (it is retained knowledge, not hard examples) |
| `M' degrades on D*` | The unlearned model suffers measurable utility loss on D\* |
| No access to M' during construction | D\* is built from D_forget and M alone |

**Goal:** Construct D\* that maximally captures the collateral utility degradation induced by unlearning, using only pre-unlearning information.

---

## 4. Method: Semantic Filtering

The pipeline has three steps.

### Step 1 — Semantic Anchor Extraction

Compute the mean embedding of all samples in D_forget using the base model M (e.g., mean over last-layer hidden states at the [EOS] position):

> `V_forget = mean_{x ∈ D_forget} h_M(x)`

V_forget is the centroid of the forget set in representation space — the "semantic signature" of the knowledge to be removed.

### Step 2 — Vulnerable Subset Retrieval

For each sample w in Wikitext, compute:

> `score(w) = cosine_sim(h_M(w), V_forget)`

Retain samples with `score(w) ≥ θ` to form D\*. These samples are semantically close to the forget content in representation space, making them likely to share parameters perturbed by unlearning.

**Why this satisfies the irrelevance constraint.** Semantic similarity in embedding space does not imply lexical or content overlap. Samples in D\* are *topically related* to the forget set but are genuine Wikitext — real-world retained knowledge, not reproductions of D_forget. We verify this by checking that `sim(D*, D_forget) ≤ δ` at the token/content level.

### Step 3 — Side Effect Quantification

After unlearning, compute the **Side Effect Index (SEI)** over any evaluation set S:

> `SEI(S) = Avg_PPL_M'(S) / Avg_PPL_M(S)`

- `SEI(S) ≈ 1.0` → minimal collateral impact on S  
- `SEI(S) > 1.0` → model utility degrades on S after unlearning

**Main claim:** `SEI(D*) >> SEI(Wikitext_full)`, demonstrating that our semantic filter concentrates the damage signal rather than diluting it.

---

## 5. Experimental Setup

**Model.** Llama-3.2-1B-Instruct.

**Unlearning task.** TOFU forget10 setting: unlearn 10% of fictitious author biographies.

**Unlearning algorithm.** GradDiff. We evaluate across 40 hyperparameter configurations (learning rate ∈ {1e-5 ... 5e-5}, retain weight α ∈ {1, 2, 5, 10}, epochs ∈ {5, 10}) and select Pareto-optimal models that balance Forget Quality and Model Utility according to the open-unlearning benchmark.

**Datasets.**
- Forget set: TOFU forget10
- Evaluation corpus: Wikitext-2-raw-v1

**Baselines for comparison.**
- Global Wikitext SEI (no filtering)
- Random subset of Wikitext matched in size to D\* (controls for subset size effect)
- Semantic neighbor sets at varying thresholds θ (ablation)

---

## 6. Results (Planned)

**Main result — Figure 1.** Bar chart comparing SEI across: (a) full Wikitext, (b) random Wikitext subset, (c) D\* (semantic filter). Expected: SEI(D\*) significantly exceeds both baselines.

**Main result — Figure 2.** SEI(D\*) as a function of semantic similarity threshold θ, showing that increasing θ (stricter filtering → tighter semantic neighborhood) yields higher SEI — validating that proximity to the forget set drives vulnerability.

**Irrelevance check.** Show that content-level similarity between D\* and D_forget remains low (sim ≤ δ), confirming D\* represents genuine collateral damage rather than near-duplicate forget content.

**Across algorithms.** Replicate with RMU and DPO to show the filtering approach generalizes across unlearning methods (even as the absolute damage magnitude varies by algorithm).

---

## 7. Supporting Analyses (Appendix / §6)

These analyses motivate *why* semantic filtering works and provide mechanistic context. They are not part of the core pipeline.

**A — Task vector orthogonality.** Unlearning vectors from GradDiff, RMU, and DPO on the same forget set are nearly orthogonal in parameter space, suggesting each algorithm damages a different neighborhood of unrelated knowledge. This explains why collateral corruption is algorithm-specific and motivates checking multiple algorithms in §6.

**B — Representation-level localization (PCA / CKA / Fisher).** Layer-wise analysis shows that unlearning-induced representation drift is concentrated in the upper transformer layers and confined to the principal directions of the forget set's activation subspace. This provides geometric evidence that semantic proximity in embedding space is a valid proxy for shared parameter sensitivity — the theoretical basis for why the semantic filter works.

---

## 8. Conclusion

Standard model utility evaluation after unlearning uses global corpus perplexity, which systematically dilutes concentrated collateral damage. We show that the vulnerability is semantically localized — samples close to the forget set in representation space are disproportionately affected — and propose a semantic filtering pipeline that identifies this vulnerable subset before unlearning, using only the forget set and the base model. The resulting D\* provides a targeted, sensitive evaluation signal that global metrics miss, and the Side Effect Index (SEI) quantifies the damage in a straightforward, comparable way across algorithms and settings.

---

## Appendix Outline

- **A.1** Task Vector Orthogonality Analysis (full results across algorithm pairs)
- **A.2** Representation-Level Analysis: PCA Similarity, PCA Shift, CKA, Fisher Information
- **A.3** Ablation: Semantic anchor layer choice, embedding pooling strategy, similarity threshold θ
- **A.4** Dataset statistics for D\* at varying thresholds

---

*Last updated: 2026-03-12*
