---
loại: "quyết-định"
mã: "DECISION-STAGE03-REPRESENTATION-FIRST-108"
dự_án: "long-range-temporal-alignment"
trạng_thái: "bị-thay-thế"
chuẩn_hiện_hành: false
thay_thế: []
bị_thay_thế_bởi: ["[[23_GOAL_LADDER_AND_ACCEPTANCE_20261105]]"]
tham_chiếu: ["18_STAGE03_MECHANISM_BEFORE_SCALE", "V2_TARGETED_CONTINUAL_ALIGNMENT_CONTRACT", "23_GOAL_LADDER_AND_ACCEPTANCE_20261105"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-11-05"
---

# Stage 03 Next Steps for Temporal-Qwen: Representation-First Evidence Before Data Scale

> [!WARNING] TÀI LIỆU BỊ THAY THẾ VỀ ĐỊNH NGHĨA MỤC TIÊU / SUPERSEDED IN PART (2026-11-05)
> Tài liệu này được thay thế về mặt xác định mục tiêu nghiệm thu bởi [ADR 23: GOAL LADDER AND ACCEPTANCE](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md).
> - Trong ADR 19, MCQA từng được xem là "downstream behavioral probe và evaluator-sanity test", cấm lấy độ chính xác MCQA làm mục tiêu chính.
> - Theo quyết định phạm vi ngày 2026-11-05 của chủ đề tài (User), **GOAL 1 (YEAR-only closed-answer temporal QA)** quy định độ chính xác câu trả lời đóng (MCQA 4 lựa chọn A/B/C/D + nhị phân Yes/No) là **kết quả bắt buộc tiên quyết** (primary required outcome). Mọi thay đổi biểu diễn hoặc logit shift chỉ là cơ chế giải thích bổ trợ, không thể thay thế cho độ chính xác trả lời đúng.
> - Các nguyên tắc phương pháp luận của ADR 19 về kiểm soát shortcut thực thể, paired statistics, không random split trên near-duplicate, và audit rò rỉ vẫn giữ nguyên giá trị ràng buộc.

## Executive Summary

**Recommendation: do not scale the Stage 03 training dataset yet.** The best next step is a **representation-first verification cycle** in which the partner supplies FoNE / Ticktack / temporal-core / Seam checkpoints and your side evaluates them under one frozen training/evaluation contract. MCQA should remain a **downstream behavioral probe and evaluator-sanity test**, not become the object that the research repeatedly modifies until accuracy improves.

This recommendation follows directly from the latest project evidence. The repaired wrong-year intervention now produces a measurable change—mean logit delta about `0.118655`, with `4/36` prediction flips on eligible cases—but this proves only that the temporal treatment can reach the model, not that the resulting answer tracks the correct temporal fact. The four-rotation MCQA diagnostic remains much more troubling: about `24.20%` accuracy over 624 variants, `488/624` predictions concentrated on one answer letter, and `0/156` source questions correct under all four rotations. The current semantic-counterfactual pilot is also too small and unsuccessful to support a causal representation claim.

The earlier Run 05.2 comparison likewise does not justify scaling. Arm B reached `44/180 = 24.44%`, while Base/BOTH-OFF reached `58/180 = 32.22%`; the paired McNemar comparison had discordances `b=37`, `c=51` and `p≈0.165`, so the observed difference is inconclusive rather than proof of either equivalence or definite harm. The later forensic audit also corrected historical denominator and lineage mistakes: DEV Relation was 482 rather than 60 examples, clean DEV MCQA was 120 rather than 40, and 60 of the TempUN-180 cases shared source lineage with NWP training while 120 were source-nonoverlapping under the available IDs.

The correct scientific split is therefore:

| Responsibility | Primary task now | What not to optimize yet |
|---|---|---|
| **Representation partner** | Produce auditable checkpoints for FoNE, Ticktack, static LoRA, temporal-core/MaT-LoRA-derived variants, Seam variants; expose internal temporal states where possible | Downstream MCQA accuracy alone |
| **Your Stage 03 work** | Objectives, deterministic evaluator, causal counterfactuals, MCQA shortcut diagnostics, paired/clustered statistics, retention | Repeatedly adding MCQA examples to compensate for weak representation |
| **Joint evidence** | Show `time → representation → hidden state → logits → correct temporal answer` under matched controls | Claiming mechanism success merely because logits changed |

The primary scientific claim should therefore become:

> **Temporal-Qwen improves the internal encoding and causal use of temporal information relative to matched non-temporal/static adaptation controls.**

MCQA accuracy is supporting evidence. It is **not** the definition of success.

That framing is also better aligned with the source methods. Ticktack explicitly proposes a sexagenary representation, polar coordinates and temporal representational alignment; FoNE changes the representation of numerical values through Fourier features; and the current arXiv body for the temporal-core work describes a shared-basis factorization in which temporal variation is concentrated in a compact \(F_t\) through \(\Delta W(t)=BF_tA\). These are fundamentally representation/mechanism proposals.

**Stage decision as of October 18, 2026:**

> **GO:** audit + inference diagnostics + counterfactual representation probes + one-variable smoke training.  
> **NO-GO:** larger MCQA corpus or full Stage 03 training run.  
> **Promotion condition:** causal temporal representation must first survive matched controls and produce semantically correct temporal changes.

## Scientific Objective and Decision Boundary

The most important correction is conceptual: **Stage 03 should stop asking “How do we raise MCQA accuracy?” and start asking “Can we prove that the temporal representation is better?”**

FoNE's original design is quite specific: detected numbers are replaced by a special `[NUM]` token, the Fourier representation is added at that number position, and the paper defines specialized numerical decoding. It is therefore not scientifically safe to interpret every project-specific use of a Fourier vector as an exact reproduction of FoNE. Ticktack similarly proposes a particular year representation using the sexagenary cycle, polar coordinates and representational alignment during post-training. The current temporal-core formulation at arXiv:2602.11965 separates fixed shared bases from a time-varying core,

\[
\Delta W(t)=B F_t A,
\]

so \(F_t\) itself is an obvious first-class object to instrument rather than inferring temporal activity only from final QA accuracy.

A naming caution matters for the thesis/report: the current body of arXiv:2602.11965 describes the approach as **LoTiC, Low-rank Time-varying Core**, even though older/cached material refers to **MaT-LoRA**. Unless the partner implementation follows the current paper exactly, the safest label is **“MaT-LoRA/LoTiC-inspired project variant”** rather than claiming exact reproduction. The paper's current formulation uses shared bases and a compact time-varying core and considers flow, recurrent, and direct time-to-core parameterizations.

The latest project evidence says the unresolved problem is **mechanism identification**, not simply insufficient data. In Run 05.2, turning off Seam produced only about `0.000369` mean logit change and zero prediction changes, whereas turning off the LoRA path produced about `1.754` logit change and 88 changed predictions. These numbers do **not** prove that Seam is useless or that LoRA contains correct temporal knowledge; they show that the current checkpoint's observable output is far more sensitive to the adapter path than the tested Seam path. The original Run 05.2 receipts also established compact, distinct adapter checkpoints and a frozen-backbone training contract, although later forensic work superseded several earlier reported denominators.

A useful current-state visualization is:

```text
TempUN-180 strict accuracy — current audited comparison

Base / BOTH-OFF       32.22%  ████████████████
Stage03 Arm B         24.44%  ████████████
                      ↑
              paired p ≈ 0.165
              therefore inconclusive

Four-rotation MCQA    24.20%  ████████████
                      0/156 fully rotation-consistent
```

These values are diagnostic, not a leaderboard across perfectly identical protocols; the rotation test has 624 variants rather than 180 original cases.

The **dataset-rescue boundary** should therefore be explicit:

- Creating a small, source-clean **counterfactual evaluation set** is justified because it makes the representation hypothesis testable.
- Balancing or permuting options during a tiny MCQA smoke run is justified as a **shortcut-control intervention**.
- Repeatedly adding training MCQA examples until the model beats Base is **not** justified at this stage; that would blur whether the improvement came from better temporal representation, more supervision, benchmark adaptation, or answer-position learning.

The next scientific result should ideally be of the form:

\[
\text{Temporal variant}
>
\text{matched static variant}
\]

on a **pre-registered temporal causal endpoint**, while the training objective, data exposure, decoding, parser and evaluation set remain fixed.

## Experimental Architecture and Ablation Matrix

The experiments should be split into two layers. First, perform **inference-only representation ablations**. Only after those are interpretable should you perform a tiny objective smoke test.

The reason to retain static LoRA as a control is methodological rather than historical. Original LoRA freezes pretrained weights and introduces low-rank trainable updates; it therefore gives a clean PEFT control without necessarily containing an explicit temporal state. The current temporal-core paper explicitly argues that independent LoRA factors do not themselves establish a common temporal coordinate system and instead introduces shared bases plus \(F_t\).

**Recommended ablation matrix**

| ID | Static MLP-LoRA | FoNE | Ticktack | Temporal core / MaT-LoRA-derived | Seam | Scientific purpose | Priority |
|---|---:|---:|---:|---:|---:|---|---|
| **A0 Base** | — | — | — | — | — | Absolute pretrained baseline | Essential |
| **A1 Static** | ✓ | — | — | — | — | Separate generic task adaptation from temporal representation | Essential |
| **A2 FoNE** | ✓ | ✓ | — | — | — | Test numeric/Fourier representation contribution | Essential |
| **A3 Ticktack** | ✓ | — | ✓ | — | — | Test cyclic/year representation contribution | Essential |
| **A4 Seam** | ✓ | — | — | — | ✓ | Test external injection contribution without temporal encoding combination | High |
| **A5 Tick+Seam** | ✓ | — | ✓ | — | ✓ | Test whether explicit temporal encoding benefits from the external injection path | Essential |
| **A6 FoNE+Tick** | ✓ | ✓ | ✓ | — | — | Test representation combination before dynamic adapter | High |
| **A7 Static Full** | ✓ | ✓ | ✓ | — | ✓ | Strong static temporal-representation control | Essential |
| **A8 Dynamic** | replaced | optional | ✓ | ✓ | — | Test dynamic \(F_t\) mechanism against A3/A6 | Essential |
| **A9 Dynamic+Seam** | replaced | optional | ✓ | ✓ | ✓ | Full project-style temporal mechanism | Essential |

This is deliberately **not** a brute-force factorial search. With unspecified compute, the objective is to estimate the important causal contrasts, not optimize every combination.

The most important contrasts are:

\[
A3-A1
\]

for the value of Ticktack-like representation;

\[
A5-A3
\]

for the incremental effect of Seam;

\[
A8-A3
\]

for the value of the dynamic temporal core over matched static adaptation; and

\[
A9-A8
\]

for Seam's incremental value on the dynamic representation.

FoNE should be evaluated carefully rather than automatically retained. The original FoNE paper's architecture is number-token-specific, and its reported strengths are primarily numerical representation and arithmetic. A project-specific 40D FoNE/Can-Chi vector may be scientifically useful, but the result should be described as the project's temporal encoding variant rather than evidence that the original FoNE method itself succeeds or fails.

**Minimal experiment plan and budgets**

| Experiment | Inputs | Proposed budget | Compute class | Required output | Decision value |
|---|---|---:|---|---|---|
| **Artifact audit** | Existing checkpoints + Run 05.2 receipts | All artifacts, no model training | CPU / storage | lineage + SHA + adapter inventory | Critical |
| **Deterministic checkpoint sweep** | A0–A9 as available | One fixed evaluation pass/cell | Low GPU | raw predictions + SCOREBOARD | Very high |
| **Treatment-delivery probe** | 32–64 single-year pairs | true/wrong/zero/off controls | Low GPU | \(z,F_t,\Delta W\), logit traces | Critical |
| **Hidden-state sensitivity** | Same pair subset, top 4–6 cells | selected layers/positions | Low–moderate GPU/RAM | layerwise sensitivity file | Critical |
| **Grounded gold-flip diagnostic** | 32–64 gold-flip pairs + controls | one deterministic pass/cell | Low GPU | causal semantic scoreboard | Critical |
| **MCQA rotation diagnostic** | Existing MCQA set | four rotations; no new training data | Low GPU | semantic consistency + label bias | High |
| **Objective smoke** | Best 1–2 representation variants | **40–100 steps**, one screening seed | Moderate GPU | Control vs permutation Treatment | Conditional |
| **Finalist replication** | Only successful smoke | at least 3 seeds | Moderate GPU | cluster-aware replicated result | Conditional |
| **Large-data training** | Expanded training corpus | **0 budget now** | High GPU | — | Current NO-GO |

The proposed 40–100-step smoke budget is a deliberately small screening range, not a claim that this is an optimal training schedule. The existing Stage 03 run used 40 optimizer steps, so a similar-scale experiment is appropriate for determining whether a single change fixes a known pathology before spending materially more compute.

The **only training-side change I would permit initially** is online option permutation with correct semantic-label remapping, because it addresses a known answer-position shortcut while leaving the representation hypothesis untouched. It should not be described as a representation innovation. If it improves option entropy but not temporal gold-flip accuracy, its conclusion is simply: **evaluator/training hygiene fixed; temporal reasoning still unproven.**

## Data Protocol, Metrics, and Statistical Inference

The strongest improvement to Stage 03 is not “more data”; it is **better causal eligibility**.

The current project already demonstrated why this matters: 144 of 180 TempUN cases were multi-year bypasses under the single-year conditioning logic, while only 36 were eligible for the repaired wrong-year intervention. The earlier `C_WRONG_YEAR` path had also failed to actually perturb those eligible years before the caller was repaired. Therefore, the old equality of `C_TRUE` and `C_WRONG_YEAR` could not legitimately be interpreted as temporal invariance.

Every evaluation case should receive one of these statuses:

| Status | Use |
|---|---|
| `ELIGIBLE_GOLD_FLIP` | **Primary causal semantic endpoint** |
| `ELIGIBLE_GOLD_SAME` | Negative control for excessive sensitivity |
| `MULTI_YEAR_BYPASS` | Report separately; exclude from single-year effect denominator |
| `AMBIGUOUS_YEAR` | Exclude |
| `MISSING_YEAR` | Exclude |
| `TREATMENT_NOT_DELIVERED` | Implementation failure, not a model result |
| `UNKNOWN_LINEAGE` | Secondary analysis only; not clean primary evidence |

A proper **gold-flip counterfactual pair** should hold entity, relation, sentence structure and all non-temporal content fixed while changing the relevant time from \(t_1\) to \(t_2\), where the externally verified correct answer changes. For example, if a proposition is false in \(t_1\) and true in \(t_2\), the pair must contain validated gold labels for both states. Arbitrary `year + 37` perturbations remain useful for testing whether a signal reaches the model, but they are not semantic counterfactual evidence unless the new year produces a known change in the correct answer. The project's repaired +37 diagnostic should therefore remain under **treatment delivery**, not **causal temporal reasoning**.

The first set need not be large. A manually or programmatically verified **32–64 gold-flip-pair diagnostic set**, plus similar gold-same controls, is enough to expose mechanism failures cheaply. Only after observing a plausible effect should the **evaluation set** be enlarged for statistical power. That is fundamentally different from enlarging the training corpus to rescue accuracy.

For every pair, capture the entire causal pathway:

\[
t
\rightarrow z(t)
\rightarrow F_t
\rightarrow \Delta W(t)
\rightarrow h_\ell(t)
\rightarrow \ell(t)
\rightarrow \hat y(t)
\]

where applicable. The current temporal-core paper makes \(F_t\) the explicit compact temporal state, so direct instrumentation is particularly appropriate for temporal-core checkpoints.

**Mechanism metrics**

For temporal input encoding:

\[
D_z = \|z(t_1)-z(t_2)\|_2.
\]

For each dynamic adapter module \(m\):

\[
D_F^{(m)}
=
\|F_{t_1}^{(m)}-F_{t_2}^{(m)}\|_F,
\]

and preferably:

\[
D_W^{(m)}
=
\|\Delta W^{(m)}(t_1)-\Delta W^{(m)}(t_2)\|_F.
\]

For hidden state at layer \(\ell\):

\[
S_\ell
=
\frac{
\|h_\ell(t_1)-h_\ell(t_2)\|_2
}{
\max(\|h_\ell(t_1)\|_2,\epsilon)
}.
\]

Also record cosine similarity so a norm increase is distinguishable from a direction change.

At the output level, record mean absolute logit delta, logit \(L_2\) delta, and—more scientifically important—the **gold-margin shift**:

\[
M(t)
=
\ell_{\text{gold}(t)}
-
\max_{j\neq \text{gold}(t)} \ell_j.
\]

For a real gold-flip counterfactual, the key question is not merely whether logits change; it is whether the changed year increases the margin of the **new correct semantic answer**.

Matched controls should include same-year repeated forward, zero temporal feature, representation OFF, LoRA/adapter OFF, Seam OFF, and a random perturbation whose norm is approximately matched to the temporal perturbation. This distinguishes genuine temporal selectivity from a network that reacts strongly to any perturbation.

**Behavioral metrics**

Report strict accuracy, invalid rate, macro recall/F1, per-answer recall, prediction entropy, position-specific accuracy, semantic permutation consistency and counterfactual-pair measures:

\[
\text{BothCorrect}
=
P[
\hat y(t_1)=y(t_1)
\land
\hat y(t_2)=y(t_2)
],
\]

and:

\[
\text{CorrectFlip}
=
P[
\hat y(t_1)=y(t_1),
\hat y(t_2)=y(t_2),
y(t_1)\neq y(t_2)
].
\]

These are far more informative for the thesis than a single aggregate MCQA score.

**Statistical tests**

For the same cases evaluated by two systems, use **paired McNemar** on correctness rather than treating model accuracies as independent proportions. McNemar's original test was specifically developed for differences between correlated proportions.

However, ordinary McNemar assumes independent matched pairs. When several questions come from the same source article, event, entity bundle, or temporal cluster, that assumption can fail. Methodological work on clustered matched-pair binary data explicitly develops adjusted tests for this setting. For Temporal-Qwen, I would therefore report ordinary paired McNemar for transparency **and** a cluster-aware primary analysis grouped at the strongest available provenance unit, preferably `event_id` or `source_record_id`.

For continuous endpoints such as \(D_F\), hidden-state sensitivity and gold-margin changes, use cluster-bootstrap confidence intervals, resampling whole source/event clusters rather than individual rows.

The previous internal power calculation estimated that, under the observed Run 05.2 discordance structure, roughly 384 independent pairs would be needed for about 80% power to detect a 10 percentage-point paired effect, and roughly 683 for a 7.5-point effect. Those values are useful **planning estimates, not universal sample-size requirements**; they should be recomputed once the new source-clean counterfactual set provides its own discordance rate and cluster structure.

## Audit Artifacts and Reproducibility Contract

Artifact integrity is not administrative overhead here; it is part of the experimental method. The project has already encountered checkpoint aliasing, inactive or improperly disabled adapters, denominator errors, incorrect joins, and a wrong-year treatment that had not actually been delivered.

PEFT makes the distinction between base model and adapter explicit. Hugging Face documents that a PEFT checkpoint normally stores adapter state in `adapter_model.safetensors` or `.bin`, its `adapter_config.json`, and metadata rather than a copy of the complete base model. Hugging Face also exposes explicit adapter activation/deactivation methods; a loaded adapter and an active adapter should therefore never be treated as synonymous in an audit.

That is especially relevant to the project's historical `adapter off` error class. Every condition should record:

```text
loaded adapters
active adapters
expected temporal modules
actual temporal modules
Seam state
hook positions
hook count during forward
hook count after cleanup
temporal feature hash
checkpoint SHA256
canonical tensor fingerprint
base-weight fingerprint
```

The Qwen2.5-0.5B-Instruct repository currently ships a generation configuration with sampling enabled (`do_sample=true`, temperature `0.7`, top-p `0.8`, top-k `20`). For a paired causal evaluation, Stage 03 should explicitly override that with deterministic decoding rather than depending on repository defaults; Hugging Face's generation documentation identifies greedy generation as the non-sampling regime.

**Required artifact checklist**

| Artifact | Before run | After run | Audit requirement |
|---|---:|---:|---|
| Base model ID + immutable revision | ✓ | ✓ | Must match all compared conditions |
| Tokenizer ID/revision | ✓ | ✓ | Hash/config recorded |
| Parent checkpoint SHA/tensor fingerprint | ✓ | ✓ | No alias ambiguity |
| Adapter file + config | ✓ | ✓ | Expected modules and parent verified |
| Adapter active-state trace | ✓ | ✓ | ON/OFF proven, not inferred |
| `GATES.json` | ✓ | ✓ | Fail-closed gate results |
| Canonical eval manifest | ✓ | ✓ | Manifest SHA frozen |
| Generation config | ✓ | ✓ | Deterministic primary evaluator |
| Raw predictions / generations | — | ✓ | Mandatory; aggregate scores are derived |
| `SCOREBOARD.json` | — | ✓ | Regenerated from raw rows |
| `SPLIT_CONTAMINATION_MATRIX.json` | ✓ | ✓ | Source/event/time lineage |
| Mechanism trace \(z,F_t,\Delta W,h,\ell\) | — | ✓ | Required for representation claim |
| Paired-statistics receipt | — | ✓ | McNemar \(b,c,p\), CIs, cluster analysis |
| Loss/gradient trace for training | — | ✓ | Needed only for smoke/full training |
| NTP retention output | — | ✓ | Same held-out tokens across arms |
| SHA256 manifest | — | ✓ | Seal completed run |
| `RUN_VERDICT.md` | — | ✓ | Explicit GO/NO-GO and failed gates |

A raw evaluation row should be reconstructible at least from:

```text
run_id
example_id
source_record_id
event_id
temporal_cluster_id
checkpoint_sha256
tensor_fingerprint
condition
year_true
year_used
treatment_status
gold_semantic_answer
gold_display_label
raw_generation
parsed_prediction
is_valid
is_correct
seed
prompt_template_sha
parser_sha
generation_config_sha
```

This also prevents another denominator problem: `SCOREBOARD` should be **computed from raw rows**, never manually copied into the report.

The contamination matrix should distinguish exact-example overlap, normalized-text overlap, source-record overlap, event overlap, temporal-cluster overlap and unknown lineage. The later forensic Stage 03 audit found 60 source-lineage-overlap and 120 source-nonoverlap TempUN cases, so a one-bit `clean=true/false` field is already demonstrably too coarse for this project.

## Acceptance Gates and Six-Week Execution Plan

The promotion logic should be sequential. This avoids demanding publication-grade significance from a 32-pair smoke test while still preventing expensive scale-up based on weak evidence.

**Gate to run a smoke training experiment**

`GO_SMOKE` requires:

1. Checkpoint parent and tensor identity resolve correctly.
2. Base weights remain frozen for PEFT-only variants.
3. Adapter inventory and active state are explicit.
4. Adapter-OFF + Seam-OFF reproduces Base within a predeclared numerical tolerance.
5. The wrong-year intervention truly recomputes the temporal input.
6. Eligible year changes produce a measurable internal representation change.
7. Raw rows reconstruct all metrics and denominators.
8. A validated source-clean gold-flip counterfactual diagnostic exists.

Failure of any one of these should keep the project inference-only.

**Gate to scale training/data**

`GO_SCALE` should require all of the following:

| Gate | Acceptance condition |
|---|---|
| **Mechanism** | Temporal representation shows selective \(z/F_t/h/\ell\) sensitivity beyond same-year and random-perturbation controls |
| **Semantic causal** | Temporal variant beats matched static/no-temporal control on preregistered gold-flip pairs |
| **Statistical** | Cluster-aware 95% CI on the primary scaled-evaluation effect excludes zero |
| **Replication** | Finalist effect has the same direction across at least three seeds |
| **MCQA hygiene** | No answer position takes >50% of balanced-rotation predictions; every answer label has nonzero recall |
| **Permutation robustness** | Inverse-mapped semantic consistency improves materially from the present `0/156` baseline |
| **Lineage** | Primary holdout has no known prohibited source/event/time-cluster overlap |
| **Retention** | NTP retention remains within the preregistered project non-inferiority margin; the current project proposal uses ≤5% relative PPL degradation |
| **Attribution** | Scale-up changes only the intervention that succeeded in the matched smoke experiment |
| **Artifact integrity** | SHA/checkpoint/evaluator/adapter gates still pass |

The key distinction is that **more evaluation pairs may be needed before GO_SCALE**, but that does not imply more **training** data. Statistical power and training-data size solve different problems.

**Prioritized action plan**

| Priority | Action | Main resource | Expected output | Target window |
|---|---|---|---|---|
| **Now** | Freeze evaluator and audit all Stage 03 receipts/checkpoints | CPU/storage | canonical manifest, lineage, corrected scoreboard | Oct 19–25 |
| **Next** | Inference-only A0–A9 checkpoint sweep | One GPU | comparable raw predictions + ablation table | Oct 26–Nov 1 |
| **Next** | Gold-flip pairs + \(F_t\)/hidden/logit probes | CPU curation + one GPU | causal mechanism traces | Nov 2–8 |
| **Conditional** | One-variable objective smoke | One GPU | Control vs online-permutation Treatment | Nov 9–15 |
| **Conditional** | Replicate only representation/objective finalists | One GPU or small queue | three-seed result + cluster stats | Nov 16–22 |
| **Decision** | Recompute power, evaluate gates, either scale or stop/refine | CPU + report time | `GO_SCALE` or `NO_GO_SCALE` | Nov 23–29 |

The week-four smoke should **not automatically happen** just because the calendar reaches November 9. It happens only if the audit and representation gates from the first three weeks pass.

## Final Summary and Operating Rule

The scientific endpoint for the coming six weeks should be an auditable chain of evidence:

\[
\boxed{
\text{year intervention}
\rightarrow
\text{representation change}
\rightarrow
\text{temporal-core/hidden-state change}
\rightarrow
\text{gold-directed logit change}
\rightarrow
\text{correct counterfactual answer}
}
\]

with Base, static MLP-LoRA and representation-off controls held constant. If that chain cannot be established at small scale, increasing the amount of Stage 03 training data would make the experiment more expensive without resolving the central scientific uncertainty.
