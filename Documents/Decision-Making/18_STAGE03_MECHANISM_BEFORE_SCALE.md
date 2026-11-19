---
loại: "quyết-định"
mã: "DECISION-STAGE03-MECHANISM-BEFORE-SCALE-107"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["V2_TARGETED_CONTINUAL_ALIGNMENT_CONTRACT"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-10-18"
---

# Stage 03 Scale-Up Decision Report: Refine the Mechanism Before Scaling Training

## Executive Summary

**Recommendation: choose Option A now — continue refining and auditing Stage 03 — and do not authorize a large-dataset/full training run yet.** Option B should be treated as a **conditional promotion state**, not the next default step.

This recommendation is not based on caution alone. The latest project evidence shows that the principal uncertainty is still **identifiability of the temporal mechanism**, not lack of optimization scale. The repaired `C_WRONG_YEAR` intervention now reaches the model—the latest diagnostic reports mean logit change of about `0.118655` and prediction changes on `4/36` eligible items—but that establishes only **treatment delivery / mechanism sensitivity**, not correct temporal causal reasoning. The same latest audit reports a four-way option-rotation diagnostic over 624 MCQA variants at about `24.20%` accuracy, with `488/624` predictions concentrated on `b`, and `0/156` source questions answered correctly under all four rotations. A tiny semantic-counterfactual diagnostic of five pairs produced `0/5` both-correct/appropriate-answer-flip cases. These are diagnostics, not proof of impossibility, but they are strong enough to block a positive causal claim and to make immediate scaling scientifically premature.

The older Run 05.2 aggregate result also does not justify scaling. Arm B scored `44/180 = 24.44%` while the Base/BOTH-OFF condition scored `58/180 = 32.22%`; the paired discordances were `b=37, c=51` with McNemar `p≈0.16544`. Thus the observed difference is statistically inconclusive, not evidence of equivalence and not conclusive evidence that LoRA is harmful. Moreover, only 120 of the 180 evaluation questions were `source_record_id`-nonoverlapping with the NWP training source under the available lineage fields; 60 had source overlap. On that 120-case slice, Arm B was `27/120 = 22.50%` versus Base `33/120 = 27.50%`.

The evaluator and mechanism audits have already discovered problems that could have made a much larger training run uninterpretable: the old `C_WRONG_YEAR` caller did not actually perturb the year on the 36 eligible cases; 144/180 cases are multi-year bypasses and therefore are not causal-treatment observations; historical DEV denominators were misreported as 60 instead of 482 for Relation and 40 instead of 120 for clean MCQA; a prior leakage check joined an empty/wrong `example_id` field; and historical “adapter off” logic sometimes removed temporal coordinates without truly disabling the LoRA residual.

The external literature reinforces the same decision. LoRA is explicitly defined around freezing pretrained weights and learning low-rank updates, and Hugging Face PEFT exposes explicit adapter activation/deactivation APIs precisely because “adapter loaded” and “adapter active” are distinct states. PEFT checkpoints normally contain adapter parameters rather than the base model. The current MaT-LoRA-related arXiv work at `2602.11965` makes temporal evolution itself part of the low-dimensional temporal core and reports frozen backbones and explicit time-dependent core dynamics; that makes direct `F_t` instrumentation central to validating a project derived from this idea. Ticktack injects temporal representation into the input representation and post-trains a temporal alignment objective, and its published experiments span models of roughly 3B–13B parameters—not the project's Qwen2.5-0.5B setup—so simply adding training scale to the present implementation is not an experimental reproduction of Ticktack.

### Decision at a glance

| Decision dimension | Continue Stage 03 refinement | Scale to full training now |
|---|---:|---:|
| Temporal treatment is delivered | **Partially established** | Does not fix causal ambiguity |
| Semantic temporal effect established | **No** | **No** |
| MCQA position/label bias resolved | **No** | Scaling may amplify it |
| Adapter ON/OFF implementation | Repaired, should be re-audited after every change | Higher cost of discovering a regression |
| Evaluator denominators/joins | Corrections exist; need canonicalization | Risk of expensive invalid result |
| Clean primary holdout | Not yet adequate for strong causal claim | Existing 180 should not be sole primary set |
| Current statistical power | Low for modest paired effects | More training does not increase evaluation power |
| Information gained per GPU unit | **High** | **Low until mechanism is identifiable** |
| Recommended status | **GO** | **NO-GO** |

**Decision rule:** authorize full training only after **all hard preflight gates** in the final section pass, followed by one small, preregistered, one-variable Stage 03 smoke experiment that produces a positive effect on a source-clean, semantic gold-flip counterfactual test without reintroducing label collapse or materially degrading the fixed NTP retention set.

Stage 01 and Stage 02 should **not be retrained** by default. Stage 01 needs report/protocol reconciliation only. Stage 02 should receive inference-only P12 reconstruction only if its claimed full-180 × three-seed raw generations, loader identity, and SHA lineage cannot be verified. That is consistent with the project's current audit policy.

## Evidence Base and Scientific Interpretation

### The key distinction is treatment delivery versus causal temporal reasoning

The project should explicitly separate four evidence levels:

| Evidence level | Scientific question | Current Stage 03 status |
|---|---|---|
| **Treatment validity** | Did changing the intended year actually change the encoded input? | **Now provisionally yes on eligible diagnostics** |
| **Mechanism sensitivity** | Did that treatment alter `F_t`, Seam output, hidden/logit state? | **Partially measured; insufficiently instrumented** |
| **Semantic causal validity** | Did changing time change the answer in the gold-consistent direction? | **Not established** |
| **Task improvement** | Does temporal conditioning beat a matched non-temporal control on held-out data? | **Not established** |

That hierarchy matters because the original Run 05.2 `C_TRUE == C_WRONG_YEAR` result could not be interpreted as temporal invariance: the intervention was not actually executed correctly for the 36 eligible examples. The later repair reporting `Δlogits≈0.118655` and four prediction flips on those 36 examples resolves the **delivery** problem, but a nonzero logit delta is not, by itself, evidence that the model understands the changed temporal facts.

The correct next causal experiment is therefore not another arbitrary `year + 37` perturbation. It is a **grounded counterfactual pair where the changed year changes the correct answer**. For example, a proposition whose truth status is demonstrably different at `t₁` and `t₂` should produce a correspondingly different gold answer. Each pair should have source provenance, a unique semantic gold under each year, and a stable non-temporal wording except for the temporal intervention. This converts the experiment from “does the feature influence logits?” to “does the time-conditioned mechanism causally track the time-dependent truth?”

This is much closer to what the temporal-alignment literature actually seeks to demonstrate. Ticktack represents year information through a sexagenary-cycle coordinate system, temporally encodes it into input representations, and then performs post-training alignment; its paper explicitly frames the goal as associating time points with relevant knowledge. The project therefore needs evidence at the *semantic* end of that chain, not just at the encoding end.

### Ticktack, FoNE, and MaT-LoRA do not imply that “more data” is the missing variable

There are important method-fidelity differences.

Ticktack's current arXiv v3 states that it uses a sexagenary representation, polar coordinates, temporal encoding injected into associated input embeddings, and a temporal alignment objective. It introduces TempLS spanning 75,000 BCE to 2025 and evaluates representative open-source LLMs in the 3B–13B range. The present project uses Qwen2.5-0.5B and a different mixture of gated Seam conditioning, project-specific Dual-cycle MaT-LoRA-style modules, and Stage 03 NTP/Relation/MCQA objectives. It should therefore be presented as a **derived experimental system**, not an exact Ticktack reproduction.

Likewise, the current arXiv version at `2602.11965` describes temporal adaptation through a shared low-dimensional adapter manifold with a structured time-varying core; its implementation section explicitly records time-dependent `F_t` dynamics, keeps backbone parameters frozen, and reports results over multiple temporal-domain benchmarks including Qwen3-0.6B and LLaMA-3-8B configurations. That is a strong reason to log the project's `F_t` directly rather than infer temporal activity from final accuracy alone.

FoNE is also narrower than the project's historical “add a Fourier vector” experiment. The FoNE paper replaces detected numbers with a special `[NUM]` token, adds FoNE to that token embedding, and introduces number-specific decoding machinery; its evaluations are primarily numerical/arithmetic tasks. Therefore Stage 01's negative additive-FoNE result should remain scoped to **the implementation tested by this project**. It is not evidence that Fourier number representations as a whole are invalid. The project's own Stage 01 audit already reaches that appropriately narrow conclusion and does not justify retraining the stage.

### Current evidence says “mechanism ambiguity,” not “dataset too small”

The newest project evidence points to three competing explanations that need to be distinguished before scale-up:

**Temporal conditioning may be too weak or too poorly positioned.** In Run 05.2, disabling the current Seam changed mean logits by only about `0.000369` and changed `0/180` discrete predictions, whereas disabling LoRA changed logits by about `1.754` and changed `88/180` predictions. This does not prove that “LoRA contains the temporal knowledge”; it establishes that, in the current implementation and checkpoint, the Seam at its tested placement has negligible measured leverage relative to the adapter path.

**The objective may be inducing response-position collapse rather than reasoning.** Arm B originally selected `c` for `127/156` MCQA examples and `No` for all 24 binary examples. After four cyclic option permutations, the latest diagnostic still reports a massive output-letter concentration (`b` on `488/624` variants) and zero source questions that remain correct under all four rotations. This is the signature of a problem that more examples *might* eventually reduce, but scaling before understanding whether it comes from target construction, exposure, decoding, loss balance, or adapter dynamics would conflate those hypotheses.

**The existing benchmark is not a clean causal test bed.** Sixty of 180 questions share `source_record_id` lineage with NWP training data under the forensic join, while 120 are source-nonoverlap under available identifiers. The contamination literature shows why such exposure must be reported: training/evaluation overlap can inflate benchmark performance. At the same time, controlled work also shows that source-only contamination does not behave identically to full source-plus-target contamination, so the project should call these cases **source-lineage overlap**, not automatically claim exact benchmark memorization without text-level evidence.

The scientifically efficient response is therefore to isolate these explanations with cheap interventions before increasing training scale.

## Evaluator, Adapter, and Dataset Integrity

### Evaluator integrity must become a hard gate

The denominator mistakes are not cosmetic. A result reported as `37/60` and the same model actually scoring `299/482` happen to produce similar-looking percentages in this case, but they imply completely different uncertainty, case composition, and pairing structure. The clean MCQA development denominator was likewise 120 rather than 40. A scientific result should therefore be generated from per-case records, never by copying a previously summarized denominator.

Every evaluation row should be reconstructible as:

```text
run_id
checkpoint_sha256
canonical_tensor_fingerprint
base_model_revision
dataset_sha256
split_sha256
example_id
source_record_id
event_id / temporal_cluster_id
protocol_id
prompt_template_sha
parser_sha
decode_config_sha
seed
gold
raw_generation
parsed_prediction
is_valid
is_correct
temporal_treatment_status
source_overlap_status
```

Aggregate metrics should then be computed from those rows, with hard assertions such as:

```text
n == len(unique(example_id))
correct == sum(is_correct)
invalid == sum(not is_valid)
sum(prediction_distribution.values()) == n
```

Deterministic paired evaluation is particularly important here. The official Qwen2.5-0.5B-Instruct generation configuration currently specifies sampling by default (`do_sample=true`, temperature `0.7`, top-p `0.8`, top-k `20`). A causal paired test that intends to attribute a prediction difference to temporal treatment should therefore explicitly override stochastic decoding—for example `do_sample=False`—or else preregister and replicate stochastic sampling with matched random streams. The Qwen model configuration also confirms `<|im_end|>`/EOS token ID `151645`, consistent with the project's repaired Stage 03 supervision contract.

Two preregistered evaluator views should be retained rather than retroactively choosing whichever produces the better result:

| Evaluator | Purpose | Scientific status |
|---|---|---|
| Strict generation + strict parser | End-to-end format and answer behavior | Primary behavioral outcome |
| Choice-logit scoring over valid labels | Separates semantic preference from generation-format failure | Secondary diagnostic |

The primary metric must be selected before the next smoke run. The secondary metric can explain failures, but should not silently replace the primary afterward.

### Adapter/LoRA correctness needs explicit execution traces

The original LoRA formulation freezes pretrained weights and adds trainable low-rank components. Hugging Face's current PEFT documentation similarly states that adapter checkpoints contain adapter state rather than the base model, that adapters can be explicitly disabled for base-model inference, and that loading an adapter is not necessarily the same as activating it.

For this project, that implies the following execution trace must be recorded for *every* causal condition:

```text
condition=C_TRUE
expected_adapter_modules=72
loaded_adapter_modules=72
enabled_adapter_modules=72
seam_enabled=true
temporal_feature_sha=...
F_t_digest_by_module=...
hook_positions=[...]
hook_count_during_forward=...
hook_count_after_forward=0
base_weight_sha_before=...
base_weight_sha_after=...
```

and for `C_LORA_OFF`:

```text
enabled_adapter_modules=0
```

rather than merely `coords_40d=None`.

The historical project bug where `use_adapter=False` omitted temporal coordinates but left the `DualCycleMaTLoRALinear` residual active is exactly why this distinction matters. Run 05.2's repaired BOTH-OFF/Base parity and compact adapter checkpoints are encouraging, but every architectural change in current Stage 03 can reintroduce the same class of bug.

### `F_t` should be promoted to a first-class experimental artifact

For every intended temporal module `m` and paired years `t₁,t₂`, store:

\[
\Delta F_m(t_1,t_2)
=
\left\|F^{(m)}_{t_1}-F^{(m)}_{t_2}\right\|_F
\]

alongside:

\[
\|\Delta W_m(t)\|_F,\qquad
\|h^\text{post}_m(t_1)-h^\text{post}_m(t_2)\|_2,
\qquad
\|\ell(t_1)-\ell(t_2)\|_2.
\]

The external temporal-core literature gives a clear methodological precedent for treating the temporal core itself as the object that evolves with time; the current `2602.11965` implementation description, for example, explicitly specifies time-dependent `F_t` dynamics and frozen backbones.

A useful diagnostic figure for the Stage 03 report would be a heatmap with **modules on the y-axis, counterfactual year-pairs on the x-axis, and `log10 ||ΔF_t||` as the cell value**. A second plot should show layerwise hidden-state/logit effect. These make it immediately visible whether conditioning is entering all 72 intended modules, dying in early layers, or being overwhelmed by a time-insensitive adapter component.

The current Seam should receive the same treatment. Compare the current token-14 position against the preregistered dynamic assistant-trigger location near `<|im_start|>assistant\n`, but call this an **inference sensitivity study**, not proof that the relocated Seam has been trained effectively. Only a subsequent matched smoke run can test efficacy.

### Dataset cleanliness should be defined at several levels

A single boolean `clean` is inadequate. Use at least:

```text
exact_example_overlap
normalized_text_overlap
source_record_overlap
event_overlap
temporal_cluster_overlap
unknown_lineage
```

The 60 Run 05.2 cases sharing source lineage should therefore remain a distinct diagnostic slice; the 120 others are `source_nonoverlap`, not automatically “blind,” because absence of a source identifier in another training source is not proof of non-exposure. Benchmark-contamination research likewise emphasizes that exposure can undermine evaluation validity and that contamination needs to be assessed at benchmark/sample level rather than assumed absent.

For the next decisive Stage 03 experiment, a **new primary holdout split grouped before splitting by source/event/time lineage** should be created. Legacy TempUN-180 remains useful as a secondary comparability benchmark, but it should no longer carry the entire burden of the causal claim.

## Statistical Power, Sample Size, and Compute Risk

The current Run 05.2 paired comparison has 88 discordant pairs out of 180 (`q≈0.489`) and an absolute paired difference of `14/180≈7.78` percentage points. McNemar power depends fundamentally on the discordant-pair probabilities, not simply total `n`.

Using the observed `q≈0.489`, the observed alternative split among discordants (`51/88` in one direction), and a two-sided exact conditional McNemar approximation, the present `n=180` has only about **30% power** for an effect of the size actually observed. That makes `p≈0.165` entirely unsurprising. It means the result is **underpowered and inconclusive**, not evidence that the two models are equivalent.

For planning, the first-order approximation:

\[
n
\approx
q\,
\frac{
(z_{1-\alpha/2}+z_{1-\beta})^2
}{
\delta^2
}
\]

with two-sided `α=.05`, 80% power, and `q=.489` yields approximately:

| Paired absolute effect to detect | Independent matched-pair `n`, approximate |
|---:|---:|
| 15 pp | 171 |
| 10 pp | 384 |
| 7.5 pp | 683 |
| 5 pp | 1,535 |
| Run 05.2 observed 7.78 pp | 635 |

Questions sharing a `source_record_id`, event, or source article violate the independent-pair assumption. Clustered matched-pair binary data explicitly require variance adjustment or cluster-aware tests when several paired observations belong to a common cluster.

The practical consequence is critical: **scaling the training set does nothing to repair an underpowered or clustered evaluation design.** Before the full run, choose a scientifically meaningful minimum detectable effect: default **10 percentage-point clean-set paired improvement**, requiring roughly 384 independent pairs at the current discordance rate before clustering inflation; or 7.5-point target requiring roughly 683.

### Compute-risk comparison

| Experiment | GPU / compute risk | Evidence produced | Expected decision value | Acceptance criterion |
|---|---|---|---|---|
| Artifact/hash/join audit | Negligible GPU | Provenance and valid denominators | **Very high** | All joins/hash contracts resolve |
| Re-evaluate existing checkpoints | Low | Correct Base/C1/A/B comparisons | **Very high** | Reproducible paired rows |
| 4-fold option permutations | Low inference cost | Label-position dependence | **Very high** | Collapse absent / quantified |
| `F_t` + activation tracing | Low–moderate inference overhead | Direct mechanism evidence | **Very high** | Treatment propagates through intended modules |
| Expand grounded counterfactual holdout | Primarily data/research cost | Actual causal temporal test | **Critical** | Unique gold changes with time |
| Seam-location inference ablation | Low | Sensitivity to injection site | High | Generation-adjacent site produces measurable pathway effect |
| One-variable smoke training, several seeds | Moderate | Does the proposed fix work? | **High after audits** | Beats matched static control without collapse |
| Full larger-dataset training | High | Final scale result | **Low right now** | Not authorized until all gates pass |

## Recommended Stage 03 Experimental Program

### Highest-priority inference-only experiments

| Priority | Experiment | Primary question | Expected useful outcome |
|---|---|---|---|
| **Immediate** | Base/C1/A/B deterministic re-evaluation on one fixed manifest | Are all comparisons actually paired? | Canonical scoreboard |
| **Immediate** | `C_TRUE`, corrected `C_WRONG_YEAR`, zero-feature, LoRA-off, Seam-off on eligible cases | Where does the causal effect enter/vanish? | Treatment → `F_t` → hidden → logits trace |
| **Immediate** | Four cyclic MCQA permutations | Is accuracy tied to answer position? | Position-invariant semantic accuracy |
| **Immediate** | Per-label precision/recall/confusion/entropy | Is one-letter collapse driving aggregate accuracy? | Quantified output prior |
| **Immediate** | Source-overlap versus source-nonoverlap slices | Is performance dependent on lineage exposure? | Separate generalization claims |
| **Next** | Grounded time-counterfactual pairs with gold flips | Does temporal conditioning alter answers correctly? | First real causal-semantic evidence |
| **Next** | Token-14 Seam versus assistant-trigger Seam | Is injection location suppressing leverage? | Architecture-localization evidence |
| **Next** | Fixed-token Base/Parent/A/B NTP CE/PPL | Is task adaptation causing retention loss? | Proper paired retention estimate |

For MCQA reporting, add at minimum macro recall, per-label recall, prediction entropy, position-specific accuracy, semantic permutation consistency, invalid-output rate, and source-clean slices.

### Only one change should enter the first training smoke test

The first intervention candidate is:

> **Online option permutation during MCQA training with correctly remapped gold labels**, while leaving architecture, temporal representation, NTP/Relation objectives, optimizer, checkpoint parent, and target-token budget fixed.

The smoke comparison:

```text
Control:
Parent + existing Stage03 objective/data
same optimizer steps
same supervised target-token count
same seed set
same clean split

Treatment:
Control + online MCQA option permutation only
```

Then evaluate both on the same unpermuted and rotated holdouts plus the grounded temporal-counterfactual set.

### Concrete preflight and evaluation commands

```powershell
# Contract and scientific-diagnostic unit tests
py -3.12 -m pytest `
  tests/test_run052_contract_invariants.py `
  tests/test_stage03_diagnostic_probes.py `
  -q

# Recompute Run 05.2 forensic metrics directly from raw artifacts
py -3.12 scripts/audit_run052_discrepancies.py

# Regenerate / verify Run 05.2 artifact hashes after derived audit outputs are finalized
py -3.12 scripts/generate_run05_2_sha256_manifest.py

# Project and knowledge-vault integrity
py -3.12 scripts/validate_project.py
py -3.12 scripts/validate_knowledge_vault.py

# Repository diff: raw receipts/checkpoints should not have been mutated
git status --short
git diff -- Reports configs scripts tests Documents
```

## Eight-Week Research Timeline

The plan starts on October 19, 2026, placing the full-run decision at the end:

- **Week 1 (Oct 19 - Oct 25):** Canonicalize Run 05.2 denominators, joins, SHAs.
- **Week 2 (Oct 26 - Nov 01):** Deterministic Base/C1/A/B re-evaluation.
- **Week 3 (Nov 02 - Nov 08):** $F_t$, Seam, LoRA ON/OFF and treatment instrumentation.
- **Week 4 (Nov 09 - Nov 15):** Expand grounded gold-flip counterfactuals & complete option rotations and label-bias analysis.
- **Week 5 (Nov 16 - Nov 22):** Cluster structure, power simulation, preregistration.
- **Week 6 (Nov 23 - Nov 29):** One-variable matched smoke experiment (Online Option Permutation).
- **Week 7 (Nov 30 - Dec 06):** Multi-seed clean-holdout replication and retention.
- **Week 8 (Dec 07 - Dec 13):** Hard-gate review and full-training GO or NO-GO milestone.

## Full-Training Promotion Gate (22-Item Checklist)

1. [ ] **Checkpoint identity:** parent SHA, adapter file SHA, canonical tensor fingerprint, model revision, tokenizer revision, dataset SHA, evaluator SHA, and git commit all resolve to real artifacts.
2. [ ] **Frozen backbone:** base-model weights are unchanged before/after the smoke optimizer run; optimizer groups contain no frozen `base_layer` tensors.
3. [ ] **Adapter inventory:** exactly the expected 72 temporal adapter modules are loaded; no missing/unexpected adapter keys; active/inactive state is recorded for every condition.
4. [ ] **Base parity:** `LoRA-OFF + Seam-OFF` reproduces Base logits within the preregistered numerical tolerance under an identical deterministic evaluator.
5. [ ] **Adapter sensitivity:** at least the designated positive-control inputs show nonzero LoRA-ON versus LoRA-OFF effects, proving that the enabled path is actually active.
6. [ ] **State hygiene:** temporal coordinates reset per example, Seam/forward hooks fire only where expected, and live hook count returns to zero after evaluation.
7. [ ] **Treatment proof:** every causal-eligible pair uses its *actual perturbed year* to recompute the temporal representation; true and counterfactual input features differ.
8. [ ] **`F_t` proof:** all intended temporal modules emit recorded `F_t`; the corrected counterfactual produces measurable `F_t` differences after the smoke stage rather than merely changing an upstream year variable.
9. [ ] **Causal denominator:** multi-year bypass cases are excluded from the treatment-effect denominator; treated, bypassed, ambiguous, and missing-year cases are reported separately.
10. [ ] **Evaluator determinism:** primary paired generation explicitly disables sampling or uses a preregistered matched-randomness design; generation configuration is hashed.
11. [ ] **Denominator integrity:** every score can be regenerated from unique case records; `n`, correct, invalid, prediction counts, and paired IDs reconcile exactly.
12. [ ] **Single-gold integrity:** every single-answer MCQA example has exactly one valid gold answer.
13. [ ] **Primary holdout lineage:** the decisive evaluation set has zero *known* source/event/time-group overlap with Stage 03 training; unknown lineage is not mislabeled clean.
14. [ ] **Duplicate audit:** normalized prompt/options/answer hashes and near-duplicate checks find no prohibited train/test duplicates in the primary holdout.
15. [ ] **Option-bias gate:** on the balanced four-rotation set, no answer letter accounts for more than 50% of predictions, every label has nonzero recall, and the range of position-specific accuracy is no more than 10 percentage points.
16. [ ] **Semantic permutation gate:** after inverse-mapping each rotation to semantic option identity, consistency improves materially over the current collapsed checkpoint and over the matched static control.
17. [ ] **Grounded temporal gate:** a source-clean counterfactual set exists where changing time changes gold semantics; arbitrary feature perturbations are not counted as semantic counterfactual evidence.
18. [ ] **Positive small-run result:** the temporal treatment beats a matched static/no-temporal LoRA control on the preregistered semantic counterfactual endpoint, with a source-cluster-aware 95% confidence interval excluding zero and a point improvement at least as large as the preregistered minimum meaningful effect.
19. [ ] **Power:** the primary paired sample has at least 80% planned power for the preregistered minimum effect after cluster dependence is included (approx. 384 independent pairs for 10 pp, 683 for 7.5 pp).
20. [ ] **Retention:** Base/Parent/Control/Treatment are evaluated on identical held-out NTP token positions (e.g. 5% relative-PPL non-inferiority margin).
21. [ ] **One-variable attribution:** the full run scales the intervention that passed the smoke study; it does not simultaneously introduce a new Seam location, new objective, smoothing, replay, new backbone, new sampler, and larger dataset.
22. [ ] **Matched exposure:** control and treatment receive matched optimizer steps and, more importantly, matched supervised-token/task exposure.

## Final Decision

**Project-lead verdict as of October 18, 2026: `STAGE03_CONTINUE_REFINEMENT / FULL_TRAINING_NO_GO`. Promotion to `FULL_TRAINING_GO` should occur only when the causal mechanism is measurable, the evaluator is trustworthy, the primary holdout is lineage-clean and sufficiently powered, the label-position shortcut has been controlled, and one matched small-scale experiment already demonstrates the intended semantic temporal effect.**
