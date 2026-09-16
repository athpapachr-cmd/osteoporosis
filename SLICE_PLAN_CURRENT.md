# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** LIVE GPT-5.6 QUALIFICATION 7/22 PASS; H-07 TRIAGE COMPLETE / BOUNDED PROFILE+FIXTURE REMEDIATION FROZEN — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-06 runtime/eval head:** `d1470ce1c10cd69f6f9e4fe72096527157f084e9`.
> **Live execution wrapper head:** `2714715ee2c14c6a477abd78562685ae1e7ecf0b`.
> **Live qualification:** `35140165842` — 22 executed / 7 PASS / 15 FAIL.
> **Live-failure checkpoint:** `5498f88ce8e9a67090863a7a5c9c272d3cced23e`; verification `35140838915` — SUCCESS.
> **H-07 triage:** 5 provider semantic failures / 4 fixture overconstraints / 6 ontology-profile ambiguities / 0 demonstrated deterministic defects.
> **Writer:** one bounded PR-1 implementation writer; operational owner is `CURRENT_OPERATIONAL.md`.

## 1. Objective

PR-1 accepts a pasted Heidi transcript and returns structured, non-authoritative candidates for immediate clinician review through strict Core contracts and deterministic Osteoporosis Module-01 mapping.

```text
PASTE HEIDI TRANSCRIPT
→ protected Core endpoint
→ ephemeral processing
→ strict semantic candidates
→ deterministic target mapping
→ transient preview
→ NO authoritative write
```

PR-1 does not own Accept/Edit/Reject-to-record, authoritative persistence, PR-2 or real-patient pilot activation.

## 2. Frozen invariants

- transcript/candidates remain ephemeral and non-authoritative;
- no DB/encounter/browser-storage/log persistence;
- provider cannot choose runtime/storage paths;
- Module-01 deterministic code owns mappings;
- every candidate remains `proposed` and clinician-review-required;
- preserve speaker/source, polarity, temporality, certainty and semantic type;
- preserve history vs objective result vs clinician interpretation;
- preserve option vs recommendation vs preference vs acceptance vs final decision;
- relative/vague timing never becomes invented exact date;
- identifiable transcript use remains behind a separate privacy/provider approval gate.

## 3. Proven engineering baseline

H-01..H-06 remain closed deterministically. The H-06 runtime head `d1470ce1c10cd69f6f9e4fe72096527157f084e9` passed full gate `35139716235`; canonical checkpoint verification `35140063802` passed. The live trigger head passed deterministic gate `35140165777`.

Run `35140165842` then reached the selected GPT-5.6 provider path with secure masked Actions credential, PHI approval false, accepted strict Structured Outputs schema and all 22 frozen synthetic/de-identified cases executed.

Result:

```text
7 PASS
15 FAIL
promotion threshold: failed=0
promotion gate: FAIL
```

## 4. H-07 triage disposition

The 15 failures are not one homogeneous model problem.

### A — provider semantic failures

```text
speaker_ambiguity
unrelated_general_clinical_text
prescription_not_administration
third_party_treatment_history
negated_treatment_exposure
```

These conflict with established scope/source/negation/administration distinctions. Remediation must strengthen the provider profile, never weaken deterministic guards or fixtures.

### B — fixture/default-deny overconstraints

```text
followup_vague
followup_exact
frax_original_adjusted
out_of_range_runtime_values
```

In these cases all frozen required assertions passed and the only failure was an additional source-supported concept:

- follow-up cases: `followup.task_type`;
- FRAX cases: `frax.tool_name`.

The permitted fixture correction is narrow and value-constrained:

```text
followup.task_type = followup_visit
frax.tool_name = frax
```

No wildcard or concept-only allow rule is allowed.

### C — ontology/provider-profile ambiguity

```text
fracture_relative_time
options_one_final
repeated_fracture_event_grouping
referral_not_completed_result
self_correction_date
planned_not_done_administration
```

The current profile enumerates concept keys but does not fully define provider-facing value kinds, code sets or concept ownership by semantic role. Live coded evidence proves the relevant concept family was returned but not the exact required contract. Because candidate payloads were intentionally not logged, hidden values/source fields must not be guessed retroactively.

### D — deterministic defect

No current evidence demonstrates a mapper/evaluator correctness defect when a provider assertion exactly satisfies the frozen contract.

## 5. Frozen H-07 remediation contract

### 5.1 Provider value contract

The provider profile must explicitly publish value kinds and code sets required by the current mapped ontology, including at minimum:

```text
fracture.site:
  kind=code
  vertebral | hip | distal_radius | proximal_humerus | pelvis | other

treatment.agent / administration.agent / decision.selected_agent:
  kind=code
  none | alendronate | risedronate | ibandronate_oral | zoledronate |
  ibandronate_iv | denosumab | teriparatide | romosozumab | raloxifene |
  hormone_therapy | other

treatment.status:
  kind=code
  planned | active | completed | stopped | holiday | unknown

administration.status:
  kind=code
  done | due | overdue | missed | planned | not_applicable

decision.type:
  kind=code
  start | continue | stop | switch | defer | no_drug_treatment |
  complete_course | consolidate | refer | uncertain

patient.acceptance:
  kind=code
  accepted | declined | undecided

followup.task_type:
  kind=code
  lab | DXA | administration | followup_visit | referral | VFA_or_imaging |
  adherence_check | exercise_or_falls | nutrition | other
```

The profile must also identify the expected value kind for booleans, dates, numeric/quantity fields and other concepts exercised by the 22-case suite.

### 5.2 Semantic ownership

Freeze these provider-side rules:

```text
historical/actual treatment episode → treatment.*
option discussed → decision.selected_agent + semantic_type=option_discussed
clinician recommendation → decision.selected_agent + semantic_type=clinician_recommendation
final selected treatment → decision.selected_agent + semantic_type=final_decision
administration.* → explicit actual/planned administration event only
prescription/recommendation alone → never administration.*
uncertain whether administration occurred → uncertain_needs_review; no administration status
unrelated non-osteoporosis narrative → clinical.unmapped_narrative only; no osteoporosis task creation
third-party fact → speaker=third_party
explicit never/not received → polarity=negative; do not infer positive treatment status
```

For an affirmed event/fact, polarity should be `positive` unless the source explicitly negates or leaves it unclear.

### 5.3 Four fixture corrections

Only these fixture relaxations are authorized:

- `followup_vague`: allow `followup_task / followup.task_type / code=followup_visit / speaker=clinician`;
- `followup_exact`: same narrow allow rule;
- `frax_original_adjusted`: allow `objective_result / frax.tool_name / code=frax / speaker=clinician` with mapped risk-assessment tool target;
- `out_of_range_runtime_values`: same narrow FRAX-tool allow rule.

All other failed cases must be addressed by provider-profile clarification, not fixture relaxation.

## 6. Required deterministic regressions

Before a third live run:

- provider-profile tests must assert the exact code/value contracts and semantic ownership rules above;
- fixture tests must assert only the four narrow allowlist additions;
- prescription/recommendation cases must continue to forbid administration truth;
- third-party and negated exposure cases must retain source/polarity guards;
- unrelated narrative must retain no mapped osteoporosis follow-up assertion expectation;
- all 52 existing focused tests plus new H-07 tests pass;
- inherited protected-clinical, Medical Report, navigation and scope gates pass.

Checkpoint exact remediation SHA/run before live execution.

## 7. Third live qualification gate

Only after deterministic checkpoint:

```text
frozen 22 cases
+ hardened provider profile
+ four narrow fixture corrections
+ secure synthetic credential
+ PHI approval=false
→ GPT-5.6 live qualification
→ failed=0 required
```

If failures remain, checkpoint exact coded evidence before any further mutation. Do not repeatedly tune against individual live outputs without a new canonical finding.

## 8. Release boundary

Even a future 22/22 provider PASS does not by itself authorize production. H-05 synchronous-provider blocking and executable browser lifecycle evidence remain release blockers/debt, and identifiable transcript use retains its separate privacy/provider gate.

```text
live provider promotion gate now: FAIL
release ready: NO
release PR: NO
merge/deploy: NO
identifiable transcript use: NO
PR-2 / real pilot: NO
```
