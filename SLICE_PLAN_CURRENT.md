# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** POST-H10 TWO-CASE TRIAGE COMPLETE / H-11 BOUNDED SOURCE-SEMANTIC NORMALIZATION AUTHORIZED — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-08 deterministic head:** `5eeddeba664814b38d55bad8231afb5b33448eae`.
> **Deterministic/inherited gate:** `35446869081` — SUCCESS.
> **H-09 exact deterministic head:** `00985996fe7c32a9596c5440d27223552d290d06`.
> **H-09 deterministic/inherited gate:** `35448153136` — SUCCESS.
> **H-10 exact deterministic head:** `eee45f9bab1fa0eba9fc36dea8fdcbf61dd89d5a`.
> **H-10 deterministic/inherited gate:** `35455400599` — SUCCESS.
> **Focused PR-1 tests:** 70 PASS.
> **Frozen live suite:** 22 synthetic/de-identified cases.
> **Pre-H09 live qualification:** `35446962808` — 22 PASS / 0 FAIL.
> **Post-H09 live qualification:** `35448273993` — 19 PASS / 3 FAIL.
> **Post-H10 live qualification:** `35455512992` — **20 PASS / 2 FAIL**.
> **Writer:** one bounded PR-1 implementation writer; operational owner is `CURRENT_OPERATIONAL.md`.

## 1. Objective

PR-1 accepts a pasted Heidi transcript and returns strict, structured, non-authoritative candidates for clinician review through Core contracts and deterministic Osteoporosis Module-01 mapping.

```text
HEIDI TRANSCRIPT
→ protected ephemeral extraction
→ semantic candidates
→ deterministic runtime mapping
→ transient clinician preview
→ NO authoritative write
```

PR-1 excludes authoritative persistence, Accept/Edit/Reject-to-record, PR-2 and real-patient pilot activation.

## 2. Frozen safety/privacy invariants

- raw transcript and candidates remain ephemeral/non-authoritative;
- no DB/encounter/browser-storage/log persistence;
- provider never chooses runtime/storage paths;
- deterministic Module-01 code owns mapping;
- candidates remain proposed + clinician-review-required;
- preserve speaker/source, polarity, temporality, certainty and semantic type;
- preserve history vs objective result vs interpretation;
- preserve option vs recommendation vs preference vs acceptance vs final decision;
- vague timing never becomes invented exact date;
- identifiable transcript use remains behind a separate privacy/provider approval gate.

## 3. Proven path through first live qualification

H-01..H-06 are closed deterministically. The safe Actions credential and strict Structured Outputs path are proven. The selected GPT-5.6 live run `35140165842` executed all 22 frozen synthetic/de-identified cases and returned:

```text
7 PASS
15 FAIL
```

That evidence was checkpointed and then triaged read-only before mutation.

## 4. H-07 read-only triage

The 15 failures were classified as:

```text
A PROVIDER_SEMANTIC_FAILURE       5
B FIXTURE_ORACLE_OVERCONSTRAINT  4
C ONTOLOGY_PROFILE_AMBIGUITY     6
D DETERMINISTIC_CONTRACT_DEFECT  0 demonstrated
```

The triage was frozen at `71cb10a21551ad2da356990cc74a02153b7e3fb2` and verified by workflow `35141347844`.

## 5. H-07 bounded remediation

### 5.1 Exact provider-facing value contracts

The provider profile now explicitly states value kinds and current code sets for concepts used by the promotion suite, including:

```text
fracture.site:
  code: vertebral | hip | distal_radius | proximal_humerus | pelvis | other

agents:
  none | alendronate | risedronate | ibandronate_oral | zoledronate |
  ibandronate_iv | denosumab | teriparatide | romosozumab | raloxifene |
  hormone_therapy | other

treatment.status:
  planned | active | completed | stopped | holiday | unknown

administration.status:
  done | due | overdue | missed | planned | not_applicable

decision.type:
  start | continue | stop | switch | defer | no_drug_treatment |
  complete_course | consolidate | refer | uncertain

patient.acceptance:
  accepted | declined | undecided

followup.task_type:
  lab | DXA | administration | followup_visit | referral | VFA_or_imaging |
  adherence_check | exercise_or_falls | nutrition | other
```

Exact value-kind guidance is also present for boolean, integer, number, quantity and date concepts.

### 5.2 Semantic ownership

Provider-side rules are explicit:

```text
historical/actual treatment exposure → treatment.*
option discussed → decision.selected_agent + option_discussed
recommendation → decision.selected_agent + clinician_recommendation
final selected treatment → decision.selected_agent + final_decision
administration.* → explicit actual/planned administration event only
prescription/recommendation alone → no administration truth
uncertain administration → uncertain_needs_review, no administration status
third-party treatment → speaker=third_party
patient never received agent → polarity=negative, no positive status
unrelated non-osteoporosis narrative → no osteoporosis task creation
```

### 5.3 Default-deny fixture policy preserved

Only four narrowly demonstrated, source-supported additions were made:

- `followup_vague` may additionally return exact `followup.task_type=followup_visit`;
- `followup_exact` may additionally return exact `followup.task_type=followup_visit`;
- `frax_original_adjusted` may additionally return exact `frax.tool_name=frax`;
- `out_of_range_runtime_values` may additionally return exact `frax.tool_name=frax`.

These rules constrain semantic type, speaker, value kind/code and deterministic mapping. No wildcard permission was added.

Hard A-case protections remain frozen, including:

- no administration truth from prescription/recommendation;
- no positive treatment truth from explicit negation;
- third-party attribution preserved;
- unrelated narrative does not create osteoporosis task truth;
- uncertain administration remains non-authoritative/uncertain.

## 6. Deterministic acceptance evidence

The exact H-07 remediation head is:

```text
e066c87bf6b42c2c56c80b2b71b43bc909510d39
```

Workflow `35142115023` completed SUCCESS with:

- Python syntax PASS;
- browser syntax PASS;
- **56 focused PR-1 tests PASS**;
- 6 inherited protected-clinical tests PASS;
- 24 inherited Medical Report tests PASS;
- workspace/navigation PASS;
- bounded PR-1 scope PASS.

Two preceding failed deterministic attempts were harness-only string-matching findings in the new H-07 profile tests. The corrections changed only test wording and did not mutate provider-facing profile, fixtures or runtime behavior.

Therefore H-07 bounded remediation is now deterministically proven.

## 7. Third live qualification evidence

Workflow `35142415250` passed the secure synthetic-only boundary and fixture-count checks, then executed all 22 cases through `gpt-5.6`.

```text
18 PASS
4 FAIL
```

The four coded failures were:

- `negative_history_vs_negative_investigation`: unexpected `vfa.modality`;
- `referral_not_completed_result`: unexpected `clinical.unmapped_narrative`;
- `prescription_not_administration`: expected recommendation concept missing and `decision.selected_agent` returned instead;
- `planned_not_done_administration`: unexpected `followup.task_type`.

Read-only source/profile triage classifies the first, second and fourth as narrow fixture/oracle overconstraints because the extras are directly source-supported and already valid under the provider ontology. The prescription case instead demonstrates conflicting provider guidance: general recommendation ownership points to `decision.selected_agent`, while a later special clause also permits `treatment.agent`.

For safety, recommendation/prescription semantics should converge on `decision.selected_agent`, because it remains non-authoritative/ambiguous unless the candidate is an explicit `final_decision`; `treatment.agent` targets a treatment episode.

## 8. H-08 bounded remediation — deterministic PASS

The authorized H-08 changes are now implemented:

- exact `vfa.modality=VFA` allowance;
- exact clinician `clinical.unmapped_narrative` allowance for the explicit absence-of-result statement;
- exact `followup.task_type=administration` allowance for the scheduled administration task;
- prescription/recommendation canonicalized to `clinician_recommendation + decision.selected_agent`;
- prescription-only `treatment.agent` blocked;
- focused H-08 regressions added without weakening default-deny.

The first run after implementation, `35446828051`, had one harness-only H-07 wording assertion failure after 59 other focused tests passed. No provider-facing or fixture behavior changed in the correction.

Final exact deterministic head:

```text
5eeddeba664814b38d55bad8231afb5b33448eae
```

Workflow `35446869081` completed SUCCESS:

- Python syntax PASS;
- browser syntax PASS;
- **60 focused PR-1 tests PASS**;
- 6 inherited protected-clinical tests PASS;
- 24 inherited Medical Report tests PASS;
- workspace/navigation PASS;
- bounded scope PASS.

Therefore H-08 is deterministically proven.

## 9. Fourth live qualification — PASS

Workflow-only trigger head:

```text
85d3a11ddff676a4569d1951af79aa2c28175e5d
```

Trigger-head deterministic workflow `35446962951` completed SUCCESS.

Live synthetic qualification workflow `35446962808` passed the secure boundary and frozen fixture-count checks, then returned:

```text
22 PASS
0 FAIL
provider-eval summary: total=22 failed=0
```

All 22 frozen case IDs passed under strict Structured Outputs and the default-deny oracle. The provider declaration remained `openai / gpt-5.6 / synthetic_eval / phi_approval=false`.

The selected-model synthetic promotion gate is therefore satisfied.

## 10. Independent promotion review — HOLD

The fresh independent READ-ONLY review verified:

- clean branch ancestry from current `main`;
- H-08 deterministic head and all exact workflow evidence;
- genuine live `gpt-5.6` execution with synthetic-only purpose and `phi_approval=false`;
- all 22 frozen case IDs PASS and `failed=0`;
- bounded H-07/H-08 fixture changes rather than blanket default-deny relaxation.

It nevertheless found a HIGH deterministic semantic-boundary defect: known runtime concepts can still map under incompatible semantic/polarity states because `transcript_target_guard.py` does not fully enforce the provider profile's semantic ownership.

Examples that must fail closed include:

```text
clinician_recommendation + treatment.agent
clinician_recommendation + administration.status=done
negative polarity + fracture.site=hip
```

Disposition:

```text
HOLD_FOR_SEMANTIC_REMEDIATION
```

## 11. H-09 deterministic semantic boundary — PASS

H-09 adds local fail-closed enforcement at the Module-01 target boundary rather than relying on provider prompt obedience.

Deterministic ownership now is:

```text
treatment.* runtime truth
  → patient_history_fact only

administration.* runtime truth
  → patient_history_fact | objective_result | followup_task only

negative polarity
  + fracture presence / treatment episode / administration event concept
  → ambiguous, never positive mapped runtime truth
```

Incompatible semantic combinations remain visible for clinician review as `ambiguous` with explicit reason codes.

Focused adversarial coverage proves at minimum:

- clinician recommendation cannot create `treatment.agent` runtime truth;
- option semantics cannot create `treatment.status` runtime truth;
- recommendation/final-decision semantics cannot create `administration.*` truth;
- negated fracture/treatment/administration presence fails closed;
- valid historical treatment remains mapped;
- valid actual administration remains mapped;
- valid planned follow-up administration remains mapped.

The first run `35448114022` had one new-test harness defect after 64 other focused tests passed. A test-only correction yielded final exact head:

```text
00985996fe7c32a9596c5440d27223552d290d06
```

Workflow `35448153136` completed SUCCESS with **65 focused tests**, all inherited protected-clinical and Clinical Documents regressions, navigation and scope guard PASS.

Medium independent-review debts remain recorded and unchanged:

- authorized free-text content is not value-constrained by the promotion oracle;
- duplicate authorized candidates are not globally cardinality-limited;
- unknown adapter exceptions have broad availability classification.

These were not part of the blocking HIGH H-09 patch.

## 12. Post-H09 live qualification — FAIL CHECKPOINT

The workflow-only trigger head was:

```text
d2c878cc036b656b273a3684d3726c2fb8ae8353
```

Trigger-head deterministic gate `35448274111` completed SUCCESS.

Live synthetic run `35448273993` passed the secure synthetic-only boundary and frozen fixture-count checks, then returned:

```text
19 PASS
3 FAIL
provider-eval summary: total=22 failed=3
```

Failures:

```text
speaker_ambiguity
  required_assertion_0_missing
  unexpected_assertion_treatment.agent

unrelated_general_clinical_text
  unexpected_assertion_clinical.unmapped_narrative
  semantic_count_clinician_recommendation_mismatch

referral_not_completed_result
  unexpected_assertion_clinical.unmapped_narrative
```

The post-H09 provider call itself was healthy; this is semantic/evaluator evidence, not infrastructure failure.

The prior pre-H09 22/22 remains valid historical evidence but cannot substitute for post-runtime-change qualification.

## 13. Post-H09 three-case triage

The failure checkpoint was verified by workflow `35448460778` before triage.

Classification:

```text
speaker_ambiguity
  → STALE FIXTURE MAPPING AFTER H-09

unrelated_general_clinical_text
  → PROVIDER/PROFILE SEMANTIC AMBIGUITY

referral_not_completed_result
  → PROVIDER/PROFILE OPTIONAL-NARRATIVE METADATA AMBIGUITY
```

The H-09 target guard itself is not implicated by the latter two failures.

H-10 must keep the existing safety meaning:

- uncertain treatment mention is review-only/ambiguous, not a mapped episode;
- unrelated generic clinician deferral does not create recommendation/task truth;
- DXA referral remains a follow-up task and “no result yet” never becomes completed-result truth.

Because the two narrative cases intersect the independent evaluator findings, H-10 also closes:

- blank/unbound evidence authorization for optional free-text assertions;
- exact duplicate authorized candidate false-PASS behavior.

## 14. H-10 bounded remediation

Authorized files/behavior may cover provider profile, the exact affected fixtures, promotion evaluator and focused tests only.

Required contracts:

```text
speaker_ambiguity treatment.agent
  → ambiguous / SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE

generic unrelated clinician deferral
  → no clinician_recommendation
  → no osteoporosis followup_task

DXA referral + no result yet
  → followup_task for referral
  → optional clinician_interpretation + clinical.unmapped_narrative only
  → source-bound evidence
  → no DXA result concept

authorized free-text allowance
  → evidence_contains must match candidate evidence snippet

exact duplicate candidate
  → promotion failure
```

Default-deny remains intact. No H-09 guard relaxation and no wildcard allowances.

## 15. H-10 deterministic acceptance — PASS

H-10 closes the triaged stale fixture/profile/evaluator gaps without relaxing H-09.

Implemented contracts:

- uncertain treatment mention remains `uncertain_needs_review` but maps `ambiguous`, never to a treatment episode;
- generic unrelated clinician deferral does not become `clinician_recommendation` or osteoporosis `followup_task`;
- optional no-result narrative is clinician interpretation only and source-bound;
- optional free-text narrative authorization requires `evidence_contains` match against the candidate evidence snippet;
- exact duplicate provider candidates fail promotion.

The first H-10 run `35455364544` had one test-only H-08 expected-dict mismatch after 69 focused PASS. A test-only correction produced:

```text
eee45f9bab1fa0eba9fc36dea8fdcbf61dd89d5a
```

Workflow `35455400599` completed SUCCESS with:

```text
70 focused PASS
6 protected-clinical PASS
24 Clinical Documents / Medical Report PASS
navigation PASS
bounded scope PASS
```

## 16. Post-H10 live qualification — FAIL CHECKPOINT

Trigger head `9d77df8569a0037ec51204e6724a0c62c4bcc552`.

Trigger-head deterministic gate `35455512998` completed SUCCESS.

Live run `35455512992` passed the synthetic-only/provider/fixture-count boundaries, then returned:

```text
20 PASS
2 FAIL
provider-eval summary: total=22 failed=2
```

Failures:

```text
negative_history_vs_negative_investigation
  unexpected_assertion_vfa.action

prescription_not_administration
  unexpected_assertion_clinical.unmapped_narrative
```

## 17. Post-H10 two-case triage

The failure checkpoint was verified by `35455697350`.

Disposition:

```text
negative_history_vs_negative_investigation
  source says VFA result exists/is referenced
  safest action = already_available_reviewed
  do not infer current-encounter performed

prescription_not_administration
  "no administration is mentioned"
  = absence-of-documentation narrative
  != administration event
  != negative administration truth
```

## 18. H-11 bounded normalization

Authorized changes:

- provider rule for referenced VFA result → `vfa.action=already_available_reviewed` unless explicit performance;
- exact source-bound `vfa.action` allowance in the VFA case;
- provider rule for no-administration-mentioned wording → `clinician_interpretation + clinical.unmapped_narrative`;
- exact source-bound narrative allowance in the prescription case;
- focused deterministic regressions.

H-09 target guards and H-10 source-bound/default-deny/duplicate protections remain unchanged.

Required sequence:

```text
H-11 patch
→ full deterministic/inherited gate
→ canonical checkpoint
→ live 22-case GPT-5.6 qualification
→ fresh independent promotion review
```

H-05 and browser lifecycle remediation remain blocked.

## 19. Release boundary

A future 22/22 selected-model PASS still does not by itself authorize production. H-05 synchronous-provider blocking and executable browser lifecycle evidence remain production-release blockers/debt, and identifiable transcript processing retains its separate privacy/provider gate.

```text
current live promotion PASS: YES, 22/22 on run 35446962808; independent review HOLD_FOR_SEMANTIC_REMEDIATION
release ready: NO
release PR: NO
merge/deploy: NO
identifiable transcript use: NO
PR-2 / real pilot: NO
```
