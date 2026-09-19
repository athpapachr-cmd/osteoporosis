# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** LIVE GPT-5.6 PROMOTION GATE PASS 22/22 / FRESH INDEPENDENT READ-ONLY EVIDENCE REVIEW REQUIRED — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-08 deterministic head:** `5eeddeba664814b38d55bad8231afb5b33448eae`.
> **Deterministic/inherited gate:** `35446869081` — SUCCESS.
> **Focused PR-1 tests:** 60 PASS.
> **Frozen live suite:** 22 synthetic/de-identified cases.
> **Fourth live qualification:** `35446962808` — **22 PASS / 0 FAIL**.
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

## 10. Independent evidence review gate

Before any production-release engineering, a fresh independent READ-ONLY review must inspect the exact branch/evidence and determine whether the H-01..H-08 closure and 22/22 live result are sufficient to advance.

The reviewer may not mutate code/canonicals, merge/deploy, process identifiable transcripts, or start PR-2.

Only after that evidence review is checkpointed may work proceed to the separate production-release blockers:

- H-05 synchronous provider execution inside the async single-worker web process;
- executable browser lifecycle/BFCache/logout cleanup evidence.

These remain open even after 22/22.

Required sequence:

```text
22/22 live PASS
→ canonical evidence checkpoint
→ fresh independent READ-ONLY review
→ author checkpoint of disposition
→ H-05 + browser lifecycle release engineering
→ release-candidate review
```

## 11. Release boundary

A future 22/22 selected-model PASS still does not by itself authorize production. H-05 synchronous-provider blocking and executable browser lifecycle evidence remain production-release blockers/debt, and identifiable transcript processing retains its separate privacy/provider gate.

```text
current live promotion PASS: NO, pending third run
release ready: NO
release PR: NO
merge/deploy: NO
identifiable transcript use: NO
PR-2 / real pilot: NO
```
