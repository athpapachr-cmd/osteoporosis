# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — H-09 DETERMINISTIC SEMANTIC BOUNDARY HARDENING PROVEN / NEW 22-CASE GPT-5.6 QUALIFICATION AUTHORIZED — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-08 deterministic head:** `5eeddeba664814b38d55bad8231afb5b33448eae`.
> **H-08 deterministic/inherited gate:** `35446869081` — SUCCESS.
> **H-09 exact deterministic head:** `00985996fe7c32a9596c5440d27223552d290d06`.
> **H-09 deterministic/inherited gate:** `35448153136` — SUCCESS.
> **Focused PR-1 tests:** 65 PASS.
> **Frozen synthetic qualification suite:** 22 synthetic/de-identified cases.
> **Fourth live qualification:** `35446962808` — **22 PASS / 0 FAIL**; secure synthetic-only boundary and frozen 22-case fixture count passed.
> **Safe credential/schema boundary:** CLOSED; Actions secret available/masked, PHI approval false, strict Structured Outputs accepted.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

The Product Owner authorized bounded PR-1 implementation and supplied the safe repository Actions credential prerequisite. Authority remains limited to synthetic qualification, bounded remediation and implementation-candidate preparation. It does not authorize production/PHI use, merge/deploy, PR-2 or real-patient pilot activity.

## Proven engineering baseline

H-01 through H-06 remain closed deterministically. Live run `35140165842` then executed all 22 cases and returned 7 PASS / 15 FAIL. That result was checkpointed at `5498f88ce8e9a67090863a7a5c9c272d3cced23e` and verified by `35140838915`.

A read-only H-07 triage classified those failures as:

```text
A provider semantic failures       5
B fixture/oracle overconstraints   4
C ontology/profile ambiguities     6
D demonstrated deterministic bugs  0
```

The triage contract was frozen at `71cb10a21551ad2da356990cc74a02153b7e3fb2` and verified by `35141347844`.

## H-07 bounded remediation — COMPLETE deterministically

The remediation is intentionally narrow. It changes provider-facing ontology/profile guidance, only four source-supported fixture permissions, and focused deterministic tests. It does not change endpoint behavior, mapper/evaluator logic, authoritative-write boundaries or production configuration.

### Provider profile hardening

The osteoporosis provider profile now explicitly defines exact provider-facing value kinds and code sets for the mapped concepts exercised by the promotion suite, including:

- fracture-site codes;
- treatment/administration/decision agent codes;
- treatment and administration statuses;
- decision types;
- patient acceptance codes;
- follow-up task codes with exact `DXA` casing and `followup_visit`;
- boolean/integer/number/quantity/date expectations.

It also freezes semantic ownership:

```text
actual/historical treatment → treatment.*
option discussed → decision.selected_agent + option_discussed
clinician recommendation → decision.selected_agent + clinician_recommendation
final selected plan → decision.selected_agent + final_decision
administration.* → only explicit actual/planned administration event
prescription/recommendation alone → never administration truth
uncertain administration occurrence → uncertain_needs_review, no administration status
third-party fact → speaker=third_party
explicit never/not received → polarity=negative, no positive treatment status
unrelated non-osteoporosis narrative → no osteoporosis task creation
```

### Four narrow fixture corrections only

The default-deny oracle remains intact. Only these source-supported extras are now allowed, with exact semantic/source/value/mapping constraints:

- `followup_vague`: `followup.task_type=followup_visit`;
- `followup_exact`: `followup.task_type=followup_visit`;
- `frax_original_adjusted`: `frax.tool_name=frax`;
- `out_of_range_runtime_values`: `frax.tool_name=frax`.

No wildcard permission and no blanket relaxation were introduced. Hard administration/source/negation cases remain fail-closed.

## Exact deterministic evidence

The H-07 remediation lifecycle contained two harness-only wording corrections before the gate could close. Neither changed provider profile, fixtures or runtime behavior. The final exact head is:

```text
e066c87bf6b42c2c56c80b2b71b43bc909510d39
```

Workflow `35142115023` passed:

- Python syntax — PASS;
- browser syntax — PASS;
- **56 focused PR-1 tests — PASS**;
- inherited protected-clinical regressions — 6 PASS;
- inherited Medical Report regressions — 24 PASS;
- inherited workspace/navigation regression — PASS;
- bounded PR-1 scope verification — PASS.

Therefore the bounded H-07 remediation is deterministically proven and the canonical barrier may advance to the next live qualification attempt.

## H-08 third live qualification — CHECKPOINTED

Workflow `35142415250` executed the same frozen 22-case synthetic/de-identified suite through `gpt-5.6`.

Boundary checks passed before provider execution:

- Actions `OPENAI_API_KEY` available and masked;
- `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false`;
- frozen fixture count = 22;
- strict Structured Outputs/provider schema path accepted.

Live result:

```text
18 PASS
4 FAIL
```

Exact coded failures:

```text
negative_history_vs_negative_investigation
  unexpected_assertion_vfa.modality

referral_not_completed_result
  unexpected_assertion_clinical.unmapped_narrative

prescription_not_administration
  missing_expected_concepts
  required_assertion_0_missing
  unexpected_assertion_decision.selected_agent

planned_not_done_administration
  unexpected_assertion_followup.task_type
```

Read-only triage against the frozen H-07 profile/fixtures found:

- three source-supported extras are fixture/oracle overconstraints: `vfa.modality=VFA`, `clinical.unmapped_narrative` for the explicit absence-of-result statement, and `followup.task_type=administration` for an explicitly scheduled administration;
- `prescription_not_administration` exposes an ontology/profile ambiguity: the profile generally assigns clinician recommendation to `decision.selected_agent`, while a later special clause also permits `treatment.agent`. Because `treatment.agent` maps to a treatment episode and `decision.selected_agent` remains ambiguous unless final_decision, the safer canonical recommendation representation is `clinician_recommendation + decision.selected_agent`.

No mapper/evaluator defect is demonstrated by these four coded failures.

## H-08 bounded remediation — COMPLETE deterministically

The bounded H-08 patch made only the authorized changes:

- exact source-supported permission for `vfa.modality=VFA` in `negative_history_vs_negative_investigation`;
- exact source-supported `clinical.unmapped_narrative` permission for the explicit no-result statement in `referral_not_completed_result`;
- exact `followup.task_type=administration` permission for the explicitly scheduled administration task;
- prescription/recommendation semantics canonicalized to `clinician_recommendation + decision.selected_agent`, with `treatment.agent` forbidden for prescription-only wording;
- focused H-08 regressions added; default-deny remains intact.

An initial deterministic run `35446828051` reached 59 PASS / 1 FAIL in the focused suite. The sole failure was a brittle pre-existing H-07 string assertion (`must NEVER` vs `and NEVER`) after the safety sentence was reworded without semantic relaxation. A harness-only assertion correction produced exact head:

```text
5eeddeba664814b38d55bad8231afb5b33448eae
```

Workflow `35446869081` completed SUCCESS with:

- Python syntax PASS;
- browser syntax PASS;
- **60 focused PR-1 tests PASS**;
- inherited protected-clinical regressions PASS;
- inherited Clinical Documents regressions PASS;
- inherited workspace/navigation regression PASS;
- bounded PR-1 scope PASS.

No mapper/evaluator relaxation, production configuration change or PHI processing occurred.

## Fourth live GPT-5.6 qualification — PASS

The workflow-only trigger head is:

```text
85d3a11ddff676a4569d1951af79aa2c28175e5d
```

The trigger-head deterministic gate `35446962951` completed SUCCESS across syntax, focused PR-1 tests, inherited clinical regressions, Clinical Documents regressions, navigation and bounded scope.

Live synthetic provider workflow `35446962808` then passed:

- synthetic-only execution boundary — PASS;
- provider/model declaration — `openai / gpt-5.6 / purpose=synthetic_eval / phi_approval=false`;
- frozen promotion fixtures — `total=22 ids_unique=true`;
- all 22 case-level oracle checks — PASS;
- final summary — `total=22 failed=0`.

Therefore the selected-model synthetic promotion gate is satisfied for the frozen 22-case suite.

This does **not** authorize production, PHI use, release PR, merge/deploy or PR-2.

## Independent promotion review — HOLD_FOR_SEMANTIC_REMEDIATION

A fresh independent READ-ONLY review verified the exact ancestry, deterministic chain and live GPT-5.6 evidence. It independently confirmed that workflow `35446962808` is genuine selected-model evidence with all 22 frozen cases PASS and `failed=0`.

The review did **not** invalidate that live result. It found one material deterministic-boundary defect before PR-1 semantic promotion can close:

- schema-valid provider output may still map `treatment.*` / `administration.*` into runtime targets under semantically incompatible states such as `clinician_recommendation`;
- a negated fracture/event presence assertion such as `polarity=negative + fracture.site=hip` can still become positive mapped runtime truth;
- the Module-01 target guard therefore does not yet fully enforce the provider-as-untrusted architecture.

Independent disposition:

```text
HOLD_FOR_SEMANTIC_REMEDIATION
```

Separate non-semantic release blockers remain unchanged: H-05 async/provider execution, executable browser lifecycle/BFCache/logout evidence, and the identifiable-transcript privacy/provider gate.

The review also noted two medium evaluator-hardening debts (unconstrained authorized free-text content and duplicate authorized candidate cardinality) and one low adapter diagnostic-classification weakness. These are recorded but are not the smallest blocking remediation action.

## H-09 bounded remediation contract

Authorized H-09 mutation is limited to the deterministic Module-01 target boundary plus focused adversarial tests:

1. `treatment.*` runtime mapping must fail closed when the candidate semantic type is not an actual/history treatment state;
2. `administration.*` runtime mapping must fail closed outside actual/history/objective/follow-up administration-event semantics;
3. negated fracture/treatment/administration presence/event assertions must not become positive mapped runtime truth;
4. existing valid scheduled-administration follow-up behavior must remain mapped;
5. no provider prompt/profile relaxation, evaluator/default-deny relaxation, endpoint behavior change, persistence change or production configuration change is authorized.

Use deterministic `ambiguous` mappings with explicit reason codes rather than silently discarding provider assertions.

## H-09 deterministic semantic boundary — COMPLETE deterministically

The bounded H-09 patch now enforces the previously missing local semantic boundary:

- all `treatment.*` episode concepts are mapped only for `patient_history_fact`; incompatible option/recommendation/decision semantics become `ambiguous` with `SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE`;
- all `administration.*` event concepts are mapped only for `patient_history_fact`, `objective_result` or `followup_task`; recommendation/option/decision semantics become `ambiguous` with `SEMANTIC_TYPE_NOT_ALLOWED_FOR_ADMINISTRATION_EVENT`;
- negated fracture presence, treatment episode and administration event concepts become `ambiguous` with `NEGATED_ASSERTION_NOT_POSITIVE_RUNTIME_VALUE`;
- legitimate actual treatment history, actual administration and scheduled follow-up administration remain mapped.

The first H-09 gate `35448114022` reached 64 focused PASS / 1 FAIL. The sole failure was a test-harness double-mapping error in the newly added scheduled-administration regression; runtime code was unchanged by the correction.

Final exact H-09 head:

```text
00985996fe7c32a9596c5440d27223552d290d06
```

Workflow `35448153136` completed SUCCESS:

- Python syntax PASS;
- browser syntax PASS;
- **65 focused PR-1 tests PASS**;
- 6 inherited protected-clinical tests PASS;
- 24 inherited Clinical Documents / Medical Report tests PASS;
- workspace/navigation PASS;
- bounded PR-1 scope PASS.

The independent HIGH deterministic semantic-boundary finding is therefore closed deterministically. This does not yet restore promotion PASS because runtime semantic code changed after the previously recorded 22/22 provider qualification.

## Exact next action

After this checkpoint itself verifies, rerun the same frozen 22-case synthetic/de-identified suite through `gpt-5.6` on the H-09 branch state.

Required sequence:

```text
H-09 canonical checkpoint
→ checkpoint verification PASS
→ workflow-only live trigger
→ frozen 22-case GPT-5.6 qualification
→ failed=0 required
→ canonical live evidence checkpoint
→ fresh independent promotion review
```

Do not begin H-05 or browser lifecycle remediation until the post-H-09 independent promotion review allows advancement.

## Explicitly blocked

Until a live 22-case qualification passes and its evidence receives fresh independent review:

- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no production credential/config mutation;
- no claim that deterministic H-07 closure equals live promotion PASS.

## Separate production-release blockers/debt

**H-05 remains OPEN:** synchronous provider execution inside the async single-worker web process must be remediated before production enablement/deploy.

Executable browser lifecycle/BFCache/logout cleanup evidence also remains release debt.
