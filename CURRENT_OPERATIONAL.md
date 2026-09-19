# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — H-08 THIRD LIVE QUALIFICATION CHECKPOINTED AT 18/22 / BOUNDED FOUR-CASE REMEDIATION AUTHORIZED — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-07 deterministic head:** `e066c87bf6b42c2c56c80b2b71b43bc909510d39`.
> **H-07 deterministic/inherited gate:** `35142115023` — SUCCESS.
> **Focused PR-1 tests:** 56 PASS.
> **Frozen synthetic qualification suite:** 22 synthetic/de-identified cases.
> **Third live qualification:** `35142415250` — 18 PASS / 4 FAIL; credential/schema/fixture-count boundaries passed; exact coded failures checkpointed.
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

## Exact next action

Perform one bounded H-08 remediation only:

1. allow the three exact source-supported extras above with full semantic/source/value constraints; no wildcard/default-deny relaxation;
2. remove the contradictory recommendation/prescription `treatment.agent` permission and make clinician recommendation use `decision.selected_agent`;
3. update the `prescription_not_administration` fixture to require `clinician_recommendation + decision.selected_agent=denosumab`, while retaining the hard prohibition on administration truth;
4. add focused deterministic regressions for all four H-08 findings;
5. run the full PR-1 deterministic/inherited gate and checkpoint the exact SHA/run before any fourth live provider qualification.

After deterministic PASS, deliberately trigger the same frozen 22-case suite through `gpt-5.6`. Promotion still requires `failed=0` and then a fresh independent READ-ONLY evidence review.

Do not begin H-05 remediation or browser lifecycle work until this semantic/provider gate is resolved.

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
