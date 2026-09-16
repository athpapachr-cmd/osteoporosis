# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — H-07 BOUNDED REMEDIATION DETERMINISTICALLY PROVEN / THIRD LIVE 22-CASE GPT-5.6 QUALIFICATION AUTHORIZED — NOT RELEASE READY.
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
> **Previous live qualification:** `35140165842` — 7 PASS / 15 FAIL; evidence triaged and checkpointed.
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

## Exact next action

Deliberately trigger the bounded synthetic-only workflow against this verified H-07 branch state and execute the same frozen 22-case suite through `gpt-5.6`.

Promotion rule remains:

```text
22 frozen cases
→ strict Structured Outputs
→ default-deny oracle
→ failed=0 required
→ checkpoint exact live evidence
→ fresh independent READ-ONLY evidence review
```

If any case fails, checkpoint the exact coded evidence before further mutation. Do not tune repeatedly against hidden provider payloads or relax fixtures beyond the frozen H-07 contract without a new canonical finding.

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
