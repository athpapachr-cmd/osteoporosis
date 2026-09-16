# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — IMPLEMENTATION ACTIVE.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Bootstrap main:** `805c4fe4e723dacafc351ccf695ef01b66600079`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Activation branch:** `docs/pr1-transcript-capture-activation-2026-09-16`.
> **Runtime implementation branch:** not yet created; create only after this activation checkpoint is merged.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner explicitly instructed `Ok. Go. Ξεκίνησε` after fresh closeout identified Heidi-first capture / PR-1 as the next primary roadmap target. This authorizes bounded implementation of the frozen PR-1 extraction slice, including branch creation, code, tests/evals and implementation-candidate preparation.

This does **not** automatically authorize merge, production deployment, enabling identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutations.

## Fresh verification completed before activation

The archived corrected PR-1 v3 design was re-read from `docs/pr1-replan-v3-clinic-utilities` and checked against current `main`.

Current runtime still confirms the required target seams:

```text
encounter_archetype
anthropometrics.weight_kg / current_height_cm
fracture_history.events[]
risk_context.*
risk_assessment.*
step3.dxa / step3.vfa / step3.labs
step4.treatment_episodes[]
step4.administrations[]
step4.decision
step4.tasks[]
```

`patient-registry.js` still persists the complete active encounter object as the protected encounter `payload`, so deterministic mapping must target the actual browser/runtime namespace rather than YAML-only vocabulary.

Current OpenAI verification also supports the frozen provider pattern: Responses API structured parsing with Pydantic is available, GPT-5.6 is available through Responses API, and the current Python SDK still retries selected errors by default unless `max_retries=0` is set.

No material REPLAN blocker was found.

## Active implementation scope

Implement only PR-1:

```text
protected transcript paste/intake
→ explicit size + sanitized validation boundary
→ provider-neutral structured semantic extraction
→ deterministic osteoporosis target mapping
→ mapped / ambiguous / unmapped ephemeral candidate preview
→ no authoritative write
```

Required privacy boundary:

```text
raw transcript: ephemeral only
candidate preview: ephemeral only
DB write: none
localStorage/sessionStorage transcript/candidates: none
provider retries: disabled
identifiable transcript use: blocked until separate transcript-specific provider/privacy gate is explicitly closed
```

## Exact next action

After this activation checkpoint is merged:

1. create `feat/pr1-transcript-capture-v1-2026-09-16` from fresh `main`;
2. implement Core transcript contracts/router/service/provider abstraction;
3. implement Module-01 osteoporosis profile + deterministic target mapper;
4. add isolated ephemeral browser transcript UI;
5. add deterministic privacy/contract/mapping/UI tests and synthetic provider eval harness;
6. run exact-head regression gate;
7. checkpoint implementation/test evidence before opening any release PR.

## Explicitly deferred / forbidden in PR-1

- Accept/Edit/Reject-to-record or any authoritative patient write;
- PR-2 inline population/persistence;
- real identifiable Heidi transcript transmission before separate privacy/data-control approval;
- persistence of transcript, evidence snippets or candidates;
- provider-authored application target paths;
- exact-date invention from vague timing;
- collapsing option/recommendation/preference/final decision semantics;
- Practice Review coaching, KPI changes, treatment recommendation, pilot activation;
- unrelated Medical Report, Physio, RF, Calendar or Clinical Learning mutation.
