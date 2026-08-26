# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh source main at correction start:** `f2ea2c81630f7c13cb1d7ea64fa462a62dafed0c`.
> **Current major phase:** Baseline/pilot integrity + Clinical Practice Review foundation.
> **Active phase plan:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — PR-1 Transcript Intake + Candidate Extraction v1, REPLAN-corrected design v3.
> **ACTIVE CANONICAL WRITER/LOCK:** ChatGPT — `docs/pr1-replan-v3-clinic-utilities` — documentation/design/roadmap correction only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **RUNTIME IMPLEMENTATION:** PR-1 NOT STARTED and NOT AUTHORIZED.
> **CALENDAR / DIGITAL SECRETARY:** intentionally paused.
> **CLINIC UTILITIES DETOUR:** approved for near-term planning, NOT ACTIVE; source websites not yet inspected.

This file is the sole owner of operational **NOW**. Do not infer mutation authority from chat history, `HANDOFF_CURRENT.md`, an old PR body or the roadmap alone.

---

# 1. Current runtime foundation

Latest merged runtime hardening before PR-1:

```text
PR #29 — fix: preserve completed encounter finalization
merge commit: 0a2147b8ae5fb8316bde16c8fbb4c0d96aba2194
```

Server lifecycle rule:

```text
draft + ordinary Save                 → draft
draft + Finish Visit                  → completed
completed + no-op Save                → completed
completed + content/date modification → amended
amended + later Save                  → amended
```

## Live synthetic browser smoke — PASSED 3/3

On 2026-08-26 the product owner confirmed:

1. `completed` + no-op Save remained `completed`;
2. material change + Save became `amended`;
3. reload/reopen remained `amended` and loadable.

Operational conclusion:

```text
PR #29 finalization integrity gate = CLOSED
```

`TODO.md` is corrected in the current docs branch so this completed gate is no longer shown as pending.

---

# 2. Baseline/patient persistence truth

Already proven before PR-1:

- Baseline Audit Steps 1–6 implemented;
- P1–P8 pre-pilot hardening implemented;
- `labs_date` + Step-6 conflict clear-on-collapse implemented;
- 14-scenario synthetic form smoke passed;
- PostgreSQL clinical storage configured and previously verified online;
- protected browser-session `/clinical/*` access implemented;
- Patient → Encounters[] + LabSnapshots[] persistence implemented;
- patient search/load + encounter save/reload browser smoke passed;
- longitudinal laboratory save/history browser smoke passed;
- duplicate lab-history UI removed and `Νέες αναλύσεις` reset smoke passed;
- encounter finalization browser smoke passed 3/3.

Important caveat:

> Do not claim whole-service GDPR/privacy compliance. Legacy public `/osteoporosis/*`/CORS exposure remains a separate production-security concern.

---

# 3. Product boundary — Personal Clinical Excellence, not osteoporosis audit only

The product target is a reusable **Personal Clinical Excellence System before, during and after the clinic**. Osteoporosis remains Module 01/proving ground.

Permanent implementation question:

```text
Is this reusable Clinical Excellence Core behavior?
or
Is this osteoporosis-specific Module 01 clinical content?
or
Is this cross-module Clinic Operations / Utility behavior?
```

For PR-1:

```text
CORE
transcript intake / validation / semantic candidate envelope
→ provider boundary
→ provenance / speaker / temporality / uncertainty
→ privacy / logging / failure behavior
→ module dispatch / candidate transport

MODULE 01
osteoporosis concept profile
→ fracture / risk / DXA / VFA / lab / treatment / task semantics
→ deterministic mapping to actual runtime targets
```

For the requested near-term utility detour:

```text
physiotherapy referral generator
radiofrequency request/PDF workflow
→ Clinic Utilities / Clinical Operations
→ not Osteoporosis Module 01 logic
```

---

# 4. Baseline methodology gate

Clinical-use sequence remains:

```text
5 pilot encounters
→ one deliberate usability/data-contract refinement
→ freeze Baseline Form + KPI contract
→ 30 scored consecutive unique encounters
→ baseline lock
```

During scored baseline:

- routine KPI coaching hidden;
- routine Practice Review coaching hidden;
- safety-critical exception path allowed;
- intervention exposure recorded if methodology changes.

PR-1 may be engineered/tested as capture infrastructure without activating routine coaching.

---

# 5. PR-1 pre-code review — REPLAN finding

A fresh read-only design-review conversation bootstrapped from `main` and inspected the actual runtime/schema seams. It found three material defects in design v2:

1. **schema/runtime target drift** — YAML names do not always equal the actual persisted browser payload paths;
2. **singular candidate under-modeling** — composite facts require `components[]` and potentially multiple deterministic target mappings;
3. **PHI-sensitive validation boundary** — standard validation/error paths must not be allowed to echo transcript input.

Disposition:

```text
REPLAN WITHIN PR-1
!= new roadmap phase
!= permission to implement yet
```

The current docs branch rewrites `SLICE_PLAN_CURRENT.md` as **v3** with the corrected contract.

---

# 6. PR-1 v3 corrected design — frozen direction pending final verification

Key corrections now canonical on the branch:

- candidate = composite semantic assertion with `components[]`;
- `target_mappings[]` are deterministic module output, never provider-authored;
- Module 01 mapping is versioned against `osteoporosis_runtime_targets_v1`, the actual persisted runtime namespace;
- unsupported/lossy concepts remain `ambiguous`/`unmapped` rather than being forced into unrelated fields;
- request contract has explicit body/transcript ceilings and forbids unnecessary patient identifiers;
- endpoint owns sanitized request validation/errors;
- provider adapter is isolated from Core/module business logic;
- no implicit SDK retries for transcript transmission;
- raw transcript/candidates remain ephemeral and content-free logs are enforced;
- browser cleanup includes `pagehide` and defensive `pageshow`/BFCache reset;
- deterministic tests and a synthetic Greek provider eval suite are required;
- exact provider model/SDK version is verified at implementation time rather than frozen as a permanent product invariant.

Runtime implementation remains:

```text
NOT STARTED
NOT AUTHORIZED
```

---

# 7. Exact next PR-1 action after this canonical correction closes

The next PR-1 conversation should be a **fresh final design verification**, not implementation.

It must:

```text
1. bootstrap fresh main + all six canonicals
2. verify v3 against actual current runtime paths
3. verify current official provider/API/SDK facts
4. identify remaining contradiction/REPLAN trigger, if any
5. present a compact final implementation contract
6. STOP and wait for explicit product-owner IMPLEMENT
```

Only after explicit `IMPLEMENT`:

```text
claim one runtime writer lock
→ create implementation branch
→ implement frozen PR-1 only
```

---

# 8. Near-term controlled detour — Clinic Utilities / Clinical Operations

The product owner has requested a small near-future detour to integrate two existing standalone clinic websites into the Clinical Excellence workspace.

## 8.1 Physiotherapy referral text generator

Desired outcome:

- inspect the existing source website first;
- preserve/refine referral-text generation;
- integrate into Clinical Excellence navigation/workspace;
- align styling with the shared Clinical Excellence visual system;
- determine persistence/linkage only from actual source/workflow needs.

## 8.2 Radiofrequency treatment request/PDF workflow

Desired outcome:

- inspect the existing source website first;
- preserve/refine PDF generation;
- align visual design with Clinical Excellence;
- add protected durable request tracking with minimum states:

```text
pending
approved_awaiting_application
completed
```

- provide list/filter views by state;
- preserve historical requests;
- allow **Repeat from previous** by cloning reusable fields from an old request into a **new request**, never rewriting the historical original;
- reconfirm/edit cloned values before resubmission;
- link to patient registry where appropriate and keep identifiable data out of the public repository.

No rejected/cancelled state is frozen until the actual workflow/source inspection demonstrates a need.

## 8.3 Detour activation rule

This is approved roadmap work but **not the active runtime slice today**.

When the product owner chooses to activate it:

```text
fresh canonical bootstrap
→ explicitly pause/switch active slice in CURRENT_OPERATIONAL
→ inspect both source sites read-only
→ classify reusable Core/Clinic Utility vs source-specific behavior
→ freeze one small implementation slice
→ claim one writer lock
→ implement
```

Do not overlap Clinic Utilities runtime mutation with an active PR-1 runtime writer.

The two source websites are not present in the current osteoporosis repository tree; their source location/files must be provided or located before design can be frozen.

---

# 9. Calendar / Digital Secretary state — paused

Already present:

- Clinical Calendar API/store/UI foundation;
- Baseline sidebar navigation;
- osteoporosis-only categories/filtering.

Not yet present/proven:

- live Setmore appointment ingestion;
- structured `visit_reason` feed;
- CareTask/Zadarma reminder workflow.

Permanent rule:

```text
Appointment != CareTask
```

Do not modify Digital Secretary as part of PR-1 or the current docs correction.

---

# 10. Explicitly forbidden during this canonical correction

```text
WRITE PR-1 runtime code
CREATE PR-1 implementation branch
AUTO-WRITE transcript candidates
PERSIST/LOG transcript content
MODIFY Calendar/Secretary runtime
IMPLEMENT Clinic Utilities before their source inspection + slice switch
COMMIT identifiable patient/utility-request data
```

---

# 11. Conversation handoff contract

At every material transition preserve:

```text
fresh main/source identity
active writer/branch/PR
what was designed
what was implemented
what was tested
what was merged/deployed
what remains unverified
current blocker/HOLD
exact next action
forbidden overlapping action
```

A fresh conversation must be able to reconstruct project truth from the six canonicals without relying on chat memory.
