# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-25 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Current major phase:** Baseline/pilot integrity + Clinical Practice Review foundation.
> **Active phase plan:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — PR-1 Transcript Intake + Candidate Extraction v1.
> **ACTIVE WRITER/LOCK:** NONE after the canonical-control-plane update is merged.
> **RUNTIME IMPLEMENTATION:** PR-1 NOT STARTED.
> **CALENDAR / DIGITAL SECRETARY:** intentionally paused.

This file is the sole owner of operational **NOW**. Do not infer current mutation authority from `HANDOFF_CURRENT.md`, chat history or old PR text.

---

# 1. Current runtime foundation

Latest merged runtime hardening before this canonical update:

```text
PR #29 — fix: preserve completed encounter finalization
merge commit: 0a2147b8ae5fb8316bde16c8fbb4c0d96aba2194
```

Server lifecycle rule:

```text
draft + ordinary Save               → draft
draft + Finish Visit                → completed
completed + no-op Save              → completed
completed + content/date modification → amended
amended + later Save                → amended
```

The code-level transition contract has focused deterministic tests.

## Evidence still pending

The following live synthetic browser smoke has **not yet been recorded as passed in canonicals**:

1. load a synthetic `completed` encounter and Save without changing content → remains `completed`;
2. change one synthetic field/date and Save → becomes `amended`;
3. reload/reopen → remains `amended` and loadable.

Until this is completed, state must be described as:

```text
MERGED / DEPLOY PATH STARTED HISTORICALLY
!= LIVE FINALIZATION SMOKE VERIFIED
```

A fresh session should verify current Render deploy identity/status before claiming production verification.

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
- duplicate lab-history UI removed and `Νέες αναλύσεις` reset smoke passed.

Important caveat:

> Do not claim whole-service GDPR/privacy compliance. Legacy public `/osteoporosis/*`/CORS exposure remains a separate production-security concern.

---

# 3. Baseline methodology gate

The 5-case pilot remains next clinical-use gate after finalization integrity is smoke-verified.

```text
5 pilot encounters
→ one deliberate usability/data-contract refinement
→ freeze Baseline Form + KPI contract
→ 30 scored consecutive unique encounters
→ baseline lock
```

During the scored baseline:

- routine KPI coaching hidden;
- routine Practice Review coaching hidden;
- safety-critical exception path allowed;
- any intervention exposure must be recorded if methodology changes.

---

# 4. Active next engineering slice — PR-1

Active approved design:

```text
PR-1 — Transcript Intake + Candidate Extraction v1
```

Purpose:

> Paste a Heidi transcript and convert it into structured **candidate** clinical data aligned to the existing schema, without allowing the model to write authoritative patient data.

Frozen first-slice boundary:

```text
paste transcript
→ protected extraction endpoint
→ validated typed candidates
→ candidate preview
→ no authoritative encounter write
→ no raw transcript persistence
```

Full design/acceptance/REPLAN triggers live in `SLICE_PLAN_CURRENT.md`.

---

# 5. Practice Review program state

The broader Clinical Practice Review architecture is now a first-class product program in `CLINICAL_EXCELLENCE_PLAN.md` and `TODO.md`.

Planned stages:

```text
PR-1 transcript extraction candidates
PR-2 clinician review/accept/reject/edit + merge
PR-3 Quick Practice Review shadow mode
PR-4 Deep Review / Red Team / Decision Reconstruction
PR-5 longitudinal pattern → Signal integration
PR-6 intervention / Learning linkage
PR-7 adaptive consultation-flow presentation layer
PR-8 Patient Voice/outcome enrichment
```

Do not skip directly to automated coaching or consultation-flow rewrite before the data/provenance contracts exist.

---

# 6. Calendar / Digital Secretary state — paused

Already present:

- Clinical Calendar API/store/UI foundation;
- Baseline sidebar navigation;
- osteoporosis-only categories/filtering.

Not yet present/proven:

- live Setmore appointment ingestion;
- structured `visit_reason` feed;
- CareTask/Zadarma reminder workflow.

Product-owner decision: pause this track for now and continue independent Clinical Excellence programming.

Permanent rule:

```text
Appointment != CareTask
```

Do not modify the Digital Secretary as part of PR-1.

---

# 7. Exact next authorized actions

```text
1. verify current Render deploy/runtime identity for PR #29
2. run the three synthetic encounter-finalization smokes
3. if PASS, record the result in CURRENT_OPERATIONAL + changelog/TODO as appropriate
4. claim one writer lock for PR-1
5. inspect current auth/logging/UI/schema/model-integration seams
6. implement only the frozen PR-1 candidate-extraction boundary on a feature branch
7. use synthetic/de-identified transcript fixtures only
8. focused tests/review
9. PR → squash merge → auto-deploy
10. production smoke that does not insert real patient transcript into public logs/tests
```

If the finalization smoke fails, PR-1 does not outrank the data-integrity defect; fix/replan the integrity problem first.

---

# 8. Explicitly deferred / forbidden for the active slice

During PR-1 do not:

```text
AUTO-WRITE extracted transcript values into patient records
PERSIST raw transcript
LOG raw transcript
SHOW routine Practice Review coaching during scored baseline
CHANGE KPI scoring contract
REDESIGN entire Baseline form
MODIFY Digital Secretary / Setmore / Zadarma
CREATE fake Calendar appointments for CareTasks
BUILD polished Clinical Excellence Home before data contracts
COMMIT identifiable patient/transcript fixtures
```

---

# 9. Conversation handoff contract

Any session ending after a material step must update this file so a new conversation can resume without prior chat.

At minimum preserve:

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

If those cannot be reconstructed from this file plus `SLICE_PLAN_CURRENT.md`, the operational handoff is incomplete.
