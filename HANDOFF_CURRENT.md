# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-23 23:35 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** production-safe patient persistence + real 5-case pilot preparation; Calendar/Secretary feed temporarily deferred
> **Current module:** Module 01 — Osteoporosis

This file contains current operational truth only. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Baseline / patient persistence status

The prospective Baseline Audit Steps 1–6 are implemented. P1–P8, `labs_date`, Step-6 conflict clear-on-collapse, the explicit 14-scenario form smoke test, and the subsequent patient-persistence/laboratory-history browser smokes have passed.

Production persistence is confirmed on the live Render service:

```text
database_dialect=postgresql
database_url_configured=True
storage_mode=online_database
clinical_key_configured=True
```

Clinical authentication uses a Secure/HttpOnly/SameSite=Strict browser-session cookie. The patient-centric production layer supports:

```text
Patient
├── Encounters[]
└── LabSnapshots[]
```

PostgreSQL is the durable source of truth; browser localStorage is only a working cache. Patient search/load, encounter Save/reload and longitudinal lab snapshots have been browser-smoke-tested successfully.

Public repository rule remains absolute: no identifiable patient data, clinical exports or secret values may be committed.

---

## 2. Current production-hardening slice — encounter finalization integrity

PR #29:

```text
fix: preserve completed encounter finalization
branch: fix/encounter-finalization-integrity
```

Problem being closed: the browser adapter sends ordinary Save as `draft`, including after a completed encounter is reopened. Without a server guard, a completed record could silently regress from `completed` to `draft`.

New server rule:

```text
draft + Save              → draft
draft + Finish Visit      → completed
completed + no-op Save     → completed
completed + content change → amended
amended + later Save       → amended
```

A later content/date change after completion is therefore explicitly represented as an amendment rather than making the encounter look like an unfinished draft or untouched original completion.

Focused deterministic unit tests cover the transition contract. The repository currently has no configured CI status for this path; validation is scoped code/diff review plus a small live browser smoke after Render auto-deploy.

Required live smoke after merge/deploy:

1. load a synthetic completed encounter and press Save without changing content → status must remain `completed`;
2. change one synthetic field and Save → status must become `amended`;
3. reload/reopen → amended encounter remains loadable and still reports `amended`.

---

## 3. Clinical Calendar / Digital Secretary — intentionally paused

The Clinical Calendar foundation/navigation/osteoporosis-only filter are already merged and live. Current Calendar categories are:

```text
osteoporosis_first
osteoporosis_review
osteoporosis_unspecified
prolia
aclasta
```

The Digital Secretary remains the external integration owner for Setmore / Cal.com / Zadarma. Live Setmore appointments are **not yet being ingested** into `clinical_appointments`.

The product owner has explicitly deferred Calendar/Secretary changes for now. Do not create a parallel Secretary implementation path merely to keep Calendar work moving.

Current duration alignment remains:

```text
osteoporosis_first   60 min
osteoporosis_review  40 min
Aclasta              60 min
Prolia               10 min
```

Future Secretary work is expected to preserve a structured `visit_reason` from the Cal.com field `What is this meeting about?`, carry it to Setmore comment/notes, and then feed the Clinical Calendar. That work is deferred and is not a prerequisite for continuing independent Clinical Excellence programming.

Appointments remain distinct from CareTasks. Lab reminders, treatment due dates, results review and patient notifications must not be represented as fake appointments.

---

## 4. Pilot gate remains unchanged

The 5-case pilot uses real clinical encounters but is a usability/capture pilot, not a scored performance phase.

During the 5 pilot cases:

- no live KPI coaching/red-green performance feedback;
- safety-critical behavior may remain active;
- do not revise the Baseline form after each case unless there is a safety, data-loss or persistence defect;
- measure completion time, friction, missing/ambiguous fields and persistence behavior;
- after all 5, perform one deliberate refinement;
- then freeze form + KPI applicability/calculation rules before the 30-case scored baseline.

Do not implement the full §20 refinement backlog before the 5 pilot encounters.

---

## 5. Independent work that may continue while Calendar is deferred

Priority order remains:

```text
production data integrity / safety
→ real 5-case pilot
→ one evidence-driven post-pilot refinement
→ freeze Baseline Form + KPI contract
→ Core object schema v1 / standards & competency expansion
→ first dashboard data contract
→ broader Signal / Learning / Audit / Home implementation
```

Calendar/CareTask/Zadarma integration can resume later without blocking the above sequence. CareTask automation that depends on treatment due dates should remain separate from appointment ingestion and should not be coupled to the paused Calendar feed.

---

## 6. Exact next action

1. Review and merge PR #29 if clean.
2. Let Render auto-deploy; do not manually trigger deployment.
3. Run the three synthetic finalization smokes listed above.
4. If they pass, the persistence path is ready for the 5-case usability pilot from a data-integrity standpoint.
5. Continue independent Core/standards/data-contract programming without reopening Calendar/Secretary work until the product owner chooses to resume it.
