# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-23 08:55 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** patient-centric persistence before real 5-case pilot
> **Current module:** Module 01 — Osteoporosis

This file contains current operational truth only. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Current baseline status

The prospective Baseline Audit Steps 1–6 are implemented. P1–P8, `labs_date`, Step-6 conflict clear-on-collapse, and the explicit 14-scenario browser smoke test have passed.

The baseline form itself is therefore functionally ready. The remaining blocker before real pilot use is no longer form correctness; it is **patient-centric durable persistence** so real pilot data are not entered twice or stranded in browser-only encounter storage.

Approved sequence:

```text
patient registry + protected DB persistence
→ patient encounter timeline + lab snapshots/history
→ persistence smoke test
→ 5 real pilot encounters
→ one deliberate post-pilot refinement
→ freeze Baseline Form v1 + KPI applicability/calculation contract
→ 30 consecutive scored baseline cases
```

---

## 2. Existing backend truth verified from current code

The existing application already has a SQLAlchemy persistence layer in the legacy backend:

```text
DATABASE_URL env when configured
fallback: sqlite:///./osteoporosis.db
AssessmentORM(patient_id, created_at, input_json, output_json, ...)
```

Existing legacy endpoints already support patient-based persistence/history:

```text
POST /osteoporosis/evaluate
PUT  /osteoporosis/assessment/{assessment_id}
GET  /osteoporosis/patient/{patient_id}/latest
GET  /osteoporosis/patient/{patient_id}/history
```

The legacy Cockpit UI already has `patient_id`, load-latest and load-history behavior.

Important clarification: the **new Baseline Audit UI** was still using localStorage only and was not connected to that database layer. The current implementation slice closes that gap without discarding the legacy backend.

Render workspace currently exposes no Render-managed Postgres instance through the connector. Therefore the deployed database target is determined by the service's `DATABASE_URL` environment setting; if absent, SQLAlchemy falls back to local SQLite. Do not assume a Render-managed Postgres database merely from the existence of SQLAlchemy code.

---

## 3. Patient-centric persistence implementation in progress

Branch:

```text
feat/patient-registry-backend
```

New protected clinical-data layer:

```text
clinical_patients
clinical_encounters
clinical_lab_snapshots
```

Model direction:

```text
Patient
├── Encounters[]
├── LabSnapshots[]
├── existing/legacy Assessments[]
└── later normalized DXA / Risk / Treatment objects
```

The existing large `main.py` implementation has been preserved byte-for-byte as `legacy_main.py`. A thin new `main.py` entrypoint imports the existing FastAPI `app` and `engine`, then composes the new routers. This is composition, not a rewrite of the legacy Cockpit.

New API surface:

```text
POST /clinical/login
POST /clinical/logout
GET  /clinical/status
POST /clinical/patients
GET  /clinical/patients?query=...
GET  /clinical/patient/{patient_id}
POST /clinical/patient/{patient_id}/encounters
GET  /clinical/patient/{patient_id}/encounters
GET  /clinical/encounter/{encounter_id}
PUT  /clinical/encounter/{encounter_id}
POST /clinical/patient/{patient_id}/labs
GET  /clinical/patient/{patient_id}/labs
PUT  /clinical/lab/{lab_snapshot_id}
```

Protection uses `CLINICAL_DATA_KEY` from Render environment. Login exchanges the key for a Secure/HttpOnly/SameSite=Strict cookie; the static client never embeds the env secret in source code.

`CLINICAL_DATA_KEY` has been configured on the Render service. Never commit or print its value in the public repository.

---

## 4. Baseline Audit patient-registry client

New runtime module:

```text
static/baseline-audit/patient-registry.js
```

It adds:

- protected login state;
- Patient ID search;
- creation/opening of a patient record;
- per-patient encounter timeline;
- load of a stored encounter back into the Baseline Audit UI;
- server sync of the complete Baseline Audit encounter payload on Save/Finish;
- patient-level laboratory snapshots keyed by actual laboratory date;
- comparative laboratory table across dates;
- update of an existing same-date/source-encounter lab snapshot instead of creating duplicates;
- localStorage retained only as a working/offline cache, not the durable source of truth.

The active patient is session-scoped in the browser. Server encounter linkage is cached locally only to support editing; the durable encounter remains in the database.

---

## 5. Immediate next action before merge

1. Review branch diff and import/startup behavior.
2. Open PR and merge if clean.
3. Let Render auto-deploy; do not manually trigger after merge.
4. Verify `/clinical/login` + `/clinical/status` on the live service.
5. Run a **patient-persistence smoke test** with synthetic/non-identifiable data:

```text
A. authenticate with clinical key
B. create/search Patient ID
C. create encounter → Save → server sync
D. reload page → search patient → load encounter → values restored
E. enter labs date 1 → Save
F. enter labs date 2 → Save
G. comparative lab table shows both dates
H. edit date-2 values → Save → same snapshot updates, no duplicate
I. Finish Visit → encounter status completed
J. reopen patient → completed encounter remains loadable
```

Only after this persistence smoke passes should real pilot data be entered.

---

## 6. Pilot rules unchanged

The 5-case pilot uses real clinical encounters but remains a **usability/capture pilot**, not a scored performance phase.

During the 5 pilot cases:
- no live KPI coaching/red-green performance feedback;
- safety-critical behavior may remain active;
- do not revise the form after each case unless there is a safety, data-loss or persistence defect;
- after all 5, perform one deliberate refinement;
- then freeze form + KPI applicability/calculation rules before the 30-case scored baseline.

Public GitHub rule remains absolute: no identifiable patient data, clinical exports or secrets are committed to the repository.
