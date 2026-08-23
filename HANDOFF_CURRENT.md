# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-23 10:46 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** patient-centric production persistence + Clinical Calendar integration before real 5-case pilot
> **Current module:** Module 01 — Osteoporosis

This file contains current operational truth only. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Baseline / patient persistence status

The prospective Baseline Audit Steps 1–6 are implemented. P1–P8, `labs_date`, Step-6 conflict clear-on-collapse, the explicit 14-scenario form smoke test, and the subsequent patient-persistence/laboratory-history smoke tests have passed.

Production persistence is confirmed on the live Render service:

```text
database_dialect=postgresql
database_url_configured=True
storage_mode=online_database
clinical_key_configured=True
```

Clinical authentication now uses a Secure/HttpOnly/SameSite=Strict **browser-session cookie**; there is no fixed 12-hour Max-Age.

The patient-centric production layer now supports:

```text
Patient
├── Encounters[]
├── LabSnapshots[]
├── existing/legacy Assessments[]
└── later normalized DXA / Risk / Treatment objects
```

Verified by browser smoke testing:

- login with `CLINICAL_DATA_KEY`;
- create/search Patient ID;
- Save encounter to PostgreSQL;
- reload/search/load and restore values;
- save multiple dated laboratory snapshots;
- comparative laboratory table displays historical dates;
- `Νέες αναλύσεις` clears only current lab-entry fields and leaves historical snapshots intact;
- duplicate laboratory table was removed from Patient Registry and remains only in Step 3.

Public repository rule remains absolute: no identifiable patient data, clinical exports or secret values may be committed.

---

## 2. Digital Secretary integration truth

The existing Digital Secretary repository is `athpapachr-cmd/ortho-reception-backend-v2`.

Verified existing infrastructure:

- Setmore OAuth/client integration and appointment listing helpers;
- persistent Setmore refresh-token file on Render disk;
- Cal.com ↔ Setmore synchronization via Render cron;
- Setmore patient cache and PostgreSQL patient directory;
- Zadarma API integration, including SMS send support;
- `SYNC_ADMIN_TOKEN`-protected synchronization workflow.

Canonical ownership decision:

```text
Setmore / Cal.com / Zadarma
        ↓
Digital Secretary backend = external integration owner
        ↓
Clinical Excellence / Osteoporosis = clinical meaning, appointment classification, CareTasks, reminders
```

Setmore is the booking source of truth for the Clinical Calendar. Cal.com remains the availability/scheduling-support source and must not become the clinical appointment store.

Known duration contract from the product owner:

```text
osteoporosis_first      60 min
osteoporosis_review     60 min
Aclasta                 60 min
Prolia                  10 min
```

The Digital Secretary currently has a legacy 40-minute osteoporosis-follow-up duration in its appointment-category map. Correct this in the integration slice before relying on category-duration consistency.

---

## 3. Clinical Calendar v1 — active implementation

Active branch:

```text
feat/clinical-calendar-v1
```

New schema/data contract:

```text
schemas/clinical_calendar_contract_v1.yaml
```

New server module:

```text
clinical_calendar.py
```

New database table:

```text
clinical_appointments
```

Normalized appointment categories:

```text
osteoporosis_first
osteoporosis_review
osteoporosis_unspecified
prolia
aclasta
other
```

Important classification rule: duration alone must **not** distinguish first from review because both are 60 minutes. Explicit service/label/comment semantics outrank duration; ambiguous osteoporosis visits stay `osteoporosis_unspecified`.

New protected API surface:

```text
GET  /clinical/calendar/appointments?start=...&end=...
POST /clinical/calendar/appointments/import
```

Clinician reads use the existing browser-session clinical authentication. External server-to-server appointment ingest is separately protected by `CLINICAL_INGEST_KEY`; its value must exist only in environment configuration, never in source.

New UI:

```text
/static/clinical-calendar/
```

The first UI slice provides:

- previous/current/next week navigation;
- daily appointment columns;
- categories for osteoporosis / Prolia / Aclasta;
- weekly counts;
- explicit `osteoporosis_unspecified` count for appointments needing classification;
- link back to Baseline Audit;
- Calendar link from Patient Registry.

The Calendar deliberately does **not** represent laboratory reminders or treatment follow-up tasks as appointments.

---

## 4. Appointment vs CareTask — frozen separation

```text
Appointment
= scheduled attendance

CareTask
= clinical action that can exist with or without an appointment
```

Examples of future CareTasks:

- pre-Prolia / pre-Aclasta laboratory tests due;
- post-treatment monitoring due;
- results review;
- patient notification;
- Prolia/Aclasta administration due or overdue;
- next clinical review required.

CareTasks will later drive reminder logic and Zadarma SMS; they must not be inserted as fake calendar appointments.

---

## 5. Exact next implementation action

1. Review and merge Clinical Calendar v1 if clean.
2. Let Render auto-deploy; do not manually trigger.
3. Smoke the empty/normalized calendar UI and protected endpoints.
4. Build the **Digital Secretary → Clinical Calendar feed adapter** from Setmore appointments.
5. Use stable Setmore appointment IDs for idempotent upsert into `clinical_appointments`.
6. Correct Digital Secretary osteoporosis-review duration from 40 to 60 minutes.
7. Configure the same `CLINICAL_INGEST_KEY` on the Digital Secretary and Osteoporosis Render services.
8. Smoke previous/current/next week with synthetic/non-identifiable appointment fixtures, then with live Setmore feed.
9. Only after the feed is stable, implement CareTasks and Zadarma reminder workflow.

---

## 6. Pilot rules

The 5-case pilot uses real clinical encounters but remains a usability/capture pilot, not a scored performance phase.

During the 5 pilot cases:

- no live KPI coaching/red-green performance feedback;
- safety-critical behavior may remain active;
- do not revise the Baseline form after each case unless there is a safety, data-loss or persistence defect;
- after all 5, perform one deliberate refinement;
- then freeze form + KPI applicability/calculation rules before the 30-case scored baseline.

Calendar/CareTask infrastructure is operational support and must not change the neutral baseline KPI scoring contract.
