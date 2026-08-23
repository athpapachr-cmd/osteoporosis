# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-23 11:20 Asia/Nicosia
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

Clinical authentication uses a Secure/HttpOnly/SameSite=Strict browser-session cookie. The patient-centric production layer supports Patient → Encounters[] and LabSnapshots[] with PostgreSQL as durable source of truth; localStorage remains only a working cache.

Verified browser behavior includes patient search/load, encounter Save/reload, multiple dated laboratory snapshots, comparative lab history in Step 3, and `Νέες αναλύσεις` without loss of historical snapshots.

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

Canonical ownership:

```text
Setmore / Cal.com / Zadarma
        ↓
Digital Secretary backend = external integration owner
        ↓
Clinical Excellence / Osteoporosis = clinical meaning, appointment classification, CareTasks, reminders
```

Setmore is the booking source of truth for the Clinical Calendar. Cal.com remains availability/scheduling support and must not become the clinical appointment store.

Product-owner duration contract:

```text
osteoporosis_first      60 min
osteoporosis_review     60 min
Aclasta                 60 min
Prolia                  10 min
```

The Digital Secretary still contains a legacy 40-minute `osteoporosis_followup` mapping and must be corrected to 60 minutes in the feed integration slice.

`CLINICAL_INGEST_KEY` has been configured by the product owner for server-to-server calendar ingestion; never commit or print its value.

---

## 3. Clinical Calendar v1 — merged and live

PR #25 (`feat: clinical calendar v1 foundation`) is merged on `main` at commit:

```text
3f8a5b87e2120e8cb88cae4513359237f8ad97e5
```

The Render deployment is live.

New contract/module/table:

```text
schemas/clinical_calendar_contract_v1.yaml
clinical_calendar.py
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

Important classification rule: duration alone must not distinguish first from review because both are 60 minutes. Explicit service/label/comment semantics outrank duration; ambiguous osteoporosis visits stay `osteoporosis_unspecified`.

Protected API surface:

```text
GET  /clinical/calendar/appointments?start=...&end=...
POST /clinical/calendar/appointments/import
```

Clinician reads use the existing browser-session clinical authentication. External server-to-server ingest uses `CLINICAL_INGEST_KEY`.

Calendar UI:

```text
/static/clinical-calendar/
```

It provides previous/current/next week navigation, daily appointment columns, osteoporosis/Prolia/Aclasta categories and weekly counts.

Appointments remain distinct from CareTasks. Lab reminders, treatment due dates, results review and patient notifications must not become fake appointments.

---

## 4. Navigation decision

Current implementation branch:

```text
fix/clinical-navigation
```

Approved navigation behavior:

- Baseline Audit sidebar contains a dedicated `Ημερολόγιο` item opening `/static/clinical-calendar/`;
- the Calendar remains a sibling clinical workspace, not a card embedded inside an encounter;
- the service root `/` must no longer open the legacy Cockpit page;
- until the true Clinical Excellence Home is designed, `/` temporarily redirects to `/static/baseline-audit/`;
- the old Cockpit remains available at `/static/index.html` for legacy/reference use;
- later `/` will become the actual Clinical Excellence Home/Dashboard, with Baseline Audit, Clinical Calendar, CareTasks, safety/attention queue and learning/audit surfaces behind it.

---

## 5. Exact next implementation actions

1. Merge/deploy `fix/clinical-navigation` and smoke sidebar Calendar access + root redirect.
2. Build the Digital Secretary → Clinical Calendar feed adapter from Setmore appointments.
3. Use stable Setmore appointment IDs for idempotent upsert into `clinical_appointments`.
4. Correct Digital Secretary `osteoporosis_followup` duration from 40 to 60 minutes.
5. Preserve explicit first/review semantics; do not infer them from 60-minute duration alone.
6. Smoke previous/current/next week with live Setmore feed.
7. Only after the feed is stable, implement CareTasks and Zadarma reminder workflow.

---

## 6. Appointment vs CareTask — frozen separation

```text
Appointment
= scheduled attendance

CareTask
= clinical action that can exist with or without an appointment
```

Future CareTasks include pre-Prolia/pre-Aclasta labs, post-treatment monitoring, results review, patient notification, treatment administration due/overdue and next clinical review required. CareTasks will later drive reminder logic and Zadarma SMS.

---

## 7. Pilot rules

The 5-case pilot uses real clinical encounters but remains a usability/capture pilot, not a scored performance phase.

During the 5 pilot cases:

- no live KPI coaching/red-green performance feedback;
- safety-critical behavior may remain active;
- do not revise the Baseline form after each case unless there is a safety, data-loss or persistence defect;
- after all 5, perform one deliberate refinement;
- then freeze form + KPI applicability/calculation rules before the 30-case scored baseline.

Calendar/CareTask infrastructure is operational support and must not change the neutral baseline KPI scoring contract.
