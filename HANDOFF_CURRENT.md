# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-23 11:38 Asia/Nicosia
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

Public repository rule remains absolute: no identifiable patient data, clinical exports or secret values may be committed.

---

## 2. Digital Secretary integration truth — read-only dependency for this slice

The existing Digital Secretary repository is `athpapachr-cmd/ortho-reception-backend-v2`.

Setmore remains the booking source of truth. Cal.com remains availability/scheduling support. The Digital Secretary owns external integrations; Clinical Excellence owns clinical classification, display and later CareTasks/reminders.

The product owner explicitly decided **not to modify the Digital Secretary in the current Calendar slice**. The Clinical Calendar must therefore mirror the Secretary's current duration behavior for now:

```text
osteoporosis_first      60 min
osteoporosis_review     40 min
Aclasta                 60 min
Prolia / injection      10 min
```

The product owner may later change the Secretary's osteoporosis-review duration. When that happens, the Clinical Calendar classification contract must be updated in the same integration change; do not pre-emptively diverge now.

Read-only inspection of the current Secretary code established that the visit reason is designed to survive the booking path:

```text
telephone reason / payload.notes
→ Cal booking notes
→ Setmore shadow appointment comment
→ Setmore appointment snapshot comment
```

The Setmore comment may also contain transport metadata such as `clinic=...`, `source=...` and `cal_uid=...`. The Clinical Calendar should expose only the human visit-reason portion. This does not prove that every historical/manual appointment contains a reason; missing source semantics must fail closed rather than cause unrelated appointments to appear.

`CLINICAL_INGEST_KEY` is configured for future server-to-server ingestion; never commit or print its value.

The Digital Secretary currently has another active implementation writer/scope. Do not create a parallel Secretary branch/PR for Calendar feed work until that operational lock is released and the product owner authorizes the next Secretary slice.

---

## 3. Clinical Calendar v1 — foundation and navigation live

Clinical Calendar foundation PR #25 is merged on `main` at:

```text
3f8a5b87e2120e8cb88cae4513359237f8ad97e5
```

Navigation PR #26 is merged on `main` at:

```text
297af278e8cf93176ee4fb13b74695ab606e8dfd
```

The Render deployment for the navigation release is live. Current routes:

```text
/static/baseline-audit/
/static/clinical-calendar/
GET  /clinical/calendar/appointments?start=...&end=...
POST /clinical/calendar/appointments/import
```

The Baseline Audit sidebar contains `Ημερολόγιο`. The service root `/` temporarily redirects to the current Baseline Audit workspace; the legacy Cockpit remains at `/static/index.html`. Later `/` becomes the real Clinical Excellence Home/Dashboard.

Appointments remain distinct from CareTasks. Lab reminders, treatment due dates, results review and patient notifications must not become fake appointments.

---

## 4. Active Calendar implementation slice

Active branch:

```text
feat/osteoporosis-calendar-filter
```

Approved behavior:

- show **only** osteoporosis-related appointments: `osteoporosis_first`, `osteoporosis_review`, `osteoporosis_unspecified`, `prolia`, `aclasta`;
- unrelated appointments must not appear in the clinician-facing Calendar;
- unrelated appointments arriving through the future ingest path are not persisted in `clinical_appointments`;
- if a previously relevant source appointment is later reclassified as unrelated, remove the stale clinical copy on ingest;
- explicit medication/service/reason semantics outrank duration;
- duration may refine first-vs-review only after osteoporosis context is already established;
- duration alone must never turn a generic 40- or 60-minute visit into osteoporosis;
- current Secretary refinement is 40 min → review and 60 min → first visit only within established osteoporosis context;
- if osteoporosis is established but first/review cannot safely be resolved, use `osteoporosis_unspecified`;
- derive a clinician-facing `reason` from Setmore comment/notes while hiding `clinic/source/cal_uid` transport metadata.

This deliberately favors false negatives over false positives: if a source appointment lacks enough osteoporosis semantics, it stays excluded rather than displaying an unrelated patient in Module 01.

---

## 5. Exact next actions

1. Review the `feat/osteoporosis-calendar-filter` diff and focused classification tests.
2. Merge if clean and let Render auto-deploy; do not manually trigger deployment.
3. Smoke the protected Calendar route and confirm the UI remains osteoporosis-only.
4. Do **not** claim live Setmore appointment display until a real feed has populated `clinical_appointments`.
5. Keep the Digital Secretary unchanged for now. When its active writer lock is released and the product owner wants live feed integration, design the smallest Setmore → Clinical Calendar ingest slice using stable Setmore appointment IDs and the existing `CLINICAL_INGEST_KEY`.
6. Implement CareTasks/Zadarma reminder workflow only after the appointment feed is stable.

---

## 6. Appointment vs CareTask — frozen separation

```text
Appointment
= scheduled attendance

CareTask
= clinical action that can exist with or without an appointment
```

Future CareTasks include pre-Prolia/pre-Aclasta labs, post-treatment monitoring, results review, patient notification, treatment administration due/overdue and next clinical review required.

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
