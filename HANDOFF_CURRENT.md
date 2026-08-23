# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-23 11:58 Asia/Nicosia
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

## 2. Digital Secretary integration truth — read-only dependency

The existing Digital Secretary repository is `athpapachr-cmd/ortho-reception-backend-v2`.

Setmore remains the booking source of truth. Cal.com remains availability/scheduling support. The Digital Secretary owns external integrations; Clinical Excellence owns clinical classification, display and later CareTasks/reminders.

The product owner explicitly decided **not to modify the Digital Secretary for the current Calendar work**. The Clinical Calendar therefore mirrors the Secretary's current duration behavior:

```text
osteoporosis_first      60 min
osteoporosis_review     40 min
Aclasta                 60 min
Prolia / injection      10 min
```

The product owner may later change the Secretary's osteoporosis-review duration. When that happens, update the Clinical Calendar classification contract in the same integration change; do not pre-emptively diverge.

Read-only inspection of the current Secretary code established that the visit reason is designed to survive the booking path:

```text
telephone reason / payload.notes
→ Cal booking notes
→ Setmore shadow appointment comment
→ Setmore appointment snapshot comment
```

The Setmore comment may also contain transport metadata such as `clinic=...`, `source=...` and `cal_uid=...`. The Clinical Calendar exposes only the human visit-reason portion. This does not prove that every historical/manual appointment contains a reason; missing source semantics must fail closed rather than cause unrelated appointments to appear.

`CLINICAL_INGEST_KEY` is configured for future server-to-server ingestion; never commit or print its value.

The Digital Secretary currently has another active implementation writer/scope. Do not create a parallel Secretary branch/PR for Calendar feed work until that operational lock is released and the product owner authorizes the next Secretary slice.

---

## 3. Clinical Calendar — merged and live

Foundation PR #25:

```text
3f8a5b87e2120e8cb88cae4513359237f8ad97e5
```

Navigation PR #26:

```text
297af278e8cf93176ee4fb13b74695ab606e8dfd
```

Osteoporosis-only Calendar PR #27 (`feat: restrict Clinical Calendar to osteoporosis appointments`) was squash-merged as:

```text
c0624068d253a292a9da32cf1b2f19f902237fe5
```

Render auto-deploy:

```text
dep-da5bbdk9v7es73f2fqs0 — LIVE
```

Runtime startup completed successfully with PostgreSQL clinical storage online.

Current routes:

```text
/static/baseline-audit/
/static/clinical-calendar/
GET  /clinical/calendar/appointments?start=...&end=...
POST /clinical/calendar/appointments/import
```

The Baseline Audit sidebar contains `Ημερολόγιο`. The service root `/` temporarily redirects to the current Baseline Audit workspace; the legacy Cockpit remains at `/static/index.html`. Later `/` becomes the real Clinical Excellence Home/Dashboard.

Appointments remain distinct from CareTasks. Lab reminders, treatment due dates, results review and patient notifications must not become fake appointments.

---

## 4. Frozen osteoporosis-only Calendar behavior

The clinician-facing Calendar returns/displays only:

```text
osteoporosis_first
osteoporosis_review
osteoporosis_unspecified
prolia
aclasta
```

Rules:

- unrelated appointments do not appear in the Calendar;
- future ingest skips unrelated appointments instead of persisting them in `clinical_appointments`;
- if a previously relevant source appointment is later reclassified as unrelated, its stale clinical copy is removed on ingest;
- explicit medication/service/reason semantics outrank duration;
- duration may refine first-vs-review only after osteoporosis context is already established;
- duration alone never turns a generic 40- or 60-minute visit into osteoporosis;
- current Secretary refinement is 40 min → review and 60 min → first visit only within established osteoporosis context;
- if osteoporosis is established but first/review cannot safely be resolved, use `osteoporosis_unspecified`;
- the clinician-facing `reason` is derived from Setmore comment/notes while `clinic/source/cal_uid` transport metadata is hidden.

This deliberately favors false negatives over false positives: insufficient source semantics stay excluded rather than risking unrelated patient display inside Module 01.

Focused source tests cover medication categories, explicit first/review semantics, current 60/40 duration refinement only inside established osteoporosis context, unrelated 40/60-minute visits remaining `other`, ambiguous osteoporosis remaining unspecified, and metadata-stripped reason rendering. GitHub has no CI status configured for this head; merge evidence is scoped diff review plus focused deterministic checks, not a claimed CI run.

---

## 5. Current limitation and exact next action

The Calendar code is live, but **live Setmore appointments are not yet being ingested**. No request to `/clinical/calendar/appointments/import` was observed during the integration check before this release. Therefore do not claim that live Secretary appointments are already visible merely because the Calendar UI/filter is deployed.

Exact next action when the product owner is ready and the Digital Secretary writer lock is released:

```text
Setmore appointment feed
→ stable Setmore appointment ID
→ reason/comment + timing + patient display data
→ POST /clinical/calendar/appointments/import
→ osteoporosis-only classification/filter
→ clinical_appointments
→ weekly Clinical Calendar
```

Until then:

1. keep the Digital Secretary unchanged;
2. preserve the current 60/40/60/10 duration alignment;
3. do not introduce Cal.com as a replacement clinical appointment source merely to bypass the Secretary boundary;
4. once feed integration is authorized, smoke previous/current/next week with live Setmore data;
5. only after the feed is stable, implement CareTasks and Zadarma reminder workflow.

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
