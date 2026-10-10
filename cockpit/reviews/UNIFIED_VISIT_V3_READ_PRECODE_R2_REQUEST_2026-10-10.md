# Cockpit unified Visit V3 — independent R2 pre-code, protected read extensions

2026-10-10. **Fresh independent READ-ONLY, ONE finite pre-code review** of the two missing protected read projections and the patient identity boundary. Do not implement, merge, deploy or run a real patient pilot.

Repository `athpapachr-cmd/osteoporosis`. Base production `main=1febcfa27096df20f6f4bdeb3ccee15158634fc3`. Integration draft branch `feat/cockpit-unified-visit-v3-integration-2026-10-10`, stacked on pending draft PR #140 (`f134a359bdeb7ca8b73e90b64cc16354cb2db6c5`). The branch contains **only clinician-facing UI presentation** and a design contract. New protected backend read extensions are held until this R2 pre-code decision.

**Design owner:** `cockpit/UNIFIED_VISIT_V3_INTEGRATION_CONTRACT_2026-10-10.md` blob `9e107112020e2d98d542867c64fbb08d47e278dd`.

## Bounded questions

R1. Is it correct to extend the EXISTING `/clinical/calendar/cockpit-context` protected endpoint's `CockpitContext` to return at most **3 upcoming_today** `CockpitAppointment` entries, from the same already fetched validated Reception schedule? Are date/time boundaries (`Asia/Nicosia`, only starts after now until local midnight), freshness/unavailable semantics and no inferred patient linkage sufficient? Do not create another calendar or booking writer.

R2. Is one MINIMAL protected read-only endpoint `GET /clinical/recent-encounters?limit=3`, inside existing `clinical_data.py` owner, safe and implementable? It should return only completed/amended encounter metadata needed for the last **3 actual clinical encounters** (patient label from stored demographic name, encounter date/type, existing patient link internally), never visit payload, DOB/phone, source documents or raw transcript, no new data store.

R3. Does the new Home UI preserve identity integrity? A calendar name alone is a contextual candidate and must never automatically select `patient_id` or fetch patient records. A clinician's explicit selection via existing full-registry search establishes chosen protected patient context; existing Save remains a separate clinician action. No IDs displayed to the user or leaked into URL query, analytics or permanent browser storage.

R4. Can this read-only extension be admitted without changing or reopening previous Visit Capture R2 A1–A3, Calendar/Reception, Module 01 and PR #140 A4 review? The Dia prompt/plaintext preview is browser-only; do not turn arbitrary prose into the signed structured encounter under this authority. State any material retained privacy/real-source qualification explicitly.

Review exact existing source: `clinical_calendar.py` (`CockpitContext`, `/cockpit-context`), `clinical_data.py` (`PatientORM`, `EncounterORM`, `/patients`), `static/cockpit/index.html`, `static/cockpit/clinical-workspace.js`. Tests planned: synthetic 4+ calendar rows with 3 correctly selected, zero/no-schedule, 4+ recent signed encounters, two names shared, cross-day Cyprus, no fetching/Save on candidate click and no extra provider calls.

Verdict exactly **R2 PRE-CODE PASS / BLOCK / UNKNOWN**, finite coverage R1–R4, findings P0:P1:P2, smallest corrections, STOP. A PASS makes only the two declared protected *read projections* eligible for bounded implementation and subsequent one post-code affected review. It is not production deployment or live identifiable Dia/Heidi/GESY approval.
