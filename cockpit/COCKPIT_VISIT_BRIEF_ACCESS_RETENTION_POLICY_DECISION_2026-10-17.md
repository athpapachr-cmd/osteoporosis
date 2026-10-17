# Cockpit Visit Brief — access, confidentiality and booking-link retention decision

**Date:** 2026-10-17 (Asia/Nicosia). **Owner:** clinician Product Owner. **Status:** clinician access requirement confirmed; prior four-visit deletion proposal WITHDRAWN by Product Owner; focused security/retention design correction pending independent review; implementation not yet accepted/enabled. **Related implementation:** [PR #161](https://github.com/athpapachr-cmd/osteoporosis/pull/161), frozen reviewed code/test head `3a98f7c046a83f05e0590661118c92c972483026`, latest docs-only release HOLD. No live patient-data access or authorization to deploy.

## Product Owner-confirmed choices (do not narrow them later)

1. **Both the clinician and the secretary must be able to READ the protected medical visit summaries (Visit Brief)** and their documented previous plan, follow-up purpose, and last-recorded pending items. The secretary's role is **not** appointment-metadata-only.
2. The secretary must not silently edit, sign or authorize medical diagnoses, treatments or reviewed clinical summaries. Independent explicit booking→chart identity verification is required before each protected brief when Reception attendee continuity is unqualified; clinical linkage confirmation is a separate auditable administrative action, never a clinical Save.
3. **Only authorized clinician and secretary, not the general public**, may access identifiable data. Do not store identifiable data in the public GitHub repository or any public-readable database, object store, logs, CI artifacts, URLs, browser persistent storage, transcripts or generated publications. Authenticated HTTPS application delivery is not, by itself, public-readable storage.
4. **The four-subsequent-visit deletion proposal is rescinded.** A patient may attend once or return much later; no visit-count, six-month or 24-month automatic purge is authorized. Protect the clinical chart, encounters and pending under separate lawful medical retention. Keep only necessary technical booking associations, with a separately approved, purpose-based review/erasure process and legal-hold handling before identifiable activation; do not infer indefinite retention from the absence of a numerical deadline.

## Access authority before identifiable activation

- Provide **two separately attributable principals**, e.g. `clinician` and `secretary`, not one shared Clinical Key masquerading as named staff identity. Existing `clinical_auth.py` shared-key cookie establishes clinic access but NOT individual attribution or a role distinction; `clinical_visit_brief.py` currently records `operator_type="shared_clinical_key"`, and has no proven per-person view audit. This is an explicit implementation gap, not a claim that role-based security already exists.
- Both permitted principals: limited protected Visit Brief read on an explicitly confirmed correct booking↔patient association, including concise previous visit, plan and recorded pending checks.
- Clinician: existing clinical chart edit/Save authority remains separately protected; clinical decisions and signed encounter content may not be changed through the secretary's role.
- Secretary: no clinical Save, sign, medical-plan edit or clinical decisions. May perform an **explicit, independently verified booking↔patient administrative confirmation** only under an approved scoped permission and audit, otherwise cannot open a previously unqualified booking's brief. Never auto-match by name/phone/same UID.
- Server-side role checks and per-actor audit for view, linkage confirmation/correction/revocation and denied access; do not infer user identity from the shared key, a browser flag, or a mere frontend hide. Logout/401/403 must clear protected client data; no browser persistent PHI.
- Define offboarding/revocation, short-lived secure sessions, and a routine way for the clinician to inspect staff access history. Roles must be assigned deliberately; do not create general staff/anonymous access.

## Storage and retention boundary

- `PatientORM`, signed `EncounterORM` and clinical history are not subject to booking-link lifecycle erasure. The medical record follows applicable Cyprus law and verified clinic retention policy, not this workflow convenience threshold.
- Versioned `BookingPatientLinkORM` is a narrow booking-specific association, not a durable proof of patient continuity. Do **not** schedule automatic expiry based on visit count or elapsed time. Before actual activation, document the purpose of retaining historical links, review frequency, who may approve actual erasure, the trigger when no longer needed, and open-workflow/legal-hold exceptions. Keep per-opening independent identity verification; audit correction/revocation. The current `unlink` can retain patient ID and must not be described as true erasure.
- A later policy-qualified, explicitly authorized link erasure/de-identification must NOT delete clinical records, pending items, plans or a clinically relevant task. Any such operation must be auditable, transactional, idempotent, preserve replay-resistant minimal revision/tombstone, and prevent stale reattachment. No automatic erasure job is part of v1.
- `BookingPatientLinkAuditORM` has a distinct integrity/accountability purpose. Do **not** automatically delete security audit with link correction/revocation/erasure; first adopt documented lawful audit retention, review/de-identification and disputes/legal-hold handling.
- No numeric retention period is invented here. Clinic-specific, lawful, purpose-based link and audit review/erasure criteria remain to be established before identifiable link activation. Neither a fixed 6/24-month expiry nor indefinite retention is approved.
- Production persistence must be demonstrably private, durable, appropriately backed up, access-controlled and encrypted in transit/at rest; verify actual `DATABASE_URL` backing and the Render/other database access configuration WITHOUT exposing secrets or reading real patient rows. Absence of Render-native Postgres alone cannot prove there is no external private database.

## Release gating and minimal further work

- **DO NOT MERGE/DEPLOY/ENABLE #161 to process identifiable booking links yet.** Existing `CLINICAL_VISIT_BRIEF_LINKS_ENABLED` remains OFF. A successful previous independent code R2 (0:0:0) closes P1/P2 implementation fidelity, not these new role, access-log, storage and retention concerns.
- Target the **new affected security authority only**: (A) independently attributable clinician/secretary sessions and server-enforced limited Visit Brief permission, (B) actor-audited protected reads and administrative confirmation, (C) purpose-based technical-link lifecycle and separate audit retention policy, (D) private durable storage/config checks. Reuse existing protected auth/Clinical Data/Calendar owners; no second patient registry, clinical Save workflow, calendar/booking writer, patient data import or AI/summary engine.
- Because this changes authentication, roles and identifiable retention, classify it as **R2** for one bounded affected PRE-CODE design review and one affected POST-CODE implementation review after implementation; do not reopen already accepted Home, Visit Brief P1/P2, RF, Medical Reports or Inbox. Preserve unqualified Reception attendee-continuity manual per-opening fallback and separate automatic-one-click HOLD.
- Before any rollout with patient data: concrete clinic-specific retention/access policy and verified private storage, tested role boundaries, least-privilege migration plan and explicit Product Owner activation decision.

## External regulatory touchpoints (not an invented numeric statutory duration)

- Cyprus Patients' Rights Law 1(I)/2005, as amended, **section 15** protects confidentiality including information held by healthcare employees: https://www.cylaw.org/nomoi/enop/ind/2005_1_1/section-sc0322466d-348e-3e7d-514c-e9e029a93840-ln11a6c842-b5d9-08bb-5149-d489e25d4575.html
- Section 17 requires secure medical records and retention as determined by applicable law: https://www.cylaw.org/nomoi/enop/ind/2005_1_1/section-sc3e2b9106-53c1-b36b-1740-3c679b335cbb.html
- Cyprus Data Protection Commissioner GDPR overview: https://www.gov.cy/dataprotection/en/documents/the-regulation-gdpr/

**No actual patient data, credentials, provider calls, clinical writes, merge or deploy in this design checkpoint.**
