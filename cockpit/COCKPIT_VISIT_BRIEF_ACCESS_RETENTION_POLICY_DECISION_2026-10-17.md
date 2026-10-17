# Cockpit Visit Brief — access, confidentiality and booking-link retention decision

**Date:** 2026-10-17 (Asia/Nicosia). **Owner:** clinician Product Owner. **Status:** confirmed product access requirement; targeted security/retention design and implementation NOT YET ACCEPTED or enabled. **Related implementation:** [PR #161](https://github.com/athpapachr-cmd/osteoporosis/pull/161), frozen reviewed code/test head `3a98f7c046a83f05e0590661118c92c972483026`, latest docs-only release HOLD. No live patient-data access or authorization to deploy.

## Product Owner-confirmed choices (do not narrow them later)

1. **Both the clinician and the secretary must be able to READ the protected medical visit summaries (Visit Brief)** and their documented previous plan, follow-up purpose, and last-recorded pending items. The secretary's role is **not** appointment-metadata-only.
2. The secretary must not silently edit, sign or authorize medical diagnoses, treatments or reviewed clinical summaries. Independent explicit booking→chart identity verification is required before each protected brief when Reception attendee continuity is unqualified; clinical linkage confirmation is a separate auditable administrative action, never a clinical Save.
3. **Only authorized clinician and secretary, not the general public**, may access identifiable data. Do not store identifiable data in the public GitHub repository or any public-readable database, object store, logs, CI artifacts, URLs, browser persistent storage, transcripts or generated publications. Authenticated HTTPS application delivery is not, by itself, public-readable storage.
4. The clinician does not want automatic removal merely because six calendar months have passed: a patient may next attend after six months. For redundant historical **technical booking↔patient associations**, consider pruning after **four later actually completed visits**; keep the distinct legally retained patient chart, signed encounters and validated clinical plan independently. Previous user phrasing was 'μετά από 3-4 επισκέψεις'; four is an implementation candidate, not a verified statutory rule. An inactivity-based secondary bound (previously suggested 24 months) is **not yet Product Owner-approved**, so do not silently activate it.

## Access authority before identifiable activation

- Provide **two separately attributable principals**, e.g. `clinician` and `secretary`, not one shared Clinical Key masquerading as named staff identity. Existing `clinical_auth.py` shared-key cookie establishes clinic access but NOT individual attribution or a role distinction; `clinical_visit_brief.py` currently records `operator_type="shared_clinical_key"`, and has no proven per-person view audit. This is an explicit implementation gap, not a claim that role-based security already exists.
- Both permitted principals: limited protected Visit Brief read on an explicitly confirmed correct booking↔patient association, including concise previous visit, plan and recorded pending checks.
- Clinician: existing clinical chart edit/Save authority remains separately protected; clinical decisions and signed encounter content may not be changed through the secretary's role.
- Secretary: no clinical Save, sign, medical-plan edit or clinical decisions. May perform an **explicit, independently verified booking↔patient administrative confirmation** only under an approved scoped permission and audit, otherwise cannot open a previously unqualified booking's brief. Never auto-match by name/phone/same UID.
- Server-side role checks and per-actor audit for view, linkage confirmation/correction/revocation and denied access; do not infer user identity from the shared key, a browser flag, or a mere frontend hide. Logout/401/403 must clear protected client data; no browser persistent PHI.
- Define offboarding/revocation, short-lived secure sessions, and a routine way for the clinician to inspect staff access history. Roles must be assigned deliberately; do not create general staff/anonymous access.

## Storage and retention boundary

- `PatientORM`, signed `EncounterORM` and clinical history are not subject to a four-visit purge. The medical record follows applicable Cyprus law and verified clinic retention policy, not this workflow convenience threshold.
- Versioned `BookingPatientLinkORM` is a narrow technical association and may be pruned according to the approved **four subsequent completed visits** rule. Count only independently linked, real completed encounters for the same protected patient; do not count scheduled, cancelled, unverified or draft visits. Prevent cleanup of a link currently required for an active booking or open clinical workflow. UID uniqueness and revocation protections remain.
- Deleting an expired link must NOT delete clinical records, pending items, plans or a clinically relevant task. Link/audit cleanup must be transactional/idempotent with audit evidence of cleanup, and no reattachment from stale client or booking-only reads.
- `BookingPatientLinkAuditORM` has a distinct integrity/accountability purpose. Do **not** automatically delete security audit at the fourth visit without a documented, lawful retention/de-identification decision and policy for disputes/legal hold.
- An upper time/inactivity limit and exact audit retention/deletion policy remain to be established as legally appropriate; do not infer six months, 24 months or indefinite retention from this decision.
- Production persistence must be demonstrably private, durable, appropriately backed up, access-controlled and encrypted in transit/at rest; verify actual `DATABASE_URL` backing and the Render/other database access configuration WITHOUT exposing secrets or reading real patient rows. Absence of Render-native Postgres alone cannot prove there is no external private database.

## Release gating and minimal further work

- **DO NOT MERGE/DEPLOY/ENABLE #161 to process identifiable booking links yet.** Existing `CLINICAL_VISIT_BRIEF_LINKS_ENABLED` remains OFF. A successful previous independent code R2 (0:0:0) closes P1/P2 implementation fidelity, not these new role, access-log, storage and retention concerns.
- Target the **new affected security authority only**: (A) independently attributable clinician/secretary sessions and server-enforced limited Visit Brief permission, (B) actor-audited protected reads and administrative confirmation, (C) technical link retention and separate audit retention policy, (D) private durable storage/config checks. Reuse existing protected auth/Clinical Data/Calendar owners; no second patient registry, clinical Save workflow, calendar/booking writer, patient data import or AI/summary engine.
- Because this changes authentication, roles and identifiable retention, classify it as **R2** for one bounded affected PRE-CODE design review and one affected POST-CODE implementation review after implementation; do not reopen already accepted Home, Visit Brief P1/P2, RF, Medical Reports or Inbox. Preserve unqualified Reception attendee-continuity manual per-opening fallback and separate automatic-one-click HOLD.
- Before any rollout with patient data: concrete clinic-specific retention/access policy and verified private storage, tested role boundaries, least-privilege migration plan and explicit Product Owner activation decision.

## External regulatory touchpoints (not an invented numeric statutory duration)

- Cyprus Patients' Rights Law 1(I)/2005, as amended, **section 15** protects confidentiality including information held by healthcare employees: https://www.cylaw.org/nomoi/enop/ind/2005_1_1/section-sc0322466d-348e-3e7d-514c-e9e029a93840-ln11a6c842-b5d9-08bb-5149-d489e25d4575.html
- Section 17 requires secure medical records and retention as determined by applicable law: https://www.cylaw.org/nomoi/enop/ind/2005_1_1/section-sc3e2b9106-53c1-b36b-1740-3c679b335cbb.html
- Cyprus Data Protection Commissioner GDPR overview: https://www.gov.cy/dataprotection/en/documents/the-regulation-gdpr/

**No actual patient data, credentials, provider calls, clinical writes, merge or deploy in this design checkpoint.**
