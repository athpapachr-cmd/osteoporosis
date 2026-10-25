# Clinical Inbox B0 — protected persistent clinical attention and qualified identity

**Date:** 2026-10-25 Asia/Nicosia. **Status:** PRE-CODE DESIGN CANDIDATE / ONE affected security-identity R2 HOLD. **Authority:** clinician Product Owner approved initiating Step B after synthetic #161 acceptance; this is **design authority only**, not authority for live PHI, DDL/grants, persistent real-ID writes, provider connections, release or secrets.

**Repository / frozen baseline:** `athpapachr-cmd/osteoporosis`, production `main=3204bc7722b5bb45241c5b23d788f6f51c6fb502` (#213 one-tap synthetic UI merged/LIVE). **Current reuse:** [#208](https://github.com/athpapachr-cmd/osteoporosis/pull/208), [#209](https://github.com/athpapachr-cmd/osteoporosis/pull/209), [#212](https://github.com/athpapachr-cmd/osteoporosis/pull/212), [#213](https://github.com/athpapachr-cmd/osteoporosis/pull/213), the independently accepted [#210 ADT/referral design](https://github.com/athpapachr-cmd/osteoporosis/pull/210) and [Phase-A multi-account contract](CLINICAL_INBOX_PHASE_A_MULTI_ACCOUNT_CONTRACT_2026-10-23.md). Reuse unchanged source-revision, CAS/409, mixed-subject, stale-401, clinician-review and top-right peek oracles. Do not restart these completed reviews.

## 1. Clinical outcome and human path

**Goal:** the clinician opens the existing ✉ compact peek or Clinical Inbox, sees a durable queue of attention items, and can acknowledge **receipt/source inspection** in one tap; when a truly issuer-qualified existing ADT or verified exact referral/order relationship identifies one protected patient, a current patient link is displayed or an explicit **single** confirmation is available without entering DOB, ADT digits, fake evidence codes or a second checkbox. The same verified current item appears in Visit Brief. It survives logout/relogin, worker restart and new browser sessions. A **source inspected** state is never a reviewed/normal lab result, a signed encounter or a completed obligation.

- **Already confirmed exact prior referral:** show `Συνδεδεμένος ασθενής` with minimized qualified-source reason; clinician can press `Το είδα`; no repeat identity fields.
- **New unique issuer-qualified ADT match:** server proves live, unrevoked, conflict-free unique match against the *protected* identity owner, then UI shows the patient and **one** `Επιβεβαίωση ασθενούς`. No mandatory DOB or manual reference entry.
- **No qualified match, duplicate identifiers, weak name/sender/date/phone or bare referral:** clear `Δεν υπάρχει επαληθευμένη αντιστοίχιση`; keep **Inbox-only**. Source review may be recorded without patient projection. Never invent or manually type a synthetic secret to bypass evidence.
- **Mixed/unknown subject document:** no entire PDF or preview of another person in a patient Brief; keep Inbox-only until separately proven safe subject-specific segmentation and verified link.
- **GESY results-available notification:** merely `Αποτελέσματα διαθέσιμα — εκκρεμεί ιατρικός έλεγχος`. Does not create analyte values, accept findings, complete CareTask, send SMS, book visits or make treatment decisions.
- **Correction/new revision, even single_subject→single_subject:** atomically invalidate former patient association, qualified evidence, review/disposition and unsubmitted approvals; preserve one logical item and sanitized append-only revision history; clinician must requalify.
- **Existing task or prior encounter:** Inbox never becomes a second task, clinical encounter, email-transport or booking writer. Existing Visit Brief and ClinicalPending are read-only consumers of verified attention metadata.

**Product rule:** a technical database requirement must never become an extra clinician-facing field or click. The UI must always distinguish `Επισκοπήθηκε το εισερχόμενο` from `Ιατρικά αξιολογήθηκαν τα αποτελέσματα`.

## 2. Source-grounded current implementation and gaps

| Current source at baseline | Confirmed role / material gap |
|---|---|
| `clinical_data.py:PatientORM` | Existing `clinical_patients.patient_id` is an internal generated ID and `demographics_json` is not a verified ADT/GESY registry; never derive issuer proof from it. |
| `clinical_data.py:ClinicalPendingORM` | Existing obligation owner; free-text description is NOT qualified original-referral evidence or a new Inbox record. |
| `clinical_surgery_queue.py` | Different surgery queue ID authority; its `identity_number` must not silently join patients to incoming clinical reports. |
| `clinical_attention_phase_a.py:SyntheticAttentionStore` | In-memory, synthetic-only four fixtures; server-checked qualified fictional ADT/referral, CAS, audit, safe projection, revision invalidation. Does NOT persist real attention/identity. |
| `clinical_synthetic_161.py` | Isolated `/clinical/synthetic-161/` test authority; synthetic patients/appointments already use existing clinical test rows, but attention resets with process. Never widen this route to production PHI. |
| `clinical_visit_brief.py` | Existing protected booking→patient link, distinct per-actor access audit and Visit Brief. A booking UID is not a patient identity source. |
| `clinical_staff_auth.py` / `clinical_auth.py` | Separate staff-cookie roles exist but `COCKPIT_STAFF_AUTH_ENABLED` defaults off; shared legacy Clinical Key and role-wide `clinic-clinician`/ `clinic-secretary` principals do not prove an individual operator. Real Inbox link/review cannot accept that as an attributable clinician. |
| `clinical_data.py:build_clinical_router` | Calls `ClinicalBase.metadata.create_all` at router creation. Registering new B tables in `ClinicalBase` would risk creating tables at app boot: **do not do this**. Explicit reviewed migration only, with schema/grant attestation and no boot DDL. |
| `cockpit/migrations/20261017_*`, `20261018_*` | Existing PostgreSQL additive migration, preflight, least-grants and append-only audit patterns to reuse; NOT a grant for a new B migration. |

The normal Cockpit ✉ currently reports **no live mailbox provider**. Step B must not imply it scans real Gmail/Outlook/iCloud or has complete email coverage. Reception remains future transport owner; Clinical Excellence owns patient association and clinical review.

## 3. Proposed minimal protected owners, not live DDL

**One Clinical Excellence attention owner** within the existing private PostgreSQL clinical environment. Do not create a second patient registry, obligation writer or transport store. New tables are **conditional design contracts**, and no table/column is created by this PR.

| Logical storage owner | Minimum purpose/fields | Constraints |
|---|---|---|
| `ClinicalAttentionItem` (proposed `clinical_attention_items`) | opaque `item_id`, `origin_kind`, `source_type`, opaque account/source/part scope (nullable for authorized manual intake), stable `logical_source_item_key`, `active_source_revision`, `source_subject_scope`, nullable *internal* `patient_id`, `identity_state`, `review_state`, `disposition`, `version`, source/received/updated times | Unique verified logical document scope; mixed/unknown never patient-projected; update via transaction/CAS and version; no raw email/PDF/identifier in generic preview or audit. A manual attention row is NOT a manually accepted medical result. |
| `ClinicalAttentionEvent` (proposed `clinical_attention_events`) | opaque event/idempotency IDs, item + revision + version, event type, attributable actor ID/role, time, outcome and **sanitized** provenance | Append-only `SELECT/INSERT`; exact actor+item+payload idempotency, stale 409; actor-read/denied mutation audit. An item-update and its audit event commit atomically. |
| `PatientIdentityEvidence` (only if source provenance is independently qualified) | existing `clinical_patient_id`, kind, issuer, verified state, actor/time, source proof, key version, revocation/conflict/validity; server-only keyed blind equality index for normalized ID | **No** raw ADT in ordinary JSON/search/HTML/URLs/logs, no unqualified demographics/surgery join; key custody, rotation, encryption requirements and unique/conflict rules must be approved before an identifiable row is stored. Unqualified evidence produces NO table entry. |
| `ReferralSubjectLink` (only if a real original referral/order has been independently verified) | issuing authority/source family, qualified original order/reference equality key, one verified protected patient ID, exact item+subject scope, verification actor/time/state, version/revocation/conflict | Reference must really exist in qualified original referral/order owner. Do not infer from ClinicalPending free-text or an email sender; no cross-issuer/global bare-ID matches. |

**Design choice for B0:** qualify the storage contract and protected identity ownership first; it is acceptable for the first implementation slice to have only **empty/additive schema + disposable-PostgreSQL testing + feature OFF**. Real identifier values/clinical content and clinician-manual PHI intake stay disabled until all field-level activation gates close. Do NOT use a "temporary" global plaintext ID list, fixed hard-coded ADT lookup or shadow patient directory.

If an approved existing source owner can provide the same verified identity/referral evidence without new tables, reuse/rebind it instead of duplicating ownership. A validated implementation design must decide that on actual source evidence before migration.

## 4. Authorizations and session rules (activation minimum)

1. All **new identifiable B read/write/link/review endpoints** require an independently attributable **clinician** session on the server, with a unique real operator principal, not a browser role flag, synthetic Clinical Key header injection, static role-wide pseudoprincipal or legacy shared-key fallback. Existing `staff_enabled()` state in production is **UNKNOWN until verified without exposing secrets**. Feature remains OFF otherwise (HTTP 503/403 as appropriate). Do not silently change authentication for established unrelated paths.
2. Existing Product Owner decision permits clinician **and secretary to READ already-qualified Visit Brief** in their respective protected scopes; preserve it. This does **not** automatically grant secretary authority to approve, clinically review, retarget or dispose of incoming source evidence. New real Inbox default is **clinician-only**, except any later separately authorized minimal administrative projection. Test both roles and denied-access audit.
3. Every protected item view and mutation is actor/audit accountable, with short-lived scoped session, logout/offboard/rotation handling. A real expired or denied session clears protected client projections. Delayed older 401/response must not end the new session or resurrect a previous patient/item.
4. Verify private durable PostgreSQL storage, backups and restore, access boundaries, TLS, encryption at rest, source-key/blind-index encryption and key rotation *without reading patient rows or revealing environment secrets*. App role has no CREATE/DDL/grant authority; new table grants exact least privilege, append-only event audit, separate migration principal, preflight and post-migration catalog/grant attestation.
5. Confirm clinic-specific purpose/retention/access/deletion and legal-hold policy for (a) current attention rows, (b) previous source versions, (c) verified ADT/referral mappings, (d) access/denial/event audit, (e) copies/backups and external object references. **No invented 4-visit, 6/24-month or indefinite retention.** Correction/revocation alone is NOT verified erasure. Do not erase medical encounter or independently owned Pending when cleaning attention.
6. No raw PHI, personal identifiers, message bodies, attachment contents, real keys or emails in GitHub, CI, routine logs, URLs, metrics, browser persistent storage, simulator fixtures or unprotected previews. No global app access via synthetic route.

## 5. Atomic state and consumer contracts

- **Write:** one transaction locks/verifies the current item/source revision and actor; validates qualified, issuer+subject-scoped unique patient evidence against the existing protected patient; compares `expected_version`; applies link/review/disposition only to authorized axes; increments revision; appends actor audit/idempotency record; commits once. Conflict/revocation/mixed source/patient mismatch → fail closed and preserve current state.
- **Read:** repeat verification that linked revision is active, evidence is still valid and exactly one protected patient target remains. Brief and patient-chart projection must derive from the same `ClinicalAttentionItem` and never from browser `verified` booleans; legacy source values are untrusted.
- **Replay/revision:** same valid idempotency response projects *current safe* item; replay after a later version → 409; corrected same-subject source always revokes all old evidence, review and disposition, not just old patient_id. No stale clinical actions or old replay may recreate prior Brief.
- **Cold start:** durable state survives process/browser restart; no `SyntheticAttentionStore` fixture promoted to clinical authority; disabled new tables/roles or missing migration are explicit "not available", not a misleading empty-success Inbox.
- **Transport:** no mailbox polling, Reception handoff, IMAP, Graph, Gmail OAuth or source-completeness claims in B0/B1. These are Step C with distinct per-account provider/security gates. A future provider receipt must be separately scoped by account/native message/part/subject/version.
- **No effects:** an attention status mutation cannot sign a Visit Capture, change labs, finish ClinicalPending, send messages, book a slot, trigger SMS or infer treatment.

## 6. B execution boundaries / finite milestones

| Milestone | Bounded result | Explicit gate / authority |
|---|---|---|
| **B0 — this PR** | Source-grounded persistence/identity/access/retention contract + single focused affected PRE-CODE R2 request. **Docs only.** | Clinician start approval; one independent R2. NO code/DDL/provider/PHI. |
| **B1 — later implementation** | Separate protected attention owner and *empty* explicit migration/grants in disposable PostgreSQL; feature OFF, role/server enforcement, append-only audit, CAS, reboot/replay tests; synthetic input. | Independent B0 PASS + explicit clinician implementation authorization. Controlled migration not production-executed in B1 merely because source is merged. |
| **B2 — activation qualification** | Qualified identity/referral owner and one-click protected association over actually verified source, field-level privacy/retention, backup/restore/offboarding, realistic security tests without real PHI in CI. | Separate explicit identification of verified issuer/order authority, policy adoption, installation/real-data activation approval and one bounded affected post-code R2. |
| **C — later** | Gmail adapter, then Microsoft/iCloud as separately authorized; honest per-account coverage and recovery. | Independent provider/security qualification, consent/permissions, no silent broad mailbox access. |

**No automatic passage from B0 PASS to B1 implementation or PHI activation.** Reuse existing #208/#209/#210/#212/#213 review evidence; do not launch a second equivalent architecture review chain.

## 7. Predeclared finite R2 questions and implementation oracles

**Six affected PRE-CODE questions; one review only:**

1. **Ownership:** exactly one Clinical Excellence attention/item/audit owner; existing patient/Visit Brief/Pending intact; no second registry or source transport. Does new storage require new tables? Is existing exact qualified evidence provable?
2. **Real identity:** issuer/subject-scoped evidence only, verified referral reuse and ADT matching server-side, no mandatory DOB/typed arbitrary code, cross-patient + conflicting/revoked + mixed/unknown fail closed; unique clinician proof.
3. **Privacy/security:** per-person principal and clinician-only mutations, secretary Visit Brief access preserved, legacy-key/synthetic headers cannot authorize new real identifier endpoints, no PHI in public/browser/CI.
4. **Persistence/concurrency:** controlled migration avoiding `ClinicalBase.metadata.create_all`, schema/grants, atomic idempotency/CAS/revision and actor audit, cold-start/restart state and stale 401 safety.
5. **Lifecycle:** correct source revision invalidates all previous identity/review/disposition even same-subject, Brief excluded, deletion != erasure, purpose-specific retention/backup policy gates explicit and *not* fabricated.
6. **Product meaning:** compact peek and one-click review remain intuitive, notification != reviewed lab or resolved Pending, normal Inbox states true lack of email sources, zero outbound/provider side effects.

**Future B1 tests (not executed or authorized here):** isolated PostgreSQL migration preflight/drift/least-grants; staff clinician vs secretary vs legacy/shared role; locked/revoked source; two concurrent tabs, idempotent immediate replay and 409 stale, mixed subjects; same-subject correction; duplicate-issuer/no-source; patient Brief from current verified link only; cold-start restart; logout, current/stale 401 and principal switch; UI one-click with no input; no task/sms/lab/provider writes; no synthetic/prod data contamination. Reuse existing focused regression suites and exact-head CI, with no uncontrolled real patient read/smoke.

**Release stop rule:** any missing real actor identity, private retention policy, issuer-qualified original evidence, migration privileges, backup/restore or authorization is a **BLOCK only for that identifiable activation boundary**. Synthetic/demo work can remain available independently, but no weakened match or extra clinician password/code input is an acceptable workaround.

**Review disposition required:** `PASS|BLOCK|UNKNOWN / COMPLETE_FOR_DECLARED_SCOPE|PARTIAL / P0:P1:P2`, six Q outcomes, source-level evidence, exactly one smallest correction if BLOCK, then **STOP READ-ONLY**. Even PASS grants no merge/deploy/real PHI/provider authority.
