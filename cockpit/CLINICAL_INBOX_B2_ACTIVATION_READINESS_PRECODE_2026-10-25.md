# Clinical Inbox B2 — identifiable-activation readiness (PRE-CODE, design only)

**Date:** 2026-10-25 (Asia/Nicosia). **State:** B2 PRE-CODE CANDIDATE / ONE bounded independent affected R2 HOLD. **Authority:** Product Owner conveyed independent B1 correction-closure PASS and said to proceed to the next step; this authorizes B2 *readiness design only*. No B2 runtime implementation, code changes, live clinical identity, PHI, DDL/grants, provider connection, production config, merge or deployment.

## 0. Exact dependency and frozen boundary

- Repository `athpapachr-cmd/osteoporosis`; verified production `main` at start `3204bc7722b5bb45241c5b23d788f6f51c6fb502`.
- B0 draft [PR #214](https://github.com/athpapachr-cmd/osteoporosis/pull/214), frozen design blob `16fcdf12d41553d93f7a75b934e4c0b722605094`, independently accepted PRE-CODE for its declared scope.
- B1 draft [PR #215](https://github.com/athpapachr-cmd/osteoporosis/pull/215), frozen corrected synthetic-only head `87280499b029b2615cd064874d9071ad236da467`, independent correction R2 PASS reported by Product Owner (not authority for merge/release). B1 runs on disposable PostgreSQL only and is not mounted in normal app.
- Reuse [#210 source-qualified ADT/referral contract](https://github.com/athpapachr-cmd/osteoporosis/pull/210), prior #208/#209/#212/#213 synthetic UI/safety decisions and [Phase-A multi-account contract](CLINICAL_INBOX_PHASE_A_MULTI_ACCOUNT_CONTRACT_2026-10-23.md). No re-review of settled synthetic UX or B1 CAS tests.
- This B2 design branch stacks on **B1 draft head** for documentation diff only; B0, B1 and B2 all stay DRAFT/unmerged. B2 cannot ride into `main` by an implicit dependency merge.

## 1. Patient-visible outcome (no extra clicks)

The clinician opens the established compact Inbox/peek, sees receipt state, and presses `Το είδα` once to acknowledge **source inspection** (never "medical result reviewed"). If an existing issuer-qualified original referral or previously verified protected identity uniquely and currently matches one clinical patient, show `Συνδεδεμένος ασθενής`; reuse already-confirmed evidence without DOB/ADT re-entry. If a *new* uniquely qualified source requires clinician approval, show one `Επιβεβαίωση ασθενούς`, grounded in the existing source/card, without a second checkbox or fake code.

Unknown, ambiguous, unverified, revoked, conflicting or mixed-subject source remains `Χωρίς επαληθευμένη αντιστοίχιση` and **Inbox-only**. Do not project the source into any patient's Brief/chart. GESY notice means `Αποτελέσματα διαθέσιμα — εκκρεμεί ιατρικός έλεγχος`; it never imports values, accepts results, closes Pending or sends contact.

## 2. Source-grounded blockers to real activation

| Observed source | Verified limitation | B2-required decision |
|---|---|---|
| `clinical_staff_auth.py:match_staff_key/require_staff_actor` | Staff cookies map to shared `clinic-clinician` or `clinic-secretary` role principals; no personally attributable staff operator. | Approve a real per-person clinician identity owner (lifecycle, proofing, short-lived sessions, offboarding, credential rotation, audit); role-only/shared Clinical Key must fail closed for new B2 routes. |
| `clinical_auth.py` / `clinical_pending_one_tap.py:actor_for` | Legacy protected clinical actions can use shared clinical-key identity. | Preserve unrelated legacy behavior; **do not** reuse shared-key fallback as B2 authority. |
| `clinical_data.py:PatientORM` | Existing generated `patient_id`/demographics do not establish issuer-verified ADT or GESY beneficiary registry. | Identify and qualify an actually controlled, independently verified issuer/evidence owner before any real identity comparison. |
| `clinical_data.py:ClinicalPendingORM`, `clinical_surgery_queue.py` | ClinicalPending free text is not original referral proof; separate surgery `identity_number` is not patient-registry authority. | An original issuer-qualified referral/order must be evidenced, source-scoped and attributable; no text/name/phone/date-only join. |
| `clinical_attention_b1.py` + B1 migration | Requires local test opt-in; SQL `patient_id IS NULL` and synthetic keys; no production router. | Keep B1 intact. Real B2 needs **separate reviewed storage/activation**; never weaken B1 CHECK or import synthetic fixture as clinical record. |
| `clinical_data.py:build_clinical_router` | Calls `ClinicalBase.metadata.create_all` at router construction. | Keep B2 ORM separate from `ClinicalBase`; explicit privileged migration only with reviewed preflight/least grants, no boot DDL. |

No actual issuer-qualified original referral or individually attributable real clinician identity was verified by this repository inspection. They are explicit **OPEN gates**, not facts to presume.

## 3. B2 work sequence — smallest useful slices

**B2A — actor/source qualification contract (next bounded implementation candidate, but separately authorized).** Prepare server-level per-person clinician identity and issuer-adapter contracts in an offline, fake-actor/fake-issuer test harness. Prove two distinct clinician principals, clinician-only mutations, secretary denial (while established qualified Visit Brief secretary READ remains unchanged), shared legacy-key rejection, expiration/rotation/offboarding, tamper resistance, denial audit and no role/header spoofing. No production actor configuration, no PHI route, no real ADT value, no real source enrollment. Do not install B2A without independent post-code gate.

**B2B — protected data policy and evidence owner qualification.** Obtain Product Owner-approved and operator-confirmed field-level purpose/access, retention/deletion/legal-hold/backup-restore policy for attention current+old versions, source provenance, identity/referral evidence and append-only audit. Identify real issuer/original-referral owner or declare `NOT QUALIFIED`; support a scoped verified referral relationship only if original order/reference ownership is proved. Specify secure comparison (keyed blind index/encrypted reference), key custody/rotation, conflict/revocation, tenant/practice scope, and explicit privilege migration. Never fabricate an ADT registry or use the surgery queue as one.

**B2C — identifiable patient linkage, only after all B2B gates and new explicit real-data authority.** Server validates a real qualified source against exactly one current protected patient; mutations use actor-scoped CAS/row lock/idempotency, audit atomically and revalidate on read. On *any* verified source revision, even single-subject→single-subject, invalidate prior patient evidence/link, Brief projection, review/disposition and unsent approvals. Preserve existing clinical/Visit Brief/Pending ownership; no outbound, lab acceptance or task-writing side effects. A safe source cannot be inferred from mailbox text or a manually typed name.

**Step C — providers** is still separate: Gmail first, then separately reviewed Microsoft/Outlook/iCloud; Reception owns transport, Clinical Excellence owns clinical review. No Gmail, OAuth, IMAP, Graph, SMS or provider connection in B2 planning.

## 4. B2 implementation/release authorization matrix

| Gate | Before code | Before identifiable production activation |
|---|---|---|
| Real per-person staff principal and credential lifecycle | Approve source/contract; offline fake tests permitted under *separate B2A code authorization* | Fully implemented and attributable, role-scoped, logout/rotation/offboarding/401 verified |
| Real issuer-qualified ADT/referral/order mapping | Identify evidence/owner, or hold real linkage | Unique scope, issuer proof, validity/revocation/conflict/subject checks independently tested |
| Protected data lifecycle | Draft specific policy; no invented periods | Product Owner adopts explicit field-level purpose, access, retention, deletion/legal hold, backup restore and incident obligations |
| Migration/storage/privacy | Specify separate B2 table owner and no startup DDL | Separate review and explicit operator execution of empty/additive migration/grants, encryption/TLS/key policy and tested rollback/restore |
| Patient/Brief UX | Synthetic/offline regression: one click, weak/mixed fail closed, stale 401 protection | Clinician real-user protected testing under separately authorized patient/identity data policy |
| Production/provider | Not authorized | Separate clinician approval + B2 post-code/security R2; mailbox Step C still separate |

A R2 PRE-CODE PASS on this document is **not** B2A code authorization, B1 merge, production installation, PHI access, or live mailbox authorization.

## 5. Minimum exact tests for the later authorized B2A/B2C implementations

1. Two independent clinicians A/B with different attributable principals; secretary/shared/static/synthetic role cannot impersonate one or link/review. Preserve existing secretary qualified Visit Brief read.
2. Logout/offboard/rotation, stale 401 and overlapping browser tabs do not resurrect patient projections; no client-supplied role/proof authority.
3. Unique qualified existing referral reuse without DOB or second click; new independently qualified ADT match one explicit clinician confirmation; name/date/sender/phone/bare referral never automatic.
4. Unknown/mixed/multi-subject documents remain Inbox-only; corrected same-subject revision invalidates old linkage; forged/expired/revoked/conflicting issuer evidence fails closed.
5. Atomic item+audit, concurrent CAS 409, actor/item/payload-scoped idempotency, revalidation on protected reads, restart persistence; no mutation of ClinicalPending, lab result, appointment, message, SMS, Visit Capture or clinical chart.
6. Disposable PostgreSQL only until explicit installation; no PHI/real ADT/email/PDF in fixtures, CI, logs, URLs, browser storage or GitHub; explicit migration least grants and rollback; no ClinicalBase startup DDL.

## 6. One bounded PRE-CODE handback and STOP

Use [B2 affected readiness R2 request](reviews/CLINICAL_INBOX_B2_ACTIVATION_READINESS_PRECODE_R2_REQUEST_2026-10-25.md). Six affected questions, reuse B0/#210 prior PASS. Reviewer returns PASS/BLOCK/UNKNOWN with concrete evidence; no implementation or release. On PASS, request **separate explicit clinician approval for only B2A offline test implementation**. B2 identifiable activation remains on HOLD until all operational and policy gates are separately satisfied.
