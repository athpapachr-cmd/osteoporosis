# Cockpit — Visit Brief clinician/secretary access + private retention v1

**Date:** 2026-10-17 Asia/Nicosia. **State:** BOUNDED SECURITY DESIGN / Product Owner withdraws visit-count retention / focused PRE-CODE delta R2 HOLD. **Owner:** Cockpit Visit Intelligence. This is a *new affected authentication/retention authority*, not a reopening of the previously independently PASSED Visit Brief P1/P2 implementation. Do not merge, deploy, enable identifiable data, provision secrets or perform provider writes on this design alone.

## 1. Product Owner decision and actual gap

The Product Owner confirmed: (a) both doctor and secretary may READ the *medical* Visit Brief, including previous signed visit, prior decisions/medications, and last-recorded pending/checks; (b) the secretary may NOT author, alter or sign medical content; (c) clinic access is private, not public-readable; (d) the old four-subsequent-visit technical-link purge proposal is **withdrawn**: patients may attend only once or return years later. No visit-count-based or arbitrary 6/24-month automatic expiration is authorized. The medical chart remains independently protected. Technical associations require their own documented purpose-based data lifecycle; no automatic deadline does not imply indefinite lawful retention.

Existing `clinical_auth.py` uses one shared `CLINICAL_DATA_KEY` and a derived `clinical_session` cookie. Existing `clinical_visit_brief.py` uses this common header and writes `operator_type=shared_clinical_key`; it cannot identify a doctor versus a secretary, audit accesses by actor, or enforce secretary clinical Save exclusion. `clinical_visit_brief.py` has no link cleanup, and `VisitBriefBase.metadata.create_all` currently runs during router construction regardless of feature flag. These are confirmed implementation gaps.

The public GitHub repository contains **only code, design and synthetic tests**, not identifiable patient data. An internet-accessible HTTPS application endpoint is not the same thing as a public-readable clinical database. The actual Render service `osteoporosis` runs from `main` with auto-deploy; no Render-native Postgres instance was listed in the connected workspace, which **does not prove** whether an independently managed private Postgres database exists. Actual `DATABASE_URL` destination, backups, encryption, access/network rules and data processor terms must be verified privately *without extracting secrets or patient rows*.

## 2. Scope and visibility

In: separately accountable doctor and secretary staff sessions; protected same-origin minimal Visit Brief and targeted full-registry patient search for booking verification; actor-specific access/confirmation/revocation audit; purpose-based booking-link data lifecycle with governed correction/revocation and separate audit retention; private-storage readiness and fail-closed feature gate; exact synthetic Python/Node/Chromium proof at 1280/680/390. Keep existing appointment source, selected-date truth, signed clinical evidence, and existing Home layout and Visit Capture Save authority unchanged.

Out: any new patient/booking registry, clinician chart editor, Secretary/Cal writer, automatic identity linking from phone/name/UID, new AI tasks, automatic one-click previous patient brief, whole-clinic GDPR certification or provider credential copying. No new unrelated RF, Inbox, Reports, Home or PR-1 release work.

## 3. Actor / endpoint authority (server enforced)

Use two independently attributable staff identities, **clinician** and **secretary**, with independently rotatable/revocable credentials (expand to individual named accounts later, never one shared key mislabeled as individual audit). Reuse the same global Cockpit login entry where possible to avoid two separate logins; the server resolves staff identity, issues an HttpOnly/Secure/SameSite=Strict, short-expiry signed session **bound to credential version and role**, no credentials in JS/browser local/session storage. Require adequate secret entropy and replay/failed-login protection; add MFA or an independently verified access-control boundary before identifiable data activation. The existing shared `CLINICAL_DATA_KEY` cannot authorize a staff-labelled Visit Brief read, even if it remains supported for existing physician-only clinical modules. Secretary's role never produces legacy clinician authorization.

| Operation | Clinician | Secretary | Anonymous or unqualified shared key |
| --- | --- | --- | --- |
| See bounded appointment context | allowed after staff auth | allowed after staff auth | denied for new staff surface |
| Search protected registry for identity check | only scoped booking-verification search | only scoped booking-verification search | denied |
| Explicitly confirm selected booking↔patient and receive bounded medical Visit Brief | allowed | allowed | denied |
| Reopen same booking UID/name without new independent confirmation | **denied** until upstream attendee continuity is qualified | **denied** | denied |
| Read bounded medical Visit Brief within *that* successful independent confirmation only | allowed | allowed | denied |
| Edit/sign clinical `EncounterORM` / Visit Capture Save / medication plan | existing clinician-only Clinical Data authority | **403** | denied |
| Link correction/unlink | explicit guarded operation, audit | only if separately authorized by clinic role policy; otherwise 403 | denied |
| Read staff-access audit / authorize policy-qualified administrative erasure | clinician only | denied | denied |

**Strict same-origin protected semantics:** no patient prefetch on home listing or search; no reusable clinical content via repeated GET/POST `read` merely because UID, booking name or visible fingerprint are equal. The exact successful `confirm` response alone may carry the bounded brief. Selection must be made against independent clinical identifiers; ID/name suggestion does not establish person continuity. A 401/403/503, logout, overlay close, booking switch and late response all invalidate PHI, as the existing P1/P2 corrected controller guarantees.

Extend existing `clinical_calendar.py` and `clinical_data.py` narrowly rather than weakening their global `protected` dependencies: secretary gets only appointment read and a dedicated booking-scoped registry lookup returning the minimum identifiers to perform independent verification; never open `/clinical/patients` or unrestricted `/clinical/patient/{patient_id}/encounters` globally to the secretary. Preserve clinician's existing protected source when staff auth is not configured, but **for the new booking-link clinical read, insist on independently attributed role identity** when enabled. Validate per-request actor identity on the server and never trust a client-supplied role or browser visibility.

## 4. Explicit access audit and denial

Implement append-only protected audit for successful/denied scoped clinical brief reads (including direct confirm response), selected patient administrative confirmation/correction/revocation and cleanup, recording timestamp, immutable staff principal ID/role, booking namespace+UID, patient ID **only if authorized**, event/outcome, and revision where relevant. No patient name, medical content, national ID, raw login credentials, auth cookie, full HTTP body or clinical candidate in logs or public CI artifacts. Clinical link audit must no longer falsely attribute actors as `shared_clinical_key`; consider schema migration for databases where tables already exist rather than assuming `create_all` will alter schema.

Secretary access needs clinician-governed permission; show staff a concise confidentiality/need-to-know cue but do not impose verbose repeated warnings. Clinician can view a bounded audit history; patient data never appears in URLs. Denials can record a non-identifiable event where appropriate; do not let an unauthenticated caller choose a patient or enumerate the registry.

## 5. Purpose-based technical-link lifecycle (not medical-record retention)

The protected `clinical_booking_patient_links` sidecar is distinct from clinical records and `clinical_booking_patient_link_audit`/staff-access audit. **No visit-count rule or automatic elapsed-time purge is approved.** A patient may attend only once or return much later.

- Preserve `PatientORM`, signed `EncounterORM`, decisions, medications and clinically relevant `ClinicalPendingORM` under independently established lawful medical-record retention. Never purge them as a technical booking-link side effect.
- Minimize the booking-specific, versioned technical association. Its existence **never** establishes attendee continuity: every new opening requires independent explicit registry patient verification; repeat same-UID/name reads remain booking-only. No history prefetch.
- Before identifiable activation, adopt a concrete clinic-specific purpose-based link-retention policy: record why a historical booking-specific association remains necessary, a practicable periodic review and erasure/de-identification trigger once no longer needed, accountable authorization, legal/data-subject holds and a distinct audit retention policy. No numerical deadline is inferred; silence is not permission for indefinite storage.
- **No automatic scheduler or cleanup based on visits, calendar time or inactivity in v1.** Never delete on brief read, clinical Save or unrelated appointment activity. Provide clinician-governed, reason-recorded link correction/revocation. **Revocation is not necessarily erasure:** existing `unlink` marks a state and may retain patient ID in the protected row and audit.
- If a later approved policy authorizes erasure, use a separate auditable idempotent operation with transactional version CAS and current booking checks. Protect active/future booking, correction in flight, open follow-up, legal/data-subject hold, and any identity uncertainty. Remove/de-identify the association when eligible; preserve a minimal stale-relink-resistant tombstone/revision and assess whether residual booking UID remains personal data.
- Apply independent lawful retention, limited access, accountability, review, redaction and hold handling to staff/link audit. Do not automatically erase required audit with a booking link.
- Never silently alter source appointments, signed medical history, plans or clinical pending. Fail closed if required policy/storage qualification is absent.

## 6. Startup, storage, privacy and activation gates

**Feature flag stays OFF by default:** `CLINICAL_VISIT_BRIEF_LINKS_ENABLED` remains `0`/unset until a documented Product Owner/private-clinic activation. Keep role-auth credential config independent, and fail closed when any mandatory role-audit/retention/private-DB policy is missing. No raw passwords or keys stored in GitHub, PR/CI, ChatGPT conversation or JS. Use private service configuration with individually provisioned staff secrets and an emergency revoke/rotate process.

**DO NOT claim private/durable merely from PostgreSQL dialect or a Render TLS URL.** Before production PHI use, verify connection location/owner, SSL configuration, backup restore, network/database user least privilege, at-rest encryption and approved data processing/region; if this remains unknowable to automated tooling, the gate stays UNKNOWN and no activation. Do not fetch real patient rows or disclose `DATABASE_URL`/keys in the review.

**DDL/migration:** do not silently rely on `create_all` to alter existing tables. Use a reviewed additive migration strategy that is safe to reapply to a private database, has rollback, never dumps patient records, and does not create live patient associations while the flag is OFF. The original `create_all` call requires targeted remediation or explicit documented acceptability before deploy.

**Deployment:** first ensure separate pre-code R2 PASS on this new R2 security design, then a bounded writer on the existing DRAFT PR or a stacked child PR; synthetic tests and post-code independent affected R2; guarded squash to current verified `main` only when the policy, credentials, private storage and feature readiness are actually confirmed. Existing Render auto-deploys `main`: do not trigger a duplicate manual deploy. Verify exact merge source LIVE, allow only anonymous no-PHI smoke and user-authorized logged-in synthetic verification; do not run real patient/provider operations as release smoke. If no verified private DB/role credentials/audit retention, **HOLD merge/deploy of the patient-link feature**, even if CI succeeds. No production configuration changes in a design-only session.

## 7. Finite synthetic security/regression proof

1. Auth: absent/wrong/legacy shared-only credentials ⇒ 401/403; independently attributable clinician+secretary ⇒ role-only allowed endpoints; rotate/revoke account denies old cookie; expiring session denied. No privilege escalation from forged `role` body, cookie or raw header.
2. Secretary reads and confirms selected medical Visit Brief, but **cannot** Save/sign/alter `EncounterORM`, medical plan or Clinical Data. Clinician still can; no shared-key actor spoofing.
3. Independent booking source unique UID/fresh coverage, one-time confirmation, no same-UID automatic repeat read or other patient history. 409 on changed observation or stale CAS. Exact booking-person mismatch fails closed. Audit identifies the real role actor for permitted reads and writes without raw PHI logs.
4. Synthetic one-visit and long-return-gap patients retain complete protected chart context; no automatic visit-count/elapsed-time purge; authorized correction versus true erasure; policy-unqualified erasure denied; active/future booking, pending, legal hold, changed UID, repeat authorized erasure, and stale client re-link fail safely. No medical history/plan/pending or required audit deletion.
5. Synthetic role/logout/stale network/DOM + actual Chromium at widths 1280, 680 and 390. Existing Home #160 and Visit Capture tests pass without broad auth weakening. No browser persistent PHI, no patient data in URL or public artifacts.
6. Disabled-feature boot/startup on non-verified storage, DB migration idempotence/rollback, no live secrets or provider calls in CI, new service settings OFF by default. Exact-head focused GitHub Actions verified separately before any release.

## 8. Review scope and stop

Request exactly one fresh independent **PRE-CODE R2** over: staff role trust boundary, secretary's scoped clinical summary visibility, booking-verified search/confirm, audit attribution, purpose-based technical-link lifecycle and separate audit retention, migration/storage and release gates. Reuse all previous Visit Brief design and P1/P2 post-code PASS. If the design receives PASS, implement and test only this affected slice, then one post-code affected R2. A technical PASS never grants live data processing without separate storage/policy credential qualification.

**References:** Cyprus Patients' Rights Law 1(I)/2005 as amended by 54(I)/2025, articles 15–18, https://www.cylaw.org/nomoi/enop/non-ind/2005_1_1/full.html ; GDPR 2016/679, arts 5/9/32.
