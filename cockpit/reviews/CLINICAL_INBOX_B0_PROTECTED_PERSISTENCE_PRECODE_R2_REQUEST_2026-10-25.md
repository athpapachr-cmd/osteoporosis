# Clinical Inbox B0 — ONE independent affected PRE-CODE security + identity R2

**Date:** 2026-10-25 Asia/Nicosia. **MODE:** fresh independent READ-ONLY affected PRE-CODE R2, one bounded pass and STOP. No new architecture chain, code changes, DDL, Aiven/Render configuration, GitHub merge/deploy, credentials, patient rows, real provider connection, actual PHI or external communication.

**Repository:** `athpapachr-cmd/osteoporosis`
**Baseline production main at B0 start:** `3204bc7722b5bb45241c5b23d788f6f51c6fb502` (PR #213 guarded merged, no runtime changes in this B0).
**Source-of-truth candidate:** `cockpit/CLINICAL_INBOX_B0_PROTECTED_PERSISTENCE_PRECODE_DESIGN_2026-10-25.md` in the proposed B0 design PR. Reviewer MUST resolve and report its **exact blob SHA** at review time and compare latest PR HEAD with design freeze for any substantive changes.
**Reused approved design:** [#210](https://github.com/athpapachr-cmd/osteoporosis/pull/210), substantive blob `4f1f1111814d841989954ec0d6095635d8f2ccd7`, independent affected R2 PASS 0:0:0; Phase-A contract `cockpit/CLINICAL_INBOX_PHASE_A_MULTI_ACCOUNT_CONTRACT_2026-10-23.md`; released #208, #209, #212 and #213. Keep their exact completed security/UX/correction regression findings CLOSED. New identity/retention/persistence gates are not approved merely by previous synthetic PASS.

## Current implementation evidence to inspect (only where needed)
- `clinical_data.py`: `PatientORM` internal ID, `ClinicalPendingORM` obligation, `build_clinical_router:ClinicalBase.metadata.create_all` **boot-time DDL risk**.
- `clinical_staff_auth.py`: separate staff roles, default OFF, static role-wide principals; `clinical_auth.py`: remaining legacy shared Clinical Key path.
- `clinical_visit_brief.py`: bounded booking/patient link and staff audit; no inferred source identity.
- `clinical_synthetic_161.py` and `clinical_attention_phase_a.py`: safe synthetic-only current model, corrections, server-qualified fictional evidence, no persistent real attention.
- `clinical_surgery_queue.py`: distinct surgical queue `identity_number` is NOT a qualified clinical identity link.
- `cockpit/migrations/20261017_visit_brief_atomic_clinic_install.sql`, `20261017_visit_brief_cockpit_app_grants.sql`, `20261018_clinical_pending_checkpoints.sql`: migration/least-grant patterns, not B permissions.
- `cockpit/COCKPIT_VISIT_BRIEF_ACCESS_RETENTION_POLICY_DECISION_2026-10-17.md`: clinician + secretary limited Visit Brief read, distinct clinical write authority, no invented numeric retention.
- Existing topbar/Inbox/Brief JS and protected 401 tests only for affected seams; do not restart unchanged testing.

## Exactly six questions — affected boundary only

1. **One owner and minimal B storage:** Is proposed attention state + append-only audit owned once by Clinical Excellence, using existing patient and Visit Brief authority, without a competing Pending task/mailbox writer? Are any extra identity/referral tables conditioned on proof of qualified source ownership rather than assumed?
2. **Safety and one-click identity:** Does a prior clinician-confirmed exact issuer-qualified referral (or unique authoritative ADT) permit no typing and at most one confirmation, while weak names, dates, free-text Pending, surgery ADT, sender, mixed subjects, duplicate/conflict and revoked proof fail closed? No premature clinical result promotion?
3. **Attribution & role enforcement:** Does the activation gate correctly distinguish staff-role code presence from production ENABLED plus attributable human actor; explicitly reject role-wide/static principal or legacy key as sufficient for real ID verification; preserve secretary's existing limited Visit Brief read without granting medical review/link mutation?
4. **Controlled persistence & concurrency:** Are startup `create_all` risk and explicit migration, PostgreSQL schema preflight/grants/audit, CAS/idempotency and current source revision consumer rules fully bounded? Cold-start/restart state and current vs superseded session effects addressed?
5. **Minimization & retention:** Do plaintext ADT/body/PDF avoidance, controlled keyed equality lookups, private storage/backup, actor view audit, revoked-evidence semantics, purpose-based lifecycle including backup copies and legal hold remain hard pre-activation requirements without invented deletion duration?
6. **Practical staged delivery:** Does B0 authorize **only docs**; B1 remain synthetic/empty feature OFF under separate implementation grant; B2 real patient/identifier activation require own security and PO authority; C providers independently gated? Does normal Inbox remain honest until live coverage is proven?

## Reviewer output

Return `PASS|BLOCK|UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE|PARTIAL`, `P0:P1:P2`, six question dispositions, current main/PR HEAD, **exact design blob**, focused source references and whether B0 itself is independently safe to hand back as a design freeze candidate. If BLOCK: identify **one concrete reachable material flaw** and its smallest affected correction; no speculative second design, no new review chain. If UNKNOWN: exact missing evidence/owner and narrow verification. **STOP READ-ONLY.**

**Regardless of PASS:** Do not write code or schema, activate real identities, turn on staff auth, run migrations, read patient rows, connect email, merge or deploy. Any B1 implementation requires separately explicit clinician authorization and fresh writer gate. Independent reviewer may not self-grant implementation authority.
