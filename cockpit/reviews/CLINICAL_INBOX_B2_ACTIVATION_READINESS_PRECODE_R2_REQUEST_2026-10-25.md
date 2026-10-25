# Clinical Inbox B2 — ONE fresh independent affected PRE-CODE readiness R2

**Date:** 2026-10-25 (Asia/Nicosia). **MODE:** independent READ-ONLY, ONE bounded affected PRE-CODE R2, then STOP. NO code, merges, deployment, production DDL/grants, patient records, real issuer/ADT lookups, staff secrets, provider logins, mailbox or external communications.

Repository: `athpapachr-cmd/osteoporosis`.
- Production main at design start: `3204bc7722b5bb45241c5b23d788f6f51c6fb502`.
- B0 dependency DRAFT #214: `816d6dc9079909ddf82bce082ca8c8441dd38756`; independently accepted B0 pre-code scope.
- B1 dependency DRAFT #215: `87280499b029b2615cd064874d9071ad236da467`; Product Owner conveyed independently passed B1 correction closure; synthetic-only, no merge/release.
- B2 candidate: `cockpit/CLINICAL_INBOX_B2_ACTIVATION_READINESS_PRECODE_2026-10-25.md` on a new DRAFT branch stacked on #215. Verify **current** actual main/B0/B1/B2 HEAD and exact B2 blob before decision.
- Reuse frozen #210 issuer-qualified ADT/referral design, B0 PRE-CODE PASS, B1 R2 PASS, Phase-A and #208/#209/#212/#213 synthetic safety/UX oracles. Do NOT reopen unchanged decisions, manufacture a second full architecture chain, or demand extra clinician clicks.

## Exactly six affected questions

1. **True actor authority:** Does the plan correctly identify that `clinical_staff_auth.py` exposes role-wide rather than person-unique principals, `clinical_auth.py` and existing pending code can use shared Clinical Key, and new identifiable B2 actions must refuse all those fallback identities? Does it preserve unrelated legacy and qualified secretary Visit Brief READ?
2. **Actual issuer/referral evidence:** Does it refuse treating generated clinical patient ID, surgery identity number, ClinicalPending text, mailbox sender/name/date or bare referral as verified authority; require an actual controlled original issuer/source scope, unique current patient match, validity/revocation and conflict handling, with no mandatory DOB/fake evidence field?
3. **Safe ownership and lifecycle:** Are Clinical Excellence attention and append-only audit single owners, source/identity evidence only after qualification, real patient/backup/retention policy explicitly OPEN, and SQL startup DDL avoided? No unreviewed production migration or weakening of the B1 synthetic DB fence.
4. **Mutation and clinical meaning:** Atomic protected CAS/audit/idempotency, same-subject revision invalidation, mixed/unknown Inbox-only, one-click source inspection distinct from accepted medical results; no Brief/Pending/chart/result/outbound side effects before qualified activation.
5. **Finite first slice:** Does B2A consist ONLY of separately authorized **offline fake-person/fake-issuer actor/source contract and tests**, leaving real identity, secrets/config, PHP/PHI, production code routes, provider/mailbox and B2C authorization OFF? Is it small enough to implement/test without another generic redesign loop?
6. **Release stop and dependency:** Are B0 and B1 still DRAFT/unmerged, B2 branch stacked docs-only, and each real clinician auth/issuer proof/retention/backups/DDL/release/provider gate explicitly separate? Does B2 design PASS grant no merge, B2A code, real-data activation or Gmail/Outlook/iCloud transport?

## Required output

`PASS|BLOCK|UNKNOWN / COMPLETE_FOR_DECLARED_SCOPE|PARTIAL / P0:P1:P2` plus: independently verified current remote SHA and design blob, six dispositions with source/code references, one smallest specific correction if BLOCK, narrow missing fact if UNKNOWN, final decision whether the B2 design supports a later **separately approved B2A offline fake-only implementation**. **STOP READ-ONLY**. No code/merge/deploy or automatic B2 activation.
