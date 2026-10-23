# Clinical Inbox Phase A + Future Multi-Account — Focused PRE-CODE R2 P1/P2 Correction Closure

**Date:** 2026-10-23 (Asia/Nicosia).
**MODE:** ONE fresh independent **READ-ONLY DELTA + AFFECTED-CUMULATIVE** design-contract closure. Review only the two original material findings and their directly affected owner/identity/privacy consumers. **STOP** after verdict. The author of the correction must NOT certify their own closure.

**Repository:** `athpapachr-cmd/osteoporosis`
**Verified production base main at correction:** `32a87ec8757b21b64dd8837e362810b5a2b9b510`
**PR:** [#208](https://github.com/athpapachr-cmd/osteoporosis/pull/208) — DRAFT, unmerged, design-only
**Original frozen design HEAD:** `806f1523672a654e78a891ee12e330bb6d6eae65`
**Original review-request checkpoint:** `bd86b409d588b137ea04b6505cc2b4f50c6a0a26`
**Corrected substantive DESIGN HEAD:** `7a79ec44ae010cfc68f87c39b6f2bc301fa3cc2c`
**Corrected contract blob:** `79696ff1972c12d2156c9c7bc0741db798cf11ad`
**Corrected affected two-entry design blob:** `9f24c93d7065f858e20d922f5f706393fce40732`
**Historical original R2 request:** [2026-10-23 affected R2](CLINICAL_INBOX_PHASE_A_MULTI_ACCOUNT_PRECODE_R2_REQUEST_2026-10-23.md)
**Reuse:** [2026-10-08 pre-code R2 PASS receipt](VISIT_BRIEF_CLINICAL_INBOX_PRECODE_R2_PASS_RECEIPT_2026-10-08.md) Q1–Q6 of original two-entry design; no re-review.

## Source of closure

The 2026-10-23 independent read-only R2 handback returned **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:1:1** on the prior contract. Product Owner instructed the smallest **design-only** correction, which the author committed at corrected design HEAD above. That author correction is **NOT a PASS** and gives no runtime/release authority.

- **P1 — stable item identity versus revision receipt:** The old `source_dedupe_key` included revision while promising one stable `ClinicalAttentionItem`. Without a verified logical identity, a corrected source could create a duplicate record or leave outdated `reviewed` / approved action state. Review the stable `logical_source_item_key` and distinct `source_receipt_version_key` plus exact revision transition.
- **P2 — provider-native identity and mixed-patient leakage:** IMAP UID/UIDVALIDITY without folder scope is not sufficient; Graph message IDs can change on moves without immutable-ID qualification. A message, even one PDF, can contain multiple patients; no patient-specific projection/communication may leak another subject's content.

## Four finite closure questions

1. **P1 logical/version contract:** At the two corrected design blobs, does one **proven** source document scope retain the same opaque attention ID across actual revisions, while each immutable version/delivery retains its own idempotent receipt key? Do forwarded/reissued, cross-account or otherwise unproven messages stay separate, and manual items avoid weak text/hash deduplication?
2. **P1 atomic safety:** On a verified new revision, do CAS/record `version`, append-only audit, stale clinician 409, `needs_re_review`, old candidate/disposition/unsent approval invalidation and subject-link re-confirmation prevent stale clinical state or side effects? Are already-submitted messages historical with unknown outcomes preserved and never blindly resent?
3. **P2 provider scope:** Are IMAP account + folder/mailbox scope + UIDVALIDITY + UID explicit, including UIDVALIDITY reset/moves and incomplete recovery? Are Graph immutable IDs **deferred provider qualification**, with safe fallback when stable IDs cannot be guaranteed, and no early provider/credential activation?
4. **P2 mixed-subject and affected cumulative:** Does an unknown/mixed-patient document remain **Inbox-only**, with no whole-PDF/other-subject patient chart or Visit Brief exposure, no auto fanout or message, and explicitly separately verified safe subject segment before any patient-specific link? Check consistency of corrected contract with the original two-entry design, core/Reception/Pending owners, Phase-A **synthetic/manual-only** and per-account freshness isolation. Do not reopen settled original Q1–Q6 absent directly conflicting delta evidence.

**Evidence boundaries:** Compare `806f152...` → `7a79ec...`; verify any later PR HEAD is only documentation/current/review checkpoints with Git compare. Inspect exactly the two corrected contract/design blobs and affected root/workstream/current/review pointers. Synthetic future acceptance *oracles* are design requirements, **not** executed tests or live-provider qualification. No test, build, provider, Aiven, Render, Gmail/Apple/Microsoft account, database DDL, PHI or patient message execution.

**Return format:** `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, each original P1/P2 `CLOSED | OPEN | UNKNOWN`, exact evidence/blob references, direct new affected risk (if any), minimal correction if blocked, and **STOP**. No merge, code, deployment or release.

**After a true independent PASS only:** Product Owner may separately authorize a bounded **Phase A synthetic/manual implementation writer**; run one bounded post-code R2 fidelity review after implementation. Persistence schema/grants/roles/retention/PHI activation and future Gmail, iCloud, Hotmail/Outlook provider gates remain OPEN independently.
