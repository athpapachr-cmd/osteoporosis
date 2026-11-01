# Independent pre-code R2 closure — PASS

Reviewer: `icloud_triage_design_r2`, independent of design author. Date: 2026-11-01.

Exact corrected design target: `8c34467681b78a430cdae6742efb013f669e1aae`. Source main `b8637bcc8af0deebee3716cf7cd98bc99bf2e912`. Later status/receipt changes contain no material contract delta. Original target `65b665d3eccffc31bd48826a2953fd4d30815930` returned BLOCK0:0:1; this is its ONE bounded independent delta plus affected cumulative closure, not a new broad review.

**PASS / P0:P1:P2=0:0:0 / P2-01=CLOSED_AT_DESIGN_LEVEL / ALL_MATERIAL_FINDINGS_CLOSED=YES / COMPLETE_FOR_DECLARED_SCOPE=YES / NO_ADDITIONAL_MATERIAL_FINDING=YES within examined affected scope.**

## Independent closure evidence

- Corrected design82–83 binds category projections to mailbox scope, UIDVALIDITY and preference revision. Confirmed mutations revoke inbox AND shell peek/badge caches, reset paging and invalidate old request generations.
- Design84–85 requires fresh server qualification before cache reuse, including return from another tab without notifications. Notifications remain optional hints; old projections cannot override a confirmed newer revision.
- Design86 governs source/store qualification failure and existing transient-error cache fallback: an unqualified filtered list or empty badge cannot be presented as current.
- Design94 explicitly includes doctor-shell.js, covering the previously omitted consumer.
- Acceptance107 exercises both primed caches, empty clinical peek, a saved category/rule change, delayed successful old responses, second-tab return and qualification failure.
- This addresses existing `doctor-shell.js:412–417,532–576` and `icloud-inbox.js:510–538` without a second owner or changed auth/provider authority. Read-only All rollback (design118) remains consistent.

Original supported questions1–4 and unaffected parts of5–6 are reused. Operational storage/CAS/replay/undo, exact source identity, future-only rules, clinical exclusions and separate provisioning authority remain unchanged. No additional material finding in the affected scope examined.

## Limits and next authority

This PASS is a DESIGN closure, not an implementation, runtime, PostgreSQL/SQLite or browser validation. No reviewer edits, tests, provider/database access, publication, merge or deploy occurred. No production schema/permission/config/patient changes occurred. The finite review gate is CLOSED; do not review the review or reopen settled questions without a genuinely new material delta.

The approved plan is ready for a separate implementation decision. After implementation authority: smallest bounded runtime change, meaningful disposable SQLite/PostgreSQL and actual synthetic browser evidence, then ONE independent exact-code fidelity R2. Provisioning/merge/deploy remain separate authorized steps; do not auto-create tables at boot/GET or activate the old mail-action router.
