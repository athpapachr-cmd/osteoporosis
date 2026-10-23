# Cockpit Clinical Inbox Phase A + Future Multi-Account — Independent PRE-CODE R2

> **R2 historical handback (2026-10-23):** Original affected review returned **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:1**. Product Owner approved an author **design-only P1/P2 correction** at `7a79ec44ae010cfc68f87c39b6f2bc301fa3cc2c`. This request documents the **original** target; do not rerun the full Q1–Q6. Use the [one focused independent P1/P2 closure request](CLINICAL_INBOX_PHASE_A_MULTI_ACCOUNT_PRECODE_R2_P1_P2_CLOSURE_REQUEST_2026-10-23.md). **Independent closure is still PENDING.**

**Date:** 2026-10-23 (Asia/Nicosia).
**MODE:** ONE fresh independent READ-ONLY affected R2 contract/design review; review only this new bounded Phase-A data/persistence contract and future multi-mailbox source namespace. Reuse [original two-entrance pre-code R2 PASS](VISIT_BRIEF_CLINICAL_INBOX_PRECODE_R2_PASS_RECEIPT_2026-10-08.md) (Q1–Q6). STOP after PASS/BLOCK/UNKNOWN; do not code/merge/deploy/use mail or patient data.

**Repo:** `athpapachr-cmd/osteoporosis`.
**Base verified production main:** `32a87ec8757b21b64dd8837e362810b5a2b9b510`.
**PR #208:** https://github.com/athpapachr-cmd/osteoporosis/pull/208 (DRAFT, unmerged, design-only).
**Exact frozen DESIGN-bearing author HEAD:** `806f1523672a654e78a891ee12e330bb6d6eae65`; any later checkpoint must be documentation-only (verify Git compare).
**New design:** `cockpit/CLINICAL_INBOX_PHASE_A_MULTI_ACCOUNT_CONTRACT_2026-10-23.md`, and affected sections of `cockpit/VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`, `cockpit/PRODUCT_CONSTITUTION.md`, `CLINICAL_EXCELLENCE_PLAN.md`, `TODO.md`; exact PR author design head to be pinned after publication.
**Product Owner:** approved Phase A now, and explicitly asked to **later connect Gmail, iCloud and Hotmail/Outlook accounts**, without activating credentials now.

## Scope decision

Phase A: shared `ClinicalAttentionItem` state, appointment-independent Inbox and linked Visit Brief using **synthetic/manual typed sources only**. No Gmail, iCloud, Hotmail, Zadarma, credentials, provider intake, patient messaging, authoritative result/task mutation or deploy now. One source-neutral contract now prevents rebuilding the UI/core for future account adapters. The first external Gmail/Medicover vertical and then other accounts remain separate gated slices.

## Six finite R2 questions

1. **Owner reuse:** Does Clinical Excellence own only clinical-attention review/link state while existing `clinical_pending_one_tap` owns tasks, Reception owns delivery/communication and provider owns mailbox? Are proposed core and two entrances one lifecycle rather than duplicates?
2. **Scoped identity:** Does an opaque mailbox account + provider-native message/part/revision prevent cross-account collision, false dedupe (including forwarded mail), inferred clinical patient identification and mixed-patient attachment leakage? Are multiple accounts per provider supported?
3. **Phase-A write safety:** Are identity confirmation, clinician role, append-only actor/idempotency/version conflict, source revision invalidation and separate review vs accepted result/transport stated precisely enough to implement with existing protected source? Does any action silently create clinical facts/tasks?
4. **Privacy and retention:** Before persistent real patient/manual use, do required DDL/grants, access control, field-level retention/deletion and source encryption gates remain explicitly OPEN and fail-closed? Are synthetic fixture tests isolated and no raw email/PDF/tokens saved in repo/browser storage?
5. **UX/freshness:** Are Inbox (without booking), Visit Brief shared read, 100% source coverage vs stale-per-account and real/test/successful-key-switch/stale-401 isolation truthfully separated? Is unknown coverage never shown as no new items?
6. **Evidence and bounded next:** Do design+roadmap+current agree that Gmail-first is an integration order, **not** a hard-coded core type and not active, and that iCloud/Hotmail are postponed providers; is one bounded post-code R2 after code enough once pre-code gates close?

Return `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, P0:P1:P2, evidence by exact reviewed blobs, minimal correction, STOP. No runtime modification, no credentials, Gmail/Apple/Microsoft connection, Aiven, Render, clinician patient data or communications.
