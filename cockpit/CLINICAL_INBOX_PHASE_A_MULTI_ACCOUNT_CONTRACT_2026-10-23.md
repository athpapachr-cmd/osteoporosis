# Clinical Inbox Phase A — Shared Clinical Attention + Multi-Mailbox-Ready Contract

**Date:** 2026-10-23 (Asia/Nicosia). **State:** BOUNDED PRE-CODE DESIGN CANDIDATE; NO RUNTIME/PROVIDER/CLINICAL DATA AUTHORITY.
**Repo:** `athpapachr-cmd/osteoporosis`. **Base main:** `32a87ec8757b21b64dd8837e362810b5a2b9b510`.
**Product approval:** Clinician accepted first-step Clinical Inbox development and required FUTURE Gmail + iCloud Mail + Hotmail/Outlook account coverage.
**Reuse:** Existing independent [two-entrance design](VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md) / [independent R2 pre-code PASS](reviews/VISIT_BRIEF_CLINICAL_INBOX_PRECODE_R2_PASS_RECEIPT_2026-10-08.md); `clinical_visit_brief.py`, `static/cockpit/visit-brief.js`, protected clinical identity and G3 long-term summaries, `clinical_pending_one_tap.py` is distinct existing clinical obligation writer. No second mailbox/clinical task owner.

## 1. Clinician-visible bounded A

Click top ✉ **Κλινικά εισερχόμενα** from Home without an appointment. See a compact queue of typed **synthetic/manual** clinical signals and full view with exact status (unreviewed, needs review, reviewed, unresolved), received/source time, known/unknown identity, document availability and relevant pending action. The latest three/four *eligible* attention items may appear on Home only if proven source coverage and protected data are available. Do not invent unread counts.

Example: a **synthetic** MRI report at 09:17, no current appointment. Clinician can inspect its source context, explicitly confirm a protected patient link when actually supported, and mark a review disposition without changing original visit data. A later linked Visit Brief consumes the **same attention item ID/state**. Opening a row alone does not mark it reviewed, accept extracted results, create tasks, send messages or change a booking.

**Phase A exclusions:** live Gmail/iCloud/Hotmail; OAuth/IMAP; Reception provider intake; Gmail label reads; Zadarma outbound SMS; accepted clinical lab/MRI writes; automatic patient matching, treatment advice, generated tasks/appointments and clinical record changes from unconfirmed sources. If A is later released before provider B, label it as *manual/synthetic Clinical Inbox* and **never** as a live mailbox monitor.

## 2. Stable clinical attention owner (proposed A persistence contract)

One private **Clinical Excellence** attention store, not `ClinicalPendingORM`, Gmail or a second Visit Brief history. Proposed minimal `ClinicalAttentionItem` columns (not yet a migration):

| Field family | Minimal fields / invariants |
|---|---|
| Identity | `item_id` opaque stable UUID; `origin_kind = synthetic_fixture / clinician_manual / external_delivery`; `schema_version` |
| Source | `source_receipt_ref` opaque optional; `mailbox_account_ref` nullable opaque; `source_type = lab / imaging_mri / imaging_xray / gesy_notice / other`; `source_revision` opaque/versioned |
| Time | source-stated received timestamp (nullable/precision-marked), first intake/visible timestamp, clinician-review timestamp; no inferred specimen date or clinical deadline |
| Clinical subject | source-stated subject `candidate_only`, `patient_id` nullable, `identity_state` and explicit `link_evidence_kind`/`link_actor`/`link_at`; patient name/email alone cannot link |
| Candidate/review | typed candidate source ref/preview status, `review_state`, `disposition`, sanitized evidence reference and `clinical_actor`/`reviewed_at`; no raw body/PDF in public/browser persistence |
| Concurrency | `version` monotonically increments on patient-link/review/disposition changes, `expected_version` on every mutation, idempotency key + actor-bound append-only audit, conflict 409 |
| Cross-products | optional references to existing clinical pending task/Visit Brief and Reception communication action; never make an Inbox record the owner of task fulfillment or SMS delivery |

Separate **identity**, **review**, **disposition**, and **transport** state axes; `reviewed` does not equal accepted result, clinical completion or communicated. A source correction, patient re-link or revision invalidates dependent unsubmitted approvals. Deduped source references may have multiple display views but must retain one authoritative attention ID per explicitly scoped source item.

**Auth / authorization:** reuse existing protected clinical session/staff clinician checks and existing patient-registry exact-ID confirmation. Secretary, anonymous users, invalid/expired cookies and stale sessions cannot mutate review/link/disposition. Do not place PHI in GET URLs, logs, source docs, localStorage, browser history or GitHub. UI clears patient text/modal/action state on logout, successful principal switch (even without logout), modal close/ESC and scope switch; pending responses have generation + item + session guard before any side effect. Genuine 401 remains fail-closed; stale 401 must not log out the new principal (reuse #207 regression learning).

**Persistence activation gate (currently OPEN):** before enabling actual patient/manual entries, explicitly confirm private storage table ownership, migration/grants, field retention/deletion, clinician actor audit, role mapping, read/write/cross-session flows and recovery. Synthetic fixtures can use isolated SQLite/test only. No database DDL, long-lived real candidate storage or migration is authorized in this design step. No invented fixed retention duration.

## 3. SourceReceipt namespace for the later all-account integration

Reception always-on source orchestration is **one transport owner** with separately verified adapters. The clinical store receives only immutable, minimal source references and clinical review states.

| Future field | Contract |
|---|---|
| `mailbox_account_id` | Opaque unique per connected account (one or many Gmail, iCloud Mail, Hotmail/Outlook addresses). Never infer clinical patient identity |
| `provider_family` | `gmail`, `icloud_mail`, `microsoft_outlook` (extensible, provider selected by approved connection) |
| `provider_native_message_id` | Opaque provider/account-scoped key; Gmail ID, Microsoft Graph message ID or iCloud IMAP UID/UIDVALIDITY are NOT interchangeable |
| `provider_part_id / content_revision` | Binds attachment/document granularity and corrected/reissued report; revisions require re-review |
| `source_dedupe_key` | Namespace = account ID + provider-native message + part + revision; never global de-duplication by subject, email sender or date |
| `account_coverage` | Per-account last-complete scan, cursor/UID validity, supported folder/label filters, lookback, latency, retry/disconnected/permission state |
| `delivery` | Outbox pending/delivered/acknowledged or unknown; no lost item on Cockpit free-service cold start |

Aggregate Inbox freshness is **complete** only if each enabled approved account's coverage is complete. If any source is stale/unavailable, show explicit per-account gap; do not mislabel a partial queue as "all new messages handled". Existing clinical attention already reviewed/linked remains subject to the protected retention policy after source disconnect; disconnect one account without disabling others.

A forwarded/copied message across two accounts has **two receipts** (may be visually grouped only by deliberate review), not automatic conflation. One email may have several patients/attachments; each document scope needs its own identity confirmation. Source provider read/unread never becomes clinical review/completion.

**Integration candidates, later security review only:** Gmail API `gmail.readonly` scopes and sync cursor; Microsoft personal @hotmail/@outlook addresses via delegated Microsoft Graph, independently tested; iCloud Mail via supported account authorization or app-specific credential and secure IMAP if technically appropriate. Provider-specific authentication, encryption/secrets, authorized folder filters, GDPR/processor/retention, API quota, incremental recovery and cold-start delivery gates are **OPEN**; no credentials supplied or stored here.

## 4. Finite Phase-A evidence and gates

1. Two independent entrances (appointmentless Inbox and appointment Visit Brief) refer to the *same* protected item; no duplicated clinical task.
2. Synthetic manual item opens and remains **unreviewed** until explicit clinician action; patient link requires strong approved evidence/confirmation; cross-patient/unlinked/secretary access fails closed.
3. Review/action state persists across sessions only after a reviewed schema+retention/migration gate; same-key replay is exactly once, stale/version conflict 409, source correction requires re-review. Before that gate, no PHI fixture or production writer is allowed.
4. Close/ESC/logout/account switch clears protected display and side effects; late 401 from old request cannot log out replacement user.
5. Multiple future accounts may coexist under independent namespaces/cursors; one disconnected account cannot masquerade as complete coverage or erase another's clinical history.
6. At 1280/680/390 responsive widths, Inbox is a first-class path with source identity and honest coverage; patient chart navigation remains explicit; no fake badge.
7. First Gmail live vertical and subsequent iCloud/Hotmail integrations each require their own provider/security R2 and end-to-end delivery proof. The Phase-A synthetic/manual result must not claim live email intake.

**Review tier:** R2 pre-code affected delta, because protected patient identity, new persistent attention states and future mailbox account scope are material. Reuse Q1–Q6 original R2 PASS; assess only fields/permissions/retention and cross-account source identity. **No implementation until independent focused pre-code R2 and separate scoped writer authorization.** No release/clinical data/provider authority inferred from this candidate.
