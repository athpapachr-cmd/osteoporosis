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
| Source | `source_receipt_ref` opaque optional; `mailbox_account_ref` nullable opaque; `logical_source_item_key` stable/version-independent; `source_type = lab / imaging_mri / imaging_xray / gesy_notice / other`; `active_source_revision` opaque/versioned |
| Time | source-stated received timestamp (nullable/precision-marked), first intake/visible timestamp, clinician-review timestamp; no inferred specimen date or clinical deadline |
| Clinical subject | source-stated subject `candidate_only`, `source_subject_scope = unknown / single_subject / mixed_subjects`, `patient_id` nullable, `identity_state` and explicit `link_evidence_kind`/`link_actor`/`link_at`; patient name/email alone cannot link |
| Candidate/review | typed candidate source ref/preview status, `review_state`, `disposition`, sanitized evidence reference and `clinical_actor`/`reviewed_at`; no raw body/PDF in public/browser persistence |
| Concurrency | `version` monotonically increments on patient-link/review/disposition changes, `expected_version` on every mutation, idempotency key + actor-bound append-only audit, conflict 409 |
| Cross-products | optional references to existing clinical pending task/Visit Brief and Reception communication action; never make an Inbox record the owner of task fulfillment or SMS delivery |

Separate **identity**, **review**, **disposition**, and **transport** state axes; `reviewed` does not equal accepted result, clinical completion or communicated. A source correction, patient re-link or revision invalidates dependent unsubmitted approvals. One verified logical document scope maps to one stable attention ID across its revisions; revision-specific delivery receipts remain distinct append-only provenance. The exact required atomic transition and uncertain-source fallback are in §3A. Deduped source references may have multiple display views but must retain one authoritative attention ID per explicitly scoped source item.

**Auth / authorization:** reuse existing protected clinical session/staff clinician checks and existing patient-registry exact-ID confirmation. Secretary, anonymous users, invalid/expired cookies and stale sessions cannot mutate review/link/disposition. Do not place PHI in GET URLs, logs, source docs, localStorage, browser history or GitHub. UI clears patient text/modal/action state on logout, successful principal switch (even without logout), modal close/ESC and scope switch; pending responses have generation + item + session guard before any side effect. Genuine 401 remains fail-closed; stale 401 must not log out the new principal (reuse #207 regression learning).

**Persistence activation gate (currently OPEN):** before enabling actual patient/manual entries, explicitly confirm private storage table ownership, migration/grants, field retention/deletion, clinician actor audit, role mapping, read/write/cross-session flows and recovery. Synthetic fixtures can use isolated SQLite/test only. No database DDL, long-lived real candidate storage or migration is authorized in this design step. No invented fixed retention duration.

## 3. SourceReceipt namespace for the later all-account integration

Reception always-on source orchestration is **one transport owner** with separately verified adapters. The clinical store receives only immutable, minimal source references and clinical review states.

| Future field | Contract |
|---|---|
| `mailbox_account_id` | Opaque unique per connected account (one or many Gmail, iCloud Mail, Hotmail/Outlook addresses). Never infer clinical patient identity |
| `provider_family` | `gmail`, `icloud_mail`, `microsoft_outlook` (extensible, provider selected by approved connection) |
| `provider_native_message_locator` | Provider/account-scoped locator: Gmail message ID; qualified Microsoft Graph immutable ID; or iCloud IMAP `(folder_scope, UIDVALIDITY, UID)`; native forms are NOT interchangeable |
| `provider_part_id / content_revision` | Distinct document/attachment part scope and content version; revision never silently creates a second clinical attention item when the stable logical document is verified |
| `logical_source_item_key` | Stable version-independent `(account + normalized provider message locator + part + verified subject segment scope)`; for manual items, immutable server-issued item identity and create-idempotency; never inferred by subject/name/date/hash alone |
| `source_receipt_version_key` | `(logical_source_item_key + content_revision)` identifies one delivered revision; repeat is idempotent, a new verified revision updates the same item under conflict control |
| `account_coverage` | Per-account last-complete scan, cursor/UID validity, supported folder/label filters, lookback, latency, retry/disconnected/permission state |
| `delivery` | Outbox pending/delivered/acknowledged or unknown; no lost item on Cockpit free-service cold start |

Aggregate Inbox freshness is **complete** only if each enabled approved account's coverage is complete. If any source is stale/unavailable, show explicit per-account gap; do not mislabel a partial queue as "all new messages handled". Existing clinical attention already reviewed/linked remains subject to the protected retention policy after source disconnect; disconnect one account without disabling others.

A forwarded/copied message across two accounts has **two receipts** (may be visually grouped only by deliberate review), not automatic conflation. One email may have several patients/attachments; each document scope needs its own identity confirmation. Source provider read/unread never becomes clinical review/completion.

**Integration candidates, later security review only:** Gmail API `gmail.readonly` scopes and sync cursor; Microsoft personal @hotmail/@outlook addresses via delegated Microsoft Graph, independently tested; iCloud Mail via supported account authorization or app-specific credential and secure IMAP if technically appropriate. Provider-specific authentication, encryption/secrets, authorized folder filters, GDPR/processor/retention, API quota, incremental recovery and cold-start delivery gates are **OPEN**; no credentials supplied or stored here.

## 3A. Focused P1/P2 correction — source revision and per-patient isolation

**P1 — two different identities, two different purposes.** `logical_source_item_key` identifies one **proven** source document scope throughout its versions; `source_receipt_version_key` identifies an individual immutable version/delivery of it. A provider-backed logical key is scoped by authorized `mailbox_account_id`, provider-native message locator, document part and (only when independently verified) subject segment. A synthetic/clinician-manual item uses an immutable server-issued item identity and actor/payload-bound create idempotency, never deduplication by free text, sender, document name, hash or arrival time. For a separate reissued email/forward or ambiguous provider ID, **do not** assume revision continuity: preserve separate receipts/items and require evidence-based clinician reconciliation without merging histories or actions.

On a **verified** new revision of the same document scope, Clinical Excellence atomically locks/CAS-updates the one `ClinicalAttentionItem` (`expected_version`, otherwise HTTP 409), increments `version`, preserves append-only source/audit history, sets the active revision and `review_state=needs_re_review`, and marks older extracted candidates/current dispositions/unsent approvals stale pending fresh clinician review. A change of subject identity or mixed/single-subject scope also invalidates the old protected patient projection/link until re-confirmed. A same-revision receipt replay is exactly-once without a second item, audit mutation or clinician action. A concurrent review/approval against the superseded version fails closed (409). Already-submitted communications retain their actual historical delivery/unknown outcome and require clinician assessment; **never** erase, silently re-approve, automatically resend or treat them as accepted clinical results.

**P2 — provider locator and subject scope.** For IMAP, native identity is **(authorized account, selected folder/mailbox scope, UIDVALIDITY, UID)**, not UID or UIDVALIDITY alone; a folder rename, move or UIDVALIDITY reset requires scoped recovery with uncertain correlations left separate, not an empty-success scan or unproven merge. For Microsoft Graph, the future adapter must qualify `Prefer: IdType="ImmutableId"` for message/attachment reads and change notifications where supported, and handle its documented lifetime exceptions; if stable ID semantics cannot be proven, movement/recreation triggers explicit reconciliation rather than accidental deduplication. Gmail IDs remain account-scoped. Provider-specific native identity, locator normalization, revision provenance and recovery require **separate future provider R2** before activation; this contract neither activates nor tests a provider.

An email/message or PDF can contain different patients even within **one attachment**. An `unknown` or `mixed_subjects` item is **Inbox-only** until the clinician confirms an evidence-backed, non-leaking subject-specific document/segment and protected recipient identity. Never show the entire mixed-patient file or unsegmented candidate in one patient's chart/Visit Brief; do not silently fan out to multiple patients. Any clinician-approved segmented source must have separate source/segment reference, access boundary, explicit per-subject patient confirmation and revision invalidation; if safe segmentation cannot be proven, remain unlinked with no patient-specific projection, authoritative write or communication. No source-derived name, sender or account address proves identity.

**Phase-A test boundary:** synthetic provider-shaped IDs and mixed-subject fixtures can verify these invariants in isolated tests; real IMAP/Graph/Gmail account access, provider qualification, protected patient persistence and outbound effects stay excluded.

Reference for deferred adapters: [RFC 9051 §2.3.1.1](https://www.rfc-editor.org/rfc/rfc9051.html#section-2.3.1.1) and [Microsoft Graph immutable IDs](https://learn.microsoft.com/en-us/graph/outlook-immutable-id).

## 4. Finite Phase-A evidence and gates

1. Two independent entrances (appointmentless Inbox and appointment Visit Brief) refer to the *same* protected item; no duplicated clinical task.
2. Synthetic manual item opens and remains **unreviewed** until explicit clinician action; patient link requires strong approved evidence/confirmation; cross-patient/unlinked/secretary access fails closed.
3. Review/action state persists across sessions only after a reviewed schema+retention/migration gate; same-key replay is exactly once, stale/version conflict 409, source correction requires re-review. Before that gate, no PHI fixture or production writer is allowed.
4. Close/ESC/logout/account switch clears protected display and side effects; late 401 from old request cannot log out replacement user.
5. Multiple future accounts may coexist under independent namespaces/cursors; one disconnected account cannot masquerade as complete coverage or erase another's clinical history.
6. At 1280/680/390 responsive widths, Inbox is a first-class path with source identity and honest coverage; patient chart navigation remains explicit; no fake badge.
7. First Gmail live vertical and subsequent iCloud/Hotmail integrations each require their own provider/security R2 and end-to-end delivery proof. The Phase-A synthetic/manual result must not claim live email intake.
8. Synthetic replay of the same revision is one item/one effect; a verified revised report retains its `item_id` but requires re-review, invalidates unsent approval, and rejects stale concurrent clinician action with 409. A separate forwarded/reissued email never merges on filename/sender/content alone.
9. Synthetic IMAP same UID across different folders/UIDVALIDITY and Graph moved-message IDs cannot cross-dedupe; a reset/move shows incomplete recovery and no false `no new items` coverage.
10. Synthetic mixed-patient PDF/message never attaches its complete source or other patient's segment to one patient; without clinician-verified safe segmentation it stays Inbox-only and cannot trigger patient-specific contact.

**Review tier:** R2 pre-code affected delta, because protected patient identity, new persistent attention states and future mailbox account scope are material. Reuse Q1–Q6 original R2 PASS; assess only fields/permissions/retention and cross-account source identity. **No implementation until independent focused pre-code R2 and separate scoped writer authorization.** No release/clinical data/provider authority inferred from this candidate.
