# Cockpit — Clinician-controlled Obligation Category and Due Date v1

**Date 2026-10-22 (Asia/Nicosia). Product Owner real-use correction.**
**Scope tier R2:** Changes authoritative structured fields on an existing signed-visit clinical obligation. Independent focused pre-code design and post-code implementation reviews are required before merging, regardless of synthetic author CI.

## Problem, clinician-visible example and decision

The first-class obligations screen (#205, installed LIVE) can only display a task's recorded category and ISO `review_date`. Real tasks titled *«αναλύσεις σε 1,5 μήνα»* or *«DXA σε 1,5 χρόνο»* may be `open` with no `review_date`. This is not overdue status: `open` is task lifecycle, distinct from when follow-up is planned. Doctor must be able to adjust **category and due date** without re-signing an entire visit, replacing the task identity, or modifying signed source text.

**One-click affordance:** Existing obligations row shows an «Επεξεργασία» action in addition to the normal open-chart arrow. It opens the **existing protected patient Pending popup** with inline metadata editor already expanded. In an ordinary chart popup the same editor is reachable by «Κατηγορία / Προθεσμία». Fields: drop-down with known categories and a real calendar date picker, Save/Cancel and optional Clear Date. Immediate truthful refresh uses the same existing patient and /clinical-updates read owners. Status stays unchanged, original task text and stored signed visit stay immutable.

**Suggested relative dates (non-authoritative):** Only from a specific time expression in *the obligation title*, and only when an exact signed source-encounter `source_date` is available. For example «σε 1,5 μήνα» may suggest encounter date plus **one calendar month + approximately 15 days**; «σε 1,5 χρόνο» suggests plus **18 calendar months**. These are displayed as *suggestions*, not stored automatically. The doctor must press Use Suggestion and then Save. No open/completed inference, no back-dated comparison to today's date, no parsing from email or generic NLP. Unknown/ambiguous interval → no suggestion, manual date only. Dates use Cyprus-local calendar semantics; input and validation are strict ISO `YYYY-MM-DD`.

## Ownership and persistence — reuse, not duplicate

- Existing `ClinicalPendingORM` already owns `item_type` and `review_date`; retain row ID, patient ID, source encounter ID, original description, status, source excerpt, provenance, resolution link and checkpoints.
- Add **one** POST `/clinical/patient/{patient_id}/pending/{pending_id}/metadata` into existing `clinical_pending_one_tap.attach_pending_one_tap_routes` (not a new database or task owner). Body contains full desired category/date, expected *existing* lifecycle event revision and idempotency key; user-origin explicit Save only.
- Existing protected clinical key/session dependency plus `actor_for` rules, staff cookie precedence, signed provenance via `_signed_source`, exact patient/pending match, row lock, active-status eligibility, optimistic revision and idempotent retry. Invalid/non-clinician roles/anonymous/other patient/terminal deleted or cancelled → fail closed.
- Durable change is an **atomic** update to the two existing columns and a `metadata_corrected` event in the *existing* append-only `clinical_pending_checkpoint_events`, with before/after values, actor identity/role and timestamp. Uses existing UPDATE/INSERT grants and unique revision/index constraints. No new DDL, PostgreSQL user or deployment secret.
- On conflict, return 409 and retain user form with a reload instruction; never blindly retry a possibly committed POST using a new idempotency key. Replays of same exact request key and payload return success; conflicting reuse of key 409.
- Do **not** rewrite `EncounterORM.payload_json`, signed visit candidate, immutable source details, original task title, source provenance, or earlier audit event. Updating metadata does not cancel, resolve, recreate or change the task's clinical checkpoint stages.
- The existing signed-visit amendment may be performed later. Its token includes current pending item fields and revision and therefore rejects stale edits before reconciliation. Its existing source-item binding compares original signed values; a pending task that was not changed in the amendment remains bound to its original task ID despite later metadata overrides.
- Only known categories: test/investigation, result, document, follow_up, referral and other. Keep the legacy `investigation` value when selecting the same displayed “Εξετάσεις” category and editing only the deadline; explicit category change is saved as selected.
- Existing `project_checkpoint.history` must not mislabel metadata edit as clinical receipt/completion. The audit remains append-only regardless of what UI history chooses to display.
- Existing `GET /clinical/clinical-updates?active_only=true` currently limits to 100 most recently updated records; metadata edit changes `updated_at` and affects existing sort by design. UI continues to disclose incomplete global coverage.

## Failure/UX/security guardrails

The editor is present only for the clinician's allowed record in the existing Pending popup. The list never sends a write without explicit Save. A screen switch/logout clears proposed metadata fields and any sensitive patient text. A pending network response after navigation is not allowed to re-show or edit another patient's record. The task title is NOT itself a clinical deadline; suggestion is never persisted unless manually confirmed.

No separate Gmail, reception, clinical inbox, new role, schema or timer. No special AI call. No new PHI transfer.

## Finite acceptance evidence

1. One persisted metadata correction changes category/date in chart and updates, including independent second session; same Pending ID/source/status/step confirmation and unchanged signed visit, source excerpt/provenance. Changed or cleared due date reflected in grouping.
2. Wrong patient, invalid date (e.g., 2026-02-30), disallowed category, anonymous, invalid/secretary cookie, cancelled/deleted, signed source unavailable, uninstalled append-only table, stale revision, same-key conflicting payload → fail closed. Exact replay returns recorded success without second write/event.
3. Cross-operation concurrency with checkpoint-plan, completion, undo, cancellation, restore and signed-visit amendment retains shared revision and no stale resurrection; no guessed completion after setting future deadline.
4. Responsive 1280/680/390 real Chromium: editing from full screen invokes existing patient chart popup, form pre-fills date/category, optional explicit suggestion from anchored source, Save/Cancel, keyboard focus, no overflow, one-tap move back; no edits while unsaved clinical draft is present.
5. Reuse affected CP1/CP2/visit amendments, Home, Visit Brief and canonical checks; exactly one independent R2 pre-code and one post-code review (reuse unchanged evidence). Release/real-data authority is separate and requires PO approval.

**State:** Product Owner explicitly requests the behavior; candidate design/implementation on a DRAFT/HOLD branch. The initial release of #205 remains LIVE/unchanged until final review and separate release authorisation.
