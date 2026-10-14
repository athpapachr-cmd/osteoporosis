# Native RF v2 — Dedicated Tracking + Weekly Calendar v1 (bounded design)

**Date:** 2026-10-14. **Stage:** Product Owner APPROVED DIRECTION → **DESIGN-ONLY candidate / fresh independent PRE-CODE R2 HOLD**. No implementation or production authorization.
**Repository:** `athpapachr-cmd/osteoporosis`. Frozen design activation production `main = 244a3264f119252b1bbfc3562a00648cfdf113bd` (#152 independently reviewed and Render LIVE). Branch `design/rf-tracking-weekly-calendar-20261014`.
**Parallel work:** DRAFT PR #153 Medical Reports resumable tracking is a separate, unmerged design, **not included**. No overlapping runtime or report design edits. Reconcile shared CURRENT documentation at future merge time; no premature cross-workstream release.

## 1. Confirmed Product Owner experience

The embedded `Αιτήσεις & εκβάσεις` panel in the long RF form is hard to find. A clinician wants a **separate "Παρακολούθηση RF" page** with **Λίστα** and **Εβδομαδιαίο ημερολόγιο**, and visible direct navigation back to `Νέα αίτηση RF` / the existing A1/A2 form. Cockpit Home's RF attention card must open **Tracking**, not the buried form. A date in the RF ledger may be shown; an actual **Reception appointment with clock time** may be shown **only if there is an explicitly confirmed link to that exact RF application and a currently source-verified booking**. Do not infer it from patient name, surname, phone, type, approximate date, location or a free-text reason.

The desired UI is *iPhone-like*: compact, first-tap detail, no duplicated RF modules, no lengthy forms in the tracking list.

### Proposed protected screens

- `/clinical/clinic-utilities/rf/tracking` — main tracking screen; top **Παρακολούθηση RF | Νέα αίτηση RF**, one list/calendar switch, search, status filters, compact true counts and immediate case actions.
- `/clinical/clinic-utilities/rf` — existing form for **new A1/A2 request**. Always-visible **Παρακολούθηση RF** link. Both screens reuse **one** native RF controller/outcome implementation, not a second database or independent application. Protected route on the **existing RF router** with `Cache-Control: no-store`; legacy entry remains valid.
- Cockpit Home `static/cockpit/doctor-shell.js`: **keep existing #152 RF counter reader/semantics**, change only the RF action destination to the new `/tracking` screen, optionally a second compact "Νέα RF" shortcut in existing commands; no new attention counter source.
- On the tracking screen, choose **all issued applications by default**, not the old `mine` follow-up filter. The latter remains a named secondary filter; issuing a PDF does **not** establish personal follow-up or submitted status.
- A clicked protected RF row opens an in-page detail panel (same screen) with stage, dates, issuer provenance, decision and rejection reason if known, and contextual actions **Υποβλήθηκε**, **Εγκρίθηκε**, **Απορρίφθηκε**, **Προγραμματίστηκε**, **Πραγματοποιήθηκε**, **Διόρθωση**, **Αναλαμβάνω/Σταματώ παρακολούθηση**. Only currently valid actions are shown. Manual write path remains the existing guarded RF outcome event API. Never imply a PDF itself was submitted.
- For rejected attempts, retain the existing **Νέα διορθωμένη αίτηση** protected reuse flow. A transition to the existing form with active case context may use **same-tab transient in-memory controller state only**; direct-link/new-tab/new-login open is generic unless the original application is explicitly re-selected. **No patient/application ID, rejection reason or one-time secret in URL, hash, query, browser storage, logs or open redirect**. Cross-screen actions must not silently apply to a different RF.

A clear **Πηγή/κάλυψη δεδομένων** status shows source-backed zeros versus unavailable/partial. Examples in screenshots/tests must be invented; the user's observed "13 εκδοθείσες" is not a fixed or expected runtime fixture.

## 2. Independently sourced existing authority (no second register)

Current source on frozen main:

1. `clinic_utilities/rf/persistence.py` blob `0385e2cb56ca20b197241ca7a0f70e01acf71ed7`: immutable issued attempts `RFApplicationORM`, existing `RFApplicationProgressORM` with `stage, submitted_date, decision_date, scheduled_date, performed_date, mine_to_follow, revision` and audit, and `list_rf_application_progress()`. No appointment UID link exists. `_RF_SEARCH_SCAN_LIMIT = 10000` and 20-entry pagination mean partial/truncated results **must** be labelled partial. Do not imply that page 1 contains every application or read all "13" from a page-length assumption.
2. `clinic_utilities/rf/api.py` blob `391878ff4ff2ab55c6d03872f4a420b42c768b1c`: existing protected `POST /api/outcomes/search` (**read-intent**), `GET /api/outcomes/{application_id}` and `POST /api/outcomes/{application_id}/events` (**write**); direct RF issuance and previous-application reuse remain unchanged. Scope owner is the existing RF router and RF database only.
3. `static/clinic-utilities/rf/{index.html,app.js}` blobs `5ab322064678f7a0ca5f5959b31ec176119e52d6` / `389421107c6cf4b0f3adf3c300968fd42394f5ae`: the current panel `Αιτήσεις & εκβάσεις` exists at the top of the builder but defaults `scope="mine"`. Existing listing, selection tokens, outcome actions, optimistic revision and correction flows must be **reused/refactored**, not forked into another writable UI. `test_rf_v2_native.py` and `test_rf_v2_ui.js` already cover settled RF rules.
4. `static/clinical-calendar/{index.html,calendar.js,calendar.css}` and `clinical_calendar.py`: the osteoporosis weekly calendar is a **visual precedent only**. The protected `GET /clinical/calendar/appointments` **filters out** bookings whose categories are not `osteoporosis_first/review/unspecified`, `prolia` or `aclasta`. It **cannot** currently serve as an RF appointment feed, and must never be widened unconditionally for the existing osteoporosis UI.
5. Reception's existing private `/private/schedule-context` has UTC `fetched_at`, source coverage dates, real Cal.com booking `uid, start_at, end_at, clinic, display name, reason, category`. It is already consumed by `clinical_calendar.py` under server-to-server configured authentication. The cross-service model **does not** carry a trusted RF application ID, RF stage, proof of patient identity, or current RF↔Cal booking association. The source typically covers a bounded ~31-day local lookahead and its freshness is checked to ≤5 minutes; never claim unlimited historical/week coverage.
6. `static/cockpit/doctor-shell.js` blob `acd9340c4d2d67d30ede12426ff8784eaeb3a8d0`: source-backed RF counts distinguish **issued attempts** from current `submitted` awaiting-decision, `mine_actionable` from verified RF responsibility, and incomplete sources. This #152 reviewer-approved projection stays **unchanged except navigation**.

No source presently certifies an RF↔booking link. The first release's **RF date-only view must work without Reception**. Adding optional actual clock times is conditional/gated, not a fabricated automatic join.

## 3. Tracking list contract

One authoritative RF application record = one issued attempt. An explicitly linked corrected RF is another attempt, not an alteration of the old rejection. Search by Greek patient name (accent/case insensitive), existing identity/GESY where authorized, issue date and optional stage; issuer filter only if source-backed. Show current lifecycle labels:

- `Εκδόθηκε — υποβολή άγνωστη` = `stage="issued"`; **not** "pending approval".
- `Υποβλήθηκε — αναμένει απόφαση` = `stage="submitted"` with explicit date, current pending.
- `Εγκρίθηκε` / `Απορρίφθηκε` = source-attested manual decision, including colleague-reported decision with **unknown submission date**.
- `Προγραμματίστηκε` / `Πραγματοποιήθηκε` only from the existing manually confirmed RF stage. Approval is not actual treatment.
- `Δικές μου προς παρακολούθηση` only if existing `mine_to_follow`, never because issuer was a colleague/default profile. Claims/releases do not change real outcome.

Card: name, RF anatomical site/side and A1/A2, issued date, latest recorded stage, issuer when documented and follow-up responsibility (source-backed). Select by **application ID only** within protected in-memory client state; stable optimistic version on all event writes and stale-response isolation. No "quick action" to misapply events to another selected RF. Search/date/stage filters must be on the server for complete result semantics—not client filtering of the first 20 or only the first 10000. If the existing search API remains bounded/truncated, expose explicit **μερική κάλυψη**, withhold falsely exhaustive totals and avoid "no results" absent a confirmed complete filtered scan. Prefer a small additive extension of the RF-owned existing search rather than a new RF list owner.

The **existing RF action endpoint** remains the only authority for submission/decision/scheduling/performance. UI provides provenance/event date with explicit physician confirmation; auto-issued PDFs, calendar dates and actual Reception bookings **never** mutate `submitted/approved/scheduled/performed`. No GeSY provider submission or Cal.com booking write.

## 4. Weekly RF calendar: source-backed dates, not fictitious bookings

Reuse the osteoporosis calendar's compact week navigation/layout, date labels and responsive treatment, **not** its osteoporosis appointment categories or ingestion authority. Default current Cyprus week, Monday–Sunday; previous/next/current-week controls. Use `Asia/Nicosia` comparisons; existing RF `YYYY-MM-DD` are **local date-only** values, not UTC instants. Handle DST transitions and invalid dates without day shifts.

Primary source = authoritative existing RF progress. Show:
- `Προγραμματισμένη RF · [patient name/site] · [local date]` on `scheduled_date`, with **Χωρίς επιβεβαιωμένη ώρα** if no separately verified real booking.
- `Πραγματοποιήθηκε RF · [local date]` for source-attested `performed_date`; distinguish historical confirmed treatment from future planned entries; do not infer performance from a date that has passed.
- Optionally show recorded submission/decision as muted history when toggled, NOT as a scheduled treatment or an appointment.
- No scheduled date? Keep the issued/approved/awaiting applications in the **list** and a compact **Χωρίς ημερομηνία προγραμματισμού** area; do not insert into arbitrary day cells.
- One app may have a linked actual booking on a different date. Render the RF date-only fact **and** the independently source-confirmed booking date/time with different labels and a discrepancy warning, never silently overwrite either, double-count it as two applications or invent a reschedule/actual treatment.

**Bounded read plan:** add (only when separately authorized for code) one RF-owned protected *read-only* `POST /api/outcomes/calendar` accepting a strict local `week_start` (Monday ISO date), querying indexed ``scheduled_date/performed_date` ranges across **all** matching RF progress rows, plus bounded relevant detail by their existing RF application IDs, with explicit `complete/truncated/source_as_of`. Hard resource ceiling/error → unavailable/partial, **never** silently omit entries from a purported complete week. This endpoint is a **projection of the existing RF database**, not a second registry/provider or calendar writer. List retains the existing RF search endpoint. No reusable Google Calendar/Cal.com synchronization job, extra clinical appointments database or scheduling API in this slice.

## 5. Conditional Reception/Cal.com booking time — strictly verified link

### Current reality and optional RF-owned association

**No reliable link exists today.** Do **not** automatically match RF to a booking by name, phone, date, procedure keywords, patient ID guesses, issuer, or Cal category `other`. A list of anonymous matching suggestions cannot itself make a source link verified.

To enable future time display, define one **optional RF-owned, versioned one-to-one link** (`existing RF application ID ↔ exact source Cal.com booking UID`), under the **existing RF clinical database and protection**. If persistence is needed, at most one narrow RF-specific association sidecar (ID pair, association revision, recorded time, explicit clinician attestation, audit provenance); **NOT** a copied appointment register, new patient mapping table, secondary RF report owner, stored source snapshots or provider booking writer. A prior application ID and its immutable issued PDF remain unmodified. Require clear conflict behavior if UID is already attached to a different attempt; explicit unlink/relink with audit, no silent takeover. Existing RF outcome statuses are unchanged by linking.

**Two separate proofs are required before the UI says "Επιβεβαιωμένο ραντεβού Γραμματείας":**
1. A protected clinician explicitly selected **this RF attempt** and **that exact Cal UID** from source-backed booking choices, confirmed the association and the limited patient-context ambiguity. This is **manual association provenance**, not proof of individually authenticated clinician or national-ID identity. The shared Clinical Key remains clinic-wide.
2. At **display time**, the trusted server-only existing Reception schedule reader attests the same unique Cal UID in a **fresh (≤5-minute), complete-covered, valid source** for the requested week/date; event is active/not canceled, start/end have valid zone-aware chronology, and normalized `Asia/Nicosia` local time is available. The API may expose only the minimum linked booking facts (UID within protected server/client view, date/time, clinic, booking-source timestamp and verification label), without phone or unrelated bookings in Home.

The **existing** `GET /clinical/calendar/appointments` is **not** sufficient for RF—it currently drops `other` bookings. Any necessary bounded sharing/refactoring of `clinical_calendar.py` must **reuse its existing protected Reception source/freshness/coverage reader**, not create a second direct browser/provider reader, a new credential, a Cal booking writer or broaden the osteoporosis screen. Future implementation may expose a narrow protected RF booking-candidate/link-verification read and an RF-owned audited link/unlink action, and must separately prove permissions and source coverage. If this integration cannot meet its source/auth/coverage gate, it **stays disabled**; deliver the full RF list/week with honest **Δεν υπάρχει επιβεβαιωμένη ώρα**.

Only linked/verified appointments may expose exact times. If appointment source becomes stale/503, changes UID, disappears from an otherwise complete snapshot, is canceled, duplicate/ambiguous, outside covered window or cannot be validated, **clear previously displayed clock time immediately**, show **Σύνδεση/ώρα μη επιβεβαιωμένη — έλεγξε Γραμματεία**. Do **not** infer cancellation from an uncovered/unavailable source. If same UID reschedules, use the newly source-verified date/time, show **Διαφορά από καταχωρισμένη ημερομηνία RF** and require manual RF scheduling correction; do not auto-change status/date. No Cal.com create/reschedule/cancel/attendee/email action.

### Read race/authorization

A shared monotonic request-generation (and active selected RF ID) governs list, week, selected detail and optional booking verification. Old success A after newer 401/403/503/changed week/changed RF B may **never** restore a stale patient, UID, clock time or past authorization; old failure A cannot erase B. On Clinical Key re-auth/logout or source error, immediately hide previously protected list/detail and any linked appointment times until fresh validated success. Preserve a **truthful distinction** between unavailable RF ledger and unavailable Reception timetable: a failing Reception clock feed must **not** wipe otherwise valid RF progress dates or fabricate zero RF records. Reject malformed response, missing strict `complete/truncated` indicators or untrusted times; no logs containing PHI.

## 6. Product navigation and authority preservation

* Home RF card → `/clinical/clinic-utilities/rf/tracking`. Existing source-backed counts from `POST /api/outcomes/search` remain identical. Home **never** displays RF name, application ID, Cal UID or individual appointment time.
* Tracking ↔ existing RF creator with one action; existing A1/A2/PDF, doctor directory, source/history, saved RF drafts and valid rejected correction reuse preserved. If switching with unsaved creator form data, warn rather than discard silently. Rejected correction handoff is explicitly initiated and uses protected server-side reuse validation; no clinical identifiers in route parameters, query strings, history or browser storage.
* Shared sticky Home/Shortcuts/Clinical Key remains visible in both screens, no duplicated secret login or new provider identity. Current RF external-result audit, source verification and per-clinician follow-up semantics remain unchanged.
* Do not modify `clinic_utilities/clinical_documents/*`, Medical Report PR #153, Visit Capture Save/patient selection, Reception service, Surgery, Osteoporosis clinical-calendar classification, Cal provider, emails or notification scheduling.

## 7. Minimal possible future implementation boundaries (NOT NOW)

If independent design R2 PASS and explicit separate Product Owner implementation approval:
- **RF route/persistence:** `clinic_utilities/rf/{api.py,persistence.py}` only RF read projection, optional audited link metadata/actions behind a verified-source gate; reuse existing models/transactions, no duplicate RF applications table. Any additive schema strictly migrated/created idempotently; `create_all` must not be treated as a migration for existing columns.
- **Protected screens:** one `tracking` route/page in existing RF router, narrowly refactor `static/clinic-utilities/rf/{index.html,app.js,styles.css}` and share one existing outcome component/handlers. New small RF-owned tracking UI/CSS/JS files only as needed. No second independent outcome event implementation.
- **Conditional Reception read:** only a narrowly scoped **read-only** reuse of `clinical_calendar.py` / existing configured private source contract, with verified fresh source/coverage. No change to Reception code, upstream schema/provider/write; if impossible, hold booking-time capability and show date-only mode. No additional provider credentials.
- **Cockpit Home:** one RF link destination change in `static/cockpit/doctor-shell.js` plus existing shortcut placement if needed; no new Home source or counters.
- **Focused synthetic tests:** `test_rf_v2_native.py`, `test_rf_v2_ui.js`, affected Home/browser tests and optional protected calendar transport mocks. Native RF + Home/Chromium + Calendar (if affected) + Visit Capture + Canonical Impact CI. No real names/phone/appointments or external service interactions.

### Finite independent synthetic oracles to be implemented later

1. New dedicated page defaults **all** (13 synthetic issued attempts illustrative); issued PDF ≠ submitted, colleague approval without submitted date, verified mine and opt-in behavior, correct status/provenance/correction actions, duplicate attempts not merged, search all pages/truncated truth and no stale selected action.
2. Week Monday–Sunday in `Asia/Nicosia`, leap/DST boundaries, past scheduled≠performed, undated RF, 2+ cases same date, two different statuses/dates same RF, Monday navigation and 390/680/1280px real Chromium. No horizontal overflow or keyboard/focus regression.
3. Booked time only after explicit clinician association **and** freshly verified unique UID/status/coverage from mocked Reception. No name/date/free-text implicit match. Stale, denied, uncovered, ambiguous/missing UID, source canceled, UID rescheduled, RF date mismatch, link collision/unlink and audit; receipt never auto-changes RF stage or registers performed treatment.
4. Concurrency A→B, B→A for list, calendar, selected outcome and time linkage, logout/re-auth, 401/403/503/non-array/invalid counts, request input conflict; no stale named RF/time exposure and no false zeros. Invalid Reception affects clock only, RF date-only calendar remains.
5. Routing: Home RF tap → Tracking, tracking → existing A1/A2 form, selected rejected attempt → existing protected correction with no PHI in URL/storage; no new provider writes, cross-service privileges, patient linking or calendar appointment records. Existing #150 Native RF post-code R2, #151 nav/calendar, #152 Home and unrelated #153 report source unchanged.

## 8. Governance, R2 and STOP

This change rearranges visibility of **named protected RF cases** and may add an *explicit, separately verified* Cal UID association. It is material enough for **fresh independent PRE-CODE R2**, focused on RF clinical authority, booking link integrity, freshness, privacy and bounded blast radius. Do not claim the Reception source presently proves RF booking linkage: it does not.

The Product Owner approval in conversation is **DESIGN ONLY**. This branch changes **documentation only**. No runtime code, schema, tests, clinical data, login, Cal.com/Reception operations, merge or Render deploy. After an independent PRE-CODE R2 PASS, implementation still requires a **separate explicit Product Owner instruction**, synthetic affected CI and separate **independent POST-CODE R2**. Merge/deploy and any use of real PHI or verified real bookings require their own later authorization. Parallel #153 and its privacy gates remain untouched.

**AUTHOR WRITER RELEASED after design/request/checkpoints; reviewer must independently decide PASS/BLOCK/UNKNOWN and STOP.**
