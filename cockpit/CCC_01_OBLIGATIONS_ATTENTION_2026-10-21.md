# CCC-01 — Υποχρεώσεις ασθενών και σημερινή προσοχή

**2026-10-21 · Product Owner APPROVED start · bounded implementation candidate · NO release authorization.**

## Product-owner purpose and acceptance

Problem: Today the Tasks icon opens all first-100 active Pending records, so a future-visit lab follow-up looks like an immediate action; the icon badge reuses surgical totals. The patient has not asked for more popups or extra steps. Approved CCC plan: one milestone at a time, pilot each result before the next.

**After CCC-01:** a normal full Cockpit "Υποχρεώσεις ασθενών" screen; same protected active `pending_id` owner, grouped by *patient_id* or item_type or due bucket. Filters: patient/text query, category, status, due bucket, sort, paginated older records. One click opens existing patient chart's exact Pending popup. The Tasks icon is an attention glance for only explicitly dated due/overdue open tasks; it links to full list. Gmail and call-derived attention are *truthfully unconnected*; the separate Clinical Inbox remains independent. Normal unqualified free-text `trigger` and `external_dependency` are displayed in full detail but NEVER translated into a due event merely by phrase matching.

## Ownership / reuse and classification

- **REUSE**: protected `clinical_capture_cp1.py:/clinical-updates` and `ClinicalPendingORM`, existing `pending_view` (checkpoint/revision/provenance), one `patient-chart.js` identity-bound patient popup, Clinical Key/session authority, existing Cockpit shell and Home library.
- **EXTEND read ONLY**: API pagination `limit<=100`, `offset`, optional `query`, `category`, `status`, `due`, `sort`; `due=today` filters *recorded valid ISO dates <= Cyprus today* across ALL records **BEFORE** limit, `future` > today, `undated` when not set/invalid. `active_only=true` excludes resolved/cancelled/deleted before pagination, maintaining the existing default for legacy readers. Stable secondary `pending_id` sort. Structured `total`, `has_more` and `next_offset` only from the same protected filtered cohort; no clinical writes on reads.
- **No second data source or queue**, no inference that all undated/open items are urgent. Direct clinician-entered same-day `review_date` is the only machine-actionable trigger at CCC-01. A pending item "next visit" with no date is visible under "Χωρίς ημερομηνία" until a future milestone qualifies a visit-event trigger. An "await results" task is never closed by mail alone.
- R1 **display/projection**: unchanged protected clinical read scope and clinical semantics, no new clinician tasks, identity, write, integration or timing authority; review only after code. If a future design seeks persisted event triggers/automatic classification, that materially changes clinical semantics and is **R2**.
- NO `localStorage`, `sessionStorage`, URL patient identifiers for list, email/Reception provider, new table, or PHI in CI artifacts. Existing chart may maintain selected patient URL under its already-approved flow, not new list state.

## Explicit trust and visibility

1. Protected auth (legacy physician or qualified clinician cookie) unchanged. Never enable secretary or public list. Scope `real` continues excluding synthetic IDs; scope `test` is explicit only.
2. Full screen and popover are read-only projections. Stored label/text inserted via `textContent`, not innerHTML. Clicking an exact `patient_id/pending_id` opens the existing guarded patient detail/popup; no write by new UI.
3. Distinguish **no due work** from **read unavailable, authenticated state missing, stale fetch**, and **partially paginated** full list. If request fails or logout occurs, PHI clears and count badge is hidden. Late requests cannot repopulate.
4. Cyprus calendar-day cutoff computed by named timezone `Asia/Nicosia`, not UTC midnight or workstation timezone. Invalid/missing dates are undated, never automatically due.
5. Home attention list for surgeries/RF remains untouched. The Tasks icon badge must no longer display surgical counts; only qualified current dated clinical attention after protected response may set its count.
6. Footer of Tasks preview explicitly links "Όλες οι υποχρεώσεις" and notes that call/email attention comes only after CCC-02–06 integrations, without inventing current-source completeness.

## Integration / bounded rollout

Change only `clinical_capture_cp1.py`, `static/cockpit/index.html`, `doctor-shell.js`, `doctor-shell.css`, the installed patient-chart read list adaptation, new standalone same-origin read-only controller if needed, and precise synthetic tests. Keep Home split-view, Clinical Inbox, surgeries, RF, appointment navigation, Visit Capture Save, and Pending checkpoint/actions unchanged. CSS at 1280/680/390. Release: DRAFT PR; one independent affected R1 PASS + explicit PO release decision, then native Render and real-data READ-ONLY acceptance. Do not start CCC-02 until PO accepts CCC-01.

## Acceptance cases (one finite test set)

- Recorded future/undated trigger, pending next visit or results awaiting => full list yes; Tasks now no. Review_date today or overdue => both.
- Active-only filtering before first 100; real >100 items accessible by next page; stable unique pending IDs across pages and categories; status, patient, date/category sorting and category filtering deterministic; two same-name patients separate by `patient_id`.
- Auth 401/403, logout/re-login, stale old-success after denied request, patient switch and draft preservation fail closed.
- No false result arrival, email/call data, badge or task-completion assertion.
- Responsive native list with search/filter and direct existing popup click at 1280/680/390 without independent browser tab or new authentication step.

**Out of CCC-01:** Gmail intake and matching (CCC-02/03); decisions (CCC-04); outbound (CCC-05); reply calls (CCC-06); cross-system unified clinical chronology (CCC-07); full pilot (CCC-08).