# Native RF — Dedicated Tracking + Weekly Calendar v1: fresh independent PRE-CODE R2

**2026-10-14. MODE: ONE fresh independent READ-ONLY affected PRE-CODE R2, then STOP.**
No implementation, commits, schema, tests against production, merge, deploy, patient/PHI access, external Cal/Reception requests or provider operations.

Repository: `athpapachr-cmd/osteoporosis`.
**Frozen activation main:** `244a3264f119252b1bbfc3562a00648cfdf113bd`, after independently reviewed #152 Home Real Attention Render LIVE.
Design branch: `design/rf-tracking-weekly-calendar-20261014`.
Fresh-resolve DRAFT PR number and exact current HEAD. **Owning design** `clinic_utilities/rf/RF_TRACKING_WEEKLY_CALENDAR_V1_DESIGN_2026-10-14.md`, **expected design blob** `6eb38f365f39a80d80178264e0b0884282648e07`. Pin source identity; if blob moved, compare/reconcile rather than silently citing this fixed candidate.
**Parallel** DRAFT #153 Medical Reports design + independent R2 closure are outside this RF scope; do not merge the workstreams or modify their files.

Product Owner expressly approved **DESIGN ONLY**: a genuinely separate RF tracking page with all-application list and weekly calendar akin to Osteoporosis UI, a one-tap path to/from RF form, and optional real Reception appointment **time only for exactly source-verified RF↔booking association**. No new RF registry.

### Settled evidence to REUSE, not re-review

- Native RF v2 #149/#150 independent PRE/POST-CODE R2 PASS, protected issued attempts and outcomes/opt-in responsibility, immutable PDFs and rejection-correction history. Inspect current `clinic_utilities/rf/{api.py,persistence.py}` and `static/clinic-utilities/rf/{index.html,app.js,styles.css}`, and `test_rf_v2_native.py`/`test_rf_v2_ui.js` for affected seams.
- #151 Clinical Key / global nav / Calendar PASS; #152 Home real source-backed RF counts and surgery unaffected POST-CODE R2 PASS plus Render LIVE; original Home V4 and calendar/browser synthetic contracts.
- `static/clinical-calendar/calendar.js` is a visual precedent; inspect `clinical_calendar.py` real `GET /appointments` filtering behavior and private Reception source contract (`RECEPTION_SCHEDULE_CONTEXT_URL`). Optional cross-repo **read-only code inspection** of `athpapachr-cmd/ortho-reception-backend-v2/schedule_projection.py` is allowed, without any live network/provider call. Existing snapshots have booking UID/start/end/fetched_at/coverage but no RF application ID.
- No reopening Medical Reports #153, Reception booking M6 and provider write protocols, clinical Visit Capture authority, surgery queue or unrelated regressions.

## Seven finite required gates

**G1 — Identity/strictly bounded owner.** Exact PR DRAFT/unmerged, base/head and design blob, docs-only diff and canonical checkpoints; no runtime/backend/test/schema/deploy changed. One RF-owned tracking presentation over existing `RFApplicationORM` + `RFApplicationProgressORM` and one optional narrow booking-association relation, **not** a copied RF/patient/appointment database. Clear Product Owner approved direction but not implementation.

**G2 — RF workflow fidelity.** Default `scope=all` lists previously issued RF attempts, not only `mine`; `mine_to_follow` remains voluntary/provenance-based, including colleagues. Issued ≠ submitted, submitted current awaiting decision, approved decisions can overlap later scheduled/performed and are not added as disjoint totals. Previously issued without progress stays "υποβολή άγνωστη". List/search and pagination do not silently hide older attempts behind the 10000 scan ceiling; truncated/incomplete and unavailable are honest, "no records" only if complete. Selected application ID, optimistic revision, clinical event validation, audit, safe rejected-correction reuse and immutable PDF preserve current backend authority; no new outcome writer.

**G3 — Separate page + one-tap navigation.** `/clinical/clinic-utilities/rf/tracking` is protected, list/calendar and compact detail, links to existing `/clinical/clinic-utilities/rf` new A1/A2 creator and back, #152 Home RF tap changes destination only. A1/A2, saved drafts, doctor issuer and RF outcome action code are reused, not cloned. Selected rejected-correction handoff is deterministic/explicit, with no protected data in browser navigation URL/storage or stale selection, and unsaved creator edits not silently lost.

**G4 — Weekly RF source correctness.** RF `scheduled_date`/`performed_date` are date-only `YYYY-MM-DD`, not appointment clock times. Asia/Nicosia Monday–Sunday, DST-safe. RF scheduled past date is **not** performed; performed only if source-attested; undated remains in list. Multiple RF on a date remain multiple cases, not a single inferred appointment. Proposed week projection reads all relevant existing RF rows, not page 1; honest completeness/truncation, bounded response and no silent data omission. Osteoporosis calendar's filtered appointment data is **not reused as RF bookings**, its classification and writer unchanged.

**G5 — Reception exact booking linkage (critical).** No existing RF application ↔ Cal booking UID link—do not infer from name/phone/date/procedure/category. Explicit clinician-attested one-to-one association in RF-owned bounded metadata, with revision/audit, independently verified unique active Cal UID from trusted protected source, freshness ≤5 min, coverage for requested day/week, UTC zone-aware interval, Cyprus rendering; time only while **both proofs** remain true. Patient identity is not verified merely by an RF/Cal name match. The current clinical calendar `/appointments` intentionally excludes `other`; design must require narrowly shared/refactored protected read, not pretend it already works. Cancellations, disappearance, stale/incomplete source, reschedules, UID collision, different RF date fail closed for booking time only; RF date-only view survives. No provider booking create/reschedule/cancel/notification, no status `performed` inferred from a booking. If link verification not achievable, optional clock-time layer remains **OFF**.

**G6 — Protected failure/race + UI evidence.** Existing shared Clinical Key is not per-doctor IAM. No PHI in Home or public URLs/logs/storage/screenshots; tracking details only within protected RF UI. Separate generation/request identity for list, selected RF, week, booking, and 401/403/503/network/error/refetch, preventing stale A after newer B success/failure from showing wrong names/times; failed Reception cannot fabricate no RF or erase true RF dates. Real Chromium synthetic 390/680/1280, keyboard, two same-day records, pagination, source unavailable, re-auth and unambiguous UI. No second protected credential, browser localStorage or independent provider API.

**G7 — Proportional CI/governance.** Future changes limited to RF route/data, one RF tracking screen/shared UI handlers, Home RF destination, and existing calendar protected reader **only if needed and reviewed**; no edits to other workstreams or existing booking writer. Synthetic oracles explicitly exercise link/no-link/fresh/stale, confidence, status, manual correction, invalid dates, source authorization, race and body-only identifiers where applicable, plus RF native/Chromium, affected Home/Calendar/Visit Capture and Canonical Impact CI. Reuse already closed reviews; classify only real affected material risk. Design PASS, if any, authorizes **only a later separately PO-approved implementation**, then affected POST-CODE R2, and distinct merge/deploy/clinical-use approval.

## Required reviewer output and STOP

Return:
- `PRE-CODE R2 PASS | BLOCK | UNKNOWN`
- `COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`
- `P0:P1:P2 = x:y:z`
- exact current PR/base/HEAD, design blob and docs-only diff
- disposition **G1–G7**, any smallest concrete P0/P1/P2 findings and **minimal design-only correction** if blocked; no new general design cycle
- **STOP**.

The design author **must not self-certify** an independent PASS. Reviewer performs no source edits, merge, deploy, DB queries, actual patient data or external provider operations.
