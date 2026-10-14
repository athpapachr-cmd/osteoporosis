# Native RF — Tracking + Weekly Calendar v1: independent delta + affected PRE-CODE R2 closure

**2026-10-14. MODE: ONE fresh independent READ-ONLY delta + affected-cumulative PRE-CODE R2; then STOP.**
No implementation, test execution against real services, source commits, schema/database access, merge, deploy, patient/PHI access, external Reception/Cal.com requests, or provider operations.

- Repository: `athpapachr-cmd/osteoporosis`; DRAFT PR **#154**, branch `design/rf-tracking-weekly-calendar-20261014`. **Fresh-resolve exact PR HEAD/base/main**, do not rely on a stale author's SHA.
- Frozen activation production `main = 244a3264f119252b1bbfc3562a00648cfdf113bd` (#152 previously independently reviewed and Render LIVE).
- Original blocked design HEAD `13382205b170e2e623cf05bb63f28f06e2b7509f` and original owning design blob `6eb38f365f39a80d80178264e0b0884282648e07`. The original request blob is `7e66786d6e19692be159d51d048cbe51766ab9e6`.
- **Corrected owning design (expected exact blob): `debd6fd9d61b15bf68160ea586b7dd9b36a72efe`** at `clinic_utilities/rf/RF_TRACKING_WEEKLY_CALENDAR_V1_DESIGN_2026-10-14.md`. Check fresh HEAD blob and docs-only diff; if it moved, reconcile explicitly.
- Prior independent result **PRE-CODE R2 BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:4:1**. This candidate is an author correction, **not** an independent closure PASS.
- Parallel DRAFT Medical Reports **#153** stays entirely separate. Reuse settled Native RF v2 #149/#150, #151 shared Clinical Key/calendar, #152 Home attention, Home V4 and synthetic source contracts; do not re-review or modify their unrelated scopes.

## Mandatory bounded closure checks

**P1-01 — Complete Tracking search beyond both legacy ceilings (G2).**
Confirm current `RFOutcomeSearch.page ≤100` plus 20/page imposes ~2,000 visible-result navigation ceiling, independent of `_RF_SEARCH_SCAN_LIMIT=10000`. Corrected design must add an opt-in **cursor-capable mode on the existing RF-owned protected search endpoint**, default `all` on Tracking, using stable server-side ordering, opaque POST-body-only cursor bound to filters/expiry/high-water, bounded forward scan beyond 2,000 matched and 10,000 scanned RF attempts and no skipped older/Greek/identity/status matches. A zero-match partial batch must still offer continuation, not false empty; mark total/counts complete only with source proof. Changes to mutable RF progress between pages must not falsely claim one atomic snapshot. The **legacy `query/scope/page` Home #152 contract and counter semantics stay unchanged**; no second registry or search owner. Proposed synthetic oracle: >2,000 matches, >10,000 scans, cursor tamper/filter-mismatch/expiry, pagination and exact/partial totals.

**P1-02 — Deterministic rejected-correction handoff (G3).**
Check genuinely distinct protected Tracking / existing creator **views of one same-document RF shell/controller**, enabling same-tab **in-document** navigation with fixed generic paths, not a full-page reload carrying phantom JS state. Selected rejection ID, revision and one-time correction intent are volatile; current protected reuse API revalidates the exact selected attempt; wrong ID/revision/clinical state, lost selection or login change fail closed. Direct GET/new tab/reload/re-auth restores generic unselected view until explicit reselection. No patient/application ID in **document** URL/path/query/hash/history, browser/session/local storage or logs; existing protected API paths and audit/revision rules preserved. Dirty A1/A2 drafts remain in memory/warn before discard; back/forward and shared Key must not discard silently. No duplicated writable outcome handlers or altered PDF semantics.

**P1-03 — Mismatched booked date + true five-minute clock expiry (G5/G4).**
No existing RF application ↔ Cal booking UID source link. Require explicit audited 1:1 clinician RF-ID/UID association **AND** fresh (0..300 seconds), fully covered, source-authenticated, unique active/non-cancelled matching UID with zone-aware interval. The existing osteoporosis `/clinical/calendar/appointments` filters `other` bookings and is NOT a valid RF feed; the protected Reception snapshot has no RF ID or explicit booking-status property, so review the actual active-only projection and fail closed when unproven. **Clock time additionally requires existing manually recorded RF `scheduled_date` exactly equal to the booking's Asia/Nicosia local date.** Cross-date reschedule, missing RF date or mismatch => source-backed RF date-only facts plus discrepancy notice, **no clock**, until audited manual schedule correction and re-verification. Client clears an already visible time on or before `fetched_at+300s` **even when idle**, on focus/refetch/error/link change, while keeping valid RF dates. No provider writer, auto-clinical status, inferred cancellation from unavailable source or PHI in Home.

**P1-04 — Four stream generations + shared auth epoch (G6).**
Separate tokens/identities for list/filter/cursor, selected RF/detail/event, RF week, Reception candidate/link/clock. Each async result AND failure must match own stream token, selected ID/revision/week/context, and a shared auth epoch. 401/403/re-auth/logout/session expiry globally clears protected name/detail/link/time and any transient correction state; unrelated refreshed week does NOT cancel list. RF 503/network invalidates the affected RF view; Reception-only 503/stale/partial only clears booking time/candidates and NEVER fabricates no RF/loses valid date-only view. Old A success/error cannot supersede newer B success/error. Strict schema, zero/partial distinction and no persistence of clinical identifiers.

**P2-01 — Actual week-date indexing / migration assumption (G4/G7).**
Verify `RFApplicationProgressORM.scheduled_date` and `performed_date` are nullable strings without declared indexes in the frozen schema. Corrected design must not claim indexed baseline; explicitly choose safe/idempotent index migration only if needed **or** proven bounded complete query plan. `create_all` is not a migration. Multiple same-day and scheduled/performed RF date-only entries, DST/leap, hard ceilings and completion truth need focused synthetic coverage. No Reception writer/calendar classification changes.

## Affected cumulative G1–G7 and bounded authority

- Confirm one protected RF list/calendar and **one** existing issued-attempt/progress/event owner, no copied patient/RF registry, and no conflict with RF v2 issuer/opt-in, immutable PDF, corrected rejected attempts, source audit.
- G1 identity/docs-only governance; G2 status/search completeness; G3 route/reuse; G4 local week facts; G5 optional exact UID proof/expiry OFF-by-default; G6 fail-closed UI/auth/race; G7 proportional future synthetic RF native/browser (Chromium 390/680/1280), affected Home/Calendar/Visit Capture and Canonical Impact CI only after separate implementation authorization.
- Reconcile original design/request with corrected blob, branch `CURRENT_OPERATIONAL.md` and `clinic_utilities/rf/CURRENT.md` and PR metadata. Confirm PR remains OPEN/DRAFT/UNMERGED and **only .md files changed** from frozen main. Do not interpret metadata or a source diff as tests/deploy success.
- Any PASS is **DESIGN-ONLY**, enabling at most a later *separately PO-authorized* bounded implementation. Later affected POST-CODE R2 and distinct release/clinical-data permission required; no implementation, merge, deploy or Reception/provider activity in this review.

## Required reviewer result and STOP

Return **PRE-CODE R2 PASS | BLOCK | UNKNOWN**, **COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL**, **P0:P1:P2 = x:y:z**, exact PR/head/base/design blob, explicit dispositions **P1-01...P1-04, P2-01 and G1–G7**, source evidence and any smallest remaining design-only correction. **STOP**.

The correction author must **not self-certify** independent closure.
