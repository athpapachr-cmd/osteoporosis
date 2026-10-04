# D1.1 freshness replan delta — operational appointment context

> **STATUS:** BOUNDED REPLAN CANDIDATE / R2 SOURCE DESIGN REVIEW REQUIRED / NO RUNTIME AUTHORITY.
> **DATE:** 2026-10-04 Asia/Nicosia.
> **OSTEOPOROSIS SOURCE:** remote `main` `12be588866aee6543444eee0b435e9064c39bc3f`; D1.1 implementation branch `feat/cockpit-d1-1-global-context-2026-10-03` at `6f56093ffbf33d01078eb8a25af35eb65a1e851a` before this documentation checkpoint. Runtime/test commit `dd558e2fa8d59409d21913d2d3f2fbf885fdfece` is unchanged.
> **RECEPTION SOURCE:** remote `athpapachr-cmd/ortho-reception-backend-v2/main` `bcfa57e0c1ca7358fb898b393e1dafa7b042c238`, inspected read-only. Product Owner supplied the production schedule, urgent-event and call-agent evidence; no live patient data or production requests were inspected.
> **FROZEN PRIOR DESIGN:** `D1_1_GLOBAL_APPOINTMENT_CONTEXT_DESIGN.md` blob `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33`. Its accepted pre-code PRE-01/PRE-02 closure remains historical PASS. This delta supersedes only its operational source/freshness and directly dependent storage/transport/test seams. The prepared post-code request is HOLD, not a verdict.

## Product Owner step card

**Problem now:** the daily Clinical Calendar copy may be hours old, so a newly booked, cancelled or rescheduled visit can make Previous / Current / Next wrong even when Reception sees the actual Cal booking. **After this step:** the Cockpit strip and Reception use one short-lived, read-only projection of actual Cal bookings; a clinician can refresh the schedule without running Cal ↔ Setmore reconciliation. For example, a booked `limassol-urgent` visit with `eventTypeId=393065` appears in the actual schedule and can become Next; an open urgent slot does not. **Outside this step:** booking/availability policy, call handling, Visit Brief, communication, clinical patient matching and lab callbacks. **Why now:** Visit Brief must open from a dependable appointment context. **Next milestone:** bounded source-design review, then a separately authorized implementation/fidelity gate.

## Evidence and replan trigger

- The frozen D1.1 design §8 explicitly accepts the daily Clinical Calendar snapshot cadence and says it cannot claim live Reception parity. That premise does not meet the global operational strip's current use.
- Current Reception `main.py` has protected `GET /dashboard/schedule`, an actual Cal `get_bookings` read and `_SCHEDULE_CACHE_TTL_SECONDS=3600`. The cached return recalculates clock labels, not bookings. The visible `Συγχρονισμός` calls `/sync/both`; that endpoint runs Cal→Setmore and Setmore→Cal and does not clear `_SCHEDULE_CACHE`. Its daily heavy cron remains the background reconciliation path. The endpoint's present `take=50`, error-to-empty behavior, day bounds, omitted booking UID/event type and cache make its **current response** insufficient as the D1.1 source without a bounded correction.
- Production evidence supplied by the Product Owner says heavy synchronization during call hours degraded the call agent; actual urgent bookings include `limassol-urgent` `eventTypeId=393065`. Urgent availability, telephone windows and lab windows are availability lanes, not booked attendance.

## Smallest final source design

```text
Cal actual bookings (booking UID/status/start/end/event type)
  → Reception-owned bounded, lightweight actual-schedule reader/cache
  → Reception schedule UI + private minimal schedule-context response
  → server-to-server Cockpit context reader
  → existing protected Cockpit Previous / Current / Next presentation

Cal ↔ Setmore + daily Clinical Calendar snapshot
  → off-hours/background reconciliation and Module-01 weekly Calendar only
```

1. **One operational source owner.** Reception reuses its existing Cal client/auth and schedule read, with a shared bounded query/projection for its UI and Cockpit. Query a Nicosia-local window covering today's Previous/Current and the bounded future Next horizon; page to completeness, include all actual eligible bookings regardless of ordinary/urgent event type, and exclude cancelled/inactive bookings. Booking UID is the stable deduplication key. An availability slot without a booking UID/status is never an appointment. A failed/incomplete Cal read is an error, never a successful empty schedule. Unknown event types remain ordinary `other` context; Aclasta gets the existing exception only from an explicit, reliable type mapping.
2. **Short-lived shared freshness.** Cache the complete actual-bookings projection for at most five minutes, single-flight concurrent refreshes and bound Cal latency/load. Return the actual successful `fetched_at` and cache age; a cache hit must not reset freshness. Invalidate affected dates after Reception-owned create/cancel/reschedule success. A protected **Refresh schedule** action bypasses the cache, fetches actual bookings and redraws both schedule consumers' next request; it does not run `/sync/both`. On failure, show unavailable/stale age and retain no claim of live parity. No heavy sync or Setmore mutation occurs in the request path.
3. **Minimal cross-service read.** Reception exposes a private, bounded schedule-context response with booking UID, start/end, clinic, patient display name, source-proven human reason, event type/category needed for the existing overlap rule, and `fetched_at`; no phone, linked clinical patient ID, attendee payload, call context or raw Cal metadata. Reuse the already deployed shared Clinical Calendar ingest secret strictly server-to-server for this read, with constant-time check and no browser exposure; the source review must confirm this credential boundary. Cockpit's existing protected same-origin `/clinical/calendar/cockpit-context` becomes a proxy/selector over that response, not a browser cross-origin call or a second Cal reader. It keeps the current concise response and fail-closed unavailable state.
4. **Separate weekly clinical projection.** The daily snapshot and normalized Clinical Calendar continue for the weekly Osteoporosis view and background reconciliation, without freshness authority for the global strip. Because global `other` rows are no longer needed in that store, the not-yet-released snapshot-only `retain_other` admission and its directly affected storage oracles are superseded before release. Preserve the accepted legacy-import discard/removal, manual overrides, weekly category filter and effective-`other` phone/link minimization. Do not delete any production rows by inference; D1.1 has not been deployed, and rollback/data treatment must be checked at release.
5. **Existing semantics survive.** Reuse/extract the implemented Nicosia day, start-order Previous, single active Current, strict earlier-Aclasta exception, overlap fail-closed and cross-day Next resolver. Keep the compact Home card, next-day date, weekly link, no counters/fallback, protected browser route, and no Visit Brief handler. No phone/name-based clinical identity linkage.
6. **Reception control correction.** Relabel the visible action `Ανανέωση προγράμματος` and bind it only to cache invalidation + lightweight Cal read + redraw. Keep `/sync/both` as protected administrative maintenance, not the visible routine action. Preserve the existing off-hours heavy cron and its call-agent safety boundary.

## Affected implementation delta

| Owner | Bounded change |
|---|---|
| Reception `main.py` / existing Cal client | Extract/harden actual-bookings schedule reader; complete bounded local-day/future query, status/UID/event-type mapping, shared short cache, invalidation and scoped refresh/private response. Rebind schedule UI button; keep heavy sync and cron behavior. |
| Cockpit `clinical_calendar.py` | Reuse implemented resolver and minimal response but source its rows/fetched timestamp from Reception private projection; return unavailable on source failure/staleness. Keep protected weekly Calendar path; undo only snapshot-global `other` retention now made unnecessary. |
| Cockpit Home JS/HTML | Keep current D1.1 rendering/weekly link; update freshness wording from daily feed to actual-bookings read and distinguish stale/unavailable. No Reception secret or cross-origin browser fetch. |
| Focused tests | Replace daily-global storage expectations with source/freshness tests; preserve all semantics/privacy/week-view regressions and the prepared post-code fidelity questions that still apply. |
| Docs/checkpoints | This delta, `cockpit/CURRENT.md`, root NOW and HOLD marker in the existing request. Frozen design blob and pre-code result stay untouched as provenance. |

## Acceptance criteria for the revised D1.1 gate

1. With synthetic bookings, normal, Prolia/Aclasta and actual urgent booking `eventTypeId=393065` are eligible in one UID-deduplicated schedule; urgent/lab/telephone **availability** alone yields no appointment. Cancel/reschedule removes or updates the row after a successful fresh read; incomplete/paginated/error responses cannot masquerade as an empty complete schedule.
2. Reception and Cockpit read the same source projection. A healthy provider read is no older than five minutes under normal use; protected manual refresh forces a new read and redraw, including immediately after a booking change. A cached read reports its original fetch time. `/sync/both` cannot be invoked by the visible refresh action, and the heavy cron remains off call hours.
3. Previous/Current/Next pass existing start-order, Nicosia DST/midnight, cross-day Next, explicit Aclasta and ambiguous-overlap tests. Unknown/urgent category does not silently become Aclasta. The weekly Osteoporosis Calendar stays filtered and linked.
4. Browser and cross-service response never expose phone, linked clinical patient ID, raw Cal/attendee/call data or secrets; no name/phone clinical record linkage; auth failures fail closed. Server-side effective-`other` minimization and legacy import behavior remain intact.
5. Stale, failed or incomplete schedule reads show a plain-language unavailable/stale state with last successful fetch time when known; Cockpit never falls back to the daily Clinical Calendar as if current. A synthetic load/timeout check shows the lightweight refresh is bounded and does not invoke Cal↔Setmore or call-agent paths.
6. One affected-surface R2 pre-code delta review passes before runtime mutation. After implementation, one exact-head R2 fidelity review covers this revised source plus preserved semantics/privacy. Merge, deploy and production smoke remain separate decisions.

## Checkpoint and next lawful action

**Classification:** `REPLAN / CURRENT BLOCKER` for the D1.1 freshness owner only. The previous R2 pre-code PASS remains valid for unchanged semantics/privacy, but its daily-source assumption is superseded. The prepared post-code request against `dd558e2` is **HOLD / SUPERSEDED FOR SOURCE FIDELITY**, with no PASS/BLOCK verdict. Runtime branch bytes, PR-1, Visit Brief and D2 stay on HOLD.

**Next lawful action:** obtain the Product Owner's plain-language correction/acceptance of this source delta, then commission **one independent read-only R2 pre-code delta + affected cumulative review** of the Reception reader/auth/cache and Cockpit source swap under `PROCEDURES.md` P1/P4/P5.1. Only after that gate and a fresh writer checkpoint may a bounded two-repository implementation start. Do not run the old post-code request, modify runtime, open a release PR, merge or deploy from this checkpoint.
