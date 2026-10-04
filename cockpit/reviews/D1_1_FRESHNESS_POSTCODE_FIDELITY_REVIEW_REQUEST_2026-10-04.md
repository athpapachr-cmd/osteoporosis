# D1.1 freshness — one bounded independent R2 post-code fidelity review request

> **STATUS:** REQUEST ONLY / independent verdict pending / no merge or deploy authority.
> **DATE:** 2026-10-04 Asia/Nicosia.
> **REVIEWED SOURCE DESIGN:** `cockpit/D1_1_FRESHNESS_REPLAN_DELTA_2026-10-04.md` immutable blob `00e3b4527a8ccbfc6df1a635494da136ccb9dd33`; its independent affected cumulative pre-code closure is PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0 in `cockpit/reviews/D1_1_FRESHNESS_PRECODE_CLOSURE_PASS_2026-10-04.md`.
> **PRESERVED PRIOR DESIGN:** `cockpit/D1_1_GLOBAL_APPOINTMENT_CONTEXT_DESIGN.md` blob `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33`; its unchanged semantics/privacy pre-code PASS remains evidence. The old `cockpit/reviews/D1_1_POSTCODE_REVIEW_REQUEST.md` is HOLD / superseded for source fidelity, with no verdict.

## Exact implementation target

| Repository | Base | Branch | Runtime/test head |
|---|---|---|---|
| `athpapachr-cmd/osteoporosis` | original D1.1 base `d5da6271567a4141b2708d9fa12e673dfce37131`; fresh main at kickoff `b9beca7b1e233245f9ea3429a247a1c80a69ac7c` had unrelated PHYSIO changes | `feat/cockpit-d1-1-global-context-2026-10-03` | `9dc14bbde1005aab60922e88e021518197f0654f` |
| `athpapachr-cmd/ortho-reception-backend-v2` | fresh main `bcfa57e0c1ca7358fb898b393e1dafa7b042c238` | `feat/d1-1-fresh-schedule-projection-2026-10-04` | `b4739678016c1cd10f3afd8381e7c4de222f38ba` |

Reception Ops canonical writer claim was published on main `1b98f7f023fdd92a55938591c49a49f12af5e916` after `orctl verify` and IMPLEMENTATION gate PASS. Separate exact-head EVIDENCE and REVIEW gates passed. No Reception or Cockpit branch is merged or deployed. No live Cal, patient, Render or production smoke was performed.

## What changed

- Reception now reads actual Cal bookings through one bounded `past` + `upcoming` UID-deduplicated projection with explicit pagination, local-day/future bounds, a combined ten-second deadline, no automatic provider retry, a 20-page ceiling, five-minute single-flight cache and mutation/sync invalidation. The separate scoped Cal read has no raw response-body logging. Its allowlisted secret-protected response omits phone, call context, raw attendees, linked clinical identity and creation time; the Reception dashboard retains its existing local call/creation-time presentation.
- The visible Reception control refreshes only the lightweight schedule; `/sync/both` remains administrative, and the daily heavy cron was not changed.
- Cockpit's existing protected same-origin route now consumes that private projection server-to-server. Existing Nicosia, Previous/Current/Next, overlap and cross-day semantics operate on its rows; unavailable/stale source fails closed without using the weekly Clinical Calendar as an operational fallback. The snapshot-only global `other` retention was removed; weekly Module-01 filtering, manual overrides and effective-`other` minimization remain.
- Cockpit Home labels the last actual-bookings read and shows unavailable/last-known time where available. It sends no Reception secret from the browser.

## Focused evidence on exact runtime/test heads

- Osteoporosis: 31 focused tests PASS (`test_clinical_calendar.py`, `test_clinical_calendar_snapshot.py`, `test_cockpit_home.py`, `test_cockpit_surgery_queue_ui.py`), Python and Cockpit JS syntax PASS, diff hygiene PASS.
- Reception: 54 focused tests plus 33 subtests PASS (`test_schedule_projection.py`, `test_schedule_routes.py`, `test_clinical_calendar_feed_bridge.py`, `test_deterministic_reception_flow.py`, `test_cancellation_authorization_contract.py`, `test_dashboard_key_header_auth.py`), Python syntax and diff hygiene PASS. Existing deprecation warnings only.
- Synthetic evidence covers past-only Previous, midnight Current, upcoming Next, actual urgent `eventTypeId=393065`, cancelled exclusion, UID conflicts, incomplete pagination, page ceiling, no-retry 429 body logging, deadline, single-flight, protected private transport and refresh without heavy sync. No live source completeness or performance claim is made.

## Finite reviewer questions

1. **FRESH-01 source and completeness:** Against both exact heads and Cal's actual v2 response contract, does the `past` + `upcoming` query/pagination merge make a complete bounded actual-bookings schedule for the D1.1 local-day/Next horizon? Check `hasNextPage`, accepted status semantics, UID/status conflicts, provider-filter behavior, cancelled/rescheduled bookings, urgent event type `393065`, and absence of availability slots. Is the explicit Aclasta/Prolia slug/type mapping supported by this source, or does it require a bounded correction before fidelity PASS?
2. **FRESH-02 freshness/operation:** Does the ten-second combined deadline, no-retry scoped request, 20-page ceiling, five-minute cache, single-flight and generation invalidation avoid partial/stale publication? Check actual create/cancel/reschedule and `/sync/both` success/partial paths, forced refresh, cached `fetched_at`, and whether the Reception dashboard and Cockpit truly consume the same projection without heavy sync in call hours.
3. **FRESH-03 privacy/auth:** Is the shared ingest-secret check server-to-server only and constant-time? Is `RECEPTION_SCHEDULE_CONTEXT_URL` a release-time configuration dependency with no browser credential leak? Confirm private response and new schedule-read logs contain no phone, attendee/raw provider payload, clinical linked identity or call context; errors fail closed.
4. **Preserved D1.1 semantics:** Recheck Nicosia DST/midnight, start-order Previous, active Current, strict earlier-Aclasta exception, ambiguous overlap, cross-day Next, compact Home/weekly link and absence of clinical patient matching or Visit Brief handler. Confirm snapshot-only global `other` retention removal does not regress weekly filtering, manual overrides, legacy discard or effective-`other` minimization.
5. **Scope and evidence:** Mark each finding P0/P1/P2 with exact file/line and a reproducible scenario; distinguish test-backed behavior from live-source/installed-path UNKNOWN. Return PASS/BLOCK/UNKNOWN only for this affected cumulative R2 scope. Do not infer merge, deploy or production-smoke authority.

## Requested handback

One independent, read-only exact-head fidelity verdict with P0:P1:P2 counts, question-by-question disposition, evidence references, exact tested heads and any bounded correction required. The implementation author does not self-certify fidelity. Preserve the prior HOLD request as historical, and keep merge/deploy/smoke on HOLD.
