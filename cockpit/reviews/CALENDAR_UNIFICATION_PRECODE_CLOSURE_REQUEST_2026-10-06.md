# Calendar Unification — bounded independent R2 closure review request

> **STATUS:** REQUEST ONLY / read-only closure review / no runtime implementation authority.
> **DATE:** 2026-10-06 Asia/Nicosia.
> **BRANCH:** `docs/cockpit-calendar-unification-visit-brief-2026-10-06`
> **CORRECTED DESIGN:** `cockpit/CALENDAR_UNIFICATION_REPLAN_2026-10-06.md`
> **CORRECTED DESIGN BLOB:** `e45e6e07052b71597f230b652d4b236cab433894`

## Original material finding to close

Independent R2 pre-code review returned:

- overall: **BLOCK**
- P0:P1:P2 = **0:1:0**
- Q1 BLOCK, Q3 consequential BLOCK
- finding: the existing protected Reception schedule-context route is anchored to today, while the Osteoporosis weekly UI can navigate arbitrary previous/next weeks. Without an explicit requested-week/date contract and proof of full-week coverage, a week outside the today-anchored projection could be rendered as empty rather than unavailable.

The reviewer proposed the smallest correction:

- parameterize the protected Reception read with the requested week/date;
- prove the full requested week is covered;
- show unavailable when coverage is incomplete;
- preserve Home's existing today-default behavior.

## Correction made

The design now freezes:

1. existing `GET /private/schedule-context` no-parameter behavior remains the Home/today default;
2. the same protected route accepts optional `target_date=YYYY-MM-DD`;
3. weekly Clinical Calendar sends the displayed week's Monday;
4. Reception uses the same `actual_schedule(target_date)` owner/cache path;
5. private response includes non-sensitive completed-coverage metadata plus `fetched_at`;
6. Cockpit must prove `[week_start, week_start+7d)` is fully inside completed coverage before interpreting an empty appointments list;
7. incomplete/missing coverage, provider failure or stale source => unavailable, never empty truth;
8. cache remains anchor-date scoped;
9. no arbitrary weekly navigation limit is introduced;
10. Home remains today-default and D1.1 semantics are untouched.

## Closure questions

### C1 — original P1
Does the corrected requested-date + explicit coverage contract close the exact Q1/Q3 finding?

### C2 — affected cumulative semantics
Does the correction preserve:
- one Reception provider-read owner;
- no Cal credentials in Cockpit;
- existing Home today semantics;
- weekly previous/next navigation;
- fail-closed source behavior;
- cancellation/reschedule freshness;
- minimized private response?

### C3 — implementation sufficiency
Is the design now specific enough to implement without inventing additional source/coverage behavior?

Check that:
- requested week anchor is unambiguous;
- coverage metadata has a clear purpose;
- empty week vs unavailable is deterministic;
- no second schedule engine is required.

### C4 — no new material risk
Did the correction introduce any new material P0/P1/P2 risk within the affected surface?

## Evidence allow-list

Use only the minimum needed from:
- corrected design;
- original pre-code handback supplied by Product Owner;
- current Reception `main.py` private schedule-context route;
- current Reception `schedule_projection.py` actual_schedule/read bounds/cache;
- current Cockpit weekly calendar navigation;
- current D1.1 contract where needed.

No broad archaeology, provider research, live calls, implementation or mutation.

## Stop rule

Return one of:
- PASS / COMPLETE_FOR_DECLARED_SCOPE / original finding CLOSED
- BLOCK with concrete remaining/new material finding
- UNKNOWN only if the closure question genuinely cannot be disposed

Once C1–C4 are disposed, STOP.

## Requested handback

- corrected design blob verified;
- P0:P1:P2 counts;
- original P1 CLOSED / NOT CLOSED;
- C1–C4 dispositions;
- explicit statement whether Calendar Unification implementation may start.

No merge/deploy/smoke authority is implied.
