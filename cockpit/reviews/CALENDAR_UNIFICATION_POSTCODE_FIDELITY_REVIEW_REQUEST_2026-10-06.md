# Calendar Unification — one bounded paired exact-head R2 post-code fidelity review request

> **STATUS:** REQUEST ONLY / independent read-only verdict pending / no merge, deploy or smoke authority.
> **DATE:** 2026-10-06 Asia/Nicosia.
> **CORRECTED DESIGN:** `cockpit/CALENDAR_UNIFICATION_REPLAN_2026-10-06.md`
> **CORRECTED DESIGN BLOB:** `e45e6e07052b71597f230b652d4b236cab433894`
> **PRE-CODE CLOSURE:** PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0 / original requested-week P1 CLOSED.

## Exact implementation targets

| Repository | Base | Branch | Exact runtime/test head |
|---|---|---|---|
| `athpapachr-cmd/ortho-reception-backend-v2` | `ca8d027e2ba30cb83958fa345c0b51e6138bbeea` | `feat/calendar-unification-requested-week-2026-10-06` | `49a321531fbe8042d2b22cb6ba7b121e901879ae` |
| `athpapachr-cmd/osteoporosis` | `f9bf0189f619d440cd8828e554a52e0802895af8` | `feat/cockpit-calendar-unification-2026-10-06` | `f2fdb146366850fac659acb74bacd40e9ec88588` |

Later documentation/checkpoint commits on the Osteoporosis branch are not runtime/test review targets.

## Implemented contract

### Reception

- Existing `GET /private/schedule-context` remains the single protected provider-read seam.
- No-param behavior remains anchored to current Asia/Nicosia date.
- Optional strict ISO date-only `target_date=YYYY-MM-DD` anchors the same `actual_schedule(day)` reader/cache.
- The internal completed read records `coverage_start` and exclusive `coverage_end`.
- **Backward-compatible release seam:** no-param responses keep the deployed D1.1 public shape; coverage metadata is exposed only for requested-date reads. This is intended to permit Reception-first deployment without breaking the currently deployed Cockpit Home model that forbids unexpected response fields.
- Requested-date and no-param reads use the same bounded pagination/deadline/cache/invalidation path.
- No new Cal credential, provider client or schedule owner was added.

### Cockpit / weekly Osteoporosis Calendar

- Weekly `GET /clinical/calendar/appointments` no longer serves appointment existence from the daily snapshot store.
- It asks Reception for the displayed week's local Monday via `target_date`.
- It validates:
  - source response shape;
  - source freshness <= 5 minutes;
  - complete coverage of the requested local interval before accepting an empty week.
- Missing/stale/incomplete/unavailable source => 503 fail-closed; no daily-snapshot serving fallback.
- UI renders an explicit unavailable state rather than a false “no appointments” week and can display last-known source fetch time when available.
- Actual booking rows are filtered to the requested interval and clinically classified:
  1. explicit Reception `aclasta/prolia` category first;
  2. otherwise existing `classify_appointment("", reason, duration)`;
  3. clinician manual override last.
- Stable manual-override identity uses `cal.com:{uid}`.
- A clinician override can now exist for a live Cal row without requiring a duplicated `ClinicalAppointmentORM` snapshot row.
- Clearing an override returns to automatic classification on the next live read.
- Legacy snapshot ingest/reconciliation remains present for compatibility but is not the weekly serving source.
- Home `/cockpit-context` logic is intentionally unchanged.

## Exact focused evidence

### Reception head `49a321531fbe8042d2b22cb6ba7b121e901879ae`

- D1.1 schedule smoke correction run `37513820627`: PASS.
  - Python compile PASS.
  - focused schedule regressions: **14 passed**.
  - diff hygiene PASS.
- AC-4A validation run `37513820517`: PASS.

### Osteoporosis head `f2fdb146366850fac659acb74bacd40e9ec88588`

- Clinical Calendar snapshot tests run `37513937466`: PASS.
  - Python syntax PASS.
  - Clinical Calendar JavaScript syntax PASS.
  - deterministic Calendar tests: **24 passed**.
  - diff hygiene PASS.
- Canonical impact guard `37513937407`: PASS.
- Cockpit surgery queue `37513937510`: PASS.
- Clinical Learning L1 `37513937344`, L1B `37513937380`, L1C `37513937307`: PASS.
- Physio Cockpit `37513937379`, V5 `37513937392`, jurisdiction `37513937562`: PASS.
- Clinical Learning L0 `37513937429`: its contract validation step PASS; only its known design-only changed-file scope assertion fails because this PR intentionally changes runtime files. Treat applicability, not color, as the review question.

No live Cal/patient mutation, merge, deploy or production smoke was performed.

## Finite reviewer questions

### Q1 — requested-week source / coverage fidelity

Against both exact heads and the corrected design:

- Does strict `target_date` use the same Reception actual-schedule owner rather than create a second reader?
- Are `coverage_start` / exclusive `coverage_end` truthful only after the bounded provider read completes?
- Does the weekly consumer prove its entire requested local interval is inside completed coverage before accepting an empty result?
- Do incomplete pagination/deadline/provider failures remain unavailable rather than partial/empty publication?
- Are arbitrary previous/next weekly anchors supported without silently using the wrong cache key?

### Q2 — release compatibility / D1.1 preservation

Check specifically:

- no-param private response shape remains compatible with the currently deployed old Cockpit Home during a Reception-first deploy;
- requested-date calls receive coverage metadata;
- new Cockpit accepts both shapes;
- Home Previous/Current/Next source, overlap, freshness and cross-day semantics remain unchanged;
- Reception lightweight refresh and heavy sync behavior are untouched.

This release-order compatibility is material.

### Q3 — weekly clinical classification / manual override

Check:

- raw Cal UID maps deterministically to the existing `cal.com:{uid}` clinical override identity without double-prefixing;
- Reception explicit `aclasta/prolia` categories are preserved;
- other rows reuse the existing Osteoporosis classifier using only source-proven reason + derived duration;
- unrelated rows remain excluded unless a clinician manual override makes them relevant;
- manual override can be created/cleared without a snapshot row;
- clearing returns to automatic live classification;
- legacy snapshot/manual-override behavior is not materially regressed.

If the narrower classification-write response contract has an affected consumer beyond the weekly JS reload path, identify it concretely.

### Q4 — failure state / privacy

Check:

- source unavailable/stale/incomplete coverage never renders as an empty week;
- last-known fetch time is source freshness only and does not claim appointment truth;
- no silent snapshot fallback occurs;
- private server-to-server secret stays off browser;
- no phone, linked patient identity or raw provider payload enters the weekly response;
- no weak patient matching is introduced;
- actual urgent bookings remain eligible while availability-only urgent/lab/telephone windows remain absent because the source is actual bookings.

### Q5 — scope / evidence

For each material finding provide:
- P0/P1/P2;
- exact repository + file/line;
- reproducible scenario;
- violated design invariant;
- smallest bounded correction.

Distinguish code/test-backed behavior from live/provider UNKNOWN. Do not turn absence of production smoke into a defect by itself.

## Finite evidence rule

Use the smallest affected surface only:
- corrected Calendar Unification design;
- exact diffs/heads above;
- current focused tests/evidence above;
- current `schedule_projection.py`, private route, `clinical_calendar.py`, weekly JS and directly affected tests;
- D1.1 contract only where preservation is materially questioned.

Do **not**:
- reopen D1.1 architecture;
- review Visit Brief/Gmail/Zadarma;
- perform repo-wide archaeology;
- run live Cal/provider mutations;
- repeat passing suites merely for reassurance;
- review unrelated OR-NCONV/voice-agent work.

Once Q1–Q5 are disposed, or one decisive material BLOCK is established, STOP.

## Requested handback

Return exactly one:
- **PASS / COMPLETE_FOR_DECLARED_SCOPE**
- **BLOCK**
- **UNKNOWN** only for a materially unresolved question

Include:
- exact Reception and Osteoporosis runtime/test heads reviewed;
- corrected design blob verified;
- P0:P1:P2 counts;
- Q1–Q5 dispositions;
- any narrow non-blocking UNKNOWN;
- smallest correction if needed;
- explicit statement whether Calendar Unification may proceed to separate merge/deploy release gating.

No merge/deploy/smoke authority is inferred.
