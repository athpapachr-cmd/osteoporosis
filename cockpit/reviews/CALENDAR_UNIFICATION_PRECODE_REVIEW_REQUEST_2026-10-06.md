# Calendar Unification — bounded independent R2 pre-code review request

> **STATUS:** REQUEST ONLY / read-only review / no runtime implementation authority.
> **DATE:** 2026-10-06 Asia/Nicosia.
> **BRANCH:** `docs/cockpit-calendar-unification-visit-brief-2026-10-06`
> **BRANCH HEAD AT REQUEST CREATION:** `fc8a6f4762691f238d4ca6367fa00047d77dbfa8`
> **DESIGN:** `cockpit/CALENDAR_UNIFICATION_REPLAN_2026-10-06.md`
> **DESIGN BLOB:** `d70c2e05871fdc5bca11fd591898b9beac58d518`

## Context

Production use proved that Cal/Reception could contain a newly created Aclasta booking while the Osteoporosis Clinical Calendar remained stale until the separate daily snapshot sync succeeded. The Product Owner confirmed a Calendar Unification replan.

The intended architecture is:

```text
Cal actual bookings
→ Reception lightweight actual-schedule projection
→ protected Cockpit server-to-server read
   ├─ Home Previous/Current/Next
   └─ weekly Osteoporosis Clinical Calendar
        + existing clinical classifier
        + existing manual override
```

The free Cockpit service may cold-start. The paid Reception service is the always-on schedule reader.

## Frozen boundaries

Do not reopen:
- D1.1 Previous/Current/Next semantics;
- Cal booking ownership;
- urgent availability design;
- heavy /sync/both cadence;
- Visit Brief / Gmail / Zadarma;
- broader Module-01 architecture.

No runtime mutation is authorized by this request.

## Finite questions

### Q1 — source/ownership

Is the proposed REBIND correct and minimal?

Check:
- Cal remains appointment source of truth;
- Reception remains the single provider-read owner;
- Cockpit does not gain Cal credentials;
- weekly Calendar can consume the existing minimized Reception projection safely;
- the free-Cockpit / always-on-Reception constraint is handled correctly.

### Q2 — clinical classification/manual override

Can the weekly Calendar preserve existing semantics over live actual rows?

Check:
- stable UID-derived key compatibility with existing `cal.com:{uid}` identity;
- explicit Reception `aclasta/prolia` category first;
- existing `classify_appointment("", reason, duration)` for other rows;
- existing manual override survives refresh;
- clearing override returns to automatic classification;
- no provider write-back.

If current override routes require a persisted snapshot row, identify the smallest implementation correction needed.

### Q3 — failure/freshness

Check:
- no stale daily snapshot fallback presented as current;
- source unavailable fails closed with visible freshness/unavailable state;
- cancellation/reschedule follows Reception projection;
- no dependence on daily 04:00 UTC snapshot success;
- heavy sync remains separate.

### Q4 — preservation/privacy

Check:
- urgent/lab/telephone availability-only windows remain excluded;
- actual urgent bookings remain included;
- Home D1.1 behavior remains preserved;
- no phone/raw attendee/provider payload is added;
- no weak patient identity matching is introduced;
- existing weekly filter/manual classifications remain clinically equivalent.

## Evidence allow-list

Use only what is necessary from:

- `cockpit/CALENDAR_UNIFICATION_REPLAN_2026-10-06.md`
- `cockpit/PRODUCT_CONSTITUTION.md`
- `cockpit/CURRENT.md`
- `clinical_calendar.py`
- focused Clinical Calendar tests
- D1.1 actual-schedule contract/tests already on current main
- current Reception `schedule_projection.py` only if needed to verify the consumed contract
- applicable `AGENTS.md` / `PROCEDURES.md`

Do not perform whole-repo archaeology, broad provider research, live Cal calls, production mutation, or repeat unrelated review history.

## Stop rule

Return one of:

- PASS / COMPLETE_FOR_DECLARED_SCOPE
- BLOCK with concrete P0/P1/P2 findings
- UNKNOWN only where a material question genuinely cannot be disposed from the allow-listed evidence

Once Q1–Q4 are disposed, STOP.

## Requested handback

- exact heads inspected;
- design blob verified;
- P0:P1:P2 counts;
- Q1–Q4 disposition;
- smallest required correction if any;
- whether implementation may start.

No merge/deploy/smoke authority is inferred.
