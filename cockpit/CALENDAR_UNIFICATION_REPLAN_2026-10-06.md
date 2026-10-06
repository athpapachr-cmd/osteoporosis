# Calendar Unification Replan — 2026-10-06

> **STATUS:** PRODUCT OWNER CONFIRMED / DESIGN REPLAN ONLY / RUNTIME NOT YET AUTHORIZED.
> **Scope:** unify appointment existence/freshness across Cal, Reception, Cockpit Home and the Osteoporosis Clinical Calendar without reopening D1.1 semantics.
> **Product Owner confirmation:** 2026-10-06 — proceed with Calendar Unification, then immediately continue to Visit Brief / Lab Results workflow.
> **Fresh main at branch creation:** `f9bf0189f619d440cd8828e554a52e0802895af8`.

## 1. Problem now

The clinician currently sees three appointment views that can disagree temporarily:

1. Cal — provider source of actual bookings.
2. Reception daily schedule — fresh operational projection of actual Cal bookings.
3. Osteoporosis Clinical Calendar — a separate daily snapshot/reconciliation path.

Production evidence on 2026-10-06 showed the failure mode directly: a newly created Aclasta booking appeared immediately in Cal/Reception while the weekly Osteoporosis Calendar remained stale until a later manual heavy sync succeeded. The daily 04:00 UTC cron had failed; the later manual sync refreshed the Clinical Calendar correctly.

The product problem is therefore not Aclasta classification. It is duplicated appointment serving paths with different freshness.

## 2. Product-owner step card

**What the clinician sees today:** different calendar surfaces can disagree until the Clinical Calendar snapshot refreshes.

**What changes after this step:** all appointment-existence/time views derive from the same bounded actual-Cal schedule owned by Reception. The Osteoporosis calendar becomes a clinical projection over that same schedule rather than a separately refreshed copy.

**Example:** an 11:00 Aclasta booked in Cal appears in Reception immediately and, on opening the Osteoporosis Calendar, appears there too without waiting for the daily heavy cron.

**What stays outside this step:** booking creation/cancellation/rescheduling semantics, urgent availability, lab/telephone availability, Setmore reconciliation, Visit Brief, Gmail, SMS and clinical-record writes.

**Why this step comes first:** Visit Brief and lab-result workflows need a trustworthy single appointment context and should not inherit three competing appointment freshness models.

**Next milestone:** Visit Brief / Lab Results workflow.

## 3. Frozen architecture

```text
Cal actual bookings
    ↓
Reception lightweight actual-schedule projection
    ↓
Cockpit server-to-server protected read
    ├── Home: Previous / Current / Next
    └── Osteoporosis Clinical Calendar:
         actual booking
         + existing osteoporosis classifier
         + existing manual classification override
         → weekly clinical projection
```

### Source ownership

- **Cal** owns whether an actual booking exists and its provider timing/status.
- **Reception** owns the always-on bounded provider read, caching, pagination, invalidation and minimized operational projection.
- **Cockpit/Osteoporosis** owns the clinical interpretation of those bookings: osteoporosis / first visit / review / Prolia / Aclasta / manual override.
- The free Cockpit service is a consumer. It must not become the background synchronization owner.
- The paid Reception service remains the always-on schedule reader.

## 4. Reuse decision

**REBIND**, not a second calendar engine.

Reuse the D1.1 Reception private actual-schedule projection already used by Previous / Current / Next. Do not add Cal credentials to Osteoporosis and do not create a second Cal client.

Reuse the existing Osteoporosis `classify_appointment` semantics and manual-classification table.

## 5. Weekly Clinical Calendar serving rule

The weekly Osteoporosis Calendar must no longer require the daily snapshot store to determine whether an appointment currently exists.

### Requested-week source contract — correction after independent R2 BLOCK

The current Reception private route is today-anchored. That is sufficient for Home but not for an arbitrarily navigated weekly calendar. The unified design therefore extends the **existing** protected Reception read rather than creating another reader.

Contract:

- `GET /private/schedule-context` keeps its current no-parameter behavior for Cockpit Home: anchor = today's Asia/Nicosia date.
- The same protected route accepts an optional ISO local-date anchor, e.g. `target_date=YYYY-MM-DD`.
- The Osteoporosis weekly Calendar sends the displayed week's Monday as `target_date`.
- Reception calls the same cached `actual_schedule(target_date)` projection already used for today; no new Cal client, credentials or provider-read owner is introduced.
- The protected response exposes non-sensitive coverage metadata for the completed provider read, at minimum `coverage_start` and `coverage_end`, together with `fetched_at`.
- Coverage metadata represents the bounded local-date interval whose provider pagination completed successfully. An empty appointments list is valid **only** when that interval is explicitly complete.
- Before rendering an empty or partial week as truth, Cockpit validates that the full requested local interval `[week_start, week_start + 7 days)` is contained in the returned coverage interval.
- Missing/invalid coverage metadata, incomplete coverage, provider/page/deadline failure, or a requested week outside completed coverage => **unavailable**, never “zero appointments”.
- Existing Reception cache keys remain anchor-date scoped, so Home/today and separately requested historical/future weeks cannot silently reuse the wrong coverage.
- No UI navigation limit is invented merely to fit today's projection; previous/next week navigation remains supported through the requested-date contract.

The current provider projection already reads a bounded window around its supplied anchor date. Implementation may preserve that bounded window so long as the response proves coverage and the consumer verifies the requested week is fully contained.

For each live actual booking in the requested week:

1. derive stable clinical appointment key from the actual booking UID using the existing Cal source identity convention;
2. preserve Reception-provided explicit `aclasta` / `prolia` classification where present;
3. otherwise apply the existing Osteoporosis classifier to the source-proven reason and derived duration;
4. apply an existing clinician manual override, if one exists;
5. include only the existing osteoporosis-relevant categories;
6. keep missing/ambiguous reason as unrelated/other unless a manual override establishes relevance.

The provider schedule is not a clinical patient identity source.

## 6. Manual overrides

Manual override semantics remain unchanged.

The implementation should reuse the existing stable Cal UID-derived appointment key so an override can survive schedule refreshes without depending on a duplicated snapshot row.

If the current write route requires a stored `ClinicalAppointmentORM` row merely to create/update an override, that coupling should be removed narrowly. The override table remains the authority for clinician classification; it must not write back to Cal or Reception.

Clearing an override returns to live automatic classification.

## 7. Failure semantics

No stale daily snapshot fallback for the clinician-facing weekly calendar.

“Empty week” and “source could not prove this week” are different states.

For the weekly Calendar, Cockpit must validate both freshness and **full requested-week coverage** from the Reception response before interpreting absence of rows as absence of bookings.

If the Reception actual-schedule source is unavailable:

- fail closed;
- show a clear unavailable/stale-source state;
- preserve the last-known fetch timestamp where available;
- do not silently substitute the legacy daily snapshot and present it as current.

This matches the D1.1 freshness principle.

## 8. What happens to the existing daily Clinical Calendar snapshot

Do **not** delete it in this slice.

The daily heavy cron and snapshot delivery may continue for legacy reconciliation/compatibility while this unification is introduced. However:

- snapshot success is no longer required for the weekly calendar to show current actual bookings;
- the snapshot store is no longer the serving source for appointment existence in the weekly view;
- later removal/retirement is a separate cleanup decision after production evidence.

The heavy `/sync/both` path remains administrative/off-call-hours and is not used for interactive calendar refresh.

## 9. Availability is not an appointment

Normal, urgent, telephone and laboratory availability windows remain excluded.

Only actual accepted bookings enter the shared schedule projection.

A real urgent booking is included. An empty urgent/lab/telephone slot is not.

## 10. Privacy / identity boundary

Preserve D1.1 minimization:

- no browser-held Reception secret;
- no phone in the Cockpit schedule projection;
- no raw Cal attendees/provider body;
- no automatic clinical patient match from name/phone;
- no Visit Brief patient history until an authorized strong or clinician-confirmed link exists.

## 11. Render/cold-start constraint

Cockpit currently runs on a free Render service and may cold-start after inactivity.

Reception runs on paid Render and stays available.

Therefore unification must be pull-based from Cockpit to the already-running Reception projection when Cockpit wakes. Correctness must not depend on Cockpit being awake at 07:00 or receiving a push during a cron window.

## 12. Acceptance criteria

1. A new normal Cal booking appears in Reception and in the Osteoporosis weekly calendar from the same actual schedule source without waiting for the daily heavy snapshot.
2. A real urgent booking appears as an actual booking; urgent/lab/telephone availability-only windows do not.
3. A normal Limassol booking with source-proven reason `Aclasta` is classified Aclasta in the weekly calendar.
4. Existing osteoporosis first/review classification behavior remains unchanged for source-proven reasons and durations.
5. Existing manual overrides survive provider refresh; clearing override returns to automatic classification.
6. Cancelled/rescheduled actual bookings disappear/move according to the Reception projection without waiting for the daily snapshot.
7. Reception source unavailable, stale, or unable to prove full requested-week coverage → weekly calendar shows unavailable/freshness state, not an empty week and not a silent snapshot fallback.
8. Home Previous / Current / Next semantics remain byte/behavior compatible except for shared helper reuse if needed.
9. No Cal credential, phone, raw provider payload or weak patient matching is added to Cockpit.
10. The heavy sync cadence remains unchanged.
11. Home continues to use the existing today-default private schedule read with no target-date requirement from the browser.
12. Weekly previous/next navigation requests the displayed week's local anchor and proves the entire seven-day interval is covered before rendering “no appointments”.

## 13. Review tier

**R2 pre-code review required** because this changes the authoritative serving source for a clinical calendar projection and the interaction between live source rows and persisted manual clinical classification.

Review only:
- live schedule → weekly clinical projection;
- stable override identity;
- failure/freshness behavior;
- preservation of D1.1 and weekly classification semantics.

Do not reopen booking architecture, D1.1 overlap semantics or broader Module-01 design.
