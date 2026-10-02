# D1.1 Global Appointment Context — Independent Pre-Code Review Request

> **MODE:** fresh independent READ-ONLY R2 pre-code review.
> **PRODUCT:** Clinical Excellence Cockpit.
> **REVIEW TARGET:** `cockpit/D1_1_GLOBAL_APPOINTMENT_CONTEXT_DESIGN.md`
> **EXPECTED DESIGN BLOB:** `2b438c949e30d00a1d11d9d1431d766954320f15`
> **IMPLEMENTATION:** forbidden in this review.
> **D1.1 RUNTIME:** not started.

## Product Owner intent to preserve

The global Cockpit must show real Previous / Current / Next appointments rather than only osteoporosis appointments.

Home remains compact. After D1.1, clicking/tapping a patient will later open a floating Visit Brief, but **Visit Brief runtime is not part of D1.1**.

The Digital Secretary remains appointment/workflow owner. The weekly Osteoporosis Calendar remains Module-01 specific.

## Source-grounded fact that motivated this design

The existing Digital Secretary producer already sends a bounded complete future Cal.com snapshot to Clinical Excellence. The Clinical Excellence consumer currently classifies non-osteoporosis rows as `other` and discards them at ingest. The proposed design retains one normalized row per source appointment and keeps the existing osteoporosis endpoint filtered.

## Exact review questions

Issue **PASS** only if the design is implementable and preserves all material boundaries below.

1. **Single owner / reuse**
   - Does the design reuse the existing Digital Secretary → Clinical Calendar snapshot rather than creating a second schedule/calendar path?
   - Does retaining all rows in the existing normalized store avoid a competing appointment owner?

2. **Privacy / minimization**
   - Is retaining global patient display name + appointment reason necessary and bounded for clinician scheduling/visit preparation?
   - Is stripping `phone_e164` and `linked_patient_id` from effective `other` rows sufficient for D1.1?
   - Does the new browser projection exclude phone/link/raw transport/provider metadata?

3. **Snapshot reconciliation**
   - Does changing `_apply_import_item` from “discard unrelated” to “retain as other” preserve complete-snapshot cancellation/reschedule deletion?
   - Are manual osteoporosis classification overrides preserved?
   - Is the rollback treatment of candidate-only `other` rows explicit enough?

4. **Projection separation**
   - Does the existing `/clinical/calendar/appointments` remain osteoporosis-only?
   - Is a separate minimal `/clinical/calendar/cockpit-context` projection the correct boundary?
   - Is returning only Previous / Current / Next preferable to exposing a 30-day global list to the browser?

5. **Temporal semantics**
   - Previous and Current remain today-context.
   - Next crosses the local-day boundary and selects the earliest future stored appointment.
   - Lawful Aclasta overlap remains allowed only when explicitly classified; all other concurrent active overlaps fail closed.

6. **Freshness**
   - Does the design accurately state that D1.1 reuses the existing daily snapshot cadence and does not claim live Reception parity?
   - Is cadence/live-pull/manual-sync correctly deferred?

7. **Scope**
   - No Reception runtime mutation.
   - No new secret/config.
   - No booking/cancel/reschedule write.
   - No Visit Brief clinical-history implementation.
   - No GESY/D2 communication scope.

## Existing tests whose oracle changes

Review the design's explicit legacy-oracle disposition for:
- `test_snapshot_upserts_relevant_rows_filters_unrelated_and_removes_missing`;
- `test_snapshot_reclassification_removes_previously_relevant_row`.

A finding must distinguish a genuinely stale old oracle from a preserved invariant that the new design would violate.

## Required output

Return exactly:

```text
VERDICT = PASS | BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL
P0:P1:P2 = x:y:z
MATERIAL FINDINGS =
- <id / severity / reachable consequence / violated invariant / evidence / smallest correction>
NO ADDITIONAL MATERIAL FINDING = YES | NO
IMPLEMENTATION MAY START = YES | NO
```

Do not implement code. Do not merge or deploy. STOP after the review result.
