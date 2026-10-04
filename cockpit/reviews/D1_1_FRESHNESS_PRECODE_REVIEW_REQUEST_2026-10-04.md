# D1.1 freshness — one independent R2 pre-code delta review

## Decision and target

Decide whether the Product Owner-confirmed source/freshness delta is safe and implementable as the smallest correction to operational Previous / Current / Next. Review the affected cumulative behavior once, reusing the previous pre-code PASS for unchanged semantics/privacy. **Read-only. No runtime edit, test execution, PR, merge, deploy, Visit Brief or old post-code review.**

```text
REPOSITORY = athpapachr-cmd/osteoporosis
MAIN = 12be588866aee6543444eee0b435e9064c39bc3f
D1.1 BRANCH BEFORE REVIEW CHECKPOINT = 64d5a890b0a1ee5f3a523bf73961ef74938f72e3
NEW SOURCE DELTA = cockpit/D1_1_FRESHNESS_REPLAN_DELTA_2026-10-04.md
NEW SOURCE DELTA BLOB = a6eedf7b465a55f962321e39d3e6c76d1de5676b
FROZEN PRIOR DESIGN BLOB = 261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33
CURRENT RUNTIME/TEST HEAD = dd558e2fa8d59409d21913d2d3f2fbf885fdfece
RECEPTION REPOSITORY = athpapachr-cmd/ortho-reception-backend-v2
RECEPTION MAIN = bcfa57e0c1ca7358fb898b393e1dafa7b042c238
PRODUCT OWNER P1 = CONFIRMED, “Προχώρα”, 2026-10-04
OLD POST-CODE REQUEST = HOLD / NO VERDICT
```

Fresh-verify both remote main identities and the D1.1 branch. Bootstrap the six Osteoporosis canonicals once in AGENTS order, then PROCEDURES P4/P5/P5.1 and `cockpit/CURRENT.md`. Use the exact Reception main and design blob above; if either changed, state the concrete drift. Source paths below are an evidence map, not an invitation to browse unrelated workstreams.

## Finite evidence map

| Question | Smallest evidence and required disposition |
|---|---|
| Q1 — Actual bookings and completeness | Reception `main.py` `/dashboard/schedule` read, `CalClient.get_bookings`, `cal_setmore_sync.py`/`clinical_calendar_feed.py` only where they establish source fields/limits. Is reuse of the Cal client with a bounded, paginated, status-checked UID projection implementable? Does the design include `eventTypeId=393065` as an actual booking while excluding availability windows, and make incomplete/error reads unavailable rather than empty? Identify any source-proven filter or horizon contradiction. |
| Q2 — Freshness and call-agent isolation | Reception `main.py` cache, dashboard `runSync`, `/sync/both`, daily trigger, and directly affected booking mutation callers if needed. Can one short-lived, single-flight cache serve both consumers, with true `fetched_at`, refresh bypass/invalidation and no heavy sync in the call-hour request path? Does the design account for cache invalidation after sync and provider failure without false freshness? |
| Q3 — Protected minimal cross-service boundary | Reception `clinical_calendar_feed.py` key configuration/delivery, Reception dashboard key dependency only as necessary; Osteoporosis `clinical_calendar.py`, `clinical_auth.py`, Home fetch and frozen design privacy clauses. Is reuse of the existing shared ingest key for a private read endpoint sound and actually available on both sides? Can the projection omit phone/link/raw attendee/call data and avoid browser cross-origin credentials or patient matching? If the credential proposal is unsafe or infeasible, name the smallest alternative. |
| Q4 — Preserved cumulative D1.1 behavior and affected delta | Frozen design §§5–12, accepted PRE-01/PRE-02 closure, new delta, current `clinical_calendar.py` resolver/store/weekly route, `static/cockpit/app.js`, focused D1.1 tests. Can source substitution keep Nicosia day, start-order Previous, cross-day Next, Aclasta/overlap fail-closed, weekly filter/link and effective-other minimization? Is retiring not-yet-released global `other` storage correctly bounded? Name any acceptance oracle missing for a reachable material risk. |

For each question, report `SUPPORTED`, `MATERIAL FINDING`, or `UNRESOLVED` with the exact source and smallest correction. Existing author tests are evidence of the old runtime behavior, not proof of the new source. No live patient data, production requests, broad test run, dependency installation, web research or repeated review is needed for this pre-code decision. Expand only for a concrete caller/writer/contradiction and state why. Stop once Q1–Q4 are disposed or a decisive material BLOCK and correction path are established.

## Terminal report

```text
TARGET SOURCE DELTA BLOB = a6eedf7b465a55f962321e39d3e6c76d1de5676b
VERDICT = PASS | BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL
Q1: ...; Q2: ...; Q3: ...; Q4: ...
P0:P1:P2 = ...
MATERIAL FINDINGS = NONE | reachable behavior / violated invariant / exact evidence / smallest correction
NO ADDITIONAL MATERIAL FINDING = YES | NO
UNREVIEWED / MISSING EVIDENCE = NONE | exact gap
FILES EXAMINED = ...
STOP REASON = Q1–Q4 disposed | decisive material BLOCK | exact required evidence gap
IMPLEMENTATION MAY START = YES | NO
RELEASE AUTHORITY = NO MERGE OR DEPLOY
```

`PASS` closes only the replan's pre-code source gate. `BLOCK` permits only bounded design correction followed by one delta + affected cumulative closure under PROCEDURES P5. Do not launch the old post-code review or implement runtime from this read-only review.
