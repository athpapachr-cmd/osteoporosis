# D1.1 freshness R2 pre-code review — independent BLOCK

> **RESULT:** BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:3:0 / IMPLEMENTATION MAY START = NO.
> **TARGET DELTA BLOB:** `a6eedf7b465a55f962321e39d3e6c76d1de5676b`.
> **REQUEST:** `D1_1_FRESHNESS_PRECODE_REVIEW_REQUEST_2026-10-04.md`.
> **SOURCE IDENTITIES VERIFIED BY REVIEWER:** Osteoporosis main `12be588866aee6543444eee0b435e9064c39bc3f`, D1.1 branch `157982f`, Reception main `bcfa57e0c1ca7358fb898b393e1dafa7b042c238`.
> **MODE:** independent read-only source/contract review; no edits, tests, live requests or release action.

## Question dispositions

- **Q1 — MATERIAL FINDING D1.1-FRESH-01:** Reception's current dashboard and sync reader request only `status="upcoming"`. `CalClient.get_bookings` always sends a status parameter and separately documents `past` (`main.py:11552–11557`, `cal_setmore_sync.py:403–409`, `cal_client.py:620–643`). A fresh apparently complete schedule could omit a completed appointment needed for Previous. Smallest correction: specify bounded complete `past` **and** `upcoming` reads for the Nicosia local-day/future window, merge by UID, exclude cancelled/inactive rows, and require a source-level Previous regression.
- **Q2 — MATERIAL FINDING D1.1-FRESH-02:** Five-minute cache and single-flight do not bound a cache-miss request. The current Cal client defaults to six retries, a 20-second timeout per attempt and accepts `Retry-After` without a total deadline (`cal_client.py:39–42, 72–79, 149–183`). Smallest correction: strict end-to-end refresh deadline with scoped retry/page budget and unavailable on exhaustion; synthetic timeout/no-sync regression.
- **Q3 — MATERIAL FINDING D1.1-FRESH-03:** The shared client prints raw `resp.text` before projection (`cal_client.py:149–160`), so frequent bookings reads would log attendee content despite the minimal downstream response. Smallest correction: suppress raw booking response bodies on this reader path; log only non-identifying status/timing/error metadata. The proposed existing shared-key server-to-server authentication is otherwise supported.
- **Q4 — SUPPORTED:** existing resolver, weekly filter and effective-`other` minimization can survive source substitution. Existing tests establish only the old source behavior; new source acceptance remains future work.

```text
VERDICT = BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE
P0:P1:P2 = 0:3:0
NO ADDITIONAL MATERIAL FINDING = NO
UNREVIEWED / MISSING EVIDENCE = NONE within Q1–Q4
STOP REASON = Q1–Q4 disposed
IMPLEMENTATION MAY START = NO
RELEASE AUTHORITY = NO MERGE OR DEPLOY
```

Files examined: Osteoporosis `AGENTS.md`, `TODO.md`, `CLINICAL_EXCELLENCE_PLAN.md`, `SLICE_PLAN_CURRENT.md`, `CURRENT_OPERATIONAL.md`, `osteoporosis-change-log.md`, `PROCEDURES.md`, `cockpit/CURRENT.md`, frozen D1.1 design, freshness delta, prior pre-code closure, `clinical_calendar.py`, `clinical_auth.py`, `static/cockpit/app.js`, focused D1.1 tests; Reception `main.py`, `cal_client.py`, `cal_setmore_sync.py`, `clinical_calendar_feed.py`, `cal_config.py`, `render.yaml`, `cron_trigger.py`.

**Plan impact:** three `CURRENT BLOCKER / in-slice correction` findings. Preserve the already closed semantics/privacy questions. Next: bounded correction of FRESH-01/02/03 in the source delta, then **one** independent delta + affected cumulative closure review under PROCEDURES P5/P5.1. Do not implement runtime or run the old post-code request.
