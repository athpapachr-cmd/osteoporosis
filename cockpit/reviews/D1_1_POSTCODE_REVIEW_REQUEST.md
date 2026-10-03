# D1.1 — one independent post-code exact-head fidelity review

## Decision and immutable identities

Decide whether the bounded D1.1 implementation faithfully realizes the already accepted corrected design, with the affected preserved behavior safe for a separate release decision. This is the sole R2 post-code review request, not a repeated pre-code review or whole-repository audit. **Read-only; no implementation, merge, deployment or Visit Brief work.** Prepared 2026-10-03; no reviewer launched and no verdict implied.

```text
REPOSITORY = athpapachr-cmd/osteoporosis
BRANCH = feat/cockpit-d1-1-global-context-2026-10-03
IMPLEMENTATION HEAD = dd558e2fa8d59409d21913d2d3f2fbf885fdfece
BASE MAIN = d5da6271567a4141b2708d9fa12e673dfce37131
FROZEN DESIGN BLOB = 261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33
DESIGN PATH = cockpit/D1_1_GLOBAL_APPOINTMENT_CONTEXT_DESIGN.md
ACCEPTED PRE-CODE = PASS / COMPLETE_FOR_DECLARED_SCOPE / PRE-01 CLOSED / PRE-02 CLOSED / 0:0:0
RELEASE AUTHORITY = NOT GRANTED
```

The target is the immutable implementation commit above, not a moving branch name. A following docs-only checkpoint records that SHA and this request. It changes no runtime/test bytes. Verify that relation once; review the full changed runtime/test surface against BASE MAIN, not just the last commit. The frozen design remains byte-identical, including historical pending-review header; the supplied closure artifact and CURRENT supersede that status without changing design semantics.

## Bootstrap and finite source scope

Fresh-verify GitHub main and bootstrap once in AGENTS order: AGENTS, TODO, CLINICAL_EXCELLENCE_PLAN, SLICE_PLAN_CURRENT, CURRENT_OPERATIONAL, changelog; then applicable PROCEDURES P4/P5/P5.1 and cockpit/CURRENT. Build the manifest, honoring the runtime/release HOLD. Fresh-main drift requires only the concrete changed dependency to be inspected; do not recursively follow historical entries, registries, reviews or unrelated workstreams.

Read the corrected design and supplied pre-code closure once. Runtime/test allow-list:

- `clinical_calendar.py` — both import callers, helper, classification writer, weekly filter, new context model/resolver/auth.
- `test_clinical_calendar_snapshot.py` — changed storage oracles and protected/global/API regressions.
- `test_clinical_calendar.py` — unchanged classifier/reason sanitation evidence.
- `static/cockpit/app.js`, `static/cockpit/index.html` — global request/rendering, date/freshness/unavailable states and weekly link.
- `test_cockpit_home.py` — real JavaScript loader/renderer harness and unchanged navigation constraints.
- `clinical_auth.py` — unchanged existing cookie-to-header path, only as needed for Q3 protection.
- `test_cockpit_surgery_queue_ui.py`, `test_g4_workspace_ergonomics.js` — unchanged shared UI/navigation evidence, only as needed for Q4 preservation.

Inspect no other runtime unless a specific caller/writer/contradiction requires it; state that reason before expansion. No web research, Reception inspection/mutation, live service/data, broad tests, dependency installation, history archaeology, CI polling or repeated searches for reassurance. Local focused evidence below can be reused because the runtime/test bytes are immutable. Missing required evidence is UNKNOWN, never PASS.

## Evidence map and terminal conditions

| Question | Finite evidence sufficient for disposition |
|---|---|
| Q1 — Snapshot-only admission/reconciliation and legacy compatibility | Inspect internal default-off flag and exactly two import callers; complete snapshot prevalidation before writes; same-source/exact-window missing-row removal and unchanged override-before-category decision. Match API tests for mixed storage/weekly filtering, missing rows, invalid scope/interval, producer bounds/DST, legacy discard/removal/counters and effective relevant overrides. |
| Q2 — Every effective other write is minimized, including clear override | Inspect shared sanitizer and import-helper/manual-clear calls before commit/response. Match relevant→other/repeated-other tests and auto-other/manual-relevant/populated-fields→clear regression, removed override, weekly exclusion, minimal global response; manual promotion does not restore fields without authorized relevant upsert. No new identity/linkage authority. |
| Q3 — Minimal protected global context, Nicosia time, cross-day Next and overlap behavior | Inspect explicit response models, protection + existing session adapter, resolver query and time/order predicates. Match API tests for other in all slots, forbidden-field/metadata exclusion, cross-day Next, local-day/DST/midnight, start-order Previous, strict earlier-Aclasta exception, tied/other/reverse overlaps fail closed and key/session protection. Assess the bounded freshness representation described below against design §8; no new feed/cadence owner. |
| Q4 — Home fidelity and adjacent preservation | Inspect single global fetch and real loader/renderer tests: no osteoporosis-list fallback, later-day date/time, counters removed, weekly link preserved, conflict/unavailable/freshness notices, plain human reason/context via textContent, no phone/link/raw metadata/browser persistence or Visit Brief handler/history. Match unchanged Surgery UI + G4 navigation evidence; confirm branch file list stays bounded. |

Return immediately when Q1–Q4 have dispositions, or when a reachable decisive material BLOCK plus the directly affected correction path is established. COMPLETE_FOR_DECLARED_SCOPE means these questions are disposed; it does not require proving the whole repository has no defect. NO ADDITIONAL MATERIAL FINDING means none found in this examined scope. Do not certify earlier reviews, invent findings or widen completeness. An UNKNOWN required question leaves the gate unpassed.

## Existing author evidence

Python 3.12.2 / local synthetic SQLite + TestClient + production JS functions executed by Node:

```text
python -m pytest -q test_clinical_calendar.py test_clinical_calendar_snapshot.py test_cockpit_home.py test_cockpit_surgery_queue_ui.py
30 passed in 1.16s
node --check static/cockpit/app.js: PASS
Python syntax for changed server/tests: PASS
node test_g4_workspace_ergonomics.js: PASS
git diff --check: PASS
```

These are author evidence, not independent fidelity PASS. No PostgreSQL concurrency change/schema migration exists. No production request/smoke performed.

Freshness limitation is explicit: `source_updated_at` is the latest retained normalized-row `updated_at`; no persisted snapshot receipt ledger is introduced. Manual classification also updates a row. Empty storage has unknown freshness and UI says so. UI shows last update and visibly notes when it is not today's Nicosia date. Do not infer last successful complete-feed time or live Reception parity. After any separately authorized release, global other context first populates through the existing scheduled complete snapshot; no manual sync/cadence change is included.

## Required terminal report

```text
TARGET IMPLEMENTATION HEAD = dd558e2fa8d59409d21913d2d3f2fbf885fdfece
TARGET DESIGN BLOB = 261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33
VERDICT = PASS | BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL
QUESTION DISPOSITIONS = Q1: ...; Q2: ...; Q3: ...; Q4: ...
P0:P1:P2 = ...
MATERIAL FINDINGS = NONE | reachable behavior / violated invariant / exact source / smallest correction
NO ADDITIONAL MATERIAL FINDING = YES | NO
UNREVIEWED / MISSING EVIDENCE = NONE | exact gap
FILES EXAMINED = ...
STOP REASON = Q1–Q4 disposed | decisive material BLOCK | exact required evidence gap
FIDELITY GATE PASSED = YES | NO
RELEASE AUTHORITY = STILL SEPARATE / NO MERGE OR DEPLOY
```

PASS closes this material review gate; separate release authority and release/smoke checkpoint remain required. BLOCK permits only bounded correction + one delta/affected-cumulative closure under PROCEDURES. Visit Brief stays deferred until D1.1 closed/released. **RETURN REPORT AND STOP.**
