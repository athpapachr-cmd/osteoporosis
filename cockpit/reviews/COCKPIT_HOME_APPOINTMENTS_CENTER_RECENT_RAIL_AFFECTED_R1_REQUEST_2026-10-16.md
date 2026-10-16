# Cockpit Home appointments in center, recent encounters in rail — one independent affected R1

**MODE: fresh separate READ-ONLY POST-CODE R1, one bounded implementation-fidelity review. STOP.** No implementation, redesign, merge, deploy or patient data.

Repository `athpapachr-cmd/osteoporosis`; [DRAFT PR #160](https://github.com/athpapachr-cmd/osteoporosis/pull/160), source base verified `main=32592ee76ce396b014eb51ba717ace50fed591aa` at author activation; re-verify actual main/head. **Frozen code+test HEAD** `509beeb764b0d334f4f04b7d02797e67c2a0723e`, tree `20409de9383f88d4e29b574d9d76b556c3c204c7`. Actual PR HEAD may advance through **docs-only** closure checkpoint; verify all affected blobs identical. Read `cockpit/reviews/COCKPIT_HOME_APPOINTMENTS_CENTER_RECENT_RAIL_AUTHOR_HANDBACK_2026-10-16.md` and reuse independently closed prior Home V4/nav/attention/RF/Visit Capture reviews, no repeated broad architecture review.

## Single finite affected scope (R1–R5)

- **R1** source identity and bounded diff: only three Home HTML/CSS/JS and two existing synthetic/Chromium tests materially changed; no change to calendar API, key/session auth, protected patient identity, RF/surgery data or other clinical writers.
- **R2** layout: `#visitRecentRows` is in `#visitSidebar` with exactly 3 recent visits from same source. `#visitSidebarPrevious`, `Current`, `Next` plus #calendarState/note/retry are in central main area preceding existing attention, no duplicate IDs/handlers. Main search and Home V4 attention untouched.
- **R3** real split viewport and phone: at 1280 sidebar present, center appointments; at 680/sidebar hidden behind ☰, all three appointments **visible without opening ☰**; at 390 no horizontal overflow and meaningful date/status/slots. Sidebar opens for 3 recent visits and selecting one closes overlay, enters original patient workspace. Use existing real Chromium/synthetic CI, do not infer from CSS text alone.
- **R4** preserved behavior/security: peek on populated appointment and disabled empty slot; source unavailable (403/401/503) truthful, no invented bookings or local browser PHI; protected recent one-click opens same identity boundary, no extra Save/provider effects. Check no new clinical writes, no unauthorized details from weak appointment name.
- **R5** affected cumulative evidence: browser Home Chromium [38846799081](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846799081) and [38846802273](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802273), Visit Capture [38846802296](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802296), Surgery [38846802259](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802259), Canonical Impact [38846802167](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802167) **SUCCESS** on source/test HEAD. Reuse previous PASS, ignore only proven unrelated diff-scope failures.

## Frozen source/test blob oracle

```
static/cockpit/index.html               278741f812ccc47cbd0c768a0a042e2beb8bd6a1
static/cockpit/doctor-shell.css          c960759f498d03e079d6006579df42e1d609892d
static/cockpit/clinical-workspace.js     66d2068af2730f33e12caa6f98bbbcc84160b3e1
test_cockpit_unified_home.js             2c47fa6758699820a8d401858d13813224cd953d
test_cockpit_doctor_browser.py           e134930413352b2b82ca25b334e6d14aa57804b7
```

Return **IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE | PARTIAL**, **P0:P1:P2**, source-grounded R1–R5 and any smallest actionable problem, then **STOP**. This is not production approval. No modifications by reviewer.
