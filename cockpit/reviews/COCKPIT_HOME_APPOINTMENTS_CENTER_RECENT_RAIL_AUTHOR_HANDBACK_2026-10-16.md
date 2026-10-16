# Cockpit Home recent visits / appointments — implementation author handback

**Date** 2026-10-16; **scope** Home V4 presentation-only, Product Owner-reported split-screen usability.

- Original verified production `main`: `32592ee76ce396b014eb51ba717ace50fed591aa`.
- DRAFT [PR #160](https://github.com/athpapachr-cmd/osteoporosis/pull/160) on `feat/cockpit-home-appointments-center-recent-rail-20261016`.
- Exact frozen code+test HEAD: `509beeb764b0d334f4f04b7d02797e67c2a0723e`, tree `20409de9383f88d4e29b574d9d76b556c3c204c7`.
- Subsequent commits from author: operational/review documentation **only**; source/test blobs must remain identical.
- One existing owner: Home V4, not a new visit/calendar engine.

## Source-bounded implementation

- `static/cockpit/index.html`: move **same** `#visitRecentRows` list to the left rail `#visitSidebar`. Move **same** Previous/Now/Next controls `#visitSidebarPrevious`, `#visitSidebarCurrent`, `#visitSidebarNext`, `#calendarState`, `#calendarNote`, `#cockpitCalendarRetry` and calendar link into central first Home card; preserve attention card/source and initial search. Mark rail ☰ as recent visits.
- `static/cockpit/doctor-shell.css`: compact 3-row appointments on desktop beside attention, 3 cards horizontally at 561–990px split, vertical at phone widths; recent encounters in compact rail, no overflow.
- `static/cockpit/clinical-workspace.js`: click recent visit closes ☰ rail first at half-width before opening the already-protected selected-patient workspace; unchanged patient identity and Dia/Save boundaries. Existing one-click recent button/Peek remains.
- `test_cockpit_unified_home.js` & `test_cockpit_doctor_browser.py`: assert actual position of unique live IDs, split visibility for all three appointments while rail hidden, no horizontal scrolling, recent row click from rail closes it and opens patient, regression existing Peek/Dia/attention.
- No changes to `static/cockpit/app.js`, `static/cockpit/doctor-shell.js`, `clinical_calendar.py`, RF/surgery backend, Clinical Key, provider or medical record writers. Calendar unavailable remains unavailable with original source messages, no fabricated appointments.

## Evidence (exact frozen HEAD)

- [Cockpit Home + Chromium #38846799081](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846799081) SUCCESS; desktop 1280, half-width 680, mobile 390 browser flows.
- [Cockpit Home #38846802273](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802273) SUCCESS, including canonical HTML/JS and actual Chromium.
- [Visit Capture #38846802296](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802296) SUCCESS.
- [Surgery Queue #38846802259](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802259) SUCCESS.
- [Canonical Impact #38846802167](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38846802167) SUCCESS.
- Unrelated strict workstream diff-scope red is not an identified Home functional regression.

### Exact tested Git blob oracle

```
static/cockpit/index.html              278741f812ccc47cbd0c768a0a042e2beb8bd6a1
static/cockpit/doctor-shell.css         c960759f498d03e079d6006579df42e1d609892d
static/cockpit/clinical-workspace.js    66d2068af2730f33e12caa6f98bbbcc84160b3e1
test_cockpit_unified_home.js            2c47fa6758699820a8d401858d13813224cd953d
test_cockpit_doctor_browser.py          e134930413352b2b82ca25b334e6d14aa57804b7
UNCHANGED static/cockpit/app.js         9e7e7053eb1a861fe8441831f2cb335b287c259f
UNCHANGED clinical_calendar.py          024a95d71bf1fbd26ee24d242fb8c7081458f65e
```

**Mode/next:** single independent **affected R1 post-code implementation-fidelity review** with exact source and acceptance. Source frozen, author writer released. PR remains DRAFT/no merge/deploy/patient use. After PASS only separate PO production release authority.
