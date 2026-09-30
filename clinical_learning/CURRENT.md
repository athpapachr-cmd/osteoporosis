# Clinical Learning — CURRENT

STATUS: NAVIGATION FIX IMPLEMENTED / REGRESSION GATES PASS / PRE-MERGE

Task: add a direct return path from Clinical Learning Hub to the global Clinical Excellence Cockpit.

Scope:
- visible `← Cockpit` link in the Learning Hub header;
- target `/static/cockpit/`;
- responsive styling;

Out of scope:
- learning data contracts;
- challenge/foundation/loop behavior;
- patient records;
- Reception / Calendar;
- Cockpit Home layout beyond the destination route.

Release candidate branch:
`fix/clinical-learning-cockpit-backlink-2026-09-30`

Evidence:
- Canonical impact guard: PASS.
- Clinical Learning L1 regression gate: PASS.
- Clinical Learning L1B regression gate: PASS.
- Scope/adjacent-owner guards: PASS.
- Diff hygiene: PASS.

Exact next action:
Fresh-verify PR #125 exact head/base and merge if unchanged/mergeable.
