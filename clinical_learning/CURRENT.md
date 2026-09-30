# Clinical Learning — CURRENT

STATUS: NAVIGATION FIX MERGED / AUTO-DEPLOY IN PROGRESS

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

Release:
- PR #125 merged as `d8252e4bd39d0c8ecbc2c1a42e17a00415cb3a4e`.
- final reviewed head `bac70baf956870a8d39d6216dd035b1cdcc36b17`.
- L1, L1B and canonical-impact gates: PASS.
- Render auto-deploy triggered after merge.

Exact next action:
No further code change. Confirm the normal auto-deploy reaches LIVE; then this navigation defect is closed.
