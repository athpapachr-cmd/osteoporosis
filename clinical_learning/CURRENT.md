# Clinical Learning — CURRENT

STATUS: NAVIGATION FIX IMPLEMENTED / REGRESSION GATES PENDING

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

Exact next action:
Run the existing Clinical Learning regressions and canonical-impact guard; merge only if the exact head is green.
