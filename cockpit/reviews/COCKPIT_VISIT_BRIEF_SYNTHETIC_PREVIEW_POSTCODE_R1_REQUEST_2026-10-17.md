# Cockpit Visit Brief — synthetic-only preview, independent affected POST-CODE R1

Date: 2026-10-17 Asia/Nicosia.

**MODE:** fresh independent READ-ONLY R1, one bounded review. No code, merge, deploy, real patient rows, staff credentials or provider operations.

Repository: `athpapachr-cmd/osteoporosis`. PR [#165](https://github.com/athpapachr-cmd/osteoporosis/pull/165) DRAFT/HOLD, base verified `main=4cf219e8f3f254caeb88134577af397afc229410`.

**Exact immutable SOURCE+TEST checkpoint HEAD:** `53f30d7ff6a93d0ffacd746a0c01f055d39883c1`. Later review-only commit does not change runtime/test bytes.

## Exact blobs

- `static/cockpit/visit-brief-demo/index.html` — `975af5f156af2109ff9bfde4258a722511b824a5`
- `test_cockpit_visit_brief_demo_browser.py` — `8c33509b378a7d84f832f3fc0d6840ffade2da10`
- `.github/workflows/cockpit-visit-brief-demo.yml` — `70ae5b52b5bbfe2dd7f4a1750ce47c998012eb2c`

## Reuse settled results; examine only four finite questions

1. **No-PHI authority:** Static preview contains three intentionally invented people with no patient IDs/real appointment access, `connect-src 'none'`, no `fetch`, XHR, external resources, cookies, browser persistence or writes. It is entirely independent of `CLINICAL_VISIT_BRIEF_LINKS_ENABLED` and cannot enable real booking↔patient records; no protected endpoint bypass or production setting changed.
2. **Behavioral fidelity for a DEMO, not live Visit Brief:** Appointment click starts booking-only; synthetic clinical context requires explicit independent-verification checkbox; previous visit, plans, pending and checks display only after user opt-in; first-visit example shows genuinely missing history. No automated matching, patient-source assertions or source of truth claims. All persistent UI text clearly denotes synthetic-only nature.
3. **DOM/viewport:** Actual Chromium at 1280/680/390. Close, Escape, switch or reopen purges patient name/context and clinical pending from hidden DOM; no hidden retained previous patient metadata, no horizontal overflow, no Javascript errors. Keyboard affordances remain usable.
4. **Existing Cockpit unaffected:** Diff contains only new standalone static preview, new test/workflow and `cockpit/CURRENT.md` checkpoint. No changes to Home, existing Calendar, protected clinical APIs, Visit Capture or RF/Reports. Existing PR #161/#163/#164 remain DRAFT/HOLD and no live PHI activation is claimed.

## Author CI at exact tested HEAD

- [Preview actual Chromium 39049606750](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39049606750) SUCCESS.
- [Cockpit Home 39049606739](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39049606739) SUCCESS.
- [Cockpit Surgery 39049606807](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39049606807) SUCCESS.
- [Canonical Impact 39049606759](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39049606759) SUCCESS.

Author CI is not the independent R1 verdict.

## Required handback

Return **PASS/BLOCK/UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE/PARTIAL**, `P0:P1:P2`, specific code/test lines, and only any smallest necessary correction. Do not restart #161/#163 R2, A5 DB qualification, staff/secretary service integration, or unrelated components.

STOP READ-ONLY. Any future merge/deploy needs a separate explicit Product Owner release decision and may publish only this static synthetic page, never the protected real booking↔patient feature.
