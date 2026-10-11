# Cockpit Home V4 — A2/P1 bounded correction; one independent delta + affected R1 closure

Date: 2026-10-11. **Mode: fresh separate independent READ-ONLY, exactly one closure then STOP.**
Repository: `athpapachr-cmd/osteoporosis`. Draft [PR #145](https://github.com/athpapachr-cmd/osteoporosis/pull/145), branch `feat/cockpit-apple-inspired-home-2026-10-11`.
Production base at correction: `main=7028c1459ff51c0d6edd0df74298f742f77ccd3f`.

## Original review and fixed target

Original affected R1 request: `cockpit/reviews/CLINICIAN_HOME_V4_AFFECTED_R1_REQUEST_2026-10-11.md`.
Original source/test/CI head: `2d2e97f3b6dff0ea56d7a8b53605266df2b42ef0`.
Original independent review disposition: **BLOCK, P0:P1:P2=0:1:0**, with sole A2/P1 finding: Home text misrepresented relevant-category clinical day appointments as the *full* Reception schedule. A1/A3/A4 were supported; reuse that source/evidence; do not re-review untouched areas.

**Exact corrected UI+regression-test head: `0ca19ff8ffd738f50dc1b45b2ed853d352ba6420`.**
Frozen corrected file blobs:

- `static/cockpit/index.html` = `0903e18a865beaff0399f4f5d9df2d32dce08d6c`
- `test_cockpit_home.py` = `e28c5994e5e492fc137676f61936e1b492182add`

Only affected two source/test files changed after correction writer claim. Post-head canonical/current/review documentation commits are **not** new UI/test scope; verify exact blobs remain unchanged.

## Finite closure questions

**C1 — Close the P1 precisely.** Inspect Home `#visitHomeNote` against original wording and the contract. It must NOT claim that the date popover contains the complete Reception schedule; it should explain partial-day scope and how to reach the full Reception schedule. Check the existing popover still labels category coverage and links to Reception.

**C2 — No new behavior or authority.** The correction must be display copy only, with a focused regression assertion. Verify no change in calendar fetching/timezone, patient identity, surgery, clinical Save, network/provider access or routes; reuse original A1/A3/A4 and their existing CI evidence. No patient data or new runtime writes.

**C3 — Finite evidence.** Confirm exact corrected blobs and source/test-only correction diff. Check the applicable Cockpit Home CI at the corrected head (including synthetic Chromium if available), Visit Capture and canonical-impact results when complete. A red unrelated L0 design-only scope assertion is not a Cockpit defect; do not claim all repository checks green.

Return exactly **PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0** if C1–C3 are supported; else bounded **BLOCK** or **UNKNOWN** with smallest unresolved issue. Do not infer release authorization. **STOP**: no implementation, new design/owner/schema, independent review chain, merge, deploy or identifiable clinical data.
