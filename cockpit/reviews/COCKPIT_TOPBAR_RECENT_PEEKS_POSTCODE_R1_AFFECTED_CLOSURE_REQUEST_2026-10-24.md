# Cockpit Home — Top-right recent peeks · affected P1/P2 R1 closure request

**Date:** 2026-10-24 Asia/Nicosia. **MODE:** fresh independent, strictly **READ-ONLY**, **ONE bounded affected R1 closure** after the material findings in the original exact-head post-code R1. STOP after a verdict. No code changes, commits, merge, deploy, backend/provider calls, production writes, or real patient records.

- **Repository:** `athpapachr-cmd/osteoporosis`; **DRAFT PR:** [#209](https://github.com/athpapachr-cmd/osteoporosis/pull/209); branch `fix/cockpit-topbar-peeks-20261024`.
- **Verified main at correction start:** `bccaf10aead6394f545ec44f735aa12abcb37fed`.
- **Original tested R1 implementation:** `064fb2b31cbfe6292be2b809ee7940e94e7b03f8`; original PR doc head `4aa4320785cccf8d3b7906d46fdb1e1357b73e41`.
- **Corrected, CI-tested code+test HEAD:** `8a276a4cb6fa73525c2a1e3a1d8e91645585ca62`.
- **Original independent R1:** **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:1:1**. P1 = `clinical-inbox.js` `load(focusItemId)` shadowed the `select(id)` function with a DOM `const select`, so clicking a recent synthetic Inbox item called an element as a function, fell into the unavailable catch, and did not focus its detail. P2 = `obligations-view.js` modified for deep-focus navigation without changing its HTML asset cache version.
- **Original review request:** [Exact-head R1](COCKPIT_TOPBAR_RECENT_PEEKS_POSTCODE_R1_REQUEST_2026-10-24.md). Reuse unaffected original PASS and existing tested CI; do not start a new general UX, clinical-identity, or Inbox-provider design review.

## Exact bounded correction — only three file paths

1. `static/cockpit/clinical-inbox.js` blob `abeeb265ed787813a842b799b5e2ad81161ef65b`: rename the local account-selector DOM variable in `load()` to `accountSelect`. The untouched `select(focusItemId)` must now resolve to the actual protected synthetic-detail selection function.
2. `test_cockpit_synthetic_161.py` blob `9dde9150d59226ac88e5a3e82d96a7c8934504c3`: real Chromium 1280/680/390, click the actual “Συνθετική γνωμάτευση MRI” recent-row button, assert detail visible, correct MRI item `SYN-ATT-MRI` selected (`aria-pressed=true`), no browser page errors, cleared state after backing out, and explicit full Inbox route still works. Reuses isolated synthetic fixture; no real PHI or external provider.
3. `static/cockpit/index.html` blob `5420e3ec327f7ce1610be199971e7cf0722a2f00`: change the `obligations-view.js` URL query version from `pending-metadata-20261022` to `topbar-peeks-20261024`.

Unaffected `static/cockpit/doctor-shell.js` blob `aad84dcd9d0562aaa8e0c4fc91a4d35048eda819` and `static/cockpit/obligations-view.js` blob `4fc5016f203b5a818a8fc02453116cbd0525eeb6` are unchanged from original tested implementation.

## Author CI evidence on corrected SHA `8a276a4`

- [Home 40073786675](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40073786675) **SUCCESS**.
- [Visit Brief / synthetic #161 40073786695](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40073786695) **SUCCESS**, **51 + 14 PASS**, including the new three-width actual-browser synthetic-item-focus assertions.
- [Visit Capture / CP2 40073786661](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40073786661) **SUCCESS**, **33 + 37 PASS**.
- [Canonical impact 40073786602](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40073786602) **SUCCESS**.
- **Transparency:** An earlier parallel Visit Capture run [40073781877](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40073781877) associated with the same SHA **FAILED** on a Playwright 8-second timeout in the unrelated appointment-chart/patient-entry test `test_browser_single_login_booking_link_latest_and_independent_session`; the later complete [40073786661](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40073786661) on the same SHA **PASSED 37/37**. Inspect and explicitly disposition this isolated conflicting CI observation; do not conceal it or assume it proves a code regression.

## Only five affected closure questions

1. **P1 root-cause closure:** Confirm the `select` name-shadowing is removed and the exact recent-item click, current protected read, selected-item details, and normal full Inbox path behave; ensure the regression would fail on pre-fix code and passes on corrected SHA.
2. **P2 cache closure:** Confirm `obligations-view.js` changed query version in `index.html` and its existing focusPending implementation remains byte-identical; no stale bundle risk from this prior URL.
3. **Retained isolation and auth:** Confirm synthetic #161-only routing, no real-mail fabricated count/item, correctly cancelled stale async reads after close/log-out/principal change, and no automatic patient linking/writes.
4. **Diff/scope:** Independently verify GitHub compare original R1 doc head → corrected code+test HEAD includes exactly the three changes described, no release/identity/backend/writer changes; if a later PR HEAD is docs-only, compare runtime/test blobs.
5. **CI evidence:** Confirm exact SHA and log outcomes above, including explicit disposition of the earlier isolated timeout and later successful repeat. Reuse unchanged first-pass UX/browser 1280/680/390 and protected obligations tests.

**Required independent outcome:** `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, inspected exact PR/code SHA and file blob evidence, five bounded dispositions, and if BLOCK only the smallest material correction. **STOP.** The author correction and CI success are **not** an independent R1 PASS. PR #209 remains **DRAFT, unmerged**, with **no merge/deploy authorization**. Clinical Inbox verified-identity redesign is a separate R2 and is explicitly out of scope.
