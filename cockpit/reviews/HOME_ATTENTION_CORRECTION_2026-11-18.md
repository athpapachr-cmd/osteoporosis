# Day Home attention — review finding correction evidence

Corrects the four findings in [the independent review](HOME_ATTENTION_POSTCODE_R2_BLOCK_2026-11-18.md): bootstrap 401/403 broadcasts the existing unauthenticated event, fencing the actual surgery producer; upcoming surgery rows precede older pending rows; the existing surgery owner reprojects its already authorized queue on date change/return; a named RF carries its exact source id to an existing-owner protected detail read, independent of prior filters and list page. The exact RF request is invalidated on logout or owner navigation.

Focused evidence: `test_cockpit_home.py` 12 passed; `test_cockpit_home_attention.js` passes four-plus-old priority, date-boundary, privacy, partial, 401/403 and stale response cases; `test_cockpit_attention_summary.js` passes actual surgery owner midnight and delayed-success-after-bootstrap-denial integration. The existing Home Chromium suite passes desktop/half/mobile with prior RF filter, exact selected RF outside the fixture first page, return refresh and delayed RF detail after logout; the Office Cases browser test passes. Doctor Shell, iCloud UI and shared navigation Node regressions pass. No production clinical read/write or provider action occurred.

One independent delta plus affected cumulative R2 closure remains required; this is not a release or deployment receipt.
