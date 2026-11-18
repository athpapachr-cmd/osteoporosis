# Day Home attention — concurrent refresh correction

Corrects the [concurrent refresh finding](HOME_ATTENTION_REFRESH_RACE_BLOCK_2026-11-18.md): a Home return/visible stale refresh starts a new read via the existing surgery owner and supersedes any older generation. Initial auth/bootstrap does not duplicate the owner's already-started read. The same generation and auth epoch fences prevent older success or denial from overriding the new source; unsubmitted surgery editor data remains intact.

`test_cockpit_attention_summary.js` reproduces a held old GET, source change, new Home refresh and late old success, asserting the new item remains and the draft survives. `test_cockpit_home.py` and Office Cases browser: 13 pass. Home attention synthetic test and real Chromium desktop/half/mobile Home flows pass. No production data, provider or clinical write was used. One affected independent closure remains.
