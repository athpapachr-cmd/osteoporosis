# Cockpit home experience release

- Product Owner requested improvements to all four selectable surfaces, starting with Day; calendar shows all available appointments with an osteoporosis filter.
- Product PR #288 merged as `58cb13d51215bb9b87b040c3507585119dfb910c`; corrected candidate `84644a1` independently reviewed PASS P0:P1:P2 0:0:0, and all final PR checks passed.
- Render native deploy `dep-dbrb83u1egvs73fk4eog` went LIVE on that exact merge source at 2026-11-13 06:55:49Z.
- Day adds protected daily appointment list, direct appointment opening and shortcuts. Calendar defaults to all source-covered appointments with osteoporosis filter, Cyprus boundaries, manual classification and auth-loss clearing. Inbox counts explicitly cover the loaded page only. Office Cases uses existing surgery summary and clearer section labels.
- No-PHI production smoke: seven served assets byte-identical; actual Chromium at 1280/390 checked Day, calendar filters and Inbox/Office routes, with all clinical API calls intercepted and synthetic responses. No real patient/provider or clinical writes were performed.
- Render ERROR and HTTP 500/502/503 queries from live through 07:02Z returned no entries. Actual authenticated clinical-source coverage, real appointment/patient interactions and clinician usability were not verified in this smoke.
- Final documentation publication uses `[skip render]`.
