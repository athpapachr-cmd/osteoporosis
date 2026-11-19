# Quiet Day Home release — 2026-11-19

- User explicitly authorized publication and installation after final checks. PR #309 merge `4ecd859ec6db1f7dbcc3276b5a90d4446ec0aa9d`; final head `ea6be280ba467ec9c2d96c95d4fca32146f18ca6`.
- Independent R1 affected+cumulative closure PASS at substantive test-corrected `a0014c46f8b00ef267d47596201c31ad4a7738d3`; 15/15 final-head CI PASS. Two initial Ubuntu package download failures passed on retry. Synthetic browser smoke covered desktop, half-width and mobile.
- Native auto-deploy ignored the merge due to skip markers in inherited squash body, confirmed by Render event `evt-dbvek5h42hec73bddqmg`. One authorized standard deployment `dep-dbvekrbrjlhs73bnr370` reached LIVE at 12:25:52Z on the exact merge; no duplicate or settings change.
- No-PHI verification: six served Cockpit index/JS/CSS assets match the reviewed candidate; health 200; unauthenticated agenda 401; error-level and HTTP 500/502/503 logs empty over the release window through 12:26:57Z. No actual patient/provider write or authenticated clinician usability evaluation was performed.
