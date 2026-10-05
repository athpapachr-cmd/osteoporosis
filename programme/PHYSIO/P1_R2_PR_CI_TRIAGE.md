# PHYSIO R2 — draft PR #135 CI triage

> 2026-10-05 Asia/Nicosia. PR head examined: `7d1ba92f2046d221fb64bee9db7d76ebba2471be`. This is a candidate correction record, pending independent delta review and exact-head CI. Merge/deploy HOLD.

The canonical-impact, evidence, template, interaction, CU-1, jurisdiction and V5 gates passed at the examined head. Three PHYSIO jobs failed:

1. The isolated prototype job ran the protected Cockpit browser test, which imports `uvicorn` outside that job's minimal dependency set. The prototype's own six-case browser suite passed. The correction removes the protected test from **only** that isolated job; V4, V5, Cockpit and jurisdiction jobs retain protected production browser coverage with the full dependency set.
2. The Cockpit and V4 jobs failed the same rapid Defer → Continue interaction: while the Defer request was pending, the Continue control still looked actionable, but its click was ignored by the stale-response guard. The correction disables both decision controls synchronously until the refreshed response renders, then re-enables them with the persisted choice. Both browser suites now wait for the Defer decision to be visibly confirmed before proceeding to Continue. The review gate remains fail closed while pending.

The four Clinical Learning L0/L1/L1B/L1C jobs also failed their own scope guards because the authorized root `CURRENT_OPERATIONAL.md` checkpoint triggered those workflows. Their test steps passed. Those workflows belong to another writer and are not changed in this PHYSIO slice. Their failures remain an external CI ownership issue; they do not grant merge/release authority.

Local correction checks: JavaScript syntax and `git diff --check` pass; PHYSIO prototype browser **6/6** and protected Cockpit browser **4/4** pass. Independent exact-head delta/fidelity closure and new PR CI are required before calling the correction complete.
