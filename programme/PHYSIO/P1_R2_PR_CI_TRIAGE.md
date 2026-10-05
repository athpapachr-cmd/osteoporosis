# PHYSIO R2 — draft PR #135 CI triage

> 2026-10-05 Asia/Nicosia. PR head examined: `7d1ba92f2046d221fb64bee9db7d76ebba2471be`. This is a candidate correction record, pending independent delta review and exact-head CI. Merge/deploy HOLD.

The canonical-impact, evidence, template, interaction, CU-1, jurisdiction and V5 gates passed at the examined head. Three PHYSIO jobs failed:

1. The isolated prototype job ran the protected Cockpit browser test, which imports `uvicorn` outside that job's minimal dependency set. The prototype's own six-case browser suite passed. The correction removes the protected test from **only** that isolated job; V4, V5, Cockpit and jurisdiction jobs retain protected production browser coverage with the full dependency set.
2. The Cockpit and V4 jobs failed the same rapid Defer → Continue interaction: while the Defer request was pending, the Continue control still looked actionable, but its click was ignored by the stale-response guard. The correction disables both decision controls synchronously until the refreshed response renders, then re-enables them with the persisted choice. Both browser suites now wait for the Defer decision to be visibly confirmed before proceeding to Continue. The review gate remains fail closed while pending.

The four Clinical Learning L0/L1/L1B/L1C jobs also failed their own scope guards because the authorized root `CURRENT_OPERATIONAL.md` checkpoint triggered those workflows. Their test steps passed. Those workflows belong to another writer and are not changed in this PHYSIO slice. Their failures remain an external CI ownership issue; they do not grant merge/release authority.

Local correction checks: JavaScript syntax and `git diff --check` pass; PHYSIO prototype browser **6/6** and protected Cockpit browser **4/4** pass. Independent exact-head delta/fidelity closure and new PR CI are required before calling the correction complete.

## Correction closure at `fa9b1ff4f0f070cde3c78fb30ebd2a7a5a0a2dd3`

Independent read-only bounded delta + affected-cumulative fidelity review: **PASS / no material finding**. The reviewer verified synchronous disablement, pending export block, failed-refresh block, refreshed decision state, preserved Cases 1–5, retained protected coverage in full-dependency jobs, and no change to projection, evidence, default plan, CY_GESY or shared CU-1. No further review cycle is required for this exact correction.

Exact-head PR CI: canonical impact, evidence, template, interaction, CU-1 focused, prototype, Cockpit, V4, V5 and jurisdiction gates all **SUCCESS**. Clinical Learning L0/L1/L1B/L1C remain **FAILURE** on their own scope guards; their ownership is outside this PHYSIO writer lock. Merge/deploy remain HOLD. A subsequent docs-only checkpoint must be assessed at its own head.
