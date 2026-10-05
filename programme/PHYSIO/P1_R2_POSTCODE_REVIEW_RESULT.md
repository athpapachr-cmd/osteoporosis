# PHYSIO R2 post-code exact-head review

> **Target:** `c6eb29548a5446289a17a843c650d362b0db3866`
> **Verdict:** **BLOCK / incomplete for declared implementation scope**
> **Method:** one independent read-only exact-head review of the frozen R2-A design, reused R2-B semantics, Cases 1–5 and focused implementation diff.
> **Date:** 2026-10-05 Asia/Nicosia.

The reviewer found two material reachable behaviors at the target head:

1. **Proposed Plan location/hierarchy.** After selection, the inherited plan renderer moved the walking-aid control into an immediately visible selectable main-plan row. Selected adjuncts likewise moved beside core rehabilitation choices without subordinate labeling. The frozen design requires walking aid only in Plan additional options and selected extras to remain subordinate. The Case 3 test had unintentionally asserted the wrong promotion behavior.
2. **Incomplete explicit restriction.** Choosing a restriction type before typing its instruction sent an empty `state_or_value` to the product projection. Validation rejected the payload, clearing the live referral and showing a generic unavailable status. This is a normal input sequence and lacked R2 browser coverage.

The reviewer found the four sections, functional owner, retired selectors, swelling boundary, pattern cue and Continue/Defer, independent CU-1 block, deterministic projection, default plan and CY_GESY wiring otherwise consistent in the finite inspected scope. Three focused server tests passed in the review environment; Playwright was unavailable there. Human completion time and Product Owner output acceptance remain separate real-use evidence limits, not a code defect established by this review.

**Correction disposition:** bounded correction is in progress on the same branch. Keep the walking-aid and other additional controls in their owned Plan disclosure after selection, show non-selecting categorized summaries, and keep export unavailable during an incomplete restriction without sending an invalid draft. Add browser coverage for these paths. Obtain one independent delta + affected-cumulative closure review on the corrected exact head; do not treat this BLOCK as PASS or open the PR before closure.
