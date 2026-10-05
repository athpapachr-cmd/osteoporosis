# PHYSIO R2 post-code closure review — first correction

> **Target:** `fbf070199bc6934f59c4ac44cf984c32a8b35566`
> **Verdict:** **BLOCK / incomplete for declared scope**
> **Method:** one independent read-only delta + affected-cumulative closure of `P1_R2_POSTCODE_REVIEW_RESULT.md`.
> **Date:** 2026-10-05 Asia/Nicosia.

Both original material findings were closed at the reviewed head: selected additional choices remained in their Plan disclosure with non-selecting categorized summaries, and incomplete restriction entry stayed client-side with export disabled until valid.

The reviewer found one new material affected-consumer issue. Additional rows are created before evidence arrives. The refresh compared only the cue code. Walking aid's fallback and actual evidence codes are both `conditional_or_context_dependent`, so its initial `unavailable` visual styling persisted even when active sources were returned. Other conditional additional choices shared that path. The product projection's evidence was correct, but the visible evidence state was stale. A manual-therapy browser assertion exercised a different code transition and did not cover this case.

**Correction disposition:** refresh an additional row when either its evidence code **or** its active-source availability styling differs from the returned evidence; assert that walking aid is visibly available after the response in prototype and protected Cockpit browser tests. This bounded correction requires one independent delta + affected-cumulative closure at the new committed exact head. No PR, merge or deploy follows from this BLOCK.
