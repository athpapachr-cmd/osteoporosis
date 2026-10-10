# Independent focused R2 pre-code closure — P1-01 Calendar snapshot coverage

Date: 2026-10-10. **MODE: fresh separate READ-ONLY P1-01 correction closure; ONE finite review, then STOP.** Original full independent R2 disposition: **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:0** on checkpoint `36349d826d9d95497b306bc592a71b5995a25e51`; R2–R4 already PASS, not re-reviewed. Stage A R1 PASS preserved; PR #140 A4 stays separate.

Repository: `athpapachr-cmd/osteoporosis`, existing draft/unmerged [PR #142](https://github.com/athpapachr-cmd/osteoporosis/pull/142). Product main `1febcfa27096df20f6f4bdeb3ccee15158634fc3`; original frozen design blob `9e107112020e2d98d542867c64fbb08d47e278dd` (DO NOT mutate), original R2 request blob `fdd02f6e3eb8d2080b6f0d60c03853b1f5cdea60`.
Exact corrected R1 design: `cockpit/UNIFIED_VISIT_V3_READ_COVERAGE_P1_01_DELTA_2026-10-10.md` (verify exact blob in current GitHub branch).
Independent original verdict received: `cockpit/reviews/UNIFIED_VISIT_V3_READ_PRECODE_R2_BLOCK_RECEIPT_2026-10-10.md`.

## Only review questions

C1. Does the delta close original P1-01, by requiring `ActualSchedule.coverage_start <= today_local <= ActualSchedule.coverage_end` (both non-null, inclusive), with `today_local` computed in `Asia/Nicosia`, **in addition** to existing 5-minute `fetched_at` freshness, before any protected `/cockpit-context` success/empty/upcoming_today projection?
C2. Does incomplete/missing coverage preserve existing **503 unavailable** and prevent false empty today or advancement of last-known-successful source time, while allowing `upcoming_today=[]` for an actually covered empty day? Are two finite contrasting synthetic future-code oracles specified?
C3. Does this use existing source/projection semantics only, retaining current cross-day `next`, identity boundary, no new provider read/writer/store, and previously PASS R2–R4 and Stage A? Are there any **new material risks directly caused by this R1 correction**?

Evidence: `clinical_calendar.py` `ActualSchedule`, existing `/cockpit-context` and `/appointments` coverage guard (~lines 376–449 and 495–501 at original checkpoint); immutable original R2 design/request; exact new R1 delta and received verdict. No backend implementation, runtime tests, CI reruns, deploy, provider data, real patients, merging or broader architecture review. Pre-code closure is source/contract reasoning only.

Return **P1-01 FOCUSED R2 PRE-CODE CLOSURE PASS | BLOCK | UNKNOWN**, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, dispositions C1–C3, P0:P1:P2 (new/remaining within corrected scope), specific source evidence and **STOP**. PASS would authorize only the already declared two protected backend read additions subject to writer/CI/subsequent post-code review, not a release.
