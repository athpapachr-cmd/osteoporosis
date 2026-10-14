# Native RF — Dedicated Tracking + Weekly Calendar v1: fresh independent POST-CODE R2

**2026-10-14. MODE: ONE fresh independent READ-ONLY implementation-fidelity / affected POST-CODE R2, then STOP.**

Do not modify source, tests, PR, database/schema, current clinical records/PHI, Cal.com/Reception/provider, branch, merge or deploy. Reviewer does not execute production operations. Synthetic tests/source inspection only.

## Frozen identities and review authority

- Repository: `athpapachr-cmd/osteoporosis`; implementation **DRAFT PR #155**, branch `feat/rf-tracking-weekly-calendar-20261014`. **Exact frozen substantive implementation+test HEAD: `c291717b934f54d6f63898cd4d5b41090f764fa8`.** Fresh-resolve the current PR branch HEAD and source blob identities; any later commits must be documentation/checkpoint-only, otherwise STOP and request fresh evidence.
- Design baseline: separate design DRAFT **PR #154**, original PRE-CODE R2 P1-02 focused **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0**, approved owning design `clinic_utilities/rf/RF_TRACKING_WEEKLY_CALENDAR_V1_DESIGN_2026-10-14.md` **blob `b815cfb47cac310ab180df2b945675f08b5043dd`**, design HEAD `1ca2bc1e2a6dc9d169be536080e9ff3d2a72b228`.
- Frozen design activation/base `main=244a3264f119252b1bbfc3562a00648cfdf113bd`. Implementation branch starts at approved design commit #154. Review **affected runtime/test delta**, not historical design-only files. Reuse closed Native RF #149/#150 and Clinical Key #151, Home #152, RF Home V4 and previous G1–G7. **Parallel Medical Reports #153 remains separate.** No unreviewed merge/deploy or real data.
- **Fresh CI at exact substantive HEAD**: [RF Tracking synthetic run 38584144204](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38584144204) **SUCCESS** (6 synthetic Python tests including >2,000 matching/10,000 scanned, inherited 50 native RF cases, RF browser JS, Home JS, actual Chromium at 390/680/1280 and diff hygiene); [Cockpit Home 38584144247](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38584144247) **SUCCESS**; [Canonical Impact 38584144482](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38584144482) **SUCCESS**. Unrelated existing Clinical Learning scope gates may fail due to their adjacent-owner guard; do not claim all repository workflows green. Source or CI identity drift must be reconciled.

## Exact substantive files/blobs (all at `c291717b934f54d6f63898cd4d5b41090f764fa8`)

| Path | Expected Git blob |
| --- | --- |
| `clinic_utilities/rf/api.py` | `8c8137ef3f34eacff8bb7eb0301d1e26f76cb944` |
| `clinic_utilities/rf/persistence.py` | `71101a128511160d9674ef381d17fa764e09db90` |
| `clinic_utilities/rf/tracking.py` | `1bf6cf3853ff387e833e763a0a90a4da0ad9c6f0` |
| `static/clinic-utilities/rf/index.html` | `cb32a94feb11158a769b5e8dda9f28303ad6d182` |
| `static/clinic-utilities/rf/styles.css` | `8c8bc6fbdec3ba90145f9e0d3a8f9c67a00f868e` |
| `static/clinic-utilities/rf/app.js` | `d1aba511e56128d904265af3f06d66d8016801ef` |
| `static/cockpit/doctor-shell.js` | `82900f1e4bc4d700df11fbf654bade4a7ee34d11` |
| `test_rf_tracking_weekly.py` | `1903da786edc6335da1fbac726ab2132351efb0d` |
| `test_rf_tracking_browser.py` | `a9aa7a45feb7f1e061a9c05666ff440c77393e17` |
| `test_rf_v2_native.py` | `6aca394b30e7aad8ba16ee2bc50989c8cfc1085d` |
| `test_rf_v2_ui.js` | `bfca82ada87ac9a05ec0eecd6d98f60596907250` |
| `test_cockpit_attention_summary.js` | `067c22425f9325c65984a8590cc46ee6384deb1a` |
| `.github/workflows/rf-tracking-weekly-v1-tests.yml` | `0991981d43b13fc13f6de1abcedd4ad335b07bd9` |

## Seven affected gates — verify against *code*, not author claims

**G1 — Owner, HEAD and blast radius.** PR DRAFT/unmerged, frozen design/main reconciliation, changed-file review, one protected existing RF router/data owner and common same-document controller. No parallel #153 files, second RF/patient register, new outcome writer, live provider service operations or actual PHI. Idempotent date indexes are the only schema-level candidate, no destructive migration.

**G2 — Tracking list completeness and lifecycle.** Existing `POST /api/outcomes/search` unchanged for legacy Home `query/scope/page` reader/counters. Opt-in `mode=tracking` uses signed, filter/expiry/high-water-bound cursor exclusively in protected POST body; deterministic keyset, forward progress even on zero-match scan, unbounded total traversal via bounded 200-row batches across both previous 2,000-visible and 10,000-scan ceilings. Greek accented name/identity matching, explicit `scope=all` default and `mine_to_follow` only opt-in; issued != submitted; approval/schedule not disjoint totals; honest `has_more/scan_complete/truncated`, no fabricated full-snapshot totals.

**G3 — Dedicated page and protected correction handoff (critical).** One `/rf/tracking` protected view and `/rf` creator served from same HTML/JS shell, no full reload for selected-case correction, no identifier in document URL/history/storage or a copied outcome/action controller. Existing outcome actions still versioned and audited. New protected read-only `POST /api/reuse/rejected-correction` must load the exact immutable issued attempt and locked mutable progress in ONE transaction; `stage=decision=rejected` and exact `expected_revision` required; return same selected ID/revision/allowed serializer fields or 404/409/422 without PHI. Client checks ID/revision/view/generation/auth/dirty form before any mutation. At issue time `/api/create` requires `correction_source_expected_revision`, revalidates under locked progress in existing PDF transaction before new issued attempt/PDF commit. No prior PDF mutation. Generic legacy GET still works but cannot itself arm correction. On A→B, changed revision, 401/403, saved draft, reload/new tab/history/back and other errors, no cross-patient or partial prefill, automatic correction restore or silent dirty form loss. Explicit issuer/PDF/JPEG reconfirmation retained.

**G4 — RF date-only weekly calendar.** RF-owned `POST /api/outcomes/calendar` reads `scheduled_date/performed_date` from existing progress, Monday–Sunday local date with Nicosia/DST correctness, bounded deterministic continuation with completeness/truncation honesty, >1 RF/day, separate scheduled/performed facts, undated RF reachable from list, no inferred actual treatment from date passage. Actual nullable progress fields are *not* preindexed in frozen main; candidate adds two safe/idempotent indexes on existing database tables with non-destructive DDL and synthetic coverage. No osteoporosis `/clinical/calendar/appointments` category or booking writer change.

**G5 — Source-verified Reception booking clock strictly OFF.** No current RF attempt↔Cal UID trusted association. This implementation MUST NOT display an actual Reception booking hour or assert link proof, auto-match names/phones/dates, call Cal.com provider, or modify `submitted/approved/scheduled/performed` based on a booking. Date-only RF week works without Reception. Any later optional clock requires separate bounded authorization/implementation and fresh verification of UID, freshness, coverage, same RF date and absolute expiry (not part of this release).

**G6 — Auth/races/clinical privacy.** Shared Clinical Key protection for both views/endpoints, no invented per-doctor IAM; per-stream generations list/detail/week/clock, shared auth epoch on re-login/401/403 and clearing of protected patient data, no stale A success/error overriding B or misleading zeros. One-tap detail action remains attached to exact selected ID/revision, no PHI in public Home, URLs, history, storage or telemetry. RF 503/network and source partial failures visibly fail closed; UI works at Chromium 390/680/1280 with keyboard/accessibility, dirty saved drafts and in-document back/reload.

**G7 — Evidence/governance.** Confirm successful exact-head synthetic CI, inherited immutable PDF/doctor/issuer/outcome/draft tests, Home RF destination with old count semantics and actual browser behavior, protected error paths and no real clinical data/service access; no unrelated source edits. Canonical checkpoints indicate design #154 approved, #155 candidate POST-CODE HOLD. The PR author does not self-certify PASS; new material affected findings only, no reopening settled unrelated reviews.

## Required reviewer result

Return **IMPLEMENTATION-FIDELITY / AFFECTED POST-CODE R2 PASS | BLOCK | UNKNOWN**, **COVERAGE=COMPLETE_FOR_DECLARED_SCOPE | PARTIAL**, **P0:P1:P2=x:y:z**, current PR HEAD vs frozen substantive HEAD, each source/test blob and workflow status, explicit G1–G7 dispositions with line/source evidence, smallest bounded correction if blocked, then **STOP**.

A PASS authorizes *at most* a later distinct Product Owner-approved release path. Do not merge, deploy or use real patient data in this review. Any change after frozen substantive code/test HEAD requires fresh affected verification.
