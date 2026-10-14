# Native RF — Tracking + Weekly Calendar v1: fresh independent affected POST-CODE R2 request

**2026-10-14 · MODE: FRESH INDEPENDENT READ-ONLY, exact-source implementation-fidelity + affected cumulative review, then STOP.**

## Source identity
- Repository: `athpapachr-cmd/osteoporosis`
- **Implementation DRAFT PR #156**: `feature/rf-tracking-calendar-v1-implementation-20261014`, targeted to **design branch** `design/rf-tracking-weekly-calendar-20261014` (not main).
- Frozen design base: `1ca2bc1e2a6dc9d169be536080e9ff3d2a72b228`, owning design blob `b815cfb47cac310ab180df2b945675f08b5043dd` in DRAFT PR #154; pre-code independent P1-02-focused PASS/0:0:0 provided by Product Owner.
- Substantive implementation+test+migration candidate checkpoint: `985c5f1230eef8c40d2c8208a5af44dd4b3eee19`. **Fresh-resolve current PR head/base/main, verify changed-file manifest and blob identities**, and distinguish any subsequent documentation-only checkpoint from substantive tested source.
- **No merge, deploy, real clinical data, external provider or database activity.** This is an implementation candidate, not author self-certification.

## Frozen affected source/test blobs (at substantive checkpoint)
| Path | Blob |
|---|---|
| `clinic_utilities/rf/api.py` | `004c05fa87d65792910b08e155fb0116f1b0667c` |
| `clinic_utilities/rf/persistence.py` | `00bb1ca3c9e2a63a03663f0a64bb61d040d3154d` |
| `static/clinic-utilities/rf/app.js` | `e054fe521abf62fcccd23b488afd50f829ce6cf5` |
| `static/clinic-utilities/rf/index.html` | `9c99dbfe3e50d176c9e09fcefafba0f4ccd575be` |
| `static/clinic-utilities/rf/styles.css` | `35dc364516a756a28131e156b571d04df62c1568` |
| `static/cockpit/doctor-shell.js` | `82900f1e4bc4d700df11fbf654bade4a7ee34d11` |
| `test_rf_v2_native.py` | `1f2f37ce5950e7ae9c9ecf287c77075b0440fc1c` |
| `test_rf_v2_ui.js` | `263166c52594cb3a8e995314d903a77ec7f09828` |
| `test_cockpit_attention_summary.js` | `067c22425f9325c65984a8590cc46ee6384deb1a` |
| `.github/workflows/rf-v2-native-tests.yml` | `67ed613b88fa211142f8968c87c166bced4d2c20` |
| `clinic_utilities/rf/migrations/2026-10-14-tracking-calendar-progress-date-indexes.sql` | `b4ec4a17bcb2c4630c4e8caba2320ada8461275a` |

## Bounded mandatory checks

**F1/F2 — Source and prefill:** frozen main generic reuse GET did not check rejection/revision. New Clinical-Key-protected `POST /api/reuse/rejected-correction` must load exactly immutable selected RF plus locked mutable progress in one transaction, require stage+decision rejected and matching expected revision, serialize same selected row, return ID/revision with no-store; wrong/missing/stale/conflicting request => 404/409/422, no PHI leaks. Existing GET/reuse unchanged.

**F3/F4 — Handoff and issue:** true same-document Tracking/creator view, fixed non-sensitive document URLs; correction selection only volatile, A/B and auth/dirty guards prevent cross-patient form mutation, no saved-draft auto-correction, re-confirm issuer and imaging. Require `correction_source_expected_revision` on /api/create and recheck under progress lock in **existing** issuance transaction before PDF/new attempt, fail closed without partial creation. Preserve immutable prior PDF and non-correction A1/A2.

**G1/G2 — existing RF ownership and complete Tracking ledger:** one existing issued-attempt/progress/audit backend and outcome event writer, default all (not mine), explicit follow-up ownership, issued != submitted, filtered honest totals. Keyset continuation must reach **>2,000 matched and >10,000 scanned** attempts without silent truncation, signed filter-bound expiring POST-body-only cursor, explicit incomplete/unknown when necessary, no changes to Home #152 legacy counter semantics.

**G3 — UX/routes:** protected separate /rf/tracking path with list, selected outcome actions, week tab, one-tap builder; preserve permanent Home/Clinical Key/shortcuts. Test view switching/back/reload, retained unsaved creator state, no identifiers in browser history.

**G4 — date-only calendar:** RF existing scheduled/performed fields as local `YYYY-MM-DD`, Monday–Sunday Nicosia/DST, separate scheduled and performed facts, never booking time; bounded week read with honest 500-row truncation/no phantom completeness. Existing date columns are nullable and unindexed; check explicit PostgreSQL `CREATE INDEX CONCURRENTLY IF NOT EXISTS` migration artifact, **not executed**, and assess database query safety before release.

**G5 — optional Reception time:** current sources have **no certified RF↔Cal UID link**. Implementation intentionally displays **no Reception clock times**, with `time_verified=false`, no auto name/date linkage and no provider writer. Assess whether first-phase date-only release is safe and consistent; do **not** self-credit unimplemented optional link features.

**G6 — auth/races:** existing Clinical Key, independent list/selection/week request invalidation; 401/403/503, concurrent A/B selections and authorization epoch, error/no-data honest, no stale RF names or times; no PHI Home, local/session storage or navigation URLs. Validate both success and late-error paths and dirty creator.

**G7 — synthetic and regression evidence:** independently inspect exact-source CI for RF native/UI and >2000/>10000 synthetic traversal, Home, CU-1 and Canonical Impact. Need browser evidence for 390/680/1280 widths and keyboard/focus as applicable; **do not assume the Node DOM shim alone proves real Chromium**. Distinguish unrelated pre-existing Clinical Learning gates from actual RF failures. Existing #149/#150/#151/#152 previously settled reviews reused; Medical Reports #153 separate, no cross-workstream changes.

## Required disposition
Return **IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN**, `COVERAGE=COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2=x:y:z`, exact PR/base/head, frozen source/test/migration blobs, CI outcomes, explicit F1–F4, G1–G7, any smallest affected correction, and **STOP**. Do not implement, mutate data, merge or deploy. Even PASS only permits separate Product Owner release consideration. 
