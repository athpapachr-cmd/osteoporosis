# Cockpit Real Attention Summary — independent affected POST-CODE R2

**Date:** 2026-10-14. **MODE:** ONE fresh independent **READ-ONLY** affected implementation-fidelity R2, then STOP. No code writing, merge, deploy, real patient data, database or external/provider operations.

Repository: `athpapachr-cmd/osteoporosis`. DRAFT PR [#152](https://github.com/athpapachr-cmd/osteoporosis/pull/152), branch `feat/cockpit-attention-source-summary-20261014`. Production/base `3db1a924f77ad8aae14222e4472206b8b18940fa`.

- **Frozen substantive code+test checkpoint:** `4a17d584cd4904dc7239632dd772b0de866ab24a`. Fresh-resolve current PR HEAD, confirm DRAFT/unmerged and compare affected file blobs to the frozen checkpoint; docs-only post-code handback commits may follow it.
- **Owner-approved corrected design blob:** `0f6a9cade4c2055c776b441543f2d800492bd41f`, path `cockpit/COCKPIT_ATTENTION_REAL_SUMMARY_DESIGN_2026-10-14.md`.
- **Previous independent reviews to REUSE:** original PRE-CODE R2 BLOCK (P1-01 RF label / P1-02 surgery stale protected response), then independent delta+affected PRE-CODE R2 **PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0**, original P1-01/P1-02 CLOSED. Reuse prior Home V4 R1, RF outcomes pre/post-code R2, #151 shared nav/Clinical Key/calendar R2, previously closed G3 reports and G5 scope. Do NOT reopen unrelated workstreams.

## Frozen affected source/test blobs at substantive checkpoint

| File | Exact blob |
|---|---|
| `static/cockpit/app.js` | `9e7e7053eb1a861fe8441831f2cb335b287c259f` |
| `static/cockpit/doctor-shell.js` | `acd9340c4d2d67d30ede12426ff8784eaeb3a8d0` |
| `static/cockpit/doctor-shell.css` | `3df87f3b4c0e8241a7c0671c28dce4230587cc69` |
| `test_cockpit_attention_summary.js` | `87abdededccb2554342a873369324ff59851ec8a` |
| `test_cockpit_doctor_shell.js` | `d3ae674cadacd983fe9b4aab40eaef6b11387da6` |
| `test_cockpit_doctor_browser.py` | `d257edc730adcf93fdf8b9ff654c125ceddf0759` |
| `test_cockpit_home.py` | `762ec0a5f869d338865d0823525c65fac3c30adf` |

## Finite affected gates

**R1 — Source identity / bounded code change.** Inspect diff from production/base to exact current PR HEAD; separate documentation-only checkpoints from code. Verify no new database/store/backend/schema/provider writes, no second surgery GET reader, no new RF events/issuance path, no patient linkage, no actual PHI in code/test/screenshot sources. The already existing surgery mutations are not new product authority.

**R2 — Surgery protected queue and date semantics.** Inspect the **same** `loadSurgeryQueue()` in `app.js`: strictly monotonic generation governs success and failure on initial load, `clinical:authenticated` and post-write reloads; only latest request can publish `surgeryRows`, queue DOM, `window.CockpitSurgerySummary` and `cockpit:surgery-counts`. Re-auth invalidates immediately. Latest 401/403/503/network/non-array/invalid-row response clears name/procedure/counts and editor Home projection; older A success after newer B denied response cannot restore PHI; older A failure after newer B success cannot erase it. Synchronous sort and protected existing write paths cannot republish rejected rows. Verify source-confirmed zero, one singleton name+procedure versus 2+ count-only per exact YYYY-MM-DD, Cyprus date-only grouping, up to three sorted upcoming days, undated/overdue/invalid count, no inference of performed/completed from old dates. No surgery ID/name/phone in new routes, persistent browser storage or logs.

**R3 — RF bounded protected read semantics.** Verify **only** the existing read-intent `POST /clinical/clinic-utilities/rf/api/outcomes/search` is added to Home with `{"query":"","scope":"all","page":1}`; no `/api/outcomes/{id}/events` writes or RF issue actions. Compare `list_rf_application_progress` stage semantics: `counts.pending` = current `submitted` awaiting decision, never historical `Υποβλήθηκαν`; denominator = issued attempts, `issued_unknown` not submitted. Approved/scheduled/performed overlap and are not summed. Only source-attested `mine_actionable` is personal follow-up, not colleague auto-assignment. Validate required finite nonnegative integer count fields and literal boolean `truncated`; `truncated=true` withholds global totals. Malformed/missing/401/403/503, auth refresh and stale responses fail closed, not zero. No RF names/IDs in Home DOM/state.

**R4 — Smallest affected cumulative regressions.** Reuse settled Home V4/Visit Capture and #151 evidence. Verify Medical Reports still show truthful no-ledger/no-count state + one-tap existing link, Inbox remains unconnected/no fake unread badge, three recent visits and existing calendar-partial label unchanged, shared Clinical Key/global nav unchanged, selected-patient/Visit Capture Save authority unchanged. One-tap surgery/RF/report access; compact responsive layout at 390/680/1280 px, no overflow. Tests/screenshots use synthetic names only.

**R5 — CI and evidence / proportionality.** Inspect actual CI logs and artifacts at the frozen source/test SHA, not only status badges. Applicable [Cockpit Home tests](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38515549260), [Cockpit surgery queue](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38515549320), [Visit Capture tests](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38515549282) and [Canonical impact guard](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38515549265) reported SUCCESS at substantive checkpoint. Home workflow includes real Chromium mock browser 390/680/1280 and deterministic Node/pytest synthetic source-race checks. Reconfirm from GitHub. The pre-existing Clinical Learning L0 failure is out-of-scope; do not claim repo-wide green. Classify only real affected material defects; no new general design cycle.

## Output and STOP authority

Return `POST-CODE R2 PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, dispositions R1–R5, exact current HEAD + affected frozen blobs, any smallest affected blocker and **STOP**. Author's implementation and checks do NOT constitute an independent review. A PASS would be eligibility for a **separate Product Owner merge/deploy decision**, never automatic deployment or permission for identified patients. No implementation, merge, deploy, real clinical data or external writes in this review.
