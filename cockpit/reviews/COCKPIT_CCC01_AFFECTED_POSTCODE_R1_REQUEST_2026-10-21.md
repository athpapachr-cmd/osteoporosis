# CCC-01 — independent affected post-code R1 review request

**Date:** 2026-10-21 · **Mode: FRESH INDEPENDENT READ-ONLY / ONE bounded affected implementation-fidelity review / STOP.**

Repository: `athpapachr-cmd/osteoporosis`
DRAFT PR: [#204](https://github.com/athpapachr-cmd/osteoporosis/pull/204)
Source/base production main at activation: `7ca04734d697eaf573c59437e987b5575e5998c7`
Frozen **runtime+test exact tested HEAD**: `04622288b13d4a4bfafaffd2ef3dc37f344b8a48`. Subsequent edits to CURRENT/review only are NOT changes to this target.
Original Product Owner approval: start CCC-01, exactly one milestone at a time; CCC-02–08 are out-of-scope.

## Purpose

Verify that CCC-01 is an R1 protected read-only UI/projection extending the *existing* ClinicalPendingORM/Clinical Updates read owner, without introducing another task store, incorrect clinical attention, or unauthorized PHI exposures. A successful review is NOT a release approval.

## Affected surface / source

- `clinical_capture_cp1.py`: same protected GET `/clinical/clinical-updates`, added filtering, Greek name search, stable pagination beyond 100, valid dated due/overdue (Asia/Nicosia) before pagination. Preserve legacy default results/history, existing authorization, full registry scopes `real|test`, exact patient_id/pending_id.
- `static/cockpit/clinical-obligations.js`: native all-obligations view, due-only Tasks popover and badge, per-opening protected reads, logout/stale isolation. No storage or newly initiated clinical writes.
- `static/cockpit/index.html`, `doctor-shell.js`, `doctor-shell.css`: full library view, search/group/status/category/due/sort; one-click existing patient-chart Pending popup, correct Home/split widths; surgery totals no longer mislabeled task badges. Preserve Home surgeries/RF/Inbox/Visit Brief/Calendar.
- `static/cockpit/patient-chart.js`: invalid existing review_date must be safe text, not crash patient chart (actual synthetic Chromium finding).
- `test_cockpit_ccc01.py`, `test_cockpit_cp2.py`, `test_cockpit_doctor_browser.py`, `test_cockpit_doctor_shell.js`: scoped and inherited synthetic regressions.
- `cockpit/CCC_01_OBLIGATIONS_ATTENTION_2026-10-21.md`, `cockpit/CCC_CLINICAL_CONTINUITY_PLAN_2026-10-21.md`: frozen Product Owner rules/continuation.

## One finite review: 5 questions

1. **Clinical salience correctness:** only active tasks with **valid stored** `review_date<=current Cyprus day` appear as due/overdue in the icon. No arbitrary inference from `trigger_text`, waiting for labs/call, next visit, missing/invalid date, and no misclassification of future. All remain in protected full list; no actual task completion or clinical status writes.
2. **Source/pagination/identity:** protected same-source exact `pending_id` and patient_id, no namesake merge; full list can retrieve >100 via stable offset and filtered total/next offset with `active_only` before slicing. Query, status/category/due/sort semantics and legacy consumer response preserved. No unbounded provider calls or public routes. Schema unchanged.
3. **Read/auth/privacy:** a denied, stale, old-success, logout or synthetic-161 shell must not leak/restore PHI; the dedicated old synthetic route MUST NOT call real `/clinical/clinical-updates`. `Clinical Inbox` remains independent, does not trigger Pending reads on opening; 'unconnected email/call' truthful. All user-supplied text inserted as textContent.
4. **Workflow/UI:** mini Tasks popup contains only up to four current dated items + full view link; full screen works within existing Cockpit (not extra browser tab), desktop 1280/half 680/mobile 390, filters/search/page controls and exact one-click pre-existing Pending dialog; unsaved drafts and other screens preserved. Badge never reflects surgery total (Home surgery summary retained).
5. **Evidence reuse / finite regression:** Compare exact tested runtime/test source against GitHub CI. Frozen runtime+test `04622288b13d4a4bfafaffd2ef3dc37f344b8a48`; subsequent docs-only checkpoint `60328694c332b312eb295c3d6c9116fc2456872d` tested against the *same* runtime/test blobs. CCC-01 [39677271174](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39677271174) SUCCESS; Home [39677271394](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39677271394) SUCCESS; Visit Brief [39677271305](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39677271305) SUCCESS; Visit Capture [39677271225](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39677271225) SUCCESS; Pending Synthetic [39677271164](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39677271164) SUCCESS; Canonical Impact [39677271285](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39677271285) SUCCESS. All six success statuses independently inspectable on GitHub; do not rerun unchanged checks. Do not rerun unchanged tests merely to repeat evidence.

Known earlier failed runs were authored code/test corrections (invalid-date chart crash, first-page-only assertion, legacy surgery badge oracle, synthetic-161 unauthorized normal source read, Inbox read oracle); final targeted correction and inherited gates replace them. Do not reopen unrelated clinical architecture.

## Boundaries / output

Return **PASS/BLOCK/UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE/PARTIAL**, P0:P1:P2 and directly grounded evidence. If BLOCK, name reachable bug, severity and smallest correction. STOP, no code changes, merge, deploy, new database, provider/email/phone calls or patient reads/writes; no CCC-02+ implementation. Independent PASS is a *review outcome*, not clinician acceptance or live release authority.
