# Cockpit PR #207 — R2 source-verified in-list autonomous obligations, category/date

**Date:** 2026-10-22 (Asia/Nicosia). **MODE:** ONE fresh independent READ-ONLY R2 design-semantic + affected post-code implementation fidelity. STOP after PASS/BLOCK/UNKNOWN handback; no code, merge, deploy, PHI, Aiven or Gmail operations.

**Repo:** `athpapachr-cmd/osteoporosis`; **DRAFT/HOLD PR:** [#207](https://github.com/athpapachr-cmd/osteoporosis/pull/207).
**Production base main:** `8708a37c0d4083c3f811ced6b186e9ef7e817355`.
**Frozen code+test author HEAD:** **`a65b77235ffd9c7fe3ddfd2ecbbec108df388a13`**. Later review/CURRENT commits MUST be docs-only, verify via compare.
**Existing installed Obligations baseline:** #205; no current clinical release permission for this PR.

**Design:** `cockpit/OBLIGATION_CATEGORY_DUE_DATE_EDITOR_2026-10-22.md`, including the final PO refinement on category due/undated subgroups and in-list popup autonomy. Prior phase had an author-qualified quick editor using chart navigation, source+test `4785a583c838202a05386ca79ede0eaed25d825d`. The clinician explicitly rejected that extra navigation and requested same-screen actions. Review only the bounded **new final delta** plus affected cumulative R2 semantics; reuse previously passing patient chart/CP1/CP2 authority.

## Product expectation and example

Click ☑ Υποχρεώσεις → group **Ανά κατηγορία**. Inside e.g. **Εξετάσεις**, show visually separate headings **Με προθεσμία** / **Χωρίς προθεσμία**, with truthful dates and totals. Clicking an obligation (or ✎ Επεξεργασία) opens a modal **over that list**, not the patient chart. It allows category/due selection and explicit Save, reversible «Διαγραφή» with confirmation, and immediate «Αναίρεση διαγραφής», without losing active search/filter/grouping or jumping back to Home. «Άνοιγμα στον φάκελο ασθενούς» is optional. The existing chart's Pending popup also keeps its metadata editor.

For old titles e.g. «Αναλύσεις σε 1,5 μήνα» / «DXA σε 1,5 χρόνο» only show a *suggested* calendar date anchored to **signed original visit date**, approximate for half-month and never auto-persisted; clinician picks and explicitly saves. "Open" means unfinished, not due today.

## Frozen affected sources / evidence map

1. `clinical_pending_one_tap.py`: same protected Pending owner; one metadata POST updates only `ClinicalPendingORM.item_type`/`review_date`, append-only `metadata_corrected` event with before/after/actor/revision/idempotency. Existing lifecycle cancel/restore POST is **reused unchanged** for in-list Delete/Undo. Signed source, task ID/title/status/provenance/checkpoints not rewritten.
2. `static/cockpit/obligations-view.js`: category group nested dated/undated; `dialog#doctorObligationDialog`; clinician flag `checkpoint_write_allowed` and lifecycle revision guard; in-list metadata and reversible cancel/restore via pre-existing protected APIs; typed values and undo authority purge on logout/scope change/close/ESC; response uncertainty → reread rather than blind duplicate writes.
3. `static/cockpit/patient-chart.js`: retains existing patient chart metadata editor; **read-only** anchored dateSuggestion exported for optional use in list dialog. `static/cockpit/doctor-shell.js`: list task opener navigates to chart only on *explicit* action, not before edit/delete.
4. `static/cockpit/index.html` and `doctor-shell.css`: responsive in-place task modal and feedback/undo; no horizontal overflow at 1280/680/390; no invented task badge.
5. `test_cockpit_cp2.py`: exact real FastAPI + isolated SQLite fixture/Chromium 1280/390; category/date Save; prior signed visit unchanged; cross-session read; soft-delete confirm/Undo same-screen; no task recreation; revision/state/audit conflict/authorization checks. `test_cockpit_doctor_browser.py`: synthetic category with both due/undated, modal independent of chart, ESC purges PHI. Reuse already installed #205 tests and unchanged external Inbox.

**Exact frozen author CI SUCCESS:**
- [Home + real Chromium 39726320130](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320130)
- [Visit Capture / CP2 / metadata FastAPI + Chromium 39726320102](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320102)
- [Visit Brief / #161 synthetic 39726320189](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320189)
- [Surgery 39726320121](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320121)
- [Clinical Pending synthetic-demo 39726320082](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320082)
- [Canonical Impact 39726320087](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320087)

## Six finite review questions — no broad re-review

1. **Clinical write provenance and lifecycle:** exact patient/pending ID + signed encounter + clinician actor, intact signed original and status, metadata/lifecycle single revision stream and existing audit; no duplicate task writer or permanent delete shortcut.
2. **No involuntary action:** no auto due from title or email, no implicit task completion, no delete before confirmation, no duplicate Save on network uncertainty; same-key replay and stale/conflict treatment.
3. **Autonomous UX:** within-category due/undated headings, mobile/desktop, task modal interactions keep list visible; chart route explicitly optional; original chart editor remains independent, filters/grouping/scroll preserved after mutation.
4. **Security and privacy:** true auth backend, no task write for missing `checkpoint_write_allowed`, staff/secretary/cookie precedence, patient-name and modal inputs purged on Close/ESC/logout, no stale response repopulation; test-scope undo is not available on real scope and vice versa.
5. **Rollback/compatibility:** no DB migration, backend grant, credential or provider change; 100-row list limitation honestly displayed, existing Home/VisitBrief/VisitCapture/Pending step and surgery regressions unchanged.
6. **Evidence fidelity:** exact file blobs at frozen head, author CI outcomes, current `main`, PR branch/head; later commits docs-only. Focus on actual material defects; no extra ceremonies or broad review cycles.

## Result / boundaries

Provide `PASS | BLOCK | UNKNOWN`; `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`; `P0:P1:P2`; exact frozen head; evidence for each named question; minimal follow-up. **STOP READ-ONLY.** Author CI alone is NOT independent approval. No real patient writes, merge, deploy, Gmail access or server secret operations. Independent review and explicit separate release decision required before real-clinic use.
