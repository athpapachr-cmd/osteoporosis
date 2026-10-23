# Cockpit PR #207 — R2 source-verified in-list autonomous obligations, category/date

**Date:** 2026-10-22 (Asia/Nicosia). **MODE:** ONE FINAL focused independent READ-ONLY R2 P2 delayed-401 correction closure. Reuse the prior independent P1 CLOSED and all unaffected R2 PASS evidence; STOP after PASS/BLOCK/UNKNOWN. No code, merge, deploy, PHI, Aiven or Gmail operations. STOP after PASS/BLOCK/UNKNOWN handback; no code, merge, deploy, PHI, Aiven or Gmail operations.

**Repo:** `athpapachr-cmd/osteoporosis`; **DRAFT/HOLD PR:** [#207](https://github.com/athpapachr-cmd/osteoporosis/pull/207).
**Production base main:** `8708a37c0d4083c3f811ced6b186e9ef7e817355`.
**Frozen FINAL correction code+test author HEAD:** **`13bd91426d1f7f09ce5c0a9d17fc153df6eefce4`**. Original R2 source+test was `4088f7a1331ec2222132bd1248b69a6413dfc9f8`; first P1/P2 author fix was `ee6e1f9421a5e8c2e9f879f1adbac166c027a35e`. Only inspect the second focused fix for the remaining delayed-401 P2. Any later commits MUST be docs-only; verify via compare.
**Existing installed Obligations baseline:** #205; no current clinical release permission for this PR.

**Design:** `cockpit/OBLIGATION_CATEGORY_DUE_DATE_EDITOR_2026-10-22.md`, including the final PO refinement on category due/undated subgroups and in-list popup autonomy. Prior phase had an author-qualified quick editor using chart navigation, source+test `4785a583c838202a05386ca79ede0eaed25d825d`. The clinician explicitly rejected that extra navigation and requested same-screen actions. Review only the bounded **new final delta** plus affected cumulative R2 semantics; reuse previously passing patient chart/CP1/CP2 authority.

## Product expectation and example

Click ☑ Υποχρεώσεις → group **Ανά κατηγορία**. Inside e.g. **Εξετάσεις**, show visually separate headings **Με προθεσμία** / **Χωρίς προθεσμία**, with truthful dates and totals. Clicking an obligation (or ✎ Επεξεργασία) opens a modal **over that list**, not the patient chart. It allows category/due selection and explicit Save, reversible «Διαγραφή» with confirmation, and immediate «Αναίρεση διαγραφής», without losing active search/filter/grouping or jumping back to Home. «Άνοιγμα στον φάκελο ασθενούς» is optional. The existing chart's Pending popup also keeps its metadata editor.

For old titles e.g. «Αναλύσεις σε 1,5 μήνα» / «DXA σε 1,5 χρόνο» only show a *suggested* calendar date anchored to **signed original visit date**, approximate for half-month and never auto-persisted; clinician picks and explicitly saves. "Open" means unfinished, not due today.

## Frozen affected sources / evidence map

1. `clinical_pending_one_tap.py`: same protected Pending owner; one metadata POST updates only `ClinicalPendingORM.item_type`/`review_date`, append-only `metadata_corrected` event with before/after/actor/revision/idempotency. Existing lifecycle cancel/restore POST is **reused unchanged** for in-list Delete/Undo. Signed source, task ID/title/status/provenance/checkpoints not rewritten.
2. `static/cockpit/obligations-view.js`: category group nested dated/undated; `dialog#doctorObligationDialog`; clinician flag `checkpoint_write_allowed` and lifecycle revision guard; in-list metadata and reversible cancel/restore via pre-existing protected APIs; typed values and undo authority purge on logout/**successful principal switch**/scope change/close/ESC; response uncertainty → reread rather than blind duplicate writes.
3. `static/cockpit/patient-chart.js`: retains existing patient chart metadata editor; **read-only** anchored dateSuggestion exported for optional use in list dialog. `static/cockpit/doctor-shell.js`: list task opener navigates to chart only on *explicit* action, not before edit/delete.
4. `static/cockpit/index.html` and `doctor-shell.css`: responsive in-place task modal and feedback/undo; no horizontal overflow at 1280/680/390; no invented task badge.
5. `test_cockpit_cp2.py`: exact real FastAPI + isolated SQLite fixture/Chromium 1280/390; category/date Save; prior signed visit unchanged; cross-session read; soft-delete confirm/Undo same-screen; no task recreation; revision/state/audit conflict/authorization checks. `test_cockpit_doctor_browser.py`: synthetic category with both due/undated, modal independent of chart, ESC **and successful account-switch event** purge patient modal and undo feedback before new protected read. Reuse already installed #205 tests and unchanged external Inbox.

**Exact final P2 delayed-401 correction-head CI SUCCESS (synthetic data only; Visit Capture/CP2 37 passed):**
- [Home + actual Chromium 39766334219](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334219)
- [Visit Capture / CP1/CP2 + metadata FastAPI / Chromium 39766333973](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766333973)
- [Visit Brief / synthetic #161 39766334234](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334234)
- [Surgery 39766334096](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334096)
- [Clinical Pending synthetic demo 39766333983](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766333983)
- [Canonical Impact 39766334474](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334474)

**Focused R2 privacy correction:** a successful staff login/key switch can occur without a prior `clinical:unauthenticated` event. Earlier author-frozen `a65b772...` had an `authenticated` handler that refreshed the list but retained an already-open patient dialog and delete/undo identity. Corrected `obligations-view.js` clears dialog, typed fields and undo authority synchronously **before** `load()`; actual Chromium synthetic regression verifies it. Review this material edge once; do not infer independent PASS from author CI alone.

**Prior authored CI (superseded by corrected-head run):**
- [Home + real Chromium 39726320130](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320130)
- [Visit Capture / CP2 / metadata FastAPI + Chromium 39726320102](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320102)
- [Visit Brief / #161 synthetic 39726320189](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320189)
- [Surgery 39726320121](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320121)
- [Clinical Pending synthetic-demo 39726320082](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320082)
- [Canonical Impact 39726320087](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39726320087)

## First independent R2 BLOCK and bounded author correction — closure evidence

The prior independent source-reviewed R2 at `4088f7a1331ec2222132bd1248b69a6413dfc9f8` returned **BLOCK | COMPLETE_FOR_DECLARED_SCOPE | P0:P1:P2 = 0:1:1**. Its material findings were exactly:

- **P1:** `clinical:authenticated` correctly purged task dialog + Undo but left a typed patient-name value in `#doctorObligationsSearch`; a successful principal/key switch without a logout could expose that previous patient's search text.
- **P2:** an in-flight `undoLast()` restore reply, success OR failure, checked only `state.active`. After principal or scope change, it could repaint old feedback and start a stale list refresh.

The clinician authorized the **specific correction only**, implemented as one code+test commit **`ee6e1f9421a5e8c2e9f879f1adbac166c027a35e`** with exactly 3 files changed: `static/cockpit/obligations-view.js`, `test_cockpit_cp2.py`, `test_cockpit_doctor_browser.py`. It does **not** change `clinical_pending_one_tap.py`, auth backend, roles/grants, schema, provider, real data, lifecycle authority or other features.

For **P1**, confirm the `clinical:authenticated` handler synchronously clears modal/Undo, typed patient-name search AND scoped filters before issuing the new protected load. `test_cockpit_doctor_browser.py` asserts the search input was cleared even without prior logout.

For **P2**, confirm `undoLast()` captures the current `state.generation` and `state.scope` before the async request, and ignores **both** late successes and failures if either changed. `test_cockpit_cp2.py` uses actual Chromium with the isolated real clinical owner and a deferred local response: delayed restore **success after successful principal switch (1280px)** and delayed restore **error after real/test scope switch (390px)**; neither may repaint old feedback. Existing server-side lifecycle revisions/idempotency remain untouched.

**Author CI verified at exact `ee6e1f...`:** [Home 39760179844](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39760179844), [Visit Capture CP1/CP2 39760179853](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39760179853) (**36 passed**), [Visit Brief 39760179739](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39760179739), [Surgery 39760179796](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39760179796), [Pending Demo 39760179659](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39760179659), [Canonical 39760179866](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39760179866) all SUCCESS. **These are author CI, NOT independent closure approval.**

**Historical first-correction closure instruction (now superseded):** P1 was independently CLOSED and the normal P2 success/network-error cases passed, but the subsequent independent R2 closure identified a remaining P2 stale HTTP 401 side effect. The exact final closure scope and evidence follow below.

## Second independent R2 closure BLOCK (P2 delayed HTTP 401) and exact final correction

The subsequent independent READ-ONLY R2 closure on first-correction source+test `ee6e1f9421a5e8c2e9f879f1adbac166c027a35e` returned **BLOCK | COMPLETE_FOR_DECLARED_SCOPE | P0:P1:P2 = 0:0:1**. **P1 (typed search after successful principal switch) was independently CLOSED.** Remaining P2 was a side effect *before* the Undo generation/scope guard: `postTask()` immediately dispatched `clinical:unauthenticated` + `clinical:request-login` on a delayed 401 from a superseded Undo, potentially dismissing the replacement clinician's view. The previous 1280px stale-200 / 390px stale-network-error tests did not exercise that 401 path.

**Last narrowly authorized source+test correction:** `13bd91426d1f7f09ce5c0a9d17fc153df6eefce4`. Exactly two files changed relative to docs head `87a78fa562424a3705ff23a05516516586234025`: `static/cockpit/obligations-view.js`, `test_cockpit_cp2.py`. In the existing shared `postTask()`, capture **request generation and real/test scope before fetch**; emit invalidation and login-request events on HTTP 401 **only while that originating generation/scope is current and active**. Superseded replies still throw, but the caller's existing Undo/Save/Delete generation checks discard them without old-feedback repaint or new-session logout. **Genuine current-session 401 behavior remains fail-closed and still signals auth loss.** No backend owner, lifecycle write rules, permissions, secrets, DB schema, providers or Gmail changes.

**New actual Chromium regression:** the existing parameterized `test_browser_metadata_edition_and_relative_title_requires_explicit_save` adds 680px. It starts an Undo in the isolated real FastAPI+SQLite clinical fixture, defers the restore fetch response, performs a genuine successful `/clinical/login` again without logout, then resolves the old response as **HTTP 401**. Assert zero `clinical:unauthenticated` and zero `clinical:request-login` events, empty stale feedback, and the still-visible obligations view. The existing 1280px stale-200 and 390px stale-network-error cases remain unchanged in meaning.

**Verified author CI at exact code+test `13bd914...`:** [Capture/CP1/CP2 39766333973](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766333973) (**37 passed**), [Cockpit Home+Chromium 39766334219](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334219), [Visit Brief 39766334234](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334234), [Surgery 39766334096](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334096), [Pending Demo 39766333983](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766333983), [Canonical Impact 39766334474](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39766334474) are all **SUCCESS**. Author CI is not independent closure approval.

**FINAL closure instruction:** Inspect the exact two corrected file blobs and current 401 side-effect order; verify the 680px real-login/late-401 regression and 37-test passing logs. Reuse independently CLOSED P1 plus unchanged original R2 PASS and unaffected CI. Return `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2` with precise evidence and **STOP READ-ONLY**. No new broad design/post-code cycle. **#207 remains DRAFT/HOLD; no real clinic release until independent correction closure and separate PO go/no-go.**

## Six finite review questions — no broad re-review

1. **Clinical write provenance and lifecycle:** exact patient/pending ID + signed encounter + clinician actor, intact signed original and status, metadata/lifecycle single revision stream and existing audit; no duplicate task writer or permanent delete shortcut.
2. **No involuntary action:** no auto due from title or email, no implicit task completion, no delete before confirmation, no duplicate Save on network uncertainty; same-key replay and stale/conflict treatment.
3. **Autonomous UX:** within-category due/undated headings, mobile/desktop, task modal interactions keep list visible; chart route explicitly optional; original chart editor remains independent, filters/grouping/scroll preserved after mutation.
4. **Security and privacy:** true auth backend, no task write for missing `checkpoint_write_allowed`, staff/secretary/cookie precedence, patient-name and modal inputs purged on Close/ESC/logout/**successful account switch** (even without logout event), no stale response repopulation; test-scope undo is not available on real scope and vice versa.
5. **Rollback/compatibility:** no DB migration, backend grant, credential or provider change; 100-row list limitation honestly displayed, existing Home/VisitBrief/VisitCapture/Pending step and surgery regressions unchanged.
6. **Evidence fidelity:** exact file blobs at frozen head, author CI outcomes, current `main`, PR branch/head; later commits docs-only. Focus on actual material defects; no extra ceremonies or broad review cycles.

## Result / boundaries

Provide `PASS | BLOCK | UNKNOWN`; `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`; `P0:P1:P2`; exact frozen head; evidence for each named question; minimal follow-up. **STOP READ-ONLY.** Author CI alone is NOT independent approval. No real patient writes, merge, deploy, Gmail access or server secret operations. Independent review and explicit separate release decision required before real-clinic use.
