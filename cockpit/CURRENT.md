# Cockpit CURRENT — global Home and bounded clinical projections

> **NOW (2026-10-10): VISIT CAPTURE SMOKE UX R1 PASS — RELEASE READY / PRODUCT OWNER MERGE DECISION PENDING.** Independent R1 A1–A4 PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0 received; exact reviewed head `4c395b08e60334d1801dba6f611472a62238faa9`, six code/test/CI blobs unchanged in [PR #140](https://github.com/athpapachr-cmd/osteoporosis/pull/140), now OPEN/READY, UNMERGED. Receipt `reviews/VISIT_CAPTURE_SMOKE_UX_R1_PASS_RECEIPT_2026-10-10.md` blob `9436557ee6bf2ba384d3fbbb1b701bd9aaf5ab6e`. Focused CI PASS, existing protected backend unchanged, no live patient/provider data used. Draft/preview UX lets clinician test three Dia-style projections with synthetic/deidentified text and no patient ID/write; protected patient record mode remains separate. Picker only searches currently loaded latest 100 registered patients; acknowledged non-blocking larger-directory UX limitation. Original R2/P2 chains stay CLOSED; writer RELEASED; no more review. Production still old UX on `main=1febcfa27096df20f6f4bdeb3ccee15158634fc3` until separate merge/deploy authority. **Next: Product Owner explicit squash-merge authorization; then normal Render auto-deploy and no-PHI product smoke.** No real Dia/Heidi/GESY use authorized.

> **NOW (2026-10-09): VISIT CAPTURE DEPLOYED LIVE — PRODUCT OWNER SMOKE PENDING.** PR #138 reviewed design merged `9ec24a9a5f756b31ee8e6031b5d587c90ce91ac6`; PR #139 reviewed synthetic/manual Visit Capture merged `dd14cd0df1ca84d051ba7bf784b5b33226ed20b0`. Main checkpoint `8dbc9a78d032d710be4530875d4fa02a15628268`, Render deploy `dep-db4k2r3ncjis73dnc6p0` **LIVE** at exact source. Eight reviewed runtime/test blobs match immutable R2-passed candidate; prior P2-F2-01 closed, R2 chain CLOSED / 0:0:0. No new feature flag; authenticated clinicians can technically access the new Save route, but processing identifiable Dia/Heidi/GESY data remains unqualified/not authorized. No authenticated visual/functional product smoke was performed from this environment (direct HTTP unavailable). **Writer RELEASED.** Next: Product Owner no-PHI navigation/visual check of [Cockpit](https://ortho-reception-backend.onrender.com/static/cockpit/) → `Καταγραφή επίσκεψης`; then independently qualify real-patient workflow. No new general review or redeploy.


> **NOW 2026-10-09 — VISIT CAPTURE MERGED / RENDER AUTO-DEPLOY IN PROGRESS:** design PR #138 merged `9ec24a9…`, LIVE deploy `dep-db4jtlrncjis73dn58u0`; implementation PR #139 merged `dd14cd0df1ca84d051ba7bf784b5b33226ed20b0`, merged tree matches tested candidate tree. Render deploy `dep-db4k1l942hec73dll8j0` STARTED, NOT LIVE/SMOKED YET. Review chain PASS/0:0:0 closed. Authenticated Save endpoint will be reachable once deployed; no separate feature flag was added per Product Owner direction. **No real-patient/Dia/Heidi/GESY processing approved or executed.** Next: observe exact auto-deploy to terminal and perform no-PHI health/asset smoke; no code changes/duplicate deploy.


> **2026-10-09 PR #139 READY / APPLICABLE RELEASE CI PASSED:** main design `9ec24a9…` LIVE; PR #139 ready for review, base main, head at checkpoint `85b6b657376244605cd5b7a72e04835f4928809f`. Exact eight code/test blobs match independent R2, applicable CI PASS; inherited L0-only design scope check nonapplicable. Controlled PR squash merge authorized by Product Owner direction, then auto-deploy and no-PHI health. No real GESY/Heidi/Dia data processing.


> **2026-10-09 IMPLEMENTATION PR #139 RETARGETED TO MAIN:** parent design merge/deploy LIVE; PR #139 base `main`, head at retarget `d45620bf476f16eec20772716d6a5ed441f7e3fc`, diff contains only implementation/release scope (17 files), no reintroduced design. Mergeable true, still DRAFT. Next verify PR CI and reviewed blobs, then controlled release. No real-patient Dia/Heidi/GESY use authorized.


> **2026-10-09 ANCESTRY RECONCILIATION:** implementation branch fast-forwarded to two-parent commit `1c2eb9f5ca5a4ff6ac35aaea218fc11fcce0111c`; tree `1cee65627be8c897abc97cdc0952f73dacced3c1` unchanged, merged design main `9ec24a9a5f756b31ee8e6031b5d587c90ce91ac6` now ancestor. No byte drift. Next: re-target PR #139 to main, check diff/CI, no patient-data activity.


> **2026-10-09 DESIGN DEPLOY LIVE:** `dep-db4jtlrncjis73dn58u0` LIVE on Render Cockpit from design-only main `9ec24a9a5f756b31ee8e6031b5d587c90ce91ac6`. Visit Capture runtime NOT YET MERGED. Next: reconcile implementation PR #139 against merged design main, preserve code/test bytes; no live-data action.


> **2026-10-09 DESIGN PR #138 MERGED:** verified `main=9ec24a9a5f756b31ee8e6031b5d587c90ce91ac6`; Render auto-deploy `dep-db4jtlrncjis73dn58u0` BUILD_IN_PROGRESS (design/docs-only runtime). PR #139 remains stacked/unmerged; no Visit Capture code deployed. Next: wait for design deploy terminal state, checkpoint, then PR #139 source reconciliation.


> **2026-10-09 PR #138 READY:** design PR transitioned from draft to ready at unchanged exact head `5bc2a505758de6b24e82667bb84580d144f96b9c`; still unmerged. Next: authorized squash merge design first, then checkpoint before child implementation release.


> **NOW 2026-10-09 — CONTROLLED RELEASE AUTHORIZED / WRITER ACTIVE:** Product Owner approved progression into existing production Cockpit without a second Render preview or extra feature toggle. Authenticated Visit Capture Save will become reachable after code deployment; this is distinct from authorizing identifiable Dia/Heidi/GESY use. Only existing clinical auth, clinician Save and synthetic-only UI warning currently apply; no claim of enforceable synthetic-only mode. Release owner may merge PR #138 then #139 sequentially after immutable-source and applicable CI checks and checkpoint each transition. No new runtime edits, live patient/provider usage or new general review. STOP if changed code/test bytes or genuine applicable gate fails.


> **NOW (2026-10-09) — RELEASE CANDIDATE PREPARED / HOLD:** [PR #139](https://github.com/athpapachr-cmd/osteoporosis/pull/139) stacked DRAFT/unmerged on [parent design PR #138](https://github.com/athpapachr-cmd/osteoporosis/pull/138) DRAFT/unmerged; exact creation head `e667d8e5eb2541dd458e14a562acf2f598d1cb3c`. Release contract `releases/VISIT_CAPTURE_SYNTHETIC_RELEASE_CANDIDATE_2026-10-09.md` blob `71dcab7c492d35d2a661f7397f3bd846a11dcacc`. Corrected runtime/test head `07f68a2adc4664a4ab050b3fa10a663de591b662` unchanged; R2 closure PASS/0:0:0. PR canonical-impact, Visit Capture, Home and baseline-finalization CI PASS at pre-checkpoint head; L0 scope check is red solely because it applies an L0-only diff assertion to this legitimate non-L0 PR, though L0 contracts PASS; other unrelated checks pending. **Production merge/deploy HOLD** until distinct OFF-by-default backend Visit Capture write gate or full live-use qualification and separate explicit Product Owner release authority. No actual deployment or real-patient access. Writer RELEASED. Next: one release-enablement decision; no new broad R2 review or unrelated work.


> **2026-10-09 STACKED IMPLEMENTATION PR #139 OPEN / DRAFT / RELEASE HOLD:** [PR #139](https://github.com/athpapachr-cmd/osteoporosis/pull/139) targets `docs/cockpit-visit-brief-clinical-inbox-2026-10-07` because design PR #138 remains unmerged. Head at publication `2ebfe35c8bc2999b4d144d7d6ccdc8accf765e6e`. No code change, merge, deploy or data activation. Await bounded PR diff/canonical CI verification, then writer release; production feature gate unresolved.


> **2026-10-09 RELEASE CONTRACT FROZEN:** `releases/VISIT_CAPTURE_SYNTHETIC_RELEASE_CANDIDATE_2026-10-09.md` blob `71dcab7c492d35d2a661f7397f3bd846a11dcacc`. Candidate must be stacked as DRAFT PR on unmerged design PR #138 branch. Current code lacks separate server-side OFF-by-default Visit Capture gate; live write activation remains HOLD. No code change or merge/deploy now. NEXT: stacked draft PR and source/CI verification, then release preparation STOP.


> **NOW 2026-10-09 — VISIT CAPTURE RELEASE PREPARATION ONLY:** Product Owner authorized preparation of release candidate and PR, NOT merge/deploy or real-patient activation. Docs/PR writer only; corrected runtime/test head `07f68a2adc4664a4ab050b3fa10a663de591b662` and final independent R2 PASS / 0:0:0 unchanged. Parent design PR #138 remains DRAFT/unmerged. Proposed stacked implementation PR against its design branch. Release-readiness HOLD: no backend OFF-by-default Visit Capture activation gate exists; live processor/retention qualifications also OPEN. Do not confuse synthetic candidate PASS with safe production write access. Next: release notes/checkpoint + draft stacked PR, no merge/deploy.


> **NOW (2026-10-09) — VISIT CAPTURE FINAL R2 PASS / REVIEW CHAIN CLOSED:** independent focused C1–C3 closure PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0; original P2-F2-01 CLOSED. Received-review receipt `reviews/VISIT_CAPTURE_P2_F2_01_FOCUSED_CLOSURE_PASS_RECEIPT_2026-10-09.md` blob `adfb02e736d0db9ed77e305504a0b4a94ade9cb6`. Runtime/test frozen at `07f68a2adc4664a4ab050b3fa10a663de591b662`, CI `37963002895` SUCCESS, 20 tests. Existing F1/F3/F4/F5/F6 inherited PASS. Writer RELEASED. State IMPLEMENTED / FOCUSED TESTED / REVIEW CLOSED; NOT MERGED / NOT DEPLOYED / NOT REAL-PATIENT ACTIVATED. Draft design PR #138 remains separate/unmerged. Historical REVIEW HOLD statements below are superseded. Next action: separate bounded release-readiness and Product Owner authorization, never another reassurance review; no live Dia/Heidi/GESY/Gmail/Zadarma, patient-data or provider authority.


> **NOW — 2026-10-09 CLOSURE REVIEW HOLD:** P2-F2-01 narrow correction/test head `07f68a2adc4664a4ab050b3fa10a663de591b662` + CI `37963002895` SUCCESS; code/test frozen. Request `reviews/VISIT_CAPTURE_P2_F2_01_FOCUSED_CLOSURE_REVIEW_REQUEST_2026-10-09.md`, blob `e8a479cfbc014e70f72312d866cc1dddb2abf735`, prepared NOT DISPATCHED. Prior post-code R2 BLOCK remains pending closure; other F1/F3/F4/F5/F6 PASS reused. Writer RELEASED. Next: one independent C1–C3 closure and STOP; no implementation, merge, deploy, patient-data or provider effects.


> **NOW 2026-10-09 — CORRECTION TESTED / REVIEW HOLD:** P2-F2-01 dependency bounds corrected in `clinical_data.py` and targeted tests in `test_visit_capture.py`; runtime/test head `07f68a2adc4664a4ab050b3fa10a663de591b662`. CI run `37963002895` SUCCESS; post-test files changed only CURRENT/receipt metadata. Independent BLOCK receipt `reviews/VISIT_CAPTURE_POSTCODE_R2_P2_F2_01_BLOCK_RECEIPT_2026-10-09.md` blob `b448d45ff213c282eac8e97db19da0f29007e062`. Writer RELEASED. Only next permitted review: independent F2 + affected persistence closure, no re-review F1/F3/F4/F5/F6 except direct affected assertions; no merge/deploy.


> **2026-10-09 P2-F2-01 corrected candidate / CI PENDING:** strict external dependency model implemented in `clinical_data.py` (blob `ea9e3d39c124136f4818094aec4086e49cbd0919`); focused rejection+roundtrip tests in `test_visit_capture.py` (blob `00adc97ec0c726cc4a063470cf7a77e6f134a711`). Writer ACTIVE only until focused CI result/checkpoint; then one independent F2/affected persistence closure, no broad re-review.


> **NOW (2026-10-09): P2-F2-01 correction ACTIVE.** Independent Visit Capture post-code R2 returned BLOCK / 0:0:1 due only to unbounded `external_dependency`; F1/F3/F4/F5/F6 otherwise passed. Product Owner approved one narrow correction. Writer scoped to `clinical_data.py` strict nested dependency + `test_visit_capture.py` rejection/persistence oracles, then one focused CI and ONE F2/affected persistence closure review. Parent reviews stay CLOSED; no merge/deploy/live Dia/Heidi/GESY/Gmail/Zadarma/patient-data use.


> **POST-CODE R2 HOLD (2026-10-08):** request `reviews/VISIT_CAPTURE_DIA_HEIDI_GESY_POSTCODE_FIDELITY_REVIEW_REQUEST_2026-10-08.md` blob `85d4a7ad219e4c078f5e79252f571328d58570d6`; runtime/test head `4646c923…`, focused CI `37811359040` SUCCESS. Writer RELEASED. Next: one independent exact-head F1–F6 review; no further implementation/merge/deploy.


> **IMPLEMENTED / FOCUSED TESTED / WRITER RELEASED (2026-10-08):** runtime/test head `4646c923c728fd8613632db42e919687fc945504`; focused run `37811359040` SUCCESS for Python syntax, Visit Capture JS syntax, focused Visit Capture + affected Home tests, and diff hygiene. Later commits are CURRENT-only. No post-code R2 yet; no merge/deploy. Next: one independent exact-head post-code fidelity review, then stop.


> **CI harness checkpoint:** focused Visit Capture workflow added at `4646c923…`; result pending. Next: observe exact-head CI only.


> **Focused-test checkpoint:** eight synthetic acceptance cases + UI contract tests now exist (`322d617…`, `e1b22cc…`), but are NOT yet executed. Next: smallest CI run + syntax only; no broad suite.


> **Entry checkpoint:** Cockpit Home now links to the bounded Visit Capture surface (`70a5461…`). No other Home behavior changed. Next: focused synthetic tests/CI only.


> **UI checkpoint:** dedicated `/static/cockpit/visit-capture/` surface exists with explicit patient confirmation, transient candidate insertion, automatic 3-level preview and one Save. No live providers and no tests claimed yet. Next: Home entry + focused synthetic tests.


> **Implementation checkpoint:** backend Visit Capture boundary committed at `9e780de18383fd244862187cb9be967c0a2c87cd`; UI/tests not yet implemented or verified. No live providers/patient data. Writer remains active. Next: dedicated Cockpit capture UI + focused synthetic evidence only.


> **NOW (2026-10-08 IMPLEMENTATION ACTIVE):** Product Owner approved the reviewed synthetic/manual Visit Capture first-code slice (“Εγκρίνω”). Active implementation branch `feat/cockpit-visit-capture-v1-2026-10-08`, base reviewed docs head `5bc2a505758de6b24e82667bb84580d144f96b9c`. Scope is exactly the R2-passed boundary: confirmed-patient capture UI, transient VisitCaptureCandidateV1, stale patient/context guard, one three-level preview, one explicit protected Save to existing encounter owner, minimum bounded pending persistence, and server-enforced no post-signoff overwrite for Visit-Capture-signed encounters. No live external source/provider or patient-data activation. Next: bounded implementation + focused synthetic evidence, then writer release into independent exact-head post-code R2 HOLD.


> **NOW (2026-10-08):** Independent Visit Capture delta pre-code R2 PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0 received; chain CLOSED. Receipt `reviews/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_R2_PASS_RECEIPT_2026-10-08.md`. Prior R2-pending text below is historical. Next: Product Owner first-code checkpoint + separately scoped implementation writer before code. Narrow synthetic/manual Visit Capture only; no live Dia/Heidi/GESY/Gmail/Zadarma or patient-data activation.


> **NOW (2026-10-08):** Visit Brief + independent Clinical Inbox parent R2 = PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0 and CLOSED. New Visit Capture Dia/Heidi/GESY → Cockpit delta = DESIGN FROZEN / independent delta R2 PREPARED / NOT DISPATCHED / NO VERDICT / RUNTIME NOT STARTED.
> **Confirmed daily workflow:** Cockpit is used inside Dia; Heidi today + GESY today + immediately previous GESY are source tabs; Dia inserts one candidate into the confirmed-patient Cockpit Visit Capture surface; clinician reviews and presses one Save.
> **Mechanism:** EXTEND existing protected `clinical_patients.patient_id` + `clinical_encounters` + G3 projection. Dia/GESY/name/phone/tab text cannot choose destination patient. One structured encounter renders Snapshot / Visit Brief / Encounter Detail.
> **Safety:** candidate insertion is transient, not a record; stale patient/capture context fails closed; no silent source conflict resolution; later events do not rewrite the signed encounter; post-signoff overwrite is forbidden unless attributable amendment/revision support exists.
> **Scope:** synthetic/manual first-code only after delta R2 PASS and separate writer authority. No live Dia/Heidi/GESY provider access, Gmail, Zadarma, authoritative lab-result acceptance, booking or deployment.
> **Delta design:** `VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_2026-10-08.md` blob `7e19ce29a3c33a0d06e6c0ec485952f41c0ecf72`.
> **Delta review request:** `reviews/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_PRECODE_REVIEW_REQUEST_2026-10-08.md` blob `2e169a870d5ed04823c09d810f305dd3881defac`.
> **Exact next action:** one independent read-only delta R2; then STOP on PASS or one bounded correction/closure on BLOCK. No implementation follows automatically.

> **NOW (2026-10-07):** Calendar Unification = RELEASED / PRODUCT OWNER SMOKE VERIFIED. Visit Brief + Clinical Inbox = DESIGN CANDIDATE / R2 PRE-CODE HOLD / REQUEST PREPARED, NOT DISPATCHED / NO VERDICT / RUNTIME NOT STARTED.
> **Product authority:** direct continuation request confirms two independent entries, same-day review/communication regardless of appointment, and explicitly excludes runtime implementation.
> **Root lock:** `CURRENT_OPERATIONAL.md` alone records writer authority; the bounded design writer is RELEASED into R2 HOLD; PR-1 remains on its independent release HOLD. No Reception/Ops writer or release authority is claimed.
> **Product boundary:** `PRODUCT_CONSTITUTION.md`; phase architecture `../CLINICAL_EXCELLENCE_PLAN.md` §§33–34.

## Current bounded action

The revised design is `VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`, exact blob `8ee79a96350f9ac42059eeb2a3e836cbe9d11245`. The one prepared independent request is `reviews/VISIT_BRIEF_CLINICAL_INBOX_PRECODE_REVIEW_REQUEST_2026-10-07.md`, exact blob `5bb6de5fce76170ca65d0422a4d6ed8272aabc20`.

Next: perform that bounded independent read-only R2 on the declared candidate, dispose Q1–Q6 and stop. The request has not been dispatched; no verdict/implementation eligibility is claimed. Open provider/identity/contact/field-level retention and live-activation gates remain at design §§10–12. PR-1 retains `SLICE_PLAN_CURRENT.md`; this parallel design does not overwrite it. Design writer released; no runtime/schema/test/config/provider changes or release follow from this checkpoint.

## Published design candidate

Draft PR [#138](https://github.com/athpapachr-cmd/osteoporosis/pull/138) is OPEN against main `045798dfa28612b268f16a99262c6ecc9ca4829d`. Design-bearing commit `e305a9678183cab84df7825a4f59c5d4a841e113` contains only the 16 declared documentation/manifest files. This following publication checkpoint changes CURRENT metadata only and leaves the pinned design/request blobs unchanged. R2 request remains PREPARED / NOT DISPATCHED / NO VERDICT. NOT MERGED / NOT DEPLOYED; writers released.

## Author evidence (not independent R2)

Documentation-only scope: 16 changed Markdown/manifest files; AGENTS, PROCEDURES, PR-1 `SLICE_PLAN_CURRENT.md`, runtime/schema/tests/config and synced `sources/` unchanged. Append-only changelog, byte-identical predecessor archive/digest manifest, design/request blob pins and links verified. Local Canonical Impact guard PASS and diff hygiene PASS. No runtime test, live/provider request or independent review was run. Design is prepared, not frozen as an implementation contract. Draft-PR publication is complete; no independent review or further product transition was performed.

## Calendar Unification closeout

- Product Owner smoke PASS is supplied in the referenced coordinator conversation and reaffirmed by the direct current request (2026-10-07 checkpoint). It covers Home Previous/Current/Next and the weekly Calendar smoke requested at release; no new agent-run smoke is claimed.
- Reception reviewed runtime/test head `49a321531fbe8042d2b22cb6ba7b121e901879ae`; merge/main `e6babb5ab758d282166767c36dd7311024406afb`; deploy `dep-db2m42mi0phs7393jms0` LIVE per the prior release record.
- Cockpit reviewed runtime/test head `f2fdb146366850fac659acb74bacd40e9ec88588`; PR #137 merge `3f4d82280e3a58f25383ccd896ebcf806ec755c2`; final release main `045798dfa28612b268f16a99262c6ecc9ca4829d`; final deploy `dep-db2m7sbl550s73binhig` LIVE per the existing Ops release record and coordinator handback. Subsequent release checkpoint changes were docs only.
- Independent Calendar post-code R2 PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0; review chain CLOSED. Durable result: `reviews/CALENDAR_UNIFICATION_POSTCODE_FIDELITY_PASS_2026-10-06.md`.
- No live Cal booking/cancel/reschedule/provider mutation or heavy sync was performed in this design task. Production deployment was not re-polled or triggered.

## Preserved parallel dependencies

- PR-1 privacy/provider release HOLD remains independent.
- P0-V0 Visit Intelligence branch `b22fd5c610d20baf0f8ef3384ca16472cd2d1f92` and its `programme/OST-VISIT-INTELLIGENCE/CURRENT.md` are absent from this verified main. The reported cumulative PASS still needs its owning coordinator's reconciliation/release; it is not assumed available runtime.
- Reuse released G3 summary mechanics and protected clinical records. No P0-V0 merge/release or OST-UI authority is inferred.
- Reception/Ops remain read-only source references in this task; their separate workstream status/locks stay at their owner.

## Historical checkpoints — superseded as operational next actions

The exact outgoing root/current/design/request bytes are archived at `archive/checkpoints/2026-10-07-pre-clinical-inbox-replan/` from base `045798dfa28612b268f16a99262c6ecc9ca4829d`. The records below remain provenance only.

#### Historical implementation checkpoint

### D1.1 freshness implementation checkpoint — 2026-10-04

```text
COCKPIT BRANCH:       feat/cockpit-d1-1-global-context-2026-10-03
COCKPIT RUNTIME HEAD: 9dc14bbde1005aab60922e88e021518197f0654f
COCKPIT MAIN:         b9beca7b1e233245f9ea3429a247a1c80a69ac7c (PHYSIO-only drift)
RECEPTION BRANCH:     feat/d1-1-fresh-schedule-projection-2026-10-04
RECEPTION HEAD:       b4739678016c1cd10f3afd8381e7c4de222f38ba
RECEPTION MAIN:       bcfa57e0c1ca7358fb898b393e1dafa7b042c238
OPS WRITER CLAIM:     1b98f7f023fdd92a55938591c49a49f12af5e916
SOURCE DESIGN:        00e3b4527a8ccbfc6df1a635494da136ccb9dd33
PRE-CODE R2:          PASS / 0:0:0
FOCUSED TESTS:        Cockpit 31 PASS; Reception 54 tests + 33 subtests PASS
POST-CODE R2:         PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0 / CLOSED
OLD POST-CODE:        HOLD / SUPERSEDED FOR SOURCE FIDELITY / NO VERDICT
WRITER:               BOUNDED RELEASE SCOPE / NO RUNTIME EDIT
MERGED / DEPLOYED:    NO / NO
NEXT:                 Reception PR gate, merge, deploy; then Cockpit PR gate, merge, deploy
```

The new request is `reviews/D1_1_FRESHNESS_POSTCODE_FIDELITY_REVIEW_REQUEST_2026-10-04.md`. Release-time `RECEPTION_SCHEDULE_CONTEXT_URL` and shared existing ingest-key equality remain to be checked at the later release gate; neither was configured or tested against production now. The weekly Calendar stays on the background Clinical Calendar path. The existing source design and old post-code request remain immutable historical evidence.

### Historical D1.1 freshness pre-code replan checkpoint — 2026-10-04

```text
CURRENT REMOTE MAIN:    12be588866aee6543444eee0b435e9064c39bc3f
IMPLEMENTATION BRANCH: feat/cockpit-d1-1-global-context-2026-10-03
OLD BRANCH HEAD:       6f56093ffbf33d01078eb8a25af35eb65a1e851a
RUNTIME/TEST HEAD:     dd558e2fa8d59409d21913d2d3f2fbf885fdfece (unchanged)
RECEPTION MAIN:        bcfa57e0c1ca7358fb898b393e1dafa7b042c238
OLD POST-CODE REQUEST: HOLD / NO VERDICT
NEW SOURCE DESIGN:     D1_1_FRESHNESS_REPLAN_DELTA_2026-10-04.md
RUNTIME WRITER:        NONE / SOURCE DESIGN ONLY
MERGED / DEPLOYED:     NO / NO
PRODUCT OWNER STEP:    CONFIRMED / “Προχώρα” / 2026-10-04
SOURCE PRE-CODE:       BLOCK / COMPLETE_FOR_DECLARED_SCOPE / 0:3:0
FINDINGS:              FRESH-01 past+upcoming; FRESH-02 total budget; FRESH-03 raw logging
CORRECTION:            FRESH-01/02/03 in source delta / independent closure PASS
SOURCE PRE-CODE CLOSE: PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0
NEXT:                  fresh two-repository writer checkpoint before implementation
```

The existing snapshot feed and Clinical Calendar remain the Module-01 weekly/background path. The operational strip must use a bounded fresh actual-Cal-bookings projection owned by Reception. Actual urgent bookings are appointments; urgent, lab and telephone availability windows are not. Preserve the already accepted Previous/Current/Next, overlap, privacy, weekly-filter and clinical identity boundaries unless the new source directly requires a correction. The prior design/review evidence stays historical; this replan does not issue a post-code fidelity verdict.

### Historical D1.1 implementation-start checkpoint — 2026-10-03

```text
BASE MAIN:            d5da6271567a4141b2708d9fa12e673dfce37131
BRANCH:               feat/cockpit-d1-1-global-context-2026-10-03
DESIGN BLOB:          261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33
INDEPENDENT PRE-CODE: PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0
PRE-01 / PRE-02:      CLOSED / CLOSED
IMPLEMENTATION:       COMPLETE / FOCUSED TESTED / NO POST-CODE PASS
WRITER:               RELEASED / RUNTIME FROZEN FOR REVIEW
POST-CODE REVIEW:     REQUEST PREPARED / NOT LAUNCHED
MERGED / DEPLOYED:    NO / NO
HISTORICAL NEXT:      independent fidelity review of dd558e2fa8d59409d21913d2d3f2fbf885fdfece; now superseded by the 2026-10-04 freshness replan above
```

Supplied closure evidence is retained in `reviews/D1_1_PRECODE_CLOSURE_PASS_2026-10-03.md`. Q1–Q3 were supported, no material findings/missing evidence, stop reason Q1–Q3 disposed. The pre-code chain is stopped under PROCEDURES P5/P5.1. The corrected design is copied byte-for-byte from design branch head `ba1f7eafb500986ad07b2e6b4320ac982896eb77`; its historical pending-review status is superseded by this checkpoint without changing the reviewed blob. Reuse the existing complete snapshot feed; preserve one store, snapshot reconciliation/manual overrides, legacy discard and weekly osteoporosis filtering. No Reception/config/schema/identity/Visit Brief expansion. Review PASS is not release authority. Visit Brief follows only D1.1 closure/release.

### D1.1 implementation/test evidence — 2026-10-03

- Snapshot-only `retain_other=True` after complete scope/interval validation; default-off legacy helper still discards/removes unrelated rows with compatible counters.
- Shared minimization clears phone/link after effective other upserts and manual override clearing. Manual promotion alone cannot restore discarded fields.
- Existing weekly filter, overrides, classifier and exact source/window reconciliation preserved; one table, no migration.
- Protected `/clinical/calendar/cockpit-context` returns an explicit minimal model: today-only Previous, active Current, cross-day Next, strict earlier-Aclasta exception, other/tied-start overlaps fail closed. All timestamps are explicit UTC; day bounds use Asia/Nicosia including DST.
- Home uses the global projection; no fallback list request, phone/link/raw metadata or browser storage. Global card counters removed; weekly link retained; next-day date and freshness/unavailable notices visible. Patient display slot has a future appointment-ID interaction seam; no Visit Brief handler/history yet.
- **30 tests PASS in 1.16s:** `test_clinical_calendar.py`, `test_clinical_calendar_snapshot.py`, `test_cockpit_home.py`, `test_cockpit_surgery_queue_ui.py`, Python 3.12.2. Python/JS syntax, `test_g4_workspace_ergonomics.js`, diff hygiene PASS.
- `source_updated_at` is max retained normalized-row update time, not a new persisted snapshot receipt ledger. Empty storage has unknown freshness; UI says so. Cadence/producer unchanged. Global other rows populate through the next existing complete snapshot after release; no live parity or production smoke claimed.
- Exact implementation/test head: `dd558e2fa8d59409d21913d2d3f2fbf885fdfece`. Sole prepared independent request: `reviews/D1_1_POSTCODE_REVIEW_REQUEST.md`. Docs-only checkpoint does not alter runtime/test bytes. No implementation or release PASS is inferred from author checks.

### Product Owner correction — 2026-10-02

Production use exposed that the deployed D1 satisfies its tested local semantics but is connected to the wrong product projection for the **global** Cockpit.

Observed behavior:
- Reception / Digital Secretary already shows real clinic appointments from its existing schedule view;
- Cockpit D1 reads the Osteoporosis Clinical Calendar endpoint and today's local-day window;
- therefore a non-osteoporosis appointment — and an immediate next appointment on the following day — may be visible in Reception while Cockpit shows no Previous / Current / Next.

**D1.1 required correction:**
- Previous / Current / Next on global Cockpit must come from the existing Digital Secretary / appointment schedule truth, not from the Osteoporosis-only Clinical Calendar projection;
- reuse the Digital Secretary's existing protected schedule mechanism (currently surfaced by its dashboard via `GET /dashboard/schedule`) via the already existing complete snapshot feed rather than a new dashboard consumer or Cal.com fetcher/calendar;
- the projection must be global across appointment types/clinics;
- `Next` must resolve the immediate future appointment across the relevant schedule horizon and must not stop at the local-day boundary merely because the Home is labelled “Today”;
- the weekly Osteoporosis Clinical Calendar remains a separate Module-01 view and must not be broadened merely to make global Cockpit work;
- Digital Secretary / booking source remains owner; Cockpit remains read-only;
- reuse the existing server-side ingest authentication; apply the corrected server/browser minimization contract without dashboard cookies or new secrets.

The existing D1 Aclasta-overlap/start-order behavior remains a useful scheduling rule where the required appointment classification is available; D1.1 must preserve it or fail closed if the source cannot support that classification safely.


**Visit Brief interaction decision — Product Owner 2026-10-02:** keep Home compact. Previous / Current / Next should show only concise appointment context by default. Clicking/tapping the patient opens a floating/overlay Visit Brief rather than expanding all history inline. The overlay should prioritize: main problem/reason → relevant history → prior-visit decision/plan → pending/awaited items → what is expected/checking today → relevant communication only when it materially changes preparation. A full-record navigation action may remain available from the overlay. This is a product checkpoint only; no Visit Brief implementation authority is granted yet.

**Product priority correction:** Relevant Communication Context is not the primary next clinician-facing layer. The next meaningful Home layer after correct appointment context is a compact **Visit Brief / Patient Context** answering, in this order:

1. who is this patient / what is the main problem or reason for today's visit;
2. what matters from the relevant clinical history;
3. what was decided/said at the previous visit;
4. what remains pending or awaited;
5. what we are expecting/checking today;
6. only then, any communication that materially changes the visit preparation.

Communication remains bidirectional and Digital Secretary-owned, but it should normally appear as supporting evidence/context inside the Visit Brief rather than as the dominant card.

Where no strongly linked protected clinical patient record exists, Cockpit must not infer one from phone/name alone. It should show only the bounded appointment/operational context and fail closed for patient-specific clinical history until an authorized strong or clinician-confirmed link exists.

D1 is a small, directly visible, read-only Dashboard projection of **Previous / Current / Next** appointment context from the existing normalized Clinical Calendar. It lets the clinician see who was just seen, who is being seen now and who is next without opening the full calendar. The existing weekly Osteoporosis Clinical Calendar remains available through a link labelled **«Άνοιγμα εβδομαδιαίου ημερολογίου»** and is not replaced or removed.

**R1 classification:** bounded UI/read-only projection using unchanged protected Calendar data, with no new clinical fact authority, patient identity authority, write path or external side effect. Expected implementation scope is `static/cockpit/index.html`, `static/cockpit/app.js`, `static/cockpit/styles.css`, `test_cockpit_home.py` and this workstream checkpoint; root `CURRENT_OPERATIONAL.md` records the bounded writer lock. The Clinical Calendar and its source keep ownership. D1 authorizes no appointment write, booking/cancellation/rescheduling, second calendar, Setmore reminder, Digital Secretary workflow change, messaging transport, Visit Intelligence clinical-state write or patient-matching redesign.

**Accepted D1 semantics:** clinician attention follows appointment `start_at` order. `Τώρα` is the single active appointment; when an earlier still-active row is an Aclasta infusion slot and a later non-Aclasta appointment has started, the later appointment is Current. Other simultaneous active overlaps fail closed visibly. `Προηγούμενο` is the immediately preceding appointment in start order relative to Current, or the latest-started completed appointment when there is no Current. `Επόμενο` is the earliest appointment whose `start_at` is later than now. This is schedule context, not live patient-presence tracking. The Home may show `patient_display_name` only from the authenticated protected Calendar response for these slots; it must not render `phone_e164` or `linked_patient_id`.


### D1 implementation/tested checkpoint — 2026-10-01

```text
BASE MAIN:             87aedad3ad512e4b17a1eb737f0ff8302857aff2
BRANCH:                feat/cockpit-today-context-strip-v1-2026-10-01
SUBSTANTIVE HEAD:      b4ce49b6cdb1083d56453abb1e4cbc10c834b294
R0 DOCS-CORRECTION:    a965cec7c307c354d7a2ef98d7e2ff2937fe46ee
PR:                    #130 / MERGED
IMPLEMENTED:           YES
FOCUSED TESTED:        YES
INDEPENDENT R1 REVIEW: CLOSURE PASS / REVIEW CHAIN STOPPED
MERGED:                YES / db42903e08b11dfe52d427a410f8b15547fb5507
DEPLOYED:              YES / LIVE / dep-davbs48473hc73eust00 / source 794d96dc4e0676691a4d6333b8f47bdd335cdb6c
```

Implemented behavior:

- Cockpit Home renders **Προηγούμενο / Τώρα / Επόμενο** directly from the existing authenticated Clinical Calendar read endpoint.
- Previous / Current clinician attention follows deterministic `start_at` order. A later-started non-Aclasta visit may lawfully become Current while an earlier Aclasta infusion slot remains active; other simultaneous active overlaps fail closed visibly. Next remains the earliest future `start_at`.
- the context refreshes once per minute while the Dashboard remains open;
- `patient_display_name` is used only as protected appointment display context; D1 does not consume `phone_e164` or `linked_patient_id`;
- the existing weekly Osteoporosis calendar remains unchanged and reachable through **«Άνοιγμα εβδομαδιαίου ημερολογίου»**;
- daily-feed freshness remains stated in clinician-facing language without exposing technical sync metadata;
- no Calendar write, booking lifecycle, Setmore, Secretary, Zadarma, Visit Intelligence, OST-UI or PR-1 behavior changed.

Focused evidence on the substantive head:

- Cockpit Home tests — run `36897402962` — **SUCCESS**;
- Cockpit surgery queue — run `36897403159` — **SUCCESS**;
- Canonical impact guard — run `36897403120` — **SUCCESS**;
- Physio Knee OA Cockpit integration — run `36897403296` — **SUCCESS**;
- Physio Knee OA V5 integration — run `36897403218` — **SUCCESS**;
- Physio jurisdiction overlay — run `36897403208` — **SUCCESS**;
- Clinical Learning L1 / L1B / L1C inherited gates — **SUCCESS**.

Clinical Learning L0 run `36897403005` validated its own contracts but failed only its **design-only scope** assertion because D1 intentionally changes Cockpit runtime files. That scope gate is not applicable evidence for this R1 Cockpit implementation and does not indicate a Clinical Learning contract regression.

**Independent R1 result:** the first exact-head review returned **BLOCK** because it interpreted `Previous` as “the completed row with greatest `end_at`” and used an overlapping historical example (08:00–11:00 and 09:00–10:00).

**Final Product Owner clarification of D1 scheduling semantics:** normal appointments are sequential, but **Aclasta is a legitimate exception**. Aclasta may occupy a one-hour treatment slot while the patient is in a clinic room and the clinician may start another appointment during the last part of that infusion slot. Therefore clinician attention follows **appointment start order**, not “greatest end time”.

Accepted examples:

- 08:00–09:00 A, 09:00–10:00 B, now 11:30 → **Previous = B**;
- Aclasta 09:00–10:00 + Review 09:40–10:20, now 09:50 → **Previous = Aclasta / Current = Review**;
- Aclasta 09:00–10:00 + Review 09:40–09:55, now 10:05 → **Previous = Review**, even though Aclasta ended later.

**Bounded implementation rule:** D1 uses the ordered `start_at` sequence for Previous/Current attention. If multiple appointments are active and all earlier active rows are `aclasta` while the latest-started active row is non-Aclasta, the latest-started row is Current and the immediately preceding scheduled row is Previous. Other concurrent-current overlaps remain ambiguous and fail closed visibly. Next remains the earliest future `start_at`. Weekly Calendar preservation, identity/privacy boundaries and ownership are unchanged.

**Final bounded correction tested candidate:** substantive runtime/test head `b4ce49b6cdb1083d56453abb1e4cbc10c834b294`.

Focused evidence on that head:

- Cockpit Home — run `36904690143` — **SUCCESS**; this executes the production `appointmentContext` logic and covers sequential Previous, lawful Aclasta overlap, post-overlap Previous by later start, non-Aclasta overlap fail-closed, syntax, workspace navigation and diff hygiene;
- Cockpit Surgery Queue — run `36904690332` — **SUCCESS**;
- Canonical Impact — run `36904690084` — **SUCCESS**;
- Clinical Learning L1/L1B/L1C inherited gates — **SUCCESS**.

The persistent Clinical Learning L0 red check remains the same non-applicable design-only changed-file scope assertion; its contract validation step passes and D1 does not change Learning contracts.

**Final independent closure disposition:** all runtime/product D1 behavior above was accepted. The only remaining BLOCK was two stale canonical statements that still described the superseded `end_at` semantics. Those two statements were corrected in the R0 canonical-only delta at `a965cec7c307c354d7a2ef98d7e2ff2937fe46ee`; no runtime or test semantics changed. Under `PROCEDURES.md` P4/P5/P7 this requires no new product/implementation review, so the D1 review chain is stopped.

**Historical D1 release checkpoint:** the Product Owner authorized the lawful PR #130 merge/release step provided the current exact-head state and applicable CI remain clean. PR #130 was squash-merged as `db42903e08b11dfe52d427a410f8b15547fb5507`. Render deploy `dep-davbs48473hc73eust00` reached **LIVE** from source `794d96dc4e0676691a4d6333b8f47bdd335cdb6c`. D1 is released and its bounded writer is released. No production smoke beyond successful deploy health is claimed. **Historical follow-up proposal:** D2 Product Owner checkpoint. Superseded by the 2026-10-02 D1.1 → Visit Brief priority above; D2 is supporting context. Do not start D2 implementation before D1 is durably released.

**D2 — Relevant Communication Context** was the next concept proposed at D1 release; it is now deferred behind D1.1 / Visit Brief. Its bidirectional read boundary and phone-correlation limit live in `cockpit/PRODUCT_CONSTITUTION.md`. D2 needs its own bounded design/authority before any integration or UI work.

**Visit Brief / What Changed** remain product direction, downstream of durable P0-V0 closure/release and the OST-UI coordinator synthesis/Product Owner decision where the Module-01 interaction is affected. The current-problem Osteoporosis workspace also depends on that synthesis. A Next Visit continuity write path remains a separate bounded clinical-authority slice. None of these dependent slices starts from this checkpoint, and OST-UI's R4 completion does not authorize a prototype.

The release and follow-up records below are as-of checkpoints. Their former “next” text is historical and cannot override this current action or the owning workstream CURRENT.

---

# Cockpit Home v1 release checkpoint

> **STATUS:** COCKPIT HOME V1 RELEASE COMPLETE / PR #118 MERGED / RENDER LIVE / ROOT SMOKE VERIFIED.
> **Workstream:** Clinical Excellence Cockpit Home v1.
> **Branch:** feat/cockpit-home-v1-2026-09-27.
> **Base main:** 88ad125f0a25a471b0151eeb26e68b8b8a93c84f.
> **Exact tested head:** 7744e2142b54ba0b7c8b92c87e4e511cffffc26b.
> **Root writer lock:** unchanged; PR-1 Heidi-first transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this slice is limited to global Cockpit navigation/home presentation and Osteoporosis sidebar cleanup.

## Product-owner decisions

- / must open the global Cockpit Home, not the Osteoporosis module.
- Osteoporosis is Module 01 / proving ground, not the whole Cockpit.
- Osteoporosis sidebar must contain only Module-01 navigation.
- Global Clinic Utilities belong on Cockpit Home.
- The duplicate physiotherapy referral entry must collapse to one global tool entry.
- The top-level Heidi AI navigation item must be removed from Osteoporosis; Heidi remains an encounter-capture capability inside the Module-01 workflow and future reusable Core.
- The separate Reception/calls dashboard remains owned by ortho-reception-backend-v2; Cockpit Home links to it rather than duplicating its implementation.
- Calendar/Cal.com reason ingestion is a separate follow-up integration slice because its source-of-truth boundary spans repositories.

## Home v1 information architecture

Cockpit Home
- Today
  - Clinical Calendar
- Clinical Modules
  - Osteoporosis · Module 01 · Active
- Learning & Improvement
  - Clinical Learning Hub
- Clinic Utilities
  - Παραπεμπτικό Φυσιοθεραπείας
  - Αναρρωτική άδεια
  - Ιατρικές εκθέσεις
  - Ραδιοκύματα
- Reception
  - Call / Reception dashboard (external owner)

Module 02+ appear as future placeholders only; no fake runtime is implied.

## Safety / scope

- no patient/clinical data model change;
- no Cal.com/Setmore/Digital Secretary mutation in this slice;
- no PR-1 transcript runtime mutation;
- no duplicate copy of the Reception dashboard;
- no new browser patient persistence;
- root Clinical Auth boundary remains unchanged.

## Verification

Cockpit Home gate:

```text
run: 36273987676
head: 7744e2142b54ba0b7c8b92c87e4e511cffffc26b
result: SUCCESS
```

Verified:

- Python syntax PASS;
- Cockpit browser JavaScript syntax PASS;
- 4 deterministic Cockpit Home tests PASS;
- Osteoporosis G4 workspace/navigation regression PASS;
- diff hygiene PASS.

Behavior proven:

- / enters /static/cockpit/;
- global Cockpit Home contains Modules, Calendar, Learning, Clinic Utilities and Reception;
- Clinic Utilities are absent from the Osteoporosis sidebar;
- top-level Heidi AI is absent from the Osteoporosis sidebar;
- one global physiotherapy referral entry exists;
- Home calendar summary uses aggregate counts only and does not render patient identity fields.

## Release PR

PR: #118
URL: https://github.com/athpapachr-cmd/osteoporosis/pull/118
base: main
base_sha: 88ad125f0a25a471b0151eeb26e68b8b8a93c84f
head: feat/cockpit-home-v1-2026-09-27
head_sha at PR creation: f0f252dfaef9c08146c636e5d10fbf7ede97d553
draft: NO
merged: NO
deploy: NO

## Release completion

```text
PR: #118
merge method: squash
merge commit: b7b8779d943eb1d8db1f8966a81796bc69647c3b
merged: YES
Render service: osteoporosis / srv-d5qfk31r0fns73di596g
deploy: dep-das41frncjis73e6q47g
deploy trigger: new_commit
deploy status: live
manual redeploy: NO
```

Final PR exact head `ba2de644f4fd16e70d12f04831b28f5fd42730ec` passed all 16 checks, including Cockpit Home, Canonical Impact, G3/G2/G1, Clinical Learning L1/L1B/L1C, Clinical Documents, CU-1 and Physio inherited browser gates.

Production smoke from Render logs:

```text
GET /                 -> 307 Temporary Redirect
GET /static/cockpit/  -> 200 OK
Application startup   -> complete
Clinical storage      -> PostgreSQL online
```

Released navigation ownership:

- global Clinic Utilities live only on Cockpit Home;
- Osteoporosis remains Module 01 and no longer owns the general utilities group;
- top-level Heidi AI navigation is absent from the Osteoporosis sidebar;
- Heidi capture/exposure content remains inside the Module-01 encounter workflow;
- exactly one global physiotherapy referral entry is exposed on Home;
- Reception remains a linked separately-owned system.

## Current release state

```text
IMPLEMENTED: YES
TESTED: YES
PR #118: MERGED
DEPLOYED: YES / LIVE
ROOT -> COCKPIT HOME: VERIFIED
MODULE-01 SIDEBAR CLEANUP: VERIFIED
CAL.COM REASON BRIDGE: CLOSED / DAILY 04:00 UTC
```

## Historical Home v1 follow-up

HOLD the completed Home v1 release. The Cal.com → Clinical Calendar bridge is now CLOSED on a daily 04:00 UTC cadence; do not reopen that integration without new production failure evidence. The active Cockpit follow-up is the separately bounded Pending Surgery Queue v1.


---

# Clinical Calendar reason bridge — bounded follow-up slice

> **STATUS:** CLOSED / DAILY RUNTIME + REPO CADENCE ALIGNED.
> **Workstream:** Cal.com visit-reason normalization into the Clinical Calendar.
> **Branch:** `feat/calendar-reason-snapshot-v1-2026-09-27`.
> **Base main:** `a0cf912b0fcb5caf54f4a48f8f8a698908fbb0a4`.
> **Product-owner resume:** 2026-09-27 — continue the exact follow-up slice named above after recovery from the interrupted PR1 Βελτιώσεις Eval conversation.
> **Root writer lock:** unchanged; PR-1 Heidi-first transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this branch is limited to the Clinical Calendar integration contract and its tests/docs. It does not mutate PR-1 transcript runtime.

## Accepted boundary

- Cal.com remains the appointment-source truth for the feed.
- Reception/Digital Secretary remains the external integration owner; no Cal API credential is added to this repository.
- Clinical Calendar remains the normalized clinical consumer.
- This slice is read-only with respect to bookings: it must not create, cancel, reschedule or authorize appointments.
- Reception reason-triage / availability / booking semantics are explicitly out of scope.
- The existing Clinical Calendar classifier is reused; no second semantic reason owner is introduced.
- Snapshot reconciliation must remove stale source rows inside the declared source/time window so cancellations/reschedules do not remain as phantom appointments.
- Raw provider payloads are not persisted; only the existing normalized appointment fields are stored.

## Current implementation order

1. **DONE / MERGED / LIVE:** consumer snapshot endpoint and reconciliation contract via PR #119.
2. **DONE:** initial producer PR #152 merged after exact-head compatibility PASS.
3. **DONE / SOURCE-CORRECTED:** provider evidence showed no dedicated visit-reason `bookingFieldsResponses` field on current Limassol/Evrychou Cal.com event types. The Digital Secretary's own booking schema/create path and provider logs establish `metadata.notes` as the Secretary-created booking reason source.
4. **DONE / MERGED / LIVE:** producer correction PR #153 binds the clinical feed strictly to `metadata.notes`; missing notes stay blank and generic reason-like booking-field fallbacks remain forbidden.
5. **DONE / CONFIGURED:** consumer shared ingest authentication and producer snapshot URL/shared ingest key are live. No Cal.com custom-field configuration was added.
6. **DONE / PRODUCTION EVIDENCE:** lawful daily runs on 28–30/9 reached the bridge but exposed a 20-second free-service cold-start timeout, not an auth/provider-field defect.
7. **DONE / CORRECTED / LIVE:** Backend PR #154 raised only the Clinical Calendar delivery timeout to 90 seconds and aligned `render.yaml` to the final daily `0 4 * * *` cadence. The existing Render cron is daily and no second cron exists.
8. **CLOSED:** the next normal daily run is post-close monitoring only and may reopen the slice only on new failure evidence.

## Recovery checkpoint

```text
RECOVERY CHECKPOINT ID: COCKPIT-CALENDAR-REASON-BRIDGE-20260927-A
STATUS: CLOSED / DAILY 04:00 UTC
IMPLEMENTATION COMMITS:
- 20e5687fc4283318966fca7e680b3e62eaede43f — snapshot reconciliation endpoint + shared import path
- 83da6339c8e4c4841e910a8e6135a91bf00380e5 — canonical source / exact window validation
- 426a4037687d83179d8df5639957f621a579cfa2 — focused snapshot tests
- 3bfba4c9d13b0a4a955ed3d470e80115c75c7d84 — current snapshot/reason integration contract
- c07daaf75359b071ec2f5b51d5864127965d122a — path-scoped Clinical Calendar CI gate
- c28ba8c5901e43801ab5998781b45521dd3495e5 — DST/cardinality acceptance correction
- cb0752d770d2afc2b6828c6fa4e7c48f25c12754 — deterministic compatibility boundary tests
- bbd5c7bca0fd109797fac5a03c15a51f0f53af21 — schema contract alignment / final reviewed consumer head
VERIFICATION:
- Clinical Calendar snapshot workflow run 36308943111 / job 108590995955: PASS
- Python syntax: PASS
- deterministic Clinical Calendar tests: 12 passed in 1.29s
- diff hygiene: PASS
- Canonical impact guard run 36308943000: PASS
PR #119: MERGED — Clinical Calendar: add source snapshot reconciliation
PR URL: https://github.com/athpapachr-cmd/osteoporosis/pull/119
EVIDENCE HEAD: bbd5c7bca0fd109797fac5a03c15a51f0f53af21
MERGE COMMIT: 08251120a1743a1ac629e38626d04747a8fda1e9
PRODUCER INITIAL MERGE: PR #152 / 4923cc83a1a57a844be5c713b83af125f7c86f61
PROVIDER-EVIDENCE CORRECTION: PR #153 / d99738940ed439fedbcadde100c3993177aa556d / workflow 36310831720 / 16 tests PASS
PROVIDER REASON SOURCE: metadata.notes for Secretary-created bookings; no current custom reason-field slug observed on event types 341357/341358
CONSUMER AUTH DEPLOY: dep-daseid97lnhs738rkd1g / LIVE
PRODUCER ACTIVATION DEPLOY: dep-dasejdgjo6nc73bf2keg / LIVE
POST-ACTIVATION MANUAL SYNC: 0
LAST SAFE RESUME POINT: consumer and corrected producer are live and mutually configured; first real snapshot delivery has not yet been observed and must not be inferred from configuration alone
EXACT NEXT ACTION: verify the first existing scheduled /admin/trigger-sync produces clinical_calendar configured=true / attempted=true / sent=true; then reconcile the separate cron cadence drift
FORBIDDEN ON RESUME: PR-1 mutation, Reception reason/availability/booking behavior change, second sync cron, guessed Cal.com reason field, phone/SIP or booking/business mutation
```


---

# Pending Surgery Queue — bounded Cockpit follow-up slice

> **STATUS:** V1 MERGED / LIVE.
> **Workstream:** global Clinical Excellence Cockpit / pending surgery coordination.
> **Branch:** `feat/cockpit-surgery-queue-v1-2026-09-30`.
> **Base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Product-owner direction:** 2026-09-30 — add a pending surgery list to the dashboard with patient identity/contact data, procedure/laterality, manual ordering/sorting and surgery date.
> **Root writer lock:** unchanged; this workstream uses `cockpit/CURRENT.md` under AGENTS §4.2.

## V1 product contract

- Global Cockpit feature; not Osteoporosis-specific and not Reception booking logic.
- Reuse the existing protected Clinical Auth browser session and the existing PostgreSQL engine.
- Patient identity stays server-side; no localStorage/sessionStorage persistence.
- Pending list fields:
  - ονοματεπώνυμο;
  - ΑΔΤ / stable patient identity;
  - ημερομηνία γέννησης;
  - τύπος επέμβασης;
  - πλευρά;
  - τηλέφωνο;
  - ημερομηνία χειρουργείου.
- Manual persisted queue order supports move up/down.
- Table supports non-persistent sorting by displayed columns without silently changing the manual queue order.
- Marking a case completed removes it from the default pending list but preserves the record.
- Surgery Queue stores the requested identity/contact snapshot in its own protected server-side table. It does **not** assume generic `clinical_patients.patient_id` equals ΑΔΤ and does not silently mutate the longitudinal patient registry. Explicit future patient-linking is a separate product slice.
- Synthetic test data only; no identifiable patient data in repository or CI.

## Implementation checkpoint

- protected `clinical_surgery_queue.py` API with PostgreSQL persistence;
- protected queue-owned identity/contact fields; no implicit `clinical_patients` mutation or ΑΔΤ→generic patient-id assumption;
- create/update/list/move/complete lifecycle;
- Cockpit table with sortable columns, inline surgery date, ↑/↓ manual order, edit and complete controls;
- no browser storage of patient identity;
- focused synthetic API/UI tests + dedicated CI workflow;
- no Reception/Cal.com/booking behavior change.

## Release evidence and source-driven correction

- PR #124: `Cockpit: add protected pending surgery queue`.
- exact implementation head before this checkpoint: `a4feb0aeec452677b2b07cc283bd5c16aa2e0ec4`.
- focused workflow `36714622069`: PASS.
- Python syntax: PASS.
- Cockpit JavaScript syntax: PASS.
- surgery queue API/UI tests: **10 passed in 0.64s**.
- diff hygiene: PASS.
- Canonical impact guard `36714622173`: PASS.
- inherited Cockpit Home, Clinical Documents P1/P2, Learning L1, CU-1, Physio integration and G3 regression gates: PASS.
- **Source-driven correction after that head:** existing `clinical_patients.patient_id` is generic, not canonically ΑΔΤ. The queue contract was corrected to keep identity fields in the protected surgery row and avoid hidden mutation/duplication of longitudinal patient identity.
- corrected final head: `14707fc79e4941f65b1e180a4ff2510b420420f2`.
- corrected focused surgery workflow `36715409386`: PASS / **10 tests in 0.81s** / Python + JavaScript syntax + diff hygiene PASS.
- corrected Canonical Impact `36715409552`: PASS.
- corrected Cockpit Home, Clinical Documents P1/P2, Learning L1, CU-1, G3 and Physio Cockpit browser/integration gates: PASS.
- PR #124 merge: `3ab6a0fe0dc83047ab37925a8dddde2c032ba8fe`.
- initial merge deploy `dep-daug4j49v7es73bjk910` was superseded by the required post-merge checkpoint deploy;
- final Render deploy `dep-daug4tnf3r2c73fr2h30`: **LIVE** / source `205ca3369abf5b0dcec5a8db30b945e1748bce98`.

## Release state

- PR #124: MERGED.
- merge commit: `3ab6a0fe0dc83047ab37925a8dddde2c032ba8fe`.
- final deployed source/checkpoint: `205ca3369abf5b0dcec5a8db30b945e1748bce98`.
- Render deploy: `dep-daug4tnf3r2c73fr2h30` / **LIVE**.
- focused surgery queue evidence: 10 tests PASS.
- inherited Cockpit/Clinical Documents/Learning/CU-1/G3/Physio browser integration gates: PASS.
- patient identity remains protected in the surgery queue's server-side table; no browser storage and no implicit mutation of `clinical_patients`.

## Historical Pending Surgery Queue follow-up

Use the live Pending Surgery Queue in the Cockpit. Further additions (for example surgeon, hospital, insurer/GESY authorization, pre-op checklist, priority reason, or archive/history view) are separate product increments, not blockers for v1.


## Pending Surgery Queue v1.1 — delete action

> **STATUS:** MERGED / LIVE.

- PR #126 merged as `3ed0c9a52083eed6151c3e880eac2aad71d0cf4c`.
- Adds protected pending-row deletion with confirmation and soft-delete audit semantics.
- The visible control is now refined to a compact trash icon in the next combined UX release.

## Clinical Calendar manual classification + Surgery Queue icon — 2026-09-30

> **STATUS:** MERGED / AUTO-DEPLOY PENDING.

- PR #127 merged as `fb5e8596334372d7f5e9bc3ae6ccd8dfb4be43ea`.
- `osteoporosis_unspecified` remains the safe automatic result when the reason establishes osteoporosis but duration does not establish first/review.
- Appointment cards expose explicit choices: Πρώτη επίσκεψη / Επανέλεγχος / Prolia / Aclasta.
- Manual classification is stored in a separate protected consumer-side override table and survives future Cal.com snapshots.
- Selecting «Αυτόματο» clears the override and returns to classifier-derived behavior.
- No manual classification writes back to Cal.com or Reception.
- Surgery Queue delete is displayed as a compact trash icon while preserving confirmation + soft-delete behavior.
- Canonical impact, Clinical Calendar snapshot, Surgery Queue and Cockpit Home gates: PASS.

**As-of 2026-09-30 follow-up:** observe the automatic Render deploy and then use the live controls. Verify deploy state afresh before making a current-state claim.
