# Visit Capture Dia/Heidi/GESY → Cockpit — independent exact-head R2 post-code fidelity request

> **DATE:** 2026-10-08 Asia/Nicosia.
> **STATUS:** PREPARED / NOT DISPATCHED / READ-ONLY / NO VERDICT.
> **TIER:** R2 exact-head implementation-fidelity review.
> **REPOSITORY:** `athpapachr-cmd/osteoporosis`.
> **IMPLEMENTATION BRANCH:** `feat/cockpit-visit-capture-v1-2026-10-08`.
> **REVIEWED DESIGN BASE HEAD:** `5bc2a505758de6b24e82667bb84580d144f96b9c`.
> **FROZEN DELTA DESIGN BLOB:** `7e19ce29a3c33a0d06e6c0ec485952f41c0ecf72`.
> **PRE-CODE R2:** PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0; durable receipt `cockpit/reviews/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_R2_PASS_RECEIPT_2026-10-08.md`.
> **RUNTIME/TEST HEAD:** `4646c923c728fd8613632db42e919687fc945504`.
> **FOCUSED CI:** GitHub Actions run `37811359040` SUCCESS.
> **AUTHOR STATE:** writer RELEASED / implementation frozen for review / NOT MERGED / NOT DEPLOYED / NO LIVE PROVIDER OR PATIENT-DATA ACTIVATION.

## Exact runtime/test blobs

- `clinical_data.py` — `20b743d7f9c2b5d3a7c2afb446abd7549cd54c78`
- `static/cockpit/index.html` — `21b3c02fd5e0e6ef877ff3183a9c4d2c2afd817c`
- `static/cockpit/visit-capture/index.html` — `9aacd241d802d662d767040f2fc8b9e4e4d7e96e`
- `static/cockpit/visit-capture/app.js` — `4673b9b93db1ece7f4857d7cf44d1683ff2ab112`
- `static/cockpit/visit-capture/styles.css` — `85441f2550e9cdc10136a5b294a7ccf9a9c75e4a`
- `test_visit_capture.py` — `77b10e685042af2cfd6c6b358666c31d2fc49126`
- `test_visit_capture_ui.py` — `3e612e7ca59acbc1dc36dc71eab44f456a799e06`
- `.github/workflows/visit-capture-tests.yml` — `260dcdb3f5d9f6d39ba5caecd3a7755cfdd97904`

Commits after the runtime/test head are checkpoint documentation only. Fresh-verify the current branch head and prove that any drift from `4646c923…` does not alter runtime/test bytes before relying on the CI result.

## Scope

Review whether the implementation faithfully satisfies the already-passed delta design. Do not redesign the feature, reopen parent Visit Brief/Clinical Inbox Q1–Q6, or expand to live Dia/Heidi/GESY/Gmail/Zadarma/provider qualification.

No implementation, correction, merge, deploy, live source access, patient-data use or provider action is authorized.

## Mandatory bootstrap

Fresh-verify remote main and implementation branch. Read/apply the current six canonicals in required order, then `PROCEDURES.md`, `cockpit/CURRENT.md`, Product Constitution, frozen Visit Capture delta, pre-code PASS receipt and this request.

Inspect only the changed runtime/test files above plus the minimum existing source needed to verify reuse and regressions.

## F1 — Protected owner + identity/context fidelity

Verify that:
- destination patient is the existing internal `clinical_patients.patient_id`;
- Visit Capture does not create a competing patient/encounter truth store;
- candidate schema cannot carry/override destination patient identity;
- server-issued capture context is patient-bound;
- issuing a new context for the same browser session invalidates the prior context;
- patient version/context change fails closed;
- stale A→B synthetic case is source-proven.

PASS only if wrong-patient stale writes are prevented server-side, not merely by UI behavior.

## F2 — Candidate/provenance/comparison fidelity

Verify:
- VisitCaptureCandidateV1 is bounded and rejects unexpected fields;
- source roles remain `heidi_today | gesy_today | gesy_previous | clinician_edit`;
- GESY coding is a separate incomplete coding layer;
- provenance references must bind to declared source bindings;
- `what_changed` cannot exist when comparison basis is unavailable;
- source identity uncertainty/conflict can preview but cannot Save as normal truth;
- partial medication fields remain null rather than inferred.

Do not evaluate clinical correctness of example content.

## F3 — One encounter / three projections / one Save

Verify:
- Snapshot, Visit Brief and Encounter Detail are deterministic projections of one normalized candidate;
- projection prose is not stored as three independent authorities;
- preview is non-persistent/transient;
- UI automatically previews candidate content;
- exactly one explicit final Save creates the protected record;
- Home entry is bounded and does not alter Calendar/Reception behavior.

## F4 — Signed encounter immutability + later-event separation

Verify:
- Visit Capture Save creates a completed/signed marker inside the existing encounter owner;
- generic encounter PUT cannot silently mutate a signed Visit Capture encounter;
- implementation does not falsely claim immutable amendment history;
- later lab/other longitudinal activity does not rewrite the signed encounter;
- no autonomous amendment feature was added outside reviewed scope.

A UI-only edit lock is insufficient; server enforcement is required.

## F5 — First-class pending boundary

Verify:
- pending work is stored separately from encounter prose with stable ID;
- source encounter and patient are bound;
- responsible role is separate from external dependency;
- status/review/provenance are retained;
- no existing embedded `step4.tasks` is incorrectly treated as the new global pending owner;
- this slice does not auto-resolve pending items from later results.

## F6 — Evidence + bounded regression

Verify the exact CI run `37811359040` against runtime/test head `4646c923…`.

It must show success for:
- Python syntax;
- Visit Capture JavaScript syntax;
- focused Visit Capture tests;
- affected Cockpit Home tests;
- diff hygiene.

Map the eight frozen synthetic acceptance cases to actual tests. Confirm no live Dia/Heidi/GESY/Gmail/Zadarma access, no patient data and no deploy occurred.

Use existing evidence; do not rerun broad suites merely to review.

## Severity discipline

P0: credible wrong-patient write, protected-data exposure or autonomous unsafe clinical write.

P1: material clinical-state corruption, missing server-side signoff/immutability guard, competing truth owner, or tested implementation materially diverges from reviewed contract.

P2: bounded implementation/UX contract defect that should be corrected before merge.

Do not create findings for style, naming, speculative future completeness or live-provider gates intentionally left open.

## Stop rule

- All F1–F6 disposed with sufficient evidence → PASS / COMPLETE_FOR_DECLARED_SCOPE and STOP.
- Material reachable defect → BLOCK with smallest correction, finish only directly affected trace, STOP.
- Required evidence unavailable → UNKNOWN / PARTIAL with exact gap, STOP.

A BLOCK receives one bounded correction and then one affected closure review only. No reassurance review.

## Required handback

Return:

```text
REVIEW =
VISIT CAPTURE DIA/HEIDI/GESY → COCKPIT POST-CODE R2

REMOTE MAIN =
<fresh SHA>

TARGET =
<branch + current head + runtime/test head + exact blobs>

VERDICT =
PASS / BLOCK / UNKNOWN

COVERAGE =
COMPLETE / PARTIAL

P0:P1:P2 =
x:y:z

F1 =
...

F2 =
...

F3 =
...

F4 =
...

F5 =
...

F6 =
...

MATERIAL FINDINGS =
...

IMPLEMENTATION FIDELITY =
<concise result>

REMAINING LIVE-ACTIVATION GATES =
<finite list>

STOP =
why the review terminates here
```

A PASS is implementation-fidelity disposition only. It does not grant merge, deploy, live external-source access or patient-data activation.
