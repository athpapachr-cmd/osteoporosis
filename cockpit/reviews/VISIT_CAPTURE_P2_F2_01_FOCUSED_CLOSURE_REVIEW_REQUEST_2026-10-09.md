# Visit Capture — P2-F2-01 ONE independent focused closure review

> **STATUS:** PREPARED / NOT DISPATCHED / NO VERDICT.
> **DATE:** 2026-10-09 Asia/Nicosia.
> **MODE:** fresh independent READ-ONLY post-code correction closure, **only F2 + directly affected pending persistence**.
> **REPO:** `athpapachr-cmd/osteoporosis`
> **BRANCH:** `feat/cockpit-visit-capture-v1-2026-10-08`
> **ORIGINAL BLOCKED RUNTIME/TEST HEAD:** `4646c923c728fd8613632db42e919687fc945504`
> **CORRECTED RUNTIME/TEST HEAD:** `07f68a2adc4664a4ab050b3fa10a663de591b662`
> **CHANGED CODE BLOB:** `clinical_data.py` = `ea9e3d39c124136f4818094aec4086e49cbd0919`
> **CHANGED TEST BLOB:** `test_visit_capture.py` = `00adc97ec0c726cc4a063470cf7a77e6f134a711`
> **FOCUSED CI:** GitHub Actions `37963002895` SUCCESS at corrected runtime/test head.
> **RECEIVED ORIGINAL BLOCK:** `cockpit/reviews/VISIT_CAPTURE_POSTCODE_R2_P2_F2_01_BLOCK_RECEIPT_2026-10-09.md`, blob `b448d45ff213c282eac8e97db19da0f29007e062`.
> **FROZEN DELTA DESIGN:** `cockpit/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_2026-10-08.md`, blob `7e19ce29a3c33a0d06e6c0ec485952f41c0ecf72`.
> **ORIGINAL EXACT POST-CODE REQUEST:** `cockpit/reviews/VISIT_CAPTURE_DIA_HEIDI_GESY_POSTCODE_FIDELITY_REVIEW_REQUEST_2026-10-08.md`, blob `85d4a7ad219e4c078f5e79252f571328d58570d6`.

## Question being closed

The original independent post-code R2 was **BLOCK / COMPLETE / P0:P1:P2 = 0:0:1**. Sole finding:

**P2-F2-01:** `VisitCapturePendingCandidate.external_dependency` was an unbounded `Optional[Dict[str, Any]]`, permitting undeclared nested/raw source content to be accepted and persisted, contrary to the bounded candidate/data minimization contract.

Prior review dispositions: F1 PASS, F2 BLOCK only for P2-F2-01, F3 PASS, F4 PASS, F5 PASS subject to this dependency, F6 PASS. These dispositions are inherited; they are NOT reassigned to this reviewer.

## Exact correction

Author changed only:

1. `clinical_data.py`
   - introduced `VisitCaptureExternalDependency(BaseModel)` with `model_config = {"extra": "forbid"}`;
   - declared exactly `actor: str` (1–80 chars) and `condition: str` (1–240 chars);
   - typed candidate `external_dependency` as `Optional[VisitCaptureExternalDependency]`;
   - typed pending response to the same model;
   - explicitly serialized the validated model to a dictionary for `ClinicalPendingORM.external_dependency_json`.
2. `test_visit_capture.py`
   - tests accepted `actor/condition` preview, Save, encounter payload and pending-record roundtrip;
   - tests preview AND Save reject `raw_source_body` (unexpected nested field) with 422;
   - verifies rejected content creates neither encounter nor pending item;
   - tests oversized actor/condition rejected;
   - verifies a subsequent valid Save remains possible.

No other runtime, UI, test, provider, schema or source-integration change is part of this correction. The existing `clinical_pending_items` JSON column remains the rightful data owner.

## Mandatory evidence grounding

Fresh-verify remote main and branch head. Read six active canonicals per `AGENTS.md`, then `PROCEDURES.md` P5/P5.1, `cockpit/CURRENT.md`, frozen delta, original BLOCK receipt and this exact request. Verify original and corrected runtime/test source identities. Classify post-test branch drift; docs-only checkpoint/request commits do not invalidate unchanged runtime/test blobs.

Reuse original F1/F3/F4/F5/F6 findings and original test evidence. No broad repo tour or general new test suite.

### C1 — Contract closure

Does the new strict model prevent unexpected nested fields and oversized values entering `VisitCaptureCandidateV1` through `pending[].external_dependency`, while accepting exactly the declared legitimate shape and `null`?

### C2 — Affected persistence closure

Can any unvalidated dependency mapping still reach `ClinicalPendingORM.external_dependency_json` or the signed encounter candidate payload on the corrected Save path? Verify validated serialization, response typing, valid roundtrip, and rejected content not persisted. Confirm responsible role and external dependency remain distinct.

### C3 — Exact correction evidence

Verify corrected head `07f68a2…`, both changed blobs and CI run `37963002895` SUCCESS on that exact runtime/test head (Python syntax, JS syntax, focused tests, Cockpit Home regression, diff hygiene). Confirm no changed runtime/test files after that head. If a required check is unavailable, state the exact evidence gap, not a substitute PASS.

## STOP and authority

- C1–C3 PASS, original P2 closed, no new material directly affected defect → **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0 / ORIGINAL P2-F2-01 CLOSED / STOP**.
- Reachable material defect → BLOCK, smallest concrete correction, STOP.
- Essential evidence unavailable → UNKNOWN / PARTIAL, exact blocker, STOP.

Do not implement, edit, request a broader review, merge, deploy, access real patient data or activate Dia/Heidi/GESY/Gmail/Zadarma. This is the **one allowed affected closure review** under P5, not a re-review of prior PASS surfaces. A PASS does not grant merge/release/live activation.

## Required concise handback

```text
REVIEW = VISIT CAPTURE P2-F2-01 FOCUSED CLOSURE
REMOTE MAIN = <verified>
BRANCH/HEAD = <verified>
CORRECTED RUNTIME/TEST HEAD = <verified>
BLOBS/CI = <verified>
VERDICT = PASS / BLOCK / UNKNOWN
COVERAGE = COMPLETE / PARTIAL
P0:P1:P2 = x:y:z
C1 = PASS/BLOCK/UNKNOWN + evidence
C2 = PASS/BLOCK/UNKNOWN + evidence
C3 = PASS/BLOCK/UNKNOWN + evidence
ORIGINAL P2-F2-01 = CLOSED / OPEN / UNKNOWN
MATERIAL FINDINGS = only actual affected defects
STOP = why review ends; no merge/deploy/live authority
```
