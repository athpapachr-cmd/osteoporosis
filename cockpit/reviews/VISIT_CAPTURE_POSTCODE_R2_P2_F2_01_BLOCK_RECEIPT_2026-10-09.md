# Visit Capture post-code R2 — received P2-F2-01 BLOCK

> **Receipt date:** 2026-10-09 Asia/Nicosia.
> **Origin:** Independent review result supplied verbatim by the Product Owner in the coordinator conversation.
> **Status:** BLOCK / COMPLETE / P0:P1:P2 = 0:0:1 for the original post-code runtime/test target. This is a durable receipt, not a new reviewer verdict or a rerun.
> **Repository:** `athpapachr-cmd/osteoporosis`
> **Reviewed branch:** `feat/cockpit-visit-capture-v1-2026-10-08`
> **Reviewed current head:** `3e5f3ea50a0762fa7d519fe3b2928ea752374caa`
> **Original runtime/test head:** `4646c923c728fd8613632db42e919687fc945504`
> **Reviewed request blob:** `85d4a7ad219e4c078f5e79252f571328d58570d6`

## Sole material finding

**P2-F2-01 — unbounded pending external dependency.**

`VisitCapturePendingCandidate.external_dependency: Optional[Dict[str, Any]]` accepted and persisted arbitrary nested fields, contrary to the bounded candidate and no-raw-source-persistence contract. The correction requested by the independent reviewer was to use a strict declared dependency model with field lengths and `extra="forbid"`, plus targeted rejection/persistence evidence.

Other dispositions from the independent handback: F1 PASS; F3 PASS; F4 PASS; F5 PASS subject to the F2 dependency correction; F6 PASS. F2 otherwise passed.

## Authorized limited correction

After the Product Owner requested a simple explanation and subsequently instructed **«Προχωράμε»**, correction authority was confined to this one field and its affected persistence/tests. No general code redesign or wide review was authorized.

Implementation correction:
- `clinical_data.py`: strict model, explicit persistence serialization, typed candidate/response.
- `test_visit_capture.py`: accepted-model roundtrip, rejection of undeclared nested fields and oversized values, evidence of no rejected persistence.

The correction must be evaluated by **one fresh independent F2 + affected-persistence closure review only**, reusing the other previously settled R2 findings. No merge, deploy, provider access or patient-data activation follows from the correction or the prior review.
