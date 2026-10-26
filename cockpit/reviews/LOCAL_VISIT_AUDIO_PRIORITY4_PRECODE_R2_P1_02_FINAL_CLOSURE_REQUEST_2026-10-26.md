# Priority 4 local visit audio — focused PRE-CODE R2 P1-02 final closure

**Mode:** one independent **READ-ONLY** P1-02 delta + affected closure, then STOP. Reuse P1-01/P1-03/P1-04/P1-05 CLOSED and prior affected-cumulative PASS outside P1-02. No implementation, install, real data, merge, deploy or external write.

## Exact target

- Prior corrected design: `8c6712b56ef55731da0f4130dd5b27e053bc1894`, blob `6597c1e4b43cc36cae05e55c53b992557615c16d`.
- Prior closure: `BLOCK / COMPLETE_FOR_DECLARED_SCOPE / 0:1:0`; only P1-02 remained open.
- P1-02 corrected design: commit `9027d07152729d83972ca725feb533c0e987205d`.
- Corrected design blob: `644c36a8b557be46e4165a6cce9c63fa4ed54290`.

## Required question

Does section 6 now close the reachable patient-version race with the smallest implementable design: a protected atomic renewal/CAS endpoint receives the exact prior context ID plus patient/client IDs; loads and locks that prior row even when expired; verifies stored session/patient; locks the current patient; compares prior stored and current `patient_updated_at`; and only after all checks deletes old contexts and issues a fresh lease? Confirm missing/mismatched/mutated state returns 409 without replacement, browser mismatch/failure destroys the candidate, ordinary context-create is not renewal, no schema change is needed, and the focused tests cover expiry plus concurrent mutation.

Return `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, P1-02 `CLOSED | OPEN | UNKNOWN`, exact commit/blob, any directly affected new issue and smallest fix, and the narrow implementation/test boundary if PASS. **STOP READ-ONLY.**

A PASS authorizes only bounded implementation; it does not authorize real patient use, merge, deploy or clinical writes.
