# Priority 4 local visit audio — first PRE-CODE R2 correction-closure receipt

**Verdict received:** `BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:0`.

**Reviewed:** corrected design commit `8c6712b56ef55731da0f4130dd5b27e053bc1894`; blob `6597c1e4b43cc36cae05e55c53b992557615c16d`; closure request commit `27fd19bc3b9e24c6fb31ea44b48f2be2a694f08f`.

- P1-01 FileVault: **CLOSED**.
- P1-02 context expiry: **OPEN**. Ordinary context creation receives only patient/client IDs and adopts the current patient version; it cannot prove the row was unchanged since the original context.
- P1-03 exact protocol/result: **CLOSED**.
- P1-04 semantic distinctions/mapping: **CLOSED**.
- P1-05 decoder: **CLOSED**.

No additional affected-cumulative material finding was reported. The smallest requested correction is a protected atomic renewal/CAS endpoint that loads the exact prior context even if expired, verifies session/patient, locks the patient, compares prior stored/current `patient_updated_at`, and only after success invalidates old contexts and issues a fresh one. Implementation remains unauthorized pending focused independent P1-02 closure.
