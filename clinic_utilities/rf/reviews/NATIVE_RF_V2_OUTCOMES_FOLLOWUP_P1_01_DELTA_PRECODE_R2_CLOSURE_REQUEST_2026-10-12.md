# Native RF v2 outcomes/follow-up — P1-01 one independent pre-code delta closure (2026-10-12)

**MODE: fresh separate independent READ-ONLY DESIGN CORRECTION CLOSURE.** No implementation, merge, deploy, production changes or real clinical data.

Repository `athpapachr-cmd/osteoporosis`. Baseline verified main at correction: `6ee184f9c76bf45a6c43e1f9fae90d43c1dfde00`; **original reviewed design HEAD:** `1a45cdca31e2807882a1878f4c8f256069b5ead9`, **original design blob:** `72c7f6cdfe1ab20485bc19f405e4941f09bf59b5`.
**Corrected design:** `clinic_utilities/rf/RF_V2_OUTCOMES_FOLLOWUP_DESIGN_2026-10-12.md` on `design/rf-v2-outcomes-followup-corrections-20261012`. Fresh-resolve actual HEAD and exact changed design Git blob.

## One decision only

The original independent pre-code R2 returned **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:0**; R1/R2/R4/R5 were PASS. Sole material finding **P1-01**: a `papachristou` doctor key alone cannot attest that the issuer is this clinician because the current RF request model defaults omitted keys and the protected directory permits identity edits under a stable key. This could automatically impose personal follow-up for an RF prepared on behalf of another doctor.

Inspect the **delta** in the corrected design and **affected cumulative ownership contract**, not any previously settled RF source/PDF/JPEG/A2/drafts or general architecture. Treat pre-code acceptance as adequacy of the specification and synthetic test oracles, not proof of implemented code.

### Closure questions

- **C1 — Server-attested issuance provenance:** Explicit `doctor_key` presence is recorded separately from legacy omitted-key fallback at issue time; RF backend — not client-supplied flags — decides whether the immutable issued `doctor_snapshot` matches the trusted configured original clinic-owner identity (name + GeSY code) and stores a conservative verified status, without new per-doctor env variables.
- **C2 — Assignment authority:** On subsequent manual **Submitted** action, `mine_to_follow` auto-activates only when the issued record contains both attested `explicit` provenance and verified own issuer status. Missing, legacy, ambiguous/changed default profile and older records without attestation remain opt-in only. Colleague-reported rejection/on-request corrected RF never creates follow-up merely from data entry. No extra role engine or user workload.
- **C3 — Source consistency/scope:** Reuse `RFApplicationDraft` omitted-key behavior, server-side locked snapshot and editable `RFDoctorProfileORM` as inspected. Other RF R1/R2/R4/R5 PASS intact: issued ≠ submitted, reliable outcomes and counts, approved/rejected, actual scheduled/performed, immutable corrected applications, real patient identity, and no Home/Calendar writes. Smallest future synthetic proofs cover own explicit, colleague explicit, omitted legacy key, changed default profile identity with same key, older record without provenance, manual opt-in/out.
- **C4 — Exact source:** Fresh main/design branch identity, verified original vs corrected document delta, documentation only, all RF production code/source unchanged. One closure review, no review-of-review.

**Return:** `PRE-CODE R2 PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, explicit P1-01 CLOSED or smallest remaining material gap, exact verified design SHA, and **STOP**.

No code, tests executed, mutation, PR merge, Render deploy or real patient/provider records.