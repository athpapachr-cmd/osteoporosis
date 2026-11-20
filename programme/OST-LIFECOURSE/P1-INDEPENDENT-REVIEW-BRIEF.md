# OST-LIFECOURSE-P1 — Independent read-only review brief

> **TASK TYPE:** fresh independent design/semantic-ownership review.
> **MODE:** READ-ONLY / EXACT-CURRENT-SOURCE / PASS OR BLOCK / STOP.
> **NOT:** implementation, coordinator reconciliation, schema design, migration or Product Constitution work.

## Fresh bootstrap

Fresh-verify `athpapachr-cmd/osteoporosis/main`.
Read the six active canonicals in AGENTS order.

Then consume fully from draft PR #121 / branch `docs/ost-programme-lifecourse-bootstrap-2026-09-27`:

- `programme/PRODUCT-CONSTITUTION-V0.2.md`
- `programme/WORKSTREAM-REGISTRY.md`
- `programme/OST-LIFECOURSE/PROJECT-INDEX.md`
- `programme/OST-LIFECOURSE/CURRENT.md`
- `programme/OST-LIFECOURSE/PHASE0-PROGRAMME-RECONCILIATION.md`
- `programme/OST-LIFECOURSE/P1-SEMANTIC-OWNERSHIP-BRIEF.md`
- `programme/OST-LIFECOURSE/P1-PROGRAMME-RECONCILIATION.md`

Use the returned P1 handback as evidence, but fresh-inspect source before accepting any source-dependent conclusion.

## Exact review objective

Independently verify the P1 semantic-ownership reconciliation across these seven domains:

1. fracture / fragility;
2. DXA;
3. labs;
4. Step-4 treatment / administrations / tasks;
5. LongitudinalGuidanceProjectionV1;
6. EncounterContext contract vs runtime;
7. GuidanceRule / TherapyMilestone registries vs executable runtime.

## Required review questions

### A. Ownership correctness
Does the proposed authoritative-vs-derived split match current source/runtime truth?

### B. Duplicate-mechanism prevention
Has P1 correctly avoided creating a new patient/event/timeline/treatment/obligation/evidence stack?

### C. Safety/data integrity
Are the listed drift/duplication findings materially accurate and are their classifications reasonable?

### D. S1 separation
Is fracture/fragility runtime correction still correctly separated from P1 design ownership work?

### E. Contract/runtime claims
Are EncounterContext and G-2 registry/executor conclusions supported by current source?

### F. Parked questions
Have Q1–Q12 remained genuinely parked, with no hidden resolution smuggled into P1?

### G. Scope discipline
Does P1 remain documentation/design reconciliation rather than migration/implementation architecture?

## Prohibited actions

- no code changes;
- no schema/database/UI changes;
- no canonical mutation;
- no PR-1 changes;
- no S1 implementation;
- no P2 target architecture;
- no new review referral;
- no merge/deploy/smoke;
- no resolution of Q1–Q12.

## Required output

Return:

REVIEW SOURCE IDENTITY
→ VERIFIED OWNERSHIP FINDINGS
→ VERIFIED DUPLICATES / DRIFT
→ MATERIAL DISAGREEMENTS
→ SAFETY / DATA-INTEGRITY DISPOSITION
→ S1 SEPARATION CHECK
→ Q1–Q12 PARKING CHECK
→ P1 VERDICT = PASS | BLOCK
→ REGISTRY SYNC
→ STOP

One bounded review only. Do not create a review-of-review chain.