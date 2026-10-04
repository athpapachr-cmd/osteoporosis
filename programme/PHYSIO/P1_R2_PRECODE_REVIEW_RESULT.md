# PHYSIO P1 R2 Pre-Code Review — Terminal Result

> **TASK:** `PHYSIO-P1-R2-STRUCTURAL-IA-SAFETY-PRECODE-REVIEW`
> **DATE:** 2026-10-04 Asia/Nicosia.
> **MODE:** independent pre-code R2 review.
> **RUNTIME MUTATION:** none.
> **ROOT CANONICAL MUTATION:** none.
> **SOURCE NOTE:** the independent reviewer worked in a separate local checkout. This repository artifact records the terminal handback supplied by the Product Owner; it does not claim byte-for-byte archival of the reviewer's local file.

---

## R2-A — Information Architecture

**VERDICT: BLOCK**

The reviewer inventoried all 67 IDs in the current product contract against current locations, target owner, action and reason.

Three material design decisions remained unresolved:

1. duplicate functional concepts represented both as `finding` and `functional_impairment`;
2. overlapping pain-location representations plus the semantic distinction among reported swelling, examined effusion and hot/swollen-joint review context;
3. contract-supported but UI-hidden `walking_aid_assessment_and_training`: expose or remove from active product scope.

R2-A therefore did not permit implementation.

---

## R2-B — Safety / review-cue semantics

**VERDICT: PASS for the bounded pre-code semantic design**

The review established bounded semantics for:
- acute inflammatory/infective-joint review cue;
- SIFK / alternative-subchondral-pathology review cue.

For each, the reviewer defined:
- eligible observations;
- minimum combinations;
- non-triggering combinations;
- exact clinician-facing wording;
- explicit clinician disposition: continue routine referral or defer after reassessment.

Permanent semantics:
```text
REVIEW CUE != DIAGNOSIS
REVIEW CUE != AUTOMATIC IMAGING
REVIEW CUE != CU-1 SAFETY BLOCK
SINGLE NONSPECIFIC FINDING != AUTOMATIC ALERT
```

The review treats the combinations as conservative reassessment rules, not validated diagnostic criteria.

Existing explicit CU-1 unresolved safety concerns remain an independent fail-closed route.

---

## CU-1 dependency result

**NO SHARED CU-1 CHANGE REQUIRED**

The proposed review cues may reuse disposition language while remaining a Knee-OA product-local review layer.

The shared CU-1 safety path remains separate and explicit.

---

## Implementation readiness

`NO` at the reviewed target because R2-A remained BLOCK.

Required closure path under `PROCEDURES.md` P5:

```text
bounded correction of the three R2-A findings
→ one independent delta + affected-cumulative R2-A closure review
→ terminal PASS/BLOCK
→ STOP
```

R2-B is settled and must not be reopened without new source-proven material risk.
