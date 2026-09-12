# Knee OA v1 — post-smoke v4 release/deployment closeout

> **Refinement PR:** `#89`
> **Exact tested PR head:** `602740d12da0de8ad9a6134f8adf48f4b6f7c691`
> **Squash-merge / runtime refinement SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`
> **Render service:** `osteoporosis` / `srv-d5qfk31r0fns73di596g`
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug`
> **Deploy source commit:** `bf527e3836a18491b2758fd293b42f82e0924382`
> **Deploy status:** `live`
> **Deploy finished:** 2026-09-12 10:58:27 UTC
> **External live boundary-smoke:** run `34689920602` — SUCCESS
> **Exact-head v4 artifact:** `10297161518`
> **Artifact digest:** `sha256:3ca3d329148752adc2764781e48e6ca189359b6fbd3f69c423b4d51770e06b0b`

## 1. What changed

This bounded post-release refinement addressed Product Owner production-use findings without expanding the clinical model.

Released v4 behavior:

- compact clinical-picture surface with four equal desktop controls and a mobile 2×2 grid;
- `Πόνος`, `Δυσκαμψία`, `Αδυναμία` and `Λειτουργικότητα` each open a focused contextual sheet;
- re-tapping an active parent reopens its sheet; removal is explicit and clears dependent qualifier meaning;
- deterministic Greek referral prose no longer inserts unsupported `κυρίως` for unranked multiple pain locations;
- explicit quadriceps examination remains an examination finding rather than collapsing to generic weakness;
- passive extension deficit remains separate from subjective stiffness;
- the final treatment sentence is continuous, human Greek while preserving physiotherapist autonomy;
- mobile clinical controls retain at least 44 px target height and allow wrapping under large text enlargement, preventing horizontal overflow.

No second diagnosis, evidence update, jurisdiction change, patient persistence, clinical-rule expansion or autonomous recommendation change was introduced.

## 2. Pre-merge evidence

Exact PR head `602740d12da0de8ad9a6134f8adf48f4b6f7c691` completed all relevant PR gates successfully:

- `Physio Knee OA clinical-sheet v4 gate` run `34689630421` — SUCCESS;
- `Physio Knee OA prototype gate` run `34689630422` — SUCCESS;
- `Physio Knee OA Cockpit integration gate` run `34689630419` — SUCCESS;
- `CU-1 focused tests` run `34689630432` — SUCCESS;
- `Physio Knee OA evidence design gate` run `34689630440` — SUCCESS.

The final gate cycle also caught and corrected inherited test-path drift plus two real accessibility defects before merge: 42 px mobile targets and large-text horizontal overflow.

PR `#89` was mergeable, had zero unresolved review threads and was `0` commits behind `main` immediately before squash merge.

## 3. Deployment verification

Render auto-deploy was enabled on `main`; no duplicate manual deploy was triggered.

Deployment `dep-daiivp0jo6nc73bl6pug` became `live` with exact source commit:

`bf527e3836a18491b2758fd293b42f82e0924382`

This establishes `MERGED` and `DEPLOYED` for the v4 runtime refinement.

## 4. External production boundary smoke

A temporary no-secret GitHub Actions smoke workflow was run from a branch created directly from the runtime merge SHA and removed after completion.

Run `34689920602` completed `SUCCESS` and recorded:

```text
200 /static/clinic-utilities/physio-referral/index.html
200 /static/clinic-utilities/physio-referral/production-env.js
200 /static/clinic-utilities/physio-referral/production-finalize.js
200 /static/clinic-utilities/physio-referral/product-more-v3.js
200 /static/clinic-utilities/physio-referral/product-clinical-sheet-v4.js
200 /static/clinic-utilities/physio-referral/product-clinical-sheet-v4.css
401 /clinical/clinic-utilities/physio-referral
401 /clinical/clinic-utilities/physio-referral/api/product/bootstrap
```

The smoke also asserted that the deployed production loader references the v4 JS/CSS, the v4 surface is present, the live CSS contains the 44 px target and `white-space:normal;overflow-wrap:anywhere` mobile reflow correction, and production transport sets `synthetic_only = false`.

No clinical key, login password, session cookie, patient identifier or patient data was supplied.

## 5. Lifecycle precision

```text
V4 MERGED                                           YES
V4 DEPLOYED                                         YES
V4 PUBLIC-ASSET LIVE SMOKE                          PASS
V4 UNAUTHENTICATED AUTH-BOUNDARY LIVE SMOKE         PASS
FULL AUTHENTICATED LIVE END-TO-END SMOKE            NOT PERFORMED BY THIS CLOSEOUT
PILOT-VALIDATED                                     NO
RECEIVER-VALIDATED                                  NO
COMMERCIAL/PAID VALIDATED                           NO
SECOND DIAGNOSIS                                    NOT AUTHORIZED / NOT SELECTED
```

The original v1 release closeout remains historical evidence and is not rewritten by this record.
