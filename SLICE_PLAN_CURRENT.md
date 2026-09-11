# SLICE_PLAN_CURRENT.md — Step 6A Knee-OA qualifier refinement

> **STATUS:** IMPLEMENTED / FOCUSED TECHNICAL GATE PASS / CLOSED.
> **Slice:** `CU1-PRODUCT-KNEE-OA-PROTOTYPE-V1-1-QUALIFIERS-20260911`.
> **Branch:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11`.
> **Parent closeout:** `c3f79a192a3fcac9fb11a5245df4e93312c4522a`.
> **Tested substantive head:** `243095ca9545bd2f96be8986520aeae8c3551c27`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** NONE.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Problem solved

The Step-5 routine phenotype was clinically too coarse in pain, weakness and stiffness. Step 6A adds depth through progressive disclosure rather than a permanently larger form.

Frozen UX rule for this candidate:

```text
broad first tap
→ reveal clinically meaningful qualifier only on demand
→ retain a compact clinical-summary line in the routine flow
```

## 2. Pain refinement

`Πόνος` remains one routine choice. Optional location supports:

```text
medial_joint_line
lateral_joint_line
anterior_peripatellar
pes_anserine_region
posterior
diffuse
```

Focal locations may coexist; `diffuse` is exclusive.

Hard rule:

```text
pes-anserine symptom/tenderness location != autonomous pes-anserine bursitis diagnosis
```

## 3. Weakness refinement

Generic weakness remains the product phenotype unless the clinician explicitly qualifies it.

```text
objective
quadriceps
visible_atrophy
atrophy_location: quadriceps | peri_knee_general
```

Explicit objective/quadriceps qualifiers map to the existing canonical CU-1 finding IDs. New qualifier state replaces an older weakness-specific finding only when explicitly selected; otherwise legacy explicit findings remain unchanged.

## 4. Stiffness refinement

```text
morning
after_inactivity
morning duration: ≤30′ | >30′
```

`>30′` creates a review clue linked to NICE NG226. It does not create an alternative diagnosis, does not itself block export, and does not automatically change/select treatment.

## 5. Examination / power-user layer

Remain outside the routine surface:

```text
extension lag
joint effusion
fixed flexion deformity + optional degrees
focal tenderness + optional anatomic location
```

Fixed flexion deformity is an examination finding, not a synonym for stiffness. An explicitly selected FFD can make the existing mobility suggestion eligible through documented clinical mapping; it still does not select mobility treatment.

## 6. Deterministic referral behavior

The frozen Step-3 renderer remains the base. Step-6A adds only bounded product-local specificity:

- pain localization;
- stiffness pattern/duration;
- atrophy qualifier;
- localized tenderness;
- explicit fixed flexion deformity examination prose.

If no new qualifier is selected, Step-5 exact outputs remain unchanged.

## 7. Acceptance obtained

Exact substantive gate:

```text
workflow  Physio Knee OA prototype gate
run       34627436841
head      243095ca9545bd2f96be8986520aeae8c3551c27
result    SUCCESS
```

Coverage at that head:

```text
existing backend/HTTP                         15 / 15 PASS
new qualifier projection                      8 / 8 PASS
frozen exact Greek outputs                    15 PASS within existing suite
existing Chromium                            12 / 12 PASS
new qualifier Chromium                        6 / 6 PASS
source-summary display                        54 positions
package dependency closure                    PASS
scope/syntax                                  PASS
```

The tests explicitly cover pes-anserine without auto-bursitis diagnosis, generic weakness without objective inference, explicit quadriceps + atrophy refinement, stiffness >30′ review without treatment/block, fixed flexion deformity distinct from stiffness, focal pes-anserine tenderness, parent-deselect cleanup and mobile reflow.

## 8. Explicit remaining acceptance

The technical gate is not product-owner acceptance or independent review. Still pending: real product-owner synthetic use, final Greek copy judgment, iPhone Safari/VoiceOver, complete accessibility/contrast audit, independent clinical/physio/UX/commercial review and source-to-claim audit.

## 9. Hold / next action

Return to Step-6 product-owner testing of this exact refined candidate. Do not add more fields or a second diagnosis until that testing demonstrates a concrete need.

No production integration, real patient data, persistence, public/LAN hosting, billing/auth, autonomous evidence update, PR/merge/deploy or production smoke is authorized.
