# SLICE_PLAN_CURRENT.md — Knee-OA post-review bounded amendments v1

> **STATUS:** IMPLEMENTED / FOCUSED TECHNICAL GATE PASS / PRODUCT-OWNER VISUAL REVIEW NEXT.
> **Slice:** `CU1-PRODUCT-KNEE-OA-POST-REVIEW-AMENDMENTS-V1-20260912`.
> **Branch:** `feat/physio-knee-oa-review-amendments-v1-2026-09-12`.
> **Parent synthesis head:** `554ecfa30bf8d0c04a19510a5db0276e6844edd5`.
> **Reviewed candidate ancestry:** `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Tested substantive amendment head:** `83a5acd4e5b25413708bbda1715c58b5c78bce08`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** NONE.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. General clinical-utility rule

For this product track:

```text
clinically interesting
!= workflow-useful
!= receiver-useful
!= worth adding
```

Any proposed field, qualifier, intervention, alert, evidence cue, output sentence or Product Owner/reviewer/author suggestion must identify the downstream management/safety/handoff/workflow value, intended consumer, expected accuracy at referral time and incremental UI/evidence-maintenance burden. If incremental value is uncertain, default to progressive disclosure, test, defer or remove rather than expand.

Hard governance rule:

```text
PRODUCT OWNER REQUEST
!= CLINICAL EVIDENCE
!= RECEIVER VALUE
!= IMPLEMENTATION AUTHORITY
```

## 2. Implemented bounded amendments

### Diagnosis / readiness

- explicit tap on the Knee-OA diagnosis is the clinician assertion in the standalone prototype;
- opening/reloading alone does not assert diagnosis;
- diagnosis then appears automatically in live referral;
- the checkbox-like diagnosis-confirmation presentation is removed from the intended UX;
- missing diagnosis and side are named specifically and highlighted locally/accessibly;
- only the next unresolved prerequisite is emphasized.

### Weakness

- generic weakness remains reported/contextual;
- the old ambiguous quadriceps localization value fails closed;
- `quadriceps_exam` explicitly means an examination finding before mapping to canonical `quadriceps_weakness`;
- no MRC/dynamometry workflow was added.

### FFD / passive extension deficit

- remains advanced / optional;
- expressed as passive extension deficit, distinct from active extension lag;
- if a degree is supplied it must be positive (`1–60°` in the prototype); unknown remains unknown;
- `0° FFD`, invented degree, permanence wording and unnecessary English parenthetical are rejected/removed;
- FFD can make an existing mobility suggestion eligible but never selects treatment.

### Functional detail

No new structured main-activity / functional-baseline field was added. Current functional categories and free-text note remain the available physician-side capture until receiving-physiotherapist testing proves additional receiver value.

### Referral / autonomy

- diagnosis+side-only input produces proportionally shorter output;
- patient-specific findings/function produce richer output;
- selected rehab is framed as **indicative priorities after physiotherapy assessment**, not technique/dose/progression mandate;
- selected structured clinical information is not silently discarded.

### Evidence / suggestion UX

- routine supported/conditional source detail moves behind deliberate deeper disclosure;
- mixed guidance keeps all material opposing/neutral source positions visible immediately;
- suggestion presentation is flattened while Add, evidence access, dismissal and stale-candidate guards remain;
- qualifier groups collapse predictably when another group is opened;
- parent deselection synchronizes visible and ARIA expansion state;
- pain location remains multiselect-capable;
- mobile manual reconciliation has a direct path.

## 3. Jurisdiction architecture

Reusable model:

```text
international clinical-evidence core
+
optional jurisdiction/local-system guidance overlay
```

Current prototype context:

```text
id: CY_GESY
label: Κύπρος · ΓεΣΥ
```

This label does not yet import any local recommendation or policy. Local guidance may influence item-level product behavior only after exact authoritative source audit and explicit review. Local feasibility/reimbursement/resource policy may not be represented as stronger clinical-efficacy evidence.

Future Greece/England profiles are dormant. No routine country selector and no device-location inference are added in this slice.

## 4. Technical acceptance obtained

The first run `34672583682` failed because the existing adapter regression asserted product output must remain byte-identical to Step-3 even though post-review prose was intentionally amended. The fix preserved the frozen Step-3 renderer regression and moved amended product behavior into separate tests.

Successful substantive gate:

```text
run                                       34672654522
head                                      83a5acd4e5b25413708bbda1715c58b5c78bce08
scope + syntax                            PASS
real CU-1 / HTTP                          15 / 15 PASS
frozen Step-3 exact-output fixtures       15 PASS
post-review clinical/output               11 / 11 PASS
inherited Chromium                        12 / 12 PASS
post-review Chromium                       9 / 9 PASS
Greek source-summary coverage             54 positions
packaged dependency closure               PASS
```

## 5. Explicitly not proven

```text
actual iPhone Safari / VoiceOver
full measured accessibility/contrast acceptance
receiving-physiotherapist real-user usefulness
Cyprus/GeSY item-level recommendation fidelity
Greece/England product-market need
real-patient workflow/privacy readiness
paid conversion / retention
```

## 6. Out of scope remains

```text
second diagnosis
new functional-baseline field
routine FFD measurement
unaudited Cyprus item-level evidence
UK/GR localized content
analytics
billing/auth
patient persistence
production routing/database
PR/merge/deploy
```

## 7. Exact next action

Product Owner reviews the exact tested synthetic artifact and screenshots and gives concrete keep/remove/change feedback.

No release or expansion action is inferred from the technical PASS.
