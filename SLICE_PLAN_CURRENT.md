# SLICE_PLAN_CURRENT.md — Knee-OA post-review bounded amendments v1

> **STATUS:** IMPLEMENTATION ACTIVE / SYNTHETIC PROTOTYPE ONLY.
> **Slice:** `CU1-PRODUCT-KNEE-OA-POST-REVIEW-AMENDMENTS-V1-20260912`.
> **Branch:** `feat/physio-knee-oa-review-amendments-v1-2026-09-12`.
> **Parent synthesis head:** `554ecfa30bf8d0c04a19510a5db0276e6844edd5`.
> **Reviewed candidate ancestry:** `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Fresh main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** ACTIVE, bounded to prototype + supporting contracts/tests/canonicals.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. General utility rule

For this and future product slices:

```text
clinically interesting
!= workflow-useful
!= receiver-useful
!= worth adding
```

Any proposed field, qualifier, alert, sentence, intervention or local-guideline cue must identify its downstream management/safety/handoff/workflow value, its actual consumer, whether the referring clinician is expected to know it, and whether that incremental value justifies cognitive/evidence-maintenance cost. Product Owner, author, assistant and reviewer proposals are hypotheses until checked.

## 2. Diagnosis assertion and readiness

The product should not require a second checkbox after the clinician explicitly selects the diagnosis.

For the single-diagnosis prototype:

```text
explicit tap/select on Knee OA diagnosis
→ formal clinician assertion
→ diagnosis appears automatically in referral
```

Opening/reloading the prototype alone is not assertion.

Missing diagnosis or side:

- exact local missing-state text;
- restrained red/error outline;
- non-colour cue/text;
- concise contextual hint;
- export remains blocked.

No modal warning, stepper or checklist dashboard.

## 3. Weakness semantics

Routine weakness remains reported/contextual unless explicitly converted to an examination finding.

Allowed semantic path:

```text
Αδυναμία
→ αναφερόμενο αίσθημα αδυναμίας

Αδυναμία στην εξέταση
→ objective_weakness
→ optional quadriceps specificity
```

Quadriceps specificity must not silently transform subjective localization into canonical `quadriceps_weakness`.

No MRC/dynamometry module is added.

## 4. FFD / passive extension deficit

Evidence supports real physiotherapy relevance of passive extension loss/contracture when present, but does not justify routine physician measurement.

Keep only in advanced examination.

```text
fixed/passive extension deficit
!= active extension lag
!= permanent/irreversible deformity
```

If degrees are supplied, they must be positive; unknown measurement remains unknown. No 0° FFD.

Referral Greek should be natural, e.g. `παθητικό έλλειμμα έκτασης 10°`, without unnecessary English label.

## 5. Functional detail

No new structured main-activity / functional-baseline field in this slice.

Reason: rehabilitation literature supports baseline/function/goals within physiotherapy, but incremental value of asking the referring physician to encode another structured field is not yet proven. Current categories and existing free-text note remain the escape hatch until receiving-physiotherapist testing.

## 6. Receiver-facing output

Low-information input should not create verbose pseudo-personalized treatment prose.

- diagnosis + side only → concise referral / request for physiotherapy assessment and individualized active rehabilitation;
- patient-specific finding/function → richer clinical picture and relevant selected rehabilitation priorities;
- plan language must preserve physiotherapist choice of technique, dose and progression;
- no explicit selected clinician item may silently disappear without a reviewed ownership change.

## 7. Evidence and suggestion UX

Preserve:

```text
selection != suggestion != evidence != safety
```

Amendments:

- supported/conditional evidence first view: state + concise rationale + provenance/review date;
- mixed guidance: all material opposing/neutral positions remain visible on first disclosure;
- evidence symbols gain contextual semantic text rather than becoming a permanent badge wall;
- suggestion card chrome becomes a compact contextual line with explicit Add + evidence access;
- stale-candidate, dismissal and no-auto-selection guards remain.

## 8. Qualifier lifecycle / accessibility

- qualifier controls remain inline and multiselect-capable;
- switching to another qualifier group may collapse the prior completed group;
- parent deselect/reset clears hidden qualifier state and sets visible/ARIA expansion consistently;
- do not auto-collapse after first pain-location tap when multiple focal locations may be valid;
- missing-field error states use text/non-colour semantics as well as colour.

Actual Safari/VoiceOver and measured contrast remain later acceptance beyond Chromium regression.

## 9. Jurisdiction adaptation architecture

Introduce reusable `jurisdiction_profile` semantics:

```text
international clinical-evidence core
+
optional local-system guidance overlay
```

First prototype profile:

```text
id: CY_GESY
label: Κύπρος · ΓεΣΥ
```

Rules:

- profile is explicit configuration/account preference later, not location inference;
- no routine country selector required in the current Knee-OA screen;
- local guidance is labelled separately and can conflict with international evidence;
- local feasibility/reimbursement/resource policy must not masquerade as stronger efficacy evidence;
- item-level HIO/GeSY recommendations require exact final-source audit before they influence evidence states/defaults/suggestions;
- announcement of IT integration != verified live production integration.

Future `GR` or `UK_ENGLAND` profiles remain dormant until market/workflow validation. England is specifically not assumed to need a doctor-centric physio-referral generator because NHS MSK self-referral/FCP pathways may substantially reduce that job-to-be-done.

## 10. Out of scope

```text
second diagnosis
new baseline/goal form
routine FFD measurement
Cyprus item-level evidence import before audit
UK/GR localized content
analytics
billing/auth
patient persistence
production routing/database
PR/merge/deploy
```

## 11. Acceptance

Focused server/browser tests must prove the amended semantics and all inherited safety/export/privacy behaviors. A fresh runnable artifact/screenshots must be produced for Product Owner visual review before any further product decision.
