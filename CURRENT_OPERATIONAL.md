# CURRENT_OPERATIONAL.md — Knee-OA bounded review-amendment implementation

> **STATUS:** KNEE-OA POST-REVIEW BOUNDED AMENDMENT — IMPLEMENTATION ACTIVE.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Freshly verified `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Accepted reviewed candidate:** `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Synthesis/disposition parent:** `554ecfa30bf8d0c04a19510a5db0276e6844edd5`.
> **Implementation branch:** `feat/physio-knee-oa-review-amendments-v1-2026-09-12`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-POST-REVIEW-AMENDMENTS-V1-20260912`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** this bounded amendment session.
> **PR / merge / deploy / production smoke authority:** NONE.
> **Real-patient use / production integration:** NOT AUTHORIZED.

## 1. Product Owner authority exercised

The Product Owner authorized implementation and visual re-review of the bounded corrections after the four independent reviews and subsequent utility-gate discussion.

The general rule remains:

```text
PRODUCT OWNER REQUEST
!= CLINICAL EVIDENCE
!= RECEIVER VALUE
!= IMPLEMENTATION AUTHORITY
```

This implementation may include only items that passed or were explicitly adapted through that utility gate.

## 2. Authorized amendment scope

### Diagnosis / readiness UX

- remove the redundant checkbox-like `Διάγνωση επιβεβαιωμένη` interaction from the intended product UX;
- in the single-diagnosis prototype, use an explicit diagnosis-selection action as the clinician assertion;
- diagnosis auto-projects into the live referral after selection;
- missing diagnosis or laterality is named locally and highlighted with restrained error styling plus non-colour/text cue;
- no wizard/checklist dashboard.

### Clinical semantics

- generic weakness remains reported/contextual;
- quadriceps-specific weakness must be explicitly examination-derived rather than a sibling localization that silently creates an objective finding;
- FFD remains advanced/optional only;
- FFD wording becomes passive/fixed extension deficit, distinct from extension lag;
- no numeric `0° FFD`, no invented degree, no permanence wording;
- no new structured patient-priority/baseline field in this slice.

### Referral / physiotherapy autonomy

- low-information referrals remain proportionally concise rather than producing verbose generic treatment prose;
- treatment wording is reframed as rehabilitation priorities / physiotherapy assessment rather than technique/dose/progression instruction;
- explicit selected information must not silently disappear.

### UX / evidence

- qualifier lifecycle has predictable compact completion and visible/ARIA agreement;
- evidence cues become self-explanatory contextually without permanent badge clutter;
- first evidence disclosure is lighter for routine supported/conditional items, while all material conflicting positions remain visible immediately for mixed guidance;
- suggestion presentation is flattened while explicit add/provenance/stale guards remain;
- mobile manual-reconciliation state becomes direct and legible;
- historical safety/export guards remain authoritative.

### Jurisdiction / country architecture

- add a reusable jurisdiction-profile seam: international evidence core + optional local-system guidance overlay;
- first active profile is `CY_GESY` / `Κύπρος · ΓεΣΥ`;
- do not create a routine country dropdown or infer country from device location;
- local guidance is labelled separately and never silently overrides international evidence;
- no item-level Cyprus recommendation is imported until exact final HIO recommendation text is audited;
- future Greece/England profiles remain inactive hypotheses until workflow demand is demonstrated.

## 3. Explicit exclusions

```text
NO second diagnosis
NO new functional-baseline field
NO mandatory FFD measurement
NO UK/Greece evidence content yet
NO billing/auth/analytics
NO patient persistence
NO production router/API/DB changes
NO autonomous evidence updating
NO PR/merge/deploy
```

## 4. Acceptance gate

Fresh exact-head tests must prove at minimum:

- explicit diagnosis selection auto-renders diagnosis without redundant checkbox;
- diagnosis and side missing states are specific, local and accessible;
- generic weakness cannot become objective quadriceps weakness without explicit exam assertion;
- FFD is optional, passive/fixed, distinct from extension lag and rejects 0° when a degree is supplied;
- no new functional-baseline control exists;
- low-information output is shorter than rich patient-specific output;
- selected rehab information remains traceable while autonomy wording is less prescriptive;
- qualifier visible and ARIA states remain consistent;
- mixed-guidance evidence still shows all material source positions immediately;
- Cyprus/GeSY jurisdiction label is visible without claiming audited item-level local recommendations;
- inherited safety, manual-edit, network-failure, no-storage and mobile regressions remain green.

## 5. Exact next action

Implement the bounded changes in the synthetic loopback prototype, run focused server/browser regression, produce a fresh tested artifact for Product Owner visual review, then release the writer. No release lifecycle action follows automatically.