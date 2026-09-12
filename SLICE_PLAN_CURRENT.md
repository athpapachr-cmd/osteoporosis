# SLICE_PLAN_CURRENT.md — Knee-OA review intake, synthesis and Product Owner disposition

> **STATUS:** PRODUCT OWNER DISPOSITION PARTIALLY RECORDED / EVIDENCE-OF-UTILITY GATE ADDED / IMPLEMENTATION HOLD.
> **Slice:** `CU1-PRODUCT-KNEE-OA-CROSS-REVIEW-SYNTHESIS-V1-20260911`.
> **Branch:** `docs/physio-knee-oa-review-synthesis-v1-2026-09-11`.
> **Parent review-doc head:** `4bebd79fa4a34a04bf1341b4bd89f811ce071397`.
> **Reviewed candidate:** `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Writer:** NONE after documentation update.
> **Implementation and release authority:** NOT GRANTED.

## Scope completed

Intake of four specialist outputs and a supplementary combined output, source-role disambiguation, a complete 36-record finding ledger, explicit disagreement handling and proposed bounded follow-up. Product Owner has now clarified that proposed additions and even Product Owner requests are hypotheses until usefulness is checked. No runtime, frozen clinical contracts, evidence positions, tests or CI workflows changed in this documentation update.

## General clinical-utility gate — canonical rule

For this product track, **clinically interesting != worth collecting != worth showing != worth sending to the receiver**.

Any proposed field, qualifier, intervention, alert, evidence cue, output sentence or Product Owner/reviewer/author suggestion must answer, before implementation:

1. **What downstream decision, management, safety action, handoff understanding or workflow changes because this information exists?**
2. **Who needs it?** Referring clinician, receiving physiotherapist, patient, or product governance are different consumers.
3. **Is the referring clinician reasonably expected to know it accurately, or will the downstream professional reassess it anyway?**
4. **Does the incremental receiver/user value justify the extra tap, cognitive load, maintenance and evidence burden?**
5. **Is the support direct evidence/guideline support, receiver validation, or merely plausible clinical reasoning?** These must not be conflated.
6. If evidence shows a finding is clinically meaningful but does not show that collecting it in the referral improves the handoff, default to **optional / progressive disclosure / test first**, not routine expansion.
7. If usefulness remains uncertain, **do not add it yet**. Removal or deferral is preferable to feature accumulation.

Hard governance rule:

```text
PRODUCT OWNER REQUEST
!= CLINICAL EVIDENCE
!= RECEIVER VALUE
!= IMPLEMENTATION AUTHORITY
```

The assistant/reviewer is expected to challenge and verify Product Owner suggestions rather than implement them blindly.

## Product Owner disposition from 2026-09-12 discussion

### FFD / passive extension deficit

Current evidence supports that knee flexion contracture/loss of extension is associated with worse pain and function and that conservative stretching can improve extension ROM. It therefore has genuine physiotherapy relevance **when actually measured**.

Disposition:

- keep it **advanced / optional**, not routine or required;
- do not promote it merely because it is clinically meaningful;
- if retained, semantics must be passive/fixed extension deficit, distinct from active extension lag;
- no `0° FFD`, no invented degree, no word implying irreversibility/permanence;
- do not require the referring clinician to measure FFD for every OA referral;
- receiver-side testing still decides whether its presence in a physician referral materially improves the handoff.

### Main activity / functional goal + present baseline

Guidelines/rehabilitation consensus support functional goals, baseline assessment, activities/participation and shared decision-making **inside OA rehabilitation**. This does not by itself prove that the referring physician should supply a new structured baseline field.

Disposition:

- **do not add a new field now**;
- retain current functional categories and existing free-text escape hatch;
- test with receiving physiotherapists whether one concise priority/baseline item adds enough handoff value to justify UI surface;
- add only if receiver testing shows incremental value over the existing function block.

### Diagnosis display / confirmation control

A diagnosis that the clinician has explicitly selected should appear automatically in the live referral. There should be **no separate “include diagnosis in referral” checkbox**.

The safety distinction remains:

```text
explicit diagnosis selection/assertion
→ diagnosis auto-renders in referral

opening a fixed route without an explicit diagnosis-selection event
!= automatic diagnostic assertion
```

For the final integrated flow, diagnosis selection itself should carry the assertion. The current standalone prototype confirmation control is a temporary substitute for the missing upstream diagnosis selector and should not survive as an additional redundant checkbox in the production UX.

### Missing required fields

If a truly required prerequisite is missing (currently diagnosis assertion/selection and laterality), the UI should:

- identify the **specific** missing item;
- visually mark the relevant control locally, using restrained red/error styling **plus a non-colour cue**;
- provide a concise contextual message/bubble near the missing field when useful;
- avoid global warning modals, progress dashboards or a forest of red fields;
- clear the error state immediately when resolved.

### Cyprus / GeSY OA guideline

Treat the Cyprus-adapted OA guideline as **local-system guidance**, not as an automatically superior clinical-evidence source.

Required handling before product integration:

- verify the final official version and exact recommendation text;
- distinguish clinical evidence changes from Cyprus health-system feasibility, reimbursement/resource or implementation context;
- compare each material local divergence with NICE and other authoritative international sources;
- never allow local reimbursement/cost policy to masquerade as evidence of superior clinical effectiveness;
- label local-system guidance separately where it materially differs;
- distinguish “announced for integration into GeSY IT” from “verified live and operational in the current workflow”.

## Proposed correction sequence, subject to the utility gate

1. **Clinical/output amendment:** reported versus examined weakness; correct passive FFD semantics if FFD remains; proportional minimal output; receiver autonomy.
2. **Evidence amendment:** scope fidelity, contested ACE metadata, default/source-availability behavior, exact locators and review-date provenance; separate local Cyprus/GeSY policy context from international clinical-evidence synthesis.
3. **UX amendment:** automatic diagnosis projection after explicit diagnosis selection; no redundant include-diagnosis checkbox; specific missing-field cues; predictable qualifier completion/collapse; one visible/ARIA expansion state; lighter first evidence disclosure; compact suggestions and direct mobile manual reconciliation.
4. **Receiver validation before new clinical fields:** receiving-physiotherapist tasks decide whether FFD presentation, priority activity/baseline or other candidate fields materially improve handoff.
5. **Acceptance and later market discovery:** fresh targeted regression at the new exact candidate, scoped independent re-review, actual device/assistive-technology acceptance, aggregate referral-volume audit and later price-realistic experiment.

## Closure rules

Every original finding keeps its ID and severity in `KNEE_OA_REVIEW_FINDING_DISPOSITION_PROPOSED_V1.md`. Proposed deduplication never closes an original record automatically. Source conflicts require claim-level verification, not majority voting. The synthesis is not a fifth independent review.

A qualifier multi-select must not close after its first tap merely to satisfy an automatic-collapse slogan. A clinician-selected treatment must never silently disappear from output under an autonomy-related redesign. Safety restrictions remain explicit and are not downgraded to optional suggestions.

## Stop boundary

No implementation authority is created by these Product Owner dispositions. No code changes, clinical changes, analytics, billing, second diagnosis, PR, merge, deployment, patient use or source-to-live automated update may be inferred from this documentation step.