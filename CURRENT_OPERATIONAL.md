# CURRENT_OPERATIONAL.md — Knee-OA Product Owner disposition after independent reviews

> **STATUS:** FOUR SPECIALIST REVIEWS RECEIVED / SYNTHESIS RECORDED / PRODUCT OWNER UTILITY-GATE DISPOSITIONS RECORDED / IMPLEMENTATION HOLD.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Freshly observed main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Unchanged reviewed candidate:** `6539351c592c1dc3e49931057b63925dea3cb94d`.
> **Historical substantive implementation:** `243095ca9545bd2f96be8986520aeae8c3551c27`.
> **Review-prompt branch:** `review/physio-referral-knee-oa-independent-v1-2026-09-11` at `4bebd79fa4a34a04bf1341b4bd89f811ce071397`.
> **Synthesis branch:** `docs/physio-knee-oa-review-synthesis-v1-2026-09-11`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** NONE after this documentation update.
> **Implementation / PR / merge / deploy / public hosting / real-patient use authority:** NONE.

## 1. Review intake

Four specialist outputs and one supplementary combined review were received against the same pinned candidate. Specialist verdicts were CONDITIONAL PASS with no reported blocker at the synthetic-review stage. Those reviews are evidence for Product Owner disposition, not implementation authority.

## 2. New canonical Product Owner rule — clinical utility before feature adoption

The Product Owner explicitly requires that suggestions from the Product Owner, author, assistant or reviewer **must not be implemented blindly**.

For any proposed field, qualifier, output sentence, intervention, alert or evidence feature:

```text
clinically meaningful
!= useful in this workflow
!= useful to this receiver
!= worth the UI/cognitive/evidence-maintenance cost
```

Before addition, the product must identify the concrete change in downstream management, safety, handoff understanding, clinician decision or workflow. If that incremental value is unproven, the default is defer/test/remove rather than add.

A finding may be important for the physiotherapist to assess but still not belong in a physician referral. Conversely, information already known and reliable at referral time may have high handoff value. Receiver validation outranks feature enthusiasm.

This rule is reusable across future diagnoses and product modules.

## 3. Product Owner decisions from the current discussion

### FFD

Evidence indicates that passive extension loss/flexion contracture in knee OA correlates with worse pain/function and that conservative ROM/stretching interventions can improve extension. Therefore FFD can be physiotherapy-relevant **when actually measured**, but there is no justification to make it a routine physician-referral requirement.

Decision:

- keep advanced/optional only;
- correct semantics to passive/fixed extension deficit if retained;
- distinct from active extension lag;
- no `0° FFD`, no invented degree, no implication of irreversible permanence;
- receiver testing determines whether it earns its place in the referral long term.

### Priority activity / functional goal + current baseline

Rehabilitation guidance supports functional baseline assessment and patient-centred goals within physiotherapy. That does not establish that the referring physician should complete a new structured field.

Decision:

- do not add a new structured field now;
- retain current functional categories and free-text escape hatch;
- test incremental receiver value before any UI addition.

### Diagnosis checkbox / assertion

Diagnosis should automatically appear in the live referral after the clinician has explicitly selected/asserted it. A separate “include diagnosis” checkbox is redundant and should not exist in the final product.

The current standalone prototype's `Διάγνωση επιβεβαιωμένη` control exists because there is no upstream diagnosis-selection step. In the final integrated workflow, explicit diagnosis selection should carry the formal assertion and auto-project the diagnosis into the referral.

### Missing required fields

For genuinely required items such as diagnosis selection/assertion and laterality, show the exact missing requirement locally. Use restrained red/error styling plus a non-colour cue and concise contextual message/bubble. No modal warning sequence or checklist dashboard.

### Cyprus / GeSY OA guideline

Official HIO material confirms a Cyprus adaptation of NICE NG226 and an announcement that it will be integrated into the GeSY information system. The local guidance is clinically relevant but also health-system contextual.

Decision:

- classify it separately as local-system guidance;
- verify the final official recommendation text before importing it;
- compare material divergences with NICE and international evidence;
- separate scientific efficacy/effectiveness claims from local feasibility, reimbursement, resource or cost-policy considerations;
- never treat a GeSY/OAY policy choice as automatic proof of superior clinical effectiveness;
- distinguish planned/announced IT integration from verified live operational integration.

## 4. Accepted correction direction from prior synthesis

Subject to the utility gate, Product Owner agrees with the remaining bounded correction directions already synthesized: weakness semantics; proportional low-information referral length; physiotherapist-autonomy wording; evidence scope/metadata/locators; availability-aware defaults; explicit missing-field UX; qualifier state lifecycle; lighter first evidence disclosure; flatter suggestion presentation; mobile reconciliation; accessibility verification.

These directions remain **design/disposition decisions only** until a separate implementation gate is opened.

## 5. Exact next action

Prepare a bounded correction design/implementation slice that applies only the approved corrections above and explicitly excludes unproven new fields. Acceptance criteria must include:

- no redundant diagnosis checkbox in the intended integrated flow;
- diagnosis still requires explicit clinician selection/assertion;
- missing required inputs are locally explicit and accessible;
- FFD remains optional/advanced and semantically correct;
- no new patient-priority/baseline field unless receiver validation later supports it;
- local Cyprus/GeSY evidence is labelled and governed separately from international clinical evidence;
- all previous safety, selection-vs-suggestion and evidence-conflict invariants remain intact.

No second diagnosis, feature expansion, analytics, accounts/billing, production CU-1 rewrite, public/LAN hosting, PR, merge or deployment is authorized by this disposition step.