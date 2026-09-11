# CHANGELOG.md — Physio Referral productization

> **ROLE:** append-only supporting product-design history.
> **Parent repo authority:** root six canonicals remain authoritative for repo-wide operations.

---

## 2026-09-07 — Productization track activated

The existing CU-1 Physiotherapy Referral utility was selected as the fastest initial path toward modest recurring revenue, with an initial target of approximately €9.99/month and no immediate objective beyond roughly €1,000 monthly recurring revenue.

The intended longer-term architecture is a unified Clinical Cockpit with separately activatable modules; Physio Referral is the first focused module and should remain independently usable while being architecturally compatible with later cockpit entitlements.

The product must not be positioned as a simple text generator. Subscription value is expected to come from evidence-aware guidance, reviewed updates, clinical structure, flexibility and speed.

---

## 2026-09-07 — Single-diagnosis vertical slice selected

The product owner rejected parallel development of five conditions for the first productization experiment.

First complete vertical slice:

```text
Knee Osteoarthritis only
```

The single slice must prove the reusable architecture before a second diagnosis is added.

---

## 2026-09-07 — UX contract v1 frozen for prototype

The product owner approved a minimal, modern, mobile-first interaction model inspired by direct-manipulation first-party mobile software rather than conventional medical form UX.

Frozen principles include:

- smart evidence-aware starting plan instead of blank form;
- selectable rows/direct manipulation instead of routine checkbox grids;
- progressive disclosure by clinical context;
- compact `Περισσότερα` power-user layer with active-count memory when collapsed;
- live referral projection with no routine Generate button;
- color as an important evidence-state cue, never the only cue;
- green/strong emphasis for supported recommendations;
- neutral/blue-grey for context-dependent options;
- amber for limited/insufficient evidence;
- distinct caution state for recommendation against routine use;
- grey for not-yet-assessed evidence state;
- compact contextual evidence bubbles rather than warning boxes;
- small `i` control for concise rationale/source details;
- full bibliography only on deeper explicit request;
- evidence-backed suggestions with source/year and one-tap add;
- guideline publication year kept separate from product evidence-review date;
- normal final state shown simply as `Έτοιμη` rather than a quality score or dashboard of ticks;
- typical mobile routine path targeted at approximately 5–7 meaningful taps with little or no typing.

The UX contract was design-frozen for the Knee-OA prototype and later replanned only where Step-2 evidence review proved the five-state evidence model insufficient.

---

## 2026-09-11 — Step-2 Knee-OA evidence design frozen

The first evidence knowledge module was completed for Knee Osteoarthritis only on branch:

```text
design/physio-referral-knee-oa-evidence-v1-2026-09-11
```

Fresh base:

```text
d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37
```

Reviewed substantive head:

```text
6b82691c8431b699d20752c83b443793989f6402
```

The evidence review used current/relevant positions from VA/DoD 2026, Singapore ACE 2026, EULAR 2023 update, NICE NG226 2022, AAOS OAK3 2021 and ACR/AF 2019.

A material UX-design finding emerged: five evidence states could not honestly represent real guideline disagreement. Acupuncture was the clearest example, with reviewed major frameworks ranging from recommendation against to conditional/limited support and insufficient evidence.

The evidence-state model was therefore expanded to:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

The compact Greek surface semantic for real disagreement is:

```text
Οι οδηγίες διαφέρουν
```

A second integrity layer was added to prevent broad evidence strength from being silently transferred to narrower product items. Source positions now distinguish:

```text
direct_item_recommendation
named_component_of_broader_recommendation
broader_recommendation_only
contextual_clinical_mapping
```

The reviewed visible Knee-OA smart default is deliberately small:

```text
therapeutic exercise
progressive strengthening
education & self-management
```

with individualized physiotherapy/active rehabilitation implicit in the referral itself. Graded activity and more specific rehabilitation components are context-driven rather than universal defaults.

Manual therapy, soft-tissue techniques and acupuncture retain explicit mixed-guideline semantics. Dry needling remains excluded from the Knee-OA selectable surface; the evidence detail preserves the distinction between NICE recommendation against and AAOS/VA-DoD uncertainty. Weight management is strongly supported when overweight/obesity applies but remains advisory-only because the current CU-1 machine catalog has no dedicated selectable ID. The existing walking-aid ID is retained but is not yet exposed in the current Knee UI relevance scope.

Machine contract:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml
```

Human design:

```text
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_V1.md
```

Exact design review:

```text
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md
DESIGN PASS / MATERIAL OPEN FINDING NONE
```

Machine gates:

```text
run 34559461372 — SUCCESS on substantive head 6b82691...
run 34559680326 — SUCCESS on review-artifact head f20f01f...
```

No production CU-1 runtime/API/formatter/static UI/database mutation occurred. Step 2 closes as a design/evidence contract only.

Exact next product-design step:

```text
STEP 3 — dynamic Knee-OA referral/template contract
```
