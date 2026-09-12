# CYPRUS_GESY_OA_OVERLAY_DESIGN_REVIEW_V1

> **STATUS:** DESIGN REVIEW — PASS / PRODUCT OWNER REVIEW HOLD
> **Reviewed on:** 2026-09-12
> **Runtime/UI implementation authority:** NONE
> **Bootstrap main:** `a2fa27c7ff26d1dd22cd6f726656ca0532daab75`

## 1. Reviewed artifacts

- `CYPRUS_GESY_OA_SOURCE_AUDIT_V1.md`
- `CYPRUS_GESY_OA_DIFFERENCE_MATRIX_V1.md`
- `../JURISDICTION_OVERLAY_SCHEMA_V1.yaml`
- `CYPRUS_GESY_OA_OVERLAY_UX_DESIGN_V1.md`
- current reviewed international Knee-OA evidence contract

## 2. Hard-requirement review

| Requirement | Result | Review finding |
|---|---|---|
| international evidence core remains independent | PASS | overlay schema forbids core mutation |
| Cyprus clinical adaptation != GeSY admin/reimbursement | PASS | separate `policy_class` + separate admin audit |
| resource/cost policy != stronger clinical evidence | PASS | no cross-class evidence promotion allowed |
| planned GeSY integration != active rule | PASS | guideline publication and IT integration are separate statuses |
| no silent local overwrite | PASS | `core_state_mutated_by_overlay: false` + display-only local context |
| genuine disagreement remains visible | PASS | local-difference UX preserves international and Cyprus rows separately |
| no source voting | PASS | schema/design explicitly forbids arithmetic/source-majority activation |
| clinician autonomy unchanged | PASS | overlay cannot auto-select, auto-remove or rewrite referral text |
| no autonomous literature-to-live-rule updates | PASS | explicit activation contract requires reviewed/versioned release |
| no patient persistence | PASS | no patient object/write path introduced |
| no second diagnosis | PASS | Knee-OA only |
| Greece/England capability without implementation | PASS | generic profile schema supports future IDs but no content/profile activation |
| Steve Jobs-style progressive disclosure | PASS | agreement silent; difference detail-level cue; admin info only at real workflow seam |

## 3. Evidence-integrity review

### 3.1 Verified local clinical differences

Material Cyprus-vs-NICE differences/additions are real, including:

- electrotherapy;
- radiofrequency nerve ablation;
- podiatry referral;
- glucosamine/chondroitin;
- hyaluronan;
- PRP-related local recommendations;
- additional imaging guidance.

This proves a jurisdiction model is not hypothetical.

### 3.2 Current referral core remains stable

No verified Cyprus source requires changing the current routine referral defaults:

```text
therapeutic exercise
progressive strengthening
education / self-management
individualized active rehabilitation
```

### 3.3 International conflict remains international conflict

For manual therapy and acupuncture, a definite Cyprus position does not resolve the broader international conflict. The local overlay records which local position applies without changing the international evidence state.

### 3.4 Administrative rules remain separate

GeSY referral/access/session/documentation/provider-unit rules are operationally relevant but are not evidence of clinical efficacy. They remain outside the clinical evidence resolver.

### 3.5 Public source-version caveat

The May-2026 HIO announcement says adaptation is complete and informs providers of implementation. The currently linked OA recommendation PDF still identifies itself as a draft from December 2025. The design correctly preserves this metadata conflict instead of inventing a clean final version.

This is not enough to treat the guideline as absent: the HIO implementation announcement and current adult-guideline index support active publication. It is enough to prevent claiming that the linked PDF has a cleanly versioned final-document identity.

### 3.6 Rationale discipline

No audited product-relevant local clinical difference is labelled reimbursement/cost-driven unless an official source explicitly says so.

For PRP, a distinct local rationale was not identified in the reviewed public rationale section, so the audit records `unclear` rather than inferring a motive.

## 4. Product-utility review

Applying:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

Disposition:

- electrotherapy difference: worth machine representation; not worth a new current routine control because the product does not currently expose electrotherapy;
- acupuncture/manual-therapy local positions: worth detail-level jurisdiction context because those interventions already exist in the wider product evidence space;
- RF ablation, podiatry, glucosamine, hyaluronan, PRP, injection guidance: not routine physio-referral content;
- GeSY reimbursement/session/provider-unit mechanics: not routine clinical-referral content;
- routine core active-rehab defaults: unchanged.

## 5. Technical/product architecture review

The proposed schema is intentionally jurisdiction-generic but does not implement unused markets.

Correct ownership:

```text
InternationalEvidenceItem
  remains global reviewed truth

JurisdictionOverlayV1
  adds local position + provenance + policy class + operational status

UI
  resolves display policy only
  never mutates core evidence
```

A future implementation should keep `CY_GESY` in explicit clinician/account configuration. Patient location must not silently select a jurisdiction profile.

## 6. What is NOT authorized by this PASS

- no runtime/UI implementation;
- no evidence-contract changes;
- no new intervention;
- no electrotherapy field;
- no Cyprus badge on the main screen;
- no country selector;
- no GeSY reimbursement panel;
- no v5 merge/deploy;
- no second diagnosis.

## 7. Bounded recommendation

**`IMPLEMENT JURISDICTION OVERLAY V1`**

Exact meaning:

1. accept the separate machine/provenance architecture after Product Owner review;
2. implement only the minimum generic overlay resolver + `CY_GESY` reviewed data in a later bounded runtime slice;
3. keep current international Knee-OA evidence states unchanged;
4. keep current routine Knee-OA main UI unchanged;
5. expose local context only progressively where clinically/workflow relevant;
6. keep administrative/reimbursement rules in a distinct operational class;
7. add no Greece/England content without real market need.

This recommendation is **not** an instruction to implement from this design branch. Product Owner review is the next lifecycle boundary.

## 8. Review disposition

```text
PRIMARY-SOURCE AUDIT          PASS
DIFFERENCE MATRIX             PASS
MACHINE SCHEMA                PASS AS PROPOSED DESIGN
UX / ROUTINE-EXCLUSION POLICY PASS
RUNTIME CHANGE                NONE
PRODUCT OWNER REVIEW          NEXT
```
