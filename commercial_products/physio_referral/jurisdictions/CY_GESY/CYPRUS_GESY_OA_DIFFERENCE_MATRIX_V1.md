# CYPRUS_GESY_OA_DIFFERENCE_MATRIX_V1

> **STATUS:** DESIGN COMPARISON V1 — NO LIVE-RULE AUTHORITY
> **Reviewed on:** 2026-09-12
> **Local audit:** `CYPRUS_GESY_OA_SOURCE_AUDIT_V1.md`
> **International authority:** `clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml`

## 1. Comparison rule

This matrix compares a verified local position with the already-reviewed international product state.

It does **not** count sources, average recommendations, or let a local recommendation replace the core.

```text
LOCAL AGREEMENT
!= stronger international evidence

LOCAL DIFFERENCE
!= permission to override core

ADMIN / REIMBURSEMENT RULE
!= clinical recommendation
```

## 2. Product-relevant matrix

| Product / intervention | International Knee-OA state | Verified Cyprus / GeSY clinical position | Relationship | Current routine UI impact | Evidence-state change? |
|---|---|---|---|---|---|
| Therapeutic exercise | `recommended_or_supported`; default | Offer tailored therapeutic exercise | agreement | none | **NO** |
| Progressive strengthening | `recommended_or_supported`; default | Local muscle strengthening is explicitly within therapeutic exercise | agreement | none | **NO** |
| Education / self-management | `recommended_or_supported`; default | Exercise education/long-term adherence and structured education/behaviour package supported | agreement | none | **NO** |
| Supervised exercise | delivery mode, not universal requirement | Consider supervised sessions | agreement/conditional | none; do not force supervised delivery | **NO** |
| Weight management | `recommended_or_supported`, explicit overweight/obesity context required, not current selectable item | Supported when overweight/obese | agreement | none; keep no-inference rule | **NO** |
| Manual therapy | `guideline_conflict_or_mixed` | Cyprus follows NICE: only consider with exercise; insufficient evidence alone | local position aligns with one side of international conflict | local detail on demand only if useful | **NO** |
| Soft-tissue techniques | `guideline_conflict_or_mixed` | Included locally under manual therapy alongside exercise | local position aligns with NICE side of international conflict | no routine change | **NO** |
| Acupuncture | `guideline_conflict_or_mixed` | Do not offer | **local difference from international resolved state**; aligns with NICE source within that conflict | optional Cyprus position in evidence detail when item is inspected | **NO** |
| Dry needling | `recommendation_against_routine_use`; excluded from current Knee-OA surface | Do not offer | agreement | none | **NO** |
| Electrotherapy | no current item in international product contract | Conditional local use for short-term pain only, with active care; no functional-benefit claim | **local-only product gap + genuine Cyprus-vs-NICE difference** | none now because item is not in current product | **NO CURRENT CORE STATE TO CHANGE** |
| Walking aids | `conditional_or_context_dependent`; current Knee UI not exposed | Consider for lower-limb OA | agreement | none unless receiver/workflow evidence justifies exposure | **NO** |
| Brace / orthosis / supports / tape | `conditional_or_context_dependent`; not routine | Do not routinely offer; selected biomechanical/functional exceptions | agreement | none | **NO** |
| Radiofrequency nerve ablation | not in current physio-referral contract | Local addition for selected severe pain after non-surgical failure before surgery | local addition outside current product boundary | do not show routinely | **NO** |
| Podiatry for painful callus | not in current physio-referral contract | Local addition when callus affects gait/mobility | local addition outside current product boundary | do not show routinely | **NO** |
| Glucosamine + chondroitin | not a current physio-referral intervention | Cyprus conditionally supports alongside exercise/weight management; NICE NG226 is against glucosamine | genuine clinical difference outside product boundary | do not show | **NO** |
| Intra-articular hyaluronan | not a current physio-referral intervention | Cyprus conditionally supports selected knee OA after other options fail/are unsuitable; NICE says do not offer | genuine clinical difference outside product boundary | do not show | **NO** |
| PRP | not a current physio-referral intervention | Cyprus conditionally supports after unsuccessful hyaluronan for short-term pain; no direct NG226 recommendation | local addition outside product boundary | do not show | **NO** |
| Corticosteroid injection | outside routine physio-referral intervention selection | Consider for short-term relief when other drugs fail/are unsuitable or to support exercise | agreement with NICE | do not show routinely | **NO** |
| Routine imaging for non-surgical OA management | no routine imaging requirement in product | Do not routinely image; Cyprus adds Appendix-II guidance | agreement + local implementation detail | no new imaging requirement | **NO** |

## 3. GeSY operational-policy matrix — deliberately separate

| Local rule | Rule class | Clinical evidence relationship | Product relevance | Routine display? |
|---|---|---|---|---|
| Physio access requires GeSY referral by participating PI/SI and eligible diagnosis | administrative/access | not comparable | validates need for a GeSY referral workflow | normally silent once product configuration is `CY_GESY` |
| Maximum covered sessions per diagnosis/beneficiary | reimbursement/access | not comparable | may matter to downstream service workflow | **NO** unless a specific user decision actually depends on it |
| PHYS02 minimum one-to-one session duration + clinical-note documentation | reimbursement/documentation | not comparable | receiver/service obligation, not referral evidence | **NO** on routine referral surface |
| Provider monthly unit cap / exceptional additional-session approval | reimbursement/resource | not comparable | provider payment mechanics | **NEVER routine clinical UI** |
| OA guideline planned integration into GeSY information system | system lifecycle | not comparable | future automation context only | show only in admin/source detail if needed; never imply active enforcement |

## 4. Genuine disagreements that must remain visible if surfaced

### A. Electrotherapy

```text
NICE source position:
  against routine offering due insufficient evidence of benefit

Cyprus adapted position:
  consider selected modalities only for short-term pain relief
  + alongside therapeutic exercise / weight management
  + no claim of improved function

product action today:
  none — electrotherapy is not a current product item
```

This is the clearest proof that `international core + jurisdiction overlay` is necessary. A future electrotherapy item could not be represented honestly by one global evidence state.

### B. Acupuncture

```text
international product state:
  guideline_conflict_or_mixed

Cyprus local position:
  against, following NICE
```

The local position may be shown as **Cyprus-specific context** inside evidence detail, but must not collapse the international state to `against`.

### C. Manual therapy

```text
international product state:
  guideline_conflict_or_mixed

Cyprus local position:
  conditional adjunct with exercise
```

Again, local alignment with NICE is context, not an override.

### D. Hyaluronan / glucosamine / PRP

These are genuine Cyprus-vs-NICE differences or local additions, but they are outside the current physio-referral interaction contract. They demonstrate jurisdictional variability without creating receiver-useful referral controls.

## 5. No-change findings

The current routine Knee-OA referral core remains well aligned with Cyprus on the parts that are actually surfaced routinely:

```text
therapeutic exercise
progressive strengthening
education / self-management
individualized active rehabilitation
functional rehabilitation when relevant
```

The audit found no verified Cyprus rule requiring a change to those current defaults or to the deterministic referral prose.

## 6. Product decision from the matrix

### Current Knee-OA core evidence state

**NO CHANGE.**

No existing international evidence item should be reclassified because of the Cyprus adaptation.

### Current routine Knee-OA main UI

**NO CHANGE from jurisdiction evidence.**

The local audit does not justify another visible control, country selector, GeSY badge wall, reimbursement panel, or routine recommendation feed.

### Evidence-detail capability

A jurisdiction overlay is justified as a separate data/traceability layer because:

1. at least one clinically relevant local recommendation genuinely differs from NICE;
2. current international mixed states can coexist with a more definite local position;
3. GeSY administrative rules exist and must be prevented from masquerading as clinical evidence;
4. the initial target market is Cyprus/GeSY;
5. future jurisdictions can reuse the same mechanism without contaminating the core.

That capability should remain mostly invisible during routine use and activate progressively only when a local position is relevant to an item the clinician is already inspecting.
