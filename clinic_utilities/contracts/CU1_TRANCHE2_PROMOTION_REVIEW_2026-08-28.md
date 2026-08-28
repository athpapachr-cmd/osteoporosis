# CU-1 Tranche 2 Normalization / Promotion Review — 2026-08-28

> **RESULT:** `PROMOTED AS ACTIVE DESIGN AUTHORITY`
> **RUNTIME AUTHORIZED:** NO
> **OVERALL CU-1 DESIGN-COMPLETE:** NO — existing overall `BLOCK` remains.
> **Repository:** `athpapachr-cmd/osteoporosis`
> **Reviewed branch:** `design/cu1-history-evidence-timeline-2026-08-28`
> **Authoritative main verified before review:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`
> **PR #63 source head before tranche2 promotion work:** `2f4e190b4f84f31e917e8da0e4ee3170e7a664b4`
> **Promotion projection commit:** `f9a055424e6fb40374eadaf7c553002d7b9a25cc`

---

## 1. Scope

This review covers only the high-frequency tranche2 evidence shard:

```text
clinic_utilities/contracts/cu1_evidence_tranche2_v1.yaml
```

The immutable staging file is preserved for audit history. Its normative promoted representation is:

```text
staging shard
+
clinic_utilities/contracts/cu1_evidence_tranche2_promotion_v1.yaml
=
logical high_frequency_tranche2 active design authority
```

This is a design/data-contract promotion only. It does not authorize runtime selection, formatter integration, persistence changes, CU-2 or PR-1.

---

## 2. Promotion-gate result

| Gate | Result | Review note |
|---|---|---|
| map-key identity materialization | PASS | `evidence_id`, `claim_id`, `prompt_id`, `route_evidence_profile_id`, `sequence_id` are deterministically materialized from canonical map keys |
| required non-optional fields | PASS | required profile claim arrays and sequence metadata/default collections are supplied by the promotion projection |
| source references | PASS | every promoted claim/profile source resolves within active core + promoted tranche2 logical registry |
| claim references | PASS | dropped hybrid claim IDs are replaced before reference validation; replacement claims resolve |
| route-profile references | PASS | all 8 promoted profiles resolve to frozen route IDs and their declared sequences or explicit null evidence-gap state |
| sequence references | PASS | all 6 promoted sequences resolve to one promoted profile and referral-compatible claim authorities |
| route/subtype applicability | PASS | no tranche2 evidence is generalized across a material non-covered subtype; rotator-cuff tendinopathy remains explicitly separate from full-thickness tear |
| output-scope compatibility | PASS | no `clinician_ui_only` or `therapist_execution_detail` claim remains authority for rendered phase objectives/interventions/progression elements |
| freshness / source identity | PASS | current sources were rechecked; selected metadata were corrected and PFP 2025 current guidance was added as a distinct framework |
| cross-shard duplicate IDs | PASS | no promoted tranche2 canonical ID collides with the active core evidence shard |
| exact human evidence-scope review | PASS | recommendation direction/strength/context were reviewed route-by-route for the promoted material |

**Tranche2 promotion gate: PASS.**

---

## 3. Exact semantic corrections made before promotion

### 3.1 Plantar heel pain — silent B/C hybrid removed

The staging shard collapsed two distinct 2023 CPG positions into one `B_and_C` claim. Promotion splits them:

```text
Grade B
→ orthoses should NOT be used as isolated short-term treatment

Grade C
→ orthoses MAY be used combined with other treatment
```

They remain clinician-side adjunct evidence and do not become automatic referral-core prescription.

### 3.2 Knee OA — AAOS and EULAR de-hybridized

The staging shard used both AAOS and EULAR as evidence IDs while carrying `strong_AAOS` as if it were a shared framework strength.

Promotion replaces this with framework-specific claims:

```text
AAOS 2021
→ exercise — Strong
→ self-management — Strong
→ patient education — Strong
→ neuromuscular adjunct — Moderate

EULAR 2023 update / published 2024
→ individualized multicomponent plan
→ information / education / self-management
→ exercise with tailored dosage/progression
```

The sequence may cite compatible claims from both frameworks, but no synthetic hybrid strength is assigned to the phase.

### 3.3 Patellofemoral pain — current 2025 framework added without erasing 2019 APTA

Freshness review identified the 2025 Dutch multidisciplinary anterior-knee-pain guideline as current route-specific guidance.

Promotion therefore preserves two explicit frameworks:

```text
APTA/JOSPT 2019
→ combined hip + knee exercise
→ load-management/education context
→ runner gait retraining where applicable

Dutch multidisciplinary guideline 2025
→ quadriceps- and/or hip-focused exercise first
→ structured pain-guided volume/intensity
→ education prioritized
→ structured reassessment at 6 and 12 weeks
```

The 6/12-week items are **reassessment windows**, not automatic phase advancement and not a routine PT session-frequency prescription.

### 3.4 Lumbar stenosis — output-scope leak removed

The staging sequence used both:

```text
lss_multimodal_nonpharmacological_care       referral_core
lss_no_false_strong_recommendation           clinician_ui_only
```

as objective authority.

Promotion removes the `clinician_ui_only` claim from rendered authority. Conditional/weak wording is preserved directly in the referral-compatible claim and phase metadata.

### 3.5 Source metadata hardened

Reviewed corrections include:

```text
EULAR OA update
→ Moseng et al.
→ Ann Rheum Dis. 2024;83(6):730-740
→ DOI 10.1136/ard-2023-225041

Lumbar stenosis / neurogenic claudication CPG
→ Bussières et al.
→ DOI 10.1016/j.jpain.2021.03.147

De Quervain 2025 NMA
→ Cuenca-Zaldívar et al.
→ J Hand Ther. 2025 Nov 25; online ahead of print
→ DOI 10.1016/j.jht.2025.09.001

Dutch PFP/PT guideline
→ Ophey et al.
→ Knee Surg Sports Traumatol Arthrosc. 2025;33(2):457-469
→ DOI 10.1002/ksa.12367
```

---

## 4. Promoted route-profile state

Promoted tranche2 profiles:

```text
rep_lateral_ankle_sprain_v1
rep_plantar_heel_pain_v1
rep_rotator_cuff_related_pain_v1
rep_knee_oa_v1
rep_patellofemoral_pain_v1
rep_lumbar_stenosis_claudication_v1
rep_carpal_tunnel_v1
rep_de_quervain_v1
```

Promoted rehabilitation sequences:

```text
seq_lateral_ankle_sprain_v1
seq_plantar_heel_pain_v1
seq_rotator_cuff_related_pain_v1
seq_knee_oa_v1
seq_patellofemoral_pain_v1
seq_lumbar_stenosis_claudication_v1
```

The promotion does **not** claim that all six sequences are route-complete. Their existing evidence-bounded/incomplete states remain meaningful.

Explicit evidence-gap profiles remain:

```text
carpal tunnel syndrome
→ no validated CU-1 criteria-based rehabilitation sequence established by the reviewed authority

De Quervain
→ comparative conservative evidence exists, but a validated active progressive-loading rehabilitation sequence is not established
```

---

## 5. What promotion means — and does not mean

Promotion means:

```text
tranche2 may participate in the normative DESIGN evidence registry
its normalized identities/references/applicability/output scopes are reviewed
its evidence limitations are authoritative design facts
```

Promotion does NOT mean:

```text
all tranche2 routes are sequence-complete
all routine CU-1 routes are covered
tranche3 is promoted
CU-1 is DESIGN-COMPLETE
runtime evidence-aware generation is authorized
PR #63 may be merged solely on the basis of tranche2 promotion
```

---

## 6. Residual overall blockers after tranche2 promotion

The tranche2-specific conformance blocker is closed. The overall CU-1 gate remains `BLOCK` because at least:

```text
tranche3 still requires normalization/review/promotion
many routine routes remain pending or evidence-incomplete
route-specific history prompt coverage is not complete
route-complete fixtures are not complete
several routes correctly retain explicit evidence/progression gaps
```

---

## 7. Decision

```text
TRANCHE2 NORMALIZATION                    PASS
TRANCHE2 REFERENCE REVIEW                 PASS
TRANCHE2 APPLICABILITY REVIEW             PASS
TRANCHE2 OUTPUT-SCOPE REVIEW              PASS
TRANCHE2 FRESHNESS / SOURCE REVIEW        PASS
TRANCHE2 PROMOTION                        PASS

TRANCHE2 STATE                            ACTIVE DESIGN AUTHORITY
RUNTIME AUTHORIZED                        NO
OVERALL CU-1 DESIGN-COMPLETE              NO
OVERALL CU-1 GATE                         BLOCK
```

Exact next design action after canonical reconciliation:

```text
normalize → review → promote tranche3
```
