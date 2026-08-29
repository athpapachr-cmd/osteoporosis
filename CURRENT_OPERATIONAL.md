# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; SHARD INTEGRATION = `PASS`; ROUTE COVERAGE IN PROGRESS.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only / unmerged; reviewed route coverage is complete through matrix commit `875340f507f7de30fa4ca8de4ea53f746f9ba6f5`; this canonical commit may advance branch head.
> **Runtime evidence-aware generation:** NOT AUTHORIZED.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Current phase

CU-1 remains in clinician-quality **pre-runtime design hardening** for:

```text
coherent structured HISTORY
+ criteria-based / evidence-bounded rehabilitation progression
+ route/subtype/management-context-specific literature provenance
```

The object/evidence architecture, tranche2/tranche3 promotion and shard integration are proven. Work is now route-by-route completeness with route-specific history prompts and matching regression fixtures.

---

# 2. Proven design invariants

```text
missing history != negative history
patient statement != objective finding
approximate duration != inferred exact date
progression != elapsed time alone
route A evidence != route B evidence
subtype/context A authority != subtype/context B authority
clinician_ui_only != rendered referral authority
therapist_execution_detail != automatic referral_core
clinician instruction != evidence recommendation
patient-specific protocol != literature recommendation
explicit written protocol/healing restriction > conflicting route default
framework-specific strength != synthetic cross-framework strength
expert consensus / clinical opinion / best-practice opinion != treatment-effect estimate
no eligible comparative trials != low or very-low effect estimate
framework conflict != silent guideline consensus
population-specific trial superiority != universal protocol superiority
assessment recommendation != validated progression threshold
subjective radiating symptoms != objective neurological deficit
positive provocation test != formal cervical radiculopathy diagnosis
not_assessed neurological status != normal
```

No patient identifiers were added.

---

# 3. Normative evidence corpus

Manifest:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Active logical evidence layers now include:

```text
core_seed_registry
high_frequency_tranche2 + reviewed promotion projection
shoulder_hip_meniscus_tranche3 + reviewed projection/overlay
route_coverage_extension
shoulder_instability_route_coverage_extension
degenerative_meniscus_route_coverage_extension
patellar_tendinopathy_route_coverage_extension
thumb_cmc1_oa_route_coverage_extension
cervical_nonspecific_route_coverage_extension
cervical_radiating_route_coverage_extension
cu1_evidence_route_coverage_amendments_v1.yaml
```

Cervical regression extension:

```text
clinic_utilities/contracts/cu1_cervical_history_evidence_fixtures_v1.yaml
```

All listed active shards have passed their native or reviewed schema/promotion gate. No staged evidence shard remains.

---

# 4. Reviewed native route coverage

## Calcific rotator-cuff tendinopathy — PASS

```text
rep_calcific_rotator_cuff_v1
seq_calcific_rotator_cuff_v1
→ sequence_complete — evidence-bounded
```

Core active rehabilitation is supported. ESWT remains an explicit JOSPT-2025-vs-NICE-HTG645 conflict and is not automatically rendered.

## Glenohumeral instability/dislocation split — PASS

No generic instability sequence exists. Traumatic anterior, posterior nonoperative, atraumatic anterior nonoperative and multidirectional-instability nonoperative contexts remain distinct. Unresolved direction/cause/management blocks evidence-aware output; postoperative instability belongs to the postoperative owner. Posterior Part-II postoperative RTS evidence is suppressed from nonoperative authority.

## Glenohumeral osteoarthritis — PASS

Nonoperative and preoperative-TSA contexts use broad APTA best-practice/evidence-gap-aware sequences. Postoperative arthroplasty belongs to `postoperative_shoulder_rehabilitation`. The absence of nonsurgical comparative PT RCTs is not relabelled as a low-certainty treatment effect.

## Degenerative meniscal lesion — PASS

```text
rep_degenerative_meniscus_conservative_v1
seq_degenerative_meniscus_conservative_v1
→ sequence_complete — evidence-bounded
```

The 2025 EU-US meniscus consensus supports nonoperative treatment including PT as first approach for symptomatic degenerative lesions and supports progressive ROM/strength/neuromuscular rehabilitation. ESCAPE 5-year and OMEX 10-year randomized follow-up support exercise-based management compared with arthroscopic partial meniscectomy in common degenerative tears.

Hard boundary:

```text
MRI tear != automatic symptom generator or surgery logic
clicking/catching != true locking
true locking / unresolved structural surgical indication -> block routine sequence + reassess
acute traumatic context -> acute-meniscus owner
postoperative context -> postoperative-knee owner
```

## Patellar tendinopathy — PASS

```text
rep_patellar_tendinopathy_v1
seq_patellar_tendinopathy_v1
→ sequence_complete — single-phase evidence-bounded
```

Current evidence is deliberately reconciled rather than flattened:

```text
Cochrane 2025
→ absolute strengthening effect remains low/very-low certainty by outcome/comparator
→ no universal high-certainty efficacy statement

2026 exercise NMA
→ no clinically meaningful superiority hierarchy among contemporary loading strategies
→ HSR reasonable reference, not mandatory protocol

Breda 2021 PTLE RCT
→ population-specific PTLE signal vs eccentric-only
→ not universal PTLE-superiority authority
```

Therefore progressive tendon/quadriceps loading may be represented as the broad conservative rehabilitation direction, but eccentric/isometric/HSR/PTLE choice and dosing remain therapist execution detail. No universal numeric progression or return-to-sport threshold is rendered. ESWT is not auto-recommended.

## Thumb CMC-1 osteoarthritis — PASS

```text
rep_thumb_cmc1_oa_v1
seq_thumb_cmc1_oa_v1
→ sequence_complete — single-phase evidence-bounded
```

Current authority supports education/ergonomic principles/pacing/assistive strategies, individualized thumb/hand exercise and a CMC-support orthosis when clinically appropriate. No universal exercise programme, orthosis type or fixed wear schedule is generated. Hand-therapy assessment measures remain assessment measures and are not converted to progression thresholds.

## C1 — Nonspecific neck pain — PASS

```text
rep_nonspecific_neck_pain_v1
seq_nonspecific_neck_pain_v1
→ sequence_complete — single-phase evidence-bounded
```

The 2025 DEGAM/AWMF S3 guideline is the primary generic nonspecific-neck authority. The APTA/JOSPT 2017 Neck Pain CPG remains classification-specific supporting context and does not override newer generic-route negative recommendations for passive modalities.

C1 design:

```text
activation / physical activity
+ education / self-management
+ individualized exercise when appropriate
+ selected manual adjunct only when chosen/applicable
→ no universal progression threshold
→ no fixed PT frequency/course duration
```

Generic C1 does not auto-recommend mechanical traction, laser, electrotherapy, ultrasound or kinesiotaping. Radiating symptoms, headache-dominant presentation, dizziness and post-traumatic context require their own cervical routes rather than a generic C1 fallback.

## C2 — Neck pain with radiating upper-limb symptoms — PASS

```text
rep_neck_radiating_upper_limb_v1
seq_neck_radiating_upper_limb_v1
→ sequence_complete — single-phase evidence-bounded
```

Hard semantic boundary:

```text
subjective radiating pain / paresthesia / numbness
!= objective motor / sensory / reflex deficit
!= formal cervical radiculopathy diagnosis
```

The APTA/JOSPT 2017 `neck pain with radiating pain` category supplies the broad route authority. A 2025 cervical-radiculopathy component network meta-analysis is used only under narrower matching radicular-classification context; its traction/neurodynamic/articular-treatment components and promising multimodal combination are not converted into a mandatory physician-prescribed bundle.

C2 design:

```text
active/function-preserving rehabilitation
+ acute mobilizing/stabilizing exercise when acute
+ chronic activity/education when chronic
+ selected intermittent traction only when specifically applicable/selected
→ no fixed traction force/duration/frequency
→ no universal numeric progression threshold
```

New/progressive objective neurological deficit or possible cord/myelopathic features block routine progression and require reassessment. `not_assessed` neurological status never becomes `normal`.

Matching route-specific history prompts and dedicated cervical fixtures exist for C1 and C2.

Formal cervical reviews:

```text
clinic_utilities/contracts/CU1_CERVICAL_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/CU1_C2_RADIATING_NECK_ROUTE_REVIEW_2026-08-29.md
```

---

# 5. Current overall gate

```text
object/history/evidence-authority semantics       PASS
element-level evidence provenance                PASS
protocol override model                          PASS
tranche2 promotion                               PASS
tranche3 promotion                               PASS
shard integration                                PASS
native route-coverage gate for reviewed routes   PASS
logical-amendment gate                           PASS

routine-route evidence coverage                  FAIL
route-specific history prompt coverage           FAIL globally
route-complete fixture corpus                    FAIL globally
several route progression/evidence gaps          BLOCKED / EXPLICIT

FINAL RESULT                                     BLOCK
DESIGN-COMPLETE                                  NO
RUNTIME AUTHORIZED                               NO
```

The remaining block is route-content completeness, not shard integration.

---

# 6. Preserved evidence-gap behavior

```text
DGS → no validated disease-specific progression thresholds
De Quervain → no validated active progressive rehabilitation sequence
carpal tunnel → no validated CU-1 criteria-based PT sequence from reviewed authority
acute isolated meniscus → selected PT wording is consensus/clinical opinion; no validated staged sequence
adhesive capsulitis → current guidance exists; no universal validated phase progression
GHOA → PT may benefit by best practice; nonsurgical comparative efficacy remains unestablished
MDI → cautious framework may be described when selected; comparative exercise benefit/harm remains unknown
patellar tendinopathy → progressive loading direction may be used cautiously; no mandatory loading mode or validated universal numeric RTS threshold
thumb CMC1 OA → exercise/orthosis conservative directions supported; optimal exercise dose, universal orthosis type/wear schedule and validated progression threshold remain unestablished
C1 nonspecific neck pain → no universal evidence-derived progression threshold; passive-modality bundle is not generic core care
C2 radiating neck pain → no universal progression threshold or fixed traction prescription; disease-specific NMA evidence requires narrower matching context
```

No generic MSK or generic cervical fallback is permitted.

---

# 7. Exact next authorized action

Continue only on the existing writer, route-by-route from the reconciled matrix:

```text
1. headache_with_cervical_msk_features
2. cervical_dizziness_presentation
3. post_traumatic_neck_pain
4. remaining_wrist_hand_and_elbow_routes
5. remaining routine routes in registry order
6. define reviewed evidence-gap behavior where full staging is unsupported
7. complete route-specific history prompts + matching fixtures alongside each route
8. rerun exact design-completeness review
9. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

---

# 8. Explicitly forbidden

```text
WRITE runtime evidence recommendation logic
WRITE runtime formatter integration
CHANGE persistence/retention behavior
USE generic MSK rehabilitation fallback
USE generic cervical fallback across C1-C5
INVENT progression thresholds
USE elapsed time alone as universal progression criterion
LABEL clinician preference as guideline recommendation
LABEL therapist execution detail as physician prescription by default
USE evidence across a noncovered subtype or management context
INFER formal cervical radiculopathy from radiating symptoms or a positive provocation test
CONVERT not-assessed neurological findings into normal findings
EXPAND cervical-radiculopathy-specific NMA evidence to every symptom-only C2 case
CONVERT the C2 NMA component combination into a mandatory rehabilitation bundle
AUTO-PRESCRIBE traction or a fixed traction dose in every C2 referral
USE posterior postoperative RTS evidence as nonoperative posterior authority
SILENTLY resolve conflicting ESWT frameworks
REPRESENT best-practice GHOA opinion as comparative treatment efficacy
IMPORT postoperative arthroplasty protocol into nonoperative/preoperative GHOA route
FREEZE eccentric / isometric / HSR / PTLE as a universal patellar-tendinopathy physician protocol
IMPORT older expert numeric patellar RTS/pain thresholds as current validated clearance rules
CONVERT thumb-CMC orthosis long-term-use guidance into a fixed PT course or wear duration
CONVERT hand-therapy assessment measures into validated progression thresholds
USE thumb-CMC-specific orthosis evidence as interphalangeal/generalized hand-OA authority
MERGE PR #63 merely because individual routes passed
OPEN CU-2
RESTART PR-1
```

---

# 9. Continuity rule

A fresh session must repeat the six-canonical bootstrap, verify the then-current remote `main` and PR #63 head, inspect this exact `BLOCK`, and continue only the route-coverage/history-prompt/fixture work on the existing CU-1 design writer unless the canonical lock changes.
