# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; SHARD INTEGRATION = `PASS`; ROUTE COVERAGE IN PROGRESS.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only / unmerged; reviewed route coverage is complete through C5 matrix commit `57cc0a75e1653b0f1536f51b9495f46e844c4b41`; this canonical commit may advance branch head.
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
headache_with_cervical_MSK_features != formal_cervicogenic_headache
cervical_ROM_or_provocation_or_imaging_finding != proven_headache_causation
low_certainty_network_ranking != mandatory_protocol
dizziness_plus_neck_features != cervical_causation
clinician_entered_cervical_dizziness_diagnosis != CU1_generated_diagnosis
treatment_response_or_positive_cervical_sensorimotor_test != diagnostic_proof
route_selection != exclusion_of_vestibular_migraine_neurovascular_or_other_causes
outcome_specific_GRADE != synthetic_cross_outcome_certainty
post_traumatic_neck_pain != generic_C1_nonspecific_neck_pain
whiplash_context != every_cervical_trauma
recent_WAD != persistent_WAD
unclear_WAD_phase != inferred_phase
C5_selection != structural_clearance
WAD_grade != CU1_inferred_classification
post_traumatic_headache_dizziness_or_arm_symptoms != automatic_C3_C4_C2_disease_specific_authority
study_programme_duration_or_frequency != universal_referral_schedule
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
cervical_headache_route_coverage_extension
cervical_dizziness_route_coverage_extension
cervical_posttraumatic_route_coverage_extension
cu1_evidence_route_coverage_amendments_v1.yaml
```

Cervical regression extensions:

```text
clinic_utilities/contracts/cu1_cervical_history_evidence_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c3_cervical_headache_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c4_cervical_dizziness_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c5_post_traumatic_neck_fixtures_v1.yaml
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

## C3 — Headache with cervical musculoskeletal features — PASS

The frozen C3 route has two machine-distinct evidence profiles:

```text
presentation-only / no explicit formal CGH diagnosis
rep_c3_cervical_headache_presentation_v1
→ seq_c3_cervical_headache_presentation_v1

formal_cervicogenic_headache_diagnosis = yes
rep_c3_formal_cervicogenic_headache_v1
→ seq_c3_formal_cervicogenic_headache_v1
```

ICHD-3 is used for diagnostic-boundary semantics, not autonomous diagnosis. Current disease-specific synthesis is restricted to explicit formal/clinician-established CGH context. Both sequences are single-phase with no universal progression threshold or fixed PT course.

## C4 — Cervical dizziness presentation — PASS as context split

The frozen C4 route has two machine-distinct contexts:

```text
presentation-only / clinician_diagnosis_cervicogenic_dizziness != yes
rep_c4_cervical_dizziness_presentation_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap by design

clinician_diagnosis_cervicogenic_dizziness = yes
rep_c4_clinician_established_cervical_dizziness_v1
→ seq_c4_clinician_established_cervical_dizziness_v1
→ sequence_complete / single-phase evidence-bounded
```

Hard boundary:

```text
dizziness + neck pain/stiffness or movement association != proven cervical causation
positive cervical/sensorimotor test != accepted diagnostic criterion
treatment response != diagnostic proof
C4 selection or clinician diagnosis != alternative causes excluded
```

The Bárány Society position does not endorse routine clinical diagnostic criteria or a specific therapy. Presentation-only C4 therefore remains an explicit evidence gap. The clinician-established branch is cautious, single-phase and contains no universal progression threshold, fixed course or promise of dizziness resolution.

## C5 — Post-traumatic / whiplash-associated neck pain — PASS as temporal + mechanism + safety split

The frozen C5 route is not one homogeneous evidence population. Current active contexts are:

```text
recent explicit uncomplicated WAD
rep_c5_recent_whiplash_wad_v1
→ seq_c5_recent_whiplash_wad_v1
→ sequence_complete

persistent explicit WAD
rep_c5_persistent_whiplash_wad_v1
→ seq_c5_persistent_whiplash_wad_v1
→ sequence_complete

explicit WAD but temporal phase unclear
rep_c5_whiplash_phase_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

other post-traumatic cervical pain without matching WAD context
rep_c5_other_posttraumatic_neck_pain_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

unresolved structural / neurological safety context
rep_c5_unresolved_posttraumatic_safety_v1
→ rehabilitation_sequence_id: null
→ routine sequence blocked
```

Recent WAD uses the still-current SIRA third-edition acute guideline: stay-active advice and neck exercise retain separate Level-B authority. Manual therapy remains a selected limited-evidence Level-C adjunct. SIRA's consensus recommendation against prolonged activity reduction and its Level-A collar recommendation remain separate claims rather than a hybrid strength.

Persistent WAD uses condition-specific guideline/systematic-review context with a weak/modest exercise signal. The 2024 guided neck-specific exercise review does not authorize its observed >6-week / >=2-session-per-week study pattern as a universal referral schedule. The 2025 education-plus-exercise meta-analysis remains very-low certainty and does not establish a mandatory superior combined bundle.

The proposed SIRA fourth edition remains draft/non-approved and is not normative authority. OPTIMa/Côté 2016 supplies the persistent objective-neurological-sign safety/reassessment context; it is not generic WAD treatment authority.

Both nonblocked C5 sequences deliberately contain:

```text
progression_criteria: []
```

No universal numeric progression threshold, fixed visit frequency, total PT duration or elapsed-time-only progression rule is generated. WAD grade and structural clearance remain clinician-entered/documented rather than CU-1 inference. A patient-specific written structural/healing restriction overrides conflicting uncomplicated-WAD defaults.

Post-traumatic headache, dizziness and radiating arm symptoms remain history/context and do not automatically import C3/C4/C2 disease-specific authority.

Matching route-specific prompts and regression fixtures now exist for C1-C5.

Formal cervical reviews:

```text
clinic_utilities/contracts/CU1_CERVICAL_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/CU1_C2_RADIATING_NECK_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/CU1_C3_CERVICAL_HEADACHE_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/CU1_C4_CERVICAL_DIZZINESS_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/CU1_C5_POST_TRAUMATIC_NECK_ROUTE_REVIEW_2026-08-29.md
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

The remaining block is route-content completeness, not cervical-route evidence governance.

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
C3 cervical headache → no automatic CGH diagnosis; no durable universal manual-therapy claim, superior exercise programme or validated progression threshold
C4 presentation-only dizziness → no disease-specific rehabilitation sequence because cervical causation/diagnostic criteria are not established
C5 unclear WAD phase / non-WAD cervical trauma → no cross-phase, WAD or generic-C1 fallback; unresolved structural/neuro context remains blocked
```

No generic MSK or generic cervical fallback is permitted.

---

# 7. Exact next authorized action

Continue only on the existing writer, route-by-route from the reconciled matrix:

```text
1. lateral_elbow_tendinopathy
2. remaining_wrist_hand_and_elbow_routes
3. remaining routine routes in registry order
4. define reviewed evidence-gap behavior where full staging is unsupported
5. complete route-specific history prompts + matching fixtures alongside each route
6. rerun exact design-completeness review
7. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
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
INFER formal cervicogenic headache from neck pain, ROM restriction, tenderness, provocation or imaging
DECLARE migraine, tension-type or another headache cause excluded merely because C3 is selected
APPLY formal-CGH systematic-review effect claims to presentation-only C3
CONVERT low-certainty CGH network rankings into a mandatory or superior technique bundle
CLAIM durable universal manual-therapy benefit from the short-term low-certainty CGH signal
USE routine C3 when post-traumatic/whiplash context is the primary presentation
INFER cervical/cervicogenic dizziness from neck pain plus dizziness, cervical provocation, sensorimotor testing or response to treatment
CLAIM that alternative dizziness causes were excluded merely because C4 is selected or a clinician diagnosis is entered
APPLY cervical-dizziness treatment-effect evidence to presentation-only C4
AUTO-PRESCRIBE vestibular rehabilitation, a fixed balance programme or the exact 2023 self-exercise bundle in C4
CONVERT Mulligan cervical-ROM evidence into global dizziness/balance superiority or diagnostic proof
FLATTEN outcome-specific low/very-low manual-therapy GRADE into a synthetic certainty
USE routine C4 when post-traumatic/whiplash context is primary
TREAT every post-traumatic cervical presentation as WAD
APPLY recent-WAD authority to persistent WAD or persistent-WAD authority to recent WAD
INFER WAD temporal phase from vague duration
INFER WAD grade or structural clearance from route selection
USE generic C1 as fallback for C5 or other cervical trauma
APPLY WAD-specific treatment authority to other cervical trauma without matching context
AUTO-PRESCRIBE the 2024 guided neck-exercise study frequency/duration as a universal C5 schedule
REPRESENT education-plus-exercise as proven superior in WAD despite 2025 very-low-certainty synthesis
HYBRIDIZE SIRA consensus activity-restriction advice with Level-A collar strength
ALLOW uncomplicated-WAD stay-active/no-collar defaults to override a patient-specific structural/healing restriction
AUTO-IMPORT C2/C3/C4 disease-specific evidence merely because post-traumatic arm symptoms/headache/dizziness coexist
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
