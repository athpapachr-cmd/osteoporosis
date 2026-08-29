# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 DESIGN GATE = `BLOCK`; SHARD INTEGRATION = `PASS`; ROUTE COVERAGE IN PROGRESS.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only / unmerged; reviewed route coverage is complete through ulnar-neuropathy-at-elbow matrix commit `71a263568ca0bbb4100e89a95eb16e36d5889bbe`; canonical bookkeeping commits may advance branch head.
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

## 1.1 Deferred future Clinical Documentation direction — accepted, not active

The product owner accepted a future reusable architecture direction and it is recorded only in the permanent/roadmap owners (`AGENTS.md`, `TODO.md`). It does **not** alter the active CU-1 slice, does not authorize a new schema/object taxonomy, and does not open medico-legal implementation.

Future direction:

```text
ONE reviewed patient-specific clinical-assertion layer
+
SEPARATE literature/evidence layer
+
MANY document-specific policies/views
```

Preserved future requirements include source/provenance axis distinct from semantic claim type, unresolved competing interpretations when contradiction evidence is not decisive, `diagnosis != causation`, `temporal association != causal relationship`, and medico-legal/accident reports as a future Document Policy rather than a separate clinical engine.

Explicitly deferred until a fresh safe design boundary:

```text
NO ClinicalAssertionV1 frozen now
NO new claim-state enums now
NO medico-legal schema/runtime code now
NO CU-1 REPLAN from this future requirement
NO CLINICAL_EXCELLENCE_PLAN object-taxonomy mutation before fresh owner/schema audit
```

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
evidence recommendation direction != automatic treatment selection
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
lateral_elbow_pain_or_provocation_or_imaging_finding != automatic_lateral_elbow_tendinopathy_diagnosis
acute_or_highly_irritable_LET != subacute_or_chronic_Grade_B_loading_context
outcome_measure_recommendation != progression_or_RTW_RTS_threshold
newer_low_certainty_synthesis != automatic_do_not_offer_of_older_route_specific_CPG_recommendation
medial_elbow_pain_or_flexor_pronator_provocation_or_imaging_finding != automatic_medial_elbow_tendinopathy_diagnosis
subjective_ulnar_paresthesia != objective_ulnar_deficit_or_formal_ulnar_neuropathy
lateral_elbow_CPG_grade != medial_elbow_authority_by_analogy
low_certainty_medial_eccentric_signal != mandatory_or_superior_loading_protocol
narrative_review_phase_description != validated_progression_model
ulnar_distribution_paresthesia_or_Tinel_or_flexion_provocation != formal_cubital_tunnel_diagnosis
formal_UNE_diagnosis != mild_severity
not_assessed_ulnar_motor_status != normal_or_mild_eligibility
mild_UNE_conservative_signal != nonmild_or_unknown_severity_authority
very_low_night_splint_evidence != default_splint_protocol
heterogeneous_physio_evidence != best_nerve_gliding_manual_or_electrical_method
progressive_objective_ulnar_motor_deficit_or_atrophy != routine_mild_conservative_sequence
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
lateral_elbow_route_coverage_extension
medial_elbow_route_coverage_extension
ulnar_elbow_route_coverage_extension
cu1_evidence_route_coverage_amendments_v1.yaml
cu1_evidence_route_coverage_lateral_elbow_amendment_v1.yaml
```

Regression extensions include:

```text
clinic_utilities/contracts/cu1_cervical_history_evidence_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c3_cervical_headache_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c4_cervical_dizziness_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c5_post_traumatic_neck_fixtures_v1.yaml
clinic_utilities/contracts/cu1_lateral_elbow_fixtures_v1.yaml
clinic_utilities/contracts/cu1_medial_elbow_fixtures_v1.yaml
clinic_utilities/contracts/cu1_ulnar_elbow_fixtures_v1.yaml
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

## Lateral elbow tendinopathy — PASS as single-phase evidence-bounded route

```text
rep_lateral_elbow_tendinopathy_v1
→ seq_lateral_elbow_evidence_bounded_v1
→ sequence_complete
```

The existing 2022 APTA/JOSPT LET CPG remains the route-specific clinical-practice authority. The reviewed activation amendment preserves exact CPG grading and replaces the incomplete two-phase seed as the profile target.

Key route semantics:

```text
lateral elbow pain / local tenderness / Cozen-Mill-Maudsley-type finding / imaging abnormality
!= autonomous LET diagnosis

subacute or chronic LET
→ Grade-B resisted wrist-extensor loading
→ isometric and/or concentric and/or eccentric
→ no universal exercise dose

acute or highly irritable LET
!= automatic subacute/chronic Grade-B loading authority

high-demand occupation/sport/hobby context
→ Grade-F gradual stress/strength/endurance/motor-control reintroduction
→ conditional direction, not a second phase with invented transition threshold
```

Selected adjuncts preserve their own CPG evidence direction while remaining optional by applicability/selection: local manual therapy Grade B, dry needling Grade B, rigid taping Grade B in selected irritable context, and activity-related counterforce/wrist-support orthosis Grade F for immediate use context. Evidence direction does not equal automatic selection.

The 2024 Cochrane review and 2024 Campos synthesis limit confidence in effect magnitude/durability and prevent claims of a universally superior conservative modality; they do not turn individualized exercise into `do_not_offer`.

The original synthetic outcome-measure strength was removed during pre-PASS review. PRTEE/DASH and PSFS/high-demand scales retain Grade-A assessment authority, while ROM, pressure-pain threshold, pain-free grip and maximum grip retain Grade-B assessment authority. None becomes an automatic progression or RTW/RTS threshold.

The sequence deliberately contains:

```text
progression_criteria: []
```

No universal numeric loading progression, return-to-work/sport threshold, fixed visit frequency or total course duration is manufactured. Atypical PIN/radial motor deficit, cervical/radicular pattern, substantial mechanical block, material trauma/instability or another discordant presentation requires reassessment/correct owner rather than generic LET reassurance.

Formal LET review and fixtures:

```text
clinic_utilities/contracts/CU1_LATERAL_ELBOW_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/cu1_lateral_elbow_fixtures_v1.yaml
```

## Medial elbow tendinopathy — PASS as single-phase low-certainty evidence-bounded route

```text
rep_medial_elbow_tendinopathy_v1
→ seq_medial_elbow_evidence_bounded_v1
→ sequence_complete
```

No current medial-specific rehabilitation CPG with graded recommendations equivalent to the lateral-elbow CPG was identified. The primary treatment-effect authority is the 2026 See/Loo/Jaafar systematic review of eccentric exercise in medial epicondylitis: five small studies / 143 patients, heterogeneous protocols, no meta-analysis and overall low certainty.

Normative route behavior:

```text
activity/load modification
→ narrative clinical context, not comparative efficacy

eccentric flexor-pronator loading
→ may be considered
→ low certainty
→ not mandatory
→ not universally superior
→ no universal dose

lateral-elbow CPG grades
→ NOT medial authority by analogy
```

The 2023 clinical overview and 2024 medial-elbow differential review are used for history, management context and differential boundaries rather than as treatment-effect estimates. Subjective ring/small-finger paresthesia remains distinct from objective ulnar deficit or formal ulnar neuropathy. Material valgus/UCL instability, progressive objective ulnar motor deficit, major trauma, substantial mechanical block or another discordant presentation requires reassessment/correct owner.

The sequence deliberately contains:

```text
progression_criteria: []
```

No narrative three-phase description is promoted into a validated phase model. No universal numeric loading progression, RTW/RTS threshold, fixed visit frequency or course duration is manufactured. Manual therapy, dry needling, taping, orthosis and ESWT are not auto-labelled medial-route evidence from lateral analogy.

Formal medial-elbow review and fixtures:

```text
clinic_utilities/contracts/CU1_MEDIAL_ELBOW_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/cu1_medial_elbow_fixtures_v1.yaml
```

## Ulnar neuropathy at the elbow / cubital-tunnel presentation — PASS as mild-conservative + nonmild/safety split

The route has three evidence contexts rather than one generic peripheral-nerve protocol:

```text
explicit mild context
+ objective ulnar motor status actually assessed without material deficit
+ no atrophy/clawing
+ no unresolved structural/localization owner
→ rep_une_mild_sensory_predominant_v1
→ seq_une_mild_conservative_v1
→ sequence_complete

nonmild / severity unresolved / objective motor status not sufficiently assessed
→ rep_une_nonmild_or_severity_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

progressive motor weakness / intrinsic atrophy / clawing / material objective worsening
or material trauma / structural compression / nerve instability / discordant localization
→ rep_une_progressive_motor_or_structural_safety_v1
→ rehabilitation_sequence_id: null
→ routine sequence blocked + reassessment/correct owner
```

The 2025 Cochrane review provides only a narrow conservative signal: in mild UNE, information about movements or positions to avoid may reduce subjective discomfort. This supports cautious education and individualized modification of documented provoking positions/movements but does not establish a universal splint, nerve-gliding, exercise, visit-frequency or total-course protocol.

The 2025 night-splint systematic review remains very-low certainty and insufficient to recommend routine night splinting over advice. CU-1 therefore does not generate a splint device/type, elbow angle, nightly duration or total course. The 2022 physiotherapy systematic review does not establish a best method; nerve gliding/neurodynamic techniques, manual therapy and electrical modalities are not auto-promoted to core care.

AANEM 2022 Level-B neuromuscular-ultrasound authority remains diagnostic adjunct context only:

```text
ultrasound may help confirm/localize UNE
!= replacement for clinical/EDX evaluation
!= autonomous diagnosis
!= treatment-effect authority
```

The 2025 diagnostic Delphi remains expert-consensus candidate criteria requiring further validation and weighting; it is not a formal CU-1 diagnostic scale.

Hard UNE boundaries:

```text
ulnar-distribution paresthesia
!= objective sensory deficit
!= objective motor deficit
!= formal UNE/cubital tunnel diagnosis

positive Tinel / elbow-flexion provocation
!= definitive diagnosis

formal diagnosis = yes
!= mild severity

objective motor status = not_assessed
!= normal
!= mild conservative eligibility
```

The mild sequence deliberately contains:

```text
progression_criteria: []
```

No numeric progression/discharge threshold, fixed visit frequency or total course duration is manufactured. Progressive objective weakness/atrophy/clawing or other material neurological worsening triggers clinician/specialist reassessment; CU-1 does not generate an autonomous surgical threshold or procedure choice.

Formal UNE review and fixtures:

```text
clinic_utilities/contracts/CU1_ULNAR_ELBOW_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/cu1_ulnar_elbow_fixtures_v1.yaml
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

The remaining block is route-content completeness, not lateral/medial/ulnar elbow evidence governance.

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
LET → no validated universal loading dose, numeric high-demand transition, RTW/RTS threshold or fixed PT course; newer evidence limits effect magnitude/durability claims
medial elbow tendinopathy → low-certainty eccentric evidence only; no medial graded CPG, universal loading mode/dose, validated progression or imported lateral-CGP adjunct grades
UNE → only explicit mild context with assessed nonmaterial motor status receives narrow education/position-modification sequence; nonmild/unknown severity remains evidence gap; night splint and other physio modalities are not default core
```

No generic MSK, cervical, elbow or peripheral-nerve fallback is permitted.

---

# 7. Exact next authorized action

Continue only on the existing writer, route-by-route from the reconciled matrix:

```text
1. posterior_interosseous_nerve_supinator_syndrome
2. remaining wrist_hand_and_elbow_routes
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
USE generic elbow fallback across elbow routes
USE generic peripheral-nerve rehabilitation fallback
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
INFER lateral_elbow_tendinopathy_from_one_provocation_test_local_tenderness_grip_loss_or_imaging_finding
APPLY subacute_chronic_LET_Grade_B_resisted_loading_to_acute_highly_irritable_context_without_matching_applicability
CONVERT Grade_F_high_demand_LET_reintroduction_into_a_required_second_phase_or_invent_a_transition_threshold
FLATTEN Grade_A_LET_PROM_function_measures_and_Grade_B_impairment_measures_into_one_synthetic_strength
TREAT CPG_recommend_direction_as_automatic_manual_therapy_dry_needling_taping_or_orthosis_selection
CLAIM large_or_durable_LET_benefit_despite_2024_low_certainty_synthesis
RELABEL low_certainty_2024_LET_synthesis_as_proven_no_effect_or_do_not_offer_exercise
LABEL clinician-selected_ESWT_as_route_evidence_without_a_reviewed_applicable_claim
ALLOW PIN_motor_deficit_cervical_pattern_mechanical_block_or_material_trauma_to_fall_back_to_routine_LET
INFER_medial_elbow_tendinopathy_from_one_tenderness_provocation_or_imaging_finding
CONVERT_subjective_ulnar_paresthesia_into_objective_deficit_or_formal_ulnar_neuropathy
BORROW_lateral_elbow_CPG_grades_as_medial_route_authority
UPGRADE_low_certainty_medial_eccentric_evidence_to_mandatory_or_superior_protocol
CONVERT_narrative_medial_three_phase_description_into_validated_progression_model
AUTO_LABEL_manual_therapy_dry_needling_taping_orthosis_or_ESWT_as_medial_route_evidence_without_independent_review
ALLOW_material_UCL_valgus_instability_progressive_ulnar_motor_deficit_or_major_trauma_to_fall_back_to_routine_medial_tendinopathy
INFER_formal_UNE_or_cubital_tunnel_diagnosis_from_paresthesia_Tinel_flexion_provocation_or_ultrasound_alone
CONVERT_not_assessed_ulnar_motor_status_into_normal_or_mild_eligibility
APPLY_mild_UNE_position_education_authority_to_explicit_nonmild_or_unknown_severity_as_complete_sequence
AUTO_PRESCRIBE_night_splint_device_angle_duration_or_course_from_very_low_certainty_evidence
AUTO_PRESCRIBE_nerve_gliding_manual_therapy_or_electrical_modalities_as_best_or_mandatory_UNE_method
TREAT_AANEM_Level_B_ultrasound_as_treatment_effect_certainty_or_autonomous_diagnosis
CONVERT_progressive_motor_weakness_atrophy_or_clawing_into_routine_mild_UNE_progression
GENERATE_autonomous_UNE_surgical_threshold_or_procedure_choice
ALLOW_cervical_plexus_wrist_level_ulnar_trauma_or_structural_context_to_fall_back_to_routine_cubital_tunnel_sequence
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