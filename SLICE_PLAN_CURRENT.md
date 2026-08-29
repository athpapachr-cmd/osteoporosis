# SLICE_PLAN_CURRENT.md — CU-1 history + evidence + rehabilitation-sequence design hardening v1.14

> **STATUS:** ACTIVE PRE-RUNTIME DESIGN HARDENING — SHARD INTEGRATION PASS / ROUTE COVERAGE IN PROGRESS.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1 history-evidence-rehab-sequence v1.
> **Authoritative base:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **Runtime writer:** NONE.
> **Clinical taxonomy:** frozen and preserved unless a specific evidence conflict requires a narrow reviewed correction.
> **Runtime implementation:** NOT AUTHORIZED in this slice.
> **CU-2:** not authorized.
> **PR-1:** remains paused.

---

# 1. Problem

Clinician review identified three structural deficits after the prior formatter/dynamic-form work:

```text
A. HISTORY is under-modelled and therefore under-rendered
B. goals are flat rather than a safe criteria-based rehabilitation progression
C. rehabilitation directions are disconnected from explicit route/subtype-specific evidence provenance
```

The active slice is design/evidence hardening, not formatter polish and not runtime implementation.

---

# 2. Rehabilitation-sequence semantics

CU-1 does not prescribe routine visit frequency or total course duration.

```text
clinical objective
→ route/subtype/management-context-applicable evidence-supported intervention directions
→ criteria for progression
→ precautions / do-not-progress / escalation criteria
→ next objective when supported
```

No universal MSK sequence, generic cervical sequence, generic elbow sequence or generic peripheral-nerve sequence is permitted. A one-phase evidence-bounded sequence is valid when evidence supports only one broad direction. Unsupported later phases or thresholds are omitted and the evidence gap is explicit.

Calendar constraints are allowed only when an explicit patient-specific postoperative/fracture/orthopaedic protocol or an evidence source genuinely requires time.

---

# 3. Frozen object semantics

Normative human contract:

```text
clinic_utilities/contracts/CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
```

Normative machine schema:

```text
clinic_utilities/contracts/cu1_history_timeline_schema_v1.yaml
```

Frozen objects include:

```text
ReferralHistoryV2
HistoryProvenanceEntryV1
RouteHistoryPromptV1
RehabilitationSequenceV1
RehabilitationPhaseV1
InterventionDirectionV1
RehabilitationCriterionV1
GoalPlanV2
ReassessmentPlanV2
AuthorityReferenceV1
ProtocolConstraintV1
ClinicianModificationV1
EvidenceSourceV1
EvidenceClaimV1
RouteEvidenceProfileV1
```

No patient identifiers are introduced.

---

# 4. History contract

History may carry only explicitly supplied information such as onset/duration, mechanism, course, prior episodes/treatment, investigations, aggravating/easing factors, activity context, patient-priority activity and reviewed route-specific history items.

Hard rules:

```text
approximate duration != inferred exact date
mechanism != causal diagnosis
missing history != negative history
patient statement != objective finding
route-specific prompt != selected answer
not_assessed neurological status != normal
headache feature != formal headache diagnosis
dizziness symptom or neck association != cervical causation
post_traumatic_neck_pain != inferred_whiplash
approximate_post_traumatic_duration != inferred_WAD_temporal_phase
C5_route_selection != structural_clearance_or_WAD_grade
lateral_elbow_pain_or_provocation_or_imaging_finding != inferred_lateral_elbow_tendinopathy
LET_irritability_or_acuity != inferred_from_one_pain_score_or_test
medial_elbow_pain_or_flexor_pronator_provocation_or_imaging_finding != inferred_medial_elbow_tendinopathy
subjective_ulnar_paresthesia != objective_ulnar_deficit_or_formal_ulnar_neuropathy
ulnar_distribution_symptom_or_Tinel_or_flexion_provocation != formal_cubital_tunnel_diagnosis
formal_UNE_diagnosis != mild_severity
not_assessed_ulnar_motor_status != normal_or_mild_eligibility
lateral_forearm_pain_or_radial_tunnel_provocation != motor_PIN_diagnosis
PIN_motor_pattern != formal_PIN_diagnosis_or_etiology_or_supinator_compression_site
formal_PIN_diagnosis != spontaneous_noncompressive_etiology
not_assessed_PIN_motor_status != normal_or_pain_only_classification
```

---

# 5. Element-level evidence provenance

Every evidence-derived rendered phase objective, intervention direction, progression criterion, precaution/do-not-progress criterion and reassessment/escalation criterion must resolve to at least one active applicable `EvidenceClaim`.

Authority classes remain distinct:

```text
evidence_claim
patient_specific_protocol
clinician_instruction
```

---

# 6. Protocol precedence

```text
explicit patient-specific written protocol/healing restriction
>
conflicting route evidence element
```

The conflicting route default is suppressed rather than blended. Protocol authority is never relabelled as literature authority.

---

# 7. Evidence corpus architecture

Normative manifest:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Active evidence layers:

```text
cu1_evidence_registry_v1.yaml
→ core active authority

cu1_evidence_tranche2_v1.yaml
+ cu1_evidence_tranche2_promotion_v1.yaml
→ promotion PASS

cu1_evidence_tranche3_v1.yaml
+ cu1_evidence_tranche3_promotion_v1.yaml
+ cu1_evidence_tranche3_promotion_fix_v1.yaml
→ promotion PASS

cu1_evidence_route_coverage_v1.yaml
→ calcific cuff + GHOA native explicit-ID route coverage

cu1_evidence_route_coverage_instability_v1.yaml
→ native explicit-ID instability context split

cu1_evidence_route_coverage_meniscus_v1.yaml
→ degenerative-meniscus native route coverage

cu1_evidence_route_coverage_patellar_tendinopathy_v1.yaml
→ patellar-tendinopathy native route coverage

cu1_evidence_route_coverage_thumb_cmc1_oa_v1.yaml
→ thumb CMC-1 OA native route coverage

cu1_evidence_route_coverage_cervical_v1.yaml
→ C1 nonspecific-neck native route coverage

cu1_evidence_route_coverage_cervical_radiating_v1.yaml
→ C2 radiating-upper-limb cervical native route coverage

cu1_evidence_route_coverage_cervical_headache_v1.yaml
→ C3 presentation-only + formal-CGH context split

cu1_evidence_route_coverage_cervical_dizziness_v1.yaml
→ C4 presentation-only evidence-gap + clinician-established cervical-dizziness context split

cu1_evidence_route_coverage_cervical_posttraumatic_v1.yaml
→ C5 recent-WAD + persistent-WAD sequence coverage with explicit unclear-phase / other-trauma / safety blocked contexts

cu1_evidence_route_coverage_lateral_elbow_v1.yaml
→ lateral-elbow native route extension using the existing core profile identity

cu1_evidence_route_coverage_medial_elbow_v1.yaml
→ medial-elbow native route coverage using medial-specific low-certainty evidence without lateral-CGP grade borrowing

cu1_evidence_route_coverage_ulnar_elbow_v1.yaml
→ ulnar-neuropathy-at-elbow mild-conservative + nonmild/safety context split

cu1_evidence_route_coverage_pin_supinator_v1.yaml
→ PIN/supinator pain-only mismatch + unresolved motor + spontaneous-noncompressive + structural/progressive safety context split with no route-specific rehabilitation sequence

cu1_evidence_route_coverage_amendments_v1.yaml
→ reviewed post-merge narrowing / source-identity / grading / suppression layer

cu1_evidence_route_coverage_lateral_elbow_amendment_v1.yaml
→ reviewed LET profile-sequence redirection + exact CPG grading/direction corrections

cu1_evidence_route_coverage_pin_supinator_amendment_v1.yaml
→ reviewed PIN source-identity corrections + pain-only profile applicability narrowing
```

The amendment layers may only narrow applicability, attach a reviewed sequence, correct evidence grading/direction without inventing or upgrading evidence, correct verified source identity metadata or suppress a scope-unsafe claim. They cannot broaden evidence scope or manufacture clinical recommendations.

Work queue:

```text
clinic_utilities/contracts/cu1_evidence_coverage_matrix_v1.yaml
```

Regression oracles:

```text
clinic_utilities/contracts/cu1_history_evidence_fixtures_v1.yaml
clinic_utilities/contracts/cu1_cervical_history_evidence_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c3_cervical_headache_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c4_cervical_dizziness_fixtures_v1.yaml
clinic_utilities/contracts/cu1_c5_post_traumatic_neck_fixtures_v1.yaml
clinic_utilities/contracts/cu1_lateral_elbow_fixtures_v1.yaml
clinic_utilities/contracts/cu1_medial_elbow_fixtures_v1.yaml
clinic_utilities/contracts/cu1_ulnar_elbow_fixtures_v1.yaml
clinic_utilities/contracts/cu1_pin_supinator_fixtures_v1.yaml
```

The dedicated fixture extensions are normative design regressions for reviewed routes and do not authorize runtime behavior.

---

# 8. Disease/subtype/context specificity

Hard regression examples include:

```text
midportion Achilles != insertional Achilles
GTPS/gluteal tendinopathy != isolated trochanteric bursitis
acute isolated non-displaced meniscus != displaced/locked/repair-candidate meniscus
primary frozen shoulder != secondary stiff shoulder

calcific cuff JOSPT ESWT position
!= NICE HTG645 research-only position

traumatic anterior instability
!= posterior instability
!= atraumatic anterior instability
!= multidirectional instability

posterior postoperative Part-II rehab/RTS evidence
!= nonoperative posterior authority

nonoperative primary GHOA
!= preoperative TSA
!= postoperative arthroplasty rehabilitation

MRI degenerative meniscal tear
!= automatic symptom generator or surgical indication

true meniscal locking
!= clicking/catching

patellar PTLE superiority signal in one selected RCT
!= universal PTLE protocol superiority

thumb CMC-1 OA orthosis evidence
!= interphalangeal/generalized hand-OA authority

hand-therapy assessment measures
!= validated rehabilitation progression criteria

C1 nonspecific neck pain
!= C2 radiating upper-limb presentation
!= C3 cervical-headache presentation
!= C4 cervical-dizziness presentation
!= C5 post-traumatic neck pain

subjective radiating arm symptoms
!= objective motor/sensory/reflex deficit
!= formal cervical radiculopathy diagnosis

cervical-radiculopathy-specific component NMA
!= automatic authority for every symptom-only C2 case

headache with cervical MSK features
!= formal cervicogenic headache diagnosis

restricted cervical ROM / cervical provocation / occipital tenderness
!= sufficient CGH diagnosis

upper-cervical imaging finding
!= proven headache causation

formal-CGH systematic-review evidence
!= presentation-only C3 authority

low-certainty CGH network ranking
!= best or mandatory treatment protocol

dizziness + neck pain/stiffness or head-neck movement association
!= cervical causation

positive cervical/sensorimotor test or response to neck treatment
!= accepted cervical-dizziness diagnostic proof

clinician-entered cervical-dizziness diagnosis
!= CU-1 generated or validated diagnosis

cervical-dizziness treatment-effect evidence
!= presentation-only C4 authority

C4 route selection or clinician diagnosis
!= exclusion of vestibular migraine, BPPV, neurological, vascular, cardiovascular or otological causes

primary post-traumatic dizziness/neck presentation
!= routine C4 owner

post-traumatic cervical presentation
!= automatic WAD

recent/acute WAD
!= persistent WAD

unclear WAD temporal phase
!= inferred recent or persistent phase

other cervical trauma
!= WAD treatment-evidence authority
!= generic C1 fallback

C5 selection
!= fracture/dislocation/instability excluded
!= WAD/QTF grade assigned

post-traumatic headache/dizziness/radiating-arm symptoms
!= automatic C3/C4/C2 disease-specific authority

SIRA activity-restriction consensus
!= SIRA Level-A collar recommendation

2024 guided-exercise study frequency/duration observation
!= universal referral schedule

lateral elbow pain/tenderness/grip loss/provocation/imaging finding
!= automatic lateral elbow tendinopathy diagnosis

acute/highly irritable LET
!= subacute/chronic Grade-B resisted wrist-extensor loading authority

Grade-F high-demand LET reintroduction
!= validated second phase or numeric transition threshold

Grade-A LET PROM/function outcome measures
!= Grade-B impairment measures
!= automatic progression/RTW/RTS criteria

Grade-B LET adjunct recommendation
!= automatic adjunct selection

2024 low-certainty LET synthesis
!= proven no effect or do-not-offer exercise

medial elbow pain/tenderness/resisted wrist-flexion or pronation finding/imaging abnormality
!= automatic medial elbow tendinopathy diagnosis

subjective ring/small-finger paresthesia
!= objective ulnar deficit
!= formal ulnar neuropathy

lateral-elbow CPG recommendation grades
!= medial-elbow evidence authority by analogy

2026 low-certainty medial eccentric signal
!= mandatory eccentric-only protocol
!= universal loading-mode superiority

narrative medial rehabilitation phases
!= validated progression model

manual therapy / dry needling / taping / orthosis / ESWT evidence for lateral elbow
!= automatic medial-elbow route authority

ulnar-distribution paresthesia / night symptoms / flexion-provocation / Tinel
!= objective neurological deficit
!= formal UNE/cubital-tunnel diagnosis

formal UNE/cubital-tunnel diagnosis
!= mild severity

objective ulnar motor status not assessed
!= normal
!= eligibility for mild conservative sequence

mild UNE position-education signal
!= nonmild or severity-unclear complete rehabilitation authority

very-low-certainty night-splint evidence
!= default orthosis protocol

heterogeneous UNE physiotherapy evidence
!= best nerve-gliding/manual/electrical method

progressive objective motor weakness / intrinsic atrophy / clawing / material neurological worsening
!= routine mild conservative pathway

cervical / plexus / wrist-level ulnar / structural-trauma localization concern
!= cubital-tunnel fallback

lateral forearm pain / radial-tunnel provocation
!= motor PIN diagnosis

finger/thumb extension weakness with relatively preserved wrist extension
!= formal PIN diagnosis
!= spontaneous PIN etiology
!= supinator/arcade-of-Frohse compression

formal PIN diagnosis
!= etiology established
!= compression site established

spontaneous non-traumatic/non-compressive PIN management evidence
!= route-specific physiotherapy protocol

source-specific six-month spontaneous-PIN recovery signal
!= universal PT duration, progression criterion or surgery threshold

compressive / mass / traumatic / iatrogenic / inflammatory structural / progressive-motor PIN context
!= generic rehabilitation owner

PIN route selection
!= cervical / plexus / proximal radial / multifocal motor neuropathy exclusion
```

Unresolved material subtype/management/neurological/headache/dizziness/structural-trauma or elbow/peripheral-nerve differential safety context blocks evidence-aware sequence resolution rather than invoking a generic fallback.

---

# 9. Evidence strength, scope and conflicts

Output scopes remain:

```text
referral_core
therapist_execution_detail
clinician_ui_only
```

Evidence-governance rules:

```text
framework grade stays framework-specific
expert consensus / best-practice opinion != comparative efficacy evidence
no eligible comparative trials != low or very-low effect estimate
conflicting frameworks remain separate
conflicting framework claim != automatic referral recommendation
population-specific trial superiority != universal protocol superiority when later synthesis does not support a meaningful hierarchy
assessment recommendation != validated progression threshold
disease-specific evidence != broader symptom-presentation authority without matching applicability
positive provocation test != diagnosis
guideline framework grade != GRADE treatment-effect certainty
low-certainty network rank != protocol authority
outcome-specific GRADE certainty != synthetic cross-outcome certainty
therapeutic response != diagnostic proof
study programme frequency/duration signal != universal physician referral schedule
distinct recommendation strengths in one framework != synthetic hybrid strength
evidence recommendation direction != automatic treatment selection
newer low-certainty synthesis != automatic reversal of a route-specific CPG recommendation unless an explicit reviewed conflict exists
narrative management or phase description != comparative treatment-effect estimate or validated progression threshold
low-certainty medial eccentric evidence != mandatory protocol or universal loading superiority
AANEM diagnostic Level-B ultrasound authority != treatment-effect certainty
very-low-certainty night-splint evidence != automatic recommend or do-not-offer
small mild-case conservative RCT signal != nonmild or unknown-severity authority
spontaneous PIN conservative-management cohort signal != evidence for a specific physiotherapy programme
observed PIN recovery/surgical-review timing != universal rehabilitation timeline or autonomous surgical threshold
broad peripheral-nerve narrative rehabilitation advice != route-specific PIN treatment-effect authority
```

---

# 10. Referral output target

Detailed output order remains:

```text
ΔΙΑΓΝΩΣΗ / ΚΛΙΝΙΚΗ ΕΝΤΥΠΩΣΗ
ΙΣΤΟΡΙΚΟ
ΚΛΙΝΙΚΑ ΕΥΡΗΜΑΤΑ
ΛΕΙΤΟΥΡΓΙΚΗ ΕΠΙΒΑΡΥΝΣΗ
ΑΙΤΗΜΑ
ΣΤΑΔΙΑΚΟΙ ΣΤΟΧΟΙ ΚΑΙ ΚΡΙΤΗΡΙΑ ΠΡΟΟΔΟΥ
ΠΡΟΤΕΙΝΟΜΕΝΟΣ ΠΡΟΣΑΝΑΤΟΛΙΣΜΟΣ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΠΡΟΫΠΟΘΕΣΕΙΣ ΕΠΑΝΕΚΤΙΜΗΣΗΣ / ΚΛΙΜΑΚΩΣΗΣ
ΒΙΒΛΙΟΓΡΑΦΙΚΗ ΒΑΣΗ
```

Short output uses the same route/subtype/context authority in compressed form.

---

# 11. Current newly reviewed route coverage

### Calcific rotator-cuff tendinopathy

```text
rep_calcific_rotator_cuff_v1
seq_calcific_rotator_cuff_v1
→ sequence_complete / evidence-bounded
```

Active rehabilitation is core. ESWT remains an explicit JOSPT-vs-NICE conflict and is not automatically rendered.

### Glenohumeral instability / dislocation

Context-specific branches remain distinct: traumatic anterior, posterior nonoperative, atraumatic anterior nonoperative and multidirectional instability nonoperative. Unresolved direction/cause/management blocks evidence-aware output. Postoperative rehabilitation routes separately.

### Glenohumeral osteoarthritis

Nonoperative and preoperative-TSA profiles use broad APTA best-practice/evidence-gap-aware sequences. Postoperative arthroplasty routes separately. The absence of nonsurgical comparative PT RCTs is not relabelled as a low-certainty treatment effect.

### Degenerative meniscal lesion — conservative rehabilitation

```text
rep_degenerative_meniscus_conservative_v1
seq_degenerative_meniscus_conservative_v1
→ sequence_complete / evidence-bounded
```

First-line exercise-based conservative care is supported. True locking or another unresolved structural surgical indication blocks the routine sequence; acute traumatic and postoperative meniscal contexts use different owners.

### Patellar tendinopathy

```text
rep_patellar_tendinopathy_v1
seq_patellar_tendinopathy_v1
→ sequence_complete / single-phase evidence-bounded
```

CU-1 may describe individualized progressive tendon/quadriceps loading but must not prescribe one mandatory eccentric/isometric/HSR/PTLE regimen. Loading-mode dosing remains therapist execution detail. No validated universal numeric progression or RTS threshold is rendered.

### Thumb CMC-1 osteoarthritis

```text
rep_thumb_cmc1_oa_v1
seq_thumb_cmc1_oa_v1
→ sequence_complete / single-phase evidence-bounded
```

Current EULAR/ACR authority and recent thumb-specific syntheses support education/ergonomic principles/pacing/assistive strategies, individualized thumb/hand exercise and a CMC-support orthosis when clinically appropriate. The route does not mandate one exercise programme, orthosis type or wear schedule and has no evidence-derived progression criterion.

### C1 — Nonspecific neck pain

```text
rep_nonspecific_neck_pain_v1
seq_nonspecific_neck_pain_v1
→ sequence_complete / single-phase evidence-bounded
```

The 2025 DEGAM/AWMF S3 guideline is the primary generic C1 authority. The APTA/JOSPT 2017 CPG remains classification-specific supporting context. The route centers activation/physical activity, education/self-management and individualized exercise. Manual therapy is optional rather than mandatory. Newer generic negative recommendations prevent mechanical traction and selected passive modalities from being imported as routine C1 core care.

C1 deliberately has:

```text
progression_criteria: []
```

because no universal evidence-derived transition threshold is established. Persistent/progressive activity-limiting symptoms or neurological deterioration trigger reassessment rather than a fixed PT timeline.

### C2 — Neck pain with radiating upper-limb symptoms / radicular features

```text
rep_neck_radiating_upper_limb_v1
seq_neck_radiating_upper_limb_v1
→ sequence_complete / single-phase evidence-bounded
```

The broad route authority is the APTA/JOSPT 2017 `neck pain with radiating pain` classification. Current 2025 cervical-radiculopathy component-NMA evidence is narrower and can only apply under matching radicular-classification context.

Hard C2 invariants:

```text
radiating pain / paresthesia / numbness
!= objective neurological deficit
!= formal cervical radiculopathy diagnosis

positive Spurling / neurodynamic finding
!= formal diagnosis

not_assessed
!= normal
```

Acute mobilizing/stabilizing exercise retains its APTA Grade-C context; chronic activity/education retains Grade-B context. Intermittent mechanical traction can appear only when specifically selected/applicable in chronic radiating-pain care and CU-1 does not generate a fixed traction force, duration or frequency.

The 2025 NMA's traction/neurodynamic/articular components and promising combination remain conditional evidence context rather than a mandatory physician-prescribed bundle. New/progressive objective neurological deficit or possible cord/myelopathic features block routine progression and trigger reassessment.

C2 also deliberately has:

```text
progression_criteria: []
```

No universal numeric progression threshold is manufactured.

### C3 — Headache with cervical musculoskeletal features / formal cervicogenic headache

Two profiles preserve diagnostic authority:

```text
presentation-only:
rep_c3_cervical_headache_presentation_v1
→ seq_c3_cervical_headache_presentation_v1

explicit clinician diagnosis of CGH:
rep_c3_formal_cervicogenic_headache_v1
→ seq_c3_formal_cervicogenic_headache_v1
```

ICHD-3 is used for diagnostic-boundary semantics only. Cervical pain, ROM restriction, cervical provocation, occipital tenderness, or upper-cervical imaging do not independently authorize definitive CGH wording. Only an explicit clinician assertion can move the route into the formal-CGH profile.

APTA/JOSPT 2017 provides the neck-pain-with-headache rehabilitation framework. Acute supervised active mobility retains Grade B. Self-SNAG remains therapist execution detail. Subacute/chronic manual-therapy authority retains APTA Grade B but treatment selection remains separate from evidence authority.

For explicit formal CGH, the 2026 GRADE systematic review preserves only a low-certainty short-term manual-therapy signal, with no durable 12-month benefit established and direct exercise effect on headache outcomes still uncertain. The 2024 PT NMA's intervention rankings remain low-certainty comparative context and cannot be converted into a mandatory technique bundle or universal superiority claim.

Both C3 sequences deliberately contain:

```text
progression_criteria: []
```

No universal exercise/manual dose, numeric transition threshold, fixed PT frequency or course duration is manufactured. New/changed/progressive headache or neurological/vascular/systemic concern blocks routine progression. Primary post-traumatic/whiplash context routes to C5 review.

### C4 — Cervical/cervicogenic dizziness presentation

The C4 review results in a deliberate context split rather than a generic treatment route.

```text
presentation-only / clinician_diagnosis_cervicogenic_dizziness != yes:
rep_c4_cervical_dizziness_presentation_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

explicit clinician-established cervical/cervicogenic dizziness:
rep_c4_clinician_established_cervical_dizziness_v1
→ seq_c4_clinician_established_cervical_dizziness_v1
→ sequence_complete / single-phase evidence-bounded
```

The current Bárány Society position is used as diagnostic-boundary authority. It does not endorse routine clinical diagnostic criteria, an accepted diagnostic test or a specific treatment recommendation for cervical dizziness. Therefore neck pain/stiffness plus dizziness, movement association, positive cervical/sensorimotor findings or response to neck-directed therapy cannot establish cervical causation in CU-1.

Presentation-only C4 deliberately receives no disease-specific rehabilitation sequence. Disease-specific treatment claims from diagnosed/classified cervical-dizziness study populations cannot leak into this symptom-only context, and generic C1/C4 fallback is forbidden.

When an explicit clinician diagnosis is present, current evidence supports only cautious one-phase rehabilitation. The 2026 physiotherapy systematic review found a limited/conflicting evidence base with most included studies at high risk of bias. The 2025 manual-therapy GRADE review has outcome-specific low/very-low certainty; CU-1 does not flatten these outcomes into a single synthetic certainty. A small 2023 non-traumatic self-exercise RCT may support considering individualized active cervical rehabilitation, but its exact exercise bundle, frequency, dose and two-week follow-up are not universal protocol authority.

Manual therapy remains selected/optional. Mulligan-specific cervical-ROM findings remain therapist execution detail and do not establish balance/global superiority or diagnosis. Vestibular rehabilitation, balance, oculomotor or sensorimotor work is not automatically rendered without a matching deficit/indication or clinician instruction.

The clinician-established sequence deliberately contains:

```text
progression_criteria: []
```

No universal numeric progression threshold, fixed PT frequency/course duration or promise of dizziness resolution is manufactured. New acute/progressive dizziness or neurological, gait, otological, vascular/cardiovascular concern blocks routine progression. Primary recent trauma/whiplash context routes to C5 review.

### C5 — Post-traumatic / whiplash-associated neck pain

The C5 review is a deliberate temporal + mechanism + safety split.

```text
recent explicit uncomplicated WAD:
rep_c5_recent_whiplash_wad_v1
→ seq_c5_recent_whiplash_wad_v1
→ sequence_complete / single-phase evidence-bounded

persistent explicit WAD:
rep_c5_persistent_whiplash_wad_v1
→ seq_c5_persistent_whiplash_wad_v1
→ sequence_complete / single-phase evidence-bounded

explicit WAD but temporal phase unclear:
rep_c5_whiplash_phase_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

other post-traumatic cervical pain without explicit matching WAD context:
rep_c5_other_posttraumatic_neck_pain_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

unresolved structural / neurological safety context:
rep_c5_unresolved_posttraumatic_safety_v1
→ rehabilitation_sequence_id: null
→ routine C5 sequence blocked
```

The currently active SIRA third-edition acute WAD guideline remains the recent/acute authority. Stay-active advice and neck exercise each retain Level-B SIRA strength. Manual therapy remains a selected limited-evidence Level-C adjunct rather than automatic core care. SIRA's consensus advice against prolonged reduction of usual activities and its Level-A recommendation against routine immobilisation collars remain separate claims rather than a synthetic shared strength.

The proposed SIRA fourth edition remains draft/non-approved and is not normative authority in this review.

For persistent WAD, current guideline and synthesis evidence supports an active exercise/self-management direction but not one universal programme. The 2024 guided neck-specific exercise review's observed programme duration/frequency pattern remains therapist execution context and is not converted into a minimum physician-prescribed schedule. The 2025 education-plus-exercise meta-analysis remains very-low certainty and does not establish a mandatory superior combined bundle.

OPTIMa/Côté 2016 supplies the persistent objective-neurological-sign safety/reassessment context; it is not generic WAD exercise authority.

Both nonblocked C5 sequences deliberately contain:

```text
progression_criteria: []
```

No universal numeric progression threshold, fixed visit frequency, total PT duration or elapsed-time-only progression rule is manufactured. WAD/QTF grade and structural clearance remain explicit clinician-entered/documented context. Patient-specific structural/healing/orthopaedic restrictions override conflicting uncomplicated-WAD defaults.

Post-traumatic headache, dizziness and radiating-arm symptoms remain C5 history/context and do not automatically import C3 formal-CGH, C4 cervical-dizziness or C2 radiculopathy-specific authority.

### Lateral elbow tendinopathy

```text
rep_lateral_elbow_tendinopathy_v1
→ seq_lateral_elbow_evidence_bounded_v1
→ sequence_complete / single-phase evidence-bounded
```

The existing APTA/JOSPT 2022 CPG remains the route-specific authority and is reconciled with 2024 Cochrane/Campos evidence that limits confidence in effect magnitude and durability without converting exercise into a `do_not_offer` recommendation.

The reviewed design uses one required irritability-/task-informed phase. Education/self-management and relevant activity/load modification are available broadly. Grade-B resisted wrist-extensor loading is condition-gated to subacute/chronic LET and may use isometric, concentric and/or eccentric loading without one universal evidence-derived dose. Grade-C shoulder/scapular work is conditional on an actual proximal impairment. Grade-F high-demand stress/strength/endurance/motor-control reintroduction is a conditional direction rather than a second phase because no validated transition threshold exists.

Selected adjuncts preserve exact CPG direction and grading while remaining optional through selection/applicability gates. Manual therapy, dry needling and rigid taping retain Grade-B route authority in their reviewed contexts; activity-related counterforce/wrist support retains Grade-F immediate-context authority. `evidence recommendation direction != automatic treatment selection`.

The mandatory activation amendment removes a synthetic A/B outcome-measure strength. PRTEE/DASH and PSFS/high-demand activity scales retain Grade-A assessment authority; ROM, pressure-pain threshold, pain-free grip and maximum grip retain Grade-B assessment authority. These are tracking measures, not automatic progression or RTW/RTS thresholds.

The sequence deliberately contains:

```text
progression_criteria: []
```

No universal exercise dose, numeric high-demand transition, return-to-work/sport threshold, fixed visit frequency or total PT duration is manufactured. Atypical PIN/radial motor deficit, cervical/radicular pattern, substantial mechanical block, material trauma/instability or other discordant presentation triggers reassessment/correct-owner behavior rather than generic LET fallback.

### Medial elbow tendinopathy

```text
rep_medial_elbow_tendinopathy_v1
→ seq_medial_elbow_evidence_bounded_v1
→ sequence_complete / single-phase low-certainty evidence-bounded
```

No current medial-specific rehabilitation CPG with graded recommendations equivalent to the 2022 lateral-elbow CPG was identified. The treatment-effect authority is the 2026 See/Loo/Jaafar systematic review: five small clinical studies, 143 participants, heterogeneous eccentric-loading protocols, no meta-analysis and overall low certainty.

The reviewed route therefore allows activity/load modification as narrative clinical context and allows individualized eccentric flexor-pronator loading only as a low-certainty `consider` direction. It does not make eccentric loading mandatory, does not claim universal superiority, and does not create a universal dose, frequency or duration.

The 2023 clinical overview and 2024 medial-elbow differential review remain context/differential authority rather than comparative efficacy evidence. Lateral-elbow CPG Grade-B/Grade-F recommendations are not imported by analogy. Manual therapy, dry needling, taping, orthoses and ESWT are not automatically relabelled as medial-elbow route evidence.

Hard route boundaries:

```text
medial pain / local tenderness / resisted wrist-flexion or pronation pain / imaging abnormality
!= autonomous medial-elbow tendinopathy diagnosis

subjective ulnar paresthesia
!= objective neurological deficit
!= formal ulnar neuropathy

material valgus/UCL instability, progressive objective ulnar motor deficit, major trauma or substantial mechanical block
→ reassessment / correct owner
```

The sequence deliberately contains:

```text
progression_criteria: []
```

No narrative multi-phase description is promoted into a validated progression model, and no numeric loading transition, RTW/RTS threshold, fixed visit frequency or total PT course is manufactured.

Formal medial-elbow review and fixtures:

```text
clinic_utilities/contracts/CU1_MEDIAL_ELBOW_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/cu1_medial_elbow_fixtures_v1.yaml
```

### Ulnar neuropathy at the elbow / cubital-tunnel presentation

The UNE review is a deliberate mild-conservative + nonmild/safety context split.

```text
explicit mild clinical context
+ objective motor status actually assessed without material deficit
+ no intrinsic atrophy/clawing
+ no unresolved structural or alternative localization owner
→ rep_une_mild_sensory_predominant_v1
→ seq_une_mild_conservative_v1
→ sequence_complete / single-phase evidence-bounded

nonmild / severity unresolved / objective motor status not sufficiently assessed
→ rep_une_nonmild_or_severity_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

progressive motor weakness / intrinsic atrophy / clawing / materially worsening objective neurological deficit
or material trauma / structural compression / nerve instability / alternative localization concern
→ rep_une_progressive_motor_or_structural_safety_v1
→ rehabilitation_sequence_id: null
→ routine sequence blocked
```

The 2025 Cochrane treatment review gives only a narrow conservative signal: a small RCT in mild UNE supports cautious education about movements/positions to avoid. This does not establish a universal splint, exercise, nerve-gliding, treatment-frequency or total-course protocol.

The 2025 night-splint systematic review is very-low certainty and concludes evidence is insufficient to determine benefit over advice. CU-1 therefore does not auto-render night splinting and does not generate one device, elbow angle, nightly duration or course. The 2022 physiotherapy systematic review is heterogeneous and does not establish one best physiotherapy method; nerve gliding, manual therapy and electrical modalities remain non-core without separate reviewed authority.

AANEM 2022 neuromuscular-ultrasound Level B is preserved as a diagnostic-framework recommendation to help confirm/localize UNE in patients with suggestive symptoms/signs. It is clinician-UI authority, not treatment-effect certainty, and ultrasound does not replace clinical/EDX evaluation or independently create the symptomatic diagnosis.

The 2025 diagnostic Delphi is retained as expert-consensus candidate criteria only; further weighting and validation are required before a formal diagnostic scale. CU-1 does not use it as an autonomous diagnostic engine.

Hard UNE regressions:

```text
subjective paresthesia
!= objective sensory deficit
!= objective motor deficit
!= formal diagnosis

positive Tinel/elbow-flexion test
!= definitive diagnosis

formal diagnosis = yes
!= mild severity

objective motor status = not_assessed
!= normal
!= mild conservative eligibility

mild position-education signal
!= nonmild/unknown-severity complete sequence
```

The mild sequence deliberately contains:

```text
progression_criteria: []
```

No numeric progression/discharge threshold, fixed PT frequency or total course duration is manufactured. Progressive objective weakness, atrophy, clawing or other material neurological worsening requires clinician/specialist reassessment; CU-1 does not generate a surgical threshold or procedure choice.

Formal UNE review and fixtures:

```text
clinic_utilities/contracts/CU1_ULNAR_ELBOW_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/cu1_ulnar_elbow_fixtures_v1.yaml
```

### Posterior interosseous nerve / supinator syndrome

The PIN review closes route coverage as an explicit evidence-gap + safety/context split. No route-specific `RehabilitationSequenceV1` is authorized.

```text
pain-predominant radial-tunnel / lateral-forearm presentation without material PIN motor deficit
→ rep_pin_pain_only_or_radial_tunnel_mismatch_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap / route mismatch

PIN-pattern motor presentation with diagnosis/etiology/localization unresolved
→ rep_pin_motor_presentation_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap

explicit clinician-established spontaneous non-traumatic/non-compressive PIN palsy
+ no space-occupying lesion
+ no progressive motor safety trigger
→ rep_pin_spontaneous_noncompressive_established_v1
→ rehabilitation_sequence_id: null
→ conservative-management context supported but specific PT sequence not established

demonstrable compression/entrapment or mass
or trauma / iatrogenic / inflammatory structural cause
or progressive/materially worsening motor deficit
→ rep_pin_compressive_structural_or_progressive_v1
→ rehabilitation_sequence_id: null
→ specialist/correct-owner/safety behavior
```

The 2026 prospective multicenter spontaneous-PIN cohort supports an initial conservative-management strategy only in its narrow spontaneous, non-traumatic, non-compressive, no-space-occupying-lesion population. It does not define nerve gliding, splinting, manual therapy, electrical stimulation, strengthening dosage, visit frequency or PT duration. The route-specific systematic review found no randomized or controlled clinical trials, and current review does not identify a later high-quality controlled rehabilitation evidence base that establishes a criteria-based PIN physiotherapy sequence.

The cohort's six-month motor-recovery signal is retained only as source/population-specific management-course context. It is not converted into a universal PT duration, elapsed-time progression rule, automatic surgery trigger or procedure choice.

Hard PIN regressions:

```text
lateral forearm pain / radial-tunnel provocation
!= motor PIN diagnosis

motor pattern
!= formal PIN diagnosis
!= spontaneous etiology
!= supinator/arcade-of-Frohse compression

formal PIN diagnosis
!= etiology established
!= compression site established

not_assessed motor status
!= normal
!= pain-only classification

route selection
!= cervical / plexus / proximal radial / multifocal motor neuropathy excluded

investigation finding
!= rehabilitation protocol

spontaneous conservative-management evidence
!= specific PT effectiveness evidence
```

No nerve-gliding, splinting, manual-therapy, electrical-stimulation or fixed strengthening programme is automatically rendered. New/progressive objective motor weakness, demonstrable compression/mass, trauma/iatrogenic/structural cause or other material discordant localization requires clinician/specialist reassessment or the correct owner rather than generic elbow/peripheral-nerve fallback.

Formal PIN review and fixtures:

```text
clinic_utilities/contracts/CU1_PIN_SUPINATOR_ROUTE_REVIEW_2026-08-29.md
clinic_utilities/contracts/cu1_pin_supinator_fixtures_v1.yaml
```

---

# 12. Coverage gate

Before future runtime evidence-aware generation can be separately authorized:

```text
EVERY routine route / material subtype / material management context
→ unique applicable RouteEvidenceProfile or reviewed evidence-gap behavior

EVERY nonblocked variant
→ complete evidence-bounded RehabilitationSequenceV1

EVERY rendered evidence-derived element
→ >=1 active applicable referral-compatible EvidenceClaim

patient-specific protocol override
→ explicit non-literature authority

evidence gaps/conflicts
→ explicit

freshness
→ current or explicitly reviewed

history prompts + matching fixtures
→ route/variant-specific and reviewed
```

Generic fallback is forbidden.

---

# 13. Acceptance fixtures

The main regression oracle contains the previously reviewed route cases. Dedicated extensions now cover C1-C5 plus lateral, medial, ulnar and PIN elbow routes.

LET-specific fixtures verify:

```text
typical subacute/chronic resisted loading
acute/high-irritability does not inherit subacute/chronic Grade-B loading
high-demand Grade-F direction without numeric transition/RTW/RTS threshold
single provocation/imaging finding != autonomous diagnosis
objective PIN motor deficit -> reassessment/correct owner
cervical pattern not silently absorbed
manual therapy selected vs mandatory
manual therapy CPG grade + Cochrane uncertainty
dry needling selected not core
orthosis immediate context not long-term protocol
Grade-A vs Grade-B outcome-measure families remain distinct
outcome measures do not become progression criteria
2024 low certainty != do-not-offer exercise
ESWT not auto-evidence-authorized in this review
missing history remains missing
```

Medial-elbow fixtures verify:

```text
medial-specific profile/sequence resolution
single medial finding or imaging abnormality != autonomous diagnosis
2026 eccentric signal remains low certainty and optional
no lateral-elbow CPG grade borrowing
subjective ulnar symptoms != objective neuropathy
progressive ulnar motor deficit -> reassessment/correct owner
valgus/UCL structural concern -> reassessment/correct owner
activity modification != fixed rest protocol
narrative phase description != validated progression model
manual therapy/dry needling/taping/orthosis/ESWT not auto medial evidence
missing history remains missing
```

UNE fixtures verify:

```text
explicit mild + objectively assessed nonmaterial motor status resolves narrow sequence
not_assessed motor status != normal or mild eligibility
explicit moderate/nonmild does not inherit mild RCT as complete sequence
subjective paresthesia != objective deficit or formal diagnosis
positive Tinel/flexion provocation != formal diagnosis
Delphi consensus != validated diagnostic scale
AANEM ultrasound Level B remains diagnostic adjunct
progressive weakness/atrophy/clawing -> reassessment/safety block
night splint remains very-low-certainty and nondefault
nerve gliding/manual/electrical modalities not auto best method
cervical/plexus/wrist-level localization and trauma/instability fail closed
formal diagnosis does not bypass severity/motor-status gate
missing history remains missing
```

PIN fixtures verify:

```text
pain-only/radial-tunnel presentation does not become motor PIN
motor pattern does not create formal diagnosis or compression site
not_assessed motor status does not become normal or pain-only
formal diagnosis does not infer spontaneous etiology
2026 spontaneous conservative-management signal does not become a PT protocol
source-specific six-month signal is not universalized
compressive/mass/traumatic/iatrogenic contexts fail closed
progressive motor weakness triggers reassessment rather than rehab progression
cervical/plexus/proximal-radial/MMN differential remains open when relevant
EDX/ultrasound localization does not define rehabilitation protocol
nerve gliding/splint/manual/electrical modalities are not auto-rendered
no LET/generic-elbow/generic-peripheral-nerve fallback
missing history remains missing
```

Additional route fixtures remain required as coverage expands through the remaining registry.

---

# 14. Exact BLOCK criteria

The design cannot be declared complete while any of the following remains:

```text
routine route has no current applicable evidence profile
material subtype/context applicability is unresolved without explicit block behavior
evidence-supported direction exists but safe sequencing/evidence-gap behavior is undefined
routine route has no evidence-supported direction at all
claim scope/strength cannot be preserved
logical registry has unresolved or duplicate IDs
route-history prompt coverage is incomplete
fixtures reveal cross-route/subtype/context leakage
```

---

# 15. Current gate and exact next action

```text
object/history/evidence-authority semantics       PASS
tranche2 promotion                               PASS
tranche3 promotion                               PASS
shard integration                                PASS
calcific route coverage                          PASS
instability context-split coverage               PASS
GHOA management-context coverage                 PASS
degenerative-meniscus route coverage             PASS
patellar-tendinopathy route coverage              PASS
thumb-CMC1-OA route coverage                     PASS
C1 nonspecific-neck route coverage               PASS
C2 radiating-neck route coverage                 PASS
C3 cervical-headache route coverage              PASS
C4 cervical-dizziness context-split coverage      PASS
C5 post-traumatic temporal/context coverage       PASS
lateral-elbow-tendinopathy route coverage         PASS
medial-elbow-tendinopathy route coverage          PASS
ulnar-neuropathy-at-elbow context-split coverage  PASS
PIN/supinator evidence-gap/safety coverage        PASS
routine-route coverage                           FAIL globally
route-history prompt completeness                FAIL globally
route-complete fixture corpus                    FAIL globally

DESIGN-COMPLETE                                  NO
RUNTIME AUTHORIZED                               NO
```

Continue route-by-route:

```text
1. distal_biceps_tendon_disorder_nonoperative
2. remaining wrist_hand_and_elbow_routes
3. remaining routine routes in registry order
4. reviewed evidence-gap behavior where full staging is unsupported
5. route-specific prompts + matching fixtures alongside each route
6. exact design-completeness review
7. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

No runtime evidence-aware generation is authorized by this slice.