# SLICE_PLAN_CURRENT.md — CU-1 history + evidence + rehabilitation-sequence design hardening v1.7

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

No universal MSK sequence or generic cervical sequence is permitted. A one-phase evidence-bounded sequence is valid when evidence supports only one broad direction. Unsupported later phases or thresholds are omitted and the evidence gap is explicit.

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

cu1_evidence_route_coverage_amendments_v1.yaml
→ reviewed post-merge narrowing / source-identity / grading / suppression layer
```

The amendment layer may only narrow applicability, attach a reviewed sequence, correct evidence grading without upgrading evidence, correct verified source identity metadata or suppress a scope-unsafe claim. It cannot broaden evidence scope or manufacture clinical recommendations.

Work queue:

```text
clinic_utilities/contracts/cu1_evidence_coverage_matrix_v1.yaml
```

Regression oracles:

```text
clinic_utilities/contracts/cu1_history_evidence_fixtures_v1.yaml
clinic_utilities/contracts/cu1_cervical_history_evidence_fixtures_v1.yaml
```

The cervical fixture file is a dedicated extension; its C1/C2 cases are normative design regressions for the reviewed cervical routes and do not authorize runtime behavior.

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
```

Unresolved material subtype/management/neurological context blocks evidence-aware sequence resolution rather than invoking a generic fallback.

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

The main regression oracle contains the previously reviewed route cases. The dedicated cervical extension now additionally covers:

```text
C1 activation/self-management core
C1 acute exercise evidence-grade preservation
C1 manual therapy optional not mandatory
C1 newer generic passive-modality scope vs older classification bundle
C1 other-cervical-route fallback prevention
C1 neurological/structural concern without false reassurance
C1 medical-review window != PT course duration

C2 radiating symptoms != formal radiculopathy
C2 acute active-rehabilitation grade preservation
C2 chronic traction selected adjunct != universal treatment
C2 broad symptom route != automatic cervical-radiculopathy NMA scope
C2 NMA components != mandatory bundle
C2 progressive objective neurological deficit block
C2 possible cord-feature block
C2 no universal progression threshold
```

Additional route fixtures remain required as coverage expands through C3-C5 and the remaining registry.

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
object/history/evidence-authority semantics      PASS
tranche2 promotion                              PASS
tranche3 promotion                              PASS
shard integration                               PASS
calcific route coverage                         PASS
instability context-split coverage              PASS
GHOA management-context coverage                PASS
degenerative-meniscus route coverage            PASS
patellar-tendinopathy route coverage             PASS
thumb-CMC1-OA route coverage                     PASS
C1 nonspecific-neck route coverage              PASS
C2 radiating-neck route coverage                PASS
routine-route coverage                          FAIL globally
route-history prompt completeness               FAIL globally
route-complete fixture corpus                   FAIL globally

DESIGN-COMPLETE                                 NO
RUNTIME AUTHORIZED                              NO
```

Continue route-by-route:

```text
1. headache_with_cervical_msk_features
2. cervical_dizziness_presentation
3. post_traumatic_neck_pain
4. remaining_wrist_hand_and_elbow_routes
5. remaining routine routes in registry order
6. reviewed evidence-gap behavior where full staging is unsupported
7. route-specific prompts + matching fixtures alongside each route
8. exact design-completeness review
9. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

No runtime evidence-aware generation is authorized by this slice.
