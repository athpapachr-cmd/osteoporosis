# SLICE_PLAN_CURRENT.md — CU-1 history + evidence + rehabilitation-sequence design hardening v1.3

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

# 2. Meaning of rehabilitation sequence

CU-1 does not prescribe routine visit frequency or total course duration.

The model is:

```text
clinical objective
→ route/subtype/management-context-applicable evidence-supported intervention directions
→ criteria for progression
→ precautions / do-not-progress / escalation criteria
→ next objective when supported
```

No universal MSK sequence is permitted. A one-phase evidence-bounded sequence is valid when evidence supports only one broad rehabilitation direction; unsupported later phases or thresholds are omitted and the evidence gap is explicit.

Calendar constraints are allowed only when an explicit patient-specific postoperative/fracture/orthopaedic protocol or an evidence source genuinely requires time.

---

# 3. Frozen object semantics

Normative human contract:

```text
clinic_utilities/contracts/CU1_HISTORY_EVIDENCE_TIMELINE_V1.md
```

Normative machine design schema:

```text
clinic_utilities/contracts/cu1_history_timeline_schema_v1.yaml
```

Frozen objects:

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

History may carry only explicitly supplied information such as:

```text
onset date or approximate duration
onset pattern
mechanism / trigger
symptom course
prior episodes
prior treatment + response
relevant investigations
aggravating/easing factors
work/sport/activity context
patient-priority activity
route-specific history items
```

Every non-empty history value has explicit provenance.

Hard rules:

```text
approximate duration != inferred exact date
mechanism != causal diagnosis
missing history != negative history
patient statement != objective finding
route-specific prompt != selected answer
```

---

# 5. Element-level evidence provenance

Every evidence-derived rendered:

```text
phase objective
intervention direction
progression criterion
precaution / do-not-progress criterion
reassessment / escalation criterion
```

must resolve to at least one active applicable `EvidenceClaim`.

Authority classes remain distinct:

```text
evidence_claim
patient_specific_protocol
clinician_instruction
```

Clinician preference and written patient-specific protocols are never relabelled as literature authority.

---

# 6. Protocol precedence

Written postoperative/fracture/orthopaedic restrictions are first-class `ProtocolConstraintV1` objects.

```text
explicit patient-specific written protocol/healing restriction
>
conflicting generic route evidence element
```

The conflicting route default is suppressed rather than blended.

---

# 7. Evidence corpus architecture and route-coverage extension

Normative manifest:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Current active evidence layers:

```text
cu1_evidence_registry_v1.yaml
→ core active authority

cu1_evidence_tranche2_v1.yaml
+ cu1_evidence_tranche2_promotion_v1.yaml
→ active authority / promotion PASS

cu1_evidence_tranche3_v1.yaml
+ cu1_evidence_tranche3_promotion_v1.yaml
+ cu1_evidence_tranche3_promotion_fix_v1.yaml
→ active authority / promotion PASS

cu1_evidence_route_coverage_v1.yaml
→ native explicit-ID route coverage
→ calcific rotator-cuff route PASS

cu1_evidence_route_coverage_instability_v1.yaml
→ native explicit-ID context-scoped instability coverage
→ glenohumeral-instability split PASS

cu1_evidence_route_coverage_amendments_v1.yaml
→ reviewed post-merge narrowing/suppression layer
```

The amendment layer is intentionally narrow. It may:

```text
narrow applicability
attach a reviewed sequence
correct evidence grading without upgrading evidence
suppress a scope-unsafe claim
```

It may not silently rewrite source history, broaden evidence scope or invent recommendations.

Work queue:

```text
clinic_utilities/contracts/cu1_evidence_coverage_matrix_v1.yaml
```

Synthetic design oracle:

```text
clinic_utilities/contracts/cu1_history_evidence_fixtures_v1.yaml
```

Shard integration and the reviewed route-extension gates are PASS. Global route coverage remains incomplete.

---

# 8. Disease/subtype/context specificity

Hard regression examples now include:

```text
midportion Achilles != insertional Achilles
lateral elbow loading != Achilles loading
GTPS/gluteal-tendinopathy evidence != isolated trochanteric bursitis
full-thickness cuff != rotator-cuff tendinopathy
acute isolated non-displaced meniscus != displaced/locked/repair-candidate meniscus
primary frozen shoulder != secondary or other stiff shoulder

calcific cuff JOSPT ESWT position
!= NICE HTG645 research-only position

traumatic anterior instability
!= posterior instability
!= atraumatic anterior instability
!= multidirectional instability

posterior postoperative Part-II rehabilitation/RTS evidence
!= nonoperative posterior-instability authority

conservative traumatic-anterior RTS criteria
!= postoperative RTS authority
```

When direction, cause, recurrence or management context is material to evidence applicability, unresolved context blocks evidence-aware sequence resolution rather than triggering a generic fallback.

---

# 9. Evidence strength, scope and conflicts

Each `EvidenceClaim` carries recommendation direction, output scope, strength/certainty when available, route/subtype/context applicability and conflicts when known.

Output scopes:

```text
referral_core
therapist_execution_detail
clinician_ui_only
```

Hard evidence-governance rules:

```text
framework grade stays framework-specific
expert consensus != low-certainty trial evidence
no eligible comparative trials != very-low effect estimate
conflicting frameworks remain separate
conflicting framework claim != automatic referral recommendation
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

Short output uses the same route/subtype/context authority in compressed form and must not become generic.

---

# 11. Current reviewed route coverage

Reviewed profiles include the earlier core/tranche2/tranche3 routes plus two newly completed coverage routes.

### Calcific rotator-cuff tendinopathy

```text
rep_calcific_rotator_cuff_v1
seq_calcific_rotator_cuff_v1
→ sequence_complete / evidence-bounded
```

Core active rehabilitation is supported. ESWT remains a visible JOSPT-vs-NICE framework conflict and is not automatically rendered.

### Glenohumeral instability / dislocation

The route container now resolves into context-specific branches:

```text
rep_shoulder_instability_anterior_traumatic_v1
→ seq_shoulder_instability_anterior_traumatic_v1

rep_shoulder_instability_posterior_v1
→ seq_shoulder_instability_posterior_nonoperative_v1

rep_shoulder_instability_anterior_atraumatic_v1
→ seq_shoulder_instability_anterior_atraumatic_v1

rep_shoulder_instability_multidirectional_v1
→ seq_shoulder_instability_multidirectional_v1
```

Unresolved direction/cause/management context blocks evidence-aware output. Postoperative rehabilitation routes elsewhere and remains subordinate to explicit surgical protocol restrictions.

Coverage does not imply runtime readiness. Multiple other routine routes remain pending, incomplete or explicit evidence gaps.

---

# 12. Coverage gate

Before future runtime evidence-aware generation can be separately authorized:

```text
EVERY routine route / material subtype / material management context
→ unique applicable RouteEvidenceProfile or reviewed evidence-gap behavior

EVERY nonblocked route variant
→ complete evidence-bounded RehabilitationSequenceV1

EVERY evidence-derived rendered element
→ >=1 active applicable referral-compatible EvidenceClaim

patient-specific protocol override
→ explicit non-literature authority

evidence gaps / conflicts
→ explicit

freshness
→ current or explicitly reviewed

history prompts
→ route/variant-specific and non-inferential

fixtures
→ matching route/subtype/context regression coverage
```

Generic fallback is forbidden.

---

# 13. Acceptance fixtures

The design oracle now includes the prior safety/provenance fixtures plus route-specific regression cases for:

```text
calcific ESWT JOSPT-vs-NICE conflict
calcific refractory lavage not treated as initial phase
traumatic anterior first-time branch
recurrent anterior management-context requirement
conservative anterior RTS scope
posterior nonoperative != postoperative Part-II RTS authority
atraumatic anterior branch
MDI no false efficacy estimate
unresolved instability direction/management blocks sequence
```

Additional route-specific fixtures remain required as route coverage expands.

---

# 14. Exact BLOCK criteria

The design cannot be declared complete while any of the following remains:

```text
routine route has no current applicable evidence profile
material subtype/context applicability is unresolved without explicit block behavior
evidence-supported rehabilitation direction exists but safe sequencing/evidence-gap behavior is undefined
routine referral route has no evidence-supported rehabilitation direction at all
claim scope/strength cannot be preserved
logical shard contains unresolved or duplicate IDs
route-history prompt coverage is materially incomplete
fixtures reveal cross-route/subtype/context leakage
```

Current literature gaps are handled as explicit gaps, not as permission to invent generic sequences.

---

# 15. Current gate and exact next action

Current gate:

```text
object/history/evidence-authority semantics      PASS
tranche2 promotion                              PASS
tranche3 promotion                              PASS
shard integration                               PASS
calcific route coverage                         PASS
instability context-split coverage              PASS
routine-route coverage                          FAIL globally
route-history prompt completeness               FAIL globally
route-complete fixture corpus                   FAIL globally

DESIGN-COMPLETE                                 NO
RUNTIME AUTHORIZED                              NO
```

Continue route-by-route from the reconciled matrix:

```text
1. glenohumeral_osteoarthritis
2. degenerative_meniscal_lesion_conservative_rehabilitation
3. patellar_tendinopathy
4. thumb_cmc1_osteoarthritis
5. cervical_routes
6. remaining_wrist_hand_and_elbow_routes
7. remaining routine routes in registry order
8. define reviewed evidence-gap behavior where full staging is unsupported
9. complete route-specific history prompts and matching fixtures alongside each route
10. exact design-completeness review
11. STOP only at DESIGN-COMPLETE or a newly specific BLOCK
```

No runtime evidence-aware generation is authorized by this slice.
