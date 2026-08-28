# SLICE_PLAN_CURRENT.md — CU-1 history + evidence + rehabilitation-sequence design hardening v1.1

> **STATUS:** ACTIVE PRE-RUNTIME DESIGN HARDENING — EXACT GATE IN PROGRESS.
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
→ route/subtype-applicable evidence-supported intervention directions
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

Frozen objects now include:

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

Phase-wide citations are insufficient.

Every evidence-derived rendered:

```text
phase objective
intervention direction
progression criterion
precaution / do-not-progress criterion
reassessment / escalation criterion
```

must resolve to at least one active applicable `EvidenceClaim`.

Each element uses one explicit authority class:

```text
evidence_claim
patient_specific_protocol
clinician_instruction
```

Clinician preference and written patient-specific protocols remain machine-distinct from literature authority.

---

# 6. Protocol precedence

Written postoperative/fracture/orthopaedic restrictions are first-class `ProtocolConstraintV1` objects.

```text
explicit patient-specific written protocol/healing restriction
>
conflicting generic route evidence element
```

The conflicting route default is suppressed rather than blended. The protocol is never relabelled as a literature recommendation.

---

# 7. Evidence corpus architecture

The logical evidence registry is sharded for maintainability under:

```text
clinic_utilities/contracts/cu1_evidence_manifest_v1.yaml
```

Current shards:

```text
cu1_evidence_registry_v1.yaml
cu1_evidence_tranche2_v1.yaml
cu1_evidence_tranche3_v1.yaml
```

The manifest requires unique IDs across all shards, exact reference resolution, subtype/applicability matching, freshness checks and no cross-route fallback. Shard order has no clinical precedence.

Work queue:

```text
clinic_utilities/contracts/cu1_evidence_coverage_matrix_v1.yaml
```

Synthetic design regression oracle:

```text
clinic_utilities/contracts/cu1_history_evidence_fixtures_v1.yaml
```

---

# 8. Disease/subtype specificity

Hard regression examples:

```text
2024 midportion Achilles CPG
!= insertional Achilles authority

lateral elbow resisted wrist-extensor evidence
!= Achilles tendon-loading authority

GTPS / gluteal-tendinopathy exercise evidence
!= isolated trochanteric-bursitis authority automatically

traumatic anterior shoulder-instability consensus
!= posterior or multidirectional instability authority automatically
```

The current frozen route taxonomy may use explicit applicability conditions where this is sufficient; a taxonomy mutation is not justified merely to mirror every evidence subgroup.

---

# 9. Evidence strength, scope and conflicts

Each `EvidenceClaim` carries:

```text
recommendation direction
output scope
strength when available
certainty when available
route/subtype/applicability scope
conflicts when known
```

Output scopes:

```text
referral_core
therapist_execution_detail
clinician_ui_only
```

A therapist execution detail such as exact exercise frequency does not automatically become a physician prescription. A clinician-UI-only caveat does not automatically enter routine referral prose.

---

# 10. Referral output target

Detailed:

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

Short uses the same route/subtype authority in compressed form. It must not become generic.

When evidence cannot support a specific progression criterion, the system must not invent one merely to populate the section.

---

# 11. Current evidence-design coverage

Curated or staged route-specific profiles now include at least:

```text
deep_gluteal_piriformis_presentation
nonspecific_low_back_pain
low_back_pain_with_radiating_leg_symptoms
lumbar_spinal_stenosis_neurogenic_claudication
lateral_elbow_tendinopathy
achilles_tendinopathy — midportion
achilles_tendinopathy — insertional
lateral_ankle_sprain_rehabilitation
plantar_heel_pain_plantar_fasciitis
rotator_cuff_related_shoulder_pain
confirmed_full_thickness_rotator_cuff_tear_nonoperative
adhesive_capsulitis_frozen_shoulder
knee_osteoarthritis
patellofemoral_pain
acute_isolated_meniscal_injury_nonoperative
nonarthritic_intraarticular_hip_pain
greater_trochanteric_lateral_hip_pain_pathway — GTPS/gluteal-tendinopathy scope
median_neuropathy_at_wrist_carpal_tunnel — evidence-gap profile
de_quervain_first_dorsal_compartment_disorder — evidence-gap profile
glenohumeral_instability_dislocation — direction-specific evidence seeds only
```

Coverage does not imply runtime readiness. Several profiles are deliberately `sequence_incomplete` or `blocked_evidence_gap` because current literature does not support the required precision.

---

# 12. Coverage gate

Before any future runtime evidence-aware generation can be separately authorized:

```text
EVERY routine route / material variant
→ unique applicable RouteEvidenceProfile

EVERY nonblocked route variant
→ complete evidence-bounded RehabilitationSequenceV1

EVERY evidence-derived rendered element
→ >=1 active applicable EvidenceClaim

patient-specific protocol override
→ explicit non-literature authority

evidence gaps / conflicts
→ explicit

freshness
→ current or explicitly reviewed

history prompts
→ route/variant-specific and non-inferential

fixtures
→ exact semantic review passed
```

Generic fallback is forbidden.

---

# 13. Acceptance fixtures

The design oracle covers at minimum:

```text
chronic DGS with 8-month history + uncertain diagnosis
lateral epicondylalgia disease-specific loading
midportion Achilles subtype boundary
insertional Achilles boundary
cross-route generic-loading rejection
postoperative protocol override
fracture-healing restriction override
conflicting adjunct guidance
weak-evidence route
missing history
stale/superseded evidence
therapist execution detail excluded from referral core
clinician modification not relabelled as evidence
protocol-time rule
```

Additional route-specific fixtures remain required as route coverage expands.

---

# 14. Exact BLOCK criteria

The design cannot be declared complete while any of the following remains:

```text
routine route has no current applicable evidence profile
material subtype/applicability boundary is unresolved
evidence-supported rehabilitation direction exists but no safe criterion can be supported and no explicit evidence-gap behavior is defined
routine referral route has no evidence-supported rehabilitation direction at all
claim scope/strength cannot be preserved in the formatter contract
shard contains unresolved or duplicate IDs
route-history prompt coverage is materially incomplete
fixtures reveal cross-route/subtype leakage
```

Current literature gaps must be handled as gaps; they are not permission to invent a generic sequence.

---

# 15. Current exact next action

```text
1. exact-review the sharded evidence corpus against the frozen schema
2. reconcile coverage matrix with every accepted shard
3. continue route-by-route evidence curation for remaining routine routes
4. for evidence-gap routes, define reviewed safe behavior without invented rehabilitation claims
5. complete route-specific history prompts
6. expand route-specific fixtures
7. exact design-completeness review
8. STOP only at DESIGN-COMPLETE or BLOCK
```

No runtime evidence-aware generation is authorized by this slice.
