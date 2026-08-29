# CU-1 C2 Radiating-Neck Route Review — 2026-08-29

> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime mutation:** not authorized and not performed.  
> **Evidence shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_cervical_radiating_v1.yaml`  
> **Review mode:** exact route/reference/applicability/output-scope/freshness review.

---

## C2 — Neck pain with radiating upper-limb symptoms / radicular features

Canonical route:

```text
neck_pain_with_radiating_upper_limb_symptoms
```

Frozen cervical design requires three concepts to remain machine-distinct:

```text
subjective radiating/radicular symptoms
!= objective motor/sensory/reflex deficit
!= formal cervical radiculopathy diagnosis
```

A positive Spurling, distraction or upper-limb neurodynamic finding is a clinical finding and cannot independently promote the route to a formal diagnosis.

### Current sources reviewed

1. Blanpied PR et al. / APTA Orthopedics. **Neck Pain: Revision 2017.** J Orthop Sports Phys Ther. 2017;47(7):A1-A83. DOI `10.2519/jospt.2017.0302`. APTA Orthopedics continues to list this as its published neck-pain CPG at the review date. Its `neck pain with radiating pain` classification directly matches the broad C2 symptom-presentation route better than disease-only radiculopathy literature.
2. Nunez de Arenas-Arroyo S et al. **What components and formats of rehabilitation interventions are more effective to reduce pain in patients with cervical radiculopathy? A systematic review and component network meta-analysis.** Clin Rehabil. 2025;39(10). DOI `10.1177/02692155251365193`. Search through 2025-07-01; 36 trials, 25 interventions and eight active components.
3. DEGAM/AWMF S3 **Clinical Practice Guideline: Nonspecific Neck Pain** (2025) was reviewed only for general neck-pain safety context. Its nonspecific-treatment recommendations are not imported as C2 treatment authority because C2 is a distinct route.

### Freshness decision

No newer APTA/JOSPT route-specific CPG replacing the 2017 Neck Pain CPG was identified. The 2025 component NMA materially updates the rehabilitation evidence for **cervical radiculopathy**, but its disease-defined population is narrower than the frozen C2 route.

Therefore:

```text
APTA 2017 neck pain with radiating pain
→ broad C2 route authority

2025 cervical-radiculopathy component NMA
→ narrower conditional evidence layer
→ only when formal or clinician-established radicular classification is present
```

This prevents disease-specific evidence from being silently applied to vague arm symptoms.

### Diagnostic / classification boundary

The APTA CPG describes a radiating-pain presentation that may include arm pain, paresthesia/numbness and sometimes sensory, strength or reflex deficits, with Spurling/distraction/neurodynamic testing used in classification. The available test properties support contextual interpretation rather than a single-test diagnosis.

CU-1 therefore enforces:

```text
radiating pain/paresthesia
!= objective neurological deficit

positive Spurling
!= formal cervical radiculopathy

positive neurodynamic test
!= formal cervical radiculopathy

not-assessed motor/sensory/reflex status
!= normal neurological examination
```

### Exact treatment mapping

```text
active rehabilitation backbone
→ APTA radiating-pain framework
→ referral_core
→ no universal exercise dose or transition threshold

acute radiating pain
→ mobilizing + stabilizing exercise may be used
→ Grade C APTA 2017
→ referral_core

chronic radiating pain
→ education/counseling encouraging occupational + exercise activity
→ Grade B APTA 2017
→ referral_core

chronic intermittent mechanical traction
→ Grade B APTA 2017 when combined with other interventions
→ CU-1 keeps it conditional on selected adjunct context
→ referral_core only when selected/applicable

2025 NMA traction component
→ associated with pain reduction in cervical-radiculopathy trials
→ clinician_ui_only because disease scope is narrower than all C2 presentations

2025 NMA neurodynamic component
→ associated with pain reduction
→ therapist_execution_detail

2025 NMA articular/manual component
→ associated with pain reduction
→ therapist_execution_detail

2025 NMA promising component combination
→ moderate-confidence network finding
→ clinician_ui_only
→ not converted to a mandatory physician-prescribed bundle
```

### Why the 2025 NMA is not a mandatory bundle

The component NMA found neurodynamic techniques, cervical traction and articular treatment associated with pain reduction and identified a promising multi-component combination. That is comparative network evidence within cervical-radiculopathy trials. It does not establish that every component is required for every patient, does not define a validated dose/progression protocol, and does not expand automatically to symptom-only C2 cases.

Therefore:

```text
NMA promising combination
!= required C2 treatment package

component association
!= physician-prescribed technique/dose
```

### Rehabilitation-sequence decision

A one-phase evidence-bounded sequence is supportable:

```text
active/function-preserving rehabilitation
+ acute mobilizing/stabilizing exercise when acute
+ chronic activity/education when chronic
+ selected chronic intermittent traction only when explicitly selected/applicable
→ no universal numeric progression threshold
→ no fixed traction force/duration/frequency prescription
→ no promise to reverse an objective neurological deficit
```

`progression_criteria` remains intentionally empty.

### Safety boundary

The routine C2 sequence is not applicable when there is unresolved:

```text
new/progressive objective motor deficit
progressive/expanding objective sensory deficit
gait change / upper-motor-neuron feature
possible myelopathic/cord feature
other material structural/neurological concern
```

Those findings trigger medical reassessment rather than routine progression.

### Route-specific history prompts

The C2 shard captures separately:

```text
radiating distribution/laterality
duration/course
objective motor/sensory/reflex status when assessed
Spurling/distraction/neurodynamic findings when actually examined
formal diagnosis + MRI/EMG-NCS context when supplied
progressive neurological/cord features
functional/work/activity impact
prior treatment response
```

### Exact gate

```text
canonical route identity                                  PASS
current broad C2 route authority identified              PASS
2025 rehab evidence freshness update identified          PASS
broad C2 vs disease-specific NMA scope                    PASS
subjective symptoms vs objective deficit separation      PASS
formal diagnosis not inferred                            PASS
single-test diagnosis prevention                         PASS
not-assessed neurological status != normal               PASS
explicit payload IDs                                     PASS
source/claim/profile/sequence references                 PASS
required profile fields                                  PASS
required sequence fields                                 PASS
output-scope compatibility                               PASS
NMA bundle not physician-mandated                        PASS
traction kept context-selective                          PASS
no fixed traction dose                                   PASS
no invented progression threshold                        PASS
progressive neuro/cord safety boundary                   PASS
route-specific history prompts                           PASS
matching regression fixtures                             PASS after cervical fixture update
no generic C1/C2 fallback                                PASS

C2 ROUTE PROFILE                                         PASS
C2 REHABILITATION SEQUENCE                               COMPLETE — SINGLE-PHASE EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                                       NO
```

### C2 activation decision

C2 is eligible for active design authority only after its matching cervical fixture cases are added and focused CI passes on the resulting exact head.

C2 does not authorize C3-C5 cervical routes.
