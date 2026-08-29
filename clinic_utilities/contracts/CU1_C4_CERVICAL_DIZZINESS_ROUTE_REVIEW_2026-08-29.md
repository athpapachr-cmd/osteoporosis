# CU-1 C4 Cervical Dizziness Route Review — 2026-08-29

> **Route:** `cervical_dizziness_presentation`
> **Shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_cervical_dizziness_v1.yaml`
> **Fixture oracle:** `clinic_utilities/contracts/cu1_c4_cervical_dizziness_fixtures_v1.yaml`
> **Normative schema:** `clinic_utilities/contracts/cu1_history_timeline_schema_v1.yaml`
> **Review state:** EXACT HUMAN DESIGN REVIEW COMPLETE
> **Runtime authority:** NO

---

## 1. Frozen-route semantics preserved

The frozen cervical profile defines C4 as a symptom/presentation route with a separate optional clinician diagnosis assertion:

```text
cervical_dizziness_presentation
+
clinician_diagnosis_cervicogenic_dizziness = yes / no / not_stated
```

The reviewed design preserves the mandatory distinction:

```text
dizziness / disequilibrium / light-headedness
+ cervical musculoskeletal features
!= cervical causation
!= clinician-established cervical/cervicogenic dizziness diagnosis
```

The utility may faithfully carry a clinician-entered diagnosis when explicitly entered. It may not infer cervical causation from neck pain, stiffness, restricted ROM, symptom aggravation with head/neck movement, cervical or sensorimotor test findings, or improvement after neck-directed therapy.

**Gate:** PASS

---

## 2. Source identity, scope and freshness review

### Bárány Society Position on Cervical Dizziness — current diagnostic-boundary authority

Reviewed source:

```text
Seemungal BM et al.
The Bárány Society position on 'Cervical Dizziness'
J Vestib Res. 2022;32(6):487-499
DOI 10.3233/VES-220202
```

The current position states that evidence supporting a mechanistic link between illusory self-motion/vertigo and neck pathology or neck pain is lacking. It emphasizes common alternative explanations such as migraine/vestibular migraine and BPPV, with central vestibular and other potentially dangerous causes relevant by clinical context.

Most important governance conclusions:

```text
no accepted consensus diagnosis
no agreed diagnostic test
Bárány Society does not propose preliminary clinical diagnostic criteria for routine use
therapeutic response cannot establish the existence or diagnosis of cervical dizziness
no specific therapy is recommended by the position paper
```

The position paper also treats whiplash as outside its scope because head/brain acceleration and vestibular/TBI mechanisms make causal attribution more complex. This supports a hard C4→C5 boundary for primary post-traumatic presentations.

**Gate:** PASS

### Canli et al. 2026 physiotherapy systematic review

Reviewed source:

```text
Canli K, De Greef I, Van Looveren E, Meeus M, Cagnie B, De Meulemeester K
The effects of physiotherapy on neck pain with associated symptoms, including cervicogenic dizziness and tinnitus
BMC Musculoskelet Disord. 2026;27(1):244
DOI 10.1186/s12891-026-09664-6
```

The review included 13 studies and 785 participants. Methodological quality materially limits inference: 10 studies were high risk of bias, one had some concerns and two were low risk of bias.

Evidence posture preserved:

- no specific evidence-based physiotherapy guideline was identified for these associated-symptom presentations;
- Mulligan mobilization showed a short-term cervical-ROM signal in the reviewed evidence;
- balance was similar to placebo in the relevant short-term comparison;
- dizziness, pain, disability and other functional outcomes were limited or conflicting across the evidence base;
- intermediate- and long-term conclusions are limited;
- a technique-level ROM finding is not global treatment superiority and is not diagnostic evidence.

CU-1 therefore does not convert this review into a general physiotherapy-efficacy statement or a required protocol.

**Gate:** PASS

### Carrasco-Uribarren et al. 2025 GRADE systematic review/meta-analysis

Reviewed source:

```text
Carrasco-Uribarren A et al.
Is manual therapy effective for cervical dizziness?
BMC Musculoskelet Disord. 2025;26:659
DOI 10.1186/s12891-025-08899-z
```

Six primary RCTs and three secondary analyses were included. The review used GRADE and found outcome-specific certainty rather than one uniform evidence grade.

Exact posture preserved:

```text
dizziness intensity with upper-cervical manual therapy:
  low certainty

dizziness impact/disability and related outcomes:
  include very-low-certainty evidence

neck pain intensity:
  very-low-certainty evidence and no clear significant change in the pooled estimate
```

The review concludes that evidence favoring upper-cervical interventions for dizziness impact/intensity is low to very low certainty. CU-1 therefore does not manufacture a single synthetic `low` certainty across outcomes.

Manual therapy remains a selected adjunct, not mandatory route content, and no durable universal benefit is asserted.

**Gate:** PASS

### Piromchai et al. 2023 non-traumatic self-exercise RCT

Reviewed source:

```text
Piromchai P, Toumjaidee N, Srirompotong S, Yimtae K
The efficacy of self-exercise in a patient with cervicogenic dizziness
Front Neurol. 2023;14:1121101
DOI 10.3389/fneur.2023.1121101
```

This was a small randomized study of 32 people with non-traumatic clinician-classified cervicogenic dizziness. At two weeks, the self-exercise group had better DHI and NDI outcomes than control, while pain, cervical ROM and posturography did not significantly differ between groups.

The intervention included muscle, mobilization and oculomotor self-exercise. The exact exercise bundle, frequency and dose are population/intervention-specific study details and are not converted into a universal physician referral prescription or total physiotherapy-course duration.

**Gate:** PASS

### Tarnutzer et al. 2025 interdisciplinary dizziness guidance

Reviewed source:

```text
Tarnutzer AA et al.
Diagnosis and treatment of vertigo and dizziness: Interdisciplinary guidance paper for clinical practice
HNO. 2025;73(Suppl 3):357-369
DOI 10.1007/s00106-025-01599-z
```

The guidance supports a structured dizziness history and clinical evaluation based on timing, duration, triggers versus exacerbators, accompanying symptoms and syndrome context, with attention to potentially dangerous neurological and other causes.

CU-1 uses this source only for history/safety semantics. It is not cervical-causation evidence and does not authorize autonomous diagnostic classification by the referral utility.

**Gate:** PASS

---

## 3. Two-context C4 architecture

### A. Presentation-only C4 — explicit evidence-gap behavior

```text
rep_c4_cervical_dizziness_presentation_v1
rehabilitation_sequence_id: null
```

Applicability requires:

```text
C4 route selected
clinician_diagnosis_cervicogenic_dizziness != yes
no unresolved material dizziness safety concern
no unresolved primary post-traumatic/whiplash context
```

This context is deliberately `blocked_evidence_gap` for disease-specific rehabilitation sequencing.

Reason:

- cervical causation is not established by route selection;
- Bárány does not endorse routine clinical diagnostic criteria;
- the treatment evidence comes from populations already diagnosed/classified as cervical/cervicogenic dizziness and cannot automatically be applied as disease-specific authority to symptom-only C4;
- no generic cervical fallback is allowed.

A clinician may still add an explicit treatment instruction under the separate clinician-instruction authority model. Such an instruction must not be relabelled as literature recommendation.

**Gate:** PASS

### B. Explicit clinician-established cervical dizziness — cautious evidence-bounded sequence

```text
rep_c4_clinician_established_cervical_dizziness_v1
→ seq_c4_clinician_established_cervical_dizziness_v1
```

Applicability requires:

```text
clinician_diagnosis_cervicogenic_dizziness = yes
no unresolved material dizziness safety concern
no unresolved primary post-traumatic/whiplash context
```

The clinician-entered diagnosis may be faithfully rendered, but CU-1 does not mark it as validated by Bárány criteria or as a diagnosis generated by the software.

The one-phase sequence permits cautious individualized active cervical rehabilitation and selected manual therapy while preserving current uncertainty.

**Gate:** PASS

---

## 4. Exact claim/output-scope review

```text
c4_dizziness_neck_features_not_cervical_causation
→ clinician_ui_only
→ prevents symptom + neck-feature → causal diagnosis inference

c4_barany_no_clinical_diagnostic_criteria
→ clinician_ui_only
→ prevents CU-1 diagnostic-rule generation and treatment-response-as-proof

c4_alternative_causes_not_excluded
→ clinician_ui_only
→ prevents route selection from silently excluding vestibular/migraine/neurovascular/etc causes

c4_structured_dizziness_history_and_safety_context
→ clinician_ui_only
→ history prompt authority, not causal diagnosis

c4_current_physio_evidence_limited_conflicting_2026
→ clinician_ui_only
→ prevents false universal physiotherapy efficacy/protocol claim

c4_selected_active_cervical_rehabilitation
→ referral_core only in clinician-established context
→ cautious may-consider direction; exact bundle/dose not prescribed

c4_selected_manual_therapy_low_certainty_2025
→ referral_core only in clinician-established context and when selected as adjunct
→ outcome-specific low/very-low certainty preserved in text, no synthetic cross-outcome grade

c4_mulligan_rom_not_global_superiority_2026
→ therapist_execution_detail
→ ROM signal cannot become required technique or global superiority

c4_balance_sensorimotor_only_if_deficit_identified
→ therapist_execution_detail
→ no automatic vestibular/balance programme

c4_dizziness_safety_reassessment
→ referral_core safety authority
```

No `clinician_ui_only` or `therapist_execution_detail` claim automatically renders as referral-core authority.

**Gate:** PASS

---

## 5. Rehabilitation-sequence review

Only the clinician-established context receives a disease-specific sequence.

```text
cautious individualized active cervical rehabilitation
+ selected manual adjunct when explicitly chosen/applicable
→ no automatic vestibular rehabilitation
→ no fixed exercise/manual technique bundle
→ no fixed dose
→ no universal numeric progression threshold
→ no fixed PT frequency or total course duration
→ no promise of dizziness resolution
```

The sequence intentionally contains:

```text
progression_criteria: []
```

This is reviewed evidence-gap behavior, not missing design.

Balance/oculomotor/sensorimotor techniques remain therapist execution details and require an identified relevant deficit/treatment rationale. The small 2023 RCT does not freeze its exact protocol into physician-generated referral content.

**Gate:** PASS

---

## 6. Safety and differential boundaries

Preserved hard boundaries:

```text
neck pain + dizziness
!= cervical causation

head/neck movement aggravation
!= diagnostic proof

positive cervical torsion / head-neck differentiation / proprioceptive finding
!= accepted diagnostic criterion

improvement after neck treatment
!= diagnostic proof

C4 selected
!= migraine/BPPV/vestibular/neurological/vascular/cardiovascular/otological causes excluded

new acute/progressive dizziness or focal neurological/gait/otological/vascular concern
→ reassessment / clinician-selected diagnostic pathway before routine C4 progression

primary recent whiplash/post-traumatic context
→ C5 review
```

The utility does not autonomously determine urgency and does not generate a global `no red flags` statement.

**Gate:** PASS

---

## 7. Matching history prompts

C4 adds reviewed prompts for:

- timing/course/episode duration;
- triggers versus exacerbators, including gravity-positional versus head/neck movement versus spontaneous symptoms;
- temporal relationship to neck symptoms without causal inference;
- explicit clinician cervical-dizziness diagnosis state;
- migraine/headache context;
- otological/vestibular context and known BPPV/vestibular diagnoses;
- neurovascular/cardiovascular/neurological safety context;
- prior ENT/neurology/vestibular or other specialist evaluation when known;
- trauma/whiplash context;
- balance/gait and patient-priority functional impact;
- prior treatment and response.

Prompts are not auto-selected, and unanswered prompts do not become negative findings.

**Gate:** PASS

---

## 8. Regression fixture review

Matching fixtures are present in:

```text
clinic_utilities/contracts/cu1_c4_cervical_dizziness_fixtures_v1.yaml
```

They explicitly test:

- neck pain + dizziness does not generate a cervical-dizziness diagnosis;
- presentation-only C4 does not inherit disease-specific treatment-effect claims;
- positive cervical/sensorimotor test does not establish diagnosis;
- treatment response does not establish diagnosis;
- clinician diagnosis selects the formal profile without becoming CU-1 diagnostic validation;
- manual therapy is optional and outcome-specific certainty is preserved;
- the Mulligan ROM signal does not become balance/global superiority or a mandatory technique;
- the small 2023 self-exercise RCT does not freeze its exact protocol/dose or a two-week PT course;
- vestibular rehabilitation is not an automatic default;
- alternative causes are not silently declared excluded;
- material dizziness safety concern blocks routine progression;
- primary post-traumatic/whiplash context routes to C5 review.

**Gate:** PASS

---

## 9. Pre-PASS correction

The initial C4 shard used a single `low` machine certainty on the manual-therapy claim even though the 2025 GRADE review reports different certainty by outcome.

Before this PASS record it was corrected to:

```text
certainty_optional: not_graded
```

with the exact outcome-specific low/very-low certainty preserved in the claim and review text. This avoids manufacturing a synthetic certainty across outcomes.

**Gate after correction:** PASS

---

# 10. Formal result

```text
identity / required fields                         PASS
source identity / freshness                        PASS
claim/profile/sequence references                  PASS
route/context applicability                        PASS
presentation-vs-clinician-diagnosis boundary       PASS
Bárány no-criteria/no-causal-inference governance  PASS
alternative-cause non-exclusion behavior           PASS
output-scope compatibility                         PASS
outcome-specific GRADE preservation                PASS
2026 limited/conflicting evidence posture          PASS
active-rehabilitation scope                        PASS with narrow clinician-established context
manual-therapy optionality                         PASS
vestibular-rehab non-default behavior               PASS
history-prompt coverage                            PASS for C4
matching regression fixtures                       PASS for C4
progression-gap behavior                           PASS
C4/C5 trauma boundary                              PASS

C4 ROUTE REVIEW                                    PASS as context split
PRESENTATION-ONLY C4                               BLOCKED-EVIDENCE-GAP by design
CLINICIAN-ESTABLISHED C4                           SEQUENCE COMPLETE / EVIDENCE-BOUNDED
C4 DESIGN STATE                                    ACTIVE-DESIGN-AUTHORITY ELIGIBLE
RUNTIME AUTHORIZED                                 NO
GLOBAL CU-1 DESIGN-COMPLETE                        NO
```

Manifest activation remains a separate repository-state transition after focused CI succeeds.
