# CU-1 C3 Cervical Headache Route Review — 2026-08-29

> **Route:** `headache_with_cervical_msk_features`
> **Shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_cervical_headache_v1.yaml`
> **Fixture oracle:** `clinic_utilities/contracts/cu1_c3_cervical_headache_fixtures_v1.yaml`
> **Normative schema:** `clinic_utilities/contracts/cu1_history_timeline_schema_v1.yaml`
> **Review state:** EXACT HUMAN DESIGN REVIEW COMPLETE
> **Runtime authority:** NO

---

## 1. Frozen-route semantics preserved

The frozen cervical profile defines C3 as a route with two distinct wording states:

```text
headache_with_cervical_msk_features
+
formal_cervicogenic_headache_diagnosis = yes / no / not_stated
```

The reviewed design preserves the mandatory distinction:

```text
headache with cervical MSK features
!= formal cervicogenic headache diagnosis
```

The utility may carry an explicit clinician diagnosis when entered. It may not infer that diagnosis from neck pain, reduced/painful cervical ROM, occipital/suboccipital tenderness, trigger points, or reproduction/aggravation of headache with cervical movement or posture.

**Gate:** PASS

---

## 2. Source identity and freshness review

### ICHD-3 — current diagnostic-boundary authority

Reviewed source:

```text
International Headache Society Headache Classification Committee
International Classification of Headache Disorders, 3rd edition
11.2.1 Cervicogenic headache
Cephalalgia. 2018;38(1):1-211
```

Current online ICHD-3 entry remains available at review date.

Important scope preserved:

- cervicogenic headache is a secondary headache attributed to a cervical disorder/lesion;
- evidence of causation is required;
- reduced cervical ROM with worsening by provocative manoeuvres is only one possible causation criterion;
- common upper-cervical imaging findings are suggestive but not firm evidence of causation by themselves;
- clinical features such as side-locked pain, neck-movement provocation or pressure-provocation are not unique to cervicogenic headache;
- another ICHD-3 diagnosis must not better account for the headache.

The source is used only for diagnostic-boundary semantics. CU-1 does not implement autonomous ICHD diagnosis.

**Gate:** PASS

### APTA/JOSPT Neck Pain CPG 2017

Reviewed source:

```text
Blanpied et al.
Neck Pain: Revision 2017
J Orthop Sports Phys Ther. 2017;47(7):A1-A83
DOI 10.2519/jospt.2017.0302
```

Exact `neck pain with headache` recommendations preserved:

```text
acute:
  Grade B — supervised active mobility exercise
  Grade C — C1-2 self-SNAG may be used

subacute:
  Grade B — cervical manipulation/mobilization
  Grade C — C1-2 self-SNAG may be used

chronic:
  Grade B — cervical/cervicothoracic manipulation or mobilization
            combined with shoulder-girdle and neck stretching,
            strengthening and endurance exercise
```

The framework grade remains APTA-specific and is not relabelled as GRADE treatment-effect certainty.

The self-SNAG technique is represented as therapist execution detail. Manual therapy is evidence authority but not automatic treatment selection.

**Gate:** PASS

### Martins et al. 2026 GRADE systematic review

Reviewed source:

```text
Martins L, Collet P, Lafrance S, Demont A
Efficacy of nonsurgical interventions for the management of adults with cervicogenic headache
Ann Phys Rehabil Med. 2026;69(4):102070
DOI 10.1016/j.rehab.2025.102070
```

Search through September 2025; 29 RCTs; RoB 2.0 + GRADE.

Exact evidence posture preserved:

```text
manual therapy vs sham:
  short-term headache-intensity benefit signal
  low certainty

manual therapy at 12 months:
  durable benefit not established

exercise therapy:
  low-certainty / limited evidence for direct headache outcomes
  effect on headache symptoms remains uncertain

manual + exercise vs usual care:
  added efficacy remains uncertain
```

The review is restricted to diagnosed/classified cervicogenic headache and cannot authorize treatment-effect wording in presentation-only C3.

**Gate:** PASS

### Jung et al. 2024 PT network meta-analysis

Reviewed source:

```text
Jung A et al.
Physical Therapist Interventions to Reduce Headache Intensity, Frequency, and Duration in Patients With Cervicogenic Headache
Physical Therapy. 2024;104(2):pzad154
DOI 10.1093/ptj/pzad154
```

The NMA included diagnosed/classified CGH trials. Several combined interventions ranked highly for short-term outcomes, but key comparisons were low-certainty and the authors explicitly stated that no conclusive recommendation could be made.

Therefore:

```text
network rank
!= universal superiority
!= mandatory treatment bundle
```

Manipulation + dry needling, muscle-energy + exercise, soft-tissue + exercise, or dry needling + exercise cannot be auto-selected by CU-1 from ranking probabilities.

**Gate:** PASS

---

## 3. Profile split and applicability review

Two machine-distinct profiles are required and present.

### Presentation-only C3

```text
rep_c3_cervical_headache_presentation_v1
→ seq_c3_cervical_headache_presentation_v1
```

Applicability requires:

```text
C3 route selected
formal_cervicogenic_headache_diagnosis != yes
no unresolved material headache safety concern
no unresolved post-traumatic primary context
```

Disease-specific CGH systematic-review claims do **not** resolve here.

Generated clinical wording must remain presentation-level, e.g. headache with cervical musculoskeletal features, unless the clinician explicitly asserts CGH.

**Gate:** PASS

### Formal / clinician-established cervicogenic headache

```text
rep_c3_formal_cervicogenic_headache_v1
→ seq_c3_formal_cervicogenic_headache_v1
```

Applicability requires:

```text
formal_cervicogenic_headache_diagnosis = yes
no unresolved material headache safety concern
no unresolved post-traumatic primary context
```

The clinician-entered diagnosis may be faithfully rendered. It is not labelled as a CU-1 inference.

Current 2026/2024 disease-specific synthesis may resolve only in this profile.

**Gate:** PASS

---

## 4. Exact claim/output-scope review

```text
c3_cervical_features_not_formal_cgh
→ clinician_ui_only
→ prevents diagnosis inference

c3_ichd_causation_not_single_finding
→ clinician_ui_only
→ prevents single-finding or imaging-only causal inference

c3_apta_active_rehabilitation_context
→ referral_core
→ active cervical rehabilitation backbone

c3_acute_active_mobility_B
→ referral_core
→ acute-only Grade-B applicability

c3_self_SNAG_execution_detail_C
→ therapist_execution_detail
→ no automatic physician technique/dose prescription

c3_selected_manual_therapy_apta_context
→ referral_core authority
→ subacute/chronic only
→ evidence recommendation preserved, selection remains separate

c3_formal_cgh_manual_short_term_low_2026
→ referral_core authority only in formal CGH
→ low-certainty short-term signal only

c3_formal_cgh_exercise_headache_effect_uncertain_2026
→ clinician_ui_only
→ prevents false high-certainty direct headache-effect claim

c3_formal_cgh_nma_rankings_not_protocol_2024
→ clinician_ui_only
→ prevents ranking→protocol conversion

c3_headache_safety_reassessment
→ referral_core safety authority
```

No `clinician_ui_only` or `therapist_execution_detail` claim is used as automatic referral-core authority.

**Gate:** PASS

---

## 5. Pre-PASS corrections made during exact review

The initial native shard was corrected before this PASS record:

1. The acute Grade-B active-mobility claim was removed from a generic all-course authority reference and placed behind an explicit acute applicability condition.
2. APTA manual-therapy authority was restricted to subacute/chronic context at both claim and sequence-element level.
3. Manual-therapy evidence authority was separated from automatic treatment selection.
4. The 2026 low-certainty manual-therapy effect claim was restricted to formal/clinician-established CGH.
5. A synthetic phase-level `low` certainty label across mixed APTA and GRADE frameworks was removed; phase certainty remains `not_graded` while individual claims preserve their own certainty/strength.

**Gate after correction:** PASS

---

## 6. Rehabilitation-sequence review

Both C3 sequences are deliberately single-phase evidence-bounded sequences.

```text
active cervical mobility / cervical-scapulothoracic rehabilitation
+ course-specific active mobility when acute
+ selected manual adjunct when applicable
→ no fixed exercise dose
→ no fixed manual-therapy dose
→ no universal numeric progression threshold
→ no fixed PT frequency or total course duration
```

Both sequences intentionally contain:

```text
progression_criteria: []
```

This is not missing design. It is the reviewed evidence-gap behavior.

A materially new, markedly changed or progressive headache pattern or relevant neurological/vascular/systemic clinician concern is a safety/reassessment condition, not a progression stage.

**Gate:** PASS

---

## 7. Differential and cross-route boundaries

Preserved boundaries:

```text
cervical ROM restriction + headache provocation
!= formal CGH diagnosis

upper-cervical imaging abnormality
!= proven headache generator

known migraine / tension-type / other headache context
!= automatically excluded by C3 selection

recent whiplash / cervical trauma as primary context
→ C5 route review

material safety concern
→ routine C3 sequence blocked pending clinician disposition/reassessment
```

No generic C1 or generic MSK fallback is allowed.

**Gate:** PASS

---

## 8. Matching history prompts

C3 adds reviewed prompts for:

- headache location/pattern/frequency/intensity/duration;
- temporal relation to cervical symptoms;
- cervical provocation context;
- explicit formal-CGH clinician diagnosis state;
- other headache/differential context;
- materially new/changed/safety context;
- trauma/whiplash context;
- functional impact / patient-priority activity;
- prior headache/neck treatment and response.

Prompts are not auto-selected and do not infer negative findings when unanswered.

**Gate:** PASS

---

## 9. Regression fixture review

Matching fixtures are present in:

```text
clinic_utilities/contracts/cu1_c3_cervical_headache_fixtures_v1.yaml
```

They explicitly test:

- presentation-only C3 does not generate definitive CGH;
- ROM + provocation is not sufficient for diagnosis;
- upper-cervical imaging alone does not establish causation;
- formal CGH resolves the formal profile without false certainty;
- manual therapy remains selected rather than mandatory and does not gain a false durable-benefit claim;
- acute Grade-B mobility retains its framework grade;
- self-SNAG stays therapist execution detail;
- low-certainty NMA rankings do not become a best protocol;
- a known primary-headache context is not silently declared excluded;
- material headache safety concerns block routine progression;
- post-traumatic primary context routes to C5 review.

**Gate:** PASS

---

# 10. Formal result

```text
identity / required fields                    PASS
source identity / freshness                   PASS
external source references                    PASS
claim/profile/sequence references             PASS
route applicability                           PASS
formal-vs-presentation diagnostic boundary    PASS
output-scope compatibility                    PASS
framework-grade preservation                  PASS
newer GRADE certainty preservation            PASS
NMA ranking governance                        PASS
history-prompt coverage                       PASS for C3
matching regression fixtures                  PASS for C3
progression-gap behavior                      PASS
cross-route C3/C5 boundary                    PASS

C3 ROUTE REVIEW                               PASS
C3 DESIGN STATE                               ACTIVE-DESIGN-AUTHORITY ELIGIBLE
C3 SEQUENCE STATE                             SEQUENCE COMPLETE / EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                            NO
GLOBAL CU-1 DESIGN-COMPLETE                   NO
```

Manifest activation remains a separate repository-state transition after focused CI succeeds.
