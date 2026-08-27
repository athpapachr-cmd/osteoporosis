# Shared Fracture / Post-immobilization Physiotherapy Referral Profile v1.1 — CU-1

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Supersedes as active design:** `clinic_utilities/physio_profiles/shared_fracture_v1.md`.
> **Purpose:** own reusable fracture/post-immobilization rehabilitation semantics once across all regions, preserving site, treatment, healing/stability, immobilization, loading/use restrictions, orthopaedic instructions, pediatric/fragility context and safety.
> **Runtime:** NOT AUTHORIZED.

---

# 1. Architectural role

The shared fracture profile is one reusable route, not a collection of independent week-by-week regional protocols.

```text
REGIONAL / SHARED ENTRY
→ established fracture / post-immobilization route
→ SHARED FRACTURE PROFILE
→ site + treatment + phase + actual restrictions
→ confirmed findings / goals / rehabilitation directions
→ ShortReferralFormatter / DetailedReferralFormatter
```

Top-level structured route:

```text
fracture_rehabilitation_post_immobilization
```

Regional menus may preselect `fracture_site`, but this shared profile owns healing, stability, restriction and safety semantics.

Hard architectural rules:

```text
fracture site != rehabilitation clearance
elapsed time != union
pain reduction != fracture healing
radiograph mentioned != union confirmed
immobilization removed != unrestricted loading
surgical fixation != unrestricted loading
post-immobilization stiffness != permission for forceful mobilization
```

---

# 2. Inherited and fracture-specific invariants

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred

suspected fracture != established fracture-rehabilitation route
known fracture != healed fracture
stable at one review != permanently unrestricted
union != automatically full function
weight-bearing not stated != WBAT
upper-limb use not stated != unrestricted lifting/pushing
ROM restriction not stated != unrestricted ROM
surgeon instruction absent from record != no restriction
fragility mechanism != software diagnosis of osteoporosis
stress/insufficiency fracture != ordinary traumatic-fracture protocol
pediatric fracture != adult timeline
physeal/apophyseal injury != generic muscle strain
vertebral fracture != routine nonspecific back-pain pathway
```

---

# 3. Required structured fracture context

## 3.1 Identity and phase

```text
fracture_site
laterality: left / right / bilateral / midline / not_applicable
fracture_date_optional
fracture_phase:
  acute_established
  early_healing
  post_immobilization
  later_rehabilitation
  healed_with_residual_impairment
  not_stated
fracture_pattern_or_classification_optional
open_vs_closed_if_relevant_optional
```

The formatter never invents a classification from mechanism or shorthand imaging.

## 3.2 Fracture context

```text
fracture_context:
  traumatic
  fragility_or_insufficiency
  stress_or_bone_stress
  pediatric_physeal_or_apophyseal
  pathological_or_other_medical_context
  not_stated
```

## 3.3 Treatment

```text
treatment:
  nonoperative_observation_or_functional_treatment
  sling
  cast
  splint_or_orthosis
  boot
  closed_reduction_and_immobilization
  percutaneous_fixation
  ORIF
  intramedullary_fixation
  external_fixation
  arthroplasty_after_fracture
  other_operation
  mixed_or_staged_treatment
  not_stated
```

Exact procedure/free text may be carried.

## 3.4 Healing / stability

```text
healing_stability_status:
  documented_stable_for_current_rehabilitation
  healing_confirmed_with_restrictions
  union_or_healed_confirmed
  delayed_union_context
  nonunion_context
  malunion_context
  loss_of_reduction_or_instability_concern
  not_stated
```

```text
healing_stability_status = not_stated
→ warning
→ no generated claim of healed/stable fracture
```

## 3.5 Immobilization / support

```text
immobilization_status:
  none_currently
  sling
  cast
  removable_splint_or_orthosis
  brace
  boot
  other
  not_stated
immobilization_use_instructions_optional
```

Device presence does not define permitted movement/loading.

## 3.6 Lower-limb / axial weight-bearing

```text
weight_bearing_status:
  non_weight_bearing
  touch_down_or_toe_touch
  partial_weight_bearing
  weight_bearing_as_tolerated
  full_weight_bearing
  other_explicit_restriction
  not_applicable
  not_stated
```

No software conversion is made between these states.

## 3.7 Upper-limb use / loading

```text
upper_limb_use_status:
  no_functional_use
  hand_for_light_ADL_only
  light_use_with_explicit_limit
  no_lifting_or_pushing_through_limb
  lifting_limit_entered
  use_as_tolerated
  unrestricted_use_confirmed
  other_explicit_restriction
  not_applicable
  not_stated
```

A generic weight-bearing field is not sufficient for upper-limb fractures.

## 3.8 ROM / strengthening / impact restrictions

```text
ROM_status:
  unrestricted_ROM_confirmed
  ROM_allowed_with_specific_limits
  passive_only_or_assisted_only_context
  joint_or_direction_specific_restriction
  immobilized_no_ROM_currently
  not_applicable
  not_stated

loading_strengthening_status:
  unrestricted_loading_confirmed
  graded_loading_allowed
  strengthening_allowed_with_limits
  strengthening_not_yet_allowed
  impact_not_allowed
  sport_not_allowed
  other_explicit_restriction
  not_stated
```

Exact restrictions override generic suggestions.

## 3.9 Orthopaedic / surgical instructions

```text
orthopaedic_instructions_source:
  written_protocol_available
  explicit_clinician_instruction_entered
  patient_reported_instruction_only
  none_available_or_not_stated
follow_up_due_optional
repeat_imaging_due_optional
```

Patient-reported instructions remain labelled as patient-reported.

---

# 4. Frozen visibility / site registry

The registry controls navigation and wording only. It does not encode automatic timelines.

## 4.1 High-visibility routine choices in this workflow

```text
thoracic_or_lumbar_vertebral_compression_fragility_fracture
proximal_humerus_fracture
clavicle_fracture
distal_radius_fracture
metacarpal_fracture
phalangeal_hand_fracture
pelvic_ring_or_pubis_ramus_fracture
patella_fracture
lateral_malleolus_fracture
medial_malleolus_fracture
posterior_malleolus_fracture
bimalleolar_or_trimalleolar_fracture
calcaneus_fracture
anterior_process_calcaneus_fracture
fifth_metatarsal_fracture
other_metatarsal_fracture
phalangeal_foot_fracture
other_common_foot_fracture
```

Notes:

- distal radius is a high-visibility quick choice;
- hand/finger fractures are routinely visible;
- pubic-rami fractures are routinely visible;
- patella and ankle fractures are routinely visible;
- foot/metatarsal/toe fractures remain visible because the product owner refers them after immobilization;
- anterior-process calcaneus fracture is explicit because it occurs in the product-owner workflow and should not be hidden inside generic calcaneal wording.

## 4.2 Visible but less frequent / condition-sensitive

```text
radial_head_or_neck_fracture
olecranon_or_proximal_ulna_fracture
distal_humerus_fracture
other_elbow_fracture
distal_ulna_fracture
scaphoid_fracture
other_carpal_fracture
other_wrist_hand_fracture
tibial_plateau_or_proximal_tibia_fracture
proximal_fibula_fracture
talus_fracture
navicular_fracture
cuboid_or_cuneiform_fracture
Lisfranc_fracture_dislocation_or_post_treated_injury
other_ankle_foot_fracture
```

Scaphoid remains rare/advanced with a strong union/stability gate before loading/use progression.

## 4.3 Rare / advanced / context-only sites in this workflow

```text
humeral_shaft_fracture
radius_or_ulna_shaft_fracture
femoral_neck_fracture
intertrochanteric_fracture
subtrochanteric_fracture
femoral_shaft_fracture
acetabular_fracture
distal_femur_fracture
tibial_shaft_fracture
fibular_shaft_fracture
other_long_bone_shaft_fracture
```

Hip fracture in older adults is not a routine outpatient referral generated by the product owner; hospital/community rehabilitation context may be carried without making it a high-visibility pathway.

## 4.4 Fragility / insufficiency / bone-stress registry

```text
thoracic_or_lumbar_vertebral_compression_fragility_fracture
pelvic_or_sacral_insufficiency_fracture
subchondral_insufficiency_fracture_of_knee
femoral_neck_stress_fracture
tibial_bone_stress_injury
navicular_stress_fracture
fifth_metatarsal_stress_fracture
calcaneal_stress_fracture
other_established_bone_stress_or_insufficiency_injury
```

### SIFK / legacy SONK terminology

Preferred structured entity:

```text
subchondral_insufficiency_fracture_of_knee
```

Optional clinician-entered context:

```text
legacy_SONK_term_used
SIFK_with_osteonecrosis_or_collapse_context
other_established_subchondral_insufficiency_context
```

Frozen terminology rule:

```text
SIFK = preferred current term
SONK = legacy / historically used term, not a second autonomous diagnosis generated by software
advanced SIFK may include osteonecrosis / osteochondral collapse context when established
```

Hard rules:

```text
bone-marrow edema alone != SIFK
sudden knee pain alone != SIFK
MRI finding without clinician/diagnostic context != autonomous SIFK diagnosis
SIFK + loading status not stated → no generic strengthening / impact progression
SIFK != routine knee OA pathway
SIFK != meniscal-tear pathway merely because meniscal pathology coexists
```

## 4.5 Pediatric / adolescent registry

The pediatric fracture group remains active but low-visibility except for pelvic apophyseal avulsion, which is specifically encountered in the product-owner workflow.

```text
ASIS_apophyseal_avulsion
AIIS_apophyseal_avulsion
ischial_tuberosity_apophyseal_avulsion
lesser_trochanter_apophyseal_avulsion
other_pelvic_apophyseal_avulsion
pediatric_supracondylar_or_distal_humerus_fracture
pediatric_forearm_or_distal_radius_ulna_fracture
pediatric_tibia_or_ankle_fracture
pediatric_foot_fracture
other_physeal_or_apophyseal_fracture
```

Buckle fractures and other common pediatric fractures are not high-visibility in this workflow because the product owner rarely or never refers them.

```text
child fracture != adult rehabilitation timeline
apophyseal avulsion != muscle strain
pain-free child != fracture healed automatically
```

---

# 5. Vertebral compression / fragility fracture pathway

Vertebral compression/fragility fracture is an active shared-fracture entry.

Required context includes:

```text
established vertebral fracture diagnosis
level(s) if known
acute/subacute/healed phase
stability/spinal precautions if relevant
pain/function status
neurological status when clinically relevant
activity / lifting / movement restrictions if any
osteoporosis/fragility context only if clinician-established
```

Possible rehabilitation goals/directions after appropriate clearance:

```text
safe mobility and transfers
progressive walking / functional activity
posture / trunk extensor and lower-limb strengthening where appropriate
balance / falls-risk work where relevant
progressive ADL independence
education / safe movement within actual restrictions
```

Hard boundaries:

```text
vertebral fracture != nonspecific back-pain pathway
unstable fracture / unresolved spinal precautions != routine exercise referral
new/progressive neurological deficit or bowel/bladder/perineal concern → urgent medical pathway
no generic spinal manipulation recommendation
no loaded end-range flexion prescription from missing context
```

---

# 6. Fragility-fracture modifier

Selectable modifier:

```text
formal_fragility_fracture_context: yes / no / not_stated
known_osteoporosis_or_low_bone_strength_context: yes / no / not_stated
falls_risk_or_recurrent_falls_context: yes / no / not_assessed
```

When explicitly selected, the UI should make these goals prominent:

```text
restore safe mobility / independence
progressive strength / conditioning
balance / falls-risk reduction
walking confidence / endurance
safe ADL function
appropriate transfer / gait-aid use when relevant
```

Hard boundary:

```text
fragility fracture modifier
!= software diagnosis of osteoporosis
!= DXA recommendation unless clinician entered
!= medication decision support
!= automatic FLS workflow
```

---

# 7. Post-immobilization findings

Selectable only when present/assessed:

```text
pain / load-related pain
swelling / edema
joint stiffness
restricted active ROM
restricted passive ROM
muscle weakness
atrophy / deconditioning
grip/pinch deficit
calf or quadriceps weakness
reduced endurance
altered gait / reduced walking tolerance
reduced upper-limb functional use
balance / proprioceptive deficit
reduced dexterity
scar sensitivity / adherence after surgery
sensory change
fear / low confidence with loading or movement
work limitation
sport limitation
ADL limitation
```

```text
stiffness after cast != forceful mobilization automatically
weakness after immobilization != structural tendon rupture
pain with loading != loss of union automatically
persistent pain != normal healing automatically
```

---

# 8. Core rehabilitation directions

Nothing is globally preselected.

When permitted by documented stability/restrictions:

```text
education / self-management
edema management
safe ROM restoration within restrictions
progressive active / active-assisted movement
progressive strengthening when permitted
progressive upper-limb functional use when permitted
progressive weight-bearing / gait retraining when permitted
balance / proprioceptive rehabilitation when relevant
walking / endurance progression
functional task retraining
scar / desensitization management where relevant
home exercise programme
falls-risk / balance intervention after fragility fracture where clinically relevant
criterion-based return to work / gym / sport
```

No universal sequence or week-based protocol is generated.

---

# 9. Manual therapy / supports / adjunct policy

Possible carried supports:

```text
sling
cast
splint / orthosis
brace
boot
walking aid
crutches / frame
other prescribed support
```

These are context, not automatically physiotherapy-prescribed devices.

Manual therapy / joint mobilization is selectable only when:

```text
fracture stability/healing context supports rehabilitation
AND
ROM permissions are known
AND
no explicit restriction conflicts
```

Possible safe adjunctive techniques when relevant:

```text
manual therapy / joint mobilization within permitted ROM/loading
soft-tissue / scar techniques
edema-management techniques
sensory / desensitization strategies
```

Explicitly excluded as generator-default fracture-healing recommendations:

```text
acupuncture
dry needling
ESWT
therapeutic ultrasound to accelerate union
bone-stimulator prescription
```

These exclusions are workflow/safety decisions and do not assert universal scientific ineffectiveness.

---

# 10. Return-to-function / sport semantics

Possible progression inputs:

```text
healing/stability clearance
pain/irritability
ROM relevant to task
strength/capacity relevant to task
balance / gait / hop / landing where relevant
upper-limb grip/load tolerance where relevant
sport/work-specific demands
protective equipment / orthosis instructions
clinician/surgeon restrictions
patient confidence / functional readiness
```

Hard rule:

```text
elapsed weeks alone
→ never generate return-to-sport / unrestricted-load clearance
```

For children/adolescents, skeletal maturity, site/stability and orthopaedic advice remain explicit.

---

# 11. Safety / reassessment semantics

## 11.1 Structural / healing

```text
new trauma / reinjury before confirmed healing
increasing deformity or loss-of-reduction concern
unexpected or progressive inability to use / weight bear
new mechanical instability
persistent focal fracture-site pain with unresolved healing status
delayed-union / nonunion concern
malunion with major functional consequence
hardware failure / migration concern
new displacement concern
```

## 11.2 Neurovascular / compartment / thromboembolic

```text
new or progressive motor deficit
new or progressive sensory loss
new vascular compromise concern
severe escalating pain / tense swelling / compartment concern
lower-limb DVT / PE concern
```

## 11.3 Infection / wound

```text
fever / systemic illness with fracture/postoperative context
wound drainage / erythema / cellulitis
suspected deep infection
pin-site infection concern
nonhealing wound
```

## 11.4 CRPS

```text
possible_CRPS_features_not_formally_diagnosed
established_CRPS_diagnosis
```

```text
pain + edema + color/temperature change after fracture
!= automatic CRPS diagnosis
```

Possible features without formal diagnosis trigger reassessment semantics, not automatic labeling.

## 11.5 Vertebral/spinal

```text
new/progressive neurological deficit
bowel/bladder/perineal neurological concern
unstable fracture / unresolved spinal precautions
progressive severe pain / new trauma
```

## 11.6 Safety state

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

Disposition:

```text
reviewed_and_appropriate_to_proceed
orthopaedic_review_arranged
repeat_imaging_or_healing_review_arranged
urgent_or_same_day_assessment_arranged
routine_physiotherapy_deferred
other
```

No reassuring `fracture healed`, `stable`, `neurovascularly intact`, `no CRPS`, `no DVT` or `no infection` wording is generated from missing data.

---

# 12. Deterministic consistency rules

```text
fracture route + healing_stability_status = not_stated
→ warning; no stable/healed wording

lower-limb fracture + weight_bearing_status = not_stated
→ no progressive weight-bearing instruction

upper-limb fracture + upper_limb_use_status = not_stated
→ no unrestricted lifting/pushing/use wording

ROM_status = not_stated
→ no unrestricted ROM wording

loading_strengthening_status = not_stated
→ no unrestricted strengthening/impact wording

orthopaedic protocol present
→ exact protocol/restriction overrides shared generic suggestion

postoperative fixation + no procedure/restrictions
→ warning

immobilization removed + healing status unknown
→ no unrestricted loading inference

fracture date + elapsed interval
→ never infer union from time alone

stress/insufficiency fracture + impact/loading status unknown
→ no return-to-running/impact plan

pediatric fracture + adult timeline suggestion
→ invalid

apophyseal avulsion + generic muscle-strain route only
→ warning; fracture profile owns healing/restrictions

vertebral fracture + nonspecific back-pain route only
→ warning when vertebral fracture is clinically material

SIFK/legacy SONK + OA/meniscus route only
→ warning when established subchondral fracture is clinically material

possible CRPS features without established diagnosis
→ no CRPS label; reassessment prompt

fragility mechanism only
→ no software osteoporosis diagnosis

manual therapy selected + stability/ROM permissions unresolved
→ invalid/warning

material safety concern + no disposition
→ no routine reassuring wording
```

---

# 13. Evidence-governance boundary

Stable frozen evidence principle:

```text
early movement / loading may improve recovery in selected stable fractures
BUT
exact timing is site-, pattern-, fixation-, stability- and protocol-specific
```

Therefore v1.1 freezes restriction/state semantics, not universal timelines.

Evidence anchors reviewed during candidate/freeze work include:

- contemporary distal-radius fracture rehabilitation guidance;
- recent systematic reviews supporting shorter immobilization / earlier rehabilitation in selected distal-radius and proximal-humerus fractures;
- WAX multicentre RCT on early weight-bearing after selected operatively treated ankle fractures;
- contemporary evidence on early weight-bearing after selected tibial-plateau fixation;
- NICE hip-fracture mobilisation guidance;
- contemporary pediatric fracture literature emphasizing site/stability-specific recommendations;
- current evidence for rehabilitation after vertebral and pelvic fragility fractures;
- 2024–2026 reviews on SIFK/legacy SONK nomenclature and management.

SIFK terminology frozen from current evidence:

```text
preferred term = subchondral insufficiency fracture of the knee (SIFK)
legacy SONK terminology is not preferred as a separate disease label
osteonecrosis/collapse may represent advanced SIFK when established
```

Evidence-sensitive details to refresh before CU-2 implementation:

```text
site-specific immobilization / early-mobilization evidence
site-specific weight-bearing/use progression
postoperative fixation precautions
scaphoid / talus / navicular / 5th-metatarsal union semantics
vertebral fragility-fracture movement/exercise wording
SIFK loading and collapse-risk wording
pediatric return-to-sport wording
CRPS reassessment wording
```

---

# 14. Freeze decisions — product owner 2026-08-27

- vertebral compression/fragility fracture is an active dedicated shared-fracture entry;
- frequently referred sites include distal radius, proximal humerus, hand/fingers, pubic rami, patella, ankle, foot/toes and anterior-process calcaneus;
- clavicle fracture is directly visible;
- long-bone shaft fractures remain rare/advanced because they are not part of the routine workflow;
- distal radius is a high-visibility quick choice;
- scaphoid remains less frequent/advanced with a strong union-confirmation gate;
- hip fracture in older adults is context-only rather than a routine outpatient-referral pathway;
- pubic-rami fracture is an active referral entry;
- patella and ankle fractures are active visible entries;
- metatarsal/5th-metatarsal/foot/toe fractures remain active because they are referred after immobilization;
- fragility-fracture modifier is active and makes balance/falls/strength/independence goals prominent without becoming osteoporosis treatment decision support;
- pediatric fracture navigation remains low-visibility except for pelvic apophyseal avulsion; buckle fracture is not a routine referral;
- no fixed week-based ROM/weight-bearing/strengthening timeline is generated without explicit protocol/instruction;
- manual therapy/joint mobilization requires known stability and ROM permission;
- acupuncture, dry needling, ESWT and therapeutic ultrasound are excluded as default fracture-healing recommendations;
- SIFK is added as an established subchondral insufficiency/bone-stress entry; `SONK` is retained only as legacy/clinician-entered terminology or advanced SIFK/osteonecrosis-collapse context;
- anterior-process calcaneus fracture is added as an explicit foot fracture site.

This file is the frozen Shared Fracture / Post-immobilization clinical/content design for CU-1. Runtime implementation remains unauthorized.