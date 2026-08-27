# Shared Fracture / Post-immobilization Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** own reusable fracture/post-immobilization rehabilitation semantics once across all regions, preserving exact fracture site, treatment, healing/stability, immobilization, loading/use restrictions, orthopaedic instructions, pediatric/fragility context, safety and physiotherapist autonomy.
> **Consumes gateways from frozen profiles:** shoulder, elbow, wrist/hand, knee, hip/groin and ankle/foot; cervical/lumbar trauma remains safety-gated and vertebral-fracture inclusion is a product-owner decision in this candidate.

---

# 1. Architectural role

The shared fracture profile is **not** a long list of independent regional treatment protocols.

Canonical routing model:

```text
REGIONAL ENTRY POINT
→ established fracture / post-immobilization route
→ SHARED FRACTURE PROFILE
→ site-specific context + actual restrictions
→ confirmed goals / rehabilitation directions
→ ShortReferralFormatter / DetailedReferralFormatter
```

One shared top-level structured route:

```text
fracture_rehabilitation_post_immobilization
```

Regional menus may preselect `fracture_site`, but the shared profile owns the reusable healing/restriction/safety logic.

Hard architectural rule:

```text
fracture site != rehabilitation clearance
elapsed time != union
pain reduction != fracture healing
radiograph mentioned != union confirmed
immobilization removed != unrestricted loading
surgical fixation != unrestricted loading
```

---

# 2. Inherited CU-1 invariants

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
```

Additional fracture-specific invariants:

```text
suspected fracture != established fracture-rehabilitation route
known fracture != healed fracture
stable at one review != permanently unrestricted
union != automatically full function
post-immobilization stiffness != permission for forceful mobilization
weight-bearing status not stated != weight bearing as tolerated
upper-limb use status not stated != unrestricted pushing/lifting
no listed ROM restriction != unrestricted ROM unless that is actually known
no listed surgeon instruction != no restriction
fragility mechanism != osteoporosis diagnosis automatically
low-trauma fracture != automatically safe for immediate exercise
stress/bone-stress injury != ordinary traumatic fracture protocol
pediatric fracture != adult timeline
physeal/apophyseal injury != generic muscle strain
vertebral fracture != routine nonspecific back-pain pathway
```

---

# 3. Required structured fracture context

The profile should not produce unrestricted routine rehabilitation wording until clinically material restrictions are resolved or explicitly carried as unknown.

## 3.1 Fracture identity

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

The formatter must not invent a classification from site/mechanism/imaging shorthand.

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

These are context labels only when clinician-established.

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

Optional exact procedure/free text remains available.

## 3.4 Healing / stability state

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

Hard rule:

```text
healing_stability_status = not_stated
→ warning
→ no generated claim of healed/stable fracture
```

## 3.5 Immobilization / support state

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

Device presence does not itself define permitted movement/loading.

## 3.6 Lower-limb / axial loading status

When relevant:

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

No conversion is made between these states.

## 3.7 Upper-limb use / loading status

When relevant:

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

A generic `weight bearing` field is not sufficient for upper-limb fractures.

## 3.8 ROM / loading restrictions

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

Exact restrictions/free text override generic suggestions.

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

Patient-reported instructions may be carried as such but must not be silently upgraded to verified surgical protocol.

---

# 4. Candidate fracture-site registry

The site registry provides navigation and wording. It does **not** encode automatic timelines.

## 4.1 Shoulder girdle / arm

```text
proximal_humerus_fracture
humeral_shaft_fracture
clavicle_fracture
scapular_fracture
other_shoulder_arm_fracture
```

Frozen shoulder gateways already include proximal humerus, clavicle and scapula.

## 4.2 Elbow

```text
distal_humerus_fracture
radial_head_or_neck_fracture
olecranon_or_proximal_ulna_fracture
coronoid_fracture
other_elbow_fracture
```

## 4.3 Forearm / wrist / hand

```text
radius_or_ulna_shaft_fracture
distal_radius_fracture
distal_ulna_fracture
scaphoid_fracture
other_carpal_fracture
metacarpal_fracture
phalangeal_fracture
other_forearm_wrist_hand_fracture
```

Scaphoid remains structurally sensitive because union status may materially alter loading/use decisions.

## 4.4 Hip / pelvis / femur

```text
femoral_neck_fracture
intertrochanteric_fracture
subtrochanteric_fracture
femoral_shaft_fracture
acetabular_fracture
pelvic_ring_or_pubis_ramus_fracture
sacral_or_pelvic_insufficiency_fracture
other_hip_pelvis_femur_fracture
```

## 4.5 Knee / leg

```text
distal_femur_fracture
patella_fracture
tibial_plateau_or_proximal_tibia_fracture
proximal_fibula_fracture
tibial_shaft_fracture
fibular_shaft_fracture
other_knee_leg_fracture
```

## 4.6 Ankle / foot

```text
lateral_malleolus_fracture
medial_malleolus_fracture
posterior_malleolus_fracture
bimalleolar_or_trimalleolar_fracture
fibular_fracture_or_Maisonneuve_context
talus_fracture
calcaneus_fracture
navicular_fracture
cuboid_or_cuneiform_fracture
fifth_metatarsal_fracture
other_metatarsal_fracture
phalangeal_foot_fracture
Lisfranc_fracture_dislocation_or_post_treated_injury
other_ankle_foot_fracture
```

## 4.7 Stress / bone-stress / insufficiency sites

Candidate shared context rather than ordinary acute-fracture protocol:

```text
femoral_neck_stress_fracture
pelvic_or_sacral_stress_insufficiency_fracture
tibial_bone_stress_injury
navicular_stress_fracture
fifth_metatarsal_stress_fracture
calcaneal_stress_fracture
other_established_bone_stress_injury
```

Hard rule:

```text
bone-stress diagnosis + loading status not stated
→ no generic progressive impact / running plan
```

## 4.8 Pediatric / adolescent physeal / apophyseal fracture sites

Candidate navigation group:

```text
pediatric_clavicle_fracture
pediatric_supracondylar_or_distal_humerus_fracture
pediatric_forearm_or_distal_radius_ulna_fracture
pediatric_tibia_or_ankle_fracture
pediatric_foot_fracture
ASIS_apophyseal_avulsion
AIIS_apophyseal_avulsion
ischial_tuberosity_apophyseal_avulsion
lesser_trochanter_apophyseal_avulsion
other_pelvic_apophyseal_avulsion
other_physeal_or_apophyseal_fracture
```

Pediatric fractures use skeletal-maturity context and orthopaedic clearance/restrictions; adult healing/loading timelines are not imported.

## 4.9 Vertebral fracture — candidate inclusion pending product-owner workflow confirmation

Possible site/context:

```text
thoracic_or_lumbar_vertebral_compression_fragility_fracture
other_established_spinal_fracture_after_specialist_clearance
```

If retained, it must be separate from nonspecific cervical/lumbar pain and require established diagnosis plus stability/activity restrictions.

Possible vertebral-fracture core rehabilitation themes after appropriate clearance:

```text
safe mobility / transfers
progressive walking / functional activity
posture / trunk extensor and lower-limb strengthening as appropriate
balance / falls-risk work where relevant
progressive ADL independence
education / safe movement within actual restrictions
```

No generic spinal manipulation or loaded end-range flexion prescription is generated from the shared profile.

---

# 5. Post-immobilization findings / impairments

Selectable only when actually present/assessed:

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

Hard rules:

```text
stiffness after cast != forceful mobilization automatically
weakness after immobilization != structural tendon rupture
pain with loading != loss of union automatically
persistent pain != normal healing automatically
```

---

# 6. Core rehabilitation directions

Nothing is globally preselected.

Potential shared directions when permitted by the documented healing/restriction state:

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
scar / desensitization management when relevant
home exercise programme
falls-risk / balance intervention after fragility fracture when clinically relevant
criterion-based return to work / gym / sport
```

The profile does not prescribe a universal sequence or week-by-week protocol.

---

# 7. Return-to-function / sport semantics

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

For children/adolescents, skeletal maturity, fracture stability/site and orthopaedic advice remain explicit; no adult return-to-sport timeline is copied.

---

# 8. Fragility / insufficiency fracture context

Candidate optional context:

```text
formal_fragility_fracture_context: yes / no / not_stated
known_osteoporosis_or_low_bone_strength_context: yes / no / not_stated
falls_risk_or_recurrent_falls_context: yes / no / not_assessed
```

If a fragility fracture is explicitly established, the referral may include relevant rehabilitation goals such as mobility, strength, balance, falls-risk reduction and restoration of independence.

Hard boundaries:

```text
low-energy mechanism alone != software-diagnosed osteoporosis
fracture referral utility != osteoporosis medication decision support
fragility-fracture context may prompt coordination / follow-up wording
but does not invent DXA / drug / FLS actions not entered by clinician
```

---

# 9. Pediatric / adolescent fracture semantics

Required when relevant:

```text
age
skeletal_maturity_context_optional
physis_open_or_closed_if_known
physeal_or_apophyseal_involvement_if_established
sport / school PE goal
orthopaedic restrictions
```

Hard rules:

```text
child fracture != adult rehabilitation timeline
apophyseal avulsion != muscle strain
radiographic accessory ossification center != fracture automatically
pain-free child != fracture healed automatically
```

No rigid universal pediatric immobilization or return-to-sport duration is generated.

---

# 10. Safety / reassessment semantics

## 10.1 Structural/healing concerns

```text
new trauma / reinjury before confirmed healing
increasing deformity or loss-of-reduction concern
unexpected or progressive inability to use / weight bear
new mechanical instability
persistent focal fracture-site pain beyond expected course with unresolved healing status
delayed-union / nonunion concern
malunion with major functional consequence
hardware failure / migration concern
new displacement concern
```

These are reassessment prompts, not software diagnoses.

## 10.2 Neurovascular / compartment / thromboembolic concerns

```text
new or progressive motor deficit
new or progressive sensory loss
new vascular compromise concern
severe escalating pain / tense swelling / compartment concern
lower-limb DVT / PE concern
```

## 10.3 Infection / wound concerns

```text
fever / systemic illness with fracture or postoperative context
wound drainage / erythema / cellulitis
suspected deep infection
pin-site infection concern
nonhealing wound
```

## 10.4 CRPS / disproportionate post-fracture presentation

Possible concern state:

```text
possible_CRPS_features_not_formally_diagnosed
established_CRPS_diagnosis
```

Hard rule:

```text
pain + edema + color/temperature change after fracture
!= automatic CRPS diagnosis
```

Possible CRPS features without formal diagnosis trigger clinician reassessment semantics rather than automatic labeling.

## 10.5 Vertebral / spinal safety if included

```text
new/progressive neurological deficit
bowel/bladder/perineal neurological concern
unstable fracture or spinal precautions not resolved
progressive severe pain / new trauma
```

No routine spinal rehabilitation wording supersedes these concerns.

## 10.6 Safety state

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

Disposition when concern present:

```text
reviewed_and_appropriate_to_proceed
orthopaedic_review_arranged
repeat_imaging_or_healing_review_arranged
urgent_or_same_day_assessment_arranged
routine_physiotherapy_deferred
other
```

No reassuring `fracture healed`, `stable`, `neurovascularly intact`, `no CRPS`, `no DVT` or `no infection` wording is generated from missing information.

---

# 11. Supports / adjuncts

Fracture rehabilitation is restriction-governed rather than modality-governed.

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

Optional rehabilitation techniques when safe and relevant:

```text
manual therapy / joint mobilization only after fracture stability and ROM permissions support it
soft-tissue / scar techniques where appropriate
edema-management techniques
sensory/desensitization strategies
```

Not generator-default fracture-healing treatments:

```text
therapeutic ultrasound to accelerate union
ESWT to accelerate union
acupuncture as fracture-healing treatment
dry needling at/around an incompletely healed fracture
bone stimulator prescription
```

The utility does not claim these are universally ineffective; it simply does not make them default fracture-healing recommendations.

---

# 12. Deterministic consistency rules

```text
fracture route + healing_stability_status = not_stated
→ warning
→ no stable/healed wording

lower-limb fracture + weight_bearing_status = not_stated
→ no progressive weight-bearing instruction generated

upper-limb fracture + upper_limb_use_status = not_stated
→ no unrestricted lifting/pushing/use wording generated

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

fracture date entered + elapsed interval
→ never infer union from time alone

stress fracture + impact status unknown
→ no return-to-running/impact plan

pediatric fracture + adult timeline suggestion
→ invalid

apophyseal avulsion + generic muscle-strain pathway only
→ warning; route through fracture context

vertebral fracture + nonspecific back-pain pathway only
→ warning if established vertebral fracture is clinically material

possible CRPS features without established diagnosis
→ no CRPS label; reassessment prompt

fragility mechanism only
→ no software diagnosis of osteoporosis

material safety concern + no disposition
→ no routine reassuring referral wording
```

---

# 13. Evidence-governance boundary

Stable candidate evidence direction:

```text
early movement / loading may improve recovery in selected stable fractures
BUT
exact timing is fracture-, fixation-, stability- and protocol-specific
```

Therefore this profile freezes **restriction/state semantics**, not universal timelines.

Evidence anchors reviewed for this candidate include:

- 2024 JOSPT Clinical Practice Guideline for Distal Radius Fracture Rehabilitation;
- 2024 systematic reviews/meta-analyses supporting shorter immobilization / earlier rehabilitation in selected distal radius and proximal humerus fractures;
- 2024 WAX multicentre RCT showing early weight-bearing after selected operatively treated ankle fractures was non-inferior to delayed weight-bearing;
- 2025 systematic review/meta-analysis suggesting early weight-bearing may be safe after selected tibial plateau fixation, with heterogeneity requiring individualized protocols;
- NICE Hip Fracture guidance supporting physiotherapy assessment and mobilisation the day after surgery unless medically or surgically contraindicated;
- contemporary pediatric fracture literature showing fracture stability/site matter and that immobilization/return-to-sport recommendations vary substantially;
- 2025 scoping review showing limited evidence and heterogeneous rehabilitation strategies after pelvic fragility fracture;
- 2026 evidence supporting exercise/rehabilitation in osteoporotic vertebral fracture while acute-phase programme details remain heterogeneous.

Evidence-sensitive details to refresh before CU-2 implementation:

```text
site-specific immobilization and early-mobilization evidence
site-specific weight-bearing/use progression
postoperative fracture fixation precautions
scaphoid / talus / navicular / 5th-metatarsal high-risk union semantics
vertebral fragility-fracture movement/exercise wording
pediatric fracture return-to-sport wording
CRPS post-fracture reassessment wording
```

---

# 14. Candidate product-owner decisions required before freeze

1. Should **vertebral compression/fragility fracture** be an active shared-fracture rehabilitation entry, or remain osteoporosis/spine context only?
2. Which fracture regions do you actually refer commonly after immobilization: shoulder/proximal humerus, elbow, distal radius/wrist, hand/fingers, hip/pelvis, patella/tibial plateau, ankle/foot?
3. Do you refer **clavicle fractures** after immobilization often enough for direct visibility?
4. Do you refer **humeral/forearm/femoral/tibial shaft fractures**, or should long-bone shaft entries remain advanced/context?
5. For **distal radius fractures**, should this be a high-visibility quick choice given its frequency and the importance of ROM/grip/function after immobilization?
6. For **scaphoid fractures**, should this be rare/advanced with a strong union-confirmation gate?
7. For **hip fracture in older adults**, do you personally generate outpatient physiotherapy referrals, or is this usually handled by hospital/community services and therefore context-only in your workflow?
8. For **pelvic/pubis-rami/sacral insufficiency fractures**, do you refer to outpatient physiotherapy?
9. For **tibial plateau / patella / ankle fractures**, which are common enough to deserve direct quick choices?
10. For **metatarsal/5th-metatarsal and foot fractures**, do you refer after immobilization or primarily leave footwear/offloading to orthopaedics/podiatry?
11. Do you want a dedicated **fragility fracture** modifier that makes falls/balance/functional-independence goals prominent, without turning this utility into osteoporosis treatment decision support?
12. Do you want the **pediatric fracture group** active beyond pelvic apophyseal avulsions — e.g. supracondylar humerus, forearm/buckle, tibial/ankle fractures — or should these remain rare/context?
13. Do you agree that **no fixed week-based ROM/weight-bearing/strengthening timeline** should ever be generated unless an explicit protocol/instruction was entered?
14. Do you agree that **manual therapy/joint mobilization** should only become selectable when stability and ROM permissions are known, rather than appearing as a generic post-cast option?
15. Do you agree to exclude acupuncture/dry needling/ESWT/therapeutic ultrasound as default fracture-healing recommendations?
16. Any recurring fracture type in your practice missing from this registry?

This file remains **DESIGN CANDIDATE / NOT FROZEN** until product-owner review resolves these workflow decisions. Runtime implementation remains unauthorized.
