# Shared Muscle / Myotendinous Injury Physiotherapy Referral Profile v1.1 — CU-1

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Supersedes as active design:** `clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1.md`.
> **Purpose:** own reusable acute muscle/myotendinous injury rehabilitation semantics once across regions while preserving exact muscle/site, injury type, structural severity/context, loading restrictions, function, return-to-running/sport/work criteria and safety.
> **Runtime:** NOT AUTHORIZED.

---

# 1. Architectural role

This shared profile owns **acute muscle / myotendinous injury rehabilitation**. It does not absorb chronic tendinopathies or established tendon-repair pathways already owned by regional profiles.

```text
REGIONAL / SHARED ENTRY
→ established or clinically assessed acute muscle/myotendinous injury
→ SHARED MUSCLE / MYOTENDINOUS PROFILE
→ muscle/site + injury type + phase + structural context
→ restrictions + actual findings + functional demand
→ confirmed goals / rehabilitation directions
→ ShortReferralFormatter / DetailedReferralFormatter
```

Top-level structured route:

```text
acute_muscle_myotendinous_injury_rehabilitation
```

Hard architectural boundaries:

```text
acute muscle strain != chronic tendinopathy automatically
muscle pain != structural tear automatically
MRI/US grade != rehabilitation clearance
elapsed days/weeks != tissue readiness
pain reduction != restored load capacity
bony avulsion != muscle strain → Shared Fracture
complete free-tendon rupture / major tendon avulsion != routine strain pathway
postoperative tendon repair protocol > generic shared muscle suggestion
```

---

# 2. Inherited and muscle-specific invariants

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

pain after sprint/kick != muscle tear automatically
bruising != exact tear grade
tenderness != exact injured structure
weakness after injury != complete rupture automatically
normal walking != return-to-running clearance
pain-free jogging != return-to-sprinting clearance
strength symmetry alone != full return-to-sport readiness
one functional test != universal return-to-sport clearance
classification label != fixed rehabilitation timetable
child/adolescent apophyseal avulsion concern != muscle strain
calf pain/swelling != gastrocnemius strain until important vascular/other differential is addressed
```

---

# 3. Required structured injury context

## 3.1 Injury identity

```text
muscle_group
specific_muscle_optional
laterality: left / right / bilateral / midline / not_applicable
injury_date_optional
injury_phase:
  acute
  early_rehabilitation
  progressive_loading
  return_to_running_or_sport
  later_rehabilitation_with_residual_deficit
  not_stated
```

## 3.2 Injury type

```text
injury_type:
  clinically_assessed_muscle_strain
  confirmed_partial_muscle_or_myotendinous_tear
  confirmed_high_grade_muscle_or_myotendinous_tear_nonoperative
  muscle_contusion
  established_intramuscular_or_intratendinous_injury
  other_established_muscle_myotendinous_injury
  not_stated
```

Generic presentation wording may be carried without inventing a grade.

## 3.3 Tissue location / structural context

Optional only when established:

```text
injury_location:
  muscle_belly
  myofascial
  myotendinous_junction
  intramuscular_tendon_or_intratendinous
  proximal_tendon_region
  distal_tendon_region
  free_tendon_component
  other
  not_stated
```

Optional imaging/classification context:

```text
MRI_or_ultrasound_confirmed: yes / no / not_stated
classification_system_optional
classification_grade_optional
retraction_or_gap_cm_optional
haematoma_context_optional
number_of_tendons_involved_optional
```

```text
imaging/classification
→ may inform prognosis and structural context
→ never autonomously prescribes loading or return-to-sport timing
```

## 3.4 Management / restriction context

```text
management_context:
  conservative_rehabilitation
  specialist_review_underway
  postoperative_or_repair_protocol
  not_stated

explicit_loading_restriction_optional
explicit_ROM_or_stretch_restriction_optional
running_restriction_optional
sprinting_or_kicking_restriction_optional
sport_or_work_restriction_optional
surgeon_or_sports_medicine_instruction_optional
```

```text
postoperative_or_repair_protocol
→ exact protocol overrides shared generic suggestions
```

---

# 4. Frozen visibility / muscle registry

The registry controls navigation and wording, not automatic protocols.

## 4.1 High-visibility routine groups

### M1 — Quadriceps / rectus femoris acute injury

```text
quadriceps_rectus_femoris_muscle_injury
```

Visible subtypes/context:

```text
rectus_femoris_muscle_or_myotendinous_strain
proximal_rectus_femoris_myotendinous_injury
proximal_rectus_femoris_tendinous_or_free_tendon_injury
vastus_lateralis_medialis_intermedius_injury
quadriceps_contusion
other_quadriceps_injury
```

This is high visibility because it is common in the product-owner workflow, including recreational footballers and quadriceps contusions.

Hard boundaries:

```text
AIIS bony apophyseal avulsion → Shared Fracture
ASIS avulsion != rectus femoris injury by default
major free-tendon avulsion / major weakness / deformity / unresolved repairable injury
→ specialist disposition before routine strain wording
```

### M2 — Acute adductor injury

```text
acute_adductor_muscle_injury
```

Specific muscles:

```text
adductor_longus
adductor_brevis
adductor_magnus
gracilis
pectineus
other_adductor
```

Boundary:

```text
acute adductor strain/tear → Shared Muscle
chronic adductor-related groin pain / tendinopathy → Hip/Groin H3
```

This is high visibility because acute adductor injuries are frequent in the product-owner workflow.

### M3 — Gastrocnemius / soleus / calf injury

```text
calf_muscle_myotendinous_injury
```

Subtypes:

```text
medial_gastrocnemius_injury
lateral_gastrocnemius_injury
soleus_injury
plantaris_context_if_established
combined_calf_injury
other_calf_myotendinous_injury
```

Hard safety boundary:

```text
calf pain/swelling + unresolved DVT concern
→ no routine calf-strain wording
```

Achilles rupture is not collapsed into calf strain.

### M4 — Hamstring strain / partial myotendinous tear

```text
hamstring_muscle_injury
```

Optional muscles:

```text
biceps_femoris_long_head
biceps_femoris_short_head
semitendinosus
semimembranosus
proximal_hamstring_myotendinous_region
other_hamstring
```

Hamstring remains direct/high visibility because the product owner refers these injuries and the evidence base for loading/running/RTS is comparatively mature.

---

## 4.2 Visible but less frequent upper-limb / trunk groups

These remain directly searchable/selectable without occupying the first-line lower-limb menu.

### Pectoralis major muscle / myotendinous injury

```text
pectoralis_major_muscle_myotendinous_injury
```

Seen approximately a few times per year in the product-owner workflow.

Hard boundary:

```text
complete pectoralis-major tendon rupture / major axillary-fold deformity / major strength loss
→ structural specialist assessment
→ not routine muscle-belly strain wording
```

### Biceps muscle-belly / myotendinous injury

```text
biceps_muscle_belly_myotendinous_injury
```

Seen several times per year.

Hard boundary:

```text
proximal or distal biceps tendon rupture
→ relevant structural shoulder/elbow pathway
→ not muscle-belly strain route
```

### Abdominal-wall muscle injury

```text
abdominal_wall_muscle_injury
```

Seen several times per year and may include sport- or exertion-related abdominal-wall strain/tear when clinically established.

Hard boundary:

```text
abdominal/groin pain with unresolved hernia, visceral, pubic, hip or other non-muscular cause
→ no autonomous abdominal-muscle diagnosis
```

---

## 4.3 Rare / secondary groups

```text
acute_hip_flexor_iliopsoas_muscle_injury
tibialis_anterior_muscle_myotendinous_injury
gluteal_muscle_injury
other_thigh_muscle_injury
popliteus_muscle_injury
other_lower_leg_muscle_injury
intrinsic_foot_muscle_injury
latissimus_dorsi_or_teres_major_muscle_injury
triceps_muscle_belly_or_myotendinous_injury
forearm_muscle_strain
other_upper_limb_or_trunk_muscle_injury
other_lower_limb_muscle_injury
```

Iliopsoas/hip-flexor is rare in this workflow. Tibialis anterior is seen only occasionally and remains rare/secondary.

---

# 5. Major tear / tendon-avulsion / operative boundary

The shared route may carry a high-grade injury only when a conservative rehabilitation decision is established and no unresolved repairable structural concern remains.

High-priority contexts:

```text
complete proximal hamstring tendon avulsion / major retraction
complete proximal rectus femoris free-tendon tear / avulsion
complete proximal adductor tendon avulsion / retraction
complete pectoralis-major tendon rupture concern
proximal/distal biceps tendon rupture concern
other major free-tendon rupture
major palpable defect / sudden major loss of function
postoperative tendon or muscle repair
```

## 5.1 Product-owner retraction workflow and evidence-sensitive rule

The product owner commonly refers established conservatively managed avulsion-type injuries with **<2 cm retraction** to physiotherapy.

Frozen semantic treatment:

```text
retraction <2 cm
→ may support routine conservative/PT workflow
→ ONLY after clinician/specialist has established nonoperative/conservative management
→ never functions as autonomous software clearance
```

The 2 cm value is **not a universal deterministic threshold across all muscle/tendon injuries**. Current literature uses approximately 2 cm as an important factor in several proximal hamstring, proximal rectus-femoris and proximal adductor decision frameworks, but treatment also depends on number of tendons involved, complete vs partial avulsion, chronicity, functional demand, athlete level, symptoms and shared specialist/clinician decision.

```text
retraction >=2 cm
OR multi-tendon complete avulsion
OR major weakness/deformity
OR high-demand athlete with unresolved structural decision
→ make specialist-disposition check prominent
→ do not automatically recommend surgery
```

Routing rule:

```text
major tear / avulsion concern without disposition
→ specialist / imaging / structural reassessment
→ no routine strain rehabilitation wording

established conservative decision
→ shared muscle rehabilitation may proceed within entered restrictions

postoperative repair
→ exact surgeon/protocol restrictions > shared generic suggestion
```

Bony avulsions always route through Shared Fracture.

---

# 6. Examination findings — selectable only when assessed

## Symptoms/local findings

```text
localized pain
pain with stretch
pain with resisted contraction
pain with running/sprinting
pain with kicking/jumping/change of direction
pain with lifting/pushing/pulling where relevant
tenderness
swelling
bruising
palpable defect if actually found
muscle spasm/guarding
```

## ROM / length

```text
restricted relevant joint ROM
painful active ROM
painful passive stretch
reduced muscle-length tolerance
pain-free ROM restored if actually assessed
```

## Strength / capacity

```text
isometric strength deficit
isotonic strength deficit
eccentric/lengthened-position deficit
endurance deficit
power deficit
rate-of-force / explosive deficit if assessed
heel-raise capacity deficit for calf
knee-flexor deficit for hamstring
knee-extensor deficit for quadriceps
adduction deficit for adductor
load intolerance without quantified weakness
```

Pain-limited effort does not become a software diagnosis of complete tear.

## Functional / sport

```text
walking limitation
stairs
squat/lunge
single-leg stance
running intolerance
sprinting intolerance
acceleration/deceleration deficit
cut/change-of-direction deficit
jump/landing deficit
kicking deficit
sport-specific task deficit
work/manual-duty deficit
full training not yet tolerated
patient confidence/readiness concern
```

---

# 7. Core rehabilitation directions

Nothing is globally preselected.

Potential directions when structurally appropriate:

```text
education / activity and load modification
progressive pain-guided movement
progressive strength and load capacity
progressive isometric → isotonic → higher-load / lengthened-position work as appropriate
progressive eccentric/lengthening exposure where appropriate
kinetic-chain / adjacent-region strengthening when an actual deficit exists
progressive running exposure
progressive acceleration/deceleration
progressive sprint exposure for sprint-demand sports
progressive kicking/change-of-direction/jumping exposure where relevant
work-specific graded loading
home exercise programme
criterion-based return to training / sport / work
```

No single eccentric protocol or universal exercise sequence is frozen.

```text
hamstring
→ individualized loading + running/sprinting progression has stronger evidence/consensus base

quadriceps / adductor / calf
→ active progressive rehabilitation remains core
→ exact RTS thresholds are less strongly validated
→ avoid false precision
```

---

# 8. Return-to-running / sport / work semantics

Potential readiness domains:

```text
clinical symptom response
pain with palpation/stretch/contraction if relevant
ROM / muscle-length tolerance
strength / capacity relevant to muscle and task
high-speed or lengthened-position capacity where relevant
running tolerance
sprinting tolerance
kicking / jumping / change-of-direction tolerance where relevant
work-specific load tolerance
full or modified training exposure where relevant
subjective confidence / readiness
absence of deterioration after progressive exposure
clinician / surgeon restrictions
```

Hard rules:

```text
elapsed time alone != unrestricted RTS/work clearance
MRI healing appearance alone != RTS clearance
strength symmetry alone != universal clearance
pain-free jogging != sprint readiness
one full training session != universal proof of recovery
```

No fixed week-based return-to-running/sport timetable is generated.

---

# 9. Safety / reassessment semantics

## Structural

```text
major acute loss of function / inability to activate expected muscle
palpable defect with major weakness
complete tendon rupture / avulsion concern
major proximal rectus-femoris / proximal-hamstring / adductor avulsion concern
major pectoralis tendon rupture concern
new or progressive deformity
unexpected deterioration during rehabilitation
new trauma / reinjury
```

## Calf / vascular / compartment

```text
unexplained calf swelling / DVT concern
vascular compromise concern
severe escalating pain / tense swelling / compartment concern
new neurological deficit
```

## Haematoma / systemic

```text
large or expanding haematoma
anticoagulation context with significant bleeding concern
unexplained systemic illness / infection concern
rhabdomyolysis concern after severe exertional presentation
```

## Delayed / atypical course

```text
persistent worsening pain/swelling
failure to progress with unresolved diagnosis
myositis_ossificans_or_heterotopic_ossification_concern after significant contusion
persistent major weakness beyond expected course
other clinician concern
```

Safety state:

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

Disposition:

```text
reviewed_and_appropriate_to_proceed
imaging_or_sports_medicine_review_arranged
orthopaedic_or_surgical_review_arranged
urgent_or_same_day_assessment_arranged
routine_physiotherapy_deferred
other
```

No reassuring `no rupture`, `no DVT`, `no compartment syndrome` or similar wording is generated from missing assessment.

---

# 10. Adjunct / support policy

Acute muscle rehabilitation is loading/function governed rather than modality governed.

Treating-physiotherapist discretion rather than direct generator selection:

```text
compression
taping / sport-specific support
other short-term support
```

Clinician-selectable adjunct in the product-owner workflow:

```text
acupuncture
```

Semantics:

```text
acupuncture = optional adjunct when clinician-selected
acupuncture != core rehabilitation
acupuncture != evidence claim of accelerated tissue healing
acupuncture != substitute for progressive active loading
```

Excluded from this v1.1 generator:

```text
dry needling
ESWT as default acute-muscle healing treatment
therapeutic ultrasound as default acute-muscle healing treatment
routine electrical modalities as tissue-healing treatment
```

These exclusions are product/workflow decisions and do not assert universal scientific ineffectiveness.

---

# 11. Deterministic consistency rules

```text
muscle pain + no established injury context
→ presentation wording only; no exact tear diagnosis

MRI/US grade entered
→ carry as context; do not infer fixed rehabilitation timeline

complete-tendon / major avulsion concern + no disposition
→ no routine strain wording

retraction <2 cm + no established conservative decision
→ no autonomous PT clearance

retraction >=2 cm
→ prominent structural-disposition check, not automatic surgery recommendation

bony avulsion context
→ Shared Fracture

proximal rectus femoris + ASIS avulsion
→ invalid anatomical mapping by default

calf strain route + unresolved DVT concern
→ routine referral blocked / reassessment prompt

pain-free walking only
→ no running/sport clearance

return-to-sport request + no relevant high-demand assessment
→ warning against time-only clearance

postoperative repair + missing protocol/restrictions
→ warning

acupuncture selected + no active loading/functional rehabilitation direction
→ warning; adjunct does not replace core rehabilitation

material safety concern + no disposition
→ no routine reassuring wording
```

---

# 12. Evidence-governance boundary

Stable frozen evidence direction:

```text
progressive active loading and task-specific rehabilitation are core
return to sport/work should be criterion-based where possible
exact criteria depend on muscle, injury type, severity and sporting/work demands
hamstring evidence is more developed than quadriceps/adductor/calf evidence
2 cm retraction is context-sensitive, not a universal autonomous treatment threshold
```

Evidence anchors reviewed during candidate/freeze work include:

- London International Consensus on hamstring rehabilitation, running and return to sport;
- contemporary systematic review of lower-limb muscle-injury return-to-play criteria;
- current proximal-hamstring reviews and comparative/noninferiority evidence;
- systematic reviews and contemporary reviews of proximal adductor avulsion management;
- systematic review/meta-analysis of proximal rectus-femoris avulsion management;
- contemporary calf-strain literature emphasizing combined clinical and imaging assessment rather than imaging-only clearance.

Evidence-sensitive details to refresh before CU-2 implementation:

```text
exact hamstring high-speed-running progression
validated muscle-specific RTS test thresholds
proximal hamstring operative/nonoperative selection wording
adductor complete-avulsion operative/nonoperative wording
proximal rectus-femoris free-tendon/avulsion structural criteria
calf-specific return-to-running/RTS criteria
acupuncture adjunct evidence
```

---

# 13. Freeze decisions — product owner 2026-08-27

- quadriceps/rectus femoris, acute adductor and gastrocnemius/soleus/calf are the most common shared muscle referrals and receive highest visibility;
- hamstring strain/partial tear remains directly visible;
- proximal rectus-femoris/proximal-quadriceps myotendinous/tendon injury remains highly visible, including recreational footballers;
- acute adductor strain/tear is a separate active route from chronic adductor-related groin pain;
- gastrocnemius/soleus/calf strain is high visibility;
- quadriceps contusion is a direct choice;
- iliopsoas/hip-flexor acute strain is rare/secondary;
- tibialis-anterior injury is rare/secondary;
- pectoralis-major injury, biceps muscle-belly/myotendinous injury and abdominal-wall strain are visible but less frequent;
- major proximal hamstring/rectus-femoris/adductor avulsion requires established conservative/specialist disposition before routine rehabilitation wording;
- the clinician commonly refers conservatively managed avulsion injuries with <2 cm retraction, but the utility does not use 2 cm as autonomous clearance;
- bony avulsions remain entirely in Shared Fracture;
- no fixed week-based RTS and no autonomous clearance from MRI grade or strength symmetry alone;
- dry needling is excluded;
- acupuncture is clinician-selectable as an optional adjunct without tissue-healing efficacy claim;
- ESWT and therapeutic ultrasound are excluded as default acute-muscle healing recommendations;
- compression/taping remain treating-physiotherapist discretion rather than direct generator defaults.

This file is the frozen Shared Muscle / Myotendinous Injury clinical/content design for CU-1. Runtime implementation remains unauthorized.
