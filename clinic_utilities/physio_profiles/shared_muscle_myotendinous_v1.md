# Shared Muscle / Myotendinous Injury Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** own reusable acute muscle/myotendinous injury rehabilitation semantics once across regions while preserving exact muscle/site, structural severity, tissue location, loading restrictions, function, return-to-running/sport/work criteria and safety.
> **Runtime:** NOT AUTHORIZED.

---

# 1. Architectural role

This shared profile owns **acute muscle / myotendinous injury rehabilitation**, not every tendon disorder.

Canonical routing model:

```text
REGIONAL / SHARED ENTRY
→ established or clinically assessed acute muscle/myotendinous injury
→ SHARED MUSCLE / MYOTENDINOUS PROFILE
→ muscle/site + injury type + phase + structural context
→ actual restrictions + findings + functional demand
→ confirmed goals / rehabilitation directions
→ ShortReferralFormatter / DetailedReferralFormatter
```

Proposed top-level structured route:

```text
acute_muscle_myotendinous_injury_rehabilitation
```

Regional menus may preselect `muscle_group` or `injury_site`, but this shared profile owns the reusable acute-injury and progression semantics.

Hard architectural boundaries:

```text
acute muscle strain != chronic tendinopathy automatically
muscle pain != structural tear automatically
MRI grade != rehabilitation clearance
elapsed days/weeks != tissue readiness
pain reduction != restored load capacity
bony avulsion != muscle strain → shared fracture profile
complete free-tendon rupture / major tendon avulsion != routine strain pathway
postoperative tendon repair protocol > generic shared muscle suggestion
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

Additional muscle-injury invariants:

```text
pain after sprint/kick != muscle tear automatically
bruising != exact tear grade
tenderness != exact injured structure
weakness after injury != complete rupture automatically
normal walking != return-to-running clearance
pain-free jogging != return-to-sprinting clearance
strength symmetry alone != full return-to-sport readiness
one functional test != universal return-to-sport clearance
classification label != fixed rehabilitation timetable
full-thickness tendon/avulsion concern != routine progressive-loading pathway
child/adolescent apophyseal avulsion concern != muscle strain
calf pain/swelling != gastrocnemius strain until DVT/other differential adequately addressed
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

A generic presentation may be carried without inventing an exact grade.

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
retraction_or_gap_optional
haematoma_context_optional
```

Hard rule:

```text
imaging/classification
→ may inform prognosis/context
→ does not autonomously prescribe loading or return-to-sport timing
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

# 4. Candidate muscle-group registry

The registry controls navigation/wording, not automatic protocols.

## 4.1 Posterior thigh / hamstrings

```text
hamstring_muscle_injury
```

Optional specific muscles:

```text
biceps_femoris_long_head
biceps_femoris_short_head
semitendinosus
semimembranosus
proximal_hamstring_myotendinous_region
other_hamstring
```

Candidate high-visibility because hamstring strains are common sports injuries and current evidence provides the strongest criterion-based rehabilitation/RTS framework among muscle groups.

Important structural boundary:

```text
proximal hamstring free-tendon avulsion / major retraction concern
→ specialist structural pathway
→ not routine strain rehabilitation
```

## 4.2 Anterior thigh / quadriceps / rectus femoris

```text
quadriceps_rectus_femoris_muscle_injury
```

Subtypes/context:

```text
rectus_femoris_muscle_or_myotendinous_strain
proximal_rectus_femoris_myotendinous_injury
proximal_rectus_femoris_tendinous_or_free_tendon_injury
vastus_lateralis_medialis_intermedius_injury
quadriceps_contusion
other_quadriceps_injury
```

The Hip/Groin frozen profile directly exposes proximal rectus-femoris injury because it occurs in the product-owner workflow.

Hard boundaries:

```text
AIIS bony apophyseal avulsion → shared fracture profile
ASIS avulsion != rectus femoris injury by default
complete proximal tendon tear / major avulsion / major weakness
→ structural/sports-medicine review before routine strain wording
```

## 4.3 Adductors / groin

```text
acute_adductor_muscle_injury
```

Optional muscles:

```text
adductor_longus
adductor_brevis
adductor_magnus
graciIis
pectineus
other_adductor
```

[Implementation note before freeze: normalize the gracilis key to ASCII and spell-check all machine keys.]

Boundary with frozen Hip/Groin profile:

```text
acute adductor strain/tear → shared muscle profile
chronic adductor-related groin pain / tendinopathy → Hip/Groin H3
```

Major proximal adductor tendon avulsion/retraction may require specialist structural assessment rather than routine strain rehabilitation.

## 4.4 Hip flexor / iliopsoas

```text
acute_hip_flexor_iliopsoas_muscle_injury
```

Optional context:

```text
iliopsoas_muscle_myotendinous_injury
rectus_femoris_hip_flexor_component
sartorius_injury
tensor_fasciae_latae_injury
other_hip_flexor_injury
```

Bony ASIS/AIIS/lesser-trochanter apophyseal avulsion belongs to the shared fracture profile.

## 4.5 Calf / lower leg

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

The frozen Ankle/Foot profile already exposes gastrocnemius, soleus and calf myotendinous gateways.

Hard safety boundary:

```text
calf pain/swelling + unresolved DVT concern
→ no routine calf-strain referral wording
```

Achilles rupture is not collapsed into calf strain.

## 4.6 Other lower-limb muscle injuries — candidate secondary/advanced

```text
gluteal_muscle_injury
other_thigh_muscle_injury
popliteus_muscle_injury
other_lower_leg_muscle_injury
intrinsic_foot_muscle_injury
other_lower_limb_muscle_injury
```

These remain secondary/advanced unless product-owner workflow justifies direct visibility.

## 4.7 Upper-limb / trunk acute muscle injuries — candidate scope decision

Potential shared entries:

```text
pectoralis_major_or_minor_muscle_injury
latissimus_dorsi_or_teres_major_muscle_injury
biceps_muscle_belly_or_myotendinous_injury
triceps_muscle_belly_or_myotendinous_injury
forearm_muscle_strain
abdominal_wall_muscle_injury
other_upper_limb_or_trunk_muscle_injury
```

These are candidate only. Dedicated structural tendon routes in shoulder/elbow/wrist-hand remain authoritative when a tendon tear/repair is the established problem.

---

# 5. Major tear / tendon-avulsion / operative boundary

The shared profile may carry a high-grade injury only when a conservative rehabilitation decision is established and no unresolved repairable structural concern remains.

High-priority structural contexts:

```text
complete proximal hamstring tendon avulsion / major retraction concern
complete proximal rectus femoris free-tendon tear / avulsion concern
complete adductor tendon avulsion with significant retraction concern
complete distal/proximal tendon rupture belonging to an established regional tendon pathway
major palpable defect / sudden major loss of function after trauma
postoperative tendon or muscle repair
```

Routing rule:

```text
major tear / avulsion concern without disposition
→ specialist / imaging / structural reassessment
→ do not generate routine strain rehabilitation wording

postoperative repair
→ exact surgeon/protocol restrictions
→ shared profile may support findings/goals but does not invent progression
```

Bony avulsions always route through Shared Fracture.

---

# 6. Examination findings — selectable only when actually assessed

## 6.1 Symptoms / local findings

```text
localized pain
pain with stretch
pain with resisted contraction
pain with running
pain with sprinting
pain with kicking
pain with jumping
pain with change of direction
pain with lifting/pushing/pulling where relevant
tenderness
swelling
bruising
a palpable defect if actually found
muscle spasm/guarding
```

## 6.2 ROM / length findings

```text
restricted relevant joint ROM
painful active ROM
painful passive stretch
reduced muscle-length tolerance
pain-free ROM restored if actually assessed
```

## 6.3 Strength / capacity findings

```text
isometric strength deficit if assessed
isotonic strength deficit if assessed
eccentric/lengthened-position deficit if assessed
endurance deficit
power deficit
rate-of-force / explosive deficit if assessed
heel-raise capacity deficit for calf context
knee-flexor deficit for hamstring context
knee-extensor deficit for quadriceps context
adduction deficit for adductor context
hip-flexion deficit for hip-flexor context
load intolerance without quantified weakness
```

Pain-limited effort must not become a software diagnosis of complete tear.

## 6.4 Functional/sport findings

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

The profile does not mandate one exercise sequence or one eccentric protocol for all muscles.

Hamstring-specific evidence direction:

```text
individualized loading + running/sprinting progression
→ stronger evidence/consensus base
→ still no universal fixed dosage or timing
```

Quadriceps/adductor/calf direction:

```text
progressive rehabilitation remains active/function-oriented
BUT
return-to-sport thresholds are less strongly validated
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
absence of new deterioration after progressive exposure
clinician / surgeon restrictions
```

Hard rules:

```text
elapsed time alone
→ never generate unrestricted return-to-sport/work clearance

MRI healing appearance alone
→ never generate return-to-sport clearance

strength symmetry alone
→ not sufficient as universal clearance

pain-free jogging
→ not equivalent to sprint readiness

one full training session
→ may be relevant context but is not universal proof of complete recovery
```

For hamstring injuries, pain-free sport-specific high-speed function, ROM/length tolerance, strength/capacity and athlete readiness are particularly relevant. For adductor, quadriceps and calf injuries the evidence base for exact thresholds is less certain.

---

# 9. Safety / reassessment semantics

## 9.1 Structural concerns

```text
major acute loss of function / inability to activate expected muscle
palpable defect with major weakness
complete tendon rupture / avulsion concern
major proximal rectus-femoris or proximal-hamstring avulsion concern
large/retracting adductor tendon tear concern
new or progressive deformity
unexpected deterioration during rehabilitation
new trauma / reinjury
```

## 9.2 Calf / vascular / compartment concerns

```text
unexplained calf swelling / DVT concern
vascular compromise concern
severe escalating pain / tense swelling / compartment concern
new neurological deficit
```

## 9.3 Haematoma / bleeding / systemic concerns

```text
large or expanding haematoma
anticoagulation context with significant bleeding concern
unexplained systemic illness / infection concern
rhabdomyolysis concern after severe exertional presentation
```

## 9.4 Delayed complication / atypical course

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

No `no rupture`, `no DVT`, `no compartment syndrome` or other reassuring negative wording is generated from missing assessment.

---

# 10. Adjunct / support policy — candidate

Acute muscle rehabilitation remains loading/function governed rather than modality governed.

Possible non-prescriptive supports:

```text
compression where clinically appropriate
short-term walking aid / support if required
sport-specific taping/support where relevant
```

Candidate optional adjuncts for product-owner decision:

```text
manual therapy / soft-tissue techniques
dry needling for selected myofascial findings after appropriate phase/competence review
acupuncture
```

Not proposed as default tissue-healing recommendations:

```text
therapeutic ultrasound
ESWT
routine electrical modalities as tissue-healing treatment
```

No adjunct displaces progressive active rehabilitation.

---

# 11. Deterministic consistency rules

```text
muscle pain + no established injury context
→ presentation wording only; no exact tear diagnosis

MRI grade entered
→ may be carried; do not infer fixed rehabilitation timeline

complete-tendon / avulsion concern + no specialist disposition
→ no routine strain wording

bony avulsion context
→ shared fracture route

proximal rectus femoris + ASIS avulsion
→ invalid anatomical mapping by default

calf strain route + unresolved DVT concern
→ routine referral blocked / reassessment prompt

pain-free walking only
→ no running/sport clearance

return-to-sport request + no functional high-demand assessment
→ warning against time-only clearance

postoperative repair + missing protocol/restrictions
→ warning

adjunct selected + no active loading/functional rehabilitation direction
→ warning

material safety concern + no disposition
→ no routine reassuring wording
```

---

# 12. Evidence-governance boundary

Stable candidate evidence direction:

```text
progressive active loading and task-specific rehabilitation are core
return to sport/work should be criterion-based where possible
exact criteria depend on muscle, injury type, severity and sporting/work demands
hamstring evidence is more developed than quadriceps/adductor/calf evidence
```

Evidence anchors reviewed for this candidate include:

- London International Consensus on hamstring rehabilitation, running and return to sport;
- contemporary systematic review of lower-limb muscle injury return-to-play criteria showing strongest evidence for hamstring and weaker evidence for adductor/quadriceps/calf criteria;
- systematic reviews of acute adductor injury management;
- systematic reviews/reviews of proximal rectus-femoris tears and avulsions;
- contemporary review of calf strains emphasizing combined clinical and imaging assessment rather than imaging-only clearance.

Evidence-sensitive details to refresh before CU-2 implementation:

```text
exact hamstring loading dosage / high-speed running progression
validated return-to-running and return-to-sport test thresholds
adductor complete-tear operative vs nonoperative wording
proximal rectus-femoris free-tendon / avulsion structural criteria
calf-specific return-to-running / return-to-sport criteria
adjunct evidence
```

---

# 13. Product-owner decisions required before freeze

1. Which acute muscle injuries do you actually see/referral most often: **hamstring, quadriceps/rectus femoris, adductor, gastrocnemius/soleus, hip flexor/iliopsoas**?
2. Should **hamstring strain/partial tear** be high visibility? Candidate recommendation: yes.
3. Should **proximal rectus femoris / proximal quadriceps tendon-myotendinous injury** remain high visibility because you already said you see these in athletes?
4. For **adductor acute strain/tear**, should it be high visibility separately from the frozen chronic adductor-related groin pathway?
5. For **gastrocnemius/soleus/calf strain**, should it be high visibility?
6. Do you see **quadriceps contusions** often enough to include as a direct choice, or rare/advanced?
7. Do you see **iliopsoas/hip-flexor acute strains** often enough for direct visibility?
8. Do you see **gluteal acute muscle tears**, popliteus, tibialis-anterior muscle injuries or intrinsic-foot muscle injuries often enough to expose, or keep rare/advanced?
9. Do you refer **pectoralis major, latissimus/teres major, biceps/triceps muscle-belly, forearm or abdominal-wall strains**? If not, keep upper-limb/trunk acute muscle injuries rare/advanced.
10. For **major proximal hamstring / proximal rectus femoris / adductor tendon avulsions**, agree that routine referral requires an established conservative decision or specialist disposition first?
11. Agree that **bony avulsions** remain entirely in Shared Fracture rather than this profile?
12. Agree that no **fixed week-based return-to-running/sport** timeline is generated and no single MRI grade/strength-symmetry threshold gives automatic clearance?
13. **Dry needling:** do you refer/select it for acute muscle injuries once past the immediate acute phase, or exclude?
14. **Acupuncture:** do you refer/select it for acute muscle injuries, or exclude?
15. **ESWT / therapeutic ultrasound:** candidate recommendation is exclude as default acute-muscle healing recommendations. Agree?
16. Do you want **compression / taping** directly selectable supports, or leave these to treating physiotherapist discretion?
17. Any recurring muscle injury in your practice missing from this registry?

This file remains **DESIGN CANDIDATE / NOT FROZEN** until product-owner review resolves these workflow decisions. Runtime implementation remains unauthorized.
