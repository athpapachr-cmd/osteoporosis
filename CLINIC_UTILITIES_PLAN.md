# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; all regional v1.1 profiles and Shared Fracture / Post-immobilization v1.1 frozen; Muscle / Myotendinous Injury v1 active design candidate.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. Physiotherapy Referral v2 target

```text
1. Clinical problem / diagnosis
2. Important findings
3. Functional limitation
4. Precautions / restrictions
5. Rehabilitation goals
6. Rehabilitation direction
7. Final referral text
```

Structured intermediate model:

```text
ReferralDraft
  patient_context
  body_region
  primary_problem
  secondary_problems[]
  laterality
  chronicity
  key_findings[]
  functional_impairments[]
  precautions[]
  explicit_restrictions[]
  goals[]
  rehab_directions[]
  adjunct_options[]
  reassessment_criteria[]
  sessions_optional
  clinician_free_text_optional
```

```text
ReferralDraft
→ ShortReferralFormatter
→ DetailedReferralFormatter
```

Hard rules:

```text
suggested != examined
suggested != selected
selected != clinically mandatory
symptom != diagnosis
objective deficit != subjective symptom
provocation/special test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
brace/orthosis/taping != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

---

# 2. Frozen / active profile status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
shared_fracture_v1_1 = FROZEN
shared_muscle_myotendinous_v1 = ACTIVE DESIGN CANDIDATE / NOT FROZEN
```

Authoritative candidate:

```text
clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1.md
```

---

# 3. Shared Muscle / Myotendinous Injury v1 — active candidate frame

The shared profile owns reusable acute muscle-injury rehabilitation semantics once rather than duplicating them in Hip/Groin, Knee or Ankle/Foot.

```text
regional/shared gateway
→ acute_muscle_myotendinous_injury_rehabilitation
→ muscle group / specific muscle
→ injury type / phase / tissue location when established
→ conservative vs specialist/postoperative context
→ actual restrictions / findings / functional demands
→ confirmed goals / rehabilitation directions
```

Existing frozen regional gateways include:

```text
proximal rectus-femoris tendon/myotendinous injury
adductor strain/tear
iliopsoas / hip-flexor strain
rectus-femoris strain
hamstring strain
gastrocnemius strain
soleus strain
calf myotendinous injury
other acute regional muscle injury
```

Candidate registry also considers other lower-limb and rare upper-limb/trunk muscle injuries, pending product-owner workflow decisions.

Hard boundaries:

```text
acute strain != chronic tendinopathy
muscle pain != exact tear diagnosis
MRI classification != fixed rehabilitation timetable
elapsed time != return-to-sport readiness
bony avulsion → Shared Fracture
complete free-tendon rupture / major tendon avulsion → structural/specialist route
postoperative repair → exact protocol governs progression
```

The profile must not duplicate dedicated frozen tendon/pathology pathways, including rotator cuff, distal biceps/triceps, wrist/hand tendon repairs, Achilles tendinopathy/rupture structural context or peroneal tendon pathways.

---

# 4. Core rehabilitation / return-to-function principle

Potential active directions where structurally appropriate:

```text
education / load modification
progressive ROM / length tolerance where relevant
progressive strength / load capacity
lengthened-position / eccentric exposure where appropriate
running progression
sprinting progression for sprint-demand sports
kicking / jumping / change-of-direction progression where relevant
work-specific graded loading
home exercise programme
criterion-based return to training / sport / work
```

No one loading mode is mandatory for every muscle.

Return-to-sport/work uses multiple domains rather than elapsed time or one test:

```text
symptoms
ROM / length tolerance
strength / capacity
high-speed / task-specific exposure
running / sprinting / kicking / jumping / change of direction where relevant
work/sport demands
training exposure
patient confidence/readiness
explicit clinician/surgeon restrictions
```

Evidence governance:

```text
hamstring = comparatively mature rehabilitation/RTS consensus
adductor / quadriceps / calf = less certain exact thresholds
→ avoid false precision
```

---

# 5. Major structural / shared-profile boundaries

```text
proximal hamstring major tendon avulsion / retraction concern
proximal rectus-femoris major free-tendon tear / avulsion concern
major adductor tendon avulsion / retraction concern
other complete tendon rupture
→ specialist structural pathway before routine strain rehabilitation

bony apophyseal avulsion
→ Shared Fracture / Post-immobilization

chronic tendinopathy
→ appropriate frozen regional profile
```

Postoperative repair may use the shared profile for findings/goals, but the exact surgical protocol owns restrictions and progression.

---

# 6. Safety / consistency engine

```text
major acute weakness / palpable defect / rupture concern
→ structural reassessment semantics

calf pain/swelling + unresolved DVT concern
→ no routine calf-strain wording

large/expanding haematoma or anticoagulation bleeding concern
→ medical reassessment semantics

severe escalating pain / tense swelling / compartment concern
→ urgent pathway

new neurological/vascular deficit
→ urgent/medical reassessment

postoperative repair + missing protocol/restrictions
→ warning

pain-free walking only
→ no running/sport clearance

elapsed time or MRI grade alone
→ no automatic return-to-sport clearance

material safety concern + no disposition
→ no routine reassuring wording
```

---

# 7. Remaining shared design sequence

Current sequence:

```text
muscle / myotendinous injury — ACTIVE CANDIDATE
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next profile after the active writer lock closes.

---

# 8. Output wording rules

```text
Clinical problem + muscle/site/injury context + actual findings + functional impact.
Referral request + goals.
Actual restrictions and permitted progression.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- no unsupported exact tear grade;
- no automatic return timeline from elapsed time;
- no normal safety statement from missing data;
- preserve explicit specialist/surgical restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 9. Implementation boundary

CU-1 remains **design only**.

First implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. Do not write production physiotherapy runtime code until CU-1 is sufficiently frozen and the product owner explicitly authorizes transition to CU-2.
