# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; all regional v1.1 profiles and Shared Fracture / Post-immobilization v1.1 frozen.

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

# 2. Frozen profile status

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
```

Authoritative shared-fracture design:

```text
clinic_utilities/physio_profiles/shared_fracture_v1_1.md
```

---

# 3. Shared Fracture / Post-immobilization v1.1 frozen design

The shared profile owns restriction/healing logic once rather than duplicating fracture protocols region by region.

```text
regional/shared fracture gateway
→ fracture_rehabilitation_post_immobilization
→ fracture_site
→ treatment / phase / healing-stability
→ immobilization/support
→ lower-limb weight-bearing OR upper-limb use/loading
→ ROM / strengthening / impact restrictions
→ actual deficits/function
→ confirmed goals/directions
```

Minimum shared context:

```text
fracture site
laterality
date/phase when known
treatment / surgery when applicable
healing/stability status
immobilization/support status
lower-limb weight-bearing when relevant
upper-limb use/loading when relevant
ROM restrictions
loading / strengthening / impact restrictions
orthopaedic/surgical instructions and source
age/skeletal maturity when relevant
```

Hard rules:

```text
fracture != healed fracture
elapsed time != union
cast/sling/boot removal != unrestricted loading
fixation != unrestricted use/loading
not stated != unrestricted
exact protocol > shared generic suggestion
no universal week-based timetable
manual therapy requires known stability + ROM permission
pediatric fracture != adult timeline
fragility fracture != automatic osteoporosis diagnosis
```

High-visibility workflow entries:

```text
vertebral compression / fragility fracture
proximal humerus
clavicle
distal radius
hand / finger fractures
pubic rami
patella
ankle fractures
calcaneus / anterior-process calcaneus
5th metatarsal / other metatarsal
foot / toe fractures
```

Less frequent / advanced / context includes scaphoid with union gate, elbow fractures, tibial plateau, Lisfranc, long-bone shaft fractures and older-adult hip fracture.

## 3.1 Fragility modifier

```text
formal_fragility_fracture_context
known_osteoporosis_or_low_bone_strength_context
falls_risk_or_recurrent_falls_context
```

When selected, the utility makes strength, balance, falls-risk reduction, mobility and functional independence prominent. It does not generate osteoporosis diagnosis, DXA orders or medication decisions.

## 3.2 SIFK / legacy SONK

Preferred entity:

```text
subchondral_insufficiency_fracture_of_knee
```

Frozen terminology:

```text
SIFK = preferred current term
SONK = legacy / clinician-entered wording, not a second autonomous software diagnosis
advanced SIFK may carry osteonecrosis / osteochondral collapse when established
```

```text
bone-marrow edema alone != SIFK
sudden knee pain alone != SIFK
SIFK + loading status unknown → no generic strengthening / impact progression
```

## 3.3 Pediatric / apophyseal

Pelvic apophyseal avulsions remain the clinically useful visible pediatric gateway. Other pediatric fractures are low visibility because the product owner rarely refers them. Adult timelines are never imported.

## 3.4 Default adjunct exclusions

Not generator-default fracture-healing recommendations:

```text
acupuncture
dry needling
ESWT
therapeutic ultrasound to accelerate union
bone-stimulator prescription
```

---

# 4. Safety / consistency engine

```text
fracture + healing/stability unknown
→ warning; no healed/stable wording

lower-limb fracture + weight-bearing unknown
→ no progressive weight-bearing instruction

upper-limb fracture + use/loading unknown
→ no unrestricted lifting/pushing/use instruction

ROM/loading restriction unknown
→ no unrestricted ROM/strengthening statement

new trauma / loss of reduction / delayed union / nonunion / hardware concern
→ orthopaedic reassessment semantics

infection / wound / neurovascular / compartment / DVT-PE concern
→ medical/urgent reassessment semantics

possible CRPS
→ preserve concern; do not autonomously diagnose

vertebral fracture + unresolved spinal precautions / neurological concern
→ medical/specialist pathway

SIFK / stress / insufficiency fracture + loading status unknown
→ no generic impact progression

pediatric fracture
→ no adult timeline

material safety concern + no clinician disposition
→ no routine reassuring wording
```

---

# 5. Shared muscle / myotendinous profile — next after fracture

Important future entries already generated by regional gateways include:

```text
proximal rectus-femoris tendon/myotendinous injury
adductor strain/tear
iliopsoas/hip-flexor strain
rectus-femoris strain
hamstring strain
gastrocnemius strain
soleus strain
calf myotendinous injury
other regional acute muscle/tendon injury
```

---

# 6. Context-sensitive goals / directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

Fracture rehabilitation is restriction-governed before generic active-rehabilitation suggestions become available.

---

# 7. Remaining shared design sequence

Current broad remaining sequence is:

```text
muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next profile within CU-1.

---

# 8. Output wording rules

```text
Clinical problem + fracture/site/treatment context + actual deficits + functional impact.
Referral request + goals.
Exact restrictions / permitted progression.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- no unsupported healing/stability assertion;
- no automatic timeline from elapsed weeks;
- no normal neurological/red-flag statement from missing data;
- preserve exact orthopaedic restrictions;
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
