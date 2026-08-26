# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1 and elbow v1.1 frozen; wrist/hand v1 active design candidate.

Clinic Utilities are cross-module clinician-facing operational tools, not a new clinical Module 02.

---

# 1. Detour purpose

Integrate useful day-to-day clinic tools into the Clinical Excellence workspace, beginning with the Physiotherapy Referral Generator clinical/content redesign. RF Request/PDF workflow remains a separate later utility slice.

CU-1 currently covers physiotherapy clinical/content design only.

---

# 2. Physiotherapy Referral v2 target

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

Then:

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
orthosis != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen profile status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
```

Frozen authoritative files:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
clinic_utilities/physio_profiles/lumbar_v1_1.md
clinic_utilities/physio_profiles/shoulder_v1_1.md
clinic_utilities/physio_profiles/elbow_v1_1.md
```

---

# 4. Wrist / Hand — ACTIVE DESIGN CANDIDATE

Candidate file:

```text
clinic_utilities/physio_profiles/wrist_hand_v1.md
```

Proposed default pathways:

```text
De Quervain / first dorsal compartment disorder
thumb CMC-1 osteoarthritis / rhizarthrosis
interphalangeal / generalized hand osteoarthritis
median neuropathy at wrist / carpal tunnel syndrome
ulnar-sided wrist pain / TFCC-related presentation
wrist extensor/flexor tendinopathy / overuse disorder
trigger finger / trigger thumb
thumb MCP UCL injury / instability rehabilitation
post-traumatic wrist/hand pain or stiffness after assessed injury
postoperative wrist/hand rehabilitation — pending workflow confirmation
```

Candidate rare/advanced/context entities:

```text
Guyon's canal / ulnar neuropathy at wrist
scapholunate/lunotriquetral ligament injury or carpal instability
ECU instability/subluxation
established CRPS upper limb
inflammatory / psoriatic / crystal hand context
Dupuytren disease / post-procedure context
ganglion/mass context
mallet / boutonniere / central-slip / flexor-extensor tendon injuries
```

Key candidate rules:

- De Quervain provocation tests remain findings; current comparative evidence does not justify wording that physiotherapy alone is evidence-preferred first-line management;
- thumb CMC-1 OA and interphalangeal hand OA remain separate phenotypes;
- orthosis and exercise have a meaningful role in CMC-1 OA;
- CTS symptoms remain separate from objective neurological deficit;
- Phalen/Tinel/Durkan and upper-limb neurodynamic findings do not independently establish CTS;
- progressive thenar weakness/atrophy requires reassessment semantics;
- uncomplicated carpal-tunnel release does not automatically generate routine supervised hand therapy;
- ulnar-sided wrist pain or TFCC provocation does not establish TFCC tear;
- TFCC conservative wording must preserve DRUJ stability and structural restrictions;
- ECU instability is distinct from ECU tendinopathy;
- acute thumb-UCL instability/Stener concern is not routine unrestricted rehabilitation;
- possible CRPS features do not create a formal CRPS diagnosis;
- fractures and complex repair/healing contexts remain protocol governed.

### Wrist / hand orthosis policy — candidate

```text
thumb spica → condition-sensitive
CMC-support orthosis → evidence-supported option for CMC-1 OA
neutral-wrist night orthosis → short-term CTS symptom-management option
trigger-digit orthosis → conservative option pending workflow confirmation
injury/post-op orthosis → exact protocol/restriction governed
```

### Wrist / hand adjunct policy — candidate

```text
manual therapy / mobilization → optional where relevant
soft tissue → optional
taping → optional
dry needling → only selected myofascial/tendinous context + competence safeguard
acupuncture → optional only if product owner confirms actual wrist/hand use
ESWT → not proposed as a default wrist/hand adjunct
therapeutic ultrasound → not standard evidence-backed CTS/general wrist-hand treatment
```

Current evidence backbone includes the 2024 AAOS CTS CPG, De Quervain network meta-analysis, current CMC-1 OA rehabilitation systematic reviews/RCTs, current TFCC nonoperative evidence, trigger-digit orthosis evidence, wrist-tendinopathy literature and CRPS rehabilitation guidance.

Wrist/hand remains **NOT FROZEN** until product-owner review.

---

# 5. Shared fracture / post-immobilization profile

Fractures should be handled in one shared profile rather than duplicated region by region.

Required future context:

```text
bone/site
fracture date/phase
treatment
healing/stability status if known
immobilization/orthosis status
weight-bearing/use status
ROM/loading restrictions
surgeon/orthopaedic/hand-surgeon instructions
```

Regional entry points now include shoulder, elbow and wrist/hand fractures, including distal radius/ulna, scaphoid/carpal, metacarpal and phalangeal fractures.

Unknown healing/loading context must produce a warning rather than unrestricted rehabilitation wording.

---

# 6. Context-sensitive goals and directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

Active rehabilitation, exercise, graded activity/loading, education, task adaptation and self-management remain the conceptual backbone where appropriate.

---

# 7. Safety / consistency engine

```text
fracture rehab + missing healing/use context
→ warning

post-op/tendon-repair route + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety/infection concern + no clinician disposition
→ do not generate routine reassuring wording

unassessed neurological component
→ never generate normal wording
```

Region-specific rules live in each frozen/candidate profile.

---

# 8. Remaining regional design sequence

After wrist/hand review/freeze, current preferred working sequence is:

```text
knee / hip
→ ankle / foot
→ shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next region within CU-1.

---

# 9. Output wording rules

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation/hand-therapy direction / restrictions.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- active/function-oriented rehabilitation as core where appropriate;
- orthoses are condition-sensitive supports rather than universal defaults;
- technique-level interventions remain adjuncts;
- no unsupported diagnosis from symptoms, tests or incidental imaging;
- no normal neurological/red-flag statement from missing data;
- preserve explicit restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 10. Implementation boundary

CU-1 remains **design only**.

First implementation direction remains conceptually:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. Do not write production physiotherapy runtime code until CU-1 is sufficiently frozen and the product owner explicitly authorizes transition to CU-2.
