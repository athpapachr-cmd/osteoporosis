# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1, elbow v1.1 and wrist/hand v1.1 frozen.

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
orthosis != automatically mandatory
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
```

---

# 3. Wrist / Hand v1.1 frozen design

Frozen default pathways:

```text
De Quervain / first dorsal compartment disorder
thumb CMC-1 OA / rhizarthrosis
interphalangeal / generalized hand OA
median neuropathy at wrist / carpal tunnel syndrome
ulnar-sided wrist / TFCC-related presentation
intersection syndrome
thumb MCP collateral-ligament injury — UCL or RCL
sagittal-band injury / extensor tendon instability at MCP
digital tendon injury / deformity-specific rehabilitation
post-traumatic wrist/hand pain or stiffness after assessed injury
postoperative wrist/hand rehabilitation
```

Rare/advanced/context decisions:

```text
trigger finger/thumb → context only; not routine local physiotherapy referral
Guyon's canal → rare/advanced
scapholunate/lunotriquetral instability → rare/advanced
other ECU/FCR/FCU tendon disorders → secondary/advanced
CRPS → established-diagnosis advanced pathway
inflammatory/psoriatic/crystal disease → established medical context
Dupuytren → medical context unless specific postoperative rehab indication
ganglion/mass → medical context
fractures → shared fracture/post-immobilization profile
```

Key distinctions:

```text
De Quervain != intersection syndrome
ulnar-sided wrist pain != TFCC tear
TFCC = canonical terminology; TFCL not used as structured label
thumb UCL != RCL, but both share one collateral-ligament pathway with subtype-specific safety
MCP snapping != sagittal-band diagnosis
possible CRPS features != formal CRPS diagnosis
finger tendon repair != generic post-traumatic rehab
```

Local service rule:

- dedicated `hand therapist` availability is not assumed in Cyprus;
- generated referrals use physiotherapy / wrist-hand rehabilitation terminology;
- protocol-sensitive tendon/ligament/orthosis work may request relevant experience/competence rather than an unavailable professional title.

Frozen adjunct policy:

```text
manual therapy / mobilization → optional where relevant
soft tissue → optional
taping → optional
selected thermal strategy for OA → optional
acupuncture → excluded
dry needling → excluded
ESWT → excluded
therapeutic ultrasound → not standard CTS/general wrist-hand treatment
```

Orthosis is a separate condition-sensitive support category and exact procedure/injury protocol outranks generic suggestions.

---

# 4. Shared fracture / post-immobilization profile

Fractures remain handled once in a future shared profile rather than duplicated region by region.

Required future context:

```text
bone/site
fracture date/phase
treatment
healing/stability status
immobilization/orthosis status
weight-bearing/use status
ROM/loading restrictions
surgeon/orthopaedic instructions
```

Regional entry points now include shoulder, elbow and wrist/hand fractures.

Unknown healing/loading context must produce a warning rather than unrestricted rehabilitation wording.

---

# 5. Context-sensitive goals / directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

Active/function-oriented rehabilitation, education, self-management and graded loading/activity remain the conceptual backbone where appropriate, subject to structural/surgical restrictions.

---

# 6. Safety / consistency engine

```text
fracture rehab + missing healing/use context
→ warning

post-op/tendon-repair route + missing procedure/protocol/restrictions
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

acute tendon-laceration/rupture concern
→ structural reassessment before generic rehab

material safety/infection concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological component
→ never generate normal wording
```

Region-specific rules live in each frozen profile.

---

# 7. Remaining regional design sequence

Current broad remaining sequence is:

```text
knee / hip
→ ankle / foot
→ shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner selects the exact next region.

---

# 8. Output wording rules

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction / restrictions.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- no unsupported diagnosis from symptoms, tests or incidental imaging;
- no normal neurological/red-flag statement from missing data;
- preserve explicit restrictions;
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