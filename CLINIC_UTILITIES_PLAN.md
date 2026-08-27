# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1, elbow v1.1 and wrist/hand v1.1 frozen; Knee v1 active design candidate.

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
knee_v1 = ACTIVE DESIGN CANDIDATE / NOT FROZEN
```

Authoritative current Knee candidate:

```text
clinic_utilities/physio_profiles/knee_v1.md
```

---

# 3. Knee v1 — active candidate frame

Proposed default pathways:

```text
knee osteoarthritis
degenerative meniscal lesion/tear — conservative rehabilitation
acute isolated meniscal injury — assessed nonoperative pathway
patellofemoral pain
patellar tendinopathy
established knee ligament injury/instability — nonoperative rehabilitation
patellar instability/dislocation rehabilitation
post-traumatic knee pain/stiffness after assessed injury
postoperative knee rehabilitation — pending workflow confirmation
```

Candidate rare/secondary/context entities:

```text
quadriceps tendinopathy
iliotibial-band syndrome
pes-anserine pain / established bursitis-tendinopathy
Baker/popliteal cyst context
prepatellar/infrapatellar bursitis context
osteochondral/chondral lesion or osteochondritis dissecans
meniscal root tear / complex repair-relevant lesion
Hoffa fat-pad / plica context
inflammatory / crystal knee context
Osgood-Schlatter / Sinding-Larsen-Johansson if adolescent workflow requires
```

Key candidate distinctions:

```text
radiographic OA != automatic symptom generator
degenerative MRI meniscal tear != automatic symptom generator or surgical indication
clicking/catching != true locked knee
degenerative meniscus != acute displaced/displacing meniscus pathway
anterior knee pain != automatic patellofemoral diagnosis
cartilage/chondromalacia imaging != automatic symptomatic diagnosis
patellar-tendon imaging change != automatic tendinopathy
subjective giving-way != objective ligament instability
ligament test != autonomous tear grade
first-time patellar dislocation requires structural/osteochondral context
postoperative rehabilitation = exact procedure/protocol governed
posterior swelling != automatic Baker cyst and does not exclude DVT
```

Evidence-oriented core directions:

- knee OA: education/self-management plus individualized exercise, strengthening and functional/aerobic activity;
- common degenerative meniscal lesions: exercise-based rehabilitation first-line when no structural surgical indication is present;
- acute meniscal tears: selected non-displaced tears may enter rehabilitation, while displaced/displacing tears restricting ROM or repair-relevant lesions require timely specialist decision;
- patellofemoral pain: education plus knee-targeted with or without hip-targeted exercise as core, with taping/foot orthoses/manual/movement retraining tailored to presentation;
- patellar tendinopathy: progressive load-based rehabilitation without freezing one universal loading mode;
- ligament injury: ROM/strength/neuromuscular and criterion-based progression according to established injury and restrictions;
- postoperative knee: procedure-specific protocol and restrictions outrank generic defaults.

Candidate support policy:

```text
OA brace → condition-sensitive
patellofemoral taping/support → condition-sensitive
prefabricated foot orthosis → selected patellofemoral presentations
ligament/postoperative brace → exact plan/protocol
temporary walking aid/cane → optional when appropriate
```

Candidate adjunct questions remain open:

```text
manual therapy / soft tissue → optional where relevant
thermal OA support → optional
acupuncture for knee OA → unresolved because major guidelines differ
dry needling → unresolved; not core
ESWT for patellar tendinopathy → not default; include only if real workflow warrants evidence-sensitive option
NMES → postoperative/TKA-specific context, not generic knee OA
```

Knee remains **NOT FROZEN** until product-owner review.

---

# 4. Shared fracture / post-immobilization profile

Fractures are handled once in a future shared profile rather than duplicated region by region.

Required future context:

```text
bone/site
fracture date/phase
treatment
healing/stability status
immobilization/brace/orthosis status
weight-bearing/use status
ROM/loading restrictions
surgeon/orthopaedic instructions
```

Regional entry points now include shoulder, elbow, wrist/hand and knee fractures.

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

true locked knee / major mechanical ROM block
→ structural reassessment prompt

acute extensor-mechanism rupture concern
→ structural reassessment before routine rehab

major/multiligament instability or neurovascular deficit
→ specialist/urgent semantics

post-op route + missing procedure/protocol/restrictions
→ warning

hot swollen knee / infection / DVT concern
→ medical reassessment

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological/neurovascular component
→ never generate normal wording
```

Region-specific rules live in each frozen/candidate profile.

---

# 7. Remaining regional design sequence

After Knee review/freeze, current broad sequence is:

```text
hip
→ ankle / foot
→ shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next region within CU-1.

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
- active/function-oriented rehabilitation as core where appropriate;
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
