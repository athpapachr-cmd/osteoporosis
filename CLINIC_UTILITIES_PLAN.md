# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1 and shoulder v1.1 frozen; elbow v1 active design candidate.

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

The application generates structured referral wording without replacing the physiotherapist's assessment or prescribing a complete treatment recipe.

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
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen profile status

## 3.1 Cervical — FROZEN v1.1

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

## 3.2 Lumbar — FROZEN v1.1

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

## 3.3 Shoulder — FROZEN v1.1

```text
clinic_utilities/physio_profiles/shoulder_v1_1.md
```

Shoulder includes RCRSP, established full-thickness tear conservative rehabilitation, calcific tendinopathy, adhesive capsulitis, GH instability/dislocation, GH OA, post-traumatic assessed injury, AC/SC-joint pathways and postoperative rehabilitation. Long-head biceps is a common secondary/coexisting diagnosis; shoulder fractures route to the shared fracture profile.

---

# 4. Elbow — ACTIVE DESIGN CANDIDATE

Candidate file:

```text
clinic_utilities/physio_profiles/elbow_v1.md
```

Proposed elbow pathways:

```text
lateral elbow tendinopathy / lateral epicondylalgia
medial elbow tendinopathy / medial epicondylalgia
ulnar neuropathy at elbow / cubital tunnel presentation
distal biceps tendinopathy or established partial tear — conservative pathway
distal triceps tendinopathy or established partial tear — conservative pathway
elbow OA / degenerative painful stiffness
ligament injury / instability rehabilitation
assessed aseptic olecranon bursitis
post-traumatic elbow pain/stiffness after assessed injury
postoperative elbow rehabilitation — pending product-owner workflow confirmation
```

Candidate secondary/context items:

```text
radial tunnel / radial-PIN-related presentation
established inflammatory/crystal-disease context
myofascial/trigger-point findings
```

Key candidate rules:

- lateral/medial provocation tests remain findings rather than diagnoses;
- subjective paresthesia remains separate from objective motor/sensory deficit;
- medial tendinopathy remains separate from ulnar neuropathy and UCL pathology;
- lateral epicondylalgia remains separate from radial/PIN/cervical/intra-articular differential diagnoses;
- progressive neurological weakness/atrophy requires reassessment semantics;
- acute distal biceps/triceps rupture concern is not a routine tendinopathy pathway;
- aseptic olecranon-bursitis wording requires infection concern to be clinically addressed;
- ligament stress tests do not create instability diagnosis;
- postoperative elbow, if retained, requires exact procedure/protocol/restriction context;
- elbow fractures route to the shared fracture/post-immobilization profile.

### Elbow adjunct policy — candidate, not frozen

```text
manual therapy / mobilization → optional adjunct
dry needling → optional; lateral-elbow CPG has moderate-strength support; competence safeguard applies
acupuncture → optional; possible short-term benefit but limited certainty
taping → optional short-term adjunct
counterforce brace / wrist orthosis → activity-specific optional support, not required long-term treatment
ESWT → optional evidence-sensitive item for chronic/recalcitrant epicondylalgia only if product owner wants it
therapeutic ultrasound → not a standard evidence-backed stand-alone lateral-elbow option
```

Current high-quality evidence backbone includes the 2022 APTA/JOSPT lateral-elbow-tendinopathy CPG and current systematic reviews on medial epicondylalgia, ulnar neuropathy/cubital tunnel, distal biceps pathology, olecranon bursitis, ligament rehabilitation, postoperative elbow rehabilitation, dry needling, acupuncture and ESWT.

Elbow remains **NOT FROZEN** until product-owner review.

---

# 5. Shared fracture / post-immobilization profile

Fractures should be handled in one shared profile rather than duplicated region by region.

Required future context:

```text
bone/site
fracture date/phase
treatment
healing/stability status if known
immobilization status
weight-bearing/use status
ROM/loading restrictions
surgeon/orthopaedic instructions
```

Regional entry points include shoulder and elbow fractures such as proximal humerus, clavicle, scapula, radial head/neck, olecranon/proximal ulna, distal humerus and other relevant sites.

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

Active rehabilitation, exercise, graded activity/loading, education and self-management remain the conceptual backbone where appropriate.

---

# 7. Safety / consistency engine

The utility provides prompts, not autonomous diagnoses or treatment prohibitions.

Cross-cutting examples:

```text
fracture rehab + missing healing/use context
→ warning

post-op pathway + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety/red-flag concern + no clinician disposition
→ do not generate routine reassuring wording

unassessed neurological component
→ never generate normal wording
```

Region-specific rules live in each profile.

---

# 8. Remaining regional design sequence

After elbow review/freeze, current preferred working sequence is:

```text
wrist / hand
→ knee / hip
→ ankle / foot
→ shared fracture / post-immobilization
→ muscle / myotendinous injury
→ generalized deconditioning / balance / gait
```

The product owner may change the exact next region within CU-1.

---

# 9. Output wording rules

Preferred structure:

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction / restrictions.
Optional reassessment/communication criteria.
```

Rules:

- collaborative wording;
- active/function-oriented rehabilitation as core where appropriate;
- technique-level interventions remain adjuncts;
- no unsupported diagnosis from symptom combinations, tests or incidental imaging;
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
