# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1, lumbar v1.1, shoulder v1.1 and elbow v1.1 frozen.

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
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen profile status

## Cervical — FROZEN v1.1

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

## Lumbar — FROZEN v1.1

```text
clinic_utilities/physio_profiles/lumbar_v1_1.md
```

## Shoulder — FROZEN v1.1

```text
clinic_utilities/physio_profiles/shoulder_v1_1.md
```

## Elbow — FROZEN v1.1

```text
clinic_utilities/physio_profiles/elbow_v1_1.md
```

Frozen elbow default pathways:

```text
lateral elbow tendinopathy / lateral epicondylalgia
medial elbow tendinopathy / medial epicondylalgia
ulnar neuropathy at elbow / cubital tunnel
PIN / supinator syndrome
distal biceps tendinopathy or established partial tear — conservative pathway
elbow OA / degenerative painful stiffness
ligament injury / instability rehabilitation
post-traumatic elbow pain/stiffness after assessed injury
```

Rare/advanced/context decisions:

```text
radial tunnel syndrome → secondary/coexisting context; uncommon in workflow
olecranon bursitis → medical/context only; not routine physio primary pathway
postoperative elbow → rare advanced/future route, not default MVP
distal triceps → rare selectable myotendinous entity
anconeus pain/injury → rare selectable myotendinous entity
anconeus epitrochlearis → distinct anatomic variant; never auto-pathologized
fractures → shared fracture/post-immobilization profile
```

Neural boundary:

```text
pain-predominant radial tunnel presentation
!=
clinician-established PIN/supinator syndrome with motor-neuropathy semantics
```

Adjunct policy:

```text
manual therapy / soft tissue → optional
dry needling → optional + competence safeguard
acupuncture → optional
ESWT → optional evidence-sensitive adjunct for lateral/medial epicondylalgia
counterforce/wrist support → optional short-term/activity-specific
therapeutic ultrasound → not standard evidence-backed treatment
```

ESWT remains optional rather than standard because recent reviews are heterogeneous, especially for functional superiority/comparator effects; evidence for medial epicondylalgia is more limited than for lateral disease.

---

# 4. Shared fracture / post-immobilization profile

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

Regional entry points now include shoulder and elbow fractures, including proximal humerus, clavicle, scapula, radial head/neck, olecranon/proximal ulna, distal humerus and other relevant sites.

Unknown healing/loading context must produce a warning rather than unrestricted rehabilitation wording.

---

# 5. Context-sensitive goals and directions

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

Active rehabilitation, exercise, graded activity/loading, education and self-management remain the conceptual backbone where appropriate.

---

# 6. Safety / consistency engine

```text
fracture rehab + missing healing/use context
→ warning

rare post-op route + missing procedure/protocol/restrictions
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

Region-specific rules live in each frozen profile.

---

# 7. Remaining regional design sequence

Current preferred working sequence after elbow:

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
- technique-level interventions remain adjuncts;
- no unsupported diagnosis from symptoms, tests or incidental imaging;
- no normal neurological/red-flag statement from missing data;
- preserve explicit restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 9. Implementation boundary

CU-1 remains **design only**.

First implementation direction remains conceptually:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. Do not write production physiotherapy runtime code until CU-1 is sufficiently frozen and the product owner explicitly authorizes transition to CU-2.
