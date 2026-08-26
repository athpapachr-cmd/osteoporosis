# SLICE_PLAN_CURRENT.md — CU-1 Physiotherapy Referral v2 design

> **STATUS:** ACTIVE PRE-CODE DESIGN — runtime implementation not authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1.
> **Supporting plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile on active docs branch:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Prior active slice:** PR-1 Transcript Intake + Candidate Extraction v3, intentionally paused and archived at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

CU-1 is a bounded product-owner-approved design detour. It does not cancel PR-1 and does not create a new clinical module.

---

# 1. Product outcome

Freeze the clinical/content and deterministic-generation contract for a substantially improved Physiotherapy Referral utility before implementation.

```text
Clinical problem
→ important findings
→ functional limitation
→ precautions/restrictions
→ goals
→ rehabilitation direction
→ structured ReferralDraft
→ short/detailed referral text
```

The utility should improve referral quality and speed while preserving clinician judgment and physiotherapist autonomy.

---

# 2. Frozen architecture

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

Hard invariants:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

---

# 3. Frozen regional status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 4. Elbow — FROZEN v1.1 design

Authoritative frozen file on the active docs branch:

```text
clinic_utilities/physio_profiles/elbow_v1_1.md
```

Frozen default primary pathways:

```text
E1 lateral elbow tendinopathy / lateral epicondylalgia
E2 medial elbow tendinopathy / medial epicondylalgia
E3 ulnar neuropathy at elbow / cubital tunnel
E4 posterior interosseous nerve / supinator syndrome
E5 distal biceps tendinopathy or established partial tear — conservative pathway
E6 elbow osteoarthritis / degenerative painful stiffness
E7 ligament injury / instability rehabilitation
E8 post-traumatic elbow pain/stiffness after assessed injury
```

Frozen workflow decisions:

- radial tunnel syndrome is uncommon and remains clinician-entered secondary/coexisting context rather than a default primary pathway;
- pain-predominant radial tunnel presentation remains distinct from PIN/supinator motor-neuropathy semantics;
- PIN/supinator syndrome is a clinician-established neurological pathway; lateral forearm pain or a provocation test alone does not create it;
- olecranon bursitis is not a routine physiotherapy referral for this workflow and is removed from default primary pathways, while infection safety/context remains available;
- postoperative elbow is rare and remains an advanced/future-access route rather than a default MVP pathway;
- distal triceps and anconeus presentations remain rare selectable myotendinous entities rather than top-level pathways;
- anconeus epitrochlearis remains distinct from ordinary anconeus pain/injury and is not automatically pathological;
- fractures route to the shared fracture/post-immobilization profile;
- progressive objective ulnar/radial/PIN weakness/atrophy triggers reassessment semantics;
- acute distal-biceps/triceps rupture concern is not a routine tendinopathy pathway;
- medial tendinopathy remains separate from ulnar neuropathy and ligament pathology;
- stress/provocation tests remain findings, not diagnoses.

Frozen adjunct policy:

```text
manual therapy / mobilization → optional adjunct
soft tissue → optional adjunct
dry needling → optional + competence/availability safeguard
acupuncture → optional clinician-selected adjunct
ESWT → optional evidence-sensitive adjunct for lateral/medial epicondylalgia
counterforce brace / wrist orthosis → optional short-term/activity-specific support
therapeutic ultrasound → not standard evidence-backed treatment
```

ESWT wording remains deliberately cautious because recent systematic reviews remain heterogeneous; no claim of universal superiority or mandatory use is permitted. Evidence for medial epicondylalgia is less mature than for lateral epicondylalgia.

---

# 5. Shared fracture / post-immobilization boundary

Elbow fracture entry points include:

```text
radial head/neck fracture
olecranon/proximal ulna fracture
distal humerus fracture
coronoid fracture
complex fracture-dislocation context
```

The future shared profile requires explicit healing/stability, immobilization, ROM/loading/use restrictions and orthopaedic instructions.

```text
fracture/post-immobilization + unresolved healing/loading context
→ warning
→ no unrestricted routine rehabilitation wording
```

---

# 6. Context-sensitive defaults

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms or changes
→ only confirmed values populate ReferralDraft
```

No global pain + ROM + strength + motor-control bundle.

---

# 7. Safety / consistency engine

Cross-region rules include:

```text
fracture/post-trauma + unresolved structural/healing context
→ warning/reassessment prompt

rare post-op route + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety concern + no clinician disposition
→ do not generate routine reassuring wording

not assessed neurological component
→ never generate normal wording
```

Region-specific elbow rules live in `elbow_v1_1.md`.

---

# 8. Output wording contract

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation direction + restrictions/precautions.
Optional reassessment/feedback criteria.
```

Rules:

- collaborative wording, not over-prescription of the physiotherapist;
- active rehabilitation, education/self-management and graded loading/activity where appropriate;
- technique-level interventions remain adjunctive;
- no unsupported diagnosis from tests, symptoms or incidental imaging;
- no negative neurological/red-flag/structural statements from missing data;
- preserve explicit restrictions;
- short and detailed outputs derive from the same `ReferralDraft`.

---

# 9. Persistence / patient-linkage boundary

CU-1 does **not** freeze referral persistence yet.

Default first implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

---

# 10. Out of scope for CU-1

Do not yet:

- write production HTML/JS/CSS;
- integrate Clinical Excellence navigation;
- add patient persistence/history;
- use AI to generate the referral;
- implement RF workflow changes;
- alter Osteoporosis audit/PR-1 transcript runtime;
- create overlapping runtime writers.

---

# 11. Exact next action

```text
1. exact branch-vs-main review of elbow freeze
2. open docs-only elbow freeze PR if clean
3. independent exact-head review
4. merge only if clean
5. close canonical writer lock on main
6. product owner selects next CU-1 region
```

Runtime implementation remains unauthorized.
