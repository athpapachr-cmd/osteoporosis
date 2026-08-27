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
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **Frozen wrist/hand profile on active docs branch:** `clinic_utilities/physio_profiles/wrist_hand_v1_1.md`.
> **Prior active slice:** PR-1 Transcript Intake + Candidate Extraction v3 remains intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

CU-1 is a bounded cross-module design detour. It does not authorize runtime implementation.

---

# 1. Frozen architecture

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
orthosis != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

---

# 2. Frozen regional status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 3. Wrist / Hand — FROZEN v1.1 design

Authoritative frozen file on the active docs branch:

```text
clinic_utilities/physio_profiles/wrist_hand_v1_1.md
```

Frozen default primary pathways:

```text
WH1 De Quervain / first dorsal compartment disorder
WH2 thumb CMC-1 osteoarthritis / rhizarthrosis
WH3 interphalangeal / generalized hand osteoarthritis
WH4 median neuropathy at wrist / carpal tunnel syndrome
WH5 ulnar-sided wrist / TFCC-related presentation
WH6 intersection syndrome
WH7 thumb MCP collateral-ligament injury — UCL or RCL
WH8 sagittal-band injury / extensor tendon instability at MCP
WH9 digital tendon injury / deformity-specific rehabilitation
WH10 post-traumatic wrist/hand pain or stiffness after assessed injury
WH11 postoperative wrist/hand rehabilitation
```

Frozen workflow decisions:

- trigger finger/thumb is not a routine physiotherapy referral in the local Cyprus workflow and is context only;
- Guyon's canal remains rare/advanced;
- scapholunate/lunotriquetral injury or carpal instability remains rare/advanced;
- postoperative wrist/hand is active because these patients are seen;
- thumb collateral-ligament rehabilitation includes both UCL and RCL; UCL Stener concern and major RCL instability/subluxation require stronger reassessment semantics;
- CRPS is an established-diagnosis advanced rehabilitation pathway and must never be inferred from symptoms alone;
- mallet, central-slip/boutonniere and flexor/extensor tendon injuries are directly selectable and protocol/zone governed;
- intersection syndrome is directly selectable and remains distinct from De Quervain;
- TFCC is the canonical terminology; `TFCL` is not used as the structured label;
- fractures route to the shared fracture/post-immobilization profile;
- dedicated hand-therapist availability is not assumed; generated wording uses physiotherapy/wrist-hand rehabilitation and competence/protocol language.

Frozen adjunct policy:

```text
manual therapy / mobilization → optional where relevant
soft tissue → optional where relevant
taping → optional where relevant
selected thermal strategy for hand OA → optional
acupuncture → excluded
dry needling → excluded
ESWT → excluded
therapeutic ultrasound → not standard evidence-backed CTS/general wrist-hand treatment
```

Orthosis is a first-class condition-sensitive support category rather than a generic adjunct. Exact injury/surgical protocol outranks generic orthosis suggestions.

---

# 4. Safety / consistency highlights

```text
radial wrist pain != De Quervain or intersection syndrome automatically
intersection syndrome != De Quervain
ulnar wrist pain != TFCC tear automatically
incidental TFCC imaging finding != symptomatic lesion automatically
CTS paresthesia != objective neurological deficit
CTS neurodynamic test != diagnostic proof
uncomplicated carpal-tunnel release != automatic supervised PT
thumb-MCP stress finding != tear grade
UCL Stener concern / major RCL instability != unrestricted rehab
MCP snapping != sagittal-band diagnosis
finger tendon injury/repair = zone + healing + orthosis + protocol governed
possible CRPS features != formal CRPS diagnosis
scaphoid/fracture concern != routine post-traumatic rehab
not_assessed neurological component != normal
```

---

# 5. Shared fracture / post-immobilization boundary

Wrist/hand fractures route to the shared profile:

```text
distal radius / distal ulna
scaphoid / other carpal
metacarpal
phalangeal
other wrist/hand fracture
```

Unresolved healing/stability, immobilization, ROM or loading context prevents unrestricted rehabilitation wording.

---

# 6. Persistence / runtime boundary

Persistence is not frozen.

Default first implementation direction remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Do not write production HTML/JS/CSS, add patient persistence, integrate navigation or start CU-2 without explicit product-owner authorization.

---

# 7. Exact next action

```text
1. exact branch-vs-main review of wrist/hand freeze
2. open docs-only wrist/hand freeze PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and record resulting main state
6. product owner selects next CU-1 regional profile
```

Runtime implementation remains unauthorized.