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
> **Current detailed profile under review:** `clinic_utilities/physio_profiles/elbow_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
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

Future implementation must build a deterministic structured object before prose formatting:

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
```

The shoulder freeze includes RCRSP, established full-thickness cuff tear conservative rehabilitation, calcific tendinopathy, adhesive capsulitis, GH instability/dislocation, GH OA, post-traumatic assessed injury, AC/SC-joint pathways and postoperative shoulder rehabilitation. Shoulder-region fractures route to the shared fracture profile.

---

# 4. Elbow — ACTIVE DESIGN CANDIDATE

Authoritative candidate:

```text
clinic_utilities/physio_profiles/elbow_v1.md
```

Proposed primary pathways for product-owner review:

```text
E1 lateral elbow tendinopathy / lateral epicondylalgia
E2 medial elbow tendinopathy / medial epicondylalgia
E3 ulnar neuropathy at elbow / cubital tunnel presentation
E4 distal biceps tendinopathy or established partial tear — conservative pathway
E5 distal triceps tendinopathy or established partial tear — conservative pathway
E6 elbow osteoarthritis / degenerative painful stiffness
E7 ligament injury / instability rehabilitation
E8 assessed aseptic olecranon bursitis
E9 post-traumatic elbow pain/stiffness after assessed injury
E10 postoperative elbow rehabilitation — pending real-workflow confirmation
```

Candidate secondary/context items include radial-tunnel/PIN-related presentations, established inflammatory/crystal disease context and myofascial findings.

Key candidate safety semantics:

- subjective paresthesia remains separate from objective motor/sensory deficit;
- progressive motor weakness/atrophy in ulnar/radial distributions triggers reassessment semantics;
- acute distal-biceps or distal-triceps rupture concern must not be rendered as routine tendinopathy;
- positive epicondylalgia tests do not establish tendinopathy;
- medial pain plus ulnar symptoms must not collapse into a single diagnosis;
- stress tests do not establish ligament instability;
- aseptic olecranon-bursitis wording requires infectious concern to be addressed;
- acute hot/swollen elbow, fever/cellulitis/wound/drainage or septic-joint/bursitis concern requires medical reassessment semantics;
- fracture/dislocation and complex injury route through explicit structural/healing/protocol context;
- postoperative elbow, if retained, requires procedure/protocol/restriction context.

Current evidence review supports progressive resisted loading as the active core for epicondylalgia, with manual therapy and dry needling as selectable adjuncts for lateral elbow tendinopathy. Acupuncture may remain optional but evidence certainty is limited. Counterforce/wrist orthoses may be used short-term/activity-specifically rather than as a required long-term treatment. ESWT evidence is mixed and remains an open product decision rather than a default.

Elbow remains **NOT FROZEN** until the product owner resolves the open workflow/taxonomy questions in `elbow_v1.md`.

---

# 5. Shared fracture / post-immobilization boundary

Fractures are handled in one shared profile rather than duplicated region by region.

Elbow entry points may include:

```text
radial head/neck fracture
olecranon/proximal ulna fracture
distal humerus fracture
coronoid fracture
complex fracture-dislocation / terrible-triad context
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

The engine provides clinician-facing prompts, not autonomous diagnoses or treatment prohibitions.

Cross-region rules include:

```text
fracture/post-trauma + unresolved structural/healing context
→ warning/reassessment prompt

post-op pathway + missing procedure/protocol/restrictions
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

Region-specific elbow rules live in `elbow_v1.md`.

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
1. product-owner clinical review of clinic_utilities/physio_profiles/elbow_v1.md
2. resolve radial-tunnel pathway visibility
3. resolve olecranon-bursitis pathway visibility
4. resolve postoperative-elbow need
5. confirm ESWT/acupuncture adjunct policy
6. confirm distal biceps/triceps taxonomy
7. revise elbow candidate
8. freeze/merge only after explicit product-owner approval
9. then continue to the next CU-1 region
```

Runtime implementation remains unauthorized.
