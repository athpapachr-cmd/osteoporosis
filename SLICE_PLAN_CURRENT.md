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
> **Current detailed profile under review:** `clinic_utilities/physio_profiles/wrist_hand_v1.md` — DESIGN CANDIDATE / NOT FROZEN.
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

The utility should improve referral quality and speed while preserving clinician judgment and physiotherapist/hand-therapist autonomy.

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
orthosis != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

---

# 3. Frozen regional status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
```

Authoritative frozen files remain under `clinic_utilities/physio_profiles/*_v1_1.md`.

---

# 4. Wrist / Hand — ACTIVE DESIGN CANDIDATE

Authoritative candidate:

```text
clinic_utilities/physio_profiles/wrist_hand_v1.md
```

Proposed default primary pathways for product-owner review:

```text
WH1 De Quervain / first dorsal compartment disorder
WH2 thumb CMC-1 osteoarthritis / rhizarthrosis
WH3 interphalangeal / generalized hand osteoarthritis
WH4 median neuropathy at wrist / carpal tunnel syndrome
WH5 ulnar-sided wrist pain / TFCC-related presentation
WH6 wrist extensor/flexor tendinopathy / overuse disorder
WH7 trigger finger / trigger thumb
WH8 thumb MCP UCL injury / instability rehabilitation
WH9 post-traumatic wrist/hand pain or stiffness after assessed injury
WH10 postoperative wrist/hand rehabilitation — pending real-workflow confirmation
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

Key candidate safety/semantic decisions:

- radial-sided wrist pain or a De Quervain provocation test does not establish De Quervain disease;
- De Quervain referral wording must not claim physiotherapy alone is the evidence-preferred first-line treatment; current comparative evidence favors corticosteroid injection plus short thumb-spica immobilization;
- thumb CMC-1 OA and interphalangeal hand OA remain separate phenotypes;
- incidental hand/wrist OA imaging does not automatically establish the symptomatic pain generator;
- CTS subjective paresthesia remains separate from objective sensory/motor deficit;
- Phalen/Tinel/Durkan and upper-limb neurodynamic tests do not independently establish CTS; current AAOS guidance specifically advises against upper-limb neurodynamic testing as a diagnostic substitute;
- progressive thenar weakness/atrophy requires reassessment/specialist semantics;
- uncomplicated carpal-tunnel release does not automatically generate supervised postoperative hand therapy;
- ulnar-sided wrist pain or TFCC provocation does not establish TFCC tear;
- DRUJ instability/foveal full-tear context changes TFCC management and requires specialist/restriction awareness;
- ECU instability/subluxation remains distinct from ordinary ECU tendinopathy;
- acute thumb UCL instability/Stener concern is not routine unrestricted rehabilitation;
- possible CRPS features do not create a formal CRPS diagnosis;
- fractures route to the shared fracture/post-immobilization profile;
- tendon laceration/repair and complex surgery require exact procedure/protocol/restriction context.

Candidate orthosis/support policy:

```text
thumb spica → condition-sensitive
CMC-support orthosis → condition-sensitive, evidence-supported for CMC-1 OA
neutral-wrist night orthosis → short-term CTS symptom-management option
trigger-digit orthosis → candidate conservative option
injury/post-op orthosis → protocol governed
```

Candidate adjunct policy:

```text
manual therapy / mobilization → optional where relevant
soft tissue → optional
taping → optional
dry needling → only selected myofascial/tendinous context + competence safeguard
acupuncture → optional only if product owner confirms wrist/hand use
ESWT → not proposed as a default wrist/hand adjunct
therapeutic ultrasound → not standard evidence-backed CTS/general wrist-hand treatment
```

Wrist/hand remains **NOT FROZEN** until product-owner workflow decisions in `wrist_hand_v1.md` are resolved.

---

# 5. Shared fracture / post-immobilization boundary

Wrist/hand fracture entry points include:

```text
distal radius / distal ulna
scaphoid / other carpal fracture
metacarpal fracture
phalangeal fracture
other wrist/hand fracture
```

The future shared profile requires explicit healing/stability, immobilization/orthosis, ROM/loading/use restrictions and orthopaedic/hand-surgeon instructions.

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

post-op/tendon-repair route + missing procedure/protocol/restrictions
→ warning

adjunct selected without active rehabilitation direction
→ warning

new/progressive objective neurological deficit
→ prominent medical reassessment prompt

material safety/infection concern + no clinician disposition
→ do not generate routine reassuring wording

not assessed neurological component
→ never generate normal wording
```

Region-specific wrist/hand rules live in `wrist_hand_v1.md`.

---

# 8. Output wording contract

```text
Clinical problem + important findings + functional impact.
Referral request + goals.
Rehabilitation/hand-therapy direction + restrictions/precautions.
Optional reassessment/feedback criteria.
```

Rules:

- collaborative wording, not over-prescription of the therapist;
- active rehabilitation, education/self-management and graded loading/activity where appropriate;
- orthoses are condition-sensitive supports rather than global defaults;
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
1. product-owner clinical review of wrist_hand_v1.md
2. resolve trigger-digit pathway visibility
3. resolve Guyon's canal visibility
4. resolve postoperative wrist/hand need
5. resolve thumb-UCL and carpal-instability pathway visibility
6. resolve CRPS and digital-tendon advanced pathways
7. confirm acupuncture/dry-needling/ESWT policy
8. revise candidate
9. freeze/merge only after explicit product-owner approval
```

Runtime implementation remains unauthorized.
