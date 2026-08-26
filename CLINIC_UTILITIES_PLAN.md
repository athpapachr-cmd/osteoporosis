# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; cervical v1.1 frozen, lumbar v1 under review.

This document keeps the Clinic Utilities detour detailed without confusing operational tooling with Osteoporosis Module 01 or the reusable Clinical Excellence Core.

Permanent boundary:

```text
Clinical Excellence Core
→ reusable patient/workflow/navigation/auth/integration mechanics

Clinic Utilities
→ cross-module clinician-facing operational tools

Module 01 Osteoporosis
→ osteoporosis-specific clinical content
```

Clinic Utilities do not constitute a new clinical Module 02.

---

# 1. Detour purpose

Integrate two existing clinician-created tools into the future Clinical Excellence workspace:

1. **Physiotherapy Referral Generator** — structured clinical referral text generation.
2. **Radiofrequency Request / PDF Workflow** — request creation, PDF generation, lifecycle tracking and reuse of previous request data.

The detour should improve daily clinic workflow while preserving the broader product objective: better clinical practice before, during and after the consultation.

---

# 2. Source inspection already completed

## 2.1 Physiotherapy source

Useful capabilities to preserve:

- local/no-server operation;
- condition groups by body region;
- optional patient name/laterality/chronicity/session count;
- clinical findings;
- rehabilitation goals;
- active vs adjunct intervention wording;
- short/detailed output modes;
- copy/print;
- basic consistency warnings;
- evidence/reference section.

Design weaknesses to correct:

- checkbox catalogue rather than clinically structured flow;
- generic findings across unrelated diagnoses;
- globally preselected goals/interventions;
- under-modelled precautions/restrictions;
- direct phrase concatenation instead of a structured intermediate object;
- minimal validation;
- incomplete common pathways;
- standalone styling instead of Clinical Excellence-native presentation.

## 2.2 Radiofrequency source

Current implementation lives in `athpapachr-cmd/ortho-reception-backend-v2` and was inspected read-only.

Existing useful pieces include:

- `/rf` protected form route;
- PDF generation from Medikey / DIROS / Thermedico templates;
- radiology PDF attachment;
- previous-application lookup;
- PostgreSQL-backed `rfa_applications` table;
- existing status field;
- patient/site/history/VAS data;
- repeat-use logic.

Important operational constraint:

> RF runtime mutation is not part of CU-1 and remains blocked while the separate Digital Secretary AC-2 writer lock applies.

---

# 3. Physiotherapy Referral v2 — product outcome

The future utility should help the clinician create a concise, clinically coherent referral by moving through:

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

---

# 4. Structured intermediate model

Do not generate prose directly from checkboxes. First construct:

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

This separation is required for later persistence, reuse, auditability or AI-assisted wording refinement.

---

# 5. Condition-profile architecture

A profile may define:

```text
condition_key
body_region
display_name
candidate_findings[]
candidate_functional_impairments[]
candidate_precautions[]
candidate_goals[]
candidate_rehab_directions[]
adjunct_options[]
consistency_rules[]
required_context_when_selected[]
```

Hard rules:

```text
suggested != examined
suggested != selected
selected != clinically mandatory
symptom != objective deficit
provocation test != diagnosis
not assessed != normal
adjunct != core rehabilitation
```

---

# 6. Body-region taxonomy and current profile state

## 6.1 Cervical spine — FROZEN v1.1

Authoritative cervical design:

```text
clinic_utilities/physio_profiles/cervical_v1_1.md
```

Frozen primary pathways:

- non-specific / mechanical neck pain;
- neck pain with radiating upper-limb / radicular features;
- headache with cervical musculoskeletal features, with optional explicit formal cervicogenic-headache diagnosis;
- cervical/cervicogenic dizziness presentation, with optional explicit clinician diagnosis;
- whiplash/post-traumatic neck pain.

Cervical post-operative rehabilitation is not part of the active cervical MVP because it is not part of the product owner's actual referral workflow.

Directly selectable cervical findings/presentation modifiers include:

- ROM restriction/painful movement;
- referred shoulder-girdle/scapular pain;
- myofascial tenderness / active trigger points;
- radiating upper-limb pain/paresthesia/numbness;
- Spurling/neurodynamic findings when actually examined;
- work/ergonomic or sustained-posture aggravation;
- headache- and dizziness-related contextual findings.

Formal cervicogenic headache and cervicogenic/cervical dizziness are never inferred by the utility. They may be carried into generated wording only when explicitly asserted by the clinician.

The dizziness pathway intentionally preserves the evidence caveat that the Bárány Society does not currently endorse routine clinical diagnostic criteria or a proven cervical causal mechanism; the tool therefore supports clinician-entered truth but does not become diagnostic decision-support.

Cervical neurological screen is component-level tri-state:

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
reflexes: normal / abnormal / not_assessed
```

There is no global `no neurological deficit` checkbox and no default `no red flags` output sentence.

Technique-level adjuncts live under a secondary optional expander; active rehabilitation, exercise, education and self-management remain the conceptual core.

## 6.2 Lumbar spine — ACTIVE DESIGN CANDIDATE

Current candidate:

```text
clinic_utilities/physio_profiles/lumbar_v1.md
```

Proposed primary pathways:

- non-specific / mechanical low-back pain;
- low-back pain with radiating leg symptoms / radicular features;
- lumbar spinal stenosis / neurogenic claudication pathway.

Candidate modifiers/findings rather than top-level diagnoses:

- mobility restriction;
- movement/load/postural aggravation;
- trunk strength/endurance deficit;
- paraspinal/gluteal myofascial tenderness or trigger points;
- referred buttock/non-radicular leg pain;
- SLR/slump/neural-tension findings when actually examined.

Neurological screen inherits the same tri-state semantics as cervical. The safety model adds explicit cauda-equina-type concerns including new bladder/bowel/sexual-function disturbance and new perineal/saddle sensory change.

Routine lumbar traction is deliberately not a default adjunct because NICE and WHO recommend against routine traction. Needling/acupuncture requires explicit framework resolution before freeze because NICE and WHO recommendations differ.

## 6.3 Shoulder

Candidate problems include:

- rotator-cuff-related shoulder pain/tendinopathy;
- shoulder stiffness / adhesive capsulitis;
- calcific tendinopathy;
- proximal-biceps-related pain;
- instability/dislocation rehabilitation;
- post-traumatic shoulder rehabilitation;
- post-operative shoulder pathway where the real workflow justifies it.

Candidate findings include active/passive ROM pattern, load-related pain, weakness, painful arc where relevant, stiffness pattern, scapular control and functional limits such as overhead use, dressing, sleep and lifting.

## 6.4 Elbow

Candidate problems:

- lateral elbow tendinopathy;
- medial elbow tendinopathy;
- stiffness;
- grip weakness;
- post-fracture stiffness;
- distal biceps/triceps rehabilitation where appropriate.

## 6.5 Wrist and hand

Candidate problems:

- De Quervain tenosynovitis;
- mechanical wrist/hand pain;
- stiffness;
- reduced grip/dexterity;
- thumb CMC osteoarthritis;
- carpal tunnel conservative/post-operative rehabilitation;
- trigger-finger post-operative rehabilitation;
- distal-radius fracture/post-immobilization pathway.

## 6.6 Hip

Candidate problems:

- hip osteoarthritis;
- greater trochanteric pain syndrome;
- post total-hip arthroplasty where relevant;
- post hip-fracture rehabilitation;
- mobility/strength deficit.

Precautions include weight-bearing status, healing constraints and explicit post-operative restrictions when relevant.

## 6.7 Knee

Candidate problems:

- knee osteoarthritis;
- patellofemoral pain;
- meniscal tear/lesion conservative pathway;
- knee stiffness;
- lower-limb weakness;
- ACL reconstruction;
- collateral-ligament injury;
- patellar tendinopathy;
- total-knee arthroplasty where relevant;
- post-fracture/post-immobilization rehabilitation.

## 6.8 Ankle and foot

Candidate problems:

- lateral ankle sprain;
- chronic ankle instability;
- ankle mobility restriction;
- plantar heel pain;
- Achilles tendinopathy;
- Achilles rupture rehabilitation;
- post ankle fracture;
- posterior tibial tendon dysfunction pathway;
- post-operative foot/ankle rehabilitation where relevant.

## 6.9 Muscle strain / myotendinous injury

Required context candidates:

- muscle/group;
- site;
- date/phase;
- imaging/grade if known;
- activity/sport demand.

Rehab direction remains symptom-guided and criterion-based progressive loading rather than unsupported fixed timelines.

## 6.10 Fracture / post-immobilization

Required context:

- bone/site;
- treatment;
- healing/stability status if known;
- immobilization state;
- weight-bearing status;
- surgeon/orthopaedic restrictions.

If loading/healing restrictions are unknown, show a prominent review prompt rather than generate unrestricted mobilisation wording.

## 6.11 Shared post-operative musculoskeletal rehabilitation

This remains a general capability, not a mandatory option within every regional MVP.

When used, require:

- operation;
- date;
- surgeon/protocol;
- weight-bearing/activity restrictions;
- ROM restrictions;
- other relevant precautions.

The generator must never invent a generic post-operative protocol.

## 6.12 General deconditioning / balance / gait

Cross-regional pathway for deconditioning, balance deficit, recurrent falls/fall risk, gait retraining and generalized lower-limb weakness.

---

# 7. Context-sensitive goals and rehabilitation directions

Remove the global assumption that pain + ROM + strength + motor control + function always apply.

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items enter ReferralDraft
```

Examples:

- De Quervain should not inherit cervical motor-control wording;
- fracture rehabilitation depends on healing/loading restrictions;
- chronic OA may prioritize strength/function/activity;
- post-operative restrictions outrank generic defaults;
- radicular/radiating symptoms do not imply a promise to reverse neurological deficit.

---

# 8. Safety / consistency engine v1

The utility provides prompts, not autonomous treatment prohibitions or diagnostic decisions.

Cross-cutting rules:

```text
fracture rehab + missing weight-bearing/healing context
→ warning

post-op + missing procedure/protocol/restrictions
→ warning

manual/passive adjunct selected without active rehabilitation direction
→ warning

gait training selected without gait/function problem
→ soft warning

new/progressive objective neurological deficit
→ prominent medical reassessment warning

material safety/red-flag concern
→ require clinician disposition before routine reassuring wording

unassessed neurological component
→ never generate normal wording
```

Region-specific consistency rules belong in each frozen profile.

---

# 9. Output wording rules

Preferred structure:

```text
Clinical problem + important findings + functional impact.

Referral request + goals.

Rehabilitation direction / restrictions.

Optional reassessment/communication criteria.
```

Wording principles:

- collaborative language;
- active rehabilitation, education, self-management and graded activity/loading where appropriate;
- passive techniques remain adjunctive;
- no unsupported diagnosis from a provocation test;
- no negative neurological/red-flag statement from missing data;
- preserve explicit restrictions exactly;
- short and detailed versions derive from the same `ReferralDraft`.

---

# 10. Cockpit integration target

Future navigation candidate:

```text
Clinical Excellence Home
└── Clinical Tools
    ├── Physiotherapy Referral
    └── RF Requests
```

First physiotherapy implementation remains conceptually:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen yet. If later persisted, reuse creates a new referral identity rather than overwriting historical referral truth.

---

# 11. RF workflow target

Longer-term RF workflow:

```text
new request
→ application_generated / draft-like state
→ submitted
→ pending_approval
→ approved_awaiting_procedure
→ performed
```

Views:

```text
Pending decision
Approved — awaiting procedure
Completed / performed
All / search
```

`New from previous` clones reusable facts into a new request identity but excludes old status/approval/procedure/follow-up state unless explicitly reconfirmed.

---

# 12. Implementation order

```text
CU-1  Physiotherapy Referral v2 clinical/content + structured-draft design
CU-2  Physiotherapy Referral v2 implementation + Clinical Excellence styling
CU-3  Cockpit navigation integration / optional patient prefill boundary
CU-4  RF lifecycle/data-model design after Secretary writer lock permits
CU-5  RF clinician UI + request registry/history/reuse
CU-6  RF PDF engine ownership/migration/integration cleanup
```

---

# 13. Current design stop point

Current active work remains **CU-1 design only**.

Cervical v1.1 is frozen. Lumbar v1 is now the profile under clinical/content review. After lumbar freeze, proceed to shoulder.

Do not write production physiotherapy runtime code until CU-1 as a whole is sufficiently frozen and the product owner explicitly authorizes transition to CU-2.
