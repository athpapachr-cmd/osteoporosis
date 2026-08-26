# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** Physiotherapy Referral v2 design, followed by RF request/PDF workflow integration.

This document exists to keep the near-term Clinic Utilities detour detailed without confusing operational tooling with Osteoporosis Module 01 or the reusable Clinical Excellence Core.

Permanent boundary:

```text
Clinical Excellence Core
→ reusable patient/workflow/navigation/auth/integration mechanics

Clinic Utilities
→ cross-module clinician-facing operational tools

Module 01 Osteoporosis
→ osteoporosis-specific clinical content
```

The utilities must ultimately live inside the Clinical Excellence workspace, but they do not count as a new clinical Module 02.

---

# 1. Detour purpose

Integrate two existing clinician-created tools into the future Clinical Excellence Cockpit:

1. **Physiotherapy Referral Generator** — structured clinical referral text generation.
2. **Radiofrequency Request / PDF Workflow** — request creation, PDF generation, lifecycle tracking and reuse of previous request data.

The detour should improve daily clinic workflow while preserving the broader product objective: better clinical practice before, during and after the consultation.

---

# 2. Source inspection already completed

## 2.1 Physiotherapy source

Standalone HTML source supplied by the product owner and inspected read-only.

Current useful capabilities:

- local/no-server operation;
- condition groups by body region;
- optional patient name/laterality/chronicity/session count;
- clinical findings;
- rehabilitation goals;
- active vs adjunct intervention wording;
- short/detailed output modes;
- copy/print;
- basic consistency warnings for traction/dry needling;
- evidence/reference section.

Current design weaknesses:

- workflow is a checkbox catalogue rather than a clinically structured referral process;
- findings are too generic across diagnoses;
- generic default goals/interventions can create repetitive referrals;
- condition-specific precautions/restrictions are under-modelled;
- text generation concatenates phrases directly rather than formatting a structured intermediate referral object;
- validation is minimal;
- some common referral pathways are missing;
- styling is standalone rather than Clinical Excellence-native.

## 2.2 Radiofrequency source

Current implementation lives in `athpapachr-cmd/ortho-reception-backend-v2` and was inspected read-only.

Existing useful pieces include:

- `/rf` protected form route;
- PDF generation from Medikey / DIROS / Thermedico templates;
- radiology PDF attachment;
- previous-application lookup;
- PostgreSQL-backed `rfa_applications` table;
- existing status field with default `application_generated`;
- patient/site/history/VAS data;
- repeat-use logic.

Important operational constraint:

> `ortho-reception-backend-v2` is currently under a separate active Digital Secretary writer lock for AC-2. No RF runtime mutation is allowed until that lock is released/replanned through the Secretary canonicals.

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

The application should generate useful text without pretending to replace the physiotherapist's assessment or to prescribe every treatment technique.

---

# 4. Physiotherapy v2 structured intermediate model

Do not generate prose directly from checkboxes. First construct a deterministic structured object:

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

Then format it through:

```text
ShortReferralFormatter
DetailedReferralFormatter
```

This separation is required so later Cockpit persistence, reuse, auditability or AI-assisted wording refinement can operate on structured data rather than reverse-engineering prose.

---

# 5. Physiotherapy v2 condition-profile architecture

Each condition profile should own only the clinically relevant suggestions for that problem.

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

Profile selections are **suggestions**, not automatically asserted findings.

Hard rule:

```text
suggested != examined
suggested != selected
selected != clinically mandatory
```

---

# 6. Initial body-region taxonomy for design review

This taxonomy is a clinical-content candidate and must undergo evidence/wording review before production freeze.

## 6.1 Cervical spine

Candidate problems:

- mechanical neck pain;
- myofascial pain / trigger-point dominant presentation;
- referred shoulder-girdle pain;
- radicular-type symptoms / cervical radiculopathy pathway;
- cervical mobility restriction;
- postural/ergonomic load-related presentation.

Candidate findings:

- active ROM restriction;
- pain provocation with movement/load;
- neurological motor/sensory deficit present/absent;
- reflex change when examined;
- neural/radicular provocation when examined;
- muscle tenderness/trigger points;
- functional impact on sleep/work/driving.

Precautions / escalation:

- new/progressive objective neurological deficit;
- myelopathic features;
- significant gait/balance change;
- other red-flag concern.

Candidate goals/directions:

- symptom reduction;
- restore tolerated ROM;
- cervical/scapular motor control and endurance;
- graded activity/exposure;
- education/self-management;
- ergonomics where relevant.

Adjunct options only when clinically appropriate:

- manual therapy;
- soft-tissue techniques;
- neurodynamic techniques;
- selected traction;
- dry needling / acupuncture where appropriate and within local practice.

## 6.2 Lumbar spine

Candidate problems:

- mechanical low-back pain;
- referred leg pain / radicular-type symptoms;
- mobility restriction;
- trunk deconditioning;
- postural/load-related recurrent low-back pain.

Candidate findings:

- lumbar ROM/function limitation;
- neurological findings when examined;
- neural tension/radicular features when examined;
- load/movement provocation;
- walking/sitting/standing tolerance;
- trunk strength/endurance deficit.

Candidate goals/directions:

- remain/return active;
- graded loading;
- trunk strength/endurance/motor control;
- mobility where appropriate;
- self-management;
- return to work/activity.

## 6.3 Shoulder

Candidate problems:

- rotator-cuff-related shoulder pain/tendinopathy;
- shoulder stiffness;
- adhesive capsulitis;
- scapular control dysfunction;
- calcific tendinopathy;
- proximal biceps-related pain;
- instability/dislocation rehabilitation;
- post-traumatic shoulder rehabilitation;
- post rotator-cuff repair / other post-operative pathway.

Candidate findings:

- active vs passive ROM pattern;
- load-related pain;
- weakness;
- painful arc when relevant;
- stiffness pattern;
- scapular control;
- functional limits: overhead use, dressing, sleep, lifting.

Precautions/restrictions:

- post-operative surgeon protocol;
- instability precautions;
- acute traumatic weakness requiring reassessment.

## 6.4 Elbow

Candidate problems:

- lateral elbow tendinopathy;
- medial elbow tendinopathy;
- stiffness;
- grip weakness;
- post-fracture stiffness;
- distal biceps/triceps rehabilitation where appropriate.

Candidate findings:

- pain with resisted loading;
- grip strength/function;
- ROM;
- work/sport load intolerance;
- neurological symptoms if relevant.

Candidate rehab directions:

- load management;
- progressive forearm/grip strengthening;
- upper-limb kinetic-chain rehabilitation;
- mobility where restricted.

## 6.5 Wrist and hand

Candidate problems:

- De Quervain tenosynovitis;
- mechanical wrist/hand pain;
- stiffness;
- reduced grip/dexterity;
- thumb CMC osteoarthritis;
- carpal tunnel conservative/post-operative rehabilitation;
- trigger finger post-operative rehabilitation;
- distal-radius fracture/post-immobilization pathway.

Candidate findings:

- ROM;
- grip/pinch strength;
- dexterity/function;
- swelling;
- sensory symptoms where relevant;
- activity-specific load intolerance.

## 6.6 Hip

Candidate problems:

- hip osteoarthritis;
- greater trochanteric pain syndrome;
- post total-hip arthroplasty;
- post hip-fracture rehabilitation;
- mobility/strength deficit.

Candidate findings:

- ROM;
- hip abductor/extensor weakness;
- gait limitation;
- sit-to-stand/stairs;
- walking tolerance;
- balance/fall risk where relevant.

Precautions:

- post-operative restrictions;
- weight-bearing status;
- fracture-healing constraints.

## 6.7 Knee

Candidate problems:

- knee osteoarthritis;
- patellofemoral pain;
- meniscal tear / meniscal lesion conservative pathway;
- knee stiffness;
- lower-limb weakness;
- ACL reconstruction;
- collateral-ligament injury;
- patellar tendinopathy;
- post total-knee arthroplasty;
- post-fracture/post-immobilization rehabilitation.

Candidate findings:

- ROM;
- effusion/swelling;
- quadriceps weakness;
- instability;
- mechanical symptoms;
- gait/stairs/sit-to-stand/function.

Precautions:

- true locking/urgent surgical concern;
- acute instability requiring reassessment;
- post-operative protocol;
- weight-bearing restrictions.

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
- post-operative foot/ankle rehabilitation.

Candidate findings:

- swelling;
- ROM;
- calf/ankle strength;
- single-leg balance/proprioception;
- gait;
- loading tolerance.

## 6.9 Muscle strain / myotendinous injury

Required context candidates:

- muscle/group;
- site;
- date/phase;
- imaging/grade if known;
- activity/sport demand.

Rehab direction:

- symptom-guided and criterion-based progressive loading;
- restore ROM/strength;
- graded return to activity/sport;
- avoid unsupported fixed timelines when clinical context is incomplete.

## 6.10 Fracture / post-immobilization

Required context:

- bone/site;
- treatment;
- healing/stability status if known;
- immobilization removed/not removed;
- weight-bearing status;
- surgeon/orthopaedic restrictions.

Safety rule:

> If rehabilitation after fracture is selected and loading/healing restrictions are unknown, show a prominent review prompt rather than silently generating unrestricted mobilisation wording.

## 6.11 Post-operative musculoskeletal rehabilitation

Required context:

- operation;
- date;
- surgeon/protocol;
- weight-bearing/activity restrictions;
- ROM restrictions;
- wound/other special considerations where clinically relevant.

The generator must not invent a generic post-operative protocol.

## 6.12 General deconditioning / balance / gait

Cross-regional pathway for:

- deconditioning after illness/immobility;
- balance deficit;
- recurrent falls/fall risk;
- gait re-training;
- generalized lower-limb weakness.

This should remain distinct from disease-specific diagnosis when that diagnosis is not established.

---

# 7. Context-sensitive goals and rehabilitation directions

Remove the current global assumption that pain + ROM + strength + motor control + function always apply.

New rule:

```text
selected condition profile
→ suggest relevant goals/directions
→ clinician confirms/changes
→ only confirmed items appear in ReferralDraft
```

Examples:

- De Quervain: pain/load tolerance, thumb/wrist function, graded tendon loading; generic cervical-style motor-control wording should not appear.
- fracture rehabilitation: ROM/loading suggestions depend on healing/restrictions;
- chronic OA: strength/function/activity often dominate;
- post-operative cases: protocol restrictions outrank generic defaults.

---

# 8. Safety / consistency engine v1

The utility should provide prompts, not autonomous treatment prohibitions.

Initial rule candidates:

```text
fracture rehab + missing weight-bearing/healing context
→ warning

post-op + missing procedure/protocol/restrictions
→ warning

traction selected + no cervical radicular context
→ warning

dry needling selected + no myofascial/trigger-point context
→ soft warning

manual/passive techniques selected without active rehabilitation direction
→ warning

gait training selected without gait/function problem
→ soft warning

new/progressive neurological deficit selected
→ prominent medical reassessment warning

red-flag concern selected
→ routine referral generation should require clinician acknowledgement
```

A later evidence/safety review should decide which are informational vs blocking.

---

# 9. Output wording rules

The referral should remain collaborative and avoid over-prescribing the physiotherapist.

Preferred structure:

```text
Clinical problem + important findings + functional impact.

Referral request + goals.

Rehabilitation direction / restrictions.

Optional reassessment/communication criteria.
```

Wording principles:

- use `Παρακαλώ για εξατομικευμένο πρόγραμμα φυσικοθεραπευτικής αποκατάστασης...`;
- emphasize active rehabilitation, education, self-management and graded loading when appropriate;
- passive techniques remain adjunctive where used;
- do not convert a positive provocation test into an unsupported definitive diagnosis;
- do not state `no neurological deficit` or `no red flags` unless selected as actually assessed;
- preserve explicit surgeon/healing restrictions exactly rather than normalizing them away;
- short and detailed versions must be generated from the same ReferralDraft.

---

# 10. Cockpit integration target

Future navigation candidate:

```text
Clinical Excellence Home
└── Clinical Tools
    ├── Physiotherapy Referral
    └── RF Requests
```

Patient-aware future flow:

```text
Patient
→ Clinical Tools
→ New Physiotherapy Referral
→ optional demographic/context prefill
→ structured draft
→ generated text
→ copy / print
```

Persistence is deliberately not frozen for the first physio implementation. Start with ephemeral generation unless the user workflow demonstrates clear value from storing referral history.

If later persisted, minimum candidate record:

```text
referral_id
date
patient_link
body_region
primary_problem
sessions_optional
final_text
source = clinician_generated
status_optional
```

Reuse should create a **new referral draft**, never overwrite historical referral truth.

---

# 11. RF workflow target

Longer-term RF clinician workflow:

```text
new request
→ application_generated / draft-like state
→ submitted
→ pending_approval
→ approved_awaiting_procedure
→ performed
```

Terminal alternatives such as `rejected` or `cancelled/void` should be added only after confirming the actual clinic workflow.

Views:

```text
Pending decision
Approved — awaiting procedure
Completed / performed
All / search
```

Each record should show at minimum:

- patient;
- request/application date;
- anatomical location;
- indication;
- consumable/provider template;
- current status;
- next action;
- procedure date when performed.

`New from previous` should clone reusable facts into a new request identity but exclude old status/approval/procedure/follow-up values unless explicitly reconfirmed.

---

# 12. Implementation order for this detour

```text
CU-1  Physiotherapy Referral v2 clinical/content + structured-draft design
CU-2  Physiotherapy Referral v2 implementation + Clinical Excellence styling
CU-3  Cockpit navigation integration / optional patient prefill boundary
CU-4  RF lifecycle/data-model design after Secretary writer lock permits fresh source inspection
CU-5  RF clinician UI + request registry/history/reuse
CU-6  RF PDF engine ownership/migration/integration cleanup
```

The RF sequence may start earlier than CU-2/CU-3 only if the Digital Secretary control plane releases the relevant runtime mutation scope and the product owner explicitly reprioritizes it.

---

# 13. Current design stop point

Current active design work should now focus on **CU-1**:

> validate and freeze the physiotherapy condition-profile taxonomy, safety/consistency rules, structured `ReferralDraft`, and output wording contract.

Do not write production physio runtime code until this design is reviewed and explicitly approved for implementation.
