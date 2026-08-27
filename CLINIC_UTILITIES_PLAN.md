# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; all regional v1.1 profiles, Shared Fracture v1.1 and Shared Muscle / Myotendinous Injury v1.1 frozen.

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

Structured intermediate model remains:

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

Hard rules remain: suggested/examined/selected/mandatory are distinct; symptoms/tests/imaging do not autonomously create diagnoses; not-assessed does not mean normal; adjuncts do not replace core rehabilitation; clinician-entered diagnoses may be carried but not inferred.

---

# 2. Frozen profile status

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN
wrist_hand_v1_1 = FROZEN
knee_v1_1 = FROZEN
hip_v1_1 = FROZEN
ankle_foot_v1_1 = FROZEN
shared_fracture_v1_1 = FROZEN
shared_muscle_myotendinous_v1_1 = FROZEN
```

---

# 3. Shared Muscle / Myotendinous Injury v1.1 frozen design

Authoritative design:

```text
clinic_utilities/physio_profiles/shared_muscle_myotendinous_v1_1.md
```

The shared route owns acute muscle/myotendinous rehabilitation semantics once across regions.

```text
acute_muscle_myotendinous_injury_rehabilitation
→ muscle/site + injury type/phase/tissue context
→ conservative/specialist/postoperative disposition
→ restrictions / actual deficits / functional demand
→ confirmed goals / rehabilitation directions
```

High-visibility workflow groups:

```text
quadriceps / rectus femoris
acute adductor injury
gastrocnemius / soleus / calf
hamstring strain / partial tear
```

Visible but less frequent:

```text
pectoralis major muscle/myotendinous injury
biceps muscle-belly/myotendinous injury
abdominal-wall muscle injury
```

Rare/secondary includes iliopsoas/hip-flexor, tibialis anterior and other uncommon acute muscle injuries.

Frozen structural rules:

```text
acute strain != chronic tendinopathy
muscle pain != exact tear diagnosis
MRI/US classification != fixed rehabilitation timetable
bony avulsion → Shared Fracture
major free-tendon rupture/avulsion without disposition → specialist structural route
postoperative repair → exact protocol owns progression
<2 cm retraction + established conservative decision may proceed through routine PT workflow
<2 cm alone != autonomous clearance
>=2 cm / multi-tendon complete avulsion / major weakness-deformity / high-demand unresolved case
→ prominent specialist-disposition check
→ no automatic surgery recommendation
```

Return-to-sport/work is multi-domain and never based on elapsed time, MRI grade/appearance or strength symmetry alone.

Adjunct policy:

```text
acupuncture → optional clinician-selected adjunct; no accelerated-healing claim
dry needling → excluded
ESWT → excluded as default acute-muscle healing recommendation
therapeutic ultrasound → excluded as default acute-muscle healing recommendation
compression / taping → treating-physiotherapist discretion
```

Progressive active rehabilitation remains the core.

---

# 4. Safety / consistency engine

```text
major acute weakness / palpable defect / rupture-avulsion concern
→ structural reassessment

calf pain/swelling + unresolved DVT concern
→ no routine calf-strain wording

large/expanding haematoma or anticoagulation bleeding concern
→ medical reassessment

severe escalating pain / tense swelling / compartment concern
→ urgent pathway

new neurological/vascular deficit
→ urgent/medical reassessment

postoperative repair + missing protocol/restrictions
→ warning

significant contusion + atypical persistent/worsening course
→ myositis-ossificans/other reassessment semantics

material safety concern + no disposition
→ no routine reassuring wording
```

---

# 5. Final remaining shared design profile

Current remaining sequence is now:

```text
generalized deconditioning / balance / gait
```

This is the final currently planned shared CU-1 clinical/content profile. Its completion does **not** itself authorize runtime implementation; implementation requires a separate product-owner decision after design completeness review.

---

# 6. Output wording rules

```text
Clinical problem + muscle/site/injury context + actual findings + functional impact.
Referral request + goals.
Actual restrictions / permitted progression.
Optional reassessment/communication criteria.
```

No unsupported tear grade, automatic timeline, or reassuring safety statement from missing data.

---

# 7. Implementation boundary

CU-1 remains **design only**.

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. Do not write production physiotherapy runtime code until CU-1 is sufficiently frozen and the product owner explicitly authorizes transition to implementation.
