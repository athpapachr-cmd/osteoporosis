# CLINIC_UTILITIES_PLAN.md — controlled detour plan

> **STATUS:** SUPPORTING DESIGN PLAN — not a seventh active canonical authority.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Scope:** cross-module Clinic Utilities / Clinical Operations.
> **Current focus:** CU-1 Physiotherapy Referral v2 design; all regional v1.1 profiles, Shared Fracture v1.1 and Shared Muscle / Myotendinous Injury v1.1 frozen; Generalized Deconditioning / Balance / Gait v1 active design candidate.

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

# 2. Frozen / active profile status

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
shared_deconditioning_balance_gait_v1 = ACTIVE DESIGN CANDIDATE / NOT FROZEN
```

---

# 3. Generalized Deconditioning / Balance / Gait v1 — active candidate frame

Authoritative candidate:

```text
clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1.md
```

Shared route:

```text
functional_deconditioning_balance_gait_rehabilitation
```

Candidate presentation families:

```text
generalized deconditioning / functional decline
balance impairment / falls-risk rehabilitation
gait / mobility impairment rehabilitation
post-illness / post-hospital deconditioning
frailty-associated functional decline — established/context only
```

Hard boundaries:

```text
deconditioning != frailty automatically
one fall != recurrent falls
fear of falling != objective balance impairment
performance-test threshold != autonomous diagnosis
new unexplained gait disorder != generic deconditioning
age alone != indication for physiotherapy
assistive device != automatically mandatory
```

Potential core rehabilitation is individualized/progressive and may include resistance/strength, functional transfers, balance/coordination/stepping, power where appropriate, gait/walking, aerobic/endurance conditioning when medically appropriate, stairs/community mobility and falls-prevention exercise when indicated.

Falls management remains multifactorial when medication, vision/hearing, cardiovascular, neurological, vestibular, foot/footwear, home-environment or other risk factors are present.

---

# 4. Safety / consistency engine

```text
new focal neurological deficit / acute gait change
→ medical/neurological reassessment

syncope / presyncope / unexplained LOC
→ no generic falls-exercise-only wording

unstable chest pain / cardiopulmonary symptoms / marked unexplained breathlessness
→ medical reassessment

acute vestibular syndrome / severe new vertigo with neurological concern
→ medical/vestibular pathway

fracture or loading restriction unresolved
→ Shared Fracture restrictions govern

DVT / vascular concern, infection/systemic deterioration, acute cognitive change
→ medical/urgent reassessment semantics

abnormal TUG/gait speed/5xSTS/SPPB alone
→ measurement context only, not autonomous diagnosis

material safety concern + no disposition
→ no routine reassuring wording
```

---

# 5. Final CU-1 design sequence

```text
generalized deconditioning / balance / gait — ACTIVE CANDIDATE
```

After this profile is product-owner reviewed and frozen, CU-1 requires a **design-completeness review**. Completion of design does not authorize implementation. Runtime implementation requires a separate explicit product-owner decision.

---

# 6. Implementation boundary

CU-1 remains **design only**.

```text
ephemeral structured draft
→ generated text
→ copy / print
```

Persistence is not frozen. Do not write production physiotherapy runtime code until design completeness is reviewed and implementation is explicitly authorized.
