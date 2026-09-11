# KNEE_OA_EVIDENCE_DESIGN_V1.md — Physio Referral Step 2

> **STATUS:** DESIGN CANDIDATE — EVIDENCE REVIEWED 2026-09-11.
> **Scope:** Knee Osteoarthritis only.
> **Parent route:** `profile_id=knee`, `route_id=knee_osteoarthritis`.
> **Parent runtime:** CU-1 Physiotherapy Referral v2, read-only in this slice.
> **Machine companion:** `contracts/knee_oa_evidence_contract_v1.yaml`.
> **Runtime implementation:** NOT AUTHORIZED by this document.

---

# 1. Purpose

Define the evidence layer that will later allow the Knee-OA referral product to remain visually simple while still answering, for each clinically relevant option:

```text
Is this a core supported intervention?
Is it useful only in selected contexts?
Is evidence limited or insufficient?
Do credible guidelines materially disagree?
Is there a recommendation against routine use?
Why is the product suggesting or cautioning about it?
Which source/year supports that statement?
When was the product evidence last reviewed?
```

The product must never manufacture consensus by averaging incompatible guideline positions.

---

# 2. Reviewed primary source set

The Step-2 review uses six major guideline/CPG authorities with complementary scopes.

## `VA_DOD_OA_2026`

**VA/DoD Clinical Practice Guideline for the Non-Surgical Management of Hip & Knee Osteoarthritis, Version 3.0, 2026.**

- provider summary: https://healthquality.va.gov/HEALTHQUALITY/guidelines/CD/OA/Osteoarthritis-CPG_2026-Provider-Summary_final_20260618.pdf
- evidence window stated by the guideline: through July 2025;
- uses GRADE direction/strength;
- especially useful because it is a current 2026 systematic guideline and explicitly separates weak-for, weak-against and insufficient/neither-for-nor-against positions.

Relevant reviewed positions include:

- weight loss for knee OA with overweight/obesity: **Strong for**;
- tailored education + physical activity for knee-OA self-management: **Weak for**;
- structured PT for hip/knee OA: **Weak for**;
- insufficient evidence to prefer one structured PT type/mode over another;
- bracing in selected knee-OA patients: **Weak for**;
- TENS: insufficient evidence for or against;
- acupuncture and dry needling: insufficient evidence for or against.

## `ACE_KNEE_OA_2026`

**Singapore Agency for Care Effectiveness, Management of knee osteoarthritis — a joint effort with patients, 24 April 2026.**

- https://www.ace-hta.gov.sg/healthcare-professionals/ace-repository-for-clinical-guidelines/management-of-knee-osteoarthritis---a-joint-effort-with-patients-acg/
- knee-specific, current 2026 guidance;
- mainstay strategies: education, exercise programmes and weight management, individualized to patient profile;
- consider allied-health referral for additional non-pharmacological strategies such as supervised exercise and walking aids;
- acupuncture may be considered as an adjunct after inadequate response to conventional therapy or when preferred by the patient.

## `EULAR_CORE_2023_UPDATE`

**EULAR recommendations for the non-pharmacological core management of hip and knee osteoarthritis: 2023 update** (published 2024).

- https://pmc.ncbi.nlm.nih.gov/articles/PMC11103326/
- individualized multicomponent management plan: LoE 1a / Strength A;
- information, education and self-management: LoE 1a / Strength A;
- exercise including strength, aerobic, flexibility or neuromotor exercise with adequate dosage/progression tailored to function/preferences/services: LoE 1a / Strength A;
- delivery mode individualized, including supervised/unsupervised and land/aquatic modes: LoE 1a / Strength A;
- healthy weight/weight loss and assistive-device considerations are part of the core non-pharmacological framework.

## `NICE_NG226_2022`

**NICE NG226, Osteoarthritis in over 16s: diagnosis and management, 2022.**

- https://www.nice.org.uk/guidance/ng226/chapter/recommendations
- exercise tailored to needs should be offered to all people with OA;
- supervised therapeutic exercise may be considered;
- long-term exercise adherence is emphasized;
- weight management is core when overweight/obesity applies;
- manual therapy should only be considered for hip/knee OA alongside therapeutic exercise and there is insufficient evidence for manual therapy alone;
- acupuncture and dry needling should not be offered for OA;
- walking aids may be considered for lower-limb OA;
- braces/tape/supports should not be offered routinely unless the specified instability/biomechanical/loading and functional criteria apply.

## `AAOS_OAK3_2021`

**AAOS Management of Osteoarthritis of the Knee (Non-Arthroplasty), Third Edition, 2021.**

- https://new.aaos.org/globalassets/quality-and-practice-resources/osteoarthritis-of-the-knee/oak3cpg.pdf
- supervised, unsupervised and/or aquatic exercise over no exercise: **Strong**;
- neuromuscular training combined with traditional exercise: **Moderate** for performance-based function/walking speed;
- self-management: **Strong**;
- patient education: **Strong**;
- sustained weight loss in overweight/obese knee-OA patients: **Moderate**;
- manual therapy in addition to exercise: **Limited**;
- acupuncture may improve pain/function: **Limited**;
- dry-needling utility/efficacy unclear; additional evidence needed: **Consensus**.

## `ACR_AF_2019`

**2019 American College of Rheumatology / Arthritis Foundation Guideline, published 2020.**

- https://acrjournals.onlinelibrary.wiley.com/doi/10.1002/art.41142
- exercise: **Strong**;
- weight loss when overweight/obese: **Strong**;
- self-efficacy/self-management: **Strong**;
- balance exercise: **Conditional**;
- cane use when impact on ambulation/stability/pain warrants: **Strong**;
- tibiofemoral bracing in appropriate knee OA: **Strong**; patellofemoral bracing: **Conditional**;
- acupuncture: **Conditional for**;
- manual therapy with exercise: **Conditional against** over exercise alone;
- TENS: **Strong against** in the full guideline framework.

---

# 3. REPLAN finding — five evidence states are insufficient

The frozen Step-1 UX contract defined five evidence states:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
recommendation_against_routine_use
not_yet_assessed
```

This model cannot honestly represent a material source conflict.

## Example: acupuncture

Reviewed sources do not merely differ in strength; they differ in direction:

```text
NICE 2022        → do not offer
AAOS 2021        → limited recommendation in favour
ACR/AF 2019      → conditional recommendation in favour
ACE 2026         → consider as adjunct in selected patients
VA/DoD 2026      → insufficient evidence for or against
```

Classifying this as simply `conditional`, `limited`, or `against` would suppress clinically relevant disagreement.

Therefore Step 2 activates the explicit Step-1 REPLAN trigger and adds:

```text
guideline_conflict_or_mixed
```

The app-facing Greek semantic label should be concise:

```text
Οι οδηγίες διαφέρουν
```

The routine visual cue should be distinct from danger/red and from insufficient-evidence/amber. A restrained indigo/violet or split-state accent plus a second non-colour cue is preferred. Exact cosmetic implementation remains Step 5/prototype scope.

This sixth state is **not** a statistical meta-analysis or majority vote. It means that reviewed guideline positions materially differ in direction or practical recommendation.

---

# 4. Source-position model

Each source keeps its own position.

Normalized direction vocabulary:

```text
strong_for
for
conditional_for
weak_for
neutral_or_insufficient
conditional_against
weak_against
against
not_addressed
```

The source's native strength/wording is also preserved separately. The normalized direction is only for deterministic product logic; it must not overwrite the original framework language.

No arithmetic score is used.

---

# 5. App-facing evidence-state resolution

Deterministic semantic rules:

## `recommended_or_supported`

Use when major reviewed sources materially align in favour of the intervention/core component and there is no material current source recommending against the same use.

## `conditional_or_context_dependent`

Use when support depends on phenotype, impairment, delivery mode, patient preference, prior response, or another explicit context and there is no material direct conflict that must be shown separately.

## `limited_or_insufficient_evidence`

Use when the reviewed evidence is predominantly insufficient/uncertain/no-recommendation and no material source clearly recommends routine use or routine avoidance.

## `guideline_conflict_or_mixed`

Use when credible reviewed guidelines materially differ in recommendation direction or in a way that could change the clinician's choice.

This state must expose source-specific positions through `i` and must not be collapsed into a fake consensus.

## `recommendation_against_routine_use`

Use when reviewed guidance materially converges against routine use for the same indication/context. The UI rationale must distinguish lack of demonstrated benefit from evidence of harm when that distinction exists.

## `not_yet_assessed`

Use when the product evidence review has not evaluated the item sufficiently. It is not a negative recommendation.

---

# 6. Knee-OA core plan

The product should load a reviewed starting plan rather than a blank form.

## 6.1 Implicit core request

`physiotherapy_assessment_and_individualized_active_rehabilitation`

Evidence state:

```text
recommended_or_supported
```

The referral is itself a request for individualized PT/rehabilitation; this does not need to consume a large visible row in the routine UI.

## 6.2 Therapeutic exercise

CU-1 ID:

```text
therapeutic_exercise
```

Evidence state:

```text
recommended_or_supported
```

Default:

```text
selected
```

Rationale for `i` layer:

> Therapeutic exercise is a core non-surgical treatment for knee OA and improves pain and function. The exact programme should be individualized.

Key sources: NICE 2022, EULAR 2023 update, AAOS 2021, ACR/AF 2019, ACE 2026, VA/DoD 2026.

## 6.3 Progressive strengthening

CU-1 ID:

```text
progressive_strengthening
```

Evidence state:

```text
recommended_or_supported
```

Default:

```text
selected
```

Rationale:

> Strengthening is a well-supported component of therapeutic exercise. Exact exercise selection, dose and progression should be individualized rather than prescribed by the referral tool.

Important nuance: several guidelines support exercise but do not establish one universally superior exercise mode. The product must not convert “strengthening is supported” into an invented exact protocol.

## 6.4 Education and self-management

CU-1 ID:

```text
education_and_self_management
```

Evidence state:

```text
recommended_or_supported
```

Default:

```text
selected
```

Key sources include EULAR Strength A, AAOS Strong, ACR/AF Strong, NICE core information/support, ACE mainstay strategy and VA/DoD Weak-for tailored education/physical activity.

## 6.5 Graded activity exposure / physical activity

CU-1 ID:

```text
graded_activity_exposure
```

Evidence state:

```text
recommended_or_supported
```

Default:

```text
selected
```

The wording must remain general: progressive physical activity/loading according to tolerance and function. Do not imply that one exact progression schedule is guideline-mandated.

---

# 7. Context-driven rehabilitation components

These should not all be visually preselected. They become relevant from phenotype/function.

## `progressive_endurance_or_capacity_work`

State: `conditional_or_context_dependent`.

Relevant when reduced walking/activity/endurance is clinically important. Aerobic exercise is included within supported exercise programmes, but not every patient needs a separately labelled endurance component.

Suggested by:

```text
walking_limitation
walking_tolerance
community_mobility
```

## `mobility_exercise_when_restricted`

State: `conditional_or_context_dependent`.

Suggested when actual ROM restriction is selected. Flexibility/mobility is an accepted exercise component, but routine mobility emphasis should not be invented when ROM is not restricted.

## `neuromuscular_proprioceptive_training`

State: `conditional_or_context_dependent`.

AAOS gives Moderate support when added to traditional exercise for performance-based function/walking speed; EULAR includes neuromotor exercise within its Strength-A exercise recommendation; ACR/AF makes balance exercise conditional.

Suggested by:

```text
balance_deficit
subjective_giving_way
recurrent_instability_episode
```

The tool must not convert subjective giving-way into proven structural instability.

## `balance_stepping_recovery_training`

State: `conditional_or_context_dependent`.

Suggested only when balance/postural-control impairment is actually present or a relevant mobility/falls context is explicitly captured.

## `gait_walking_practice`

State: `conditional_or_context_dependent`.

Suggested by walking limitation/tolerance or altered gait context. It is supported as part of individualized functional rehabilitation, not as a universal stand-alone Knee-OA treatment claim.

## `functional_task_retraining`

State: `conditional_or_context_dependent`.

Suggested by selected functional limitations such as:

```text
stairs
sit_to_stand
squat
kneeling
patient_priority_activity
```

The generated referral should name the relevant task rather than add generic functional-training prose.

## `home_exercise_programme`

State: `conditional_or_context_dependent`.

Unsupervised/home exercise can be effective and is compatible with current guideline recommendations, but supervised vs unsupervised mode should be selected according to patient needs/preferences/access. VA/DoD 2026 specifically finds insufficient evidence to prefer one structured PT mode over another.

The product may include a home programme by default later if product-owner usability testing supports that choice, but should not claim that home delivery is uniquely superior.

---

# 8. Supports and assistive devices

## Walking aid

Existing canonical CU-1 ID:

```text
walking_aid_assessment_and_training
```

Evidence state:

```text
conditional_or_context_dependent
```

NICE advises considering walking aids for lower-limb OA; ACE 2026 names walking aids as an allied-health strategy; ACR/AF strongly recommends cane use when ambulation/stability/pain impact warrants it.

**Integration seam:** the ID already exists globally but is not currently exposed in the Knee profile's UI relevance list. Step 2 does not change the frozen UI scope. A later product prototype may add it as a context-driven power-user option through a bounded presentation-scope amendment without creating a new clinical ID.

## Brace / orthosis

Existing CU-1 adjunct:

```text
orthosis_or_brace_context
```

Evidence state:

```text
conditional_or_context_dependent
```

Not default-selected.

Reasoning:

- VA/DoD 2026: bracing weak-for in selected knee-OA patients;
- ACR/AF: strong support for tibiofemoral brace in appropriate symptomatic tibiofemoral OA; conditional for patellofemoral brace;
- NICE: do not routinely offer supports unless instability/abnormal biomechanical loading and additional functional criteria apply.

These positions are compatible with a selected-patient/context-dependent state rather than a routine default.

## Taping

Existing CU-1 adjunct:

```text
taping
```

Evidence state:

```text
conditional_or_context_dependent
```

Not default-selected. NICE restricts routine use of tape/supports to specific contexts; ACR/AF conditionally supports kinesiotaping. The product should not surface taping as a core intervention.

---

# 9. Manual/soft-tissue interventions — explicit mixed guidance

## Manual therapy

CU-1 adjunct:

```text
manual_therapy
```

App state:

```text
guideline_conflict_or_mixed
```

Not default-selected; no automatic positive suggestion.

Source positions:

- NICE: only consider for hip/knee OA **alongside exercise**, insufficient evidence for use alone;
- AAOS: may be used in addition to exercise, **Limited** recommendation;
- ACR/AF: manual therapy with exercise is **conditionally recommended against over exercise alone**.

Routine bubble when selected:

```text
Οι οδηγίες διαφέρουν
```

Concise `i` rationale:

> Manual therapy is considered an adjunct rather than a substitute for active rehabilitation. Guideline positions differ on whether it adds meaningful benefit beyond exercise.

## Soft-tissue techniques

CU-1 adjunct:

```text
soft_tissue_techniques
```

App state:

```text
guideline_conflict_or_mixed
```

Rationale: soft-tissue techniques overlap with manual therapy/massage evidence. AAOS provides limited support for massage as adjunct/usual care, while ACR/AF conditionally recommends against massage for OA symptom reduction and NICE does not support manual therapy alone. The product must not present soft-tissue work as a core evidence-based Knee-OA treatment.

---

# 10. Acupuncture — flagship conflict-state example

CU-1 adjunct:

```text
acupuncture
```

App state:

```text
guideline_conflict_or_mixed
```

Default:

```text
not selected
```

No automatic positive suggestion in the first prototype.

Source-specific positions:

```text
NICE 2022     against routine use / do not offer
AAOS 2021     limited in favour for pain/function
ACR/AF 2019   conditional in favour
ACE 2026      consider as adjunct after inadequate conventional response or patient preference
VA/DoD 2026   insufficient evidence for or against
```

Routine selected-state bubble:

```text
Οι οδηγίες διαφέρουν
```

`i` rationale:

> Acupuncture recommendations are not uniform across major guidelines. Some support selected adjunctive use, NICE recommends against its use for OA, and VA/DoD 2026 finds evidence insufficient for or against. It should never replace exercise/self-management in this pathway.

This item demonstrates why the conflict state is necessary.

---

# 11. Dry needling — preserve existing exclusion

The frozen Knee v1.1 clinical profile explicitly excludes dry needling from the Knee-OA selectable pathway.

Evidence review does **not** justify reopening that decision in Step 2:

```text
NICE 2022     do not offer for OA
AAOS 2021     efficacy unclear; additional evidence required
VA/DoD 2026   insufficient evidence for or against
```

The item therefore remains:

```text
product_selectable = false
cu1_knee_oa_surface = excluded
```

This is not represented to the user as “proven ineffective”; it is simply not an offered Knee-OA option in the current product surface.

---

# 12. Weight management — strong evidence, current machine-seam gap

Major current frameworks consistently treat weight management as core when overweight/obesity applies:

- VA/DoD 2026: **Strong for** weight loss in knee OA with overweight/obesity;
- NICE: weight management is a core treatment when applicable;
- EULAR: healthy weight/weight loss, LoE 1a / Strength A;
- ACR/AF: **Strong**;
- AAOS: **Moderate**;
- ACE 2026: mainstay strategy.

Clinical evidence state:

```text
recommended_or_supported
```

However, the current CU-1 option catalog has no dedicated selectable weight-management rehabilitation/referral ID even though the frozen Knee clinical profile already names weight-management support/referral when clinically relevant.

Step-2 decision:

```text
role = product_advisory_only_for_now
runtime_auto_trigger = blocked_without_explicit_overweight_obesity_context
new_CU1_id = NOT AUTHORIZED in Step 2
```

This is a documented integration gap, not permission to infer BMI or body-size status from absent data.

A later Step-3/Step-5 design may decide whether the product needs a small explicit context control and whether the referral should mention support/referral. That decision requires a bounded contract amendment rather than hidden free-text inference.

---

# 13. Suggestion policy

## Core omission suggestions

If the clinician removes one of the reviewed default core components:

```text
therapeutic_exercise
progressive_strengthening
education_and_self_management
graded_activity_exposure
```

the product may show a compact evidence-backed suggestion.

The suggestion must include:

```text
intervention
app evidence state
one source/year cue
one-tap add
optional i
```

It must not block the referral.

## Phenotype/function suggestions

Allowed only when the trigger is present in structured state.

Examples:

```text
ROM restriction
→ mobility_exercise_when_restricted

quadriceps/objective weakness or strength-related functional limitation
→ progressive_strengthening emphasis

balance deficit / relevant instability symptom
→ neuromuscular_proprioceptive_training

walking limitation
→ gait_walking_practice

stairs / sit-to-stand
→ functional_task_retraining with task-specific wording
```

Absence of a finding must not be treated as a negative finding.

## No automatic adjunct promotion

The first Knee-OA prototype must **not** automatically suggest:

```text
manual_therapy
soft_tissue_techniques
acupuncture
taping
brace
```

unless a later explicit context rule is reviewed and frozen. These remain clinician-selected/power-user options.

---

# 14. Caution/bubble policy

The UI is informative, not punitive.

Examples:

```text
limited_or_insufficient_evidence
→ Περιορισμένη τεκμηρίωση

guideline_conflict_or_mixed
→ Οι οδηγίες διαφέρουν

recommendation_against_routine_use
→ Δεν συνιστάται για συνήθη χρήση
```

The `i` layer must identify which source says what when conflict exists.

Do not use one red warning to represent both uncertainty and disagreement.

---

# 15. Provenance / freshness contract

Every material evidence item must retain:

```text
source_id
source organization/framework
source version/publication year
source-native recommendation direction/strength when available
product-normalized direction
applicability/context
reviewed_on
status
```

The user-facing product may display, for example:

```text
VA/DoD · 2026
Reviewed · Sep 2026
```

or, for conflicting guidance:

```text
Οι οδηγίες διαφέρουν
NICE 2022 · AAOS 2021 · VA/DoD 2026
Reviewed · Sep 2026
```

The review date must never be substituted for a guideline publication/version date.

---

# 16. Evidence update governance

No autonomous rule mutation.

```text
surveillance detects possible change
→ candidate evidence update
→ source review
→ classify impact
→ clinician/product-owner approval
→ versioned contract update
→ tests/review
→ later runtime release
```

Candidate impact classes:

```text
confirming_no_change
clarifying_no_state_change
potential_state_change
state_change
source_conflict_changed
source_withdrawn_or_superseded
```

A newer source does not automatically erase an older still-relevant framework. Supersession must be explicit.

---

# 17. Existing CU-1 compatibility findings

Compatible without taxonomy rewrite:

```text
therapeutic_exercise
progressive_strengthening
progressive_endurance_or_capacity_work
mobility_exercise_when_restricted
graded_activity_exposure
graded_loading
education_and_self_management
home_exercise_programme
neuromuscular_proprioceptive_training
balance_stepping_recovery_training
gait_walking_practice
functional_task_retraining
manual_therapy
soft_tissue_techniques
acupuncture
taping
orthosis_or_brace_context
```

Presentation seam only:

```text
walking_aid_assessment_and_training
→ canonical ID already exists
→ currently absent from Knee UI relevance scope
```

Machine-seam gap:

```text
weight-management support/referral
→ present in frozen Knee clinical profile
→ no dedicated current selectable option ID
→ Step 2 records advisory-only; no hidden taxonomy mutation
```

Existing exclusion preserved:

```text
dry needling
→ excluded from Knee-OA selectable profile
```

No Step-2 finding requires rewriting the broad CU-1 route taxonomy.

---

# 18. Step-2 acceptance decision

The evidence architecture is viable if the machine companion implements:

```text
source registry
+ source-specific positions
+ six-state app evidence model
+ deterministic non-arithmetic resolution
+ default/core plan
+ context-driven suggestion triggers
+ explicit conflict explanations
+ integration-gap metadata
+ freshness/review metadata
```

Runtime implementation remains a later, separately authorized step.
