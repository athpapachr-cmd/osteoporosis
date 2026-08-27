# CU-1 Core Typed Contract v1 — FROZEN DESIGN

> **STATUS:** FROZEN PRE-CODE MACHINE/DESIGN CONTRACT.
> **Slice:** CU-1 Physiotherapy Referral v2.
> **Clinical sources:** frozen regional/shared `*_v1_1.md` profiles.
> **Machine registry:** `clinic_utilities/contracts/cu1_registry_v1.yaml`.
> **Synthetic fixtures:** `clinic_utilities/contracts/cu1_design_fixtures_v1.yaml`.
> **Runtime:** NOT AUTHORIZED.

This contract resolves the prior design-completeness blockers B1–B6 without reopening the frozen clinical taxonomy.

---

# 1. Normative principles

```text
profile Markdown = clinical/content authority
this contract + registry = cross-profile machine/formatting authority
runtime must not invent new semantic states
```

Hard invariants remain:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
provocation/test finding != diagnosis
imaging finding != automatically symptomatic diagnosis
not_assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

Canonical machine identifiers are lowercase `snake_case`. Existing mixed-case profile keys are preserved only as aliases in the registry.

---

# 2. Common enums / semantic types — FROZEN

## 2.1 AssertionState

```text
not_stated
yes
no
```

Use only for an explicit clinician assertion such as `formal_carpal_tunnel_diagnosis`.

`not_stated` means no formal assertion was supplied. It does not mean absent.

## 2.2 AssessmentState

```text
not_assessed
normal
abnormal
```

Use for examination domains such as motor, sensory, reflex or vascular status.

## 2.3 PresenceState

```text
not_assessed
present
absent
```

Use when the concept is presence/absence rather than normal/abnormal.

## 2.4 Laterality

```text
left
right
bilateral
midline
not_applicable
not_stated
```

## 2.5 Visibility

```text
routine
visible_less_frequent
rare_advanced
context_only
shared_gateway
```

Visibility controls navigation only; it does not alter clinical validity or formatter certainty.

## 2.6 SafetyScreenState

```text
not_assessed
no_specific_concern_identified
concern_present
```

No `no red flags` wording is generated from `no_specific_concern_identified`; it only records the clinician-facing screen state.

## 2.7 ClinicianDisposition

```text
none_recorded
reviewed_and_appropriate_to_proceed
medical_reassessment_arranged
imaging_or_specialist_pathway_underway
urgent_or_same_day_assessment_arranged
routine_physiotherapy_deferred
other
```

Profile-specific display labels may be more specific, but the machine state must map to one of these values.

## 2.8 SafetySeverity

```text
info
soft_warning
hard_warning_ack_required
block_until_disposition
urgent_reassessment
```

Semantics:

| severity | formatter allowed? | acknowledgement required? | disposition required? | generated referral safety sentence? |
|---|---:|---:|---:|---:|
| `info` | yes | no | no | no by default |
| `soft_warning` | yes | no | no | no |
| `hard_warning_ack_required` | yes after acknowledgement | yes | no unless rule says otherwise | only if clinician explicitly selects wording |
| `block_until_disposition` | no until valid disposition | yes | yes | only disposition/context selected by clinician |
| `urgent_reassessment` | no routine referral until urgent disposition is recorded | yes | yes | never routine reassurance |

If several safety results coexist, the highest severity wins for generation blocking.

## 2.9 ProblemWordingMode

```text
presentation
formal_diagnosis
established_structural_diagnosis
postoperative
shared_structural
```

The registry determines which modes each route permits.

---

# 3. ReferralDraft v1 — FROZEN

```text
ReferralDraftV1
  contract_version: "cu1_referral_draft_v1"
  patient_context: PatientContext
  body_region: canonical profile id
  primary_problem: ProblemSelection
  secondary_problems[]: ProblemSelection
  findings[]: FindingSelection
  functional_impairments[]: CodedSelection
  precautions[]: CodedSelection
  explicit_restrictions[]: RestrictionSelection
  goals[]: CodedSelection
  rehab_directions[]: CodedSelection
  adjunct_options[]: AdjunctSelection
  measurements[]: MeasurementSelection
  safety: SafetyState
  sessions_optional: integer|null
  clinician_free_text_optional: string|null
```

The first implementation remains ephemeral. The existence of a typed contract does not authorize or require persistence.

## 3.1 PatientContext

```text
age_years_optional: integer|null
skeletal_maturity_optional: immature|mature|not_stated|null
sport_or_work_demand_optional: string|null
relevant_medical_context_ids[]
free_text_optional: string|null
```

No patient identifier is part of the CU-1 first-implementation contract.

## 3.2 ProblemSelection

```text
problem_id: stable UUID/string local to draft
profile_id: canonical profile id
route_id: canonical route id
wording_mode: ProblemWordingMode
formal_assertion_state_optional: AssertionState|null
subtype_id_optional: canonical subtype id|null
laterality: Laterality
chronicity_or_phase_optional: canonical route-specific enum|string|null
context: object
shared_target_optional: SharedTarget|null
source_route_optional: canonical route id|null
```

Rules:

```text
route_id must exist in cu1_registry_v1.yaml
subtype_id, if supplied, must be allowed by route
formal_assertion_state may only be used where registry permits it
formal_diagnosis wording requires either:
  formal_assertion_state=yes
  OR a route whose registry semantics require an established diagnosis
```

## 3.3 SharedTarget

```text
profile_id: shared_fracture|shared_muscle_myotendinous|shared_deconditioning_balance_gait
route_id: canonical shared route id
subtype_or_site_id_optional: canonical id|null
```

A regional gateway is navigation only. The shared target is the semantic owner.

## 3.4 FindingSelection

```text
finding_id: canonical finding id or profile-scoped id
state_optional: AssessmentState|PresenceState|null
laterality_optional: Laterality|null
value_optional: string|number|null
unit_optional: string|null
free_text_optional: string|null
```

If a finding is not selected, the formatter must not infer its opposite.

## 3.5 RestrictionSelection

```text
restriction_id: canonical restriction id
state_or_value: canonical enum|string
source: clinician_entered|written_protocol|patient_reported|other
notes_optional: string|null
```

`written_protocol` and `clinician_entered` outrank generic route suggestions. Patient-reported restrictions remain labelled as patient-reported in detailed output if rendered.

## 3.6 AdjunctSelection

```text
adjunct_id: canonical adjunct id
selected: true
provenance: clinician_selected|therapist_proposed_context
```

An adjunct can never satisfy the presence of a core rehabilitation direction.

## 3.7 MeasurementSelection

```text
measurement_id
value
unit_optional
component_values_optional: object
performed: true
```

Measurements quantify function; they do not autonomously create diagnoses.

---

# 4. Shared structural contexts — typed homes

## 4.1 FractureContext

When `profile_id=shared_fracture`, `ProblemSelection.context` may contain:

```text
fracture_site
fracture_date_optional
fracture_phase
fracture_context
treatment
healing_stability_status
immobilization_status
weight_bearing_status
upper_limb_use_status
rom_status
loading_strengthening_status
orthopaedic_instructions_source
follow_up_due_optional
repeat_imaging_due_optional
```

The enumerations and precedence remain those frozen in `shared_fracture_v1_1.md`.

Unknown/not-stated healing, WB/use, ROM or loading status cannot be reformatted as unrestricted.

## 4.2 MuscleMyotendinousContext

When `profile_id=shared_muscle_myotendinous`:

```text
muscle_group
specific_muscle_optional
injury_date_optional
injury_phase
injury_type
injury_location_optional
mri_or_ultrasound_confirmed: AssertionState
classification_system_optional
classification_grade_optional
retraction_or_gap_cm_optional
number_of_tendons_involved_optional
management_context
explicit_loading_restriction_optional
explicit_rom_or_stretch_restriction_optional
running_restriction_optional
sprinting_or_kicking_restriction_optional
sport_or_work_restriction_optional
surgeon_or_sports_medicine_instruction_optional
```

`retraction_or_gap_cm < 2` is contextual only and is never an autonomous clearance rule.

## 4.3 DeconditioningContext

When `profile_id=shared_deconditioning_balance_gait`:

```text
frailty_established: AssertionState
falls_history_status
injurious_fall_context: PresenceState
unable_to_rise_after_fall: PresenceState
loss_of_consciousness_with_fall: PresenceState
walking_aid_context_optional
```

SPPB/TUG/gait-speed/5xSTS values remain measurements, not diagnostic gates.

---

# 5. Route ownership / precedence — FROZEN

The following precedence applies globally:

```text
urgent safety disposition
> exact surgeon/written protocol
> shared structural owner (fracture or shared muscle when applicable)
> regional postoperative owner where registry declares postoperative_primary
> regional established structural route
> regional presentation route
> secondary problem / modifier
> adjunct
```

## 5.1 Fracture

```text
established fracture / fracture fixation / post-immobilization fracture rehabilitation
→ Shared Fracture is semantic primary owner
→ regional entry is navigation/source_route only
```

A regional postoperative route may describe surgery associated with a fracture only as context; it must not own healing/WB/ROM/loading semantics.

## 5.2 Knee postoperative exclusivity

```text
ACL nonoperative/prehab → acl_injury_instability_rehabilitation
ACL reconstruction → postoperative_knee_rehabilitation + subtype acl_reconstruction
MCL nonoperative → mcl_injury_rehabilitation
MCL repair/reconstruction → postoperative_knee_rehabilitation + subtype mcl_repair_or_reconstruction
```

## 5.3 Wrist/hand postoperative resolution

```text
flexor/extensor tendon repair
→ digital_tendon_injury_rehabilitation remains PRIMARY structural/protocol owner
→ subtype *_repair_postoperative
→ generic postoperative_wrist_hand_rehabilitation MUST NOT be selected as co-primary

thumb UCL/RCL repair/reconstruction
→ thumb_mcp_collateral_ligament_injury_rehabilitation remains PRIMARY
→ operative state lives in context
→ WH11 generic postoperative route not co-primary

sagittal-band/extensor stabilization
→ sagittal_band_injury_extensor_tendon_instability remains PRIMARY
→ operative state lives in context
→ WH11 generic postoperative route not co-primary

other wrist/hand operation without a dedicated structural owner
→ postoperative_wrist_hand_rehabilitation
```

## 5.4 Shoulder/knee generic postoperative

Where the frozen regional profile already provides a dedicated postoperative route and no more specific frozen operative route exists, the regional postoperative route is primary and exact procedure/protocol is typed context.

## 5.5 Shared muscle postoperative

```text
postoperative_or_repair_protocol
→ shared muscle generic loading suggestions become subordinate
→ exact surgical/structural route/protocol is authoritative
```

---

# 6. SafetyResult v1 — FROZEN

```text
SafetyResult
  rule_id
  severity: SafetySeverity
  message_key
  acknowledgement_required: boolean
  disposition_required: boolean
  formatter_blocked: boolean
  clinician_disposition: ClinicianDisposition
  source_profile_id
  source_route_id_optional
```

`SafetyState`:

```text
screen_status: SafetyScreenState
results[]: SafetyResult
highest_severity: derived
```

Deterministic rules:

```text
screen_status=concern_present + no coded result
→ hard_warning_ack_required

any result severity=block_until_disposition
+ clinician_disposition=none_recorded
→ formatter blocked

any result severity=urgent_reassessment
+ disposition not urgent/same-day or routine_physiotherapy_deferred
→ routine formatter blocked

no selected concern
→ never generate `no red flags`

not_assessed
→ never generate normal/reassuring negative wording
```

Profile-specific safety rule IDs may be added only if they map to this common severity behavior.

---

# 7. Formatter contract — FROZEN

Both formatters accept exactly one validated `ReferralDraftV1` and registry v1.

```text
ShortReferralFormatter(draft, registry) -> text
DetailedReferralFormatter(draft, registry) -> text
```

They do not mutate the draft and do not infer diagnoses.

## 7.1 Common generation gate

Before formatting:

```text
validate route/subtype ids
apply canonical aliases
resolve gateway to semantic owner
apply route precedence
validate required typed context for route
calculate SafetyResult set
block if highest severity requires unresolved disposition
```

Validation errors are clinician-facing and are not converted to invented clinical defaults.

## 7.2 Problem wording

Order:

```text
1. primary problem
2. clinically relevant secondary problems
3. selected findings / functional impact
4. restrictions / precautions
5. selected goals
6. core rehab directions
7. selected adjuncts
8. reassessment/disposition wording when clinician explicitly selected and appropriate
```

Rules:

```text
formal diagnosis wording only when permitted by ProblemSelection
presentation wording otherwise
not_stated/not_assessed values omitted
normal findings rendered only when explicitly assessed normal AND clinically useful
no global `no neurological deficit`
no global `no red flags`
```

## 7.3 Restrictions

Restrictions are rendered before generic rehabilitation directions when they affect activity.

Precedence:

```text
written protocol
> explicit clinician instruction
> entered shared structural restriction
> route suggestion
```

A contradictory generic suggestion is omitted, not merely followed by a warning.

## 7.4 Shared gateway rendering

A regional gateway does not create duplicate diagnoses.

Example semantic composition:

```text
Hip navigation: pelvic apophyseal avulsion
→ primary semantic route: fracture_rehabilitation_post_immobilization
→ site: asis_apophyseal_avulsion
→ optional source phrase: hip/groin region
```

The output states one fracture problem, not both a regional pseudo-diagnosis and shared fracture diagnosis.

## 7.5 Adjunct rendering

Adjuncts appear after core rehabilitation and only if explicitly selected.

Evidence caveats belong in clinician UI/metadata by default, not in routine referral prose, unless the frozen profile explicitly requires a wording qualification.

`therapist_proposed_context` is rendered as context, not as a clinician request.

## 7.6 ShortReferralFormatter

Target: compact referral suitable for routine copy/print.

Required structure:

```text
problem + key context
selected important findings/function
important restrictions
selected goals + core rehabilitation direction
selected adjuncts if any
```

Default omission rules:

```text
omit empty sections
omit detailed negative/normal screens unless materially useful
omit evidence commentary
omit low-priority secondary findings when they add no actionable information
retain all explicit restrictions and safety-relevant clinician disposition
```

No fixed character limit is frozen; semantic completeness outranks arbitrary truncation.

## 7.7 DetailedReferralFormatter

Adds, when selected/available:

```text
injury/procedure date or phase
formal subtype/context
objective neurological/neurovascular components
relevant measurements
structural/healing/immobilization context
full explicit restrictions
expanded functional impairments
secondary problems
reassessment criteria
source-labelled patient-reported protocol/restriction context
```

It still omits `not_assessed`/`not_stated` rather than rendering them as normal.

## 7.8 Free text

Free text is clinician-authored content. Formatter must:

```text
preserve meaning
normalize surrounding whitespace
not convert free text into machine diagnosis/state
not interpolate HTML
```

---

# 8. Registry ownership — FROZEN

`cu1_registry_v1.yaml` is the canonical machine namespace for:

```text
profile ids
route ids
aliases
subtype ids
gateway mappings
visibility
allowed wording modes
primary-owner precedence
common adjunct ids
```

Rules:

```text
Markdown key not in registry
→ not a valid runtime id

alias in registry
→ normalize to canonical id before validation/formatting

regional gateway
→ exact shared target from registry; runtime must not derive by string matching
```

---

# 9. Synthetic design fixtures — FROZEN TEST ORACLES

`cu1_design_fixtures_v1.yaml` is normative for design behavior. Fixtures are synthetic and contain no patient data.

Each fixture specifies:

```text
input semantic selections
expected canonical routing
expected safety severity/blocking
expected Short formatter section structure
expected Detailed formatter section structure
forbidden inference/output
```

Implementation may vary typography but must preserve the expected semantic behavior.

---

# 10. Versioning / changes

This is `cu1_referral_draft_v1` / registry `cu1_registry_v1`.

Before runtime exists, corrections require a design PR and repeat completeness review if they alter:

```text
route ownership
shared gateway target
safety blocking behavior
formatter diagnosis/restriction semantics
common enum meaning
```

After runtime exists, registry/schema changes require explicit compatibility/migration handling.

---

# 11. Persistence/privacy boundary

First implementation remains:

```text
ephemeral structured draft
→ generated text
→ copy / print
```

No database/localStorage persistence is part of this freeze. No patient identifier is required by this contract. Real patient data must never be committed to this public repository.

---

# 12. Freeze classification

```text
B1 typed structured homes = RESOLVED
B2 registry/gateway authority = RESOLVED
B3 precedence/primary ownership = RESOLVED
B4 safety severity/blocking/disposition = RESOLVED
B5 formatter interface/output/omission = RESOLVED
B6 normalized enums/tri-states/key namespace = RESOLVED
```

This classification is subject to the independent repeat design-completeness review. It is not runtime authorization.
