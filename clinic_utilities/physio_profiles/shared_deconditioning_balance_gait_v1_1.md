# Shared Generalized Deconditioning / Balance / Gait Physiotherapy Referral Profile v1.1 — CU-1

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Supersedes as active design:** `clinic_utilities/physio_profiles/shared_deconditioning_balance_gait_v1.md`.
> **Purpose:** own reusable rehabilitation semantics for generalized deconditioning and established frailty-associated functional decline while preserving weakness, coordination, falls, mobility, walking-aid and safety context without converting functional tests into autonomous diagnoses.
> **Runtime:** NOT AUTHORIZED.

---

# 1. Architectural role

Shared route:

```text
functional_deconditioning_balance_gait_rehabilitation
```

Routing model:

```text
SHARED / REGIONAL / FRACTURE CONTEXT
→ functional presentation family
→ actual weakness / coordination / falls / mobility findings
→ medical restrictions + assistive-device context
→ clinician-confirmed goals / rehabilitation directions
→ ShortReferralFormatter / DetailedReferralFormatter
```

This profile does **not** create neurological, vestibular, cardiopulmonary or frailty diagnoses from performance findings alone.

---

# 2. Frozen routine presentation families

## D1 — Generalized deconditioning / functional decline

Structured key:

```text
generalized_deconditioning_functional_decline
```

Display:

> Γενικευμένη αποδυνάμωση / μείωση φυσικής ικανότητας και λειτουργικότητας

Direct route, but not a high-frequency referral in the product-owner workflow.

Possible context:

```text
reduced activity / prolonged inactivity
multimorbidity
older-adult functional decline
reduced walking tolerance
reduced exercise tolerance
generalized weakness if assessed
poor coordination if assessed
transfer difficulty
stairs difficulty
reduced community mobility
ADL-related physical limitation
```

Hard rule:

```text
generalized weakness / fatigue without adequate medical context
!= autonomous deconditioning diagnosis
```

## D2 — Frailty-associated functional decline — established diagnosis/context only

Structured key:

```text
frailty_associated_functional_decline
```

Direct selectable route when frailty has already been clinician-established or explicitly documented.

Possible context:

```text
formal_frailty_status_or_tool_optional
multimorbidity_context
sarcopenia_context_if_established
low_physical_activity_context
falls_history_context
ADL_or_IADL_limitation
generalized weakness
poor coordination
walking-aid context
```

Hard rules:

```text
SPPB alone != frailty diagnosis
TUG alone != frailty diagnosis
gait-speed alone != frailty diagnosis
5xSTS alone != frailty diagnosis
acute illness + poor performance != stable frailty classification
```

---

# 3. Non-routine presentation families — context/findings, not top-level routine referrals

The product owner does not routinely refer isolated balance-only, gait-only or post-hospital deconditioning presentations.

Therefore these are retained as reusable context, findings and safety/coordination semantics rather than first-line routes:

```text
balance_impairment_context
gait_mobility_impairment_context
post_illness_or_post_hospital_deconditioning_context
```

Hard rules:

```text
new / unexplained gait disorder
→ not generic gait/deconditioning wording
→ medical/neurological/vestibular/structural reassessment as appropriate

post-hospital decline + medical stability not established
→ no generic exercise progression
```

Established neurological diagnoses such as Parkinson disease, stroke or peripheral neuropathy are not routine CU-1 referral pathways in this product-owner workflow.

---

# 4. Direct findings — selectable only when actually assessed

## Strength / function

```text
generalized weakness
lower-limb weakness
upper-limb functional weakness
sit-to-stand difficulty
reduced repeated-chair-rise capacity
reduced stair capacity
reduced walking endurance
reduced power / rapid-force capacity if assessed
fatigue with functional activity
```

## Coordination / postural control

```text
poor_coordination
static_balance_deficit
dynamic_balance_deficit
turning_instability
stepping_or_recovery_deficit
dual_task_difficulty_if_assessed
uneven_surface_difficulty
obstacle_negotiation_difficulty
```

Hard rule:

```text
poor coordination != cerebellar / neurological diagnosis automatically
```

## Mobility / gait

```text
slow gait speed if measured
unsteady gait
short walking distance/tolerance
assistance required for gait
walking-aid use
transfer assistance
stairs limitation
community mobility limitation
outdoor mobility limitation
```

## Fear / confidence

```text
fear_or_concern_about_falling
reduced_confidence_with_mobility
```

Hard rule:

```text
fear of falling != objective balance deficit
```

---

# 5. Falls-history model

Prominent structured history:

```text
falls_history_status:
  none_reported
  single_fall
  recurrent_falls
  not_assessed

injurious_fall_context: yes / no / not_assessed
unable_to_rise_after_fall: yes / no / not_assessed
loss_of_consciousness_with_fall: yes / no / not_assessed
near_falls_optional
fear_or_concern_about_falling_optional
```

Hard rules:

```text
one fall != recurrent falls
loss of consciousness with fall != routine falls-exercise-only pathway
injurious fall may require broader reassessment
not assessed != no falls
```

---

# 6. Functional measurement policy

## Preferred optional product-owner battery — SPPB

Preferred optional functional battery for this workflow:

```text
Short Physical Performance Battery (SPPB)
```

Store both the total and components when actually performed:

```text
SPPB_total_optional
SPPB_balance_component_optional
SPPB_4m_gait_speed_or_time_optional
SPPB_5_chair_rises_time_optional
```

Reason for preference:

```text
short
multidomain
captures standing balance + gait speed + chair-rise performance
useful for longitudinal functional change
```

Hard rule:

```text
SPPB score or component threshold
→ quantifies function / supports monitoring
→ does NOT autonomously diagnose frailty, falls risk or neurological disease
```

Other optional measures may be carried if already performed:

```text
Timed Up and Go
5-times Sit-to-Stand
30-second Chair Stand
4-m gait speed
Berg Balance Scale
Mini-BESTest
6-minute walk or other endurance measure
other validated measure
```

No single test is used as a stand-alone falls-risk classifier.

---

# 7. Walking-aid assessment / training

This is a direct clinician referral direction in the product-owner workflow.

Possible device context:

```text
no_device
single_point_stick
crutch_or_crutches
walking_frame
rollator
wheelchair_for_distance_or_mobility
other_device
not_assessed
```

Direct goals/directions:

```text
assess walking-aid suitability
optimize safe device use
train transfers / gait with device
improve mobility confidence
review device need as function changes
```

Hard rules:

```text
device currently used != definitely required long-term
no device currently used != definitely unnecessary
software does not prescribe a device solely from age or test score
```

---

# 8. Fragility-fracture gateway

Shared Fracture may directly route here when clinician-selected goals include:

```text
strength
balance
falls-risk reduction through modifiable physical factors
walking / mobility
functional independence
walking-aid assessment / training
```

Hard rule:

```text
fracture restrictions / weight-bearing / use-loading instructions
→ Shared Fracture remains authoritative
→ this profile never overrides them
```

---

# 9. Core rehabilitation directions

Nothing is globally preselected.

Potential directions when medically appropriate:

```text
progressive resistance / strengthening
functional sit-to-stand / transfer practice
coordination training
balance / stepping / recovery training when relevant
power-oriented functional exercise where safe and appropriate
gait / walking practice
stairs / obstacle / community-mobility practice
walking-aid assessment and training
home exercise programme
activity / sedentary-behaviour reduction plan
graded return to usual daily activity
falls-prevention exercise when indicated
```

The product owner does **not** use generic aerobic/endurance conditioning as a routine direct referral direction in this profile. It may still be used by the treating physiotherapist when clinically appropriate, but it is not a generator default.

Programme principles:

```text
individualized
progressive
tailored to goals, ability, comorbidity and safety
regularly reviewed
active / function-oriented
```

No universal set/repetition/intensity prescription is frozen.

---

# 10. Goals

```text
improve generalized / lower-limb strength
improve transfer independence
improve coordination
improve balance / postural control when impaired
reduce falls risk through modifiable physical factors
increase walking tolerance
improve stair negotiation
increase confidence with mobility
maintain or regain ADL-related physical independence
optimize safe walking-aid use
```

Falls wording must never guarantee prevention.

---

# 11. Safety / reassessment semantics

High-priority reassessment contexts:

```text
new focal neurological deficit
acute or rapidly progressive gait change
new inability to stand/walk without explained stable cause
syncope / presyncope / unexplained loss of consciousness
new chest pain / unstable cardiopulmonary symptoms
marked unexplained breathlessness / oxygenation concern
acute vestibular syndrome / severe new vertigo with neurological concern
new fracture / unresolved fracture restrictions
acute painful swelling / DVT or vascular concern
acute infection/systemic deterioration
new delirium/confusion or marked acute cognitive change
progressive unexplained coordination loss
```

Other clarification/coordination contexts:

```text
recurrent unexplained falls
orthostatic symptoms
significant medication-related dizziness/sedation concern
vision/hearing impairment affecting mobility
unsafe footwear / foot pathology
progressive functional loss without adequate diagnosis
```

Home-hazard assessment is **not** exposed as a routine coordination option in this Cyprus workflow because there is no practical local pathway identified by the product owner.

No reassuring negative statement is generated from missing assessment.

---

# 12. Adjunct policy

This profile is intentionally active/function centered.

Excluded as generic deconditioning/balance/gait treatments:

```text
acupuncture
dry needling
ESWT
therapeutic ultrasound
passive manual therapy as a substitute for active rehabilitation
```

Manual therapy may still belong to a separate regional musculoskeletal indication when appropriate.

---

# 13. Deterministic consistency rules

```text
one fall + no recurrent-falls history
→ do not label recurrent falls

fear of falling
→ carry separately from objective balance impairment

SPPB/TUG/gait-speed/5xSTS result
→ carry measurement; do not autonomously create frailty diagnosis

frailty not established
→ do not output formal frailty diagnosis

poor coordination + no established diagnosis
→ do not infer neurological disorder

new unexplained gait disorder
→ do not route as generic deconditioning without disposition

loss of consciousness with fall
→ no routine falls-exercise-only wording without medical context

walking aid selected
→ do not imply permanent requirement

fracture context + restrictions unresolved
→ Shared Fracture restrictions override this profile

material safety concern + no disposition
→ no routine reassuring referral wording
```

---

# 14. Evidence-governance boundary

Stable frozen direction:

```text
progressive individualized active rehabilitation is core
falls-focused exercise should address relevant balance, coordination, strength and power deficits
multicomponent functional rehabilitation is preferable to passive modality-led treatment
performance tests quantify function/change but are not autonomous diagnoses
falls management is multifactorial when non-physical risk factors are present
SPPB is preferred optional local functional battery, not a diagnostic gate
```

Evidence anchors reviewed during candidate/freeze work include:

- NICE NG249 Falls: assessment and prevention in older people and people 50+ at higher risk, 2025;
- WHO Integrated Care for Older People (ICOPE), second edition / locomotor-capacity material, 2025;
- recent systematic reviews of SPPB/TUG/gait-speed and other physical-performance measures;
- systematic-review evidence showing no single gait/balance/functional test should be used alone to predict falls with high certainty.

Evidence-sensitive details to refresh before runtime implementation:

```text
local SPPB workflow / data-entry UX
validated interpretation ranges by population
walking-aid assessment workflow
frailty instrument interoperability if later required
```

---

# 15. Freeze decisions — product owner 2026-08-27

- generalized deconditioning is directly selectable but not a frequent referral;
- isolated balance/falls-risk and gait-only referrals are not routine top-level pathways;
- post-hospital/post-illness deconditioning is not a routine route;
- frailty-associated functional decline is direct only when frailty is already established;
- SPPB is the preferred optional functional battery; no performance test is mandatory;
- falls history is prominent: single/recurrent/injurious/unable-to-rise/loss-of-consciousness states are preserved;
- walking-aid assessment/training is directly requested by the referring clinician;
- home-hazard assessment is not exposed as a routine local coordination option;
- fear of falling is direct and distinct from objective balance deficit;
- Shared Fracture may directly gateway to this profile for strength/balance/falls/independence goals;
- Parkinson disease, stroke, peripheral neuropathy and similar neurological disorders are not routine CU-1 referral pathways in this workflow;
- generic aerobic/endurance conditioning is not a routine generator direction;
- acupuncture, dry needling, ESWT and therapeutic ultrasound are excluded;
- generalized muscular weakness and poor coordination are direct findings.

This file is the frozen Shared Generalized Deconditioning / Balance / Gait clinical/content design for CU-1. Runtime implementation remains unauthorized.
