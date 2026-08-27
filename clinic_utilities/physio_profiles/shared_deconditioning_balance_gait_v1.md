# Shared Generalized Deconditioning / Balance / Gait Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** own reusable rehabilitation semantics for generalized deconditioning, balance/falls-risk impairment and gait/mobility limitation without converting functional findings or screening tests into autonomous diagnoses.
> **Runtime:** NOT AUTHORIZED.

---

# 1. Architectural role

This shared profile is the final currently planned CU-1 clinical/content profile.

Proposed shared route:

```text
functional_deconditioning_balance_gait_rehabilitation
```

Routing model:

```text
SHARED / REGIONAL / FRACTURE CONTEXT
→ functional presentation family
→ actual findings + falls/mobility context + medical restrictions
→ clinician-confirmed goals / rehabilitation directions
→ ShortReferralFormatter / DetailedReferralFormatter
```

Presentation families are intentionally separate while sharing one reusable functional-rehabilitation engine:

```text
D1 generalized_deconditioning_functional_decline
D2 balance_impairment_falls_risk_rehabilitation
D3 gait_mobility_impairment_rehabilitation
D4 post_illness_or_post_hospital_deconditioning
D5 frailty_associated_functional_decline — clinician-established/context only
```

D1 and D4 may later be combined in UI if product-owner workflow makes the distinction unnecessary.

---

# 2. Hard semantic boundaries

```text
deconditioning != frailty automatically
frailty != deconditioning automatically
one fall != recurrent-falls syndrome
fear of falling != objective balance impairment
abnormal gait speed/TUG/5xSTS/SPPB/Berg/Mini-BESTest != autonomous diagnosis
normal single test != absence of clinically relevant mobility limitation
assistive device use != diagnosis
assistive device use != automatically inappropriate
age alone != indication for physiotherapy
not assessed != normal
```

Important exclusion boundary:

```text
new / unexplained acute gait disorder
new focal neurological deficit
syncope / presyncope / unexplained loss of consciousness
acute vestibular syndrome
unstable cardiopulmonary symptoms
new fracture or unresolved fracture restriction
acute painful joint / structural condition requiring regional pathway
active infection/systemic deterioration
unresolved DVT/vascular concern
→ not generic deconditioning/balance/gait wording
→ medical/specialist/regional reassessment as appropriate
```

---

# 3. Presentation families

## D1 — Generalized deconditioning / functional decline

Structured key:

```text
generalized_deconditioning_functional_decline
```

Default wording:

> Γενικευμένη αποδυνάμωση / μείωση φυσικής ικανότητας και λειτουργικότητας

Use for a broad decline in strength, endurance, transfers, walking tolerance or daily physical capacity when no single regional musculoskeletal problem adequately owns the rehabilitation need.

Possible context:

```text
reduced activity / prolonged inactivity
multimorbidity
older-adult functional decline
reduced walking tolerance
reduced exercise tolerance
lower-limb or generalized weakness if assessed
transfer difficulty
stairs difficulty
reduced community mobility
ADL-related physical limitation
recent but resolved illness context
```

Hard rule:

```text
generalized weakness/fatigue without adequate medical context
!= automatic deconditioning diagnosis
```

## D2 — Balance impairment / falls-risk rehabilitation

Structured key:

```text
balance_impairment_falls_risk_rehabilitation
```

Default wording:

> Διαταραχή ισορροπίας / αυξημένος κίνδυνος πτώσεων με ανάγκη λειτουργικής αποκατάστασης

Falls context must preserve actual history:

```text
falls_last_12_months_optional
recurrent_falls_context_optional
injurious_fall_context_optional
unable_to_rise_after_fall_optional
loss_of_consciousness_with_fall_optional
near_falls_optional
fear_or_concern_about_falling_optional
```

Hard rules:

```text
one fall alone != recurrent-falls diagnosis
fall with loss of consciousness != routine balance-only pathway
fall with injury requiring medical treatment may need broader falls assessment
fear of falling != objective balance deficit
```

Core rehabilitation may include progressive, individualized work on:

```text
static/dynamic balance
coordination
strength
power where appropriate
stepping/recovery strategies
transfer practice
walking and turning
stairs / obstacle negotiation
functional task practice
```

Home/environmental hazard assessment is not assumed to be owned by physiotherapy; it may be recommended/recorded as coordinated falls-management context when clinically appropriate.

## D3 — Gait / mobility impairment rehabilitation

Structured key:

```text
gait_mobility_impairment_rehabilitation
```

Default wording:

> Διαταραχή βάδισης / κινητικότητας με λειτουργικό περιορισμό

Possible findings/context:

```text
slow gait
shortened walking tolerance
unsteady gait
reduced step/stride control if assessed
turning difficulty
stairs difficulty
outdoor/community mobility limitation
need for supervision or physical assistance
assistive-device context
reduced confidence with walking
```

Hard rule:

```text
unexplained new gait pattern != generic gait rehabilitation diagnosis
```

Established neurological disease may be carried as context, but this utility does not diagnose Parkinsonism, stroke, neuropathy, myelopathy, cerebellar disease or vestibular disease.

## D4 — Post-illness / post-hospital deconditioning

Structured key:

```text
post_illness_or_post_hospital_deconditioning
```

Default wording:

> Λειτουργική αποδυνάμωση μετά από νόσηση / νοσηλεία

Possible context:

```text
recent hospitalization
recent acute illness now medically stable
reduced bed-to-chair / transfer capacity
loss of walking endurance
loss of lower-limb or generalized strength
reduced ADL independence
new need for walking aid or assistance
```

Required safety principle:

```text
medically stable for exercise / rehabilitation
+ relevant restrictions known
```

The utility does not infer that every post-hospital decline is deconditioning; unresolved acute disease, delirium, hypoxia, haemodynamic instability or new neurological/structural problems require medical reassessment.

## D5 — Frailty-associated functional decline — established/context only

Structured key:

```text
frailty_associated_functional_decline
```

Use only when frailty is clinician-established or explicitly documented in the referral context.

Optional context:

```text
formal_frailty_status_or_tool_optional
multimorbidity_context
sarcopenia_context_if_established
low_physical_activity_context
recurrent_falls_context
ADL_or_IADL_limitation
```

Hard rules:

```text
TUG threshold alone != autonomous frailty diagnosis
gait-speed threshold alone != autonomous frailty diagnosis
SPPB/5xSTS alone != autonomous frailty diagnosis
acute illness + poor performance != stable frailty classification
```

Core direction is individualized multicomponent physical rehabilitation rather than a rigid frailty protocol.

---

# 4. Findings — selectable only when actually assessed

## 4.1 Strength / power / endurance

```text
generalized weakness
lower-limb weakness
upper-limb functional weakness
sit-to-stand difficulty
reduced repeated-chair-rise capacity
reduced stair capacity
reduced walking endurance
reduced exercise tolerance
reduced power / rapid-force capacity if assessed
fatigue with functional activity
```

## 4.2 Balance / postural control

```text
static balance deficit
dynamic balance deficit
single-leg balance deficit if appropriate
turning instability
stepping/recovery deficit
dual-task balance/gait difficulty if assessed
uneven-surface difficulty
obstacle-negotiation difficulty
```

## 4.3 Gait / mobility

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

## 4.4 Functional measures — context, not diagnosis

May be recorded when actually performed:

```text
Timed Up and Go
5-times Sit-to-Stand
30-second Chair Stand
4-m gait speed
Short Physical Performance Battery
Berg Balance Scale
Mini-BESTest
6-minute walk or other walking-endurance measure
other validated local measure
```

Machine fields should store measured value/unit and test name when known.

```text
performance-test result
→ quantifies impairment / monitors change
→ does not autonomously create diagnosis or treatment clearance
```

---

# 5. Falls / medical-context model

Optional structured context:

```text
falls_history_status:
  none_reported
  single_fall
  recurrent_falls
  not_assessed

fall_injury_context
loss_of_consciousness_context
orthostatic_or_dizziness_context
vision_or_hearing_context
medication_review_context
footwear_or_foot_problem_context
neurological_context
cardiovascular_context
continence_context
home_environment_context
osteoporosis_or_fragility_fracture_context
```

These are context fields, not autonomous diagnoses.

A material falls-risk factor requiring medical, medication, visual, cardiovascular, neurological, vestibular, podiatry or occupational-therapy input should support appropriate coordination rather than be silently treated by generic exercise wording.

---

# 6. Assistive-device semantics

Possible context:

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

Possible clinician goals/directions:

```text
assess device suitability if within treating professional scope
optimize safe use
progress walking confidence and independence
review need as function changes
```

Hard rules:

```text
device currently used != device definitely required long-term
no device currently used != device definitely unnecessary
software does not prescribe a device solely from age, TUG or gait speed
```

---

# 7. Core rehabilitation directions — candidate

Nothing is globally preselected.

Potential directions:

```text
progressive resistance / strengthening
functional sit-to-stand / transfer practice
balance training
coordination / stepping / reactive-balance work where appropriate
power-oriented functional exercise where safe and appropriate
gait retraining / walking practice
walking-endurance progression
aerobic/cardiorespiratory conditioning where medically appropriate
stairs / obstacle / community-mobility practice
flexibility / mobility work where an actual limitation exists
home exercise programme
activity / sedentary-behaviour reduction plan
graded return to usual daily activity
falls-prevention exercise programme when indicated
```

Programme principles:

```text
individualized
progressive
tailored to goals, ability, comorbidity and safety
regularly reviewed
active/function-oriented
```

No universal set/repetition/intensity prescription is frozen in CU-1.

---

# 8. Goals — candidate library

```text
improve lower-limb / generalized strength
improve transfer independence
improve balance / postural control
reduce falls risk through modifiable physical factors
improve gait safety and efficiency
increase walking tolerance
increase community mobility
improve stair negotiation
increase exercise tolerance
increase confidence with mobility
maintain or regain ADL-related physical independence
return toward pre-illness/pre-hospital functional level when realistic
reduce sedentary behaviour / increase safe daily activity
optimize safe assistive-device use when relevant
```

Falls outcome wording should avoid guaranteeing fall prevention.

---

# 9. Safety / reassessment semantics

## High-priority medical reassessment contexts

```text
new focal neurological deficit
acute or rapidly progressive gait change
new inability to stand/walk without explained stable cause
syncope / presyncope / unexplained loss of consciousness
new chest pain / unstable cardiopulmonary symptoms
marked unexplained breathlessness or oxygenation concern
acute vestibular syndrome / severe new vertigo with neurological concern
new fracture / unresolved fracture restrictions
acute painful swelling / DVT or vascular concern
acute infection/systemic deterioration
new delirium/confusion or marked acute cognitive change
```

## Other contexts requiring clarification/coordination

```text
recurrent unexplained falls
orthostatic symptoms
significant medication-related dizziness/sedation concern
vision or hearing impairment affecting mobility
unsafe footwear / foot pathology
major home-environment hazard concern
progressive functional loss without adequate diagnosis
```

Safety state:

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

No reassuring negative statement is generated from missing assessment.

---

# 10. Adjunct / support policy

This profile is intentionally exercise/function centered.

Not proposed as generic deconditioning/balance/gait treatments:

```text
acupuncture
dry needling
ESWT
therapeutic ultrasound
passive manual therapy as a substitute for active rehabilitation
```

Manual therapy may still be used under a separate regional musculoskeletal indication when relevant; it is not a core shared deconditioning intervention.

---

# 11. Deterministic consistency rules

```text
one fall + no gait/balance impairment documented
→ do not label recurrent falls / balance disorder

abnormal TUG/gait speed/5xSTS/SPPB
→ carry measurement; do not autonomously create frailty diagnosis

frailty not established
→ do not output formal frailty diagnosis

new unexplained gait disorder
→ do not route as generic deconditioning without disposition

loss of consciousness with fall
→ no routine falls-exercise-only wording without medical context

assistive device selected
→ do not imply permanent requirement

no device selected
→ do not imply device unnecessary

post-hospital deconditioning + medical stability not established
→ warning; no generic exercise progression

fracture context + restrictions unresolved
→ Shared Fracture restrictions override this profile

material safety concern + no disposition
→ no routine reassuring referral wording
```

---

# 12. Evidence-governance boundary

Stable candidate evidence direction:

```text
progressive individualized exercise is core for mobility decline / deconditioning
falls-prevention exercise should address relevant balance, coordination, strength and power deficits
multicomponent programmes are generally preferable to a single passive modality for frailty/functional decline
performance tests quantify impairment and change but are not autonomous diagnoses
falls management is multifactorial when non-physical risk factors are present
```

Evidence anchors reviewed during candidate work include:

- NICE NG249 Falls: assessment and prevention in older people and people 50+ at higher risk, 2025;
- WHO Integrated Care for Older People (ICOPE), second edition / locomotor-capacity materials, 2025;
- recent systematic reviews/meta-analyses of multicomponent exercise in frail older adults;
- recent systematic review/meta-analysis of post-discharge exercise after acute hospitalization in older adults.

Evidence-sensitive details to refresh before CU-2 implementation:

```text
optimal programme dose/duration by population
specific validated thresholds for local test battery
best assistive-device assessment workflow
home hazard / OT coordination workflow
condition-specific cardiopulmonary exercise precautions
```

---

# 13. Product-owner decisions required before freeze

1. Do you actually refer **generalized deconditioning / weakness** often enough for D1 to be high visibility?
2. Do you refer **balance impairment / falls risk** often enough for D2 to be high visibility?
3. Do you refer **gait/mobility impairment** without one dominant regional diagnosis often enough for D3 to be high visibility?
4. Do you see **post-hospital / post-illness deconditioning** enough to keep D4 separate, or should it be a modifier under D1?
5. Do you want **frailty-associated functional decline** as a direct selectable route when frailty is already established, or context only?
6. Which tests do you actually use or commonly receive: TUG, 5xSTS, 30-sec Chair Stand, gait speed, SPPB, Berg, Mini-BESTest, 6MWT?
7. Do you want **falls history** represented prominently as `single / recurrent / injurious / unable to rise / LOC`, as proposed?
8. Do you directly refer for **walking-aid assessment/training**, or is this mostly decided by the treating physiotherapist?
9. Do you want **home hazard assessment** exposed as a coordination option, or is this impractical in your Cyprus workflow?
10. Do you want **fear/concern about falling** as a selectable finding/goal, without treating it as objective balance impairment?
11. For patients after fragility fracture, should this shared profile be directly reachable from Shared Fracture for balance/falls/strength/independence goals? Candidate recommendation: yes.
12. Do you see neurological gait disorders (Parkinson's, stroke, neuropathy etc.) often enough that this utility should expose them as established-context subtypes, or should they remain outside routine CU-1 scope?
13. Do you want aerobic/endurance conditioning directly selectable, provided medical stability is established?
14. Agree that acupuncture, dry needling, ESWT and ultrasound remain outside this shared profile?
15. Any recurring functional/falls/mobility presentation in your practice that is missing?

This file remains **DESIGN CANDIDATE / NOT FROZEN** until product-owner review resolves these workflow decisions. Runtime implementation remains unauthorized.
