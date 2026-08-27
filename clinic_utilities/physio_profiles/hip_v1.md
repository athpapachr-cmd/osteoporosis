# Hip Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful hip/groin referral choices while preserving diagnosis-vs-finding separation, intra-articular vs extra-articular distinctions, fracture/stress-fracture and pediatric safety semantics, active rehabilitation, postoperative restrictions and physiotherapist autonomy.
> **Prior frozen regional profiles:** `cervical_v1_1.md`, `lumbar_v1_1.md`, `shoulder_v1_1.md`, `elbow_v1_1.md`, `wrist_hand_v1_1.md`, `knee_v1_1.md`.

---

# 1. Core design contract

```text
PRIMARY CLINICAL PATHWAY
+
ACTUAL FINDINGS / MODIFIERS
+
FUNCTIONAL IMPACT
+
AGE / SKELETAL-MATURITY CONTEXT WHEN RELEVANT
+
SAFETY / STRUCTURAL / POSTOPERATIVE CONTEXT
+
CONFIRMED GOALS
+
CONFIRMED REHABILITATION DIRECTIONS
```

Inherited hard invariants:

```text
suggested != examined
suggested != selected
symptom != diagnosis
pain-limited effort != structural weakness or tendon rupture
special/provocation test != diagnosis
imaging morphology != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

Additional Hip v1 candidate invariants:

```text
lateral hip pain != automatically trochanteric bursitis or gluteal tendinopathy
cam/pincer morphology != femoroacetabular impingement syndrome
FADIR/FABER finding != FAIS or labral tear
MRI labral tear != automatically symptomatic pain generator
snapping phenomenon != symptomatic snapping-hip syndrome
buttock pain != proximal hamstring tendinopathy or deep-gluteal syndrome automatically
groin pain != adductor or iliopsoas diagnosis automatically
painful weight bearing != routine physiotherapy until fracture/stress-fracture concern is appropriately addressed
```

The utility structures a referral and checks consistency. It must not autonomously diagnose hip OA, GTPS/gluteal tendinopathy, FAIS, labral tear, hip instability/dysplasia, proximal hamstring/adductor/iliopsoas tendinopathy, snapping hip, fracture/stress fracture, osteonecrosis or postoperative complication.

---

# 2. Proposed default primary hip pathways

## H1 — Hip osteoarthritis

Structured key:

```text
hip_osteoarthritis
```

Display:

> Οστεοαρθρίτιδα ισχίου

Useful findings/context:

```text
groin/anterior hip pain
lateral/posterior hip pain when clinically attributed
weight-bearing pain
start-up pain / stiffness
walking tolerance
stair limitation
sit-to-stand limitation
shoe/sock or dressing difficulty
car transfer difficulty
hip flexion/internal-rotation/extension restriction
hip strength deficit if assessed
gait deviation if assessed
balance/fall-risk context if relevant
radiographic OA context if available
BMI/weight-management context when clinically relevant
```

Hard rules:

```text
radiographic OA alone != proof that all current symptoms arise from OA
radiographic severity != automatic functional severity
```

Core rehabilitation directions may include:

```text
education / self-management
individualized strengthening
flexibility/ROM work according to actual impairment
endurance/aerobic activity
progressive functional exercise
gait/balance training when relevant
walking-aid strategy when relevant
weight-management support/referral when clinically relevant and clinician-selected
```

Manual therapy/joint mobilization may be offered where mobility/pain impairment makes it appropriate, but it does not replace active rehabilitation.

Open product-owner questions remain for acupuncture and dry needling in hip OA.

## H2 — Greater trochanteric pain syndrome / gluteal tendinopathy

Structured key:

```text
greater_trochanteric_pain_gluteal_tendinopathy
```

Default presentation wording without formal diagnosis:

> Πλάγιος πόνος ισχίου με χαρακτηριστικά greater trochanteric pain syndrome / γλουτιαίας τενοντοπάθειας

Optional clinician assertions:

```text
formal_GT​​PS_diagnosis: yes / no / not_stated
formal_gluteal_tendinopathy_diagnosis: yes / no / not_stated
formal_trochanteric_bursitis_diagnosis: yes / no / not_stated
```

Useful findings/context:

```text
lateral greater-trochanter region pain
pain lying on affected side
pain with walking/stairs/single-leg loading
pain with hip adduction/compressive positions
local palpation tenderness if examined
pain with resisted hip abduction if examined
hip abductor strength/capacity deficit if assessed
pelvic control / single-leg function if assessed
ultrasound/MRI context if available
```

Hard rules:

```text
lateral hip pain != automatically GTPS
trochanteric tenderness alone != isolated bursitis
MRI/ultrasound gluteal tendon change != automatically symptomatic diagnosis
```

Core rehabilitation direction:

```text
education / load and compression management
progressive hip-abductor/gluteal loading
progressive lower-limb strength and function
graded walking/stair/sport exposure
```

Current evidence supports education + exercise as the core/first-line rehabilitation approach.

ESWT may be evidence-sensitive in selected GTPS/gluteal-tendinopathy cases, but whether it appears as a clinician-selectable adjunct is left to the product owner.

## H3 — Hip-related groin pain / femoroacetabular impingement syndrome presentation

Structured key:

```text
hip_related_groin_pain_FAIS_presentation
```

Default wording without formal diagnosis:

> Πόνος ισχίου / βουβωνικής χώρας με μη αρθριτικά ενδαρθρικά χαρακτηριστικά

Optional clinician assertion:

```text
formal_FAIS_diagnosis: yes / no / not_stated
```

If `yes`:

> Σύνδρομο μηροκοτυλιαίας πρόσκρουσης (FAIS)

Useful context:

```text
groin/anterior hip pain
pain with hip flexion/rotation
pain with prolonged sitting
pain with squat/lunge/pivot/sport
hip ROM restriction
hip/trunk/lower-limb strength deficits if assessed
movement-pattern finding if assessed
FADIR/FABER or other provocation finding if actually examined
cam/pincer/mixed morphology if established on imaging
coexisting labral/chondral/dysplasia/instability context
```

Hard rules:

```text
cam or pincer morphology alone != FAIS
positive FADIR/FABER alone != FAIS
hip/groin pain alone != FAIS
```

Formal FAIS should preserve the accepted syndrome concept: relevant symptoms + clinical signs + appropriate imaging/morphology context.

Core rehabilitation may include:

```text
education/activity modification
hip/trunk/lower-limb strengthening
mobility work that does not repeatedly provoke impingement-range symptoms
movement-pattern retraining when relevant
balance/neuromuscular work when relevant
graded return to sport/activity
```

## H4 — Established acetabular labral tear / nonarthritic intra-articular hip pain — conservative rehabilitation

Structured key:

```text
established_acetabular_labral_or_nonarthritic_intraarticular_hip_pain
```

Display when formally established:

> Επιβεβαιωμένη βλάβη/ρήξη επιχείλιου χόνδρου (labrum) / μη αρθριτικός ενδαρθρικός πόνος ισχίου — συντηρητική αποκατάσταση

Possible subtype:

```text
symptomatic_acetabular_labral_tear
other_established_nonarthritic_intraarticular_hip_condition
```

Use only when the clinician carries an established symptomatic diagnosis and nonoperative rehabilitation is appropriate.

Hard rules:

```text
MRI/MRA labral tear != automatically symptomatic pain generator
click/catch/clicking != labral tear diagnosis
provocation test != structural tear diagnosis
```

Core rehabilitation follows impairment-based multimodal nonoperative care: activity modification, hip/trunk/lower-limb strengthening, mobility and movement-pattern work according to findings.

Open product-owner decision: keep H4 separate from H3 or collapse FAIS/labral presentations into one nonarthritic-hip pathway with formal subtypes.

## H5 — Proximal hamstring tendinopathy

Structured key:

```text
proximal_hamstring_tendinopathy
```

Default wording without formal diagnosis:

> Πόνος εγγύς οπίσθιου μηρού / ισχιακής περιοχής με χαρακτηριστικά τενοντοπάθειας εγγύς οπισθίων μηριαίων

Optional clinician assertion:

```text
formal_proximal_hamstring_tendinopathy_diagnosis: yes / no / not_stated
```

Useful context:

```text
ischial/lower-buttock localized pain
pain with prolonged sitting
pain with running/sprinting
pain with hip-flexion loading
hamstring strength/load deficit if assessed
sport/load history
imaging context if available
```

Hard rules:

```text
buttock pain != proximal hamstring tendinopathy
ischial tenderness != definitive tendon diagnosis
imaging tendon change != automatically symptomatic tendinopathy
```

Acute avulsion/major tear concern leaves this pathway.

Core rehabilitation:

```text
education/load management
progressive hamstring tendon loading
progressive kinetic-chain / lumbopelvic strength where relevant
graded sitting/running/sport tolerance
```

ESWT is not proposed as a default because newer comparative evidence does not establish clear superiority over individualized physiotherapy; product-owner workflow may determine whether therapist-proposed ESWT is merely documentable context.

## H6 — Adductor-related groin pain / adductor tendinopathy

Structured key:

```text
adductor_related_groin_pain
```

Default wording without formal diagnosis:

> Πόνος βουβωνικής χώρας σχετιζόμενος με τους προσαγωγούς

Optional clinician assertion:

```text
formal_adductor_tendinopathy_diagnosis: yes / no / not_stated
```

Useful context:

```text
adductor-region/groin pain
adductor tenderness if examined
pain with resisted adduction if examined
running/kicking/change-of-direction load
adductor strength/capacity deficit if assessed
hip/trunk/lower-limb deficits if assessed
multiple-groin-entity context
```

Hard rules:

```text
groin pain != automatically adductor-related
adductor tenderness alone != tendinopathy
multiple causes may coexist
```

The Doha terminology is used as a clinical classification aid for athletic groin pain; it is not treated as an imaging-derived diagnosis.

Core rehabilitation emphasizes progressive adductor/lower-limb/trunk loading and graded sport/function return.

Acute adductor muscle strain/tear routes to the future shared muscle/myotendinous profile rather than being duplicated here.

## H7 — Iliopsoas-related groin pain / internal snapping-hip presentation

Structured key:

```text
iliopsoas_related_groin_pain_internal_snapping_hip
```

Default wording without formal diagnosis:

> Πρόσθιος πόνος ισχίου / βουβωνικής χώρας με χαρακτηριστικά iliopsoas-related presentation

Optional clinician-entered subtypes:

```text
formal_iliopsoas_tendinopathy
symptomatic_internal_snapping_hip
other_established_iliopsoas_disorder
```

Useful context:

```text
anterior/groin pain
pain with resisted hip flexion if examined
pain with hip-flexor loading
snapping/clicking with hip motion
sport/dance/running context
hip-flexor strength/capacity deficit if assessed
```

Hard rules:

```text
anterior/groin pain != automatically iliopsoas pathology
snapping without pain/function loss != snapping-hip syndrome
snapping phenomenon != labral tear automatically
```

Core rehabilitation may include activity/load modification, progressive hip-flexor/hip/trunk/lower-limb strength and movement modification according to findings.

## H8 — Post-traumatic hip pain / stiffness after assessed injury

Structured key:

```text
post_traumatic_hip_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία ισχίου μετά από αξιολογημένη κάκωση

Use only after unresolved fracture, dislocation, femoral-neck stress/occult fracture, major tendon avulsion/rupture and neurovascular concern have been addressed as required.

Required context:

```text
injury date / phase
established structural diagnosis if any
imaging/orthopaedic context
weight-bearing restriction if any
ROM/loading restrictions if any
```

The utility never labels unassessed hip trauma as a `simple strain/sprain`.

## H9 — Postoperative hip rehabilitation — candidate active pathway

Structured key:

```text
postoperative_hip_rehabilitation
```

Display:

> Μετεγχειρητική αποκατάσταση ισχίου

Candidate procedure subtypes:

```text
total_hip_arthroplasty_THA
hip_arthroscopy_FAIS_decompression
acetabular_labral_repair_or_reconstruction
hip_arthroscopy_combined_FAIS_labral
periacetabular_osteotomy_or_other_hip_preservation
hip_abductor_gluteal_tendon_repair
proximal_hamstring_repair
other_hip_operation
```

Required context:

```text
procedure
operation date
surgeon/protocol where available
weight-bearing status
assistive-device status
ROM precautions/restrictions
loading/strengthening restrictions
repair/graft-specific precautions
wound/infection context if relevant
return-to-work/sport target
```

Hard rule:

```text
procedure-specific protocol / surgeon restriction > generic hip rehabilitation default
```

No generic postoperative timeline or universal `hip precautions` are invented.

Post-arthroscopy and tendon-repair rehabilitation must remain procedure/repair specific. Routine supervised PT after uncomplicated THA is not assumed to be universally necessary solely because THA occurred; actual indication, deficits and local/surgeon pathway matter.

Product-owner confirmation is required before H9 becomes frozen as a default pathway.

---

# 3. Candidate secondary / rare / advanced entities

## 3.1 Gluteus medius/minimus partial/full-thickness tear — established conservative pathway

Candidate role:

```text
established_gluteal_abductor_tendon_tear_nonoperative
→ primary or advanced depending product-owner frequency
```

Requires clinician/imaging-established tear and nonoperative decision.

Acute traumatic major abductor failure or unresolved repairable tear concern is not collapsed into routine GTPS.

## 3.2 External snapping hip syndrome

Candidate role:

```text
symptomatic_external_snapping_hip
→ secondary/advanced by default
```

Asymptomatic snapping is not a disorder. Lateral snapping/pain does not automatically establish ITB-related external snapping hip.

## 3.3 Developmental dysplasia / hip instability / microinstability

Established nonarthritic structural context only:

```text
established_acetabular_dysplasia
established_hip_instability_or_microinstability
```

Not inferred from hypermobility, pain or one instability test. Excessive end-range stretching/mobility suggestions should not be generated when instability is the clinical issue.

## 3.4 Osteonecrosis / avascular necrosis of femoral head

Medical/structural context only by default:

```text
known_femoral_head_osteonecrosis
```

The utility does not present rehabilitation as disease-modifying and does not infer osteonecrosis from pain alone.

## 3.5 Inguinal-related / pubic-related athletic groin pain

Candidate role:

```text
inguinal_related_groin_pain
pubic_related_groin_pain
```

Secondary/advanced context by default. True inguinal/femoral hernia or other abdominal/pelvic pathology remains medical/surgical rather than a routine physiotherapy diagnosis.

## 3.6 Deep gluteal / piriformis presentation

Do not duplicate. Route to frozen lumbar/deep-gluteal pathway:

```text
lumbar_v1_1 → deep_gluteal_piriformis_presentation
```

## 3.7 Acute muscle/myotendinous injuries

Do not duplicate. Route to future shared muscle/myotendinous profile:

```text
adductor strain/tear
hip-flexor/iliopsoas strain
rectus femoris strain
hamstring strain
other hip/pelvic muscle injury
```

## 3.8 Fractures / stress fractures

Route to future shared fracture/post-immobilization profile after diagnosis/stability is established:

```text
femoral neck fracture
intertrochanteric/subtrochanteric fracture
acetabular fracture
pelvic/rami fracture
femoral-neck stress fracture
pelvic stress/insufficiency fracture
other hip-region fracture
```

Femoral-neck stress-fracture concern is a high-priority diagnostic/safety gate and not a routine tendinopathy/OA referral.

---

# 4. Pediatric / adolescent hip — candidate navigation group

Candidate UI grouping:

```text
Παιδιά / Έφηβοι — ισχίο / βουβωνική χώρα
```

This would be a navigation/safety layer, not a diagnosis.

Potential routes:

```text
adolescent FAIS / labral presentation → H3/H4 + age/skeletal-maturity context
adductor/iliopsoas-related sports groin pain → H6/H7 + age context
pelvic/apophyseal avulsion fracture → shared fracture profile
acute muscle injury → shared muscle/myotendinous profile
```

Potential established pediatric structural contexts:

```text
Legg-Calve-Perthes disease
established developmental dysplasia / instability
```

High-priority non-routine-PT concerns:

```text
SCFE concern
septic hip concern
acute inability/refusal to bear weight without appropriate assessment
unresolved fracture/apophyseal avulsion
persistent severe night/rest pain or systemic concern
```

SCFE must never be generated as a physiotherapy diagnosis from symptoms; obligate external rotation with hip flexion or a clinically suspicious adolescent presentation requires medical/imaging evaluation.

Whether this pediatric/adolescent hip group appears in the first Hip UI is a product-owner decision based on actual workflow.

---

# 5. Findings — selectable only when actually assessed

## 5.1 Pain / symptom behaviour

```text
groin/anterior hip pain
lateral/trochanteric pain
posterior/buttock pain
ischial pain
adductor-region pain
pubic-region pain
pain with walking
pain with stairs
pain with sit-to-stand
pain with prolonged sitting
pain with lying on side
pain with shoe/sock dressing
pain with hip flexion/rotation
pain with squat/lunge
pain with running
pain with kicking
pain with jumping
pain with pivot/change of direction
pain with sexual activity if clinician-entered and relevant
night/rest pain
```

## 5.2 Mechanical symptoms

```text
clicking
catching
snapping
subjective instability/giving-way
locking/mechanical block
```

Mechanical symptoms remain findings; they do not autonomously establish labral tear, snapping-hip syndrome or instability.

## 5.3 Range of motion

```text
flexion restricted
extension restricted
internal rotation restricted
external rotation restricted
abduction/adduction restricted
painful active ROM
painful passive ROM
```

## 5.4 Strength / performance

```text
hip abductor weakness if assessed
hip extensor weakness if assessed
hip flexor weakness if assessed
adductor weakness if assessed
hamstring weakness if assessed
rotator weakness if assessed
single-leg stance/pelvic-control deficit
sit-to-stand deficit
step-up/down deficit
squat/lunge deficit
balance deficit
running/kicking/change-of-direction deficit if assessed
load intolerance without measured weakness
```

## 5.5 Special/provocation findings

Secondary expander only:

```text
FADIR finding
FABER finding
log-roll finding
Stinchfield/resisted straight-leg-raise finding
hip scour finding
gluteal tendon palpation finding
resisted hip-abduction finding
single-leg-stance lateral-hip pain finding
adductor squeeze/resisted-adduction finding
resisted hip-flexion/iliopsoas finding
proximal-hamstring provocation finding
Trendelenburg sign/gait if assessed
other clinician-entered test
```

Tests remain findings, not diagnoses.

---

# 6. Neurological / neurovascular model

Use when relevant to trauma, postoperative concern or lumbar/deep-gluteal overlap.

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
vascular_status: normal / abnormal / not_assessed
```

```text
not_assessed != normal
```

No global `neurovascularly intact` or `no neurological deficit` wording from missing data.

---

# 7. Safety / reassessment semantics

## 7.1 High-priority structural concerns

```text
acute trauma with unresolved hip/pelvic fracture concern
inability to bear weight after trauma without adequate assessment
possible occult femoral-neck fracture in an at-risk patient
exercise-related groin/hip pain with unresolved femoral-neck stress-fracture concern
acute hip dislocation concern / post-dislocation without appropriate structural assessment
acute major tendon avulsion/rupture concern
new major loss of hip-abductor function after trauma
true mechanical block / loose-body concern
```

## 7.2 Medical / inflammatory / vascular concerns

```text
hot/systemically unwell patient with acute hip pain / septic-joint concern
unexplained fever or systemic illness
rapidly progressive atraumatic pain/inability to bear weight
malignancy/systemic-disease concern
DVT/vascular concern when clinically relevant
```

## 7.3 Pediatric/adolescent concerns

```text
SCFE concern
septic hip concern
acute refusal/inability to bear weight without adequate assessment
physeal/apophyseal fracture concern
persistent severe night/rest pain or systemic concern
```

## 7.4 Postoperative concerns

```text
missing procedure/protocol/restrictions
wound/drainage/infection concern
new disproportionate swelling/pain
DVT/PE concern
new neurovascular deficit
new dislocation/instability concern
unexpected loss of function requiring surgical-team feedback
```

Safety state:

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

Disposition when concern present:

```text
reviewed_and_appropriate_to_proceed
imaging/medical reassessment arranged
orthopaedic/sports-medicine pathway underway
urgent/same-day assessment arranged
routine physiotherapy deferred
other
```

No default reassuring negatives are generated from missing information.

---

# 8. Functional limitations

```text
walking distance/tolerance
stairs
sit-to-stand
prolonged standing
prolonged sitting
shoe/sock dressing
lower-body dressing
car transfer
toilet/bath transfer
sleep / side lying
squat/lunge
running
jumping
kicking
pivot/change of direction
sport-specific activity
gym/resistance training
manual work
carrying loads
school PE / youth sport
ADLs/self-care
patient-priority activity / free text
```

---

# 9. Context-sensitive goals

Nothing is globally preselected.

Candidate goal families:

```text
reduce symptom irritability
restore safe hip ROM where impaired
improve hip/trunk/lower-limb strength
improve gluteal/abductor capacity
improve tendon load capacity
improve gait/walking tolerance
improve stair and transfer function
improve balance/neuromuscular control
improve sitting tolerance
restore dynamic hip/pelvic control
progressive return to running/kicking/pivoting
criterion-based return to sport/work
restore function within surgical/structural restrictions
self-management and load adaptation
age-appropriate return to school PE/sport
```

Condition cautions:

- OA: no structural reversal promise;
- GTPS/tendinopathy: no promise that exercise normalizes imaging;
- FAIS/labral: no claim that morphology/tear disappears with rehabilitation;
- instability/dysplasia: mobility work must not default toward excessive end-range loading;
- postoperative: exact procedure/protocol outranks generic defaults.

---

# 10. Rehabilitation directions / supports / adjuncts

## 10.1 Core active directions

```text
physiotherapy assessment and individualized active rehabilitation
education / self-management
activity/load modification
progressive hip/trunk/lower-limb strengthening
progressive tendon loading where relevant
ROM/mobility exercise where safe
neuromuscular/balance training where relevant
gait retraining where relevant
movement/running retraining where relevant
progressive functional loading
criterion-based return to work/gym/sport
home exercise programme where appropriate
```

## 10.2 Condition-sensitive supports

Possible selections:

```text
walking aid / cane strategy for OA or postoperative context
postoperative assistive device according to protocol
movement/load modification strategies
```

No routine hip brace is proposed in v1; evidence for standalone bracing in nonarthritic hip pain is conflicting/limited.

## 10.3 Optional adjunct expander — candidate

Possible items:

```text
manual therapy / joint mobilization where impairment-specific and appropriate
soft-tissue techniques where appropriate
acupuncture — unresolved product-owner decision, especially hip OA
dry needling — unresolved product-owner decision; 2025 hip-OA CPG supports short-term use for selected myofascial trigger-point presentations
ESWT for GTPS/gluteal tendinopathy — unresolved product-owner decision
ESWT for proximal hamstring tendinopathy — not default; therapist-proposed use may be documentable if desired
```

Active rehabilitation remains core. Adjunct selection never implies clinical necessity.

---

# 11. Shared fracture / muscle boundary

Hip/pelvic fracture and acute muscle injury are not duplicated here.

Shared fracture profile future entries:

```text
femoral neck
intertrochanteric / subtrochanteric
acetabulum
pelvic ring/rami
femoral-neck stress fracture
pelvic stress/insufficiency fracture
adolescent apophyseal avulsion fracture
other hip/pelvic fracture
```

Shared muscle/myotendinous future entries:

```text
adductor strain/tear
hip-flexor/iliopsoas strain
rectus-femoris strain
hamstring strain
other acute hip/pelvic muscle injury
```

Unknown healing/loading context prevents unrestricted rehabilitation wording.

---

# 12. Deterministic consistency rules

```text
H1 + x-ray OA only
→ do not automatically attribute all symptoms to OA

H2 + lateral pain only
→ do not infer GTPS/gluteal tendinopathy

H2 + trochanteric tenderness only
→ do not infer isolated bursitis

H3 + cam/pincer morphology only
→ do not infer FAIS

H3 + FADIR/FABER finding only
→ do not infer FAIS

H4 + MRI/MRA labral tear only
→ do not infer symptomatic labral pain generator

H5 + buttock/ischial pain only
→ do not infer proximal hamstring tendinopathy

H5 + acute avulsion/major tear concern
→ structural reassessment, not routine tendinopathy route

H6 + groin pain only
→ do not infer adductor-related groin pain

H7 + painless snapping
→ do not diagnose snapping-hip syndrome

H8 + unresolved fracture/dislocation/stress-fracture/major avulsion concern
→ safety prompt

H9 postoperative + missing procedure/protocol/restrictions
→ warning

pediatric/adolescent suspicious SCFE presentation
→ medical/imaging assessment; no routine PT diagnosis

exercise-related groin pain + unresolved femoral-neck stress-fracture concern
→ no routine tendon/FAIS referral wording

posterior/buttock pain + deep-gluteal context
→ avoid duplication; route to lumbar/deep-gluteal profile when appropriate

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurovascular component
→ never generate normal wording
```

---

# 13. Evidence-governance boundary

Stable candidate evidence directions:

```text
hip OA → individualized exercise/education core; manual therapy can be impairment-specific
GTPS/gluteal tendinopathy → education + exercise first-line/core
cam/pincer morphology != FAIS without syndrome context
nonarthritic hip pain including FAIS/labral → multimodal impairment-based rehabilitation is reasonable
labral imaging finding != automatically symptomatic diagnosis
proximal hamstring tendinopathy → progressive loading/education core; evidence less mature than gluteal tendinopathy
adductor-related groin pain → use clinically based Doha terminology; multiple groin entities may coexist
snapping hip → symptomatic syndrome only, not asymptomatic snapping
postoperative hip → exact procedure/protocol governed
stress-fracture concern → diagnostic/structural pathway before routine rehabilitation
```

Evidence anchors reviewed for this candidate include:

- APTA/JOSPT `Hip Pain and Mobility Deficits—Hip Osteoarthritis: Revision 2025`;
- NICE OA guidance where framework differences matter;
- 2023 APTA/JOSPT `Hip Pain and Movement Dysfunction Associated With Nonarthritic Hip Joint Pain` CPG;
- Warwick Agreement on FAIS terminology/diagnosis;
- 2024 and 2025 systematic reviews on GTPS/gluteal-tendinopathy exercise and treatments;
- 2024 review of hip tendinopathies including gluteal and proximal hamstring tendinopathy;
- 2025 randomized comparison of individualized physiotherapy vs shockwave therapy for proximal hamstring tendinopathy;
- Doha agreement terminology for groin pain in athletes plus newer reliability/application literature;
- 2023 review of snapping-hip syndrome;
- recent stress-fracture guidance/reviews emphasizing prompt femoral-neck stress-fracture recognition;
- contemporary THA rehabilitation systematic reviews.

Evidence-sensitive details to refresh immediately before CU-2 implementation:

```text
hip-OA exercise/manual-therapy/dry-needling wording
GTPS loading/compression-management and ESWT evidence
FAIS/labral nonoperative selection and escalation thresholds
PHT loading and ESWT evidence
adductor/iliopsoas rehabilitation details
post-arthroscopy procedure-specific restrictions
THA local/surgeon precautions and supervision requirements
pediatric hip routing and specialist restrictions
```

---

# 14. Product-owner decisions required before freeze

1. **Hip OA:** is this a common physiotherapy referral and should it be default H1?
2. **GTPS/gluteal tendinopathy:** do you refer these frequently? Do you still want `trochanteric bursitis` directly selectable as a clinician-entered diagnosis, while keeping GTPS/gluteal tendinopathy as the safer default terminology?
3. **FAIS:** do you see/refer enough to keep H3 as a default primary pathway?
4. **Labral tear:** separate default H4 or collapse under one nonarthritic hip-joint pathway with FAIS/labral subtypes?
5. **Proximal hamstring tendinopathy:** default primary or rare/secondary?
6. **Adductor-related groin pain:** default primary? Do you refer athletic groin/adductor cases frequently?
7. **Iliopsoas / internal snapping hip:** default primary or rare/secondary?
8. **Gluteus medius/minimus tear:** do you see confirmed partial/full-thickness abductor tears often enough for a dedicated conservative pathway, or keep advanced?
9. **Post-op hip:** do you see THA, hip arthroscopy/labral repair, gluteal tendon repair, proximal hamstring repair or hip-preservation surgery often enough to keep H9 active?
10. **Acupuncture for hip OA:** do you refer for it? If yes, keep as evidence-sensitive optional adjunct only.
11. **Dry needling:** unlike Knee v1.1, the 2025 hip-OA CPG now supports short-term use in selected myofascial trigger-point presentations. Do you want it available in Hip v1 or excluded based on your workflow?
12. **ESWT for GTPS/gluteal tendinopathy:** do you use/refer enough to keep it as evidence-sensitive optional adjunct?
13. **ESWT for proximal hamstring tendinopathy:** recommend not default; should therapist-proposed use be documentable only?
14. **Pediatric/adolescent hip:** do you see enough children/adolescents to include a navigation category? If yes, candidate scope would keep SCFE as urgent medical/safety routing, Perthes/dysplasia only when established, and apophyseal avulsion fractures in the shared fracture profile.
15. Any frequent real referral missing — e.g. symptomatic external snapping hip, hip dysplasia/instability, inguinal/pubic-related athletic groin pain, gluteal tendon tear, or something else?

This file remains a **DESIGN CANDIDATE / NOT FROZEN** until these workflow decisions are resolved. Runtime implementation remains unauthorized.
