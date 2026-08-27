# Hip / Groin Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define a compact hip/groin referral profile that matches the product owner's actual referral workflow while preserving diagnosis-vs-finding separation, intra-articular vs extra-articular distinctions, fracture/stress-fracture and tendon-avulsion safety semantics, active rehabilitation and physiotherapist autonomy.
> **Supersedes as active hip design:** `clinic_utilities/physio_profiles/hip_v1.md`.
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
STRUCTURAL / SAFETY CONTEXT
+
CONFIRMED GOALS
+
CONFIRMED REHABILITATION DIRECTIONS
```

Inherited hard invariants:

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
pain-limited effort != structural weakness or tendon rupture
special/provocation test != diagnosis
imaging morphology != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

Hip-specific invariants:

```text
lateral hip pain != automatically GTPS, trochanteric bursitis or gluteal tendinopathy
cam/pincer morphology != femoroacetabular impingement syndrome
FADIR/FABER finding != FAIS or labral tear
MRI/MRA labral tear != automatically symptomatic pain generator
snapping phenomenon != symptomatic snapping-hip syndrome
buttock pain != proximal hamstring tendinopathy or deep-gluteal syndrome automatically
groin pain != adductor or iliopsoas diagnosis automatically
painful weight bearing != routine physiotherapy until fracture/stress-fracture concern is appropriately addressed
proximal rectus-femoris injury != ASIS avulsion
AIIS avulsion is anatomically associated with the rectus-femoris origin; ASIS avulsion is classically associated with sartorius-related traction
```

The utility structures a referral and checks consistency. It must not autonomously diagnose GTPS/gluteal tendinopathy, trochanteric bursitis, FAIS, labral tear, adductor-related groin pain, proximal hamstring tendinopathy, iliopsoas disorder, snapping hip, tendon rupture, fracture/stress fracture, apophyseal avulsion, osteonecrosis or postoperative complication.

---

# 2. Frozen routine primary hip / groin pathways

## H1 — Lateral hip / greater-trochanteric pain pathway

Structured key:

```text
greater_trochanteric_lateral_hip_pain_pathway
```

Default presentation wording when no formal diagnosis is asserted:

> Πλάγιος πόνος ισχίου / περιοχή μείζονος τροχαντήρα με χαρακτηριστικά greater trochanteric pain syndrome

Direct clinician-entered formal subtypes:

```text
formal_GTPS_diagnosis
formal_gluteal_tendinopathy_diagnosis
formal_trochanteric_bursitis_diagnosis
other_established_greater_trochanteric_disorder
```

If `formal_trochanteric_bursitis_diagnosis` is selected, the generated diagnosis may faithfully state:

> Τροχαντηρίτιδα / θυλακίτιδα μείζονος τροχαντήρα

This direct subtype is retained because the product owner encounters clinically diagnosed trochanteric bursitis, while the safer unconfirmed default remains a lateral-hip/GTPS presentation.

Useful findings/context:

```text
lateral greater-trochanter region pain
pain lying on affected side
pain with walking/stairs/single-leg loading
pain with hip adduction/compressive positions
local palpation tenderness if examined
pain with resisted hip abduction if examined
hip-abductor strength/capacity deficit if assessed
pelvic-control / single-leg-function deficit if assessed
ultrasound/MRI context if available
```

Hard rules:

```text
lateral hip pain != automatically GTPS
local trochanteric tenderness alone != isolated bursitis
MRI/ultrasound gluteal tendon change != automatically symptomatic diagnosis
```

Core rehabilitation direction:

```text
education / load and compression management
progressive hip-abductor / gluteal loading
progressive lower-limb strength and function
graded walking / stair / sport exposure
```

Exercise-based rehabilitation remains the core. ESWT is **not** a generator-recommended clinician adjunct in Hip v1.1. If the treating physiotherapist proposes or uses ESWT for GTPS/gluteal tendinopathy, this may be documented as therapist-proposed treatment context without implying necessity or superiority.

## H2 — Nonarthritic intra-articular hip pain — FAIS / symptomatic labral pathway

Structured key:

```text
nonarthritic_intraarticular_hip_pain
```

Default presentation wording without a formal structural diagnosis:

> Πόνος ισχίου / βουβωνικής χώρας με μη αρθριτικά ενδαρθρικά χαρακτηριστικά

Direct clinician-entered formal subtypes:

```text
formal_FAIS_diagnosis
symptomatic_acetabular_labral_tear
other_established_nonarthritic_intraarticular_hip_condition
```

FAIS and symptomatic labral pathology are intentionally combined into one pathway because both are seen but not frequently enough in this workflow to justify separate top-level routes, and contemporary nonarthritic-hip rehabilitation uses a common impairment-based framework.

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
labral/chondral finding if established and clinically considered relevant
dysplasia/instability context if established
```

Hard rules:

```text
cam or pincer morphology alone != FAIS
positive FADIR/FABER alone != FAIS
hip/groin pain alone != FAIS
MRI/MRA labral tear alone != symptomatic labral pain generator
clicking/catching alone != labral tear diagnosis
```

When `formal_FAIS_diagnosis` is selected, the diagnosis is carried as clinician-established and should preserve the syndrome concept rather than being inferred from morphology alone.

Core rehabilitation may include:

```text
education / activity modification
hip/trunk/lower-limb strengthening
mobility work according to actual impairment without repeatedly provoking impingement-range symptoms
movement-pattern retraining when relevant
balance/neuromuscular work when relevant
graded return to sport/activity
```

## H3 — Adductor-related groin pain / adductor tendinopathy

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

This is a high-visibility routine pathway because the product owner sees and refers these cases frequently.

Useful context:

```text
adductor-region / groin pain
adductor tenderness if examined
pain with resisted adduction if examined
running/kicking/change-of-direction load
adductor strength/capacity deficit if assessed
hip/trunk/lower-limb deficits if assessed
sport/training-load context
multiple-groin-entity context
```

Hard rules:

```text
groin pain != automatically adductor-related
adductor tenderness alone != tendinopathy
positive squeeze/resisted-adduction finding != autonomous diagnosis
multiple causes of athletic groin pain may coexist
```

Core rehabilitation:

```text
load modification / education
progressive adductor strengthening/loading
hip/trunk/lower-limb strengthening according to findings
graded running/kicking/change-of-direction progression
criterion-based return to sport/function
```

Acute adductor muscle strain/tear does not duplicate this pathway; it routes to the shared muscle/myotendinous profile.

## H4 — Post-traumatic hip / groin pain or stiffness after assessed injury

Structured key:

```text
post_traumatic_hip_groin_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία ισχίου ή βουβωνικής χώρας μετά από αξιολογημένη κάκωση

Use only when unresolved fracture, dislocation, femoral-neck stress/occult fracture, major tendon avulsion/rupture and neurovascular concern have already been addressed as required.

Required context:

```text
injury date / phase
established structural diagnosis if any
imaging/orthopaedic context
weight-bearing restriction if any
ROM/loading restriction if any
```

Hard rule:

```text
unassessed hip/groin trauma != simple strain/sprain
```

---

# 3. Direct shared-profile gateways — visible because they match real workflow

These entries should be directly reachable from the Hip/Groin UI but their clinical logic is owned by shared profiles to avoid duplication.

## 3.1 Proximal rectus femoris / proximal quadriceps tendon injury in athletes

Gateway key:

```text
shared_muscle_profile_proximal_rectus_femoris_tendon_injury
```

Display:

> Κάκωση / ρήξη εγγύς τένοντα ορθού μηριαίου (proximal rectus femoris) σε αθλητή

The product owner sees these more often than many other rare hip entities.

Possible established context:

```text
proximal_rectus_femoris_tendinous_tear
proximal_rectus_femoris_myotendinous_injury
proximal_rectus_femoris_free_tendon_injury
AIIS_related_rectus_femoris_avulsion_context
other_established_proximal_quadriceps_hip_region_injury
```

Hard rules:

```text
anterior hip/thigh pain != proximal rectus femoris tear
pain with kicking != structural tear diagnosis
MRI/ultrasound finding may be carried when clinician-established but is not inferred
acute major weakness, deformity, high-grade tear or avulsion concern requires structural/sports-medicine context
```

Ownership rule:

```text
hip/groin UI entry
→ future shared muscle/myotendinous profile
→ do not duplicate full tendon-repair / acute-muscle logic here
```

Anatomical rule:

```text
proximal rectus femoris / quadriceps origin → AIIS / supra-acetabular region
ASIS avulsion != proximal rectus femoris injury by default
```

## 3.2 Pelvic apophyseal avulsion fracture in children/adolescents

Gateway key:

```text
shared_fracture_profile_pelvic_apophyseal_avulsion
```

Display:

> Αποσπαστικό κάταγμα απόφυσης λεκάνης σε παιδί / έφηβο

This is retained despite no general pediatric-hip navigation category because the product owner sees these injuries, especially ASIS avulsion.

Direct anatomical site options:

```text
ASIS_avulsion
AIIS_avulsion
ischial_tuberosity_avulsion
lesser_trochanter_avulsion
iliac_crest_avulsion
other_pelvic_apophyseal_avulsion
```

Important anatomy:

```text
AIIS avulsion → commonly rectus femoris origin
ASIS avulsion → classically sartorius-related traction
ischial tuberosity → hamstring origin
lesser trochanter → iliopsoas insertion
```

Hard rules:

```text
acute adolescent groin/hip pain after sprint/kick/jump != muscle strain automatically
apophyseal avulsion concern → imaging/structural assessment before routine unrestricted physiotherapy
known fracture + unknown healing/displacement/loading status → warning
```

Ownership rule:

```text
hip/groin UI entry
→ future shared fracture/post-immobilization profile
```

No generic pediatric/adolescent Hip category is included in v1.1.

---

# 4. Rare / advanced / secondary entities

## 4.1 Proximal hamstring tendinopathy

```text
proximal_hamstring_tendinopathy
```

Rare/secondary in this workflow.

Default presentation wording without formal diagnosis:

> Πόνος εγγύς οπίσθιου μηρού / ισχιακής περιοχής με χαρακτηριστικά τενοντοπάθειας εγγύς οπισθίων μηριαίων

Hard rules:

```text
buttock/ischial pain != proximal hamstring tendinopathy
ischial tenderness != definitive tendon diagnosis
imaging tendon change != automatically symptomatic tendinopathy
acute avulsion/major tear concern != routine tendinopathy
```

Core conservative rehabilitation uses education/load management and progressive tendon/kinetic-chain loading.

ESWT is not generator-recommended. Therapist-proposed ESWT may be documented only.

## 4.2 Iliopsoas-related groin pain / symptomatic internal snapping hip

```text
iliopsoas_related_groin_pain_internal_snapping_hip
```

Rare/secondary in this workflow.

Possible clinician-entered subtypes:

```text
formal_iliopsoas_tendinopathy
symptomatic_internal_snapping_hip
other_established_iliopsoas_disorder
```

Hard rules:

```text
anterior/groin pain != automatically iliopsoas pathology
snapping without pain/function loss != snapping-hip syndrome
snapping phenomenon != labral tear automatically
```

## 4.3 Gluteus medius/minimus partial/full-thickness tear — established conservative pathway

```text
established_gluteal_abductor_tendon_tear_nonoperative
```

Very rare/advanced in this workflow.

Requires clinician/imaging-established tear and a conservative/nonoperative decision.

Acute traumatic major abductor failure or unresolved repairable tear concern is not collapsed into H1 GTPS.

## 4.4 External snapping hip syndrome

```text
symptomatic_external_snapping_hip
```

Rare/advanced. Asymptomatic snapping is not a disorder.

## 4.5 Developmental dysplasia / hip instability / microinstability

Established structural context only:

```text
established_acetabular_dysplasia
established_hip_instability_or_microinstability
```

Not inferred from hypermobility, pain or one instability test. Excessive end-range mobility suggestions should not be generated when instability is the clinical issue.

## 4.6 Inguinal-related / pubic-related athletic groin pain

```text
inguinal_related_groin_pain
pubic_related_groin_pain
```

Rare/advanced context. True inguinal/femoral hernia or other abdominal/pelvic pathology remains medical/surgical rather than a routine physiotherapy diagnosis.

## 4.7 Hip osteoarthritis

```text
known_hip_osteoarthritis_context
```

Not a routine physiotherapy-referral pathway in this product-owner workflow because the product owner does not refer hip OA for physiotherapy.

Radiographic OA may still be carried as medical/context information where relevant to another presentation.

```text
radiographic OA != automatic current pain generator
```

## 4.8 Osteonecrosis / avascular necrosis

```text
known_femoral_head_osteonecrosis
```

Medical/structural context only. Rehabilitation is not represented as disease-modifying.

## 4.9 Postoperative hip

No routine postoperative Hip pathway is included in v1.1 because the product owner does not refer these patients.

If future workflow changes, any postoperative route must be procedure/protocol governed and must not invent generic hip precautions or timelines.

## 4.10 Deep gluteal / piriformis presentation

Do not duplicate. Route to frozen lumbar profile:

```text
lumbar_v1_1 → deep_gluteal_piriformis_presentation
```

---

# 5. Shared acute muscle/myotendinous and fracture boundaries

## 5.1 Shared muscle/myotendinous profile

Future shared entries include:

```text
proximal_rectus_femoris_tendon_or_myotendinous_injury
adductor strain/tear
hip-flexor/iliopsoas strain
rectus-femoris muscle strain
hamstring strain
other acute hip/pelvic muscle or tendon injury
```

The Hip/Groin UI exposes the proximal rectus-femoris gateway because it is common enough in the product-owner workflow, but the shared profile owns the reusable acute-injury logic.

## 5.2 Shared fracture / post-immobilization profile

Future shared entries include:

```text
femoral neck fracture
intertrochanteric / subtrochanteric fracture
acetabular fracture
pelvic ring / pubic-rami fracture
femoral-neck stress fracture
pelvic stress / insufficiency fracture
ASIS apophyseal avulsion fracture
AIIS apophyseal avulsion fracture
ischial-tuberosity avulsion fracture
lesser-trochanter avulsion fracture
other pelvic apophyseal avulsion
other hip/pelvic fracture
```

Unknown healing/stability/loading context prevents unrestricted rehabilitation wording.

Femoral-neck stress-fracture concern is a high-priority diagnostic/structural gate and not a routine FAIS/tendon referral.

---

# 6. Findings — selectable only when actually assessed

## 6.1 Pain / symptom behaviour

```text
groin/anterior hip pain
lateral/trochanteric pain
posterior/buttock pain
ischial pain
adductor-region pain
pubic-region pain
proximal anterior-thigh / rectus-femoris-region pain
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
night/rest pain
```

## 6.2 Mechanical symptoms

```text
clicking
catching
snapping
subjective instability/giving-way
locking/mechanical block
```

Mechanical symptoms remain findings; they do not autonomously establish labral tear, snapping-hip syndrome or instability.

## 6.3 Range of motion

```text
flexion restricted
extension restricted
internal rotation restricted
external rotation restricted
abduction/adduction restricted
painful active ROM
painful passive ROM
```

## 6.4 Strength / performance

```text
hip-abductor weakness if assessed
hip-extensor weakness if assessed
hip-flexor weakness if assessed
adductor weakness if assessed
hamstring weakness if assessed
rotator weakness if assessed
rectus-femoris / knee-extension weakness if actually assessed
single-leg stance / pelvic-control deficit
sit-to-stand deficit
step-up/down deficit
squat/lunge deficit
balance deficit
running/kicking/change-of-direction deficit if assessed
load intolerance without measured weakness
```

## 6.5 Special/provocation findings

Secondary expander only:

```text
FADIR finding
FABER finding
log-roll finding
Stinchfield / resisted straight-leg-raise finding
hip scour finding
gluteal-tendon palpation finding
resisted hip-abduction finding
single-leg-stance lateral-hip pain finding
adductor squeeze / resisted-adduction finding
resisted hip-flexion / iliopsoas finding
proximal-hamstring provocation finding
Trendelenburg sign/gait if assessed
other clinician-entered test
```

Tests remain findings, not diagnoses.

---

# 7. Neurological / neurovascular model

Use when relevant to trauma or lumbar/deep-gluteal overlap.

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
vascular_status: normal / abnormal / not_assessed
```

```text
not_assessed != normal
```

No global `neurovascularly intact` or `no neurological deficit` wording is generated from missing data.

---

# 8. Safety / reassessment semantics

## 8.1 High-priority structural concerns

```text
acute trauma with unresolved hip/pelvic fracture concern
inability to bear weight after trauma without adequate assessment
possible occult femoral-neck fracture in an at-risk patient
exercise-related groin/hip pain with unresolved femoral-neck stress-fracture concern
acute hip dislocation concern / post-dislocation without appropriate structural assessment
acute major tendon avulsion/rupture concern
high-grade proximal rectus-femoris tear/avulsion concern
new major loss of hip-abductor function after trauma
true mechanical block / loose-body concern
adolescent pelvic-apophyseal avulsion concern after acute sprint/kick/jump mechanism
```

## 8.2 Medical / inflammatory / vascular concerns

```text
hot/systemically unwell patient with acute hip pain / septic-joint concern
unexplained fever or systemic illness
rapidly progressive atraumatic pain/inability to bear weight
malignancy/systemic-disease concern
DVT/vascular concern when clinically relevant
```

## 8.3 Child/adolescent structural concerns

No routine pediatric Hip category exists, but the safety engine still recognizes:

```text
SCFE concern
septic hip concern
acute refusal/inability to bear weight without adequate assessment
pelvic physeal/apophyseal fracture concern
persistent severe night/rest pain or systemic concern
```

SCFE must never be generated as a physiotherapy diagnosis from symptoms.

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

No reassuring negative statement is generated from missing information.

---

# 9. Functional limitations

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
school PE / youth sport when relevant
ADLs/self-care
patient-priority activity / free text
```

---

# 10. Context-sensitive goals

Nothing is globally preselected.

Goal families:

```text
reduce symptom irritability
restore safe hip ROM where impaired
improve hip/trunk/lower-limb strength
improve gluteal/abductor capacity
improve adductor capacity
improve tendon load capacity
improve gait/walking tolerance
improve stair and transfer function
improve balance/neuromuscular control
improve sitting tolerance
restore dynamic hip/pelvic control
progressive return to running/kicking/pivoting
criterion-based return to sport/work
restore function within structural restrictions
self-management and load adaptation
age-appropriate return to school PE/sport when relevant
```

Condition cautions:

- GTPS/tendinopathy: no promise that exercise normalizes imaging;
- FAIS/labral: no claim that morphology/tear disappears with rehabilitation;
- instability/dysplasia: mobility work must not default toward excessive end-range loading;
- tendon tear/avulsion: structural diagnosis, healing and loading restrictions outrank generic strengthening;
- fracture/apophyseal avulsion: healing/loading context outranks generic exercise progression.

---

# 11. Rehabilitation directions / supports / adjuncts

## 11.1 Core active directions

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

## 11.2 Optional adjunct policy

Frozen selections:

```text
manual therapy / joint mobilization where impairment-specific and appropriate
soft-tissue techniques where appropriate
dry needling → optional clinician-selected adjunct for an appropriate myofascial context
```

Dry needling is never preselected, never a substitute for active rehabilitation and should be performed only by an appropriately trained practitioner. Hip v1.1 does not generalize disease-specific benefit beyond the evidence base used for the selected context.

Explicitly not generator-recommended:

```text
acupuncture
ESWT for GTPS/gluteal tendinopathy
ESWT for proximal hamstring tendinopathy
```

For ESWT, treating-physiotherapist-proposed use may be documented as context, but the generator does not recommend it or claim superiority over exercise-based rehabilitation.

No routine hip brace is proposed.

---

# 12. Deterministic consistency rules

```text
H1 + lateral pain only
→ do not infer GTPS/gluteal tendinopathy/trochanteric bursitis

H1 + trochanteric tenderness only
→ do not infer isolated bursitis

H2 + cam/pincer morphology only
→ do not infer FAIS

H2 + FADIR/FABER finding only
→ do not infer FAIS

H2 + MRI/MRA labral tear only
→ do not infer symptomatic labral pain generator

H3 + groin pain only
→ do not infer adductor-related groin pain

H3 + squeeze/resisted-adduction finding only
→ do not infer exact structural diagnosis

H4 + unresolved fracture/dislocation/stress-fracture/major tendon-avulsion concern
→ safety prompt

proximal rectus-femoris gateway + anterior thigh pain only
→ do not infer tendon tear

proximal rectus-femoris gateway + ASIS avulsion
→ invalid anatomic mapping; ASIS is not a rectus-femoris origin by default

AIIS avulsion context
→ may carry rectus-femoris-origin relationship when established

ASIS avulsion context
→ may carry sartorius-related apophyseal context when established

pediatric/adolescent pelvic avulsion + unknown fracture/healing/loading status
→ warning; no unrestricted rehabilitation wording

exercise-related groin pain + unresolved femoral-neck stress-fracture concern
→ no routine tendon/FAIS referral wording

posterior/buttock pain + deep-gluteal context
→ avoid duplication; route to lumbar/deep-gluteal profile when appropriate

hip OA only
→ context, not routine Hip v1.1 PT pathway in this workflow

postoperative hip
→ no routine Hip v1.1 route; procedure-specific workflow required if later activated

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurovascular component
→ never generate normal wording
```

---

# 13. Evidence-governance boundary

Stable structural decisions frozen in Hip v1.1:

```text
lateral-hip/GTPS pathway → education + exercise / progressive loading core
formal trochanteric bursitis may be carried when clinician-entered; tenderness alone does not establish it
FAIS + symptomatic labral pathology → one nonarthritic intra-articular pathway
cam/pincer morphology != FAIS without syndrome context
labral imaging finding != automatically symptomatic diagnosis
nonarthritic hip pain → multimodal impairment-based rehabilitation
adductor-related groin pain → high-visibility routine pathway
multiple athletic-groin entities may coexist
proximal hamstring tendinopathy → rare/secondary progressive-loading pathway
iliopsoas/internal snapping → rare/secondary
symptomatic snapping != painless snapping
hip OA → context only in this workflow
postoperative hip → not routine pathway in this workflow
acupuncture → excluded
dry needling → optional clinician-selected adjunct
ESWT → therapist-proposed/documentable only, not generator-recommended
proximal rectus-femoris tendon injury → direct hip/groin gateway to shared muscle/myotendinous profile
ASIS/AIIS and other pelvic apophyseal avulsions → direct gateway to shared fracture profile
AIIS ↔ rectus-femoris origin; ASIS ↔ classically sartorius traction
femoral-neck stress-fracture concern → structural diagnostic pathway before routine rehabilitation
```

Evidence anchors reviewed for this freeze include:

- APTA/JOSPT `Hip Pain and Movement Dysfunction Associated With Nonarthritic Hip Joint Pain: Revision 2023`;
- APTA/JOSPT `Hip Pain and Mobility Deficits—Hip Osteoarthritis: Revision 2025` for adjunct evidence boundaries, although hip OA is not a routine referral in this workflow;
- Warwick Agreement on FAIS terminology/diagnosis;
- 2024–2025 systematic reviews/meta-analyses on GTPS/gluteal-tendinopathy exercise and ESWT;
- 2024 review of hip tendinopathies including gluteal and proximal hamstring tendinopathy;
- Doha agreement terminology for groin pain in athletes;
- systematic reviews of proximal rectus-femoris injuries in athletes;
- systematic/scoping reviews of pelvic apophyseal avulsion fractures in adolescent athletes.

Evidence-sensitive details to refresh immediately before CU-2 implementation:

```text
GTPS loading/compression-management and ESWT wording
nonarthritic hip rehabilitation / escalation details
adductor-related groin loading progression
proximal rectus-femoris injury classification / referral thresholds
pelvic apophyseal avulsion healing/loading and specialist criteria
shared fracture and shared muscle-profile integration
```

---

# 14. Product-owner decisions incorporated

Product-owner decisions on 2026-08-27:

- hip OA is not routinely referred and is therefore context only;
- GTPS is not very frequent, but clinically diagnosed trochanteric bursitis is seen and remains directly selectable within the lateral-hip pathway;
- FAIS is seen only several times per year but remains directly selectable;
- FAIS and symptomatic labral pathology are combined into one nonarthritic intra-articular pathway;
- proximal hamstring tendinopathy is rare/secondary;
- adductor-related groin pain is common and high visibility;
- iliopsoas/internal snapping hip is rare/secondary;
- gluteus medius/minimus tears are very rare/advanced;
- postoperative hip is not a routine referral and is excluded from the active menu;
- acupuncture is excluded;
- dry needling is optional;
- ESWT for GTPS is not clinician-referred by the product owner but may be documented when proposed/used by the treating physiotherapist;
- ESWT for proximal hamstring tendinopathy is likewise documentable only, not a generator recommendation;
- no general pediatric/adolescent Hip navigation category is needed;
- proximal rectus-femoris/proximal-quadriceps tendon injuries in athletes are a meaningful real-workflow problem and receive a direct shared-muscle gateway;
- pelvic apophyseal avulsion injuries, especially ASIS avulsion in children/adolescents, are seen and receive a direct shared-fracture gateway;
- AIIS/rectus-femoris and ASIS/sartorius anatomy must remain distinct.

This file is the frozen Hip/Groin clinical/content design for CU-1. Runtime implementation remains unauthorized.
