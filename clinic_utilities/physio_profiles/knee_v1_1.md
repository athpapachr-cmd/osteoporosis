# Knee Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful knee referral choices matching the product owner's real workflow while preserving diagnosis-vs-finding separation, structural safety, pediatric/adolescent distinctions, active rehabilitation, procedure-specific restrictions and physiotherapist autonomy.
> **Supersedes as active knee design:** `clinic_utilities/physio_profiles/knee_v1.md`.

---

# 1. Core contract

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

Hard invariants:

```text
suggested != examined
suggested != selected
symptom != diagnosis
subjective giving-way != objectively demonstrated instability
pain-limited effort != structural weakness or tendon rupture
special/provocation test != diagnosis
MRI finding != automatically symptomatic diagnosis
not assessed != normal
brace/taping/orthosis != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

The utility structures a referral and checks consistency. It must not autonomously diagnose knee OA, meniscal tear, patellofemoral pain, patellar/quadriceps tendinopathy, ACL/MCL or other ligament rupture, patellar instability, Hoffa/plica pathology, osteochondral injury, pediatric traction apophysitis or postoperative complication.

---

# 2. Frozen default primary knee pathways

## K1 — Knee osteoarthritis

```text
key: knee_osteoarthritis
display: Οστεοαρθρίτιδα γόνατος
```

Optional compartment/context:

```text
medial_tibiofemoral
lateral_tibiofemoral
patellofemoral
multicompartment
not_stated
```

Useful findings/context:

```text
weight-bearing pain
start-up pain / stiffness
walking/stair/sit-to-stand limitation
squat/kneeling limitation
flexion/extension restriction
quadriceps weakness if assessed
hip/lower-limb weakness if assessed
balance/neuromuscular deficit if assessed
varus/valgus alignment context
effusion
radiographic OA context
BMI/weight-management context when clinically relevant
```

Hard rules:

```text
radiographic OA alone != proof that all symptoms arise from OA
radiographic severity != automatic functional severity
```

Core rehabilitation:

```text
education / self-management
individualized strengthening
low-impact aerobic activity
progressive functional exercise
neuromuscular/balance training when relevant
ROM/mobility when restricted
graded walking/activity exposure
weight-management support/referral when relevant and clinician-selected
```

Supports may include brace, cane/walking aid or taping when appropriate; none are globally preselected.

### Knee-OA acupuncture

The product owner refers selected knee-OA patients for acupuncture.

```text
acupuncture_for_knee_OA = optional evidence-sensitive adjunct
```

It is never preselected, never a substitute for exercise/self-management and never presented as guideline-unanimous. Major guideline positions differ, including NICE guidance recommending against acupuncture for OA.

Dry needling is excluded from Knee v1.1.

## K2 — Degenerative meniscal lesion / tear — conservative rehabilitation

```text
key: degenerative_meniscal_lesion_conservative_rehabilitation
display: Εκφυλιστική βλάβη / ρήξη μηνίσκου — συντηρητική αποκατάσταση
```

Context:

```text
medial / lateral / both / not_stated
MRI-established tear if available
OA overlap
joint-line pain/tenderness if examined
clicking/catching
true locking: yes/no/not_stated
recurrent effusion
squat/twist/pivot provocation
```

Hard rules:

```text
MRI meniscal tear != automatically symptomatic pain generator
joint-line tenderness or McMurray/Thessaly finding != definitive structural diagnosis
clicking/catching != true locked knee
```

Evidence boundary:

- exercise-based physiotherapy is first-line for common degenerative meniscal lesions when no structural surgical indication is present;
- long-term randomized evidence supports exercise-based PT as noninferior to arthroscopic partial meniscectomy for patient-reported function in common degenerative tears;
- a degenerative MRI tear does not automatically imply arthroscopy;
- true locking or unresolved structural concern exits this routine pathway.

Core rehabilitation:

```text
quadriceps/lower-limb strengthening
progressive functional loading
ROM restoration if restricted
neuromuscular/balance control
movement/load modification
progressive squat/stair/walking tolerance
graded return to work/sport
```

## K3 — Acute isolated meniscal injury — assessed nonoperative

```text
key: acute_isolated_meniscal_injury_nonoperative
display: Οξεία τραυματική κάκωση μηνίσκου — συντηρητική αποκατάσταση μετά από αξιολόγηση
```

Required context:

```text
injury date / phase
medial / lateral / not_stated
tear morphology if established
displaced/displacing: yes/no/not_stated
ROM restriction / true locking
repairable-lesion/specialist context
weight-bearing restriction
brace restriction
associated ligament/chondral injury context
```

Safety rule:

```text
acute meniscal injury
+ displaced/displacing tear OR true locked knee/major ROM block OR repairable lesion needing timely decision
→ orthopaedic reassessment / early specialist pathway
→ no routine unrestricted rehabilitation wording
```

## K4 — Patellofemoral pain

```text
key: patellofemoral_pain
default display: Πρόσθιος πόνος γόνατος με επιγονατιδομηριαία χαρακτηριστικά
formal_patellofemoral_pain_diagnosis: yes / no / not_stated
```

If formally established:

> Επιγονατιδομηριαίος πόνος

Context:

```text
peripatellar/retropatellar pain
stairs / squat / running / jumping / prolonged sitting
load-volume change
quadriceps weakness if assessed
hip strength/control deficit if assessed
movement/running pattern if assessed
foot/ankle contribution if assessed
patellar-taping response if actually tested
```

Hard rules:

```text
anterior knee pain != automatically patellofemoral pain
patellar crepitus != chondromalacia diagnosis
patellofemoral cartilage MRI finding != automatically symptomatic pain generator
```

Core:

```text
education
knee-targeted exercise
± hip-targeted exercise according to assessment
load/activity modification
graded return
```

Taping, prefabricated foot orthoses, manual therapy and movement/running retraining are condition-sensitive supports.

Adolescent PFP uses K4 with age/skeletal-maturity context; it is not automatically placed in the growth-related pediatric subgroup.

## K5 — Patellar tendinopathy

```text
key: patellar_tendinopathy
default display: Πρόσθιος πόνος γόνατος / επιγονατιδικού τένοντα με χαρακτηριστικά load-related tendinopathy
formal_patellar_tendinopathy_diagnosis: yes / no / not_stated
```

Context:

```text
inferior-pole / patellar-tendon pain
localized tenderness
jump/hop/run/squat load pain
sport/gym load history
strength/load capacity
return-to-sport target
ultrasound/MRI context
```

Hard rules:

```text
anterior pain + tendon tenderness != automatic patellar tendinopathy
ultrasound/MRI tendon change != automatically symptomatic tendinopathy
```

Core:

```text
load monitoring / education
progressive tendon-loading exercise
progressive quadriceps/lower-limb strength
graded energy-storage and jumping/running return when relevant
```

No single mandatory eccentric/isometric/heavy-slow-resistance protocol is frozen.

ESWT is not a generator default. Therapist-proposed ESWT may be documented, but the generator does not recommend it or claim superiority over exercise.

## K6 — Quadriceps tendinopathy

```text
key: quadriceps_tendinopathy
display: Τενοντοπάθεια τετρακεφάλου
```

This is a default pathway because the product owner sees and refers it.

Hard rules:

```text
suprapatellar pain != automatic quadriceps tendinopathy
imaging tendon change != automatically symptomatic diagnosis
quadriceps tendinopathy != acute quadriceps-tendon rupture
```

Core rehabilitation is progressive load/capacity restoration with quadriceps/lower-limb strengthening and graded return.

Acute traumatic loss of active extension/straight-leg raise or palpable extensor-mechanism defect exits this pathway and triggers structural reassessment.

## K7 — ACL injury / instability — nonoperative or preoperative rehabilitation

```text
key: ACL_injury_instability_rehabilitation
display: Κάκωση / ανεπάρκεια πρόσθιου χιαστού (ACL) — συντηρητική / προεγχειρητική αποκατάσταση
```

Allowed pathway context:

```text
nonoperative_rehabilitation
preoperative_preparation
```

**Post-ACL reconstruction is not K7. It routes exclusively to K13 postoperative knee with subtype `ACL_reconstruction`.**

Required context:

```text
injury date / phase
partial vs complete if established
objective instability if established
operative vs nonoperative decision
brace status
weight-bearing status
ROM restrictions
associated meniscus/chondral injury
sport/work demand
```

Hard rules:

```text
subjective giving-way != objective ACL instability
Lachman/anterior-drawer/pivot-shift finding != autonomous ACL tear diagnosis
single functional test != automatic safe return to sport
time since injury alone != return-to-sport clearance
```

Core:

```text
restore ROM according to restrictions
progressive quadriceps/hamstring/lower-limb strength
neuromuscular/proprioceptive control
running progression when appropriate
plyometric/change-of-direction progression when appropriate
criterion-based functional progression
```

## K8 — MCL injury — nonoperative rehabilitation

```text
key: MCL_injury_rehabilitation
display: Κάκωση έσω πλαγίου συνδέσμου (MCL) — συντηρητική αποκατάσταση
```

**Postoperative MCL repair/reconstruction is not K8. It routes exclusively to K13 postoperative knee with subtype `MCL_repair_or_reconstruction`.**

Required context:

```text
injury date / phase
grade if established
partial/complete if established
isolated vs combined injury
objective valgus instability if established
brace status
weight-bearing status
ROM restrictions
associated ACL/meniscus/other ligament injury
```

Hard rules:

```text
medial knee pain != MCL injury
valgus-stress pain/laxity finding != autonomous tear grade
combined ligament injury != routine isolated-MCL pathway
```

Core may include protected progressive loading, ROM restoration, strength, neuromuscular control and graded return according to stability/restrictions.

## K9 — Patellar instability / dislocation rehabilitation

```text
key: patellar_instability_dislocation_rehabilitation
display: Αστάθεια / εξάρθρημα επιγονατίδας — αποκατάσταση
```

Context:

```text
first-time vs recurrent
traumatic vs low-energy/atraumatic
skeletal maturity when relevant
reduction completed if acute
osteochondral injury assessed if relevant
MPFL context
anatomic recurrence-risk factors if established
brace status
weight-bearing/ROM restrictions
```

Hard rules:

```text
patellar apprehension != autonomous instability diagnosis
first-time dislocation != automatically routine PT without structural assessment
```

Physiotherapy is an important component of operative and nonoperative management, but osteochondral/structural context and recurrence risk remain visible. Bracing is condition-sensitive, not a universally beneficial long-term intervention.

Post-MPFL reconstruction or other stabilization surgery routes to K13.

## K10 — Iliotibial-band syndrome

```text
key: iliotibial_band_syndrome
default display: Πλάγιος πόνος γόνατος με χαρακτηριστικά συνδρόμου λαγονοκνημιαίας ταινίας
formal_ITB_syndrome_diagnosis: yes / no / not_stated
```

This is directly selectable despite lower frequency in the product-owner workflow.

Context:

```text
running/cycling load
lateral femoral-condyle region pain
repetitive flexion-extension provocation
training-volume/surface/change context
hip/lower-limb strength/control if assessed
running biomechanics if assessed
```

Hard rule:

```text
lateral knee pain != automatically ITB syndrome
```

Core: load modification, progressive lower-limb strength/capacity and graded activity/running progression.

## K11 — Pes anserine region pain / established tendinobursitis

```text
key: pes_anserine_region_pain_or_tendinobursitis
default display: Πόνος περιοχής χήνειου ποδός / pes anserine region
formal_pes_anserine_bursitis_or_tendinopathy_diagnosis: yes / no / not_stated
```

This is directly selectable because it is common in the product-owner workflow.

Context:

```text
inferomedial knee pain
localized tenderness
stair/walking load provocation
hamstring/adductor loading context
knee-OA overlap
local swelling if present
```

Hard rules:

```text
pes-region tenderness != automatically bursitis
medial knee pain != automatically pes pathology
hot/erythematous/infectious-appearing swelling != routine PT pathway
```

Core: load modification, progressive hamstring/adductor/lower-limb strength, mobility when relevant and functional progression.

## K12 — Post-traumatic knee pain / stiffness after assessed injury

```text
key: post_traumatic_knee_pain_stiffness
display: Μετατραυματικός πόνος / δυσκαμψία γόνατος μετά από αξιολογημένη κάκωση
```

Use only after unresolved fracture, major ligament instability, displaced meniscal lesion, extensor-mechanism rupture, osteochondral injury or neurovascular concern has been addressed.

Required context:

```text
injury/date or phase
established structural diagnosis if any
imaging/orthopaedic context
immobilization history
current ROM/loading/weight-bearing restrictions
```

The utility never labels an unassessed knee trauma as a `simple sprain`.

## K13 — Postoperative knee rehabilitation

```text
key: postoperative_knee_rehabilitation
display: Μετεγχειρητική αποκατάσταση γόνατος
```

All postoperative knee cases route through K13 rather than through the corresponding nonoperative injury pathway.

In the product-owner workflow, **meniscus repair and partial meniscectomy are the most commonly seen postoperative knee referrals**.

Procedure subtype:

```text
meniscus_repair
partial_meniscectomy
ACL_reconstruction
MCL_repair_or_reconstruction
MPFL_reconstruction_or_patellar_stabilization
total_knee_arthroplasty_TKA
unicompartmental_knee_arthroplasty
PCL_or_multiligament_reconstruction
cartilage_or_osteochondral_procedure
patellar_or_quadriceps_tendon_repair
other_knee_operation
```

Required context:

```text
procedure
operation date
surgeon/protocol where available
weight-bearing status
brace status
ROM restrictions
loading/strengthening restrictions
graft/repair-specific precautions
wound/infection context if relevant
return-to-work/sport target
```

Hard rule:

```text
procedure-specific protocol / surgeon restriction > generic knee rehabilitation default
```

### Meniscus surgery distinction

```text
partial_meniscectomy
→ primarily criterion/milestone-based progression according to clinical recovery

meniscus_repair / reconstruction
→ time + criterion-based progression
→ lesion/repair-specific WB/ROM/loading restrictions preserved
```

No generic postoperative timeline is invented.

### ACL reconstruction boundary

Post-ACLR rehabilitation is exercise-based and criterion-informed, with graft/procedure/surgeon restrictions preserved. Return-to-sport is never generated from elapsed time alone.

### TKA boundary

TKA-specific evidence-supported strategies such as early mobilization, progressive strength/ROM and, when appropriate, early quadriceps NMES may be selected within TKA context. These do not leak into generic OA, meniscus, PFP or ligament pathways.

---

# 3. Pediatric / adolescent knee category — UI grouping, not diagnosis

Frozen navigation group:

```text
Παιδιά / Έφηβοι — γόνατο
```

This is a grouping layer, not a diagnostic umbrella.

## P-K1 — Osgood-Schlatter disease

```text
key: Osgood_Schlatter_disease
display: Νόσος Osgood-Schlatter / αποφυσίτιδα κνημιαίου κυρτώματος
```

Use when clinician-established or carried as the working diagnosis.

Context:

```text
age / skeletal maturity
tibial-tubercle localized pain/tenderness
running/jumping/sport load
growth-spurt context if relevant
quadriceps/hamstring/calf flexibility or strength if assessed
functional limitation
```

Core:

```text
education
activity/load modification rather than automatic total rest
symptom-guided progressive strengthening/loading
mobility/flexibility work when an actual restriction exists
graded return to sport/activity
```

No rigid universal exercise protocol or fixed return-to-sport timeline is generated because treatment evidence remains limited/heterogeneous.

## P-K2 — Sinding-Larsen-Johansson syndrome

```text
key: Sinding_Larsen_Johansson_syndrome
display: Σύνδρομο Sinding-Larsen-Johansson
```

Use when clinician-established or carried as the working diagnosis.

Context:

```text
age / skeletal maturity
inferior-patellar-pole localized pain/tenderness
running/jumping load
growth-related context
functional limitation
imaging context if available
```

Hard rules:

```text
inferior-pole pain in an adolescent != automatically SLJ
SLJ != patellar tendinopathy by default
```

Core is conservative load/activity modification with progressive restoration of mobility, strength and sport tolerance; no rigid universal protocol is generated.

## Pediatric routing rule

```text
adolescent PFP → K4 + pediatric/adolescent context
meniscal injury → K2/K3/K13 as appropriate + pediatric context
ACL injury → K7 if nonoperative/prehab; K13 after ACLR
MCL injury → K8 if nonoperative; K13 after surgery
patellar instability/dislocation → K9; K13 after stabilization surgery
fracture → shared fracture profile + pediatric context
osteochondral/OCD lesion → advanced structural pathway
```

Optional age context:

```text
age_years_optional
skeletal_maturity: immature / mature / not_stated
```

---

# 4. Rare / advanced / secondary entities

## 4.1 PCL / LCL / posterolateral-corner / combined ligament injury

```text
PCL_injury
LCL_injury
posterolateral_corner_injury
combined_multiligament_knee_injury
```

These are not default top-level buttons. Established injuries require exact stability, neurovascular, operative/nonoperative and restriction context. Combined/multiligament injury is never collapsed into routine ACL or MCL rehabilitation.

## 4.2 Distal hamstring insertional pathology

Rare selectable secondary entity because the product owner sees it occasionally.

```text
pes_distal_hamstring_insertional_pathology
semimembranosus_or_medial_hamstring_insertional_pathology
biceps_femoris_distal_insertional_pathology
other_established_distal_hamstring_tendon_pathology
```

Pain location alone does not establish tendon diagnosis. Progressive load-based rehabilitation may be selected when appropriate.

## 4.3 Hoffa fat-pad / synovial plica

Rare clinician-entered context:

```text
established_Hoffa_fat_pad_pain_or_impingement
established_synovial_plica_syndrome
```

Hard rules:

```text
anterior knee pain != Hoffa/plica diagnosis
MRI fat-pad signal or visible plica != automatically symptomatic diagnosis
```

## 4.4 Baker / popliteal cyst

Medical/context only:

```text
known_Baker_or_popliteal_cyst_context
```

Posterior swelling does not autonomously establish Baker cyst and never excludes DVT/vascular pathology.

## 4.5 Prepatellar / infrapatellar bursitis

Medical/context only in this workflow, not a routine physiotherapy primary pathway. Septic bursitis concern remains medical.

## 4.6 Osteochondral/chondral lesion or osteochondritis dissecans

Rare/advanced structural context:

```text
established_chondral_or_osteochondral_lesion
osteochondritis_dissecans
```

Imaging does not automatically establish symptom causality. Unstable lesion/loose body/mechanical block requires specialist context. In children/adolescents, skeletal maturity and lesion stability remain explicit.

## 4.7 Meniscal root tear / complex repair-relevant lesion

Rare/advanced structural pathway rather than routine K2. Requires explicit orthopaedic context.

## 4.8 Gastrocnemius strain / myotendinous injury

The product owner sees gastrocnemius sprain, but it routes to the future shared **muscle / myotendinous injury profile** rather than being duplicated as a knee diagnosis.

## 4.9 Inflammatory / crystal knee context

```text
known_inflammatory_arthritis_knee_involvement
known_gout_or_crystal_disease_context
```

Only when established. Acute hot swollen monoarthritis remains a medical diagnostic issue.

---

# 5. Findings — only when actually assessed

## Symptoms / function

```text
medial/lateral joint-line pain
anterior/peripatellar pain
patellar-tendon pain
quadriceps-tendon/suprapatellar pain
posterior/popliteal pain
pes-anserine-region pain
ITB/lateral-condyle region pain
tibial-tubercle pain
inferior-patellar-pole pain
distal hamstring insertional pain
walking/stairs/sit-to-stand/squat/kneeling pain
running/jumping/landing/pivot pain
prolonged-sitting pain
night pain
```

## Mechanical

```text
clicking
catching
subjective giving-way
true locking / major mechanical ROM block
recurrent instability episode
patellar subluxation/dislocation history
```

`clicking/catching` and `true locked knee` remain distinct.

## Swelling / ROM / performance

```text
effusion if assessed
acute hemarthrosis context
localized bursal swelling
posterior swelling
flexion restriction
extension restriction
extension lag
painful active/passive ROM
fixed flexion contracture
quadriceps/hamstring/hip/calf weakness if assessed
single-leg squat / step-down / sit-to-stand deficit
balance/proprioception deficit
hop/running/landing/change-of-direction deficit if assessed
```

Extension lag after trauma is not treated as benign until extensor-mechanism integrity is appropriately assessed.

## Special/provocation findings

```text
joint-line tenderness
McMurray / Thessaly finding
Lachman / anterior drawer / pivot shift
posterior drawer / posterior sag
valgus / varus stress
patellar apprehension
patellar compression/grind-type finding
other clinician-entered test
```

Tests remain findings, not diagnoses.

---

# 6. Neurological / neurovascular model

When relevant:

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
vascular_status: normal / abnormal / not_assessed
```

Possible specific context:

```text
common-peroneal motor/sensory status
foot-drop concern
pulses/perfusion if actually assessed
```

```text
not_assessed != normal
```

No `neurovascularly intact` wording is generated from missing data.

---

# 7. Safety / reassessment

High-priority structural concerns:

```text
acute trauma with unresolved fracture concern
true locked knee / major mechanical ROM block
acute displaced/displacing meniscal tear concern
acute extensor-mechanism rupture concern
new inability to straight-leg raise after acute injury
acute patellar dislocation not appropriately assessed/reduced
major/multiligament instability
new neurovascular deficit after trauma
large acute hemarthrosis with unresolved structural injury
osteochondral loose-body / unstable lesion concern
```

Medical/vascular concerns:

```text
hot swollen knee / septic arthritis concern
systemic illness with acute knee swelling
wound/drainage/cellulitis or postoperative infection concern
acute calf swelling/tenderness with DVT concern
PE/cardiopulmonary concern
unexplained rapidly progressive swelling
acute crystal/inflammatory monoarthritis not established
```

Pediatric/adolescent concerns:

```text
acute physeal/apophyseal fracture concern
persistent severe night/rest pain or systemic concern
inability to bear weight after trauma without adequate assessment
unstable osteochondral/OCD concern
acute extensor-mechanism injury mimicking traction-apophysitis pain
```

Osgood/SLJ labels must not explain away atypical/high-risk presentations.

Postoperative concerns:

```text
missing procedure/protocol/restrictions
wound/infection concern
new disproportionate swelling/pain
DVT/PE concern
new neurovascular deficit
loss of expected extensor-mechanism function
unexpected progressive ROM loss requiring surgical-team feedback
```

Safety state:

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

When concern present, clinician disposition may be:

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

# 8. Functional limitations / goals

Selectable limitations:

```text
walking distance/tolerance
stairs up/down
sit-to-stand
prolonged standing
squat/kneeling/floor transfer
car transfer/driving
running
jumping/landing
pivot/change of direction
sport/gym
manual work/carrying loads
school PE / youth sport
ADLs/self-care
sleep disturbance
patient-priority activity / free text
```

Goal families:

```text
reduce symptom irritability
restore safe knee ROM
improve quadriceps/lower-limb strength
improve gait/walking/stair function
improve balance/neuromuscular control
improve tendon load capacity
restore dynamic knee/patellar stability
progressive return to running/jumping/pivoting
criterion-based return to sport/work
restore function within surgical/structural restrictions
self-management/load adaptation
age-appropriate return to school PE/sport
```

Nothing is globally preselected.

---

# 9. Rehabilitation directions and supports

Core active directions:

```text
physiotherapy assessment and individualized active rehabilitation
education / self-management
activity/load modification
progressive strengthening
quadriceps/hamstring/calf/hip/lower-limb strengthening where relevant
ROM/mobility where safe
neuromuscular/proprioceptive/balance training where relevant
gait retraining where relevant
movement/running retraining where relevant
progressive tendon loading where relevant
progressive functional loading
criterion-based return to work/gym/sport
age-appropriate graded return to youth sport/school activity
home exercise programme where appropriate
```

Condition-sensitive supports:

```text
knee-OA brace
patellofemoral taping/support
prefabricated foot orthosis for selected PFP
ligament brace according to plan
postoperative brace according to protocol
walking aid / cane
```

Hard rule:

```text
brace/tape/orthosis suggested != automatically required
exact injury/surgical protocol > generic support suggestion
```

Optional adjuncts:

```text
manual therapy / joint mobilization where impairment-specific
soft-tissue techniques where appropriate
taping where relevant
thermal strategy for selected OA symptoms
acupuncture for selected knee OA → optional evidence-sensitive adjunct
```

Excluded as clinician-generated Knee v1.1 adjuncts:

```text
dry needling
routine/default ESWT
```

ESWT may be documented as therapist-proposed in patellar tendinopathy. NMES is procedure/context-specific, especially postoperative TKA, not generic OA.

---

# 10. Shared fracture / post-immobilization boundary

Knee-region fractures route to shared fracture profile:

```text
patella fracture
distal femur fracture
proximal tibia / tibial plateau fracture
proximal fibula fracture
other knee-region fracture
```

Future shared required context:

```text
fracture site
date/phase
treatment
healing/stability status
immobilization/brace status
weight-bearing status
ROM/loading restrictions
orthopaedic instructions
age/skeletal maturity when relevant
```

Unknown healing/loading context prevents unrestricted rehabilitation wording.

---

# 11. Deterministic consistency rules

```text
K1 + x-ray OA only
→ do not automatically attribute all symptoms to OA

K2 + MRI degenerative tear only
→ do not auto-assert symptomatic meniscal pain generator

K2 + true locked knee
→ structural reassessment prompt

K3 + displaced/displacing or repairable-lesion concern
→ specialist prompt before generic rehabilitation

K4 + anterior pain only
→ do not infer PFP

K4 + cartilage/chondromalacia imaging only
→ do not infer symptomatic PF diagnosis

K5/K6 + tendon imaging only
→ do not infer symptomatic tendinopathy

K6 + acute loss of straight-leg raise
→ extensor-mechanism reassessment prompt

K7 + postoperative ACL reconstruction
→ invalid route; use K13 subtype ACL_reconstruction

K7 + subjective giving-way only
→ do not infer ACL rupture/instability

K7 + elapsed time only
→ no return-to-sport clearance

K8 + postoperative MCL repair/reconstruction
→ invalid route; use K13 subtype MCL_repair_or_reconstruction

K8 + medial pain/valgus test only
→ do not infer exact MCL grade

K8 + combined instability
→ leave isolated-MCL routine pathway

K9 + acute first-time dislocation + unresolved osteochondral/structural assessment
→ no routine unrestricted PT wording

K10 + lateral pain only
→ do not infer ITB syndrome

K11 + pes tenderness only
→ do not infer bursitis

K12 + unresolved fracture/locked knee/extensor rupture/major instability
→ safety prompt

K13 + missing procedure/protocol/restrictions
→ warning

meniscectomy and meniscus-repair protocol treated identically
→ warning

TKA-specific NMES outside postoperative/TKA context
→ invalid

posterior swelling + no established Baker cyst
→ do not infer Baker cyst or dismiss DVT

P-K1 tibial-tubercle pain only
→ do not autonomously diagnose Osgood-Schlatter

P-K2 inferior-patellar-pole pain only
→ do not autonomously diagnose SLJ

pediatric structural injury + generic pediatric category only
→ require appropriate structural pathway

hot swollen knee + diagnostic uncertainty
→ medical reassessment

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurovascular component
→ never generate normal wording
```

---

# 12. Generated wording examples

### Knee OA

> Οστεοαρθρίτιδα του [side] γόνατος με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση με έμφαση σε προοδευτική ενδυνάμωση, λειτουργική άσκηση, κινητικότητα/νευρομυϊκό έλεγχο όπου ενδείκνυται και εκπαίδευση αυτοδιαχείρισης. [Brace/taping/acupuncture only if explicitly selected.]

### Degenerative meniscal lesion

> Εκφυλιστική βλάβη / ρήξη [medial/lateral] μηνίσκου του [side] γόνατος, με [selected findings] και περιορισμό σε [selected activities], για συντηρητική αντιμετώπιση. Παρακαλώ για προοδευτική αποκατάσταση δύναμης, κινητικότητας, νευρομυϊκού ελέγχου και ανοχής στη λειτουργική φόρτιση. Δεν υπονοείται χειρουργική ένδειξη από την απεικονιστική ρήξη και μόνο.

### ACL nonoperative/prehab

> Κάκωση / ανεπάρκεια ACL του [side] γόνατος, για [nonoperative/preoperative] αποκατάσταση. Παρακαλώ για προοδευτική αποκατάσταση ROM, δύναμης, νευρομυϊκού ελέγχου και λειτουργικής σταθερότητας σύμφωνα με τους περιορισμούς, με criterion-based progression προς τρέξιμο/άθληση όπου απαιτείται.

### MCL nonoperative

> Κάκωση MCL του [side] γόνατος, [grade/stability if established], για συντηρητική αποκατάσταση. Παρακαλώ για προστατευμένη και προοδευτική κινητοποίηση/φόρτιση, αποκατάσταση δύναμης και νευρομυϊκού ελέγχου σύμφωνα με τη σταθερότητα, το brace και τους καταγεγραμμένους περιορισμούς.

### Meniscus repair postoperative

> Μετεγχειρητική αποκατάσταση μετά από συρραφή μηνίσκου του [side] γόνατος, επέμβαση [date]. Παρακαλώ για φυσιοθεραπεία σύμφωνα με το ακριβές πρωτόκολλο της συρραφής και τους περιορισμούς σε φόρτιση, ROM, brace και ενδυνάμωση, με time- και criterion-based progression.

### Partial meniscectomy postoperative

> Μετεγχειρητική αποκατάσταση μετά από μερική μηνισκεκτομή του [side] γόνατος, επέμβαση [date]. Παρακαλώ για criterion-based αποκατάσταση κινητικότητας, δύναμης, βάδισης και λειτουργικής φόρτισης σύμφωνα με την κλινική πρόοδο και τυχόν ειδικές χειρουργικές οδηγίες.

### Osgood-Schlatter

> Νόσος Osgood-Schlatter του [side] γόνατος σε [age/adolescent context], με [selected findings] και περιορισμό σε [sport/school activity]. Παρακαλώ για εκπαίδευση και εξατομικευμένη συντηρητική αποκατάσταση με προσαρμογή φορτίου, προοδευτική ενδυνάμωση/κινητικότητα όπου ενδείκνυται και σταδιακή επιστροφή σε δραστηριότητα χωρίς άκαμπτο προκαθορισμένο χρονοδιάγραμμα.

### Sinding-Larsen-Johansson

> Σύνδρομο Sinding-Larsen-Johansson του [side] γόνατος σε [age/adolescent context], με [selected findings]. Παρακαλώ για συντηρητική αποκατάσταση με διαχείριση φόρτισης, προοδευτική βελτίωση κινητικότητας/δύναμης και σταδιακή επιστροφή σε αθλητική δραστηριότητα σύμφωνα με την ανοχή και την κλινική εξέλιξη.

---

# 13. Evidence-governance boundary

Frozen structural decisions:

```text
exercise / education = core for knee OA
radiographic OA != automatic symptom generator
acupuncture = optional evidence-sensitive knee-OA adjunct only
dry needling = excluded
degenerative MRI meniscal tear != automatic symptom generator
exercise-based PT = first-line for degenerative meniscal lesions
true locked knee != routine degenerative-meniscus rehabilitation
acute displaced/displacing tear with ROM block = structural/specialist pathway
meniscus repair != partial meniscectomy rehabilitation logic
anterior knee pain != automatic PFP
PFP core = education + knee-targeted ± hip-targeted exercise
patellar/quadriceps tendinopathy core = progressive load-based rehabilitation
routine ESWT = not a generator default
ACL and MCL = separate top-level nonoperative/prehab pathways
all ACL/MCL postoperative care = K13 only
subjective giving-way != objective ligament instability
ligament special test != autonomous tear grade
patellar dislocation requires structural/osteochondral context
ITB/pes-anserine remain diagnosis-sensitive
postoperative rehabilitation = exact procedure/protocol governed
TKA-specific interventions do not leak into generic OA pathways
pediatric/adolescent category = UI grouping, not diagnosis
Osgood-Schlatter and SLJ = distinct growth-related pathways
pediatric structural conditions use standard pathways + age/skeletal-maturity context
possible Baker cyst != exclusion of DVT
prepatellar bursitis = medical/context only
Hoffa/plica = rare clinician-entered context
fractures route to shared fracture/post-immobilization profile
gastrocnemius strain routes to shared muscle/myotendinous profile
```

Evidence anchors reviewed include:

- AAOS knee-OA guidance;
- NICE OA guidance where framework differences matter;
- 2024 AAOS Acute Isolated Meniscal Pathology CPG;
- formal EU-US Meniscus Rehabilitation 2024 consensus, operative and nonoperative parts;
- 5-year ESCAPE randomized trial of exercise PT vs arthroscopic partial meniscectomy for degenerative tears;
- 2024 patellofemoral-pain best-practice guide;
- contemporary progressive-loading evidence for patellar tendinopathy;
- AAOS ACL guidance and Aspetar ACL-reconstruction rehabilitation CPG/protocol;
- ESSKA 2024 formal consensus on first-time patellar dislocation;
- APTA Physical Therapist Management of TKA Revision 2026;
- 2026 Osgood-Schlatter treatment review and prior conservative-treatment systematic review;
- 2024 Sinding-Larsen-Johansson scoping review.

Refresh immediately before CU-2 implementation:

```text
OA brace/taping/manual-therapy/acupuncture wording across frameworks
PFP support-selection evidence
meniscus lesion/repair-specific rehabilitation milestones
ACL progression/return-to-sport language
MCL grade-specific bracing/loading details
patellar-instability brace/return-to-sport details
patellar/quadriceps tendon loading progression
ITB/pes-anserine evidence
TKA 2026 detailed recommendations
procedure-specific postoperative restrictions
Osgood/SLJ exercise/return-to-sport evidence
```

---

# 14. Product-owner decisions incorporated

Product-owner decisions on 2026-08-27:

- postoperative knee is active; meniscal repair and meniscectomy are seen most often;
- degenerative and acute traumatic meniscal pathways remain separate;
- ACL and MCL are separate top-level pathways;
- postoperative ACL/MCL route through K13 only to avoid duplicate primary pathways;
- patellar instability/dislocation is default;
- quadriceps tendinopathy is default;
- ITB syndrome is directly selectable despite lower frequency;
- pes-anserine pathology is default because it is seen frequently;
- acupuncture is optional evidence-sensitive for selected knee OA;
- dry needling is excluded;
- ESWT is not default; therapist-proposed use may be documented in patellar tendinopathy;
- taping, knee braces and foot orthoses are condition-sensitive supports;
- children/adolescents have a dedicated navigation category;
- Osgood-Schlatter and Sinding-Larsen-Johansson are directly selectable growth-related pathways;
- Baker cyst and prepatellar bursitis are medical/context only because they are not routinely referred;
- distal hamstring insertional pathology and Hoffa/plica are rare selectable context;
- gastrocnemius strain belongs in the future shared muscle/myotendinous profile.

This file is the frozen knee clinical/content design for CU-1. Runtime implementation remains unauthorized.