# Ankle / Foot Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define a clinically useful ankle/foot referral profile matching the product owner's real referral workflow while preserving diagnosis-vs-finding separation, acute-vs-chronic instability semantics, tendon and plantar-heel distinctions, pediatric growth-related conditions, podiatry/orthosis boundaries, fracture/bone-stress and Lisfranc/syndesmosis safety, active rehabilitation and physiotherapist autonomy.
> **Supersedes as active ankle/foot design:** `clinic_utilities/physio_profiles/ankle_foot_v1.md`.
> **Prior frozen regional profiles:** `cervical_v1_1.md`, `lumbar_v1_1.md`, `shoulder_v1_1.md`, `elbow_v1_1.md`, `wrist_hand_v1_1.md`, `knee_v1_1.md`, `hip_v1_1.md`.

---

# 1. Core design contract

```text
PRIMARY CLINICAL PATHWAY
+
ACTUAL FINDINGS / MODIFIERS
+
FUNCTIONAL IMPACT
+
FOOTWEAR / PODIATRY / ORTHOSIS / BRACE CONTEXT WHEN RELEVANT
+
SAFETY / STRUCTURAL / HEALING CONTEXT
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
subjective instability != objective instability
pain-limited effort != tendon rupture or structural weakness
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
brace / taping / orthosis / heel lift != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

Ankle / Foot-specific invariants:

```text
lateral ankle pain after inversion != automatically uncomplicated lateral ankle sprain
recurrent giving-way != automatically chronic ankle instability
positive ligament test != autonomous tear grade
Achilles pain != automatically tendinopathy
Achilles imaging change != automatically symptomatic tendinopathy
midportion Achilles != insertional Achilles protocol automatically
plantar heel pain != automatically plantar fasciitis
heel spur != automatic pain generator
heel pain != automatically plantar fascia pathology
medial ankle/arch pain + flat foot != automatically posterior-tibial tendon dysfunction / PCFD
lateral ankle pain != automatically peroneal tendinopathy
snapping behind fibula != routine sprain/tendinopathy
forefoot pain != automatically metatarsalgia / Morton neuroma / plantar-plate injury
radiographic 1st-MTP OA != automatically symptomatic hallux rigidus
pediatric heel pain != automatically Sever disease
pediatric flexible flatfoot != automatically pathological
accessory navicular on imaging != automatically symptomatic accessory-navicular syndrome
hot swollen neuropathic foot != routine physiotherapy presentation
```

The utility structures a referral and checks consistency. It must not autonomously diagnose fracture, syndesmotic injury, Lisfranc injury, Achilles rupture, tendon tear/subluxation, chronic ankle instability, plantar fasciitis, PCFD, Morton neuroma, plantar-plate tear, tarsal tunnel syndrome, stress fracture/bone-stress injury, Charcot neuroarthropathy, infection or postoperative complication.

---

# 2. Frozen routine primary ankle / foot pathways

## AF1 — Acute / postacute lateral ankle sprain after appropriate structural assessment

Structured key:

```text
lateral_ankle_sprain_rehabilitation
```

Display:

> Διάστρεμμα έξω πλαγίου συνδεσμικού συμπλέγματος ποδοκνημικής — αποκατάσταση

This is a high-visibility routine pathway in the product-owner workflow.

Use only when lateral ankle sprain has been clinically assessed and unresolved fracture, syndesmotic/high-ankle injury, peroneal dislocation/major tear, Achilles rupture and major osteochondral injury concerns have been addressed as required.

Useful context:

```text
injury date / phase
first-time vs recurrent sprain
mechanism if known
weight-bearing ability
swelling / bruising
pain with weight bearing
ankle dorsiflexion ROM
balance / dynamic postural control
walking / stair / running limitation
ATFL/CFL-region tenderness if examined
anterior drawer / talar tilt finding if assessed
external support already used
```

Hard rules:

```text
inversion mechanism + lateral pain != uncomplicated LAS automatically
ligament tenderness/test != autonomous sprain grade
unknown imaging/assessment != proof of no relevant structural injury
```

Core rehabilitation direction:

```text
protection according to severity
progressive weight bearing / gait normalization
protected ROM
progressive ankle / lower-limb strength
balance / proprioceptive / neuromuscular training
progressive functional loading
criterion-based return to work/sport
```

Taping is a directly visible optional support. Brace may be carried when already prescribed or clinically selected, but is not a mandatory generator default. Severe injury may require short immobilization when medically indicated; no universal duration is generated.

## AF2 — Achilles tendinopathy — midportion / insertional

Structured key:

```text
achilles_tendinopathy
```

Direct subtypes:

```text
midportion_achilles_tendinopathy
insertional_achilles_tendinopathy
other_established_achilles_tendinopathy
```

Default wording without formal diagnosis:

> Πόνος Αχιλλείου τένοντα με χαρακτηριστικά τενοντοπάθειας

Optional clinician assertion:

```text
formal_achilles_tendinopathy_diagnosis: yes / no / not_stated
```

Both midportion and insertional Achilles tendinopathy are common enough in this workflow to remain directly selectable under one top-level pathway.

Useful context:

```text
midportion vs insertional pain location
morning / start-up stiffness
pain with walking/running/jumping
training/load change
single-leg heel-raise capacity
plantar-flexor strength/endurance if assessed
ankle dorsiflexion if assessed
local tendon tenderness/thickening if examined
imaging context if available
```

Hard rules:

```text
Achilles-region pain != tendinopathy automatically
imaging tendon thickening/degeneration != automatically symptomatic diagnosis
midportion loading details != insertional loading details automatically
acute pop + bruising + marked plantar-flexion weakness / rupture concern != tendinopathy pathway
```

Core direction:

```text
education / load modification without routine complete rest
progressive tendon loading
plantar-flexor strength/endurance
kinetic-chain / neuromuscular work where relevant
graded walking/running/jumping return
```

No single loading method is frozen as uniquely superior. Eccentric, concentric, heavy-slow or other progressive loading strategies may be individualized. Insertional disease must not automatically receive deep-dorsiflexion/compression loading copied from a midportion protocol.

Heel lift is a directly visible optional support when clinically appropriate, especially for temporary symptom/load modification. It is never mandatory or represented as disease-modifying.

ESWT is directly selectable because the product owner refers selected Achilles patients for it. Evidence labeling must remain explicit:

```text
Achilles ESWT = evidence-conflicted / uncertain optional adjunct
!= routine recommendation
!= replacement for progressive loading
!= proven superior to exercise-based rehabilitation
```

## AF3 — Plantar heel pain / plantar fasciitis presentation

Structured key:

```text
plantar_heel_pain_plantar_fasciitis
```

Default wording without formal diagnosis:

> Πελματιαίος πόνος πτέρνας με χαρακτηριστικά plantar fasciitis

Optional clinician assertion:

```text
formal_plantar_fasciitis_diagnosis: yes / no / not_stated
```

This is an important high-visibility pathway in the product-owner workflow.

Useful context:

```text
medial plantar heel pain
first-step morning pain
pain after rest
pain with prolonged standing/walking
plantar-fascia / medial-calcaneal tenderness if examined
gastrocnemius/soleus flexibility or ankle dorsiflexion if assessed
foot/ankle muscle strength/capacity
footwear/load context
BMI/weight context when clinically relevant
```

Hard rules:

```text
plantar heel pain != plantar fasciitis automatically
heel spur != automatic pain generator
calcaneal tenderness + high-impact load / osteoporosis-risk context may require stress-fracture consideration
burning/neuropathic symptoms != routine plantar-fascia diagnosis
heel-centre pressure pain != plantar fascia pathology automatically
```

Core/support options:

```text
plantar-fascia-specific stretching
gastrocnemius/soleus stretching where relevant
foot/ankle resistance exercise
manual therapy where impairment-specific
taping as short-term adjunct
night splint when clinically selected for persistent first-step morning pain
```

Foot orthosis is not a generic generator default; it may be carried as podiatry/clinician-selected context within multimodal management.

ESWT is directly selectable for selected plantar heel pain/plantar fasciopathy cases:

```text
plantar-heel ESWT = evidence-supported optional adjunct
!= mandatory
!= automatic first-line before active rehabilitation
```

## AF4 — Posterior tibial tendon dysfunction / flexible progressive collapsing foot deformity

Structured key:

```text
posterior_tibial_tendon_dysfunction_progressive_collapsing_foot_deformity
```

Default wording without formal structural diagnosis:

> Έσω πόνος ποδοκνημικής / ποδικής καμάρας με χαρακτηριστικά δυσλειτουργίας οπισθίου κνημιαίου τένοντα

Possible clinician-established subtype:

```text
formal_posterior_tibial_tendinopathy
established_flexible_progressive_collapsing_foot_deformity
other_established_posterior_tibial_tendon_disorder
```

This remains a routine selectable pathway because the product owner refers these patients.

Useful context:

```text
medial ankle/arch pain
acquired arch change if established
hindfoot valgus / forefoot abduction if assessed
single-leg heel-raise performance if assessed
posterior-tibial strength/load deficit
flexible vs rigid deformity context
walking/standing limitation
podiatry / orthosis / AFO context
```

Hard rules:

```text
flat foot alone != PCFD/PTTD diagnosis
medial ankle pain alone != posterior-tibial tendinopathy
rigid/progressive deformity != generic tendon-loading pathway
```

Core conservative direction:

```text
load management
posterior-tibial / foot / calf strengthening according to stage and tolerance
functional lower-limb strengthening
walking/function progression
```

Orthosis/AFO/footwear strategy is **not** a generic physiotherapy generator default in this product-owner workflow. These items may be documented when already prescribed/selected and may trigger podiatry coordination where appropriate.

## AF5 — Peroneal tendon disorder — conservative rehabilitation

Structured key:

```text
peroneal_tendon_disorder_nonoperative
```

Subtypes:

```text
peroneal_tendinopathy
confirmed_partial_peroneal_tendon_tear_nonoperative
other_established_peroneal_tendon_disorder
```

This is a routine selectable pathway of intermediate visibility: neither highly frequent nor rare in the product-owner workflow.

Useful context:

```text
posterolateral/lateral ankle pain
pain with resisted eversion if assessed
peroneal strength/load deficit
running/cutting load
coexisting lateral ankle instability
imaging context if available
snapping/subluxation history
```

Hard rules:

```text
lateral ankle pain != peroneal tendinopathy
pain with eversion != tendon tear diagnosis
snapping/subluxation behind fibula != routine tendinopathy
confirmed partial tear requires established diagnosis and nonoperative decision
```

Acute dislocation/subluxation or major tear concern leaves routine conservative referral and requires structural reassessment.

## AF6 — Mechanical metatarsalgia / forefoot overload presentation

Structured key:

```text
mechanical_metatarsalgia_forefoot_overload
```

Default wording:

> Μηχανικού τύπου μεταταρσαλγία / υπερφόρτιση προσθίου ποδός

This remains a routine selectable pathway because the product owner refers these patients.

Useful context:

```text
plantar metatarsal-head pain
standing/walking/running load
callus/pressure pattern if examined
shoe intolerance
MTP motion / toe-deformity context
fat-pad / plantar-plate concern if established
podiatry / footwear / offloading context
```

Hard rules:

```text
forefoot pain != metatarsalgia automatically
metatarsalgia is a symptom-region label and does not identify exact structural cause
stress fracture / plantar-plate rupture / inflammatory disease / Morton neuroma concern must not be hidden under generic overload wording
```

Footwear modification, metatarsal offloading/pads and orthoses are not generic PT defaults. They may be documented when selected by the clinician/podiatrist or used in a specific offloading plan.

## AF7 — Assessed post-traumatic ankle / foot pain or stiffness

Structured key:

```text
post_traumatic_ankle_foot_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία ποδοκνημικής ή άκρου ποδός μετά από αξιολογημένη κάκωση

Use only after unresolved fracture, Lisfranc/syndesmotic injury, Achilles rupture, major tendon dislocation/tear, osteochondral injury and neurovascular concern have been addressed as required.

Required context:

```text
injury date / phase
established diagnosis if any
imaging/orthopaedic context
weight-bearing status
immobilization/brace status
ROM/loading restrictions
```

Hard rule:

```text
unassessed ankle/foot trauma != simple sprain
```

---

# 3. Pediatric / adolescent ankle-foot navigation group — FROZEN

UI grouping:

```text
Παιδιά / Έφηβοι — ποδοκνημική / άκρος πόδας
```

This is a navigation layer, not a diagnostic umbrella.

## PAF1 — Calcaneal apophysitis / Sever disease

Structured key:

```text
calcaneal_apophysitis_Sever_disease
```

Default wording without formal diagnosis:

> Πόνος πτέρνας σε παιδί/έφηβο με χαρακτηριστικά αποφυσίτιδας πτέρνας (Sever)

Useful context:

```text
age / skeletal-maturity context
sport / running / jumping load
posterior or plantar calcaneal pain
activity-related symptoms
calf flexibility / ankle dorsiflexion if assessed
footwear/load context
```

Hard rules:

```text
pediatric heel pain != Sever automatically
night/rest pain / systemic concern / acute trauma / focal atypical bone tenderness != routine Sever pathway
```

Rehabilitation remains conservative and symptom/load guided; no rigid universal protocol is generated.

Heel lifts may be directly selectable as a temporary symptom/load-management support when clinically appropriate.

## PAF2 — Symptomatic accessory navicular

Structured key:

```text
symptomatic_accessory_navicular
```

Default wording without formal diagnosis:

> Έσω πόνος άκρου ποδός / περιοχή σκαφοειδούς με γνωστό ή ύποπτο συμπτωματικό accessory navicular context

Optional clinician assertion:

```text
formal_symptomatic_accessory_navicular_diagnosis: yes / no / not_stated
```

Hard rules:

```text
accessory navicular on imaging != symptomatic diagnosis automatically
medial foot pain != accessory-navicular syndrome automatically
```

Podiatry/orthotic assessment may be appropriate and may be documented. Orthosis is not auto-generated as mandatory.

## PAF3 — Symptomatic flexible flatfoot

Structured key:

```text
symptomatic_pediatric_flexible_flatfoot
```

Default wording:

> Συμπτωματική εύκαμπτη πλατυποδία παιδιού/εφήβου

Use only when symptoms/function warrant treatment.

Hard rules:

```text
asymptomatic flexible flatfoot != disease requiring treatment
flexible flatfoot != rigid flatfoot
atypical rigidity / neurological signs / progressive deformity != routine flexible-flatfoot pathway
```

Core direction may include functional strengthening, activity modification and gait/foot function work according to findings. Orthotic/podiatry strategy may be documented when selected; it is not a universal default.

---

# 4. Rare / advanced / context entities

## 4.1 Chronic ankle instability / recurrent lateral ankle sprain

```text
chronic_ankle_instability_recurrent_sprain
```

Rare/secondary in the product-owner workflow.

Default wording without formal diagnosis:

> Υποτροπιάζοντα διαστρέμματα / αίσθημα αστάθειας ποδοκνημικής με χαρακτηριστικά chronic ankle instability

Hard rules:

```text
subjective giving-way != objective mechanical instability
recurrent pain != CAI automatically
brace/taping != stand-alone rehabilitation
```

Core rehabilitation emphasizes proprioceptive/neuromuscular exercise, balance/dynamic postural control, strength, ROM where impaired and criterion-based return to sport/function.

## 4.2 Syndesmotic / high-ankle sprain

```text
established_syndesmotic_ankle_injury
```

Very rare/advanced in this workflow. Do not infer from pain location or squeeze/external-rotation testing alone. Unresolved instability or Maisonneuve concern requires structural/specialist context.

## 4.3 Tarsal tunnel / tibial-nerve entrapment

```text
established_tarsal_tunnel_or_tibial_nerve_entrapment
```

Rare neurological pathway/context.

Possible findings if actually assessed:

```text
burning/paresthesia plantar foot
medial ankle/tarsal-tunnel symptoms
sensory deficit
intrinsic weakness if assessed
Tinel-type finding
provocation with prolonged standing/walking
EMG/NCS / ultrasound / imaging context if available
```

Hard rules:

```text
burning/paresthesia alone != tarsal tunnel syndrome
Tinel finding alone != diagnosis
negative/normal single test != reliable exclusion
```

Progressive motor deficit, objective neurological loss or uncertain diagnosis requires reassessment rather than automatic routine rehabilitation.

## 4.4 Heel fat-pad pain / fat-pad syndrome presentation

```text
heel_fat_pad_pain_presentation
```

Rare/secondary differential within plantar heel pain.

Default wording:

> Πόνος κεντρικής περιοχής πτέρνας με χαρακτηριστικά heel fat-pad pain

Hard rules:

```text
central heel pain != fat-pad syndrome automatically
plantar heel pain != plantar fasciitis automatically
focal calcaneal bone tenderness / trauma / bone-stress concern != fat-pad pathway
```

Offloading/footwear/podiatry strategies may be documented if clinically selected.

## 4.5 Morton neuroma / interdigital nerve presentation

```text
established_morton_neuroma
interdigital_neuralgic_forefoot_presentation
```

Seen clinically but only rarely referred for physiotherapy; therefore rare/context rather than routine PT pathway.

Hard rules:

```text
forefoot burning/numbness != Morton neuroma automatically
Mulder-type finding != autonomous diagnosis
```

## 4.6 Plantar-plate / lesser-MTP instability

```text
established_plantar_plate_injury_or_lesser_MTP_instability
```

Very rare/advanced in this workflow. Generic metatarsalgia must not hide an established tear or instability.

## 4.7 Anterior tibial / extensor / FHL tendon disorders

```text
anterior_tibial_tendinopathy
extensor_tendon_disorder
flexor_hallucis_longus_tendon_disorder
```

Rare tendon entries. Major rupture leaves routine tendinopathy rehabilitation and requires structural assessment.

## 4.8 Osteochondral lesion of talus / ankle joint

```text
established_osteochondral_lesion_talus
```

Rare/advanced structural context. Imaging finding alone does not establish symptom causation. Any weight-bearing/loading restriction must be preserved.

## 4.9 Hallux rigidus / 1st-MTP osteoarthritis

```text
known_first_MTP_osteoarthritis_hallux_rigidus_context
```

Context only; not a routine physiotherapy-referral pathway in this workflow.

```text
radiographic 1st-MTP OA != automatic symptomatic pain generator
```

## 4.10 Ankle osteoarthritis

```text
known_ankle_osteoarthritis_context
```

Context only; the product owner does not refer ankle OA routinely for physiotherapy.

## 4.11 Hallux valgus

```text
hallux_valgus_context
```

Medical/podiatry/footwear context only. The utility must not imply physiotherapy reverses structural deformity.

## 4.12 Charcot neuroarthropathy / neuropathic foot

```text
known_Charcot_neuroarthropathy
neuropathic_foot_context
```

This is **not** a routine physiotherapy pathway.

High-priority rule:

```text
active hot / swollen / erythematous neuropathic foot or Charcot concern
→ urgent medical / diabetic-foot / offloading pathway
→ routine strengthening/loading deferred until appropriately managed
```

The user has encountered Charcot clinically, so it remains highly visible as a safety/context entity even though it is uncommon.

## 4.13 Inflammatory / crystal / infected foot context

```text
gout_or_crystal_disease
inflammatory_arthritis
infection_or_wound_context
```

Medical context only when established. Hot swollen joint/foot, systemic illness or nonhealing wound requires medical reassessment.

---

# 5. Postoperative ankle / foot — rare advanced route only

No routine top-level postoperative ankle/foot pathway is included in v1.1.

The product owner only rarely sees postoperative Achilles reconstruction/repair.

Rare advanced postoperative subtype:

```text
postoperative_Achilles_repair_or_reconstruction
```

Required context:

```text
operation date
exact repair / reconstruction
surgeon / protocol
weight-bearing status
boot / cast / brace status
ROM restrictions
loading/strengthening restrictions
wound/infection context
return-to-work/sport target
```

Hard rule:

```text
exact procedure / surgeon protocol > generic Achilles rehabilitation default
```

Other postoperative ankle/foot procedures remain future/advanced and are not exposed as routine menu items.

---

# 6. Direct shared-profile gateways

## 6.1 Shared fracture / post-immobilization profile

Foot/ankle gateway sites include:

```text
lateral / medial / posterior malleolus
bimalleolar / trimalleolar
fibula / Maisonneuve context
talus
calcaneus
navicular
cuboid / cuneiform
5th metatarsal / other metatarsals
phalanges
Lisfranc fracture-dislocation / stable post-treated injury
ankle/foot stress fracture / bone-stress injury
other foot/ankle fracture
```

Unknown healing/stability, weight-bearing, immobilization or loading status prevents unrestricted rehabilitation wording.

## 6.2 Shared muscle / myotendinous profile

```text
gastrocnemius strain
soleus strain
calf myotendinous injury
other acute lower-leg/foot muscle injury
```

## 6.3 Achilles rupture gateway

```text
established_Achilles_rupture
→ shared structural/postoperative restriction architecture
```

Acute rupture concern is never routed as Achilles tendinopathy.

---

# 7. Findings — selectable only when actually assessed

## 7.1 Pain / symptom location

```text
lateral ankle
medial ankle
anterior ankle
posterior ankle / Achilles
plantar heel
central heel / fat-pad region
posterior heel
medial arch
lateral foot
midfoot
1st MTP
lesser metatarsal heads / forefoot
interdigital burning/paresthesia
other foot region
```

## 7.2 Symptom behaviour / function

```text
first-step morning pain
start-up stiffness
weight-bearing pain
walking limitation
stairs
uneven ground
single-leg stance
heel raise
squat
running
jumping
landing
cutting/change of direction
push-off limitation
shoe intolerance
prolonged standing
work limitation
sport limitation
school PE / youth sport limitation
```

## 7.3 Swelling / mechanical / instability findings

```text
swelling
bruising
clicking/catching
snapping
subjective giving-way
locking/mechanical block
recurrent sprain history
```

## 7.4 ROM

```text
dorsiflexion restricted
plantar flexion restricted
inversion restricted
eversion restricted
1st-MTP dorsiflexion restricted
painful active ROM
painful passive ROM
```

## 7.5 Strength / performance

```text
plantar-flexor weakness/endurance deficit
single-leg heel-raise deficit
inversion weakness
posterior-tibial capacity deficit
eversion/peroneal weakness
anterior-tibial weakness
intrinsic-foot weakness/capacity deficit
balance deficit
dynamic postural-control deficit
hop/landing deficit
running/cutting deficit
load intolerance without measured weakness
```

## 7.6 Special/provocation findings — secondary expander only

```text
anterior drawer finding
talar-tilt finding
syndesmosis squeeze finding
external-rotation stress finding
Thompson/Simmonds finding
Achilles palpation / Royal London / arc-sign finding if used
Windlass finding
calcaneal squeeze finding
single-leg heel-raise finding
posterior-tibial tendon palpation/resisted inversion finding
peroneal resisted-eversion finding
peroneal subluxation/snapping finding
Mulder-type forefoot finding
Tinel/tarsal-tunnel finding
other clinician-entered test
```

Tests remain findings, not diagnoses.

---

# 8. Neurological / neurovascular model

Use when clinically relevant:

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
vascular_status: normal / abnormal / not_assessed
```

```text
not_assessed != normal
```

No `neurovascularly intact`, `no neurological deficit` or similar normal wording is generated from missing data.

---

# 9. Safety / reassessment semantics

High-priority structural/medical concerns:

```text
acute trauma with unresolved ankle/foot fracture concern
Ottawa-rule/imaging concern not appropriately resolved when clinically applicable
suspected syndesmotic/high-ankle instability or Maisonneuve injury
Lisfranc injury / plantar midfoot bruising / midfoot instability concern
acute Achilles rupture concern
acute peroneal tendon dislocation/subluxation or major tear concern
stress-fracture / bone-stress injury concern, especially navicular / 5th metatarsal / calcaneus
true locked ankle / major mechanical block
new major neurovascular deficit
disproportionate swelling/pain or compartment concern
hot swollen joint / infection concern
DVT/vascular concern
acute hot swollen neuropathic foot / Charcot concern
nonhealing wound / diabetic-foot concern
rapidly progressive atraumatic pain or inability to bear weight
systemic / inflammatory / malignancy concern
```

Pediatric/adolescent safety concerns:

```text
acute trauma / suspected physeal or avulsion injury
atypical persistent night/rest pain
systemic illness
rigid painful flatfoot / atypical deformity
focal bone tenderness / stress-injury concern
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
podiatry/diabetic-foot pathway arranged
urgent/same-day assessment arranged
routine physiotherapy deferred
other
```

No reassuring negative statement is generated from missing information.

---

# 10. Functional limitations

```text
walking distance/tolerance
stairs
uneven-ground walking
single-leg stance
prolonged standing
push-off
heel raise
squat/lunge
running
jumping
landing
cutting/change of direction
sport-specific activity
gym/resistance training
manual work
shoe tolerance
school PE / youth sport
ADLs/self-care
patient-priority activity / free text
```

---

# 11. Context-sensitive goals

Nothing is globally preselected.

Goal families:

```text
reduce symptom irritability
restore safe ankle/foot ROM where impaired
improve calf / ankle / foot strength and endurance
improve Achilles / tendon load capacity
improve balance / proprioception / dynamic postural control
improve walking / stair / uneven-ground tolerance
restore push-off / heel-raise capacity
restore gait mechanics where relevant
progressive return to running/jumping/cutting
criterion-based return to sport/work
improve footwear tolerance
self-management and load adaptation
age-appropriate return to school PE/sport
restore function within fracture / surgical / structural restrictions
```

Condition cautions:

- Achilles: no promise that loading normalizes imaging;
- insertional Achilles: no default deep-compression loading;
- plantar heel pain: no claim that heel spur is the pain generator;
- PCFD: no promise that exercise alone reverses structural collapse;
- pediatric flexible flatfoot: no treatment claim for asymptomatic physiological variants;
- fracture/rupture/post-op: exact healing/protocol restrictions outrank generic strengthening.

---

# 12. Rehabilitation directions / supports / adjuncts

## 12.1 Core active directions

```text
physiotherapy assessment and individualized active rehabilitation
education / self-management
activity / load modification
progressive ankle / calf / foot / lower-limb strengthening
progressive tendon loading where relevant
ROM/mobility exercise where safe
balance / proprioceptive / neuromuscular training
movement / gait / running retraining where relevant
progressive functional loading
criterion-based return to work/gym/sport
home exercise programme where appropriate
```

## 12.2 Supports — frozen visibility policy

Directly visible selectable supports where clinically relevant:

```text
taping
heel lift
```

Other supports are condition-specific and usually documented as selected/prescribed context rather than generic PT defaults:

```text
ankle brace
foot orthosis
AFO
metatarsal pad / offloading
rocker / footwear modification
night splint
boot / immobilization device
```

The product owner commonly refers to podiatry for many orthotic/footwear/offloading decisions. The referral generator should support podiatry coordination without converting these devices into automatic physiotherapy prescriptions.

## 12.3 Adjuncts

```text
manual therapy / soft tissue → optional where impairment-specific and appropriate
dry needling → optional clinician-selected adjunct, especially where a myofascial/plantar-heel context is appropriate
acupuncture → excluded
```

Dry needling is never preselected and availability/competence may be limited locally.

ESWT policy is diagnosis-specific:

```text
plantar heel pain / plantar fasciopathy
→ optional evidence-supported adjunct

Achilles tendinopathy
→ optional evidence-conflicted adjunct because it is used in the product-owner workflow
→ never presented as routine, necessary or superior to progressive loading
```

Therapeutic ultrasound is not a default evidence-backed treatment for lateral ankle sprain or plantar heel pain.

---

# 13. Deterministic consistency rules

```text
AF1 + inversion/lateral pain only
→ do not infer uncomplicated lateral ankle sprain

AF1 + unresolved syndesmotic / fracture / peroneal-dislocation concern
→ structural reassessment prompt

CAI context + subjective giving-way only
→ do not infer objective mechanical instability

CAI + brace only
→ do not generate stand-alone brace rehabilitation plan

AF2 + Achilles pain/imaging only
→ do not infer symptomatic tendinopathy

AF2 + acute rupture concern
→ leave tendinopathy route

AF2 insertional + generic deep-dorsiflexion loading
→ warning; insertional compression context must be respected

AF3 + plantar heel pain only
→ do not infer plantar fasciitis

AF3 + heel spur only
→ do not attribute pain to spur automatically

heel-centre pressure pain
→ consider fat-pad differential; do not force plantar-fascia diagnosis

AF4 + flat foot only
→ do not infer PCFD/PTTD

AF4 + rigid/progressive deformity
→ no generic tendon-only pathway

AF5 + lateral pain only
→ do not infer peroneal tendinopathy

AF5 + snapping/subluxation behind fibula
→ structural reassessment context

AF6 + forefoot pain only
→ do not infer exact structural cause

AF6 + established plantar-plate instability / Morton / stress fracture
→ do not hide under generic metatarsalgia

PAF1 + pediatric heel pain only
→ do not infer Sever disease

PAF2 + accessory navicular imaging only
→ do not infer symptomatic accessory-navicular syndrome

PAF3 + asymptomatic flexible flatfoot
→ no treatment pathway generated

tarsal-tunnel symptoms + Tinel only
→ do not infer tarsal tunnel syndrome

hot/swollen neuropathic foot / Charcot concern
→ medical/offloading pathway; routine loading deferred

fracture / Lisfranc / bone-stress injury + unknown healing/loading status
→ warning; no unrestricted rehabilitation wording

not_assessed neurovascular component
→ never generate normal wording
```

---

# 14. Evidence-governance boundary

Stable structural decisions frozen in Ankle / Foot v1.1:

```text
acute lateral ankle sprain → high-visibility routine pathway
CAI → rare/secondary but exercise-based rehabilitation core
Achilles → one high-visibility pathway with midportion/insertional subtypes
Achilles loading → progressive loading core; no single loading method frozen as uniquely superior
plantar heel pain → high-visibility routine pathway
heel spur != automatic pain generator
posterior tibial / flexible PCFD → routine pathway with podiatry/orthosis context
peroneal tendon disorder → intermediate-visibility routine pathway
mechanical metatarsalgia → routine pathway
hallux rigidus / ankle OA → context only
Morton / plantar plate / anterior tibial / OLT / syndesmosis → rare/advanced
pediatric navigation → Sever + symptomatic accessory navicular + symptomatic flexible flatfoot
asymptomatic pediatric flexible flatfoot != treatment pathway
Charcot → visible safety/medical context, never routine exercise pathway
orthoses/AFO/offloading devices → not generic PT defaults; often podiatry-context
acupuncture → excluded
dry needling → optional
ESWT plantar heel → evidence-supported optional adjunct
ESWT Achilles → evidence-conflicted optional adjunct only
```

Evidence anchors reviewed for this freeze include:

- JOSPT/APTA `Lateral Ankle Ligament Sprains: Revision 2021` CPG;
- JOSPT/APTA `Achilles Pain, Stiffness, and Muscle Power Deficits: Midportion Achilles Tendinopathy: Revision 2024` CPG;
- JOSPT/APTA `Heel Pain—Plantar Fasciitis: Revision 2023` CPG;
- 2024 systematic review/meta-analysis of ESWT for plantar fasciopathy;
- 2026 systematic review/meta-analysis of shockwave therapy for midportion and insertional Achilles tendinopathy;
- recent reviews/guidance on progressive collapsing foot deformity/posterior tibial tendon dysfunction;
- 2024–2026 literature on pediatric flexible flatfoot and symptomatic accessory navicular;
- IWGDF active Charcot neuro-osteoarthropathy guideline;
- 2026 systematic review documenting diagnostic heterogeneity in tarsal tunnel syndrome.

Evidence-sensitive details to refresh immediately before CU-2 implementation:

```text
Achilles midportion vs insertional loading details
ESWT wording for Achilles and plantar heel pain
plantar-heel dry-needling / taping / heel-lift details
PCFD staging / podiatry / orthosis terminology
pediatric flatfoot / accessory-navicular management wording
Charcot urgent-routing language
shared fracture and shared muscle-profile integration
```

---

# 15. Product-owner decisions incorporated

Product-owner decisions on 2026-08-27:

- lateral ankle sprain is common and remains a high-visibility routine pathway;
- chronic ankle instability/recurrent sprain is rare/secondary;
- Achilles tendinopathy is common and keeps both midportion and insertional subtypes under one pathway;
- plantar heel pain/plantar fasciitis is an important routine pathway;
- posterior tibial tendon/PCFD is referred and remains routine;
- peroneal tendon disorders are intermediate frequency and remain routinely selectable;
- hallux rigidus/1st-MTP OA is not routinely referred and is context only;
- mechanical metatarsalgia is referred and remains routine;
- Morton neuroma is seen but only rarely referred for physiotherapy and is therefore rare/context;
- postoperative ankle/foot referrals are rare, with Achilles reconstruction/repair the main occasional example; postoperative care is therefore advanced rather than routine;
- ankle OA is not routinely referred and is context only;
- syndesmotic/high-ankle sprain is very rare/advanced;
- children/adolescents with Sever disease, symptomatic accessory navicular and symptomatic flexible flatfoot are encountered and therefore receive a pediatric navigation group;
- dry needling is allowed as optional despite limited local availability;
- acupuncture is excluded;
- ESWT is used/referred for both plantar heel pain and Achilles, but evidence labels remain diagnosis-specific;
- taping and heel lifts are directly visible supports;
- braces, orthoses, AFOs, metatarsal pads and footwear/offloading modifications are not generic defaults and many such decisions route to or coordinate with podiatry;
- tarsal tunnel is a rare neurological pathway;
- heel fat-pad pain is a rare/secondary plantar-heel differential;
- plantar-plate injury is very rare/advanced;
- anterior tibial tendon disorder is rare;
- Charcot has been encountered and remains a high-visibility medical/offloading safety context rather than a physiotherapy pathway.

This file is the frozen Ankle / Foot clinical/content design for CU-1. Runtime implementation remains unauthorized.
