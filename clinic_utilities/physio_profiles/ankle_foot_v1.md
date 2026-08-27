# Ankle / Foot Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful ankle/foot referral choices while preserving diagnosis-vs-finding separation, acute-vs-chronic instability semantics, tendon and plantar-heel distinctions, fracture/bone-stress and Lisfranc/syndesmosis safety, orthosis/brace boundaries, active rehabilitation and physiotherapist autonomy.
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
FOOTWEAR / ORTHOSIS / BRACE CONTEXT WHEN RELEVANT
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
selected != mandatory
symptom != diagnosis
subjective instability != objective instability
pain-limited effort != tendon rupture or structural weakness
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
brace / taping / orthosis != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

Additional Ankle / Foot v1 candidate invariants:

```text
lateral ankle pain after inversion != automatically simple lateral ankle sprain
recurrent giving-way != automatically chronic ankle instability
positive ligament test != autonomous tear grade
Achilles pain != automatically tendinopathy
Achilles imaging change != automatically symptomatic tendinopathy
plantar heel pain != automatically plantar fasciitis
heel spur != automatic pain generator
medial ankle/arch pain + flat foot != automatically posterior-tibial tendon dysfunction / PCFD
lateral ankle pain != automatically peroneal tendinopathy
snapping behind fibula != routine ankle sprain
forefoot pain != automatically metatarsalgia or Morton neuroma
1st-MTP radiographic OA != automatically symptomatic hallux rigidus
pediatric heel pain != automatically Sever disease
```

The utility structures a referral and checks consistency. It must not autonomously diagnose fracture, syndesmotic injury, Lisfranc injury, Achilles rupture, tendon tear/subluxation, chronic ankle instability, plantar fasciitis, PCFD, Morton neuroma, stress fracture/bone-stress injury, Charcot neuroarthropathy, infection or postoperative complication.

---

# 2. Proposed default primary ankle / foot pathways

## AF1 — Acute / postacute lateral ankle sprain after appropriate structural assessment

Structured key:

```text
lateral_ankle_sprain_rehabilitation
```

Default display:

> Διάστρεμμα έξω πλαγίου συνδεσμικού συμπλέγματος ποδοκνημικής — αποκατάσταση

Use when lateral ankle sprain has been clinically assessed and unresolved fracture, syndesmotic/high-ankle injury, tendon dislocation/rupture and major osteochondral injury concerns have been addressed as required.

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
ligament laxity/test finding if actually assessed
external support already used
```

Hard rules:

```text
inversion mechanism + lateral pain != automatically uncomplicated LAS
ligament tenderness/test != autonomous sprain grade
negative/unknown imaging != proof of no relevant structural injury
```

Core rehabilitation direction:

```text
protection according to severity
progressive weight bearing / gait normalization
protected ROM
progressive strength
balance / proprioceptive / neuromuscular training
progressive functional loading
criterion-based return to work/sport
```

Brace or taping may be condition/phase-sensitive supports. Severe injuries may require short immobilization when medically indicated; the generator does not prescribe a universal duration.

## AF2 — Chronic ankle instability / recurrent lateral ankle sprain presentation

Structured key:

```text
chronic_ankle_instability_recurrent_sprain
```

Default wording without formal diagnosis:

> Υποτροπιάζοντα διαστρέμματα / αίσθημα αστάθειας ποδοκνημικής με χαρακτηριστικά chronic ankle instability

Optional clinician assertion:

```text
formal_chronic_ankle_instability_diagnosis: yes / no / not_stated
```

Useful context:

```text
number/history of sprains
recurrent giving-way
patient-perceived instability
functional instability during sport/work
balance / dynamic postural-control deficit
hop/landing/change-of-direction deficit if assessed
ankle ROM deficit
strength deficit
objective ligament laxity if actually assessed
prior rehabilitation / brace use
```

Hard rules:

```text
subjective giving-way != objective mechanical instability
recurrent pain != chronic ankle instability automatically
brace/taping != stand-alone rehabilitation
```

Core rehabilitation:

```text
proprioceptive / neuromuscular exercise
balance and dynamic postural-control training
ankle / lower-limb strength
ROM restoration where impaired
landing / cutting / sport-specific progression when relevant
criterion-based recurrent-sprain prevention
```

## AF3 — Achilles tendinopathy

Structured key:

```text
achilles_tendinopathy
```

Candidate direct subtypes:

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

Useful context:

```text
midportion vs insertional pain location
morning/start-up stiffness
pain with walking/running/jumping
training/load change
single-leg heel-raise capacity
plantar-flexor strength/endurance if assessed
ankle dorsiflexion if assessed
local tendon thickening/tenderness if examined
imaging context if available
```

Hard rules:

```text
Achilles-region pain != tendinopathy automatically
imaging tendon thickening/degeneration != automatically symptomatic diagnosis
midportion evidence != automatically transferable to insertional disease
acute pop + bruising + marked plantar-flexion weakness / positive rupture concern != tendinopathy pathway
```

Core direction:

```text
education / load modification without routine complete rest
progressive tendon loading
plantar-flexor strength/endurance
kinetic-chain / neuromuscular work where relevant
graded walking/running/jumping return
```

The generator does not freeze one loading method as uniquely superior; eccentric, concentric, heavy-slow and other progressive loading strategies may be individualized. Insertional disease must avoid blindly importing deep-dorsiflexion/compression loading from midportion protocols.

## AF4 — Plantar heel pain / plantar fasciitis presentation

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

Useful context:

```text
medial plantar heel pain
first-step morning pain
pain after rest
pain with prolonged standing/walking
plantar fascia / medial calcaneal tenderness if examined
gastrocnemius/soleus flexibility / ankle dorsiflexion if assessed
foot/ankle muscle strength/capacity
footwear/load context
BMI/weight context when clinically relevant
```

Hard rules:

```text
plantar heel pain != plantar fasciitis automatically
heel spur != automatic pain generator
calcaneal tenderness + high-impact load / osteoporosis-risk context may require stress-fracture consideration
neuropathic/burning symptoms != routine plantar-fascia diagnosis
```

Core/support options:

```text
plantar-fascia-specific stretching
gastrocnemius/soleus stretching where relevant
foot/ankle resistance exercise
manual therapy where impairment-specific
taping as short-term adjunct
foot orthosis only as part of multimodal management, not automatic stand-alone treatment
night splint may be condition-sensitive for persistent first-step morning pain
```

## AF5 — Posterior tibial tendon dysfunction / progressive collapsing foot deformity — flexible conservative pathway

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

Useful context:

```text
medial ankle/arch pain
acquired arch change if established
hindfoot valgus / forefoot abduction if assessed
single-leg heel-raise performance if assessed
posterior-tibial strength/load deficit
flexible vs rigid deformity context
orthosis/AFO already prescribed or considered
walking/standing limitation
```

Hard rules:

```text
flat foot alone != PCFD/PTTD diagnosis
medial ankle pain alone != posterior-tibial tendinopathy
rigid/progressive deformity != generic tendon-loading pathway
```

Core conservative direction may include:

```text
load management
posterior-tibial / foot / calf strengthening according to stage and tolerance
functional lower-limb strengthening
orthosis/AFO strategy when clinically appropriate
footwear modification where relevant
walking/function progression
```

## AF6 — Peroneal tendon disorder — conservative rehabilitation

Structured key:

```text
peroneal_tendon_disorder_nonoperative
```

Candidate subtypes:

```text
peroneal_tendinopathy
confirmed_partial_peroneal_tendon_tear_nonoperative
other_established_peroneal_tendon_disorder
```

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
snapping/subluxation behind fibula != routine tendinopathy; structural assessment may be required
confirmed tear requires established diagnosis and nonoperative decision
```

## AF7 — 1st-MTP osteoarthritis / hallux rigidus — conservative functional pathway

Structured key:

```text
first_MTP_osteoarthritis_hallux_rigidus
```

Display when clinician established:

> Οστεοαρθρίτιδα 1ης μεταταρσιοφαλαγγικής / hallux rigidus — συντηρητική αντιμετώπιση

Useful context:

```text
1st-MTP pain
painful/restricted dorsiflexion
push-off limitation
walking/running limitation
shoe intolerance
radiographic OA context
footwear / rocker-sole / orthosis context
```

Hard rule:

```text
radiographic 1st-MTP OA != automatically symptomatic hallux rigidus
```

Evidence for nonoperative interventions is limited compared with ankle sprain, Achilles and plantar heel pain, so the generator should avoid claiming one orthosis/shoe strategy is clearly superior.

## AF8 — Mechanical metatarsalgia / forefoot overload presentation

Structured key:

```text
mechanical_metatarsalgia_forefoot_overload
```

Default wording:

> Μηχανικού τύπου μεταταρσαλγία / υπερφόρτιση προσθίου ποδός

Useful context:

```text
plantar metatarsal-head pain
standing/walking/running load
callus/pressure pattern if examined
shoe intolerance
MTP motion / toe deformity context
fat-pad / plantar-plate concern if established
```

Hard rules:

```text
forefoot pain != metatarsalgia automatically
metatarsalgia is a symptom-region label and does not identify the exact structural cause
stress fracture / plantar-plate rupture / inflammatory disease / Morton neuroma concern must not be hidden under generic overload wording
```

Footwear modification, metatarsal offloading/pads and orthoses may be condition-sensitive supports, with limited evidence for exact device superiority.

## AF9 — Assessed post-traumatic ankle / foot pain or stiffness

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

The utility never relabels unassessed ankle/foot trauma as a `simple sprain`.

## AF10 — Postoperative ankle / foot rehabilitation — candidate pathway

Structured key:

```text
postoperative_ankle_foot_rehabilitation
```

Candidate procedures:

```text
lateral_ligament_repair_or_reconstruction
Achilles_tendon_repair
peroneal_tendon_repair_or_stabilization
posterior_tibial_tendon_or_flatfoot_reconstruction
ankle_arthroscopy_or_osteochondral_procedure
hallux_or_forefoot_surgery
other_foot_ankle_operation
```

Required context:

```text
procedure
operation date
surgeon/protocol
weight-bearing status
boot/cast/brace/orthosis status
ROM restrictions
loading/strengthening restrictions
repair-specific precautions
wound/infection context
return-to-work/sport target
```

Hard rule:

```text
exact procedure / surgeon protocol > generic foot-ankle rehabilitation default
```

Product-owner confirmation is required before AF10 becomes a frozen routine pathway.

---

# 3. Candidate rare / advanced / context entities

## 3.1 Syndesmotic / high-ankle sprain

Candidate role:

```text
established_syndesmotic_ankle_injury
→ advanced structural pathway if nonoperative rehab is appropriate
```

Do not infer from pain location or squeeze/external-rotation test alone. Unresolved instability requires structural/specialist context.

## 3.2 Osteochondral lesion of talus / ankle joint

```text
established_osteochondral_lesion_talus
```

Advanced structural context; imaging finding alone does not establish symptom causation. Weight-bearing/loading restrictions if present must be preserved.

## 3.3 Ankle osteoarthritis

```text
ankle_osteoarthritis
```

Candidate role: primary vs rare/context depends on actual workflow. A 2026 AAOS CPG now addresses ankle OA, but the referral generator should reflect whether the product owner actually sends these patients to PT.

## 3.4 Morton neuroma / interdigital nerve presentation

```text
established_morton_neuroma
interdigital_neuralgic_forefoot_presentation
```

Candidate default: medical/context or rare, not automatically a PT pathway.

Hard rules:

```text
forefoot burning/numbness != Morton neuroma automatically
Mulder-type finding != autonomous diagnosis
```

## 3.5 Plantar plate / lesser-MTP instability

```text
established_plantar_plate_injury_or_lesser_MTP_instability
```

Rare/advanced structural context. Generic metatarsalgia must not hide an established tear/instability.

## 3.6 Tarsal tunnel / tibial-nerve entrapment

```text
established_tarsal_tunnel_or_tibial_nerve_entrapment
```

Rare neurological context. Burning/paresthesia alone does not establish entrapment.

## 3.7 Anterior tibial / extensor / flexor hallucis-longus tendon disorders

Candidate rare tendon entries:

```text
anterior_tibial_tendinopathy
extensor_tendon_disorder
flexor_hallucis_longus_tendon_disorder
```

Only if clinically established/appropriate; major rupture leaves routine tendinopathy pathway.

## 3.8 Hallux valgus

```text
hallux_valgus_context
```

Medical/structural/footwear context by default. The utility should not imply physiotherapy reverses structural deformity.

## 3.9 Inflammatory / crystal / neuropathic foot

Established medical context only:

```text
gout_or_crystal_disease
inflammatory_arthritis
neuropathic_foot
Charcot_neuroarthropathy
```

Acute hot swollen neuropathic foot / Charcot concern is a high-priority medical/offloading pathway, not routine exercise referral.

---

# 4. Pediatric / adolescent ankle-foot candidate navigation group

Candidate UI grouping:

```text
Παιδιά / Έφηβοι — ποδοκνημική / άκρος πόδας
```

Potential dedicated growth-related pathway:

```text
calcaneal_apophysitis_Sever_disease
```

Default wording without formal diagnosis:

> Πόνος πτέρνας σε παιδί/έφηβο με χαρακτηριστικά αποφυσίτιδας πτέρνας (Sever)

Hard rules:

```text
pediatric heel pain != Sever automatically
night/rest pain, systemic concern, acute trauma, focal bone tenderness or atypical course → medical/structural reassessment
```

Possible pediatric context only:

```text
symptomatic_flexible_flatfoot
accessory_navicular_syndrome_or_posterior_tibial_insertion_context
```

Whether this pediatric/adolescent group appears in v1 depends on actual workflow.

---

# 5. Direct shared-profile gateways

## 5.1 Shared fracture / post-immobilization profile

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

## 5.2 Shared muscle / myotendinous profile

```text
gastrocnemius strain
soleus strain
calf myotendinous injury
other acute lower-leg/foot muscle injury
```

## 5.3 Achilles rupture gateway

Candidate direct regional gateway:

```text
established_Achilles_rupture
→ shared structural/postoperative or fracture-style restriction framework depending final shared architecture
```

Acute rupture concern is never treated as Achilles tendinopathy.

---

# 6. Findings — selectable only when actually assessed

## 6.1 Pain / symptom location

```text
lateral ankle
medial ankle
anterior ankle
posterior ankle / Achilles
plantar heel
posterior heel
medial arch
lateral foot
midfoot
1st MTP
lesser metatarsal heads / forefoot
interdigital burning/paresthesia
other foot region
```

## 6.2 Symptom behaviour / function

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
```

## 6.3 Swelling / mechanical / instability findings

```text
swelling
bruising
clicking/catching
snapping
subjective giving-way
locking/mechanical block
recurrent sprain history
```

## 6.4 ROM

```text
dorsiflexion restricted
plantar flexion restricted
inversion restricted
eversion restricted
1st-MTP dorsiflexion restricted
painful active ROM
painful passive ROM
```

## 6.5 Strength / performance

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

## 6.6 Special/provocation findings — secondary expander only

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

# 7. Neurological / neurovascular model

Use when clinically relevant:

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
vascular_status: normal / abnormal / not_assessed
```

No `neurovascularly intact`, `no neurological deficit` or similar normal wording is generated from missing data.

---

# 8. Safety / reassessment semantics

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
uneven ground
standing tolerance
single-leg stance
heel raise
squat/lunge
running
jumping
landing
cutting/change of direction
push-off
shoe tolerance
work/manual duties
gym/resistance exercise
sport-specific activity
school PE / youth sport
ADLs/self-care
patient-priority activity / free text
```

---

# 10. Context-sensitive goals

Nothing is globally preselected.

Candidate goal families:

```text
reduce symptom irritability
restore safe ankle/foot ROM where impaired
normalize gait / progressive weight bearing
improve plantar-flexor capacity
improve inversion/eversion strength
improve intrinsic-foot / arch-support capacity where relevant
improve balance / proprioception / dynamic postural control
improve tendon load capacity
improve walking/standing tolerance
improve push-off
improve footwear tolerance
reduce recurrent-sprain risk
progressive return to running/jumping/cutting
criterion-based return to work/sport
restore function within fracture/surgical restrictions
self-management and load adaptation
age-appropriate return to PE/sport
```

---

# 11. Supports / adjuncts — candidate policy

## 11.1 Condition-sensitive supports

```text
ankle brace
ankle taping
walking aid / gait support
boot/immobilization context when medically prescribed
foot orthosis
AFO
heel lift
metatarsal pad/offloading strategy
footwear modification / rocker-sole strategy
night splint for selected plantar-heel presentation
```

These are never globally preselected.

## 11.2 Optional adjunct questions for product-owner review

Candidate options:

```text
manual therapy / joint mobilization → condition/impairment sensitive
soft-tissue techniques → optional
dry needling → evidence exists for plantar heel pain but workflow decision required
acupuncture → conflicting/insufficient evidence for ankle sprain; workflow decision required
ESWT → potentially relevant to plantar heel pain / Achilles / selected tendinopathy, but not a default generator recommendation without product-owner confirmation
```

Therapeutic ultrasound should not be a default recommendation for acute lateral ankle sprain or plantar fasciitis.

---

# 12. Deterministic consistency rules

```text
AF1 + inversion mechanism/lateral pain only
→ do not infer uncomplicated lateral ankle sprain

AF1 + unresolved syndesmosis / fracture / peroneal-dislocation concern
→ safety prompt; no routine LAS wording

AF2 + subjective giving-way only
→ do not infer mechanical instability

AF2 + brace selected
→ brace does not replace proprioceptive/neuromuscular rehabilitation

AF3 + Achilles pain only
→ do not infer tendinopathy

AF3 + acute rupture concern
→ leave tendinopathy pathway

AF3 midportion protocol + insertional subtype
→ do not blindly import deep-dorsiflexion/compression loading

AF4 + plantar heel pain only
→ do not infer plantar fasciitis

AF4 + heel spur imaging only
→ do not infer pain causation

AF5 + flat foot alone
→ do not infer PCFD/PTTD

AF5 + rigid/progressive deformity
→ do not generate unrestricted generic tendon-loading plan

AF6 + lateral pain only
→ do not infer peroneal tendinopathy

AF6 + snapping/subluxation concern
→ structural reassessment context

AF7 + radiographic 1st-MTP OA only
→ do not infer symptomatic hallux rigidus

AF8 + generic forefoot pain
→ do not hide stress fracture / plantar-plate / inflammatory / neural differential

post-trauma + unresolved Lisfranc / fracture / syndesmosis / rupture concern
→ safety prompt

known fracture + unknown healing/weight-bearing/loading status
→ warning; no unrestricted rehab wording

pediatric heel pain + atypical/red-flag context
→ do not label Sever disease automatically

not_assessed neurovascular component
→ never generate normal wording
```

---

# 13. Evidence-governance boundary

Stable candidate evidence directions:

```text
acute lateral ankle sprain → protection/support by severity + progressive weight bearing + structured exercise
chronic ankle instability → proprioceptive/neuromuscular exercise core; brace/taping not stand-alone
midportion Achilles tendinopathy → progressive tendon loading first-line; complete rest not routine
plantar heel pain/plantar fasciitis → stretching + resistance exercise + impairment-specific manual therapy; taping short-term; orthoses multimodal not isolated
PCFD/PTTD → orthosis/AFO and exercise may support selected flexible/early conservative cases; evidence quality less mature
peroneal disorders → diagnosis-sensitive conservative care; subluxation/dislocation/major tear requires structural context
hallux rigidus / 1st-MTP OA → conservative evidence limited; avoid device-superiority claims
mechanical metatarsalgia → offloading/orthotic strategies may reduce pressure; exact structural cause remains important
Sever disease → conservative treatment standard; no rigid universal protocol
```

Evidence anchors reviewed for this candidate include:

- JOSPT/APTA `Ankle Stability and Movement Coordination Impairments: Lateral Ankle Ligament Sprains Revision 2021`;
- JOSPT/APTA `Achilles Pain, Stiffness, and Muscle Power Deficits: Midportion Achilles Tendinopathy Revision 2024`;
- JOSPT/APTA `Heel Pain — Plantar Fasciitis: Revision 2023`;
- 2025/2026 systematic review/meta-analysis of orthotic treatment in progressive collapsing foot deformity;
- ESSKA-AFAS peroneal-tendon consensus plus contemporary peroneal-instability review;
- 2024 Cochrane/reviews for Morton neuroma;
- 2024 Cochrane/review evidence for hallux rigidus / 1st-MTP OA;
- 2024–2025 systematic reviews of conservative treatment for calcaneal apophysitis (Sever disease);
- recent metatarsalgia orthotic systematic review/meta-analysis;
- 2026 AAOS ankle-osteoarthritis CPG as a current evidence anchor if ankle OA is retained in workflow.

Evidence-sensitive details to refresh immediately before CU-2 implementation:

```text
Achilles insertional-vs-midportion loading details
plantar heel adjunct / ESWT / dry-needling wording
PCFD staging and orthosis/AFO semantics
peroneal tear/subluxation conservative thresholds
syndesmotic and Lisfranc routing
ankle-OA PT/support wording
pediatric Sever/accessory-navicular routing
shared fracture / Achilles rupture / muscle integration
```

---

# 14. Product-owner decisions required before freeze

1. **Lateral ankle sprain:** do you refer these frequently enough to keep AF1 as a high-visibility default pathway?
2. **Chronic ankle instability / recurrent sprains:** common/default or rare?
3. **Achilles tendinopathy:** do you see enough to keep default? Should `midportion` and `insertional` remain subtypes of one pathway or be separate top-level choices?
4. **Plantar fasciitis / plantar heel pain:** common/default?
5. **Posterior tibial tendon dysfunction / PCFD:** do you refer these enough for default visibility?
6. **Peroneal tendinopathy / partial tear:** default or rare/advanced?
7. **Hallux rigidus / 1st-MTP OA:** do you refer to physiotherapy, or should it be medical/context only?
8. **Mechanical metatarsalgia:** do you refer enough for a routine pathway?
9. **Morton neuroma:** do you send these for PT/orthotic management, or should this remain medical/context only?
10. **Postoperative foot/ankle:** do you see/referral enough Achilles repair, ligament reconstruction, ankle arthroscopy, tendon repair or forefoot surgery to keep AF10 active?
11. **Ankle OA:** do you refer these? A 2026 AAOS guideline now exists, but actual workflow should decide visibility.
12. **Syndesmotic/high-ankle sprain:** default structural pathway or rare/advanced?
13. **Pediatric/adolescent group:** do you see enough `Sever disease`, symptomatic flexible flatfoot or accessory-navicular cases to include it?
14. **Dry needling:** plantar-heel CPG supports it, but do you want it available in Ankle/Foot based on your referral workflow?
15. **Acupuncture:** do you refer for ankle/foot conditions? Candidate recommendation is to exclude unless workflow says otherwise.
16. **ESWT:** do you send for plantar fasciitis, Achilles tendinopathy or other ankle/foot tendinopathy, or should therapist-proposed use be documentable only?
17. **Brace / taping / orthoses / AFO / heel lifts / metatarsal pads / footwear modification:** keep as condition-sensitive supports, never generic defaults? Candidate recommendation: yes.
18. Any frequent real referral missing — e.g. `accessory navicular`, `tarsal tunnel`, `anterior tibial tendinopathy`, `FHL`, `plantar-plate injury`, `hallux valgus`, `ankle impingement`, `osteochondral talus`, `Morton neuroma`, or something else?

This file remains **DESIGN CANDIDATE / NOT FROZEN** until these workflow decisions are resolved. Runtime implementation remains unauthorized.
