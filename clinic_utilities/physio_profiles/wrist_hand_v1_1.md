# Wrist / Hand Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-27.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful wrist/hand referral choices for the product owner's real referral workflow while preserving diagnosis-vs-finding separation, neurological/tendon/ligament safety semantics, function/dexterity, orthosis logic and physiotherapist autonomy.
> **Supersedes as active wrist/hand design:** `clinic_utilities/physio_profiles/wrist_hand_v1.md`.
> **Prior frozen regional profiles:** `cervical_v1_1.md`, `lumbar_v1_1.md`, `shoulder_v1_1.md`, `elbow_v1_1.md`.

---

# 1. Core design contract

```text
PRIMARY CLINICAL PATHWAY
+
ACTUAL FINDINGS / MODIFIERS
+
FUNCTIONAL IMPACT / DEXTERITY
+
SAFETY / REASSESSMENT CONTEXT
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
subjective paresthesia != objective sensory deficit
pain-limited effort != structural weakness or tendon rupture
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
orthosis != automatically mandatory
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

The utility structures a referral and checks consistency. It must not autonomously diagnose De Quervain disease, osteoarthritis, carpal tunnel syndrome, TFCC tear, ligament instability, sagittal-band injury, tendon rupture, CRPS or inflammatory hand disease.

## 1.1 Cyprus workflow / service-availability rule

The product owner reports that dedicated `hand therapist` services are not routinely available in Cyprus.

Therefore generated referrals use:

```text
physiotherapy / wrist-hand rehabilitation
```

and may request:

```text
physiotherapist with relevant wrist/hand rehabilitation and orthosis/protocol experience where available
```

The generator must not make care contingent on access to a professional title/service that is not routinely available locally.

For injuries in which splint/orthosis design or protected tendon/ligament progression is critical, wording should emphasize competence and adherence to the exact surgical/structural protocol rather than the title `hand therapist`.

---

# 2. Frozen default primary wrist / hand pathways

## WH1 — De Quervain stenosing tenosynovitis / first dorsal compartment disorder

Structured key:

```text
de_quervain_first_dorsal_compartment_disorder
```

Display:

> Στενωτική τενοντοελυτρίτιδα De Quervain / πάθηση 1ου ραχιαίου διαμερίσματος

Default wording without formal diagnosis:

> Κερκιδικός πόνος καρπού/βάσης αντίχειρα με χαρακτηριστικά 1ου ραχιαίου διαμερίσματος

Optional clinician assertion:

```text
formal_de_quervain_diagnosis: yes / no / not_stated
```

Useful findings/context:

```text
radial-styloid / first-dorsal-compartment pain
APL/EPB-region tenderness
pain with thumb abduction/extension loading
pain with repetitive thumb use / lifting / caregiving
Finkelstein/Eichhoff/WHAT-type finding if actually examined
symptom irritability
work/caregiving/sport load
prior injection/immobilization context if relevant
```

Hard rule:

```text
positive Finkelstein/Eichhoff/WHAT-type test alone != definitive De Quervain diagnosis
```

Evidence boundary:

- current comparative evidence supports corticosteroid injection with short thumb-spica immobilization as an important first-line medical-management strategy;
- physiotherapy may support activity/load modification and functional recovery but must not be represented as clearly superior first-line monotherapy;
- rehabilitation may include restoration of thumb/wrist function and progressive loading when appropriate;
- ESWT is not part of this frozen profile.

## WH2 — Thumb CMC-1 osteoarthritis / rhizarthrosis

Structured key:

```text
thumb_cmc1_osteoarthritis
```

Display:

> Οστεοαρθρίτιδα βάσης αντίχειρα / ριζάρθρωση CMC-1

Use when clinician-established or carried as the clinician's working diagnosis.

Useful findings/context:

```text
thumb-base pain
pain with key pinch / tip pinch
pain with opening jars/containers
pain with gripping/turning keys
reduced pinch strength if assessed
reduced grip strength if assessed
CMC-1 tenderness
grind-type finding if actually examined
adduction contracture / MCP compensation if present
functional opposition/reach deficit
radiographic OA context optional
```

Hard rules:

```text
positive grind test alone != OA diagnosis
radiographic OA alone != proof that all current symptoms arise from CMC-1
```

Rehabilitation directions may include:

```text
CMC-support orthosis when appropriate
thumb/hand exercise
pinch/grip strategy
joint-protection education
thenar/intrinsic/first-web-space function where relevant
assistive/adaptive strategies
load/activity modification
```

No structural reversal promise is generated.

## WH3 — Interphalangeal / generalized hand osteoarthritis

Structured key:

```text
interphalangeal_hand_osteoarthritis
```

Display:

> Οστεοαρθρίτιδα άκρας χείρας / μεσοφαλαγγικών αρθρώσεων

Possible established distribution:

```text
DIP-predominant
PIP-predominant
thumb-IP involvement
multi-joint hand OA
other clinician-entered hand-OA distribution
```

Useful findings/context:

```text
joint pain/stiffness
functional grip/pinch limitation
reduced digit ROM
bony enlargement/deformity if relevant
activity-related pain
morning stiffness duration if relevant
radiographic context optional
```

The utility does not infer OA from age, pain or radiographs alone. Inflammatory arthritis, psoriatic disease and crystal disease remain separate medical contexts.

Possible rehabilitation directions:

```text
maintain/improve digit motion
hand-strength exercise where appropriate
joint-protection strategies
task adaptation
assistive devices
selected orthoses where useful
```

## WH4 — Median neuropathy at the wrist / carpal tunnel syndrome

Structured key:

```text
median_neuropathy_at_wrist_carpal_tunnel
```

Default wording without formal diagnosis:

> Συμπτωματολογία μέσου νεύρου στην περιοχή του καρπιαίου σωλήνα

Optional clinician assertion:

```text
formal_carpal_tunnel_diagnosis: yes / no / not_stated
```

If `yes`:

> Σύνδρομο καρπιαίου σωλήνα / μέση νευροπάθεια στον καρπό

Subjective symptoms:

```text
paresthesia/numbness thumb-index-middle-radial-ring distribution
night symptoms
symptoms with sustained wrist position/use
flick/shaking history
dropping objects / subjective clumsiness
proximal radiation if present
```

Objective findings only if actually assessed:

```text
median-distribution sensory deficit
thenar/APB weakness
thumb-abduction/opposition weakness
thenar atrophy
grip/pinch deficit
```

Provocation/context:

```text
Tinel at carpal tunnel
Phalen/reverse-Phalen
Durkan/carpal compression
CTS-6 elements if explicitly assessed
EMG/NCS context
ultrasound context
```

Hard rules:

```text
paresthesia alone != objective neurological deficit
positive Phalen/Tinel/Durkan alone != definitive diagnosis
upper-limb neurodynamic test != diagnostic proof of CTS
```

Conservative directions may include education/activity-position modification and short-term neutral-wrist night orthosis when appropriate. Exercise/gliding/manual approaches must not be represented as proven long-term disease-modifying treatment.

Progressive thenar weakness/atrophy or worsening objective median deficit requires reassessment/specialist semantics.

Post-release rule:

```text
uncomplicated carpal-tunnel release != automatic supervised postoperative physiotherapy
```

A post-release referral requires an actual indication such as stiffness, scar/problem, functional deficit, complication, complex procedure or explicit surgeon instruction.

## WH5 — Ulnar-sided wrist / TFCC-related presentation

Structured key:

```text
ulnar_sided_wrist_tfcc_presentation
```

Terminology rule:

```text
TFCC = triangular fibrocartilage complex
```

`TFCL` is not used as the canonical label because contemporary anatomy/clinical literature uses TFCC and documents historical terminology inconsistency.

Default wording without formal diagnosis:

> Ωλένιος πόνος καρπού με χαρακτηριστικά TFCC / ωλένιας πλευράς του καρπού

Optional clinician assertion:

```text
formal_tfcc_injury_diagnosis: yes / no / not_stated
```

If `yes`, generated wording may carry the established diagnosis:

> Κάκωση / ρήξη τριγώνου ινοχόνδρινου συμπλέγματος (TFCC)

Important context:

```text
traumatic vs degenerative onset
central vs peripheral/foveal lesion if established
DRUJ stability: stable / unstable / not_assessed
clicking/catching if present
pain with forearm rotation
pain with axial loading / push-up / chair-rise
foveal tenderness/sign if examined
press/TFCC-load-type finding if examined
MRI/arthrogram/arthroscopy context if available
ulnar variance/structural context only when relevant
```

Hard rules:

```text
ulnar-sided wrist pain != TFCC tear
foveal/press/load test != structural diagnosis
incidental TFCC imaging finding != automatically symptomatic lesion
```

Stable selected injuries may follow a conservative pathway. DRUJ instability, established foveal detachment/full-thickness lesion or clinically important mechanical instability changes management and triggers specialist/restriction semantics.

Possible rehabilitation directions:

```text
protected activity/loading
wrist/forearm ROM as safe
progressive grip/forearm strength
proprioceptive/dynamic-stability work
brace/orthosis when indicated
graded return to axial loading/work/sport
```

## WH6 — Intersection syndrome

Structured key:

```text
intersection_syndrome
```

Display:

> Σύνδρομο διασταύρωσης τενόντων / intersection syndrome

Optional clinician assertion:

```text
formal_intersection_syndrome_diagnosis: yes / no / not_stated
```

Default presentation wording without a formal diagnosis:

> Ραχαιοκερκιδικός πόνος περιφερικού αντιβραχίου/καρπού με χαρακτηριστικά της περιοχής διασταύρωσης τενόντων

Useful findings/context:

```text
dorsoradial distal-forearm pain proximal to radial styloid
localized tenderness/swelling if present
crepitus with wrist/thumb movement if actually present
pain with repetitive wrist extension/flexion
rowing/racquet/skiing/weight-training/manual-work context
proximal vs distal intersection if established
```

Hard rule:

```text
radial wrist pain != automatically intersection syndrome
intersection syndrome != De Quervain
```

The utility must preserve the anatomical distinction from the first dorsal compartment. Conservative rehabilitation may include activity/load modification, temporary protective splint/brace when appropriate and progressive mobility/strength/return to activity. No rigid evidence-based loading protocol is frozen.

## WH7 — Thumb MCP collateral-ligament injury — UCL or RCL

Structured key:

```text
thumb_mcp_collateral_ligament_injury_rehabilitation
```

Display:

> Κάκωση πλαγίου συνδέσμου MCP αντίχειρα — UCL / RCL — αποκατάσταση

Required subtype:

```text
ulnar_collateral_ligament_UCL
radial_collateral_ligament_RCL
other_established_thumb_mcp_collateral_injury
```

Required context where relevant:

```text
acute vs chronic
partial vs complete if established
stable vs unstable
operative vs nonoperative
immobilization/orthosis status
ROM/loading/pinch restrictions
specialist/surgeon context
```

UCL-specific context:

```text
Stener-lesion concern addressed if relevant
```

RCL-specific context:

```text
MCP subluxation/instability context if established
```

Stress pain/laxity remains a finding and does not independently establish tear grade.

Safety boundary:

```text
acute thumb MCP injury
+ marked instability / suspected complete tear / unresolved displaced UCL-Stener concern / major RCL instability or subluxation concern
→ specialist reassessment prompt
→ no unrestricted rehabilitation wording
```

Stable injuries designated for conservative rehabilitation may include protected motion, progressive pinch/grip, thumb stabilizer strength and graded return according to restrictions.

## WH8 — Sagittal-band injury / extensor tendon subluxation at MCP

Structured key:

```text
sagittal_band_injury_extensor_tendon_instability
```

Display:

> Κάκωση sagittal band / αστάθεια-υπεξάρθρημα εκτείνοντα τένοντα στην MCP

Use only when clinician-established or clearly documented as the working structural diagnosis.

Required context:

```text
digit
acute vs chronic
traumatic vs atraumatic/inflammatory if established
extensor tendon subluxation/dislocation: yes/no/not_stated
ability to actively extend MCP/digit
orthosis/splint plan if known
operative vs nonoperative plan
```

Hard rules:

```text
MCP pain/swelling != sagittal-band tear
snapping alone != definitive sagittal-band diagnosis
```

Conservative rehabilitation for appropriate acute closed injuries may involve protective/relative-motion or MCP-positioning orthosis and protocol-based controlled motion. Chronic injury, persistent tendon instability or failed conservative treatment may require hand-surgery reassessment.

Because the evidence is largely lower-level/heterogeneous, the generator must not invent a universal splint design or duration.

## WH9 — Digital tendon injury / deformity-specific rehabilitation

Structured key:

```text
digital_tendon_injury_rehabilitation
```

Directly selectable subtype:

```text
mallet_finger
central_slip_injury
boutonniere_injury_or_deformity
extensor_tendon_injury_nonoperative
extensor_tendon_repair_postoperative
flexor_tendon_injury_nonoperative
flexor_tendon_repair_postoperative
other_established_digit_tendon_injury
```

This is a true referral pathway because the product owner refers these injuries for rehabilitation.

Required context:

```text
digit
injury zone when known
injury/repair date
complete vs partial if established
operative vs nonoperative
repair technique/protocol if available
healing/stability status
orthosis/splint requirement
active/passive motion restrictions
loading/strengthening restrictions
surgeon/hand-surgery instructions
```

Hard safety boundary:

```text
acute laceration or trauma
+ new inability to actively flex/extend expected joint/digit
+ unresolved tendon rupture/laceration concern
→ urgent/timely structural assessment
→ no routine generic rehabilitation wording
```

Protocol rule:

```text
exact tendon zone / repair protocol / surgeon restriction
>
generic early-motion preference
```

Current evidence supports early controlled motion in many repaired tendon scenarios and relative-motion approaches in selected extensor repairs, but evidence and protocol differ by tendon, zone and repair. No one universal tendon protocol is generated.

## WH10 — Post-traumatic wrist / hand pain or stiffness after assessed injury

Structured key:

```text
post_traumatic_wrist_hand_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία καρπού ή άκρας χείρας μετά από αξιολογημένη κάκωση

Use only after unresolved fracture, dislocation, tendon rupture/laceration, major ligament injury or unstable joint injury has been addressed as required.

Required context:

```text
injury/date or phase
established structural diagnosis if any
imaging/orthopaedic/hand-surgery context
immobilization history
current ROM/loading/use restrictions
```

The utility never labels an unassessed traumatic wrist/hand injury as a `simple sprain`.

## WH11 — Post-operative wrist / hand rehabilitation

Structured key:

```text
postoperative_wrist_hand_rehabilitation
```

Display:

> Μετεγχειρητική αποκατάσταση καρπού / άκρας χείρας

This is an active pathway because the product owner sees postoperative wrist/hand patients.

Required context:

```text
operation/procedure
operation date
surgeon/protocol when available
immobilization/orthosis status
wound/scar status if relevant
ROM restrictions
loading/strengthening restrictions
weight-bearing/use restrictions
specific tendon/ligament repair precautions
return-to-work/sport constraints
```

Examples:

```text
tendon repair
thumb collateral-ligament repair/reconstruction
sagittal-band/extensor stabilization
TFCC repair
CMC arthroplasty
selected fracture fixation with explicit healing context
Dupuytren procedure
other wrist/hand surgery
```

No generic postoperative timeline is invented.

Specific CTS rule remains:

```text
uncomplicated carpal-tunnel release + no specific rehabilitation indication
→ do not auto-generate supervised postoperative physiotherapy
```

---

# 3. Rare / advanced / context entities

## 3.1 Guyon's canal / ulnar neuropathy at wrist — rare

Structured key:

```text
formal_guyon_canal_ulnar_neuropathy
```

Default symptom wording when not formally diagnosed:

> Συμπτωματολογία ωλενίου νεύρου στην περιοχή του καρπού / καναλιού Guyon

Possible objective findings:

```text
ulnar-distribution sensory deficit according to lesion zone
interosseous/intrinsic weakness
pinch weakness / Froment-type finding if assessed
hypothenar/interosseous atrophy
```

Because mass/ganglion, repetitive compression and other structural causes may occur, progressive or unexplained objective deficit requires reassessment.

## 3.2 Scapholunate / lunotriquetral injury / carpal instability — rare advanced

Possible keys:

```text
scapholunate_ligament_injury
lunotriquetral_ligament_injury
other_established_carpal_instability
```

Dorsal pain, clicking or Watson/scaphoid-shift findings do not establish instability. Established structural instability requires specialist context and explicit restrictions.

## 3.3 Other established wrist tendon disorder — secondary/advanced

Possible subtypes:

```text
ECU_tendinopathy
ECU_instability_or_subluxation
FCR_tendinopathy
FCU_tendinopathy
other_extensor_tendinopathy
other_flexor_tendinopathy
```

De Quervain and intersection syndrome remain separate default pathways. ECU snapping/subluxation must not be collapsed into ordinary tendinopathy.

## 3.4 CRPS — established diagnosis advanced pathway

Structured key:

```text
formal_CRPS_upper_limb_diagnosis
```

Display:

> Σύνδρομο σύνθετου περιοχικού πόνου (CRPS) άνω άκρου — λειτουργική αποκατάσταση

Hard rule:

```text
pain + swelling + color/temperature change + stiffness != automatic CRPS diagnosis
```

Only an established clinician diagnosis may activate this pathway.

Possible rehabilitation directions:

```text
graded functional use/movement
ROM and progressive task exposure
desensitization where appropriate
edema/hand-function management
graded motor imagery / mirror-based strategies where appropriate
coordination with pain/medical management
```

No rigid CRPS protocol is generated.

## 3.5 Trigger finger / trigger thumb — medical/context only

The product owner does not routinely refer trigger digit for physiotherapy in the local Cyprus workflow because dedicated hand-therapy services are not available and this is not part of their routine referral practice.

Possible context:

```text
known_trigger_finger_or_thumb
prior_injection_context
prior_surgery_context
post_trigger_release_specific_rehab_indication
```

The generator does not present physiotherapy as routine treatment for trigger digit.

## 3.6 Inflammatory / crystal hand context

Directly selectable only when established:

```text
known_rheumatoid_or_other_inflammatory_wrist_hand_involvement
known_psoriatic_hand_involvement
known_gout_or_crystal_disease_context
```

Acute hot swollen joint, flexor-sheath infection concern or unresolved inflammatory/infectious differential requires medical reassessment.

## 3.7 Dupuytren disease

Medical/context only unless postoperative rehabilitation is specifically indicated:

```text
known_Dupuytren_contracture
post_Dupuytren_procedure_rehabilitation_context
```

Physiotherapy is not represented as disease-modifying treatment for Dupuytren disease.

## 3.8 Ganglion / mass

Medical/context only:

```text
known_ganglion_or_other_established_benign_mass_context
```

The utility does not diagnose a ganglion from appearance alone or recommend aspiration as physiotherapy.

---

# 4. Examination findings — selectable only when actually assessed

## 4.1 Pain / symptom behaviour

```text
radial wrist pain
thumb-base pain
dorsoradial distal-forearm pain
dorsal central wrist pain
ulnar-sided wrist pain
volar wrist pain
palmar hand pain
specific MCP/PIP/DIP pain
pain with grip
pain with pinch
pain with wrist extension/flexion
pain with forearm rotation
pain with axial loading / push-up
pain with thumb use
pain with typing/mouse/phone
pain with manual tools/work
night symptoms
snapping/clicking/catching if present
```

## 4.2 Range of motion

```text
wrist flexion restricted
wrist extension restricted
radial deviation restricted
ulnar deviation restricted
forearm pronation restricted
forearm supination restricted
thumb CMC/MCP/IP restriction
thumb opposition deficit
digit flexion deficit
digit extension deficit
joint-specific contracture
painful active ROM
painful passive ROM
```

## 4.3 Strength / dexterity

```text
grip strength reduced if assessed
key-pinch strength reduced if assessed
tip-pinch strength reduced if assessed
three-jaw-chuck pinch reduced if assessed
wrist flexion weakness
wrist extension weakness
thumb abduction/opposition weakness
intrinsic-hand weakness
digit flexion/extension weakness
extensor lag if assessed
fine-motor/dexterity deficit
load intolerance without measured weakness
```

Pain-limited effort must not become structural weakness/rupture automatically.

## 4.4 Special/provocation findings

Secondary expander only:

```text
Finkelstein/Eichhoff/WHAT-type finding
CMC grind-type finding
Phalen/reverse-Phalen
Tinel at carpal tunnel
Durkan/carpal compression
fovea sign
TFCC load/press-type finding
DRUJ ballottement/stability finding
Watson/scaphoid-shift finding
thumb-MCP UCL stress finding
thumb-MCP RCL stress finding
ECU synergy/snapping finding
sagittal-band/extensor subluxation observation
other clinician-entered test
```

Tests are findings, not diagnoses.

---

# 5. Neurological model

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
```

Optional detailed fields:

```text
median-distribution sensory status
thenar/APB motor status
ulnar-distribution sensory status
intrinsic-hand motor status
superficial-radial sensory status if relevant
grip/pinch status
atrophy: present / absent / not_assessed
```

```text
not_assessed != normal
```

No global `no neurological deficit` wording is generated from missing components.

---

# 6. Safety / reassessment semantics

## 6.1 High-priority concerns

```text
acute trauma with unresolved fracture concern
snuffbox/scaphoid-fracture concern after trauma
acute trauma with unresolved dislocation/carpal instability
acute thumb MCP UCL/RCL complete-tear or major instability concern
unresolved UCL Stener-lesion concern
major RCL instability/subluxation concern
acute flexor/extensor tendon rupture or laceration concern
new inability to actively flex/extend expected digit/joint after injury
new/progressive median or ulnar motor deficit/atrophy
new neurovascular deficit after trauma
rapidly progressive swelling / severe structural concern
```

## 6.2 Infection / inflammatory concerns

```text
hot swollen joint / septic arthritis concern
wound/drainage/cellulitis
flexor tendon-sheath infection concern
systemic illness with acute hand swelling
unexplained rapidly progressive atraumatic swelling
other infectious/inflammatory concern
```

## 6.3 Other material concerns

```text
suspected CRPS not formally established
severe unremitting/progressive non-mechanical pain
rapidly progressive atraumatic loss of hand/wrist function
unexplained mass with progressive neurological deficit
true mechanical locking/unstable carpal symptoms requiring structural assessment
persistent extensor tendon subluxation/instability after sagittal-band injury
other clinician concern
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
hand/orthopaedic specialist pathway underway
urgent/same-day assessment arranged
routine physiotherapy deferred
other
```

No default `no red flags`, `no fracture`, `no tendon rupture`, `stable wrist`, `no infection` or `neurovascularly intact` wording is generated from missing information.

---

# 7. Functional limitations

```text
grip / handshake
key pinch
fine pinch / small-object handling
opening jars/containers
turning key/door handle
using cutlery
buttoning/zips/clothing
writing
keyboard/mouse use
phone use
cooking/food preparation
carrying bags
lifting child/object
pushing up from chair/floor
manual tools / screwdriver / repetitive work
personal care / grooming
medication packaging
racquet/club sport
cycling/handlebar load
gym / weight training
musical instrument / craft
work/manual duties
ADLs/self-care
patient-priority activity / free text
```

---

# 8. Context-sensitive goals

Nothing is globally preselected.

Common goal families:

```text
reduce symptom irritability
restore safe functional ROM
improve grip/pinch capacity where appropriate
improve tendon load tolerance where appropriate
improve thumb/digit stability where relevant
improve dynamic wrist/DRUJ stability where relevant
restore fine-motor/dexterity function
optimize orthosis self-management where selected
protect healing tendon/ligament/repair
restore functional use within protocol
progressive return to work/sport/manual tasks
joint-protection/task adaptation for OA
self-management/load adaptation
```

Condition cautions:

- CTS: do not promise reversal of thenar atrophy or long-term disease modification from exercise;
- TFCC/ligament: structural stability/restrictions outrank generic strengthening;
- tendon repair: exact zone/repair protocol outranks generic progression;
- CRPS: no guaranteed reversal or rigid protocol;
- OA: no structural reversal promise.

---

# 9. Rehabilitation directions

## 9.1 Core active directions

```text
physiotherapy assessment and individualized wrist/hand rehabilitation
education / activity and load modification
therapeutic exercise
progressive tendon loading where appropriate
wrist/forearm strengthening where relevant
grip/pinch strengthening where relevant
thumb/intrinsic-hand strengthening where relevant
ROM/mobility exercise where safe
fine-motor/dexterity retraining
joint-protection/task adaptation
assistive-device strategy where relevant
proprioceptive/dynamic-stability work for ligament/TFCC contexts
graded return to work/gym/sport/manual tasks
home exercise programme where appropriate
```

No single loading mode, orthosis or protocol is mandatory across conditions.

## 9.2 Orthosis / splint category

Orthosis is a condition-sensitive support category rather than a generic adjunct.

Possible selections:

```text
thumb-spica orthosis
CMC-support orthosis
neutral-wrist night orthosis
activity-specific wrist orthosis
relative-motion orthosis when injury/protocol appropriate
MCP/digit-positioning orthosis for sagittal-band injury when appropriate
mallet/central-slip protocol-specific orthosis
flexor/extensor tendon-repair protective orthosis
thumb-MCP collateral-ligament protective orthosis
other protocol-specific orthosis
```

Hard rules:

```text
orthosis suggested != automatically required
exact surgeon/injury protocol > generic orthosis suggestion
```

## 9.3 Optional adjuncts

Frozen wrist/hand adjunct list:

```text
manual therapy / joint mobilization where appropriate
soft-tissue techniques where appropriate
taping where appropriate
thermal/heat strategy for selected hand-OA symptoms where clinically appropriate
```

Explicitly excluded from Wrist/Hand v1.1 by product-owner decision:

```text
acupuncture
dry needling
ESWT
```

These exclusions are profile/workflow decisions and do not claim that every possible use of those techniques is scientifically ineffective.

Therapeutic ultrasound is not presented as a standard evidence-backed treatment for CTS or general wrist/hand pathology.

---

# 10. Shared fracture / post-immobilization boundary

Wrist/hand fractures route to the shared fracture profile:

```text
distal radius fracture
distal ulna fracture
scaphoid fracture
other carpal fracture
metacarpal fracture
phalangeal fracture
other wrist/hand fracture
```

Required future shared context:

```text
fracture site
date/phase
treatment
healing/stability status
immobilization/orthosis status
ROM restrictions
loading/use restrictions
orthopaedic/hand-surgery instructions
```

```text
fracture route + unresolved healing/loading context
→ warning
→ no unrestricted routine rehabilitation wording
```

Scaphoid concern after acute trauma remains a reassessment issue until adequately assessed.

---

# 11. Deterministic consistency rules

```text
WH1 + one De Quervain provocation test only
→ do not infer definitive De Quervain diagnosis

WH1 + wording says PT is evidence-preferred first-line over medical management
→ invalid

WH2 + imaging OA only
→ do not auto-assert symptomatic CMC-1 OA

WH3 + painful/swollen joints + no established OA diagnosis
→ preserve inflammatory/infectious differential

WH4 formal CTS diagnosis != yes
→ presentation wording only unless clinician provides diagnosis/context

WH4 + upper-limb neurodynamic positive
→ not diagnostic proof of CTS

WH4 + progressive thenar weakness/atrophy
→ prominent reassessment prompt

uncomplicated carpal-tunnel release + no rehab indication
→ no automatic supervised postoperative PT

WH5 ulnar-sided pain + fovea/load test only
→ do not infer TFCC tear

WH5 + DRUJ instability / established foveal detachment
→ specialist/restriction prompt before generic strengthening

WH6 radial wrist pain only
→ do not infer intersection syndrome or De Quervain

WH7 thumb-MCP stress finding only
→ do not infer tear grade

WH7 UCL + unresolved Stener/complete instability concern
→ specialist prompt

WH7 RCL + major instability/subluxation concern
→ specialist prompt

WH8 MCP pain/snapping only
→ do not infer sagittal-band tear

WH8 + persistent extensor subluxation/chronic failed conservative care
→ reassessment/specialist prompt

WH9 tendon injury + missing zone/healing/repair restrictions
→ warning

WH9 acute loss of active flexion/extension + rupture/laceration concern
→ structural reassessment prompt

WH10 post-traumatic + unresolved fracture/dislocation/tendon/major instability concern
→ safety prompt

WH11 postoperative + missing procedure/protocol/restrictions
→ warning

possible CRPS features without established diagnosis
→ do not auto-label CRPS

formal CRPS selected
→ function-restoration/multidisciplinary wording; no rigid protocol

trigger digit selected as routine primary physio pathway
→ invalid for frozen local workflow

acupuncture/dry needling/ESWT selected
→ invalid for Wrist/Hand v1.1

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological component
→ never generate normal wording
```

---

# 12. Generated wording examples

## 12.1 De Quervain presentation

Without formal diagnosis:

> Κερκιδικός πόνος του [side] καρπού/βάσης αντίχειρα με χαρακτηριστικά 1ου ραχιαίου διαμερίσματος, [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και αποκατάσταση με έμφαση στην προσαρμογή φορτίου, τη λειτουργική κινητικότητα και τη σταδιακή αποκατάσταση της ανοχής στη χρήση του αντίχειρα/καρπού. [Selected orthosis only if confirmed.]

If formally established, `De Quervain` may be stated explicitly.

## 12.2 Thumb CMC-1 OA

> Οστεοαρθρίτιδα βάσης αντίχειρα / ριζάρθρωση CMC-1 του [side] χεριού, με [selected findings] και περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση με άσκηση/ενδυνάμωση όπου ενδείκνυται, εκπαίδευση προστασίας άρθρωσης και λειτουργικών στρατηγικών και [selected CMC orthosis if selected].

## 12.3 Carpal-tunnel presentation

> Συμπτωματολογία μέσου νεύρου στην περιοχή του [side] καρπιαίου σωλήνα, με [selected subjective symptoms] και [selected objective findings only if actually assessed]. Παρακαλώ για συντηρητική φυσιοθεραπευτική αξιολόγηση/αντιμετώπιση με εκπαίδευση και τροποποίηση επιβαρυντικών θέσεων/δραστηριοτήτων, [selected neutral-wrist orthosis if selected] και παρακολούθηση τυχόν αντικειμενικών νευρολογικών μεταβολών.

## 12.4 TFCC-related presentation

> Ωλένιος πόνος του [side] καρπού με χαρακτηριστικά TFCC/ωλένιας πλευράς και [selected findings], με λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη αποκατάσταση με προστατευμένη/σταδιακή φόρτιση, κινητικότητα και δύναμη/σταθερότητα όπου ενδείκνυται, σύμφωνα με την καταγεγραμμένη κατάσταση της DRUJ και τυχόν περιορισμούς.

Only a clinician-established TFCC diagnosis may be stated definitively.

## 12.5 Intersection syndrome

> [Clinician-established intersection syndrome / dorsoradial intersection-type presentation] του [side] άνω άκρου με [selected findings] και περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αποκατάσταση με τροποποίηση φορτίου, κατάλληλη προστασία/orthosis όπου ενδείκνυται και σταδιακή επαναφορά κινητικότητας, δύναμης και ανοχής στη δραστηριότητα.

## 12.6 Thumb MCP collateral ligament

> Κάκωση [UCL/RCL] της MCP άρθρωσης του [side] αντίχειρα, [established grade/stability if entered], για [conservative/postoperative] αποκατάσταση. Παρακαλώ για προστατευμένη κινητοποίηση και προοδευτική αποκατάσταση σταθερότητας, δύναμης λαβής/τσιμπήματος και λειτουργίας σύμφωνα με τους καταγεγραμμένους περιορισμούς και το διαθέσιμο πρωτόκολλο.

## 12.7 Sagittal-band injury

> Κάκωση sagittal band του [digit/side] με [selected extensor instability/subluxation findings if established]. Παρακαλώ για φυσιοθεραπευτική αποκατάσταση με κατάλληλη προστασία/orthosis και ελεγχόμενη κινητοποίηση σύμφωνα με το στάδιο, τη σταθερότητα και το διαθέσιμο πρωτόκολλο, με επανεκτίμηση εάν επιμένει η αστάθεια του εκτείνοντα.

## 12.8 Digital tendon injury / repair

> [Established tendon injury/repair] του [digit/side], zone [if known], ημερομηνία [if entered]. Παρακαλώ για εξειδικευμένη φυσιοθεραπευτική αποκατάσταση με προστατευμένη κινητοποίηση και orthosis σύμφωνα με το ακριβές πρωτόκολλο του τένοντα/επέμβασης και τους καταγεγραμμένους περιορισμούς. Δεν εφαρμόζεται γενικό πρωτόκολλο όταν υπάρχει ειδική χειρουργική οδηγία.

## 12.9 Postoperative wrist/hand

> Μετεγχειρητική αποκατάσταση [procedure] του [side] καρπού/χεριού, επέμβαση [date if entered]. Παρακαλώ για φυσιοθεραπευτική αποκατάσταση σύμφωνα με το διαθέσιμο χειρουργικό πρωτόκολλο και τους καταγεγραμμένους περιορισμούς σε κινητικότητα, φόρτιση, ενδυνάμωση, χρήση του άκρου και orthosis.

---

# 13. Evidence-governance boundary

Stable structural decisions frozen in v1.1:

```text
provocation/special test != diagnosis
subjective paresthesia != objective neurological deficit
pain-limited effort != tendon rupture
radial wrist pain != automatically De Quervain/intersection syndrome
intersection syndrome != De Quervain
ulnar-sided wrist pain != automatically TFCC tear
incidental TFCC imaging finding != automatically symptomatic diagnosis
CTS neurodynamic test != diagnostic proof
uncomplicated carpal-tunnel release != automatic supervised PT
CMC-1 OA and interphalangeal hand OA remain distinct
thumb UCL and RCL are both active collateral-ligament pathways
UCL Stener concern != routine rehab
major RCL instability/subluxation != routine unrestricted rehab
sagittal-band injury is structural/protocol governed
finger flexor/extensor tendon rehab is zone/repair/protocol governed
possible CRPS features != formal CRPS diagnosis
fractures route to shared fracture/post-immobilization profile
orthoses are condition-sensitive supports
local service availability must not require a dedicated hand-therapist title
acupuncture/dry needling/ESWT are excluded from this regional profile
```

Evidence anchors reviewed for this freeze include:

- 2024 AAOS Management of Carpal Tunnel Syndrome CPG;
- 2023 systematic review/network meta-analysis of De Quervain treatments;
- recent systematic reviews/meta-analyses and trials of CMC-1 OA orthosis/exercise;
- hand-OA guidance/frameworks;
- recent nonoperative traumatic TFCC systematic review and TFCC clinical literature;
- contemporary TFCC anatomical terminology review;
- intersection-syndrome review literature;
- thumb MCP UCL/RCL review literature;
- systematic review of sagittal-band injury/extensor-tendon subluxation treatment;
- 2023/2026 relative-motion orthosis evidence for finger extensor repairs;
- 2024 systematic review/meta-analysis of zone-II flexor tendon rehabilitation;
- CRPS diagnostic/treatment guidance and rehabilitation reviews.

Evidence-sensitive items to refresh immediately before CU-2 implementation:

```text
exact De Quervain rehabilitation/loading role
CMC-1 orthosis type/duration and exercise dosage
hand-OA exercise/orthosis dosing
CTS short-term splinting/glide wording
TFCC immobilization/loading progression by lesion/stability
intersection-syndrome immobilization/loading details
thumb UCL/RCL conservative/postoperative protocols
sagittal-band orthosis design/duration
mallet/central-slip/boutonniere protocols
flexor/extensor tendon zone-specific protocols
CRPS rehabilitation evidence
postoperative procedure-specific protocols
```

---

# 14. Product-owner decisions incorporated

Product-owner decisions on 2026-08-27:

- trigger finger/thumb is **not** a routine physiotherapy referral in the Cyprus workflow and is context only;
- Guyon's canal remains rare/advanced;
- postoperative wrist/hand is an active pathway because these patients are seen;
- thumb collateral-ligament injury is an active pathway and must include both **UCL and RCL**;
- scapholunate/lunotriquetral instability remains rare/advanced;
- CRPS is an established-diagnosis advanced rehabilitation pathway;
- mallet finger, central-slip/boutonniere and flexor/extensor tendon injuries are directly selectable rehabilitation pathways;
- acupuncture and dry needling are excluded from Wrist/Hand;
- ESWT is excluded from Wrist/Hand, including De Quervain;
- intersection syndrome is directly selectable;
- TFCC is an active pathway;
- sagittal-band injury is an active pathway;
- the profile must reflect that dedicated hand-therapist services are not routinely available locally.

This file is the frozen wrist/hand clinical/content design for CU-1. Runtime implementation remains unauthorized.