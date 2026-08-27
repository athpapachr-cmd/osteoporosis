# Wrist / Hand Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful wrist/hand referral choices while preserving diagnosis-vs-finding separation, neurological/tendon/ligament safety semantics, function/dexterity, active rehabilitation, orthosis logic and hand-therapist autonomy.
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
CONFIRMED REHABILITATION / HAND-THERAPY DIRECTIONS
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

The utility may structure a referral and provide consistency prompts. It must not autonomously diagnose De Quervain disease, hand osteoarthritis, carpal tunnel syndrome, TFCC tear, ligament instability, trigger digit, tendon rupture, CRPS or inflammatory hand disease.

---

# 2. Proposed primary wrist / hand pathways

## WH1 — De Quervain stenosing tenosynovitis / first dorsal compartment disorder

Structured key:

```text
de_quervain_first_dorsal_compartment_disorder
```

Display:

> Στενωτική τενοντοελυτρίτιδα De Quervain / πάθηση 1ου ραχιαίου διαμερίσματος

Default wording when no formal diagnosis is asserted:

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
pain with grasping/lifting infant or repetitive thumb use
Finkelstein/Eichhoff/WHAT-type provocation if actually examined
symptom irritability
work/caregiving/sport load context
prior injection or immobilization context optional
```

Hard rule:

```text
positive Finkelstein/Eichhoff/WHAT-type test alone != definitive diagnosis
```

Evidence boundary:

- current comparative evidence favors corticosteroid injection combined with short thumb-spica immobilization as a first-line medical-management strategy;
- the referral utility may support rehabilitation after/alongside medical management but must not state that physiotherapy alone is the evidence-preferred first-line treatment;
- load/activity modification, recovery of thumb/wrist function and progressive tendon loading may be used as rehabilitation directions when clinically appropriate;
- ESWT is not proposed as a standard De Quervain adjunct in v1 because the comparative evidence is too limited/uncertain for a routine recommendation.

## WH2 — Thumb CMC-1 osteoarthritis / rhizarthrosis

Structured key:

```text
thumb_cmc1_osteoarthritis
```

Display:

> Οστεοαρθρίτιδα βάσης αντίχειρα / ριζάρθρωση CMC-1

Use when clinician-established or sufficiently documented as the working diagnosis.

Useful findings/context:

```text
thumb-base pain
pain with key pinch / tip pinch
pain with opening jars/containers
pain with gripping/turning keys
reduced pinch strength if assessed
reduced grip strength if assessed
CMC-1 tenderness
grind-type test finding if actually examined
adduction contracture / MCP compensation if present
functional opposition/reach deficit
radiographic OA context optional
```

Hard rules:

```text
positive grind test alone != OA diagnosis
radiographic OA alone != proof that all current symptoms arise from CMC-1
```

Rehabilitation/hand-therapy directions may include:

```text
thumb CMC-support orthosis when appropriate
hand/thumb exercise
pinch/grip strategy and joint-protection education
thenar/intrinsic and first-web-space function where relevant
assistive/adaptive strategies for ADLs
load/activity modification
```

The generator must not promise structural reversal. Current evidence supports orthoses and exercise/multimodal hand therapy for symptom/function improvement, with effect size and durability varying by intervention and follow-up.

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
bony enlargement/deformity if clinically relevant
activity-related pain
morning stiffness duration if relevant
radiographic context optional
```

Hand-OA classification criteria are not a diagnosis engine for the referral utility. The generator must preserve the clinician's actual diagnosis and must not infer OA from age plus hand pain or from incidental radiographs.

Rehabilitation directions may include exercise, maintenance/improvement of motion and strength, joint-protection strategies, task adaptation, assistive devices and selected orthoses where clinically useful.

Inflammatory arthritis, psoriatic disease and crystal disease remain separate medical contexts and are not inferred from painful/swollen joints.

## WH4 — Median neuropathy at the wrist / carpal tunnel syndrome

Structured key:

```text
median_neuropathy_at_wrist_carpal_tunnel
```

Default presentation wording without formal diagnosis:

> Συμπτωματολογία μέσου νεύρου στην περιοχή του καρπιαίου σωλήνα

Optional clinician assertion:

```text
formal_carpal_tunnel_diagnosis: yes / no / not_stated
```

If `yes`, generated wording may state:

> Σύνδρομο καρπιαίου σωλήνα / μέση νευροπάθεια στον καρπό

Subjective symptoms:

```text
paresthesia/numbness thumb-index-middle-radial ring distribution
night symptoms
symptoms with sustained wrist position/use
hand shaking/flick sign history
dropping objects / subjective clumsiness
pain radiating proximally if present
```

Objective findings only when actually assessed:

```text
median-distribution sensory deficit
thenar/APB weakness
thumb-abduction/opposition weakness
thenar atrophy
grip/pinch deficit
```

Provocation/context findings:

```text
Tinel at carpal tunnel
Phalen/reverse-Phalen finding
Durkan/carpal-compression finding
CTS-6 elements if explicitly assessed by clinician
EMG/NCS context if available
ultrasound context if available
```

Hard rules:

```text
paresthesia alone != objective neurological deficit
positive Phalen/Tinel/Durkan alone != definitive diagnosis
upper-limb neurodynamic test != diagnostic proof of CTS
```

The 2024 AAOS CPG allows CTS-6 as a diagnostic approach in appropriate clinician use and advises against using MRI or upper-limb neurodynamic testing as diagnostic substitutes. The referral generator itself does not calculate or diagnose CTS unless a dedicated clinician-entered score/diagnosis is explicitly supported later.

Conservative rehabilitation directions may include education, activity/position modification and short-term symptom-management strategies such as neutral-wrist night orthosis where appropriate. Tendon/nerve gliding or other exercise may be exposed as optional short-term rehabilitation content, but the generator must not claim that exercise/manual therapy/modalities provide proven long-term disease modification.

Progressive thenar weakness/atrophy or materially worsening objective median deficit requires reassessment/specialist semantics.

Post-release rule:

```text
carpal tunnel release
!= automatic supervised postoperative therapy referral
```

The 2024 AAOS CPG recommends against routine supervised postoperative therapy after uncomplicated carpal-tunnel release. A post-release hand-therapy referral should therefore require an actual indication such as specific stiffness, scar/problem, functional deficit, complication, complex procedure/context or clinician/surgeon instruction.

## WH5 — Ulnar-sided wrist pain / TFCC-related presentation

Structured key:

```text
ulnar_sided_wrist_tfcc_presentation
```

Default wording when no formal structural diagnosis is asserted:

> Ωλένιος πόνος καρπού με χαρακτηριστικά TFCC / ωλένιας πλευράς του καρπού

Optional clinician assertion:

```text
formal_tfcc_injury_diagnosis: yes / no / not_stated
```

If `yes`, generated wording may state the clinician-entered established diagnosis, for example:

> Κάκωση / ρήξη τριγώνου ινοχόνδρινου συμπλέγματος (TFCC)

Required/important context where relevant:

```text
traumatic vs degenerative onset
central vs peripheral/foveal lesion if established
DRUJ stability: stable / unstable / not_assessed
clicking/catching if present
pain with forearm rotation
pain with axial loading / push-up / chair-rise
foveal tenderness/sign if examined
press test or TFCC-load-type finding if examined
MRI/arthrogram/arthroscopy context if available
ulnar variance/structural context only if clinically relevant
```

Hard rules:

```text
ulnar-sided wrist pain != TFCC tear
foveal/press/load test != structural diagnosis
incidental TFCC imaging finding != automatically symptomatic lesion
```

Nonoperative management can produce good outcomes in selected stable injuries, but current evidence is heterogeneous. DRUJ instability, foveal full-thickness detachment or clinically important persistent mechanical instability materially changes prognosis/management and should trigger specialist/reassessment semantics rather than generic strengthening.

Possible rehabilitation directions in an appropriate conservative pathway:

```text
protected activity/loading
wrist/forearm ROM as safe
progressive grip and forearm strength
proprioceptive/dynamic-stability work
brace/orthosis when indicated
graded return to axial loading, racquet/gym/manual work
```

## WH6 — Wrist extensor/flexor tendinopathy / overuse disorder

Structured key:

```text
wrist_tendinopathy_overuse_disorder
```

Clinician-entered subtype:

```text
ECU_tendinopathy
FCR_tendinopathy
FCU_tendinopathy
intersection_syndrome
other_extensor_tendinopathy
other_flexor_tendinopathy
other_established_wrist_tendon_disorder
```

De Quervain remains WH1 rather than being hidden inside this group.

Useful findings/context:

```text
localized tendon-region pain/tenderness
pain with resisted subtype-specific loading
pain with gripping/forearm rotation
repetitive work/sport/gym exposure
load intolerance
swelling/crepitus if actually present
```

Hard rules:

```text
localized pain + one resisted test != automatic structural tendon diagnosis
ECU snapping/subluxation != ordinary ECU tendinopathy
```

Rehabilitation may include education/load modification, progressive subtype-specific loading, grip/forearm strengthening, ROM/mobility as needed and graded return to work/sport. Splint/orthosis use is condition- and irritability-specific rather than mandatory.

## WH7 — Trigger finger / trigger thumb

Structured key:

```text
trigger_digit_stenosing_flexor_tenosynovitis
```

Display:

> Εκτινασσόμενος δάκτυλος / trigger finger-thumb — στενωτική τενοντοελυτρίτιδα καμπτήρων

Candidate primary pathway because hand therapy and orthotic management may be used conservatively, but product-owner workflow confirmation is required before freeze.

Useful context/findings:

```text
which digit
pain/tenderness near A1 pulley if present
clicking/catching
locking frequency
ability to actively unlock
fixed contracture if present
functional grip/dexterity limitation
diabetes/inflammatory context optional if clinically relevant
prior injection context optional
```

Triggering/locking is a clinical feature but the generator should carry the clinician's working diagnosis rather than diagnose solely from a selected symptom.

Conservative hand-therapy options may include activity modification, orthosis/splint and selected movement/tendon-gliding strategies. Injection is a medical treatment decision and must not appear as a physiotherapy technique. Current evidence supports orthoses as a reasonable nonoperative option; injection remains an important medical-management option.

Severe fixed locking/contracture or progressive functional loss may require reassessment rather than indefinite routine therapy wording.

## WH8 — Thumb MCP ulnar collateral ligament injury / instability rehabilitation

Structured key:

```text
thumb_mcp_ucl_injury_rehabilitation
```

Display:

> Κάκωση ωλένιου πλαγίου συνδέσμου MCP αντίχειρα / skier's-gamekeeper's thumb — αποκατάσταση

Use only with clinician-established injury context.

Required context where relevant:

```text
acute vs chronic
partial vs complete if established
stable vs unstable
Stener-lesion concern addressed if relevant
operative vs nonoperative plan
immobilization/brace status
ROM/loading/pinch restrictions
```

Valgus stress pain/laxity does not independently establish tear severity or a Stener lesion.

Hard safety boundary:

```text
acute thumb MCP injury
+ marked instability / suspected complete tear / suspected Stener lesion
+ unresolved specialist assessment
→ reassessment/hand-surgery pathway prompt
→ no routine unrestricted rehabilitation wording
```

If the injury is stable and designated for conservative rehabilitation, directions may include protected motion, progressive pinch/grip and thumb stabilizer strength, proprioception and graded return according to restrictions.

Product-owner workflow confirmation required before freeze.

## WH9 — Post-traumatic wrist / hand pain or stiffness after assessed injury

Structured key:

```text
post_traumatic_wrist_hand_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία καρπού ή άκρας χείρας μετά από αξιολογημένη κάκωση

Use only after clinically important unresolved fracture, dislocation, tendon rupture/laceration, major ligament injury or unstable joint injury has been addressed as required.

Required context:

```text
injury/date or phase
established structural diagnosis if any
imaging/orthopaedic/hand-surgery context if relevant
immobilization history
current ROM/loading/use restrictions
```

The utility must not label an unassessed traumatic wrist/hand injury as a `simple sprain`.

## WH10 — Post-operative wrist / hand rehabilitation — workflow confirmation required

Structured key:

```text
postoperative_wrist_hand_rehabilitation
```

Display:

> Μετεγχειρητική αποκατάσταση καρπού / άκρας χείρας

Candidate for inclusion if this reflects the product owner's real workflow.

Required context:

```text
operation/procedure
operation date
surgeon/hand-surgeon protocol when available
immobilization/orthosis status
wound/scar status if relevant
ROM restrictions
loading/strengthening restrictions
weight-bearing/use restrictions
specific tendon/ligament repair precautions
return-to-work/sport constraints
```

Examples may include tendon repair, ligament repair/reconstruction, CMC arthroplasty, selected fracture fixation after healing/status is clear, Dupuytren procedure or other hand surgery.

No generic postoperative protocol may be invented.

Specific CTS rule:

```text
uncomplicated carpal-tunnel release
→ do not automatically prescribe routine supervised therapy
```

---

# 3. Candidate rare / advanced / secondary entities

## 3.1 Ulnar neuropathy at the wrist / Guyon's canal

Proposed role:

```text
clinician-entered rare neurological pathway or secondary diagnosis/context
```

Default symptom wording if not formal:

> Συμπτωματολογία ωλενίου νεύρου στην περιοχή του καρπού / καναλιού Guyon

Possible clinician assertion:

```text
formal_guyon_canal_ulnar_neuropathy: yes / no / not_stated
```

Potential objective findings:

```text
ulnar-distribution sensory deficit depending lesion zone
interosseous/intrinsic weakness
pinch weakness / Froment-type sign if assessed
hypothenar/interosseous atrophy
```

Etiology can include repetitive compression, ganglion/mass, trauma or other structural causes, so unexplained/progressive motor deficit should trigger medical/hand-surgery reassessment. Product-owner decision required: default primary vs rare/advanced.

## 3.2 Scapholunate / lunotriquetral ligament injury or carpal instability

Proposed role:

```text
clinician-established rare/advanced structural pathway
```

Possible subtypes:

```text
scapholunate_ligament_injury
lunotriquetral_ligament_injury
other_established_carpal_instability
```

Dorsal central pain, clicking, Watson/scaphoid-shift or other provocative findings do not establish instability. Established instability/structural lesion requires exact restrictions and specialist context.

## 3.3 ECU instability / subluxation

Distinct from ordinary ECU tendinopathy.

```text
formal_ECU_instability_or_subluxation
```

Snapping/pain alone does not establish subsheath tear or instability. Traumatic recurrent snapping with structural concern should not be treated as generic tendinopathy without assessment.

## 3.4 Complex regional pain syndrome — established diagnosis only

Proposed role:

```text
formal_CRPS_upper_limb_diagnosis
```

Display if formally established:

> Σύνδρομο σύνθετου περιοχικού πόνου (CRPS) άνω άκρου — λειτουργική αποκατάσταση

The utility must never infer CRPS from pain, swelling, color/temperature change or stiffness alone.

If established, rehabilitation is function-restoration oriented and often multidisciplinary. Candidate directions may include:

```text
graded functional use and movement
ROM and progressive task exposure
desensitization where appropriate
edema/hand-function management
graded motor imagery / mirror-based strategies where appropriate
coordination with pain/medical management
```

Evidence quality for individual rehabilitation techniques is variable; the generator must not prescribe one rigid CRPS protocol.

Product-owner decision required: expose as rare advanced pathway vs context only.

## 3.5 Inflammatory / crystal hand context

Directly selectable only when already established:

```text
known_rheumatoid_or_other_inflammatory_wrist_hand_involvement
known_psoriatic_hand_involvement
known_gout_or_crystal_disease_context
```

This is not inferred from swelling/pain/deformity. Acute hot swollen joint or tendon-sheath infection/inflammatory diagnostic uncertainty remains a medical reassessment issue.

## 3.6 Dupuytren disease

Not proposed as a routine physiotherapy primary diagnosis in v1.

Possible context:

```text
known_Dupuytren_contracture
post_Dupuytren_procedure_hand_therapy_context
```

The generator should not present physiotherapy as disease-modifying treatment for Dupuytren disease. Post-procedure therapy belongs under the exact postoperative/protocol context if used.

## 3.7 Ganglion cyst / mass

Medical/context only:

```text
known_ganglion_or_other_established_benign_mass_context
```

A wrist/hand mass, unexplained swelling or compressive neurological symptoms require appropriate medical assessment; the utility does not diagnose a ganglion from appearance alone and does not recommend aspiration as a physiotherapy technique.

## 3.8 Digital tendon / deformity-specific injuries

Candidate future/advanced entries if the product owner's workflow requires them:

```text
mallet_finger
boutonniere_injury_or_deformity
central_slip_injury
flexor_tendon_injury
extensor_tendon_injury
other_digit_tendon_injury
```

These require exact structural, healing and orthosis/protocol context. The current candidate does not yet promote them to default primary pathways.

---

# 4. Examination findings — selectable only when actually assessed

## 4.1 Pain / symptom location and behaviour

```text
radial wrist pain
thumb-base pain
dorsal central wrist pain
ulnar-sided wrist pain
volar wrist pain
palmar hand pain
digital/A1-pulley-region pain
specific joint pain
pain with grip
pain with pinch
pain with wrist extension/flexion
pain with forearm rotation
pain with axial loading / push-up
pain with thumb use
pain with typing/mouse/phone
pain with manual tools/work
night symptoms
```

## 4.2 Range of motion

```text
wrist flexion restricted
wrist extension restricted
radial deviation restricted
ulnar deviation restricted
forearm pronation restricted
forearm supination restricted
thumb CMC/MCP/IP ROM restriction
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
fine-motor/dexterity deficit
load intolerance without measured weakness
```

Pain-limited effort must not become structural weakness/rupture automatically.

## 4.4 Special/provocation-test findings

Secondary expander only:

```text
Finkelstein-type finding
Eichhoff-type finding
WHAT-type finding
CMC grind-type finding
Phalen/reverse-Phalen finding
Tinel at carpal tunnel
Durkan/carpal-compression finding
fovea sign
TFCC load/press-type finding
DRUJ ballottement/stability finding
ECU synergy/snapping finding
Watson/scaphoid-shift finding
thumb-MCP UCL stress finding
triggering/locking observed
other clinician-entered test
```

Tests are findings, not diagnoses.

---

# 5. Neurological model

Neurological status is component-specific and used only when relevant/assessed.

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

Hard invariant:

```text
not_assessed != normal
```

No global `no neurological deficit` wording is generated from missing components.

---

# 6. Safety / reassessment semantics

The utility provides prompts, not autonomous diagnoses or emergency decisions.

## 6.1 High-priority wrist / hand concerns

```text
acute trauma with unresolved fracture concern
acute trauma with unresolved dislocation/carpal instability concern
snuffbox/scaphoid-fracture concern after trauma
acute thumb-MCP UCL complete-tear/Stener concern
acute flexor/extensor tendon rupture or laceration concern
new inability to actively flex/extend a digit after injury
new/progressive median or ulnar motor deficit/atrophy
new neurovascular deficit after trauma
rapidly progressive swelling / severe structural concern
```

## 6.2 Infection / inflammatory concerns

```text
hot swollen joint / septic arthritis concern
wound/drainage/cellulitis
tendon-sheath infection concern
systemic illness with acute hand swelling
unexplained rapidly progressive atraumatic swelling
other infectious/inflammatory concern
```

## 6.3 Other material concerns

```text
suspected CRPS not yet established
severe unremitting/progressive non-mechanical pain
rapidly progressive atraumatic loss of hand/wrist function
unexplained mass with progressive neurological deficit
true mechanical locking/unstable carpal symptoms requiring structural assessment
other clinician concern
```

## 6.4 Safety state and disposition

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present

clinician_disposition when concern present:
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

Candidate fields:

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
pushing up from chair / floor
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

## WH1 De Quervain

Possible suggestions:

- reduce symptom irritability;
- improve tolerance of thumb/wrist use;
- restore functional ROM as needed;
- progressive APL/EPB-region load tolerance where appropriate;
- improve caregiving/work/grip function;
- self-management/load adaptation.

## WH2 CMC-1 OA

Possible suggestions:

- reduce pain during pinch/grip;
- improve functional pinch/grip capacity;
- improve thumb mobility/stability where relevant;
- optimize joint-protection/task strategies;
- improve ADLs/work;
- effective orthosis self-management where selected.

## WH3 hand OA

Possible suggestions:

- preserve/improve digit motion;
- maintain/improve hand strength;
- improve function/dexterity;
- joint-protection/assistive strategies;
- support self-management and activity.

## WH4 CTS

Possible suggestions:

- reduce nocturnal/positional symptom provocation;
- improve activity/position self-management;
- preserve hand function and strength where possible;
- monitor objective median motor/sensory status;
- support short-term symptom control while definitive management pathway remains clinician-led.

Do not promise reversal of established thenar atrophy or long-term disease modification from exercise/manual therapy.

## WH5 TFCC / ulnar wrist

Possible suggestions:

- reduce irritability under protected loading;
- restore safe wrist/forearm motion;
- improve grip/forearm strength;
- improve dynamic stability/proprioception;
- graded return to axial loading/work/sport within structural restrictions.

## WH6 tendon overuse

Possible suggestions:

- progressive subtype-specific tendon load tolerance;
- grip/forearm strength/endurance;
- movement/load adaptation;
- graded work/sport return.

## WH7 trigger digit

Possible suggestions:

- reduce triggering/irritability;
- improve functional digit motion;
- optimize orthosis use if selected;
- preserve grip/dexterity;
- avoid prolonged ineffective conservative care when severity progresses.

## WH8 thumb UCL

Possible suggestions:

- protected restoration of motion;
- progressive pinch/grip stability;
- thumb stabilizer strength;
- return to work/sport according to stability/protocol.

## WH9/WH10 trauma/post-op

Goals derive from established injury/procedure and explicit restrictions. Unrestricted ROM/strength is never assumed.

---

# 9. Rehabilitation / hand-therapy directions

## 9.1 Core active directions

```text
physiotherapy / hand-therapy assessment and individualized active rehabilitation
education / activity and load modification
therapeutic exercise
progressive tendon loading where appropriate
wrist/forearm strengthening where relevant
grip/pinch strengthening where relevant
thumb/intrinsic-hand strengthening where relevant
ROM/mobility exercise where restricted and safe
fine-motor/dexterity retraining where relevant
joint-protection and task adaptation where relevant
assistive-device strategy where relevant
proprioceptive/dynamic-stability work for ligament/TFCC contexts
graded return to work/gym/sport/manual tasks
home exercise programme where appropriate
```

No single loading mode, orthosis or protocol is mandatory for every wrist/hand condition.

## 9.2 Orthosis / splint options — condition sensitive

Orthosis is treated as a separate support category rather than a generic adjunct.

Possible selections:

```text
thumb-spica orthosis
CMC-support orthosis
neutral-wrist night orthosis
activity-specific wrist orthosis
trigger-digit orthosis
injury/protocol-specific protective orthosis
other clinician/hand-therapist-selected orthosis
```

Hard rule:

```text
orthosis suggested != orthosis automatically required
```

Condition-specific evidence matters:

- thumb CMC-1 OA: orthosis has meaningful evidence for pain/function support;
- De Quervain: thumb-spica immobilization has stronger evidence when used with injection than as stand-alone treatment;
- CTS: wrist orthosis may be used for short-term symptom management, but the 2024 CPG focuses on lack of proven long-term benefit for many conservative modalities;
- postoperative CTS: routine immobilization is not recommended after uncomplicated release;
- trigger digit: orthoses are a reasonable nonoperative option, with outcome varying by severity and design.

## 9.3 Optional adjunct expander

Candidate optional items:

```text
manual therapy / joint mobilization where appropriate
soft-tissue techniques
taping
dry needling — only for an appropriate selected myofascial/tendinous context
acupuncture — clinician-selected only; evidence-sensitive
heat/thermal strategy for hand OA where clinically appropriate
```

### Manual therapy

May be selected for a relevant mobility/pain impairment, especially within multimodal CMC/hand rehabilitation, but must not be presented as a structural cure or as proven long-term disease-modifying CTS treatment.

### Dry needling

No universal wrist/hand indication is frozen. If exposed, it should require an actual myofascial/tendinous target plus the established cross-region competence/availability safeguard. It should not be presented as standard treatment for CTS, TFCC injury or OA.

### Acupuncture

Could remain an optional clinician-selected adjunct for selected wrist/hand pain conditions if this matches real workflow, but the evidence differs substantially by diagnosis. Product-owner confirmation required before freeze.

### ESWT

Not proposed as a standard wrist/hand adjunct in v1:

- De Quervain network evidence is too sparse/high-risk for routine recommendation;
- CTS 2024 AAOS evidence does not support shockwave as a proven long-term nonoperative treatment;
- no cross-condition wrist/hand ESWT default is justified.

Product owner may still request a narrowly defined condition-specific option if it reflects actual practice, but it would remain evidence-sensitive.

### Therapeutic ultrasound

Not presented as a standard evidence-backed treatment for CTS or general wrist/hand pathology.

---

# 10. Shared fracture / post-immobilization boundary

Wrist/hand fractures route to the shared fracture profile rather than duplicate fracture logic here.

Regional routes include:

```text
distal radius fracture
distal ulna fracture
scaphoid fracture
other carpal fracture
metacarpal fracture
phalangeal fracture
other wrist/hand fracture
```

Future shared required context:

```text
fracture site
date/phase
treatment
healing/stability status
immobilization/orthosis status
ROM restrictions
loading/use restrictions
orthopaedic/hand-surgeon instructions
```

```text
fracture route + unresolved healing/loading context
→ warning
→ no unrestricted routine rehabilitation wording
```

Scaphoid-specific concern after acute trauma remains a safety/reassessment issue until adequately assessed.

---

# 11. Deterministic consistency rules

```text
WH1 + one De Quervain provocation test only
→ do not infer definitive De Quervain diagnosis

WH1 + generated wording says PT is evidence-preferred first-line over medical management
→ invalid

WH2 + imaging OA only + no clinician-established symptomatic diagnosis
→ do not auto-assert symptomatic CMC-1 OA

WH3 + swollen painful joints + no established OA diagnosis
→ do not infer OA; preserve inflammatory/infectious differential

WH4 formal CTS diagnosis != yes
→ presentation wording only unless clinician explicitly supplies diagnostic context

WH4 + upper-limb neurodynamic test positive
→ do not use as diagnostic proof of CTS

WH4 + progressive thenar weakness/atrophy
→ prominent medical/hand-surgery reassessment prompt

uncomplicated carpal-tunnel release + no specific rehab indication
→ do not auto-generate supervised postoperative therapy

WH5 ulnar-sided pain + fovea/load test only
→ do not infer TFCC tear

WH5 + DRUJ instability / foveal full-tear context
→ specialist/restriction prompt before generic strengthening

WH6 + ECU snapping/instability context
→ do not collapse into ordinary ECU tendinopathy

WH7 severe fixed locking/contracture or progressive loss
→ medical reassessment prompt

WH8 + acute instability/Stener concern unresolved
→ specialist prompt; no unrestricted rehabilitation

WH9 post-traumatic + unresolved fracture/dislocation/tendon rupture/major instability concern
→ safety prompt

WH10 postoperative + missing procedure/protocol/restrictions
→ warning

possible CRPS symptoms without established diagnosis
→ do not auto-label CRPS; prompt clinician reassessment

formal CRPS selected
→ functional restoration/multidisciplinary wording; no rigid protocol

adjunct selected + no active rehabilitation direction
→ warning

dry needling selected
→ competence/availability reminder

ESWT selected without a future explicitly supported wrist/hand indication
→ evidence/context warning

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological component
→ never generate normal wording
```

---

# 12. Generated wording examples

## 12.1 De Quervain presentation

Without formal diagnosis:

> Κερκιδικός πόνος του [side] καρπού/βάσης αντίχειρα με χαρακτηριστικά 1ου ραχιαίου διαμερίσματος, [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική/hand-therapy αξιολόγηση και αποκατάσταση με έμφαση στην προσαρμογή φορτίου, τη λειτουργική κινητικότητα και τη σταδιακή αποκατάσταση της ανοχής στη χρήση του αντίχειρα/καρπού. [Selected orthosis/adjunct only if confirmed.]

If formally established, `De Quervain` may be used explicitly.

## 12.2 Thumb CMC-1 OA

> Οστεοαρθρίτιδα βάσης αντίχειρα / ριζάρθρωση CMC-1 του [side] χεριού, με [selected findings] και περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη hand therapy με άσκηση/ενδυνάμωση όπου ενδείκνυται, εκπαίδευση προστασίας άρθρωσης και λειτουργικών στρατηγικών και [selected CMC orthosis if selected].

## 12.3 Carpal-tunnel presentation without formal diagnosis

> Συμπτωματολογία μέσου νεύρου στην περιοχή του [side] καρπιαίου σωλήνα, με [selected subjective symptoms] και [selected objective findings only if actually assessed]. Παρακαλώ για συντηρητική αξιολόγηση/αντιμετώπιση με εκπαίδευση και τροποποίηση επιβαρυντικών θέσεων/δραστηριοτήτων, βραχυπρόθεσμη υποστήριξη με [selected orthosis if selected] και παρακολούθηση τυχόν αντικειμενικών νευρολογικών μεταβολών.

No `normal motor/sensory status` is added unless actually assessed.

## 12.4 TFCC-related presentation

Without formal structural diagnosis:

> Ωλένιος πόνος του [side] καρπού με χαρακτηριστικά TFCC/ωλένιας πλευράς και [selected findings], με λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη αποκατάσταση με προστατευμένη/σταδιακή φόρτιση, βελτίωση κινητικότητας και δύναμης/σταθερότητας όπου ενδείκνυται, σύμφωνα με την καταγεγραμμένη κατάσταση της DRUJ και τυχόν περιορισμούς.

Only a clinician-established TFCC diagnosis may be named definitively.

## 12.5 Trigger digit

> [Clinician-established trigger finger/thumb] του [digit/side], με [selected triggering/locking/function findings]. Παρακαλώ για hand-therapy αξιολόγηση και συντηρητική αντιμετώπιση με [selected orthosis if selected], προσαρμογή επιβαρυντικών δραστηριοτήτων και λειτουργική αποκατάσταση. Επανεκτίμηση αν παραμένει σημαντικό locking/contracture ή λειτουργική επιδείνωση.

## 12.6 Postoperative wrist/hand

If retained after product-owner confirmation:

> Μετεγχειρητική αποκατάσταση [procedure] του [side] καρπού/χεριού, επέμβαση [date if entered]. Παρακαλώ για hand therapy σύμφωνα με το διαθέσιμο χειρουργικό πρωτόκολλο και τους καταγεγραμμένους περιορισμούς σε κινητικότητα, φόρτιση, ενδυνάμωση, χρήση του άκρου και orthosis.

---

# 13. Evidence-governance boundary

Stable structural decisions proposed for wrist/hand:

```text
provocation/special test != diagnosis
subjective paresthesia != objective neurological deficit
pain-limited effort != tendon rupture
radial-sided wrist pain != automatically De Quervain
ulnar-sided wrist pain != automatically TFCC tear
incidental imaging finding != automatically symptomatic diagnosis
CTS neurodynamic test != diagnostic proof
uncomplicated carpal-tunnel release != automatic supervised therapy
CMC-1 OA and interphalangeal hand OA remain distinct phenotypes
thumb UCL instability/complete tear concern != routine unrestricted rehab
ECU instability != ordinary ECU tendinopathy
possible CRPS features != formal CRPS diagnosis
fractures route to shared fracture/post-immobilization profile
active/function-oriented rehabilitation remains core where appropriate
orthoses are condition-sensitive supports rather than universal defaults
adjunct techniques remain optional
```

Current evidence anchors reviewed for this candidate:

- 2024 AAOS Management of Carpal Tunnel Syndrome CPG;
- 2023 JAMA Network Open systematic review/network meta-analysis of De Quervain treatments;
- 2024 systematic review/meta-analysis of physical-therapy interventions for thumb CMC OA;
- 2024 network meta-analysis of nonoperative CMC-1 OA treatments;
- 2025 multicenter RCT of orthosis + exercise vs orthosis alone for CMC-1 OA;
- ACR/AF hand-OA guidance and EULAR hand-OA management/classification framework;
- 2024 systematic review of nonoperatively treated traumatic TFCC tears;
- 2024 review/update of TFCC injuries in athletes;
- current wrist-tendinopathy/ECU rehabilitation literature;
- systematic review of trigger-digit orthoses and 2025 randomized conservative-treatment trial;
- European/hand-surgery literature for Guyon's canal syndrome;
- 2022 CRPS Practical Diagnostic and Treatment Guidelines, 5th edition, and subsequent rehabilitation reviews.

Evidence-sensitive items to refresh immediately before CU-2 implementation:

```text
exact De Quervain rehabilitation/loading role after injection/immobilization
CMC-1 orthosis type/duration and exercise dosage
hand-OA exercise/orthosis dosing
CTS short-term splinting and glide wording vs long-term outcome claims
TFCC immobilization/loading progression by lesion/stability
trigger-digit orthosis design/duration
thumb-UCL rehabilitation thresholds/protocols
ECU/FCR/FCU/intersection rehabilitation evidence
CRPS rehabilitation evidence
postoperative hand-therapy indications/protocols
```

---

# 14. Product-owner decisions required before freeze

1. **Trigger finger / thumb:** do you refer these often enough for hand therapy to remain a default primary pathway, or should it be medical/context only?
2. **Guyon's canal / ulnar neuropathy at wrist:** default primary neurological pathway or rare/advanced because you see it infrequently?
3. **Post-operative wrist/hand:** do you see tendon repairs, CMC surgery, Dupuytren procedures, ligament repairs or other hand surgery often enough to keep a default post-op pathway?
4. **Thumb UCL / skier's-gamekeeper's thumb:** keep as a default primary pathway or rare/post-traumatic advanced route?
5. **Scapholunate/lunotriquetral instability:** common enough in your referrals to promote from rare/advanced to a primary pathway?
6. **CRPS:** expose as an established-diagnosis advanced pathway because you refer these patients for function restoration, or keep only as safety/context?
7. **Mallet finger / boutonniere / central-slip / flexor-extensor tendon injuries:** do these need dedicated hand-therapy pathways in your workflow, or should they remain advanced tendon/post-op entries?
8. **Acupuncture and dry needling:** retain as optional wrist/hand adjuncts for selected pain/tendon/myofascial contexts, or omit because you do not use them here?
9. **ESWT:** current recommendation is to omit it from default wrist/hand options, including De Quervain, because the evidence is not strong enough for routine use. Do you want any narrowly defined wrist/hand ESWT option despite that?
10. Any common wrist/hand entity missing from the candidate, particularly one you routinely send to a physiotherapist/hand therapist?

This file remains a **DESIGN CANDIDATE / NOT FROZEN** until those real-workflow decisions are resolved. Runtime implementation remains unauthorized.
