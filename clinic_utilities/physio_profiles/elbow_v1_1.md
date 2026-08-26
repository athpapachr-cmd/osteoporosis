# Elbow Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-26.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful elbow referral choices while preserving diagnosis-vs-finding separation, neurological/tendon-rupture safety semantics, active rehabilitation, and physiotherapist autonomy.
> **Supersedes as active elbow design:** `clinic_utilities/physio_profiles/elbow_v1.md`.
> **Prior frozen regional profiles:** `cervical_v1_1.md`, `lumbar_v1_1.md`, `shoulder_v1_1.md`.

---

# 1. Core design contract

```text
PRIMARY CLINICAL PATHWAY
+
ACTUAL FINDINGS / MODIFIERS
+
FUNCTIONAL IMPACT
+
SAFETY / REASSESSMENT CONTEXT
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
subjective paresthesia != objective sensory deficit
pain-limited effort != structural weakness or tendon tear
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

The utility structures a referral and checks internal consistency. It must not autonomously diagnose tendinopathy, nerve entrapment, tendon rupture, instability, bursitis, inflammatory disease or intra-articular pathology.

---

# 2. Frozen primary elbow pathways

## E1 — Lateral elbow tendinopathy / lateral epicondylalgia

Structured key:

```text
lateral_elbow_tendinopathy
```

Display:

> Πλάγια επικονδυλαλγία / τενοντοπάθεια εκτεινόντων του καρπού (tennis elbow)

Preferred terminology is `lateral elbow tendinopathy` / `lateral epicondylalgia` rather than assuming a histological inflammatory `epicondylitis` process.

Useful context:

```text
lateral epicondyle/common-extensor-region pain
pain with gripping
pain with resisted wrist extension
pain with lifting/carrying with forearm pronated
pain with racquet/manual-tool/work activity
symptom irritability
work/sport load context
```

Optional clinician assertion:

```text
formal_lateral_elbow_tendinopathy_diagnosis: yes / no / not_stated
```

Cozen/Mill/Maudsley-type tests, local tenderness, pain-free grip loss or imaging abnormalities may support the clinical picture but do not independently create the diagnosis.

```text
lateral elbow pain != automatically lateral elbow tendinopathy
```

Radial/PIN, cervical, intra-articular and instability causes remain relevant when the presentation is atypical.

## E2 — Medial elbow tendinopathy / medial epicondylalgia

Structured key:

```text
medial_elbow_tendinopathy
```

Display:

> Έσω επικονδυλαλγία / τενοντοπάθεια καμπτήρων-πρηνιστών (golfer's elbow)

Useful context:

```text
medial epicondyle/flexor-pronator origin pain
pain with resisted wrist flexion
pain with resisted forearm pronation
pain with gripping/lifting
throwing/golf/manual-work load
coexisting ulnar-neural symptoms if present
```

Optional clinician assertion:

```text
formal_medial_elbow_tendinopathy_diagnosis: yes / no / not_stated
```

Medial pain plus paresthesia must preserve tendon and ulnar-neural findings separately. No single loading mode is globally mandatory.

## E3 — Ulnar neuropathy at the elbow / cubital tunnel syndrome

Structured key:

```text
ulnar_neuropathy_at_elbow
```

Default wording without formal diagnosis:

> Συμπτωματολογία ωλενίου νεύρου στην περιοχή του αγκώνα

Optional clinician assertion:

```text
formal_ulnar_neuropathy_cubital_tunnel_diagnosis: yes / no / not_stated
```

If `yes`, generated wording may state:

> Ωλένια νευροπάθεια στον αγκώνα / σύνδρομο κυβοειδούς σωλήνα

Subjective symptoms:

```text
paresthesia ring/small finger
numbness ring/small finger
medial elbow discomfort
symptoms with sustained elbow flexion
night symptoms
symptoms with leaning/direct pressure over medial elbow
```

Objective findings only when actually assessed:

```text
ulnar-distribution sensory deficit
intrinsic-hand weakness
grip/pinch weakness
visible intrinsic atrophy
ulnar-clawing or other established motor sign
```

Provocation/context findings:

```text
Tinel over cubital tunnel
elbow-flexion provocation
ulnar neurodynamic finding
ulnar-nerve subluxation/snapping if actually observed
```

Hard rules:

```text
paresthesia alone != objective neurological deficit
positive Tinel/elbow-flexion test != definitive diagnosis
```

Progressive motor weakness, intrinsic atrophy or materially worsening objective deficit requires clinician reassessment.

## E4 — Posterior interosseous nerve / supinator syndrome

Structured key:

```text
posterior_interosseous_nerve_supinator_syndrome
```

Default wording if no formal diagnosis is asserted:

> Συμπτωματολογία εν τω βάθει κλάδου του κερκιδικού / οπισθίου μεσόστεου νεύρου στην περιοχή του υπτιαστή

Optional clinician assertion:

```text
formal_pin_supinator_syndrome_diagnosis: yes / no / not_stated
```

If `yes`, generated wording may state:

> Σύνδρομο οπισθίου μεσόστεου νεύρου / supinator syndrome

This pathway is deliberately separated from pain-predominant radial tunnel presentation.

Useful objective findings when actually assessed:

```text
finger-extension weakness
thumb-extension weakness
radial/PIN-pattern motor deficit
wrist-extension pattern abnormality if clinically relevant
muscle atrophy if present
```

Possible context:

```text
proximal dorsal/radial forearm symptoms
provocation with resisted supination where relevant
compression/entrapment around supinator/arcade of Frohse if established
EMG/NCS/imaging context if available
```

Hard rules:

```text
lateral forearm pain alone != PIN/supinator syndrome
radial-tunnel provocation alone != motor neuropathy
objective radial/PIN motor deficit != routine epicondylalgia
```

New or progressive motor weakness requires medical/specialist reassessment semantics.

## E5 — Distal biceps tendinopathy / established partial tear — conservative pathway

Structured key:

```text
distal_biceps_tendon_disorder_nonoperative
```

Display:

> Τενοντοπάθεια / επιβεβαιωμένη μερική ρήξη περιφερικού τένοντα δικεφάλου — συντηρητική αποκατάσταση

Subtype:

```text
distal_biceps_tendinopathy
confirmed_partial_distal_biceps_tear
other_established_distal_biceps_disorder
```

For partial tear:

```text
clinician/imaging-established tear
traumatic vs degenerative onset if known
current nonoperative management decision
activity-demand context
restrictions if any
```

Useful findings:

```text
anterior antecubital pain
distal-biceps tenderness
pain with resisted elbow flexion
pain with resisted supination
objective supination weakness
objective flexion weakness
```

Acute eccentric injury with bruising/deformity and marked new supination/flexion weakness or unresolved rupture concern requires timely reassessment; complete rupture must not be silently routed through tendinopathy.

## E6 — Elbow osteoarthritis / degenerative painful stiffness

Structured key:

```text
elbow_osteoarthritis_degenerative_stiffness
```

Display:

> Οστεοαρθρίτιδα αγκώνα / εκφυλιστική επώδυνη δυσκαμψία

Use when the clinician considers the degenerative diagnosis established.

Useful context/findings:

```text
pain with loading/use
flexion deficit
extension deficit
pronation/supination restriction
stiffness after rest
crepitus if relevant
mechanical end-range symptoms
functional loss in reach, grooming, feeding or work
imaging OA context optional
```

Imaging OA alone does not prove symptom causation. True mechanical locking, rapidly progressive loss of motion or unresolved loose-body/intra-articular concern triggers reassessment.

## E7 — Elbow ligament injury / instability rehabilitation

Structured key:

```text
elbow_ligament_instability_rehabilitation
```

Display:

> Κάκωση συνδέσμων / αστάθεια αγκώνα — αποκατάσταση

Clinician-established subtype:

```text
ulnar_medial_collateral_ligament_injury
lateral_collateral_ligament_complex_injury
posterolateral_rotatory_instability
other_established_elbow_instability
```

Required context where relevant:

```text
traumatic vs repetitive/throwing onset
partial vs complete injury if established
operative vs nonoperative management
current specialist/stability context
brace/ROM/loading restrictions
throwing/sport demands
```

Stress/provocation tests remain findings and do not create the diagnosis. No universal return-to-sport timeline is generated.

## E8 — Post-traumatic elbow pain / stiffness after assessed injury

Structured key:

```text
post_traumatic_elbow_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία αγκώνα μετά από αξιολογημένη κάκωση

Use only after important unresolved fracture, dislocation, major tendon rupture or unstable ligament injury has been addressed as required.

Required context:

```text
injury/date or phase
established structural diagnosis if any
imaging/orthopaedic context if relevant
immobilization history
current ROM/loading/use restrictions
```

The utility must not label an unassessed traumatic elbow as a `simple sprain`.

---

# 3. Rare / advanced elbow pathways and secondary entities

## 3.1 Radial tunnel syndrome — secondary/coexisting clinician-entered context

The product owner rarely sees radial tunnel syndrome, so it is not a default primary pathway.

Possible entries:

```text
formal_radial_tunnel_syndrome
radial_tunnel_pain_presentation_not_formally_diagnosed
```

Radial tunnel remains distinguishable from E4 PIN/supinator syndrome:

```text
radial tunnel = predominantly pain-predominant presentation
PIN/supinator syndrome = motor-neuropathy pathway when formally established / objectively relevant
```

The literature uses inconsistent nomenclature and some authors conceptualize a spectrum; the generator therefore preserves the actual clinician diagnosis and actual objective deficit rather than inferring labels.

## 3.2 Distal triceps disorder — rare selectable myotendinous entity

Structured secondary/advanced key:

```text
distal_triceps_tendon_disorder
```

Possible clinician-entered subtype:

```text
distal_triceps_tendinopathy
confirmed_partial_distal_triceps_tear
other_established_distal_triceps_disorder
```

Useful findings:

```text
posterior elbow/distal-triceps pain
pain with resisted extension
pain with pushing/pressing
objective extension weakness
local insertion tenderness
```

Acute injury with palpable defect/deformity or marked extension weakness and unresolved rupture concern requires reassessment.

## 3.3 Anconeus-related pain / injury — rare selectable myotendinous entity

Structured key:

```text
anconeus_myotendinous_pain_or_injury
```

Use only when the clinician has actually localized/diagnosed an anconeus-related presentation. It is not inferred from generic posterolateral elbow pain.

Useful context/findings:

```text
posterolateral elbow pain localized to anconeus region
local tenderness
pain with relevant extension/stabilization loading
sport/gym/manual-load context
established muscle/tendon injury if known
```

`anconeus epitrochlearis` is not the same entity. It is an anatomic variant that may rarely relate to ulnar nerve compression and is not automatically pathogenic when seen on imaging.

## 3.4 Olecranon bursitis — medical/context only, not a primary physio referral pathway

The product owner does not routinely refer olecranon bursitis to physiotherapy.

Selectable context only when relevant:

```text
known_olecranon_bursitis_context
known_gout_or_inflammatory_bursal_context
```

Safety rule:

```text
posterior swelling + fever/warmth/erythema/cellulitis/wound/drainage/systemic illness/unresolved infection concern
→ medical reassessment
→ no routine physiotherapy reassurance
```

The generator does not recommend aspiration or injection.

## 3.5 Post-operative elbow — rare advanced route, not active default MVP pathway

Because the product owner sees postoperative elbow patients only rarely, this is not shown among the default primary choices. A future/advanced route may expose:

```text
postoperative_elbow_rehabilitation
```

Required context before any generated wording:

```text
operation/procedure
operation date
surgeon/protocol when available
immobilization/brace status
ROM restrictions
loading/strengthening restrictions
weight-bearing/use restrictions
return-to-sport/work constraints
```

No generic postoperative timeline may be invented.

## 3.6 Inflammatory / crystal disease context

Directly selectable only when already established:

```text
known_rheumatoid_or_other_inflammatory_elbow_involvement
known_gout_or_crystal_disease_context
```

Acute hot swollen joint or unresolved septic/inflammatory differential remains a reassessment issue.

---

# 4. Examination findings — selectable only when actually assessed

## 4.1 Pain / symptom behaviour

```text
lateral elbow pain
medial elbow pain
anterior/antecubital pain
posterior elbow pain
posterolateral/anconeus-region pain
proximal dorsal/radial forearm pain
pain with gripping
pain with lifting/carrying
pain with pushing/pressing
pain with pulling/curling
pain with pronation/supination
pain with throwing/racquet sport
pain with manual tools/work
night symptoms
pressure intolerance over medial/posterior elbow
```

## 4.2 Range of motion

```text
elbow flexion restricted
elbow extension restricted
forearm pronation restricted
forearm supination restricted
painful active ROM
painful passive ROM
mechanical end-range symptoms
```

## 4.3 Strength / load tolerance

```text
pain-free grip reduced
maximum grip reduced if assessed
painful resisted wrist extension
painful resisted wrist flexion
painful resisted pronation
painful resisted supination
objective elbow-flexion weakness
objective elbow-extension weakness
objective wrist/finger/thumb-extension weakness
intrinsic-hand weakness
load intolerance without measured weakness
```

## 4.4 Special/provocation-test findings

Secondary expander only:

```text
Cozen-type finding
Mill-type finding
Maudsley/middle-finger-resistance finding
medial flexor-pronator provocation
Tinel at cubital tunnel
elbow-flexion ulnar-nerve provocation
ulnar neurodynamic finding
Hook-test or other distal-biceps finding
valgus-stress finding
moving-valgus-stress/milking finding
varus/PLRI provocation finding
radial-tunnel/PIN provocation finding
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
ulnar-distribution sensory status
intrinsic-hand motor status
radial/PIN motor status
median-nerve status where relevant
grip/pinch status
atrophy: present / absent / not_assessed
```

```text
not_assessed != normal
```

No global `no neurological deficit` wording is generated from missing components.

---

# 6. Safety / reassessment semantics

High-priority concerns:

```text
acute trauma with unresolved fracture concern
acute trauma with unresolved dislocation/instability concern
acute distal-biceps rupture concern
acute distal-triceps rupture concern
new/progressive objective ulnar/radial/PIN/median motor deficit
new neurovascular deficit after trauma/dislocation
true locked elbow / major unresolved mechanical block
rapidly progressive post-traumatic swelling or urgent structural concern
```

Other material concerns:

```text
hot swollen joint / septic arthritis concern
olecranon swelling with unresolved infection concern
fever/cellulitis/wound/drainage around elbow
systemic/inflammatory/malignancy concern
severe unremitting/progressive non-mechanical pain
rapidly progressive atraumatic loss of motion
other clinician concern
```

Safety state:

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

Clinician disposition when concern present:

```text
reviewed_and_appropriate_to_proceed
imaging/medical reassessment arranged
orthopaedic/specialist pathway underway
urgent/same-day assessment arranged
routine physiotherapy deferred
other
```

No default `no red flags`, `no rupture`, `stable elbow`, `non-septic bursitis` or `neurovascularly intact` wording is generated from missing information.

---

# 7. Functional limitations

```text
grip/handshake
opening jars/containers
lifting/carrying
pushing/pressing
pulling/curling
turning a key/screwdriver/forearm rotation
writing/typing/mouse use
manual tools/repetitive work
feeding/grooming/reaching hand to mouth/head
sleep due to elbow flexion/pressure
racquet sport
golf
throwing/overhead sport
gym/weight training
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
improve pain-free grip and load tolerance
improve relevant muscle/tendon strength/endurance
restore tolerated functional ROM where impaired
reduce neural provocation where relevant
preserve/improve neurological function where appropriate
improve dynamic stability when instability pathway selected
graded return to work/gym/racquet/throwing demands
improve self-management and load modification
```

Condition cautions:

- cubital tunnel/PIN: do not promise reversal of established atrophy or motor deficit;
- partial tendon tear: respect established restrictions and do not promise structural healing;
- instability: criterion-based progression within restrictions;
- OA/stiffness: improve function/mobility rather than promise structural reversal;
- post-traumatic/rare post-op: actual injury/procedure restrictions outrank generic goals.

---

# 9. Rehabilitation directions

Core active directions:

```text
physiotherapy assessment and individualized active rehabilitation
education / activity and load modification
therapeutic exercise
progressive tendon loading where appropriate
wrist-extensor strengthening/endurance for lateral tendinopathy
flexor-pronator strengthening/endurance for medial tendinopathy
grip strengthening where relevant
elbow flexor/extensor and forearm-rotation strengthening where relevant
ROM/mobility exercise where restricted and safe
proximal shoulder/scapular strengthening only when an actual impairment exists
dynamic-stability/proprioceptive work for ligament/instability contexts
graded return to work/gym/racquet/throwing demands
home exercise programme where appropriate
```

No single loading mode or fixed protocol is mandatory.

Neural directions may include:

```text
education/activity modification
reduction of prolonged provocative elbow flexion/direct cubital-tunnel pressure
night-position modification/splinting where appropriate
selective neural-mobility/neurodynamic work when clinically appropriate
strength/function work according to actual objective deficit and irritability
```

Optional adjuncts:

```text
manual therapy / joint mobilization
soft-tissue techniques
dry needling
acupuncture
taping
counterforce brace or wrist-support orthosis for activity-related epicondylalgia
ESWT for lateral or medial epicondylalgia when clinically selected
```

### Dry needling

Optional, particularly for lateral epicondylalgia/myofascial findings. Competence/availability safeguard remains mandatory.

### Acupuncture

Retained as a clinician-selectable adjunct for lateral/medial elbow pain when clinically appropriate and practitioner competence exists. It must not displace active rehabilitation or be presented as mandatory.

### ESWT

Retained by product-owner preference for lateral and medial epicondylalgia.

```text
ESWT = optional evidence-sensitive adjunct
!= default
!= mandatory
!= universally superior
```

Current evidence is heterogeneous; some recent syntheses support pain benefit in lateral epicondylalgia while functional superiority and comparator-specific benefit are inconsistent. Evidence for medial epicondylalgia is less mature. Production wording must therefore remain cautious and be refreshed before CU-2.

### Counterforce brace / wrist orthosis

May be exposed as short-term/activity-specific support rather than a required long-term treatment.

### Therapeutic ultrasound

Not presented as a standard evidence-backed elbow treatment.

---

# 10. Shared fracture / post-immobilization boundary

Elbow fractures route to the shared fracture profile:

```text
radial head/neck fracture
olecranon/proximal ulna fracture
distal humerus fracture
coronoid fracture
other elbow-region fracture
```

Future shared required context:

```text
fracture site
date/phase
treatment
healing/stability status
immobilization/brace status
ROM restrictions
loading/use restrictions
orthopaedic instructions
```

```text
fracture route + unresolved healing/loading context
→ warning
→ no unrestricted routine rehabilitation wording
```

---

# 11. Deterministic consistency rules

```text
E1 + one provocation test only
→ do not infer definitive tendinopathy diagnosis

E1 + objective radial/PIN deficit
→ prompt against collapsing into epicondylalgia

E2 + ulnar paresthesia
→ preserve tendon and neural findings separately

E3 formal cubital-tunnel diagnosis != yes
→ presentation wording only

E3 + progressive motor weakness/atrophy
→ prominent reassessment prompt

E4 formal PIN/supinator diagnosis != yes
→ symptom/deficit wording only

E4 + new/progressive finger/thumb-extension weakness
→ medical/specialist reassessment prompt

radial tunnel pain-only context + no objective motor deficit
→ do not auto-label PIN motor syndrome

E5 partial distal-biceps tear + no established diagnosis/imaging context
→ warning

acute biceps injury + rupture concern
→ reassessment before routine tendon rehab

rare distal-triceps pathway + acute marked extension weakness/defect
→ reassessment

anconeus-region pain only
→ do not infer anconeus injury automatically

olecranon swelling + unresolved infection concern
→ no routine physio reassurance

E7 ligament/instability + no established diagnosis/context
→ do not infer instability from stress test alone

E8 post-traumatic + unresolved fracture/dislocation/major tendon/instability concern
→ safety prompt

rare post-op route + missing procedure/protocol/restrictions
→ warning

adjunct selected + no active rehabilitation direction
→ warning

dry needling selected
→ competence/availability reminder

ESWT selected outside lateral/medial epicondylalgia
→ warning

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological component
→ never generate normal wording
```

---

# 12. Generated wording examples

## Lateral elbow tendinopathy

> Πλάγια επικονδυλαλγία / τενοντοπάθεια εκτεινόντων του [side] καρπού με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με προοδευτική φόρτιση των εκτεινόντων, βελτίωση της δύναμης/αντοχής και της λαβής όπου ενδείκνυται, τροποποίηση φορτίου και σταδιακή επάνοδο στις απαιτούμενες δραστηριότητες. [Selected adjuncts if present.]

## Medial elbow tendinopathy

> Έσω επικονδυλαλγία / τενοντοπάθεια καμπτήρων-πρηνιστών του [side] άνω άκρου με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη ενεργητική αποκατάσταση με προοδευτική φόρτιση του flexor-pronator μηχανισμού, βελτίωση της λαβής/αντοχής και σταδιακή επιστροφή σε [selected work/sport]. [Selected adjuncts if present.]

## Ulnar-neuropathy presentation without formal diagnosis

> Συμπτωματολογία ωλενίου νεύρου στην περιοχή του [side] αγκώνα με [selected subjective symptoms] και [selected objective findings only if actually assessed]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και συντηρητική αποκατάσταση με εκπαίδευση/τροποποίηση επιβαρυντικών θέσεων και δραστηριοτήτων και επιλεγμένες νευροδυναμικές/λειτουργικές παρεμβάσεις όπου ενδείκνυνται. [Monitoring/reassessment criteria if selected.]

## PIN / supinator syndrome

Only when clinician-established:

> Σύνδρομο οπισθίου μεσόστεου νεύρου / supinator syndrome του [side] άνω άκρου, με [selected objective motor findings and symptoms]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση/αποκατάσταση σύμφωνα με την καθορισμένη νευρολογική διάγνωση, με παρακολούθηση της κινητικής λειτουργίας και επανεκτίμηση σε περίπτωση επιδείνωσης.

## Distal biceps conservative pathway

> [Clinician-established distal biceps tendinopathy / partial tear] του [side] άνω άκρου, για συντηρητική αποκατάσταση, με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για προοδευτική αποκατάσταση της φόρτισης, της κάμψης/υπτιασμού και της λειτουργικής δύναμης σύμφωνα με την κλινική εικόνα και τους καταγεγραμμένους περιορισμούς.

---

# 13. Evidence-governance boundary

Stable structural decisions frozen in v1.1:

```text
provocation/special test != diagnosis
subjective neural symptom != objective neurological deficit
pain-limited effort != tendon tear
lateral epicondylalgia != radial tunnel/PIN/cervical/intra-articular pathology
medial tendinopathy != ulnar neuropathy/UCL pathology
radial tunnel pain presentation != PIN/supinator motor syndrome
complete distal biceps/triceps rupture concern != routine tendinopathy referral
olecranon bursitis is not a routine primary physio pathway for this user
fractures route to shared fracture/post-immobilization profile
postoperative elbow is rare/advanced rather than a default MVP pathway
active/function-oriented rehabilitation remains core
adjunct techniques remain optional
```

Evidence anchors reviewed for this freeze include:

- JOSPT/APTA 2022 Lateral Elbow Pain and Muscle Function Impairments CPG;
- recent systematic reviews/meta-analyses on exercise, dry needling, acupuncture and ESWT for epicondylalgia;
- 2024 systematic review of radial tunnel diagnostic heterogeneity;
- 2024–2025 literature on radial nerve/PIN nomenclature and pain-to-motor compression spectrum;
- 2024 review of electrodiagnosis/ultrasound for ulnar entrapment at the elbow and current conservative-treatment syntheses;
- recent distal biceps/triceps and elbow-instability/postoperative rehabilitation literature;
- literature on anconeus epitrochlearis as a usually incidental anatomic variant that can occasionally coexist with ulnar compression.

Evidence-sensitive production wording to refresh before CU-2:

```text
exact loading dosage/progression
relative eccentric/isometric/concentric programme role
dry-needling effect estimates
acupuncture effect estimates
ESWT indication/comparator evidence, especially medial epicondylalgia
splint/orthosis recommendations
cubital-tunnel conservative treatment
PIN/supinator conservative-management evidence
ligament/throwing return-to-sport criteria
postoperative protocols if advanced route is exposed
```

---

# 14. Freeze decision

Product-owner decisions incorporated 2026-08-26:

- radial tunnel syndrome is uncommon in real workflow and remains secondary/context rather than a default primary pathway;
- PIN / supinator syndrome is retained as a distinct clinician-established neurological pathway because objective motor involvement has different semantics from pain-predominant radial tunnel presentation;
- olecranon bursitis is not a routine physiotherapy referral and is removed from default primary pathways, while infection safety/context remains available;
- postoperative elbow is rare and therefore an advanced/future-access route rather than a default active MVP pathway;
- ESWT is retained for lateral and medial epicondylalgia as an optional evidence-sensitive adjunct;
- acupuncture is retained as an optional adjunct;
- distal triceps and anconeus presentations are seen but rarely and remain selectable rare myotendinous entities rather than top-level default pathways;
- anconeus epitrochlearis is kept distinct from ordinary anconeus pain/injury and is not treated as automatically pathological;
- fractures remain routed to the shared fracture/post-immobilization profile.

This file is the frozen elbow clinical/content design for CU-1. Runtime implementation remains unauthorized.
