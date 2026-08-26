# Elbow Physiotherapy Referral Profile v1 — CU-1 DESIGN CANDIDATE

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful elbow referral choices while preserving diagnosis-vs-finding separation, neurological/tendon-rupture safety semantics, active rehabilitation, and physiotherapist autonomy.
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

Inherited hard invariants:

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

The generator may organize clinical information and issue consistency prompts. It must not autonomously diagnose tendinopathy, nerve entrapment, tendon rupture, instability, bursitis, inflammatory disease or intra-articular pathology.

---

# 2. Proposed primary elbow pathways

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

Important differential boundary:

```text
lateral elbow pain
!= automatically lateral elbow tendinopathy
```

Cervical/radial-neural/intra-articular/instability causes remain possible when the pattern is atypical.

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

A medial pain provocation test does not establish tendinopathy, and medial pain plus paresthesia must not be collapsed into one diagnosis.

Current evidence is less mature than for lateral elbow tendinopathy. Eccentric or progressive flexor-pronator loading may be used within a multimodal programme, but no single loading mode is frozen as uniquely superior.

## E3 — Ulnar neuropathy at the elbow / cubital tunnel syndrome

Structured key:

```text
ulnar_neuropathy_at_elbow
```

Default presentation wording if a formal diagnosis is not asserted:

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

Objective findings — selectable only if actually assessed:

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
ulnar-nerve tension/neurodynamic finding
ulnar-nerve subluxation/snapping if actually observed
```

Hard rule:

```text
paresthesia alone != objective neurological deficit
positive Tinel/elbow-flexion test != definitive diagnosis
```

Conservative-management directions may include education/activity modification and avoidance of prolonged provocative elbow flexion/pressure, with splinting/position modification where appropriate. Neurodynamic techniques may be considered selectively; evidence does not support one universal physiotherapy technique.

Progressive motor weakness, intrinsic atrophy or a materially worsening objective deficit requires clinician reassessment rather than routine reassuring referral language.

## E4 — Distal biceps tendinopathy / established partial tear — conservative pathway

Structured key:

```text
distal_biceps_tendon_disorder_nonoperative
```

Display:

> Τενοντοπάθεια / επιβεβαιωμένη μερική ρήξη περιφερικού τένοντα δικεφάλου — συντηρητική αποκατάσταση

Subtype/context:

```text
distal_biceps_tendinopathy
confirmed_partial_distal_biceps_tear
other_established_distal_biceps_disorder
```

For a partial tear, required context should include:

```text
clinician/imaging-established partial tear
traumatic vs degenerative onset if known
current nonoperative/conservative management decision
activity-demand context
restrictions if any
```

Useful findings:

```text
anterior antecubital pain
distal-biceps tenderness
pain with resisted elbow flexion
pain with resisted supination
supination weakness if objectively assessed
flexion weakness if objectively assessed
```

The generator must not diagnose a partial tear from pain or weakness alone.

Hard safety boundary:

```text
acute eccentric-load injury
+ sudden pain/bruising/deformity
+ marked new supination/flexion weakness or unresolved rupture concern
→ prompt timely clinician/orthopaedic reassessment
→ do not default to routine tendinopathy rehabilitation wording
```

Complete distal biceps rupture should not be silently routed through the tendinopathy pathway.

## E5 — Distal triceps tendinopathy / established partial tear — conservative pathway

Structured key:

```text
distal_triceps_tendon_disorder_nonoperative
```

Display:

> Τενοντοπάθεια / επιβεβαιωμένη μερική ρήξη περιφερικού τένοντα τρικεφάλου — συντηρητική αποκατάσταση

Subtype/context:

```text
distal_triceps_tendinopathy
confirmed_partial_distal_triceps_tear
other_established_distal_triceps_disorder
```

Useful findings:

```text
posterior elbow/distal-triceps pain
pain with resisted elbow extension
pain with pushing/pressing
objective extension weakness if assessed
local tenderness at triceps insertion
```

Complete rupture or acute major extension deficit is not a routine tendinopathy pathway.

Safety boundary:

```text
acute posterior-elbow injury
+ palpable defect/deformity or marked new extension weakness
+ unresolved rupture concern
→ clinician/orthopaedic reassessment prompt
```

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

Imaging OA alone does not prove that all current symptoms are generated by the OA.

A true mechanical lock, rapidly progressive loss of motion or unresolved loose-body/intra-articular concern should trigger reassessment rather than routine mobility wording.

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
current stability/specialist assessment context
brace/ROM/loading restrictions
throwing/sport demands
```

Valgus stress, moving-valgus-stress, milking manoeuvre, varus stress or PLRI provocation findings remain findings and do not create a formal instability diagnosis.

Rehabilitation may emphasize protected ROM where indicated, progressive strength, flexor-pronator/extensor-supinator and proximal-chain capacity as relevant, dynamic stability, and criterion-based return to throwing/sport. No single timeline is generated automatically.

## E8 — Olecranon bursitis — assessed aseptic/noninfectious pathway

Structured key:

```text
olecranon_bursitis_assessed_aseptic
```

Display:

> Θυλακίτιδα ωλεκράνου — αξιολογημένη μη σηπτική / άσηπτη

This pathway is deliberately safety-gated because septic and aseptic bursitis can overlap clinically.

Required context:

```text
clinician considers septic/infectious concern addressed
acute vs chronic/recurrent
trauma/pressure context if known
known gout/RA/inflammatory context if relevant
current medical treatment/restrictions if relevant
```

Findings may include:

```text
posterior olecranon swelling
local tenderness
pressure intolerance
skin/soft-tissue irritation if assessed
```

Hard safety rule:

```text
olecranon swelling
+ fever OR significant warmth/erythema/cellulitis OR wound/drainage OR systemic illness OR unresolved infection concern
→ medical reassessment prompt
→ do not generate routine aseptic-bursitis physiotherapy reassurance
```

The generator does not recommend aspiration or injection; these are medical management decisions. For established aseptic bursitis, pressure/load modification and supportive rehabilitation may be documented where useful.

## E9 — Post-traumatic elbow pain / stiffness after assessed injury

Structured key:

```text
post_traumatic_elbow_pain_stiffness
```

Display:

> Μετατραυματικός πόνος / δυσκαμψία αγκώνα μετά από αξιολογημένη κάκωση

Use only after clinically important unresolved fracture, dislocation, major tendon rupture or unstable ligament injury has been addressed as required.

Required context:

```text
injury/date or phase
established structural diagnosis if any
imaging/orthopaedic context if relevant
immobilization history
current ROM/loading/use restrictions
```

The generator must not label an unassessed traumatic elbow as a `simple sprain`.

## E10 — Post-operative elbow rehabilitation — workflow confirmation required before freeze

Structured key:

```text
postoperative_elbow_rehabilitation
```

Display:

> Μετεγχειρητική αποκατάσταση αγκώνα

Candidate for inclusion if it reflects the product owner's real workflow.

Required context:

```text
operation/procedure
operation date
surgeon/protocol when available
immobilization/brace status
ROM restrictions
loading/strengthening restrictions
weight-bearing/use restrictions if relevant
return-to-sport/work constraints
```

Examples may include distal biceps/triceps repair, ligament reconstruction/repair, elbow arthrolysis, arthroplasty or surgery after complex injury.

No generic postoperative timeline may be invented. Current literature shows substantial protocol heterogeneity even for specific operations/injuries.

---

# 3. Candidate secondary diagnoses / modifiers

## 3.1 Radial tunnel syndrome / radial nerve-PIN-related presentation

Proposed default role:

```text
clinician-entered secondary/coexisting diagnosis or differential-context modifier
```

Possible entries:

```text
formal_radial_tunnel_syndrome
radial_nerve_or_posterior_interosseous_neuropathy
radial_tunnel_presentation_not_formally_diagnosed
```

Reason for not making it an automatic top-level diagnosis in v1:

- radial tunnel syndrome remains diagnostically heterogeneous/controversial;
- it may coexist with lateral elbow tendinopathy;
- electrodiagnostic studies may be normal in pain-predominant radial tunnel syndrome;
- posterior interosseous neuropathy with objective motor deficit has different implications from pain-only radial tunnel presentation.

Findings such as radial-tunnel tenderness or pain with resisted middle-finger extension/supination must not create the diagnosis automatically.

Progressive finger/wrist extension weakness or other objective radial motor deficit requires medical reassessment semantics.

Product-owner decision before freeze: keep secondary/context only vs promote a clearly clinician-established radial-tunnel pathway.

## 3.2 Inflammatory arthritis / crystal disease context

Directly selectable context when already established:

```text
known_rheumatoid_or_other_inflammatory_elbow_involvement
known_gout_or_crystal_disease_context
```

This is context, not an inferred diagnosis from swelling/pain.

Acute hot swollen joint, systemic illness or unresolved septic/inflammatory differential remains a reassessment issue.

## 3.3 Myofascial findings

Directly selectable when actually examined:

```text
common extensor muscle tenderness
flexor-pronator tenderness
brachioradialis/supinator tenderness
biceps/triceps myofascial tenderness
active trigger points
myofascial pain presentation
```

---

# 4. Examination findings — selectable only when actually assessed

## 4.1 Pain / symptom behaviour

```text
lateral elbow pain
medial elbow pain
anterior/antecubital pain
posterior elbow pain
pain with gripping
pain with lifting/carrying
pain with pushing/pressing
pain with pulling/curling
pain with pronation/supination
pain with throwing/racquet sport
pain with manual tools/work
night symptoms
pressure intolerance over olecranon/medial elbow
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
maximum grip strength reduced if assessed
painful resisted wrist extension
painful resisted wrist flexion
painful resisted pronation
painful resisted supination
objective elbow-flexion weakness
objective elbow-extension weakness
objective wrist/finger-extension weakness
intrinsic-hand weakness
load intolerance without measured weakness
```

## 4.4 Special/provocation-test findings

Secondary expander only:

```text
Cozen-type finding
Mill-type finding
Maudsley/middle-finger-resistance finding
medial epicondyle flexor-pronator provocation
Tinel at cubital tunnel
elbow-flexion ulnar-nerve provocation
ulnar neurodynamic finding
Hook-test or other distal-biceps finding
valgus-stress finding
moving-valgus-stress/milking finding
varus/PLRI provocation finding
radial-tunnel provocation finding
other clinician-entered test
```

Tests are findings, not diagnoses.

---

# 5. Neurological model

Neurological status is component-specific and only recorded when relevant/assessed.

```text
motor: normal / abnormal / not_assessed
sensory: normal / abnormal / not_assessed
```

Optional detailed fields:

```text
ulnar-distribution sensory status
intrinsic-hand motor status
radial/PIN motor status
median-nerve status where clinically relevant
grip/pinch status
atrophy: present / absent / not_assessed
```

Hard invariant:

```text
not_assessed != normal
```

No global `no neurological deficit` statement is generated from missing components.

---

# 6. Safety / reassessment semantics

The generator provides prompts, not autonomous diagnoses or emergency decisions.

## 6.1 High-priority elbow concerns

```text
acute trauma with unresolved fracture concern
acute trauma with unresolved dislocation/instability concern
acute distal-biceps rupture concern
acute distal-triceps rupture concern
new/progressive objective ulnar/radial/median motor deficit
new neurovascular deficit after trauma/dislocation
true locked elbow / major unresolved mechanical block
rapidly progressive post-traumatic swelling or other urgent structural concern
```

## 6.2 Other material concerns

```text
hot swollen joint / septic arthritis concern
olecranon bursitis with unresolved infection concern
fever/cellulitis/wound/drainage around elbow
systemic/inflammatory/malignancy concern
severe unremitting/progressive non-mechanical pain
rapidly progressive atraumatic loss of motion
other clinician concern
```

## 6.3 Safety state and disposition

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present

clinician_disposition when concern present:
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

Candidate fields:

```text
grip/handshake
opening jars/containers
lifting/carrying
pushing/pressing
pulling/curling
turning a key/screwdriver/forearm rotation
writing/typing/mouse use
manual tools/repetitive work
feeding/grooming/face care
reaching hand to mouth/head
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

## E1 lateral elbow tendinopathy

Possible suggestions:

- reduce symptom irritability;
- improve pain-free grip and wrist-extensor load tolerance;
- improve strength/endurance;
- graded return to work/manual tools/racquet/gym demands;
- address relevant proximal/scapular deficits if present;
- improve self-management and load modification.

## E2 medial elbow tendinopathy

Possible suggestions:

- improve flexor-pronator load tolerance;
- improve grip/function;
- restore tolerance for throwing/golf/manual work;
- progressive strength/endurance;
- self-management/load adaptation.

## E3 ulnar neuropathy

Possible suggestions:

- reduce nerve irritation/provocation;
- improve tolerance of sleep/work positions;
- preserve/improve function and strength where deficits are reversible/appropriate;
- improve activity/position self-management;
- monitor objective neurological status where clinically relevant.

Do not promise reversal of established motor deficit or atrophy.

## E4/E5 distal biceps/triceps

Possible suggestions:

- reduce tendon irritability;
- progressively restore flexion/supination or extension load tolerance;
- improve relevant strength/endurance;
- graded return to lifting/pushing/pulling/gym/work;
- respect established tear/protocol restrictions.

## E6 OA/stiffness

Possible suggestions:

- improve tolerated functional ROM;
- reduce pain/irritability;
- maintain/improve strength;
- improve ADLs/work/function;
- support physical activity/self-management.

## E7 instability/ligament injury

Possible suggestions:

- restore protected/functionally appropriate ROM;
- improve dynamic stability and strength;
- graded return to throwing/sport/work;
- criterion-based progression within restrictions.

## E8 bursitis

Goals should focus on pressure/load tolerance and function after medical safety context is resolved; the generator does not frame physiotherapy as treatment of suspected infection.

## E9 post-traumatic / E10 postoperative

Goals derive from the established injury/procedure and explicit restrictions; unrestricted ROM/strength is never assumed.

---

# 9. Rehabilitation directions

## 9.1 Core active directions

```text
physiotherapy assessment and individualized active rehabilitation
education / activity and load modification
therapeutic exercise
progressive tendon loading where appropriate
wrist extensor strengthening/endurance for lateral tendinopathy
flexor-pronator strengthening/endurance for medial tendinopathy
grip strengthening where relevant
elbow flexor/extensor and forearm-rotation strengthening where relevant
ROM/mobility exercise where restricted and safe
proximal shoulder/scapular strengthening when an actual impairment is identified
dynamic-stability/proprioceptive work for ligament/instability contexts
graded return to work/gym/racquet/throwing demands
home exercise programme where appropriate
```

No single loading mode or fixed protocol is mandatory for every elbow condition.

## 9.2 Condition-specific neural direction

For ulnar-neuropathy presentations, selectable directions may include:

```text
education/activity modification
avoid/reduce prolonged provocative elbow flexion or direct cubital-tunnel pressure where appropriate
night-position modification/splinting where appropriate
selective neural-mobility/neurodynamic work when clinically appropriate
strength/function work according to objective deficit and irritability
```

Splinting/activity modification have the clearest conservative evidence base; no single physiotherapy modality is universally established as superior.

## 9.3 Optional adjunct expander

```text
manual therapy / joint mobilization
soft-tissue techniques
dry needling
acupuncture
taping
counterforce brace or wrist-support orthosis for activity-related epicondylalgia
ESWT for chronic epicondylalgia — evidence-sensitive optional item pending product-owner preference
```

### Manual therapy

For lateral elbow tendinopathy, local mobilization/manipulation may improve short-term pain and pain-free grip and can accompany resisted exercise. Cervical/thoracic/wrist techniques should only appear when relevant impairments are actually identified.

### Dry needling

The 2022 lateral-elbow CPG gives a moderate-strength recommendation for tendon or trigger-point dry needling for lateral elbow tendinopathy, and later systematic reviews support short-term pain/function benefit. It remains optional in this product and retains the cross-region competence/availability safeguard.

```text
dry needling selected
→ competence/availability reminder
```

### Acupuncture

May remain a clinician-selected adjunct for lateral/medial elbow pain when clinically appropriate and provider competence exists. Current syntheses suggest possible short-term benefit for lateral epicondylalgia, but certainty is limited; the generator must not call it mandatory or clearly superior to active rehabilitation.

### Counterforce brace / wrist orthosis

For lateral elbow tendinopathy, the current CPG does not support a clear intermediate/long-term benefit but allows use during aggravating activity for immediate pain/strength benefit. Therefore it may be offered as a short-term/activity-specific option rather than a required treatment.

### ESWT

Evidence for chronic lateral elbow tendinopathy remains mixed across reviews. Some comparisons suggest later benefit versus corticosteroid injection, while other syntheses find no clinically important advantage over sham/control or other options. Therefore v1 proposes:

```text
ESWT = optional secondary item for chronic/recalcitrant epicondylalgia
!= default
!= required
!= universally superior
```

Product-owner workflow confirmation is required before freeze.

### Therapeutic ultrasound

Do not present stand-alone therapeutic ultrasound as a standard evidence-backed lateral-elbow treatment; the 2022 CPG found conflicting evidence.

---

# 10. Shared fracture / post-immobilization boundary

Elbow-region fractures route to the shared fracture profile rather than duplicate fracture logic here.

Regional routes include:

```text
radial head/neck fracture
olecranon/proximal ulna fracture
distal humerus fracture
coronoid fracture
other elbow-region fracture
```

Required shared context later:

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

Complex fracture-dislocations/terrible-triad injuries remain fracture/post-traumatic/postoperative contexts governed by explicit surgical/orthopaedic restrictions; no universal early-motion schedule is generated.

---

# 11. Deterministic consistency rules

```text
E1 lateral tendinopathy + only one provocation test
→ do not infer definitive tendinopathy diagnosis from the test

E1 + radial-neural pattern/objective radial deficit
→ prompt to consider radial/PIN differential rather than collapsing into epicondylalgia

E2 medial tendinopathy + ulnar paresthesia
→ preserve tendon and neural findings separately

E3 formal cubital-tunnel diagnosis != yes
→ use symptom/presentation wording unless clinician explicitly asserts diagnosis

E3 + progressive motor weakness/atrophy
→ prominent medical/specialist reassessment prompt

E4 partial distal-biceps tear selected + no established diagnosis/imaging context
→ warning

acute biceps injury + marked supination/flexion weakness/rupture concern
→ reassessment prompt before routine tendon rehab wording

E5 partial distal-triceps tear selected + no established diagnosis context
→ warning

acute triceps injury + marked extension weakness/rupture concern
→ reassessment prompt

E7 ligament/instability selected + no clinician-established injury/context
→ do not infer instability from stress test alone

E8 aseptic olecranon bursitis + infection concern unresolved
→ do not generate aseptic/routine physio wording

E9 post-traumatic + unresolved fracture/dislocation/major tendon/instability concern
→ safety prompt

E10 postoperative + missing procedure/protocol/restrictions
→ warning

adjunct selected + no active rehabilitation direction
→ warning

dry needling selected
→ competence/availability reminder

ESWT selected + no chronic/recalcitrant epicondylalgia context
→ soft evidence/context warning

material safety concern + no clinician disposition
→ no routine reassuring wording

not_assessed neurological component
→ never generate normal wording
```

---

# 12. Generated wording examples

## 12.1 Lateral elbow tendinopathy

> Πλάγια επικονδυλαλγία / τενοντοπάθεια εκτεινόντων του [side] καρπού με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με προοδευτική φόρτιση των εκτεινόντων, βελτίωση της δύναμης/αντοχής και της λαβής όπου ενδείκνυται, τροποποίηση φορτίου και σταδιακή επάνοδο στις απαιτούμενες δραστηριότητες. [Selected adjuncts if present.]

## 12.2 Medial elbow tendinopathy

> Έσω επικονδυλαλγία / τενοντοπάθεια καμπτήρων-πρηνιστών του [side] άνω άκρου με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη ενεργητική αποκατάσταση με προοδευτική φόρτιση του flexor-pronator μηχανισμού, βελτίωση της λαβής/αντοχής και σταδιακή επιστροφή σε [work/sport selected].

## 12.3 Ulnar-neuropathy presentation without formal diagnosis

> Συμπτωματολογία ωλενίου νεύρου στην περιοχή του [side] αγκώνα με [selected subjective symptoms] και [selected objective findings only if actually assessed]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και συντηρητική αποκατάσταση με εκπαίδευση/τροποποίηση επιβαρυντικών θέσεων και δραστηριοτήτων και επιλεγμένες νευροδυναμικές/λειτουργικές παρεμβάσεις όπου ενδείκνυνται. [Monitoring/reassessment criteria if selected.]

No `normal motor/sensory status` is added unless explicitly assessed and selected.

## 12.4 Formal cubital tunnel diagnosis

Only when clinician-asserted:

> Ωλένια νευροπάθεια στον αγκώνα / σύνδρομο κυβοειδούς σωλήνα, με [selected symptoms/findings]. Παρακαλώ για συντηρητική φυσιοθεραπευτική αντιμετώπιση με έμφαση στην εκπαίδευση, την αποφόρτιση/τροποποίηση επιβαρυντικών θέσεων, τη λειτουργικότητα και την παρακολούθηση τυχόν αντικειμενικών νευρολογικών μεταβολών.

## 12.5 Distal biceps conservative pathway

> [Clinician-established distal biceps tendinopathy / partial tear] του [side] άνω άκρου, για συντηρητική αποκατάσταση, με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για προοδευτική αποκατάσταση της φόρτισης, της κάμψης/υπτιασμού και της λειτουργικής δύναμης σύμφωνα με την κλινική εικόνα και τους καταγεγραμμένους περιορισμούς.

No statement that surgical assessment is unnecessary is generated.

## 12.6 Elbow OA/stiffness

> Οστεοαρθρίτιδα / εκφυλιστική επώδυνη δυσκαμψία του [side] αγκώνα με περιορισμό [selected ROM] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη ενεργητική αποκατάσταση με στόχο τη βελτίωση της ανεκτής κινητικότητας, της δύναμης και της λειτουργικότητας, καθώς και την υποστήριξη αυτοδιαχείρισης/δραστηριότητας.

## 12.7 Postoperative elbow

If the pathway is retained after product-owner confirmation:

> Μετεγχειρητική αποκατάσταση του [side] αγκώνα μετά από [procedure], επέμβαση [date if entered]. Παρακαλώ για φυσιοθεραπευτική αποκατάσταση σύμφωνα με το διαθέσιμο χειρουργικό πρωτόκολλο και τους καταγεγραμμένους περιορισμούς σε εύρος κίνησης, φόρτιση, ενδυνάμωση και χρήση του άνω άκρου.

---

# 13. Evidence-governance boundary

Stable structural decisions proposed for elbow:

```text
provocation/special test != diagnosis
subjective neural symptom != objective neurological deficit
pain-limited effort != tendon tear
lateral epicondylalgia must remain distinguishable from radial/PIN/cervical/intra-articular causes
medial tendinopathy must remain distinguishable from ulnar neuropathy and UCL pathology
complete distal biceps/triceps rupture concern != routine tendinopathy referral
aseptic olecranon bursitis wording requires infection concern to be addressed
fractures route to shared fracture/post-immobilization profile
postoperative rehabilitation requires operation/protocol/restriction context
active/function-oriented rehabilitation remains core
adjunct techniques remain optional
```

Current evidence anchors reviewed for this candidate:

- Lucado et al. `Lateral Elbow Pain and Muscle Function Impairments` — JOSPT/APTA CPG 2022, DOI 10.2519/jospt.2022.0302;
- 2024 systematic review/meta-analysis of non-invasive therapies for lateral elbow tendinopathy;
- 2024 updated systematic review/meta-analysis of dry needling for lateral epicondylitis;
- 2025/2026 evidence syntheses on acupuncture and medial epicondylalgia;
- 2025 Cochrane review and prior systematic reviews for ulnar neuropathy at the elbow;
- 2024/2025 reviews and 2026 systematic review for distal biceps tendinopathy/partial tears;
- 2024/2025 systematic/scoping reviews of UCL and postoperative elbow rehabilitation;
- 2023 systematic review of non-surgical aseptic olecranon bursitis management;
- 2024 systematic review demonstrating diagnostic heterogeneity of radial tunnel syndrome;
- recent systematic reviews showing mixed evidence for ESWT in lateral elbow tendinopathy.

Evidence-sensitive items to refresh immediately before CU-2 production implementation:

- exact loading dosage and progression;
- relative role of eccentric/isometric/concentric programmes;
- dry-needling effect estimates and technique wording;
- acupuncture evidence;
- ESWT role;
- splint/orthosis recommendations;
- cubital-tunnel conservative-treatment evidence;
- ligament/throwing return-to-sport criteria;
- distal biceps/triceps conservative vs specialist thresholds;
- postoperative protocol wording.

---

# 14. Product-owner decisions required before freeze

1. **Radial tunnel syndrome:** keep as secondary/coexisting clinician-entered diagnosis/context, or make a dedicated primary pathway because it appears often enough in real workflow?
2. **Olecranon bursitis:** retain as a primary elbow pathway with strict infection gate, or keep it only as secondary/medical context because physiotherapy referral is uncommon?
3. **Post-operative elbow:** include in the active elbow MVP, as with shoulder, or exclude because the product owner rarely sees these patients?
4. **ESWT for chronic lateral/medial epicondylalgia:** expose as an optional evidence-sensitive adjunct, or omit from elbow v1.1?
5. **Acupuncture:** retain as an optional elbow adjunct, consistent with the clinician's lumbar/shoulder referral workflow?
6. **Distal biceps and triceps:** keep separate primary pathways as proposed, or place triceps under a broader tendon-injury/shared myotendinous profile because it is uncommon?
7. Any frequently referred elbow entity missing from this candidate — for example isolated radial-head/radiocapitellar pathology, plica, inflammatory elbow disease, or another sport-specific pathway?

This file remains a **design candidate** until those real-workflow decisions are resolved. Runtime implementation remains unauthorized.
