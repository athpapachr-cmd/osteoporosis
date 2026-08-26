# Cervical Physiotherapy Referral Profile v1.1 — CU-1 FROZEN DESIGN

> **STATUS:** FROZEN CLINICAL/CONTENT DESIGN — product-owner approved 2026-08-26.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful cervical referral choices without turning the referral generator into a diagnostic engine or prescribing the physiotherapist's full treatment plan.
> **Supersedes as active design:** `clinic_utilities/physio_profiles/cervical_v1.md`.

---

# 1. Core design contract

The cervical referral profile uses:

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
selected != mandatory
symptom != objective deficit
test finding != diagnosis
not assessed != normal
adjunct != core rehabilitation
```

The utility may structure and check consistency. It must not infer a diagnosis, decide medical urgency autonomously, or prescribe the physiotherapist's complete treatment plan.

---

# 2. Primary cervical pathways

## C1 — Non-specific / mechanical neck pain

Structured key:

```text
nonspecific_neck_pain
```

Display:

> Μη ειδική / μηχανικού τύπου αυχεναλγία

Use when the main problem is axial/mechanical cervical pain without a more specific neurological, headache, dizziness or traumatic pathway.

Common modifiers/findings may include mobility restriction, movement/load-related pain, referred shoulder-girdle pain, myofascial tenderness/trigger points, sustained-posture or ergonomic aggravation, and recurrent/chronic presentation.

## C2 — Neck pain with radiating upper-limb symptoms / radicular features

Structured key:

```text
neck_pain_with_radiating_upper_limb_symptoms
```

Display:

> Αυχεναλγία με ακτινοβολία στο άνω άκρο / ριζιτικού τύπου χαρακτηριστικά

Use when the clinical presentation supports a cervical source with radiating arm pain, paresthesia or other radicular-type features.

Mandatory semantic separation:

```text
subjective radiating/radicular symptoms
!=
objective motor/sensory/reflex deficit
!=
formal cervical radiculopathy diagnosis
```

Optional clinician assertion:

```text
formal_cervical_radiculopathy_diagnosis: yes / no / not_stated
```

A positive Spurling test, radiating pain or neurodynamic finding alone must never cause the utility to assert cervical radiculopathy.

## C3 — Headache with cervical musculoskeletal features / formal cervicogenic headache

Structured key:

```text
headache_with_cervical_msk_features
```

Default display:

> Κεφαλαλγία με αυχενικά μυοσκελετικά χαρακτηριστικά

Optional clinician assertion:

```text
formal_cervicogenic_headache_diagnosis: yes / no / not_stated
```

If `yes`, generated wording may state:

> Αυχενογενής κεφαλαλγία

If not explicitly selected, the utility must use cautious symptom/presentation wording and must not infer a cervicogenic-headache diagnosis from neck pain, ROM restriction or trigger points alone.

## C4 — Cervical/cervicogenic dizziness pathway

Structured key:

```text
cervical_dizziness_presentation
```

Default display:

> Ζάλη με αυχενικά μυοσκελετικά χαρακτηριστικά

Optional clinician assertion:

```text
clinician_diagnosis_cervicogenic_dizziness: yes / no / not_stated
```

If `yes`, generated wording may state the clinician's diagnosis as:

> Αυχενογενής / αυχενικής προέλευσης ζάλη

Safety/evidence boundary:

- dizziness is a symptom and must remain separate from a clinician-entered diagnosis;
- the utility must not infer cervical causation from neck pain plus dizziness;
- the utility must not claim that vestibular, neurological, vascular, migraine-related or other causes have been excluded unless the clinician explicitly documents the relevant assessment/context;
- the referral generator is not a dizziness diagnostic decision-support tool.

Evidence governance note: the Bárány Society currently considers the causal construct of cervical dizziness uncertain and does not endorse routine clinical diagnostic criteria. This does not prevent the utility from faithfully carrying a clinician's explicit diagnosis; it prevents the software from generating that diagnosis automatically.

## C5 — Whiplash-associated / post-traumatic neck pain

Structured key:

```text
post_traumatic_neck_pain
```

Display:

> Μετατραυματική / whiplash-associated αυχεναλγία

Use when physiotherapy is considered appropriate after cervical trauma.

Required context should make it easy to record:

```text
date/phase of injury
known structural status when relevant
current restrictions/precautions when relevant
```

The utility must not silently claim that fracture, instability or other significant structural injury has been excluded.

## Post-operative cervical rehabilitation

Post-operative cervical rehabilitation is deliberately **not part of the active cervical MVP taxonomy** because it is not part of the product owner's current clinical workflow.

The broader Clinic Utilities architecture may retain a shared post-operative musculoskeletal pathway for future use elsewhere. It must not appear as a routine cervical primary-problem option in this profile.

---

# 3. Findings / modifiers — selectable only when actually assessed or elicited

## 3.1 Pain distribution and symptom behaviour

```text
axial cervical pain
occipital pain/headache
referred pain to shoulder girdle/scapular region
radiating upper-limb pain
paresthesia
numbness
unilateral / bilateral symptom distribution
night/sleep disturbance
movement-related aggravation
sustained-posture aggravation
work/ergonomic load aggravation
```

`referred shoulder-girdle pain` is intentionally directly selectable because it is a common clinically useful presentation. It remains a symptom-distribution modifier, not an automatic independent diagnosis.

## 3.2 Cervical mobility

```text
active ROM restricted
painful ROM
specific directional restriction optional
```

ROM impairment is never globally preselected.

## 3.3 Myofascial findings

```text
muscle tenderness
increased tone
clinically active trigger points
myofascial pain presentation
```

These are directly selectable because they are common referral-relevant findings/presentations in the product owner's practice.

They remain findings/presentation modifiers by default. If the clinician independently considers a formal myofascial diagnosis established, it may be represented in clinician free text or a future explicit diagnosis assertion; the utility must not derive that diagnosis from a trigger-point checkbox.

## 3.4 Headache-specific findings/context

Optional, only when assessed:

```text
headache temporally associated with cervical symptoms
headache provoked/aggravated by cervical movement or sustained posture
restricted/painful cervical movement associated with headache
occipital/suboccipital tenderness or other relevant cervical finding
```

These findings may support referral content but never independently establish cervicogenic headache.

## 3.5 Dizziness-specific symptom/context fields

Optional, only when elicited/assessed:

```text
dizziness / disequilibrium / light-headedness
symptoms associated with neck movement or cervical symptom flare
balance-related functional limitation
other dizziness context free text
```

No checkbox in this group may automatically assert a cervical aetiology.

## 3.6 Provocation / neural findings

```text
Spurling reproduces concordant symptoms
upper-limb neurodynamic test positive/negative
symptom relief with distraction optional
```

A provocation test is a finding, not a standalone diagnosis.

---

# 4. Neurological-screen model

Subjective neural symptoms and objective neurological findings must remain separate.

```text
subjective_neural_symptoms
  radiating_pain
  paresthesia
  numbness
  distribution / laterality

objective_root_screen
  motor: normal / abnormal / not_assessed
  sensory: normal / abnormal / not_assessed
  reflexes: normal / abnormal / not_assessed

objective_details
  motor_detail_optional
  sensory_detail_optional
  reflex_detail_optional
```

Hard invariant:

```text
NOT_ASSESSED
→ must never generate NORMAL wording
```

There is no global `no neurological deficit` checkbox.

Formatter rules:

```text
motor=normal + sensory=normal
→ may state only that no motor or sensory deficit was identified at the current examination

reflexes=not_assessed
→ reflex status is omitted

any component=abnormal
→ state the actual selected abnormal finding; never produce generic reassurance
```

---

# 5. Cord / neurological safety concerns

Separate potential cord/upper-motor-neuron concerns from ordinary radicular symptoms:

```text
new_or_progressive_objective_motor_deficit
progressive_or_expanding_sensory_loss
new_gait_or_balance_change
hand_clumsiness_or_dexterity_change
upper_motor_neuron_or_possible_myelopathy_concern
other_cord_or_neurological_concern
```

Bilateral symptoms alone are not an automatic high-priority trigger; their significance depends on the associated neurological and clinical context.

---

# 6. General safety / reassessment semantics

This is a clinician-facing consistency/safety layer, not an autonomous diagnostic algorithm.

## 6.1 Safety-screen state

```text
safety_screen_status:
  not_assessed
  no_specific_concern_identified
  concern_present
```

If `concern_present`, candidate concern categories include:

```text
progressive objective neurological deficit
possible cord/myelopathic feature
relevant trauma / possible fracture or instability
systemic/infectious/malignancy/inflammatory/vascular concern
severe/unremitting/progressive non-mechanical pain or other clinician red-flag concern
other clinician concern
```

## 6.2 Clinician disposition

When a material concern is present, support explicit clinician disposition such as:

```text
reviewed_and_appropriate_to_proceed
medical_reassessment_arranged
specialist_or_imaging_pathway_underway
routine_physiotherapy_deferred
other
```

The utility does not select the disposition.

There is no default `no red flags` output sentence. Absence of a selected concern never becomes proof that a complete red-flag assessment was performed.

---

# 7. Functional limitations

Candidate fields:

```text
sleep disturbance
driving / head-turning limitation
desk/computer work or sustained-posture limitation
lifting/carrying limitation
upper-limb or overhead activity limitation
exercise/sport limitation
ADL/self-care limitation
work absence or reduced work tolerance
balance-related activity limitation where relevant
patient-priority activity / free text
```

At least one functional limitation should be encouraged when useful for goal-setting, but it is not mandatory.

---

# 8. Context-sensitive goal suggestions

Goals are suggestions only and are never globally preselected.

## C1 — Non-specific/mechanical neck pain

Possible suggestions:

- reduce symptom irritability;
- restore comfortable functional cervical ROM if restricted;
- improve cervical/scapular strength and endurance where relevant;
- improve motor control where relevant;
- improve tolerance of selected work/driving/ADL tasks;
- graded return to exercise/activity;
- improve self-management and recurrence-management capability.

## C2 — Radiating/radicular features

Possible suggestions:

- reduce cervical/upper-limb symptom irritability;
- improve functional cervical/upper-limb tolerance;
- restore strength/endurance where affected and appropriate;
- progressive return to activity/exercise;
- neural mobility/neurodynamic rehabilitation where indicated;
- preserve/restore function while monitoring neurological status.

The wording must not promise reversal of an objective neurological deficit.

## C3 — Headache/cervicogenic-headache pathway

Possible suggestions:

- reduce headache frequency/intensity;
- improve cervical mobility if impaired;
- improve cervical/scapular endurance/control where relevant;
- improve tolerance of provoking activities/postures;
- improve self-management.

## C4 — Cervical/cervicogenic dizziness pathway

Possible suggestions:

- reduce dizziness-related functional limitation;
- improve tolerance of head/neck movement where relevant;
- improve cervical function when impaired;
- improve balance/activity confidence where a relevant deficit has been identified;
- graded return to relevant activity;
- improve self-management.

Do not promise that cervical physiotherapy will resolve dizziness or imply that other dizziness causes have been excluded.

## C5 — Whiplash/post-traumatic

Possible suggestions:

- graded return to normal movement/activity;
- restore mobility/function;
- address fear/avoidance only when identified;
- improve cervical/scapular endurance/control where relevant;
- education/reassurance and self-management;
- work/activity re-entry.

---

# 9. Rehabilitation directions

## 9.1 Core active directions

The default conceptual backbone is active and function-oriented rehabilitation:

```text
physiotherapy assessment and individualized active rehabilitation
therapeutic exercise
progressive cervical/scapular strengthening and endurance where relevant
mobility exercise where restricted
graded activity/exposure
education and self-management
home exercise programme where appropriate
activity/load or ergonomic adaptation where relevant
```

Generated wording should remain collaborative and should not dictate a fixed technique bundle.

## 9.2 Optional adjunct expander

All technique-level adjuncts belong under:

> Προαιρετικές επικουρικές τεχνικές

Nothing is preselected.

Available options:

```text
manual therapy / mobilization
soft-tissue techniques
neurodynamic techniques
selected traction for appropriate radiating/radicular presentations
dry needling for relevant myofascial/trigger-point presentation
acupuncture as an optional adjunct where clinically appropriate
```

Additional dizziness-pathway note:

- manual therapy may be selected as an adjunct when clinically appropriate;
- exercise/active rehabilitation remains represented separately;
- the utility must not present any technique as a diagnostic test for cervicogenic dizziness or as proof of cervical causation.

---

# 10. Deterministic consistency rules

```text
C2 selected
+ motor/sensory/reflex all not_assessed
→ prompt: consider documenting current neurological status

Spurling positive
+ no radiating/radicular symptoms
→ soft warning: provocation finding alone does not establish a radicular diagnosis

neurodynamic technique selected
+ no radiating/neural context
→ soft warning

traction selected
+ no C2 radiating/radicular pathway
→ warning

dry needling selected
+ no trigger-point/myofascial finding
→ soft warning

any adjunct selected
+ no active/function-oriented rehabilitation direction
→ warning

progressive objective neurological deficit
→ high-priority medical reassessment prompt

possible cord/myelopathy concern
→ high-priority medical reassessment prompt

relevant trauma
+ unresolved structural/restriction context
→ safety prompt

material safety concern
+ no clinician disposition
→ do not generate routine reassuring wording

formal_cervicogenic_headache_diagnosis != yes
→ do not output definitive `cervicogenic headache`

clinician_diagnosis_cervicogenic_dizziness != yes
→ do not output definitive `cervicogenic dizziness`

no selected safety concern
→ must not generate `no red flags`
```

---

# 11. Generated wording contract

All output derives only from confirmed `ReferralDraft` values.

## 11.1 Generic short pattern

> [Clinician-confirmed clinical problem/presentation] με [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με στόχο [selected goals], με έμφαση στην εκπαίδευση, αυτοδιαχείριση και σταδιακή επάνοδο στη δραστηριότητα. [Selected additional rehabilitation directions/restrictions if present.]

## 11.2 Formal cervicogenic headache example

Only when `formal_cervicogenic_headache_diagnosis=yes`:

> Αυχενογενής κεφαλαλγία με [selected cervical findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση με στόχο τη βελτίωση της αυχενικής λειτουργίας, τη μείωση της κεφαλαλγίας και την επάνοδο στις επιλεγμένες δραστηριότητες. Επικουρικές τεχνικές μπορούν να χρησιμοποιηθούν ανάλογα με τη φυσιοθεραπευτική αξιολόγηση και τις αντενδείξεις.

## 11.3 Formal cervicogenic/cervical dizziness example

Only when `clinician_diagnosis_cervicogenic_dizziness=yes`:

> Αυχενογενής / αυχενικής προέλευσης ζάλη με συνοδά αυχενικά μυοσκελετικά ευρήματα [selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη αποκατάσταση με στόχο τη βελτίωση της αυχενικής λειτουργίας, της ανοχής στην κίνηση και της σχετικής λειτουργικότητας, με ενεργητικό πρόγραμμα και αυτοδιαχείριση ως βασικό κορμό. Η επιλογή τυχόν επικουρικών τεχνικών παραμένει στη φυσιοθεραπευτική αξιολόγηση.

The formatter must not add statements that vestibular/neurological/vascular causes were excluded unless explicit clinician-entered context supports that wording.

## 11.4 Myofascial / trigger-point dominant presentation example

> Αυχεναλγία με έντονα μυοπεριτονιακά ευρήματα / ενεργά trigger points σε [selected region if entered], με [selected referred shoulder-girdle pain or other findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση με ενεργητικό πρόγραμμα, εκπαίδευση και σταδιακή επάνοδο στη δραστηριότητα. Τεχνικές μαλακών μορίων ή dry needling μπορούν να χρησιμοποιηθούν επικουρικά εφόσον κριθούν κατάλληλες κατά τη φυσιοθεραπευτική αξιολόγηση.

This wording describes the clinician-selected presentation; it does not automatically assert a separate myofascial pain syndrome diagnosis.

## 11.5 Referred shoulder-girdle pain example

> Μη ειδική / μηχανικού τύπου αυχεναλγία με αναφερόμενο άλγος προς την ωμική ζώνη/ωμοπλάτη, [other selected findings] και λειτουργικό περιορισμό σε [selected activities]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη ενεργητική αποκατάσταση σύμφωνα με τα παραπάνω ευρήματα και τους επιλεγμένους λειτουργικούς στόχους.

## 11.6 Radiating symptoms with incomplete neurological assessment

> Αυχεναλγία με ακτινοβολία/παραισθησίες προς το [side] άνω άκρο, με [only actually documented findings]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένο ενεργητικό πρόγραμμα αποκατάστασης με προσαρμογή στην ερεθιστικότητα και τη λειτουργική εικόνα.

No negative neurological statement is generated from missing data.

## 11.7 Radiating symptoms with assessed normal motor and sensory findings

Only when `motor=normal` and `sensory=normal`:

> Κατά τον παρόντα έλεγχο δεν διαπιστώθηκε κινητικό ή αισθητικό έλλειμμα.

Reflexes are mentioned only if actually assessed.

## 11.8 Objective neurological deficit

> Αυχεναλγία με ακτινοβολία προς το [side] άνω άκρο και [selected clinical features]. Κατά τον παρόντα έλεγχο καταγράφηκε [specific selected motor/sensory/reflex finding]. Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένη αποκατάσταση με προσαρμογή στην κλινική εικόνα. Ιατρική επανεκτίμηση σε περίπτωση νέας ή περαιτέρω προοδευτικής νευρολογικής μεταβολής.

If the deficit is already progressive, the safety/disposition layer takes precedence before routine referral wording is finalized.

---

# 12. Evidence-governance boundary

Stable structural decisions frozen in v1.1:

```text
presentation != diagnosis
subjective symptoms != objective deficit
not assessed != normal
clinician diagnosis may be carried faithfully but never inferred
active/function-oriented rehabilitation is the core direction
adjunct techniques remain optional
safety prompts support clinician consistency; they do not diagnose
```

Evidence-sensitive technique wording must be rechecked immediately before CU-2 production implementation.

Key evidence notes reviewed for this freeze:

- APTA/JOSPT Neck Pain CPG 2017 remains the published neck-pain CPG and is under revision.
- Contemporary nonspecific-neck-pain guidance continues to emphasize activation, exercise and self-management and is more restrictive about routine traction outside appropriate radicular contexts.
- Cervical/cervicogenic dizziness remains diagnostically controversial: the Bárány Society does not currently endorse routine clinical diagnostic criteria or a proven cervical causal mechanism.
- Systematic reviews report possible benefit from cervical manual therapy for selected cervical-dizziness populations, but certainty ranges from moderate in older synthesis to low/very-low in newer synthesis, and the literature has substantial diagnostic heterogeneity.

Therefore the utility may support a clinician-entered cervicogenic/cervical dizziness diagnosis and referral, but it must not infer the diagnosis or encode a mandatory technique-specific treatment recipe.

---

# 13. Freeze decision

Product-owner approved on 2026-08-26:

- keep non-specific/mechanical neck pain;
- keep radiating/radicular-feature pathway with strict neuro semantics;
- support explicit formal cervicogenic headache diagnosis;
- add cervical/cervicogenic dizziness pathway with explicit clinician-diagnosis assertion and evidence caveat;
- keep whiplash/post-traumatic pathway;
- remove post-operative cervical pathway from the active cervical MVP;
- keep trigger-point/myofascial and referred shoulder-girdle pain directly selectable as actual findings/presentation modifiers;
- retain active rehabilitation as the default conceptual backbone;
- keep technique-level adjuncts in an optional expander;
- never generate `no neurological deficit` or `no red flags` from unassessed/missing data.

This file is the frozen cervical clinical/content design for CU-1. Runtime implementation remains unauthorized until CU-1 as a whole is frozen and the product owner explicitly moves to CU-2.
