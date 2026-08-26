# Cervical Physiotherapy Referral Profile v1 — CU-1 design candidate

> **STATUS:** DESIGN CANDIDATE — product-owner review required before freeze.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Purpose:** define clinically useful cervical referral choices without turning the referral generator into a diagnostic engine or prescribing the physiotherapist's full treatment plan.

---

# 1. Design decision — simplify the current cervical taxonomy

The current MVP separates several items that are better represented as a **primary clinical problem plus modifiers/findings**.

Current examples such as:

```text
mechanical neck pain
myofascial pain / trigger points
referred shoulder-girdle pain
radicular-type symptoms / positive Spurling
mobility restriction
postural/ergonomic load
```

should not all remain equivalent top-level diagnoses.

Proposed v2 structure:

```text
PRIMARY CLINICAL PROBLEM
+
MODIFIERS / FINDINGS
+
FUNCTIONAL IMPACT
+
SAFETY / PRECAUTIONS
```

This reduces duplication and avoids claiming separate diagnoses when some selections are really symptom distribution, examination findings or contributing factors.

---

# 2. Proposed primary cervical problem choices

## C1 — Non-specific / mechanical neck pain

Use when the main problem is axial/mechanical cervical pain without a more specific neurological or traumatic pathway.

Possible modifiers:

- mobility deficit;
- movement/load-related pain;
- referred pain to shoulder girdle;
- myofascial tenderness / trigger points;
- work/ergonomic aggravation;
- recurrent/chronic presentation.

## C2 — Neck pain with radiating upper-limb symptoms / radicular features

Use when the consultation supports a cervical source with radiating arm symptoms or radicular features.

Do **not** automatically label `cervical radiculopathy` solely because Spurling is positive or arm pain is present.

Structured distinction:

```text
radiating/radicular-type symptoms
!=
confirmed objective neurological deficit
!=
definitive cervical radiculopathy diagnosis
```

If the clinician has made a formal radiculopathy diagnosis, that may be stated explicitly. Otherwise use cautious wording such as:

> cervical pain with radiating upper-limb symptoms / clinical features compatible with nerve-root irritation.

## C3 — Cervicogenic headache / headache associated with cervical dysfunction

Use only when the clinician has assessed the headache pattern and considers a cervical musculoskeletal contribution plausible.

This pathway is missing from the MVP and should be added because headache-related neck pain is a distinct common rehabilitation presentation.

## C4 — Whiplash-associated / post-traumatic neck pain

Use for post-traumatic cervical presentations where physiotherapy is appropriate and acute serious structural injury has been excluded/managed.

This should remain separate because symptom irritability, movement coordination, reassurance and graded activity may differ from ordinary non-traumatic mechanical neck pain.

## C5 — Post-operative / protocol-governed cervical rehabilitation

Do not create a unique cervical post-op protocol inside the generic profile.

Instead route to the shared post-operative rehabilitation pathway and require:

- procedure;
- date;
- surgeon/protocol;
- movement/loading restrictions;
- collar or other restrictions when relevant.

---

# 3. Modifiers / findings — only if actually assessed

## 3.1 Pain distribution / symptom behaviour

Candidate structured fields:

```text
axial cervical pain
occipital pain/headache
referred pain to shoulder girdle/scapular region
radiating pain/paresthesia to upper limb
unilateral / bilateral
night pain / sleep disturbance
movement-related aggravation
sustained-posture aggravation
```

## 3.2 Cervical mobility

```text
active ROM restricted
painful ROM
specific directional restriction optional
```

Do not automatically select ROM impairment for every cervical referral.

## 3.3 Neurological screen

When a radicular/radiating pathway is selected, make the neurological screen easy and prominent.

Candidate fields:

```text
motor deficit: none / present / not assessed
sensory deficit: none / present / not assessed
reflex abnormality: none / present / not assessed
myotomal weakness details optional
dermatomal sensory change details optional
```

The output must distinguish:

```text
not assessed
!=
normal
```

## 3.4 Provocation / neural features

Optional findings only when examined:

```text
Spurling reproduces concordant symptoms
upper-limb neurodynamic test positive/negative
symptom relief with distraction optional
```

A positive test is a finding, not a standalone diagnosis.

## 3.5 Myofascial findings

```text
muscle tenderness
increased tone
clinically active trigger points
```

These are modifiers/findings rather than default primary diagnoses.

## 3.6 Functional impact

Candidate fields:

```text
sleep disturbance
driving limitation
desk/computer work limitation
lifting/carrying limitation
overhead/upper-limb activity limitation
exercise/sport limitation
ADL limitation
work absence/reduced tolerance
```

At least one functional limitation should be encouraged when the referral is intended to communicate rehabilitation goals, but it should not be mandatory if not relevant.

---

# 4. Safety / reassessment prompts

These are clinician-facing prompts, not autonomous diagnoses.

Prominent reassessment/escalation prompts if selected or entered:

```text
new/progressive objective motor deficit
progressive sensory loss
bilateral neurological symptoms
new gait/balance disturbance
hand clumsiness / possible cord symptoms
upper-motor-neuron concern
major trauma / possible instability or fracture
systemic/red-flag concern
unexplained severe/progressive night pain
```

If one of these is present, routine referral generation should require clinician acknowledgement and should not silently produce reassuring wording such as `χωρίς νευρολογικά ελλείμματα`.

A generic `no red flags` checkbox is acceptable only if the clinician actively confirms that red-flag screening was performed; it must never be preselected.

---

# 5. Context-sensitive goal suggestions

## C1 Non-specific/mechanical neck pain

Suggested, not preselected:

- reduce pain/irritability;
- restore comfortable functional cervical ROM if restricted;
- improve cervical and scapular strength/endurance;
- improve motor control where clinically relevant;
- improve tolerance of work/driving/ADLs;
- graded return to exercise/activity;
- self-management and recurrence prevention.

## C2 Radiating/radicular features

Suggested:

- reduce cervical/upper-limb symptom irritability;
- improve functional cervical/upper-limb tolerance;
- restore strength/endurance where affected;
- progressive activity/exercise;
- neural mobility/neurodynamic rehabilitation where indicated;
- preserve/restore function while monitoring neurological status.

Do not automatically promise restoration of a neurological deficit through physiotherapy wording.

## C3 Cervicogenic headache

Suggested:

- reduce headache frequency/intensity;
- improve cervical mobility if impaired;
- improve cervical/scapular endurance and control;
- improve tolerance of provoking activities/postures;
- self-management.

## C4 Whiplash/post-traumatic

Suggested:

- graded return to normal movement/activity;
- restore mobility/function;
- reduce fear/avoidance where present;
- improve cervical/scapular endurance/control;
- education/reassurance and self-management;
- work/activity re-entry.

---

# 6. Rehabilitation-direction suggestions

These should remain broad enough to respect the physiotherapist's assessment.

## Core active directions

```text
individualized therapeutic exercise
progressive cervical/scapular strengthening and endurance
mobility exercise where restricted
graded activity/exposure
education and self-management
home exercise programme
ergonomic/activity modification where relevant
```

## Adjunct options

```text
manual therapy / mobilization
soft-tissue techniques
neurodynamic techniques
selected mechanical intermittent traction for appropriate radiating/radicular presentations
dry needling for clinically relevant myofascial trigger-point presentation
acupuncture as adjunct analgesic option where clinically appropriate
```

Adjuncts must not become the only rehabilitation direction.

### Traction rule

The published APTA/JOSPT Neck Pain CPG (2017) supports mechanical intermittent cervical traction **combined with exercise/manual approaches** for chronic neck pain with radiating pain, but the guideline is currently listed by APTA Orthopedics as under revision.

Therefore v2 should:

```text
allow traction as an optional adjunct
only when a radiating/radicular pathway is selected
never preselect traction
never phrase traction as mandatory
```

---

# 7. Proposed UI structure for cervical region

```text
Αυχενική μοίρα

A. Κύριο κλινικό πρόβλημα
[ ] Μη ειδική / μηχανικού τύπου αυχεναλγία
[ ] Αυχεναλγία με αντανάκλαση/ακτινοβολία στο άνω άκρο ή ριζιτικού τύπου χαρακτηριστικά
[ ] Κεφαλαλγία με πιθανή αυχενική μυοσκελετική συνιστώσα
[ ] Μετατραυματική / whiplash-associated αυχεναλγία
[ ] Μετεγχειρητική αποκατάσταση → κοινό post-op pathway

B. Κλινικά ευρήματα
[ ] Περιορισμός ενεργητικής κινητικότητας
[ ] Πόνος με κίνηση/φόρτιση
[ ] Αναφερόμενο άλγος προς ωμική ζώνη
[ ] Μυϊκή ευαισθησία / trigger points
[ ] Ριζιτικού τύπου αναπαραγωγή συμπτωμάτων
[ ] Χωρίς κινητικό έλλειμμα — μόνο αν ελέγχθηκε
[ ] Χωρίς αισθητικό έλλειμμα — μόνο αν ελέγχθηκε
[ ] Κινητικό έλλειμμα
[ ] Αισθητικό έλλειμμα
[ ] Αντανακλαστικό εύρημα

C. Λειτουργικός περιορισμός
[ ] Ύπνος
[ ] Οδήγηση
[ ] Εργασία/υπολογιστής
[ ] Άρση/μεταφορά
[ ] Άθληση/άσκηση
[ ] Άλλο

D. Προφυλάξεις / επανεκτίμηση
[ ] Προοδευτικό νευρολογικό έλλειμμα
[ ] Διαταραχή βάδισης/ισορροπίας ή πιθανή μυελοπάθεια
[ ] Άλλο red-flag concern

E. Στόχοι / κατεύθυνση αποκατάστασης
context-sensitive suggestions → clinician confirms
```

---

# 8. Output wording examples

## 8.1 Short — mechanical neck pain

> Κλινική εικόνα μηχανικού τύπου αυχεναλγίας με [επιλεγμένα ευρήματα] και λειτουργικό περιορισμό σε [επιλεγμένες δραστηριότητες]. Παρακαλώ για εξατομικευμένο πρόγραμμα φυσικοθεραπευτικής αποκατάστασης με έμφαση σε ενεργητική κινητοποίηση όπου χρειάζεται, προοδευτική ενδυνάμωση/αντοχή του αυχένα και της ωμοπλατοθωρακικής ζώνης, εκπαίδευση, αυτοδιαχείριση και σταδιακή επάνοδο στις συνήθεις δραστηριότητες. Επικουρικές τεχνικές μπορούν να χρησιμοποιηθούν ανάλογα με τη φυσιοθεραπευτική αξιολόγηση και τις αντενδείξεις.

## 8.2 Short — radiating/radicular features without objective deficit

> Αυχεναλγία με ακτινοβολία/ριζιτικού τύπου συμπτώματα προς το [δεξί/αριστερό] άνω άκρο, με [Spurling/άλλο εύρημα] και χωρίς εμφανές κινητικό ή αισθητικό έλλειμμα κατά τον παρόντα έλεγχο. Παρακαλώ για εξατομικευμένη ενεργητική αποκατάσταση με προοδευτικές ασκήσεις κινητικότητας/σταθεροποίησης και ενδυνάμωσης, εκπαίδευση και αυτοδιαχείριση. Νευροδυναμικές τεχνικές, manual therapy ή διαλείπουσα μηχανική έλξη μπορούν να χρησιμοποιηθούν επικουρικά όπου ενδείκνυνται. Ιατρική επανεκτίμηση αν εμφανιστεί νέο ή προοδευτικό νευρολογικό έλλειμμα ή άλλη μη αναμενόμενη επιδείνωση.

## 8.3 Detailed — objective neurological deficit present

If an objective deficit is selected, the generated wording should explicitly state what was found and avoid the generic reassurance sentence.

Example pattern:

> Αυχεναλγία με ακτινοβολία προς το αριστερό άνω άκρο και κλινικά ριζιτικά χαρακτηριστικά. Κατά τον παρόντα έλεγχο καταγράφηκε [specific motor/sensory/reflex deficit]. Παραπομπή για φυσιοθεραπευτική αξιολόγηση και αποκατάσταση με προσαρμογή στην ερεθιστικότητα και στενή παρακολούθηση της νευρολογικής εικόνας. Επανεκτίμηση από ιατρό σε περίπτωση επιδείνωσης ή προόδου του ελλείμματος.

The system must not turn this into a routine `no deficit` template.

---

# 9. Proposed deterministic consistency rules

```text
radicular pathway selected
+ neurological screen entirely unassessed
→ prompt: consider documenting current neurological status

Spurling positive selected
+ no radicular/radiating symptoms
→ soft warning: positive provocation alone does not establish a radicular diagnosis

traction selected
+ no radiating/radicular pathway
→ warning

dry needling selected
+ no myofascial tenderness/trigger-point finding
→ soft warning

objective progressive neurological deficit selected
→ prominent reassessment warning

possible myelopathic/gait-balance concern selected
→ high-priority medical reassessment prompt

`no neurological deficit` selected
+ any motor/sensory/reflex deficit also selected
→ hard inconsistency warning

`no red flags` selected
+ any red-flag concern selected
→ hard inconsistency warning
```

---

# 10. Evidence status / production caveat

For the cervical profile, the published APTA Orthopedics/JOSPT neck-pain guideline currently available is the 2017 revision, and APTA Orthopedics lists Neck Pain as a guideline revision in development.

Implication:

> CU-1 may freeze the **structure and safety semantics** now, but detailed intervention wording should undergo one final evidence check immediately before CU-2 production implementation so the generator does not hard-code a guideline recommendation that is superseded during development.

---

# 11. Product-owner decisions needed before freeze

1. Keep `Μηχανικού τύπου αυχεναλγία` as the familiar display label, with the structured key `nonspecific_mechanical_neck_pain`?
2. Add explicit `Κεφαλαλγία με πιθανή αυχενική συνιστώσα`?
3. Add `Whiplash / μετατραυματική αυχεναλγία`?
4. Move `trigger points`, `referred shoulder-girdle pain`, `mobility restriction` and `postural/ergonomic load` from top-level conditions to findings/modifiers as proposed?
5. For radicular presentations, should the UI offer one compact combined neurological-screen row or separate motor/sensory/reflex controls?
6. Keep traction/dry needling/acupuncture visible as optional adjuncts, or hide them under an `Προαιρετικές επικουρικές τεχνικές` expander?

No runtime implementation should begin until these cervical-profile decisions are reviewed as part of CU-1.
