# CU-1 Formatter Language / Referral-Prose Contract EL v1

> **STATUS:** MAINTENANCE AMENDMENT — clinician-facing product-quality defect correction.
> **Scope:** generated referral prose and clinician-facing display labels only.
> **Does not reopen:** clinical taxonomy, route ownership, safety logic, validation rules, persistence.

## 1. Defect addressed

Production v1 can expose mechanically humanized machine IDs and produces insufficiently differentiated Short and Detailed outputs. This is unacceptable for a clinician-facing referral generator.

Corrected invariants:

```text
final referral prose = Greek
machine ids = never visible in generated referral
output = natural clinician-authored referral style, not a serialized field dump
short != detailed in structure and information density
clinical meaning and safety semantics = unchanged
```

## 2. Short referral

Target: routine referral that a clinician could plausibly write directly.

Default structure: 2–4 compact natural-language sentences.

1. Opening referral sentence: primary problem/diagnosis or presentation + laterality when relevant.
2. One sentence summarizing the most actionable selected findings and functional impact.
3. One sentence requesting the selected core physiotherapy goals/directions; explicit restrictions must remain visible when present.
4. Optional brief clinician-authored note.

Do not emit section labels merely because fields exist. Do not enumerate low-value metadata. Selected adjuncts may appear only as a brief final clause when explicitly clinician-selected.

## 3. Detailed referral

Target: fuller medical referral when more clinical context is useful.

Default structure:

```text
ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ

Κλινική εικόνα
<natural paragraph: primary problem, relevant secondary problem, key findings and function>

Περιορισμοί / προφυλάξεις
<only when explicit restrictions/precautions exist>

Στόχοι και κατευθύνσεις αποκατάστασης
<natural paragraph or compact bullets with selected goals and core rehab directions>

Πρόσθετα κλινικά στοιχεία
<only when route-specific structural/postoperative context, measurements or clinician note add value>
```

Detailed mode must normally carry materially more context than Short while still reading as clinician prose.

## 4. Greek terminology authority

Every selectable ID that can appear in generated prose must map to an explicit Greek clinician-facing phrase. Runtime fallback to underscore replacement or English machine-ID humanization is forbidden in generated referral prose.

If a selected renderable ID lacks a Greek label, generation must fail closed with a formatter-contract error rather than expose the machine identifier.

Clinical route display labels may come from frozen profile `display` / `default display` text when Greek; otherwise an explicit Greek formatter label is required.

## 5. Safety and inference

This amendment changes language/presentation only. Existing safety/validation semantics remain authoritative.

```text
not_assessed != normal
unselected != absent
symptom != diagnosis
imaging finding != automatically symptomatic diagnosis
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

No reassuring negative may be generated from omission.

## 6. Acceptance examples

### Short — knee OA

> Παραπέμπεται για φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας δεξιού γόνατος, με πόνο στη φόρτιση και δυσκολία στη βάδιση και στις σκάλες. Παρακαλώ για εξατομικευμένο πρόγραμμα ενδυνάμωσης, βελτίωσης της κινητικότητας και προοδευτικής λειτουργικής φόρτισης, με στόχο τη βελτίωση της βάδισης και της καθημερινής λειτουργικότητας.

### Detailed — same case

> **ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ**
>
> **Κλινική εικόνα**  
> Ο ασθενής παραπέμπεται για αποκατάσταση λόγω οστεοαρθρίτιδας δεξιού γόνατος. Αναφέρονται πόνος κατά τη φόρτιση, περιορισμός της βάδισης και δυσκολία στις σκάλες, με κλινικά διαπιστωμένη αδυναμία τετρακεφάλου μόνο όταν αυτή έχει πράγματι καταγραφεί.
>
> **Στόχοι και κατευθύνσεις αποκατάστασης**  
> Παρακαλώ για εξατομικευμένο ενεργητικό πρόγραμμα με προοδευτική ενδυνάμωση, ασκήσεις κινητικότητας όπου υπάρχει περιορισμός, νευρομυϊκή/λειτουργική επανεκπαίδευση και σταδιακή αύξηση της ανοχής στη βάδιση και στις σκάλες.

These examples define prose quality and structural distinction; they do not preselect clinical options.
