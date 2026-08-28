# CU-1 Formatter Language / Referral-Prose Contract EL v1

> **STATUS:** MAINTENANCE AMENDMENT CANDIDATE — product-quality defect correction.
> **Scope:** clinician-facing referral prose only.
> **Does not reopen:** clinical taxonomy, route ownership, safety logic, validation rules, persistence.

## 1. Product defect addressed

The production v1 formatter can emit machine-like prose because selectable IDs are humanized mechanically. This creates mixed-language output and insufficient distinction between Short and Detailed modes.

The corrected formatter must satisfy all of the following:

```text
final referral prose = Greek
machine ids = never visible in generated referral
output = natural clinician-authored referral style, not a dump of selected fields
short != detailed in both structure and information density
clinical meaning/safety invariants = unchanged
```

## 2. Short referral

Target: routine referral a clinician could plausibly write directly.

Default structure: 2–4 compact natural-language sentences.

1. Opening referral sentence: problem/diagnosis or presentation + laterality when relevant.
2. One sentence summarizing the most actionable selected findings/functional limitations.
3. One sentence requesting the selected core physiotherapy goals/directions, including explicit restrictions when present.
4. Optional brief clinician-authored note when supplied.

Do not emit section labels merely because fields exist. Do not enumerate low-value metadata. Selected adjuncts may be included only as a short final clause when clinician-selected.

## 3. Detailed referral

Target: fuller medical referral suitable when more clinical context is useful.

Default structure:

```text
ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ

Κλινική εικόνα
<natural paragraph: primary problem, relevant secondary problem, key findings and function>

Περιορισμοί / προφυλάξεις
<only when explicit restrictions/precautions exist>

Στόχοι και κατευθύνσεις αποκατάστασης
<natural paragraph/bullets with selected goals and core rehab directions>

Πρόσθετα κλινικά στοιχεία
<only when route-specific structural/postoperative context, measurements or clinician note add value>
```

Detailed mode should normally carry materially more context than Short, but still read as clinician prose rather than a serialized data object.

## 4. Greek terminology and label authority

All selectable IDs that can appear in generated prose must map to explicit Greek clinician-facing labels/phrases. Runtime fallback to underscore replacement or English machine-ID humanization is forbidden in generated referral prose.

If a selected renderable ID lacks a Greek label, generation must fail closed with a formatter-contract error rather than exposing the machine identifier.

Clinical route display labels may continue to come from frozen profile `display` / `default display` content when available, provided the resulting label is Greek. Otherwise an explicit Greek formatter label is required.

## 5. Safety and inference

The language amendment does not change any safety/validation semantics.

Still mandatory:

```text
not_assessed != normal
unselected != absent
symptom != diagnosis
imaging finding != automatically symptomatic diagnosis
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
```

No reassuring negative is generated from omission.

## 6. Acceptance examples

### Short — knee OA

> Παραπέμπεται για φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας δεξιού γόνατος, με πόνο στη φόρτιση και δυσκολία στη βάδιση και στις σκάλες. Παρακαλώ για εξατομικευμένο πρόγραμμα ενδυνάμωσης, βελτίωσης της κινητικότητας και προοδευτικής λειτουργικής φόρτισης, με στόχο τη βελτίωση της βάδισης και της καθημερινής λειτουργικότητας.

### Detailed — same case

> **ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ**
>
> **Κλινική εικόνα**  
> Ο ασθενής παραπέμπεται για αποκατάσταση λόγω οστεοαρθρίτιδας δεξιού γόνατος. Αναφέρονται πόνος κατά τη φόρτιση, περιορισμός της βάδισης και δυσκολία στις σκάλες, με κλινικά διαπιστωμένη αδυναμία τετρακεφάλου όπου αυτή έχει πράγματι καταγραφεί.
>
> **Στόχοι και κατευθύνσεις αποκατάστασης**  
> Παρακαλώ για εξατομικευμένο ενεργητικό πρόγραμμα με προοδευτική ενδυνάμωση, ασκήσεις κινητικότητας όπου υπάρχει περιορισμός, νευρομυϊκή/λειτουργική επανεκπαίδευση και σταδιακή αύξηση της ανοχής στη βάδιση και στις σκάλες.

These examples define prose quality and structural distinction; they do not add or preselect clinical options.
