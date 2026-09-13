# SLICE_PLAN_CURRENT.md — Physiotherapy Referral Knee-OA V5.1 examination refinement

> **STATUS:** APPROVED / ACTIVE IMPLEMENTATION.
> **Activated:** 2026-09-13 Asia/Nicosia.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-1-EXAM-2026-09-13`.
> **Branch:** `feat/physio-knee-oa-v5-1-exam-discoverability-2026-09-13`.
> **Bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Writer:** ChatGPT / bounded Physio Knee-OA V5.1 scope.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Problem

Post-release real-device use identified four bounded product issues:

1. V5 first-tap/second-tap detail is efficient but still mildly hidden.
2. `Μυϊκή αδυναμία στην εξέταση` is too non-specific for an examination finding.
3. `Περισσότερα → Εξέταση` underrepresents ROM, crepitus, tenderness and objective stability findings that are clinically useful in knee OA assessment.
4. Reviewed guideline/source URLs exist in the data model but the source link is not sufficiently visible in the evidence sheet.

## 2. Evidence-informed design

### 2.1 Strength

The 2026 systematic review/meta-analysis `PMID 41767519` reports lower knee extensor and flexor strength in symptomatic knee OA, with larger pooled deficits in extensors. A 2023 systematic review of longitudinal studies also found associations for low extensor and flexor strength with structural worsening.

Therefore examination vocabulary may distinguish extension and flexion weakness without implying a causal mechanism or automatic treatment choice.

Final UI choices are mutually exclusive:

```text
Αδυναμία έκτασης γόνατος / τετρακεφάλου
Αδυναμία κάμψης γόνατος / ισχιοκνημιαίων
Αδυναμία κάμψης και έκτασης γόνατος
```

No separate generic objective-weakness choice remains in the focused exam surface.

### 2.2 Range of motion

The same 2026 meta-analysis supports clinically meaningful loss of both flexion and extension ROM in symptomatic knee OA. EULAR diagnostic recommendations identify restricted movement as a useful clinical sign.

The examination group becomes:

```text
Εύρος κίνησης
└─ Περιορισμός εύρους κίνησης
   ├─ Υστέρηση ενεργητικής έκτασης (extension lag)
   ├─ Παθητικό έλλειμμα έκτασης
   ├─ Περιορισμός ενεργητικής κάμψης
   └─ Περιορισμός παθητικής κάμψης
```

The parent is progressive disclosure. A generic parent without a subtype may render generic restricted ROM but must not fabricate whether the deficit is active/passive or flexion/extension.

The existing optional degree entry remains attached only to passive extension deficit. No flexion degree entry is added in this slice.

### 2.3 Crepitus

EULAR identifies crepitus among the most useful signs and OARSI includes cracking/grating in examination. Add:

`Κριγμός στην κίνηση`

as a simple objective finding under Examination.

### 2.4 Tenderness

Tenderness is clinically relevant but must remain localisation, not diagnosis.

Use one progressive palpation group with:

```text
Αρθρική ευαισθησία
- Έσω
- Έξω

Οστική ευαισθησία
- Έσω
- Έξω

Άλλη εστιακή ευαισθησία
- Χήνειος πόδας
- Εκτατικός μηχανισμός
```

Bony tenderness is an established OA examination finding and must not by itself be treated as atypical or as evidence of SIFK.

### 2.5 Stability

OARSI describes joint stability/alignment as part of OA examination. General knee-examination literature supports directional ligament assessment.

Use objective examination labels:

```text
Αστάθεια σε βλαισότητα
Αστάθεια σε ραιβότητα
Πρόσθια αστάθεια / ΠΧΣ
Οπίσθια αστάθεια / ΟΧΣ
```

Do not use `προσθιοπίσθια αστάθεια` as a combined vague label.
Do not map objective laxity to subjective giving-way/recurrent-instability semantics.

### 2.6 Swelling / enlargement

Do not add a new enlargement control.

Rationale:
- bony enlargement is a recognized OA sign;
- the product already has `effusion`/swelling concepts;
- another routine swelling/enlargement distinction is not currently workflow-useful enough to justify more surface complexity.

### 2.7 Atypical-feature review bubble

NICE NG226 recommends against routine imaging unless atypical features or concern for an alternative/additional diagnosis exists. Examples include recent trauma, prolonged morning stiffness, rapid worsening/deformity and a hot swollen joint.

This slice does not add SIFK/SONK diagnosis inference.

The current `morning stiffness >30′` review clue should gain a restrained visible bubble. Existing unresolved safety flags retain their stronger blocking semantics.

No bony-tenderness-only bubble.

## 3. Discoverability microcopy

When inactive `Πόνος`, `Δυσκαμψία` or `Αδυναμία` is first selected:

```text
[active symptom card]
Πατήστε ξανά για προαιρετικές λεπτομέρειες
```

Requirements:
- appears only while the generic symptom is selected;
- disappears/replaces appropriately when detail count is present;
- must not look like validation error or mandatory next step;
- no popup on first tap;
- Function remains a first-tap chooser and does not need this hint.

## 4. Source links

International source rows already carry validated source-level `locator` URLs.
Local `CY_GESY` rows already carry `source_provenance.source_url`.

New presentation:

```text
NICE NG226 · 2022   ↗
Κύπρος · ΟΑΥ        ↗
```

The visible source label or adjacent external-link control opens the reviewed official source in a new tab with `noopener noreferrer`.

The existing deeper provenance/details remain available.

## 5. Data and semantic contract

All new detailed findings are product-local, ephemeral qualifiers unless an existing CU-1 finding is semantically exact enough to map safely.

Do not add new international evidence items merely because an exam finding exists.
Do not use exam findings to auto-select therapy.
Do not infer an additional diagnosis.

## 6. Referral prose

If explicitly selected, the referral may describe the finding in the clinical-picture paragraph.

Preferred language:

```text
αδυναμία έκτασης γόνατος / τετρακεφάλου κατά την εξέταση
αδυναμία κάμψης γόνατος / ισχιοκνημιαίων κατά την εξέταση
αδυναμία κάμψης και έκτασης γόνατος κατά την εξέταση
κριγμό κατά την κίνηση
περιορισμό ενεργητικής κάμψης
περιορισμό παθητικής κάμψης
αρθρική ευαισθησία έσω/έξω
οστική ευαισθησία έσω/έξω
ευαισθησία στον εκτατικό μηχανισμό
αστάθεια σε βλαισότητα/ραιβότητα
πρόσθια ή οπίσθια αστάθεια κατά την εξέταση
```

Avoid repetitive prose if generic CU-1 findings and product-local details overlap.

## 7. Testing requirements

Focused tests must prove:

1. first tap selects generic symptom and shows hint without opening the sheet;
2. second tap opens detail sheet;
3. hint does not appear for Function;
4. each weakness choice renders correct distinct prose;
5. combined flexion+extension weakness is distinct and deterministic;
6. ROM parent/details preserve active/passive + flexion/extension distinctions;
7. no active-flexion detail is mislabeled as extension lag;
8. crepitus renders only when selected;
9. tenderness type/laterality renders without diagnosis inference;
10. bony tenderness alone does not generate atypical review clue;
11. stability findings are objective and do not become subjective giving-way;
12. `>30′` morning stiffness produces a review bubble/clue without changing treatment or diagnosis;
13. international source headings expose safe reviewed links;
14. Cyprus source heading exposes safe reviewed source link;
15. evidence states, selection defaults, safety, no-storage and `CY_GESY` invariants remain unchanged;
16. inherited V5/browser/Cockpit/package regressions pass.

## 8. Out of scope

No:
- SIFK/SONK diagnosis;
- second diagnosis;
- new imaging rule or automatic imaging recommendation;
- new evidence-state classification;
- treatment auto-selection;
- persistence;
- analytics/billing;
- new jurisdiction profile;
- new swelling/enlargement control;
- new flexion-degree measurement field;
- ligament-test protocol/tutorial.

## 9. Replan triggers

Replan instead of patching around the design if:

- existing CU-1 state cannot safely represent the product-local detail without semantic corruption;
- source-link presentation requires weakening URL validation;
- new exam state changes safety or treatment selection unexpectedly;
- the examination sheet becomes materially harder to scan on mobile;
- another owner acquires overlapping writer scope.

## 10. Exit gate

Implementation-complete means:

```text
bounded diff
+ exact-head focused tests
+ inherited V5/clinical-sheet/Cockpit/jurisdiction gates
+ browser acceptance
+ package closure
+ canonical candidate state updated
```

Merge/deploy remain a separate Product Owner release decision after implementation/review.