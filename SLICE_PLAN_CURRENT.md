# SLICE_PLAN_CURRENT.md — Knee-OA v1 v5 optional refinement + prose correction

> **STATUS:** ACTIVE / BOUNDED IMPLEMENTATION.
> **Slice:** `CU1-PRODUCT-KNEE-OA-POSTUSE-V5-20260912`.
> **Parent production runtime:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Branch:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12`.
> **Writer:** bounded v5 correction only.
> **PR / MERGE / DEPLOY:** HOLD.

## 1. Product findings to correct

### A. Optional refinement must feel optional

For routine symptom parents (`Πόνος`, `Δυσκαμψία`, `Αδυναμία`):

```text
first tap while inactive
→ select generic symptom
→ remain on main surface

second tap while active
→ open focused refinement sheet
```

No qualifier is required. A generic symptom remains valid structured input. `Λειτουργικότητα` continues to open its chooser because the product has no generic function-only state.

### B. Weakness / atrophy clarity

New UI must distinguish concepts explicitly:

- `Μυϊκή αδυναμία στην εξέταση`
- `Αδυναμία τετρακεφάλου στην εξέταση`
- `Ατροφία τετρακεφάλου`

Do not expose bare `Τετρακέφαλος` and bare `Περιαρθρικά` as sibling controls. Existing `peri_knee_general` payload acceptance may remain for backward compatibility, but v5 must not create it through the routine/advanced UI.

### C. Pain prose deduplication

If structured pain-location qualifiers exist, they own pain-location specificity for product prose. Older overlapping `joint_line_pain` / `anterior_peripatellar_pain` findings must not generate a second location phrase in the same sentence.

### D. Referral readability

Generated referral should use two human-readable paragraphs when a clinical/functional section and a physiotherapy plan are both present:

```text
[indication + clinical picture + functional impact]

[physiotherapy assessment / active plan + additional goal emphasis]
```

Replace machine-like labels:

```text
Επιπλέον στόχος: ...
Επιπλέον στόχοι: ...
```

with connected natural prose, preferably `Παράλληλα, επιδιώκεται ...` unless exact grammar requires a narrower variant.

## 2. Preserved invariants

- symptom != objective finding != diagnosis;
- qualifier refinement remains optional;
- explicit quadriceps weakness requires examination semantics;
- atrophy is a distinct examination finding, not a synonym for weakness;
- evidence/suggestion/safety state unchanged;
- no hidden stale dependent data;
- manual text reconciliation unchanged;
- More-v3/Favorites behavior unchanged except for label simplification inside Examination;
- no new patient persistence.

## 3. Acceptance

Must prove at exact head:

- inactive Pain/Stiffness/Weakness first tap does not open the sheet;
- second tap opens the correct sheet;
- generic symptom can be exported without any qualifier;
- Function chooser remains explicit;
- no ambiguous duplicate quadriceps/periarticular controls in routine or advanced exam UI;
- pain qualifiers do not duplicate `joint line` wording;
- two-paragraph referral formatting survives clipboard/manual-edit paths;
- no literal `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` in generated Knee-OA output;
- inherited v2/v3/v4, CU-1 and protected Cockpit regressions pass.

## 4. Release boundary

Implementation/test only. Product Owner reviews exact tested candidate before any PR/merge/deploy decision.
