# CURRENT_OPERATIONAL.md — Knee-OA v1 v5 post-use correction active

> **STATUS:** PHYSIO REFERRAL KNEE-OA V1 — V4 remains released/live; bounded V5 correction ACTIVE on branch.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Released runtime baseline:** PR `#89`, SHA `bf527e3836a18491b2758fd293b42f82e0924382`, Render `dep-daiivp0jo6nc73bl6pug`, live boundary-smoke `34689920602` SUCCESS.
> **Active branch:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12`.
> **ACTIVE RUNTIME / DESIGN WRITER:** bounded Knee-OA v5 correction only.
> **PR / MERGE / DEPLOY AUTHORITY:** NONE until exact-head gates and Product Owner review.
> **Second diagnosis:** NOT AUTHORIZED.

## 1. Trigger

Authenticated Product Owner use of the released Knee-OA surface exposed four bounded defects:

1. selecting a routine clinical symptom immediately opens a modal refinement sheet, making optional refinement feel mandatory;
2. weakness/atrophy controls in `Περισσότερα → Εξέταση` are semantically crowded and visually ambiguous (`Τετρακέφαλος` appears both as weakness detail and atrophy location; `Περιαρθρικά` is not self-explanatory);
3. detailed pain qualifiers can coexist with older pain-location findings and generate duplicate/garbled Greek wording;
4. generated referral prose is too continuous: clinical picture and physiotherapy plan need paragraph separation, while machine-like `Επιπλέον στόχος:` wording should be replaced by natural prose.

## 2. Bounded v5 correction

Allowed changes:

- routine `Πόνος / Δυσκαμψία / Αδυναμία`: first tap selects the parent without opening a sheet; re-tap opens optional refinement;
- `Λειτουργικότητα` may still open its chooser because no generic functional-impairment state exists;
- simplify weakness/exam labels and remove new creation of vague peri-knee atrophy location from the UI while preserving backward compatibility in validation;
- deduplicate specific pain-location semantics before deterministic rendering;
- paragraph break before physiotherapy assessment/plan;
- replace `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` with natural connected Greek wording.

Hard boundaries:

```text
no new diagnosis
no evidence update
no safety-rule change
no new jurisdiction rule
no patient persistence
no analytics
no autonomous recommendation behavior
```

## 3. Release hold

V4 production remains authoritative until v5 passes exact-head server/clinical/browser/Cockpit regressions and receives Product Owner visual/use acceptance. No PR/merge/deploy claim may be made early.
