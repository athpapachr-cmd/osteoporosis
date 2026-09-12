# SLICE_PLAN_CURRENT.md — Knee-OA v1 v5 tested candidate

> **STATUS:** IMPLEMENTED / EXACT-HEAD TECHNICAL GATE PASS / PRODUCT OWNER REVIEW NEXT.
> **Slice:** `CU1-PRODUCT-KNEE-OA-POSTUSE-V5-20260912`.
> **Parent production runtime:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Branch:** `fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12`.
> **Tested substantive head:** `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **Successful gate:** `34693751545`.
> **Artifact:** `10298267537`, digest `sha256:a22fd402c49efcf3a9db20cc7751ed74363264437b911cd62c9de772fd520315`.
> **Writer:** NONE.
> **PR / MERGE / DEPLOY:** HOLD pending Product Owner review.

## 1. Implemented bounded corrections

### Optional refinement interaction

For routine symptom parents:

```text
Πόνος / Δυσκαμψία / Αδυναμία
first inactive tap → select generic symptom, no sheet
second active tap   → open optional focused refinement sheet
```

No qualifier is required for export. `Λειτουργικότητα` keeps its explicit chooser on first tap because the model has no useful generic function-only state.

### Weakness / atrophy simplification

Visible exam choices are now semantically explicit:

- `Μυϊκή αδυναμία στην εξέταση`
- `Αδυναμία τετρακεφάλου στην εξέταση`
- `Ατροφία τετρακεφάλου`

The UI no longer exposes bare `Τετρακέφαλος` or bare `Περιαρθρικά`. Existing legacy `peri_knee_general` validation remains backward-compatible but is not newly created by v5 UI.

### Pain prose deduplication

Structured pain-location qualifiers own location specificity. Legacy overlapping pain-location findings cannot add a second location phrase to the same referral sentence.

### Referral readability

The deterministic referral uses a paragraph break between clinical/functional information and the physiotherapy plan when both exist. Machine-like `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` labels are replaced by connected human prose.

## 2. Preserved invariants

```text
symptom != objective finding != diagnosis
atrophy != weakness
suggestion != clinician selection
evidence state != selection state
manual text != structured state
```

Preserved unchanged:

- CU-1 taxonomy and safety authority;
- evidence-state model;
- suggestion semantics;
- More-v3 / Favorites architecture apart from exam-label simplification;
- manual-text reconciliation;
- jurisdiction strategy;
- single Knee-OA diagnosis vertical;
- no patient persistence / analytics.

## 3. Exact acceptance evidence

Run `34693751545` at substantive head `a47357c602120d3678e8f2f23b99775e616c79e1` completed `SUCCESS` with:

- focused v5 projection/prose tests 3/3 PASS;
- real server/HTTP tests 15 PASS;
- frozen Step-3 exact-output fixtures 15 PASS;
- qualifier/clinical tests 11/11 PASS;
- protected Cockpit integration 6/6 PASS;
- focused v5 Chromium PASS;
- inherited browser suites v2/v3/v4 PASS;
- protected Cockpit Chromium PASS;
- adjacent-owner isolation PASS;
- package closure PASS.

The tested rich-case referral has no `κυρίως`, no duplicated `στη μεσάρθρια γραμμή` tail, contains the clinical/plan paragraph break, and contains no literal `Επιπλέον στόχος:` label.

## 4. Exact next action

Product Owner visually/use-tests the exact tested v5 candidate and returns concrete `keep / change / remove` feedback.

Do not open a PR, merge, deploy, activate a second diagnosis or broaden scope until that review is complete and explicit release authority is granted.
