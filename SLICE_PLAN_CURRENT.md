# SLICE_PLAN_CURRENT.md — Knee-OA v1 bounded v5 usability/prose refinement

> **STATUS:** ACTIVE / DESIGN FROZEN FOR BOUNDED IMPLEMENTATION.
> **Branch:** `fix/physio-knee-oa-usability-v5-2026-09-12`.
> **Bootstrap main:** `90e4377fc92e76d767a4b911a0dcff523b50b71e`.
> **Production runtime before slice:** `bf527e3836a18491b2758fd293b42f82e0924382`.
> **Authenticated live production smoke:** run `34681808255`, attempt `3` — SUCCESS.
> **Writer:** active branch above only.
> **Diagnosis vertical:** Knee Osteoarthritis only.

## 1. Problem

Product-owner use found three bounded issues after the released v4 clinical-picture refinement:

1. `Πόνος`, `Δυσκαμψία` and `Αδυναμία` open a modal sheet on first selection even though their qualifiers are optional;
2. weakness/examination wording exposes too many overlapping controls and makes two distinct quadriceps semantics look duplicated;
3. dense deterministic referral prose lacks a visual paragraph boundary, uses the mechanical label `Επιπλέον στόχος`, and can fuse a richer pain qualifier with a legacy specific pain phrase.

These are workflow/presentation defects, not evidence changes.

## 2. In scope

### 2.1 Optional-detail interaction

For `Πόνος`, `Δυσκαμψία`, `Αδυναμία`:

```text
first tap while inactive
→ assert the existing parent fact only
→ update live referral
→ do NOT open modal

second tap while active
→ open existing focused detail sheet
```

The active card should remain discoverable as having optional detail without adding a permanent new form control.

`Λειτουργικότητα` remains different: its value is the selected functional impairment(s), so opening its focused choice sheet on tap remains correct rather than inventing a generic function fact.

### 2.2 Weakness detail reduction

Routine focused sheet becomes exactly:

- `Μυϊκή αδυναμία στην εξέταση` → existing generic objective-weakness semantic;
- `Αδυναμία τετρακεφάλου` → existing explicit quadriceps examination semantic;
- `Εμφανής ατροφία τετρακεφάλου` → existing visible-atrophy semantic with `atrophy_location=quadriceps`.

Remove `Περιαρθρικά` and the separate nested atrophy-location decision from the routine surface. Do not delete the underlying compatibility enum merely to simplify presentation.

### 2.3 Referral readability

Presented deterministic referral should use:

```text
[indication + clinical picture + functional impact]

[physiotherapy assessment + active plan + nonredundant functional priority]
```

Use `Επιπρόσθετη λειτουργική προτεραιότητα:` / plural equivalent instead of `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` on the presented surface.

### 2.4 Pain-location overlap correction

When a detailed pain qualifier already expresses a more specific location, presentation must not fuse it with an overlapping legacy specific phrase. The correction is output/presentation reconciliation only; structured clinician-selected state is not silently rewritten.

## 3. Explicitly out of scope

- Cyprus/GeSY evidence or administrative rules;
- any jurisdiction-overlay implementation;
- evidence-state/default/suggestion changes;
- safety-rule changes;
- second diagnosis;
- new persistence or analytics;
- patient data;
- new treatment recommendations;
- autonomous evidence updates;
- broad More-v3 redesign;
- new functional-baseline field.

## 4. Product principles preserved

```text
CLINICALLY INTERESTING != WORKFLOW-USEFUL != RECEIVER-USEFUL != WORTH ADDING
PRODUCT OWNER REQUEST != EVIDENCE != IMPLEMENTATION AUTHORITY
```

This slice reduces decisions rather than adding controls. Power remains underneath; routine surface remains sparse; detailed state remains reachable by progressive disclosure.

## 5. Acceptance tests

1. inactive Pain/Stiffness/Weakness first tap selects parent and leaves dialog closed;
2. second tap on that active parent opens its detail sheet;
3. explicit remove still clears dependent qualifier meaning;
4. Function still opens its detail sheet immediately;
5. weakness sheet has exactly the three routine choices above and no `Περιαρθρικά` routine choice;
6. atrophy choice yields explicit quadriceps atrophy prose;
7. clinical/functional and physiotherapy-plan sections are separated by a blank line in projected/copied text;
8. presented text contains no `Επιπλέον στόχος:` or `Επιπλέον στόχοι:`;
9. detailed medial/pes pain plus overlapping legacy joint-line pain does not render a fused `...χηνείου ποδός στη μεσάρθρια γραμμή` phrase;
10. inherited evidence/safety/manual-edit/mobile/no-browser-storage gates remain PASS.

## 6. REPLAN triggers

Stop and replan if the correction would require:

- changing frozen CU-1 clinical meaning;
- inventing a new parent functional state;
- changing evidence/default recommendations;
- deleting compatibility data needed by existing structured states;
- changing jurisdiction semantics;
- weakening an inherited safety or stale-text guard.

## 7. Release boundary

Implementation/test completion does not itself imply merge/deploy. After exact-head PASS, update canonicals and present the bounded result for release review. No second diagnosis or Cyprus/GeSY runtime rule is authorized by this slice.