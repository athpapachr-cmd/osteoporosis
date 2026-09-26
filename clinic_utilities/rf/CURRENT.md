# RF CURRENT — learned medication dictionary

> **STATUS:** IMPLEMENTED / TESTED / HOLD BEFORE RELEASE PR.
> **Workstream:** native RF v2 Clinic Utility.
> **Branch:** `feat/rf-learned-medication-dictionary-2026-09-26`.
> **Base main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Exact tested implementation head:** `6ebd1667ce2640076c84a5b81ad031614f654385`.
> **Regression workflow:** `36222611828` — SUCCESS.
> **Scope:** medication parsing/classification UX only.
> **Root writer lock:** unchanged; PR-1 remains the repo-wide CURRENT_OPERATIONAL owner.

## Product-owner evidence

The clinician reported that real RF medication paste recognized too few medications in practice and explicitly requested a mechanism that lets unknown drugs be manually classified and remembered for future use.

This is new material product evidence and reopens only the bounded RF medication-recognition surface.

## Implemented behavior

```text
paste medication list
→ curated built-in dictionary
→ clinician-learned server-side alias dictionary
→ explicit unrecognized candidates
→ clinician chooses NSAID / other
→ alias is saved as clinician-confirmed
→ next parse recognizes the alias
```

The UI now supports both requested correction paths:

1. **Unknown panel**
   - unmatched pasted lines remain visible;
   - editable suggested medicine name;
   - explicit `ΜΣΑΦ + μάθηση` / `Άλλο + μάθηση`;
   - classification triggers a re-parse.

2. **Manual entry inside the two existing medication columns**
   - `+ Χειροκίνητη προσθήκη ΜΣΑΦ`;
   - `+ Χειροκίνητη προσθήκη άλλου`;
   - current application can use the row without learning it;
   - `Αποθήκευση στο λεξικό` is an explicit separate clinician action.

3. **Learned dictionary management**
   - list clinician-confirmed aliases;
   - reclassify NSAID ↔ other;
   - delete an incorrect learned alias;
   - learned parser results are visibly marked `✓ learned · clinician-confirmed`.

## Storage / safety contract

- learned aliases are stored server-side in `clinic_rf_medication_aliases`;
- only medication alias/display/classification/provenance metadata are stored;
- pasted source lines are not stored in the learned dictionary;
- dose and duration are not dictionary columns;
- aliases containing dose-like tokens are rejected;
- browser localStorage/sessionStorage are not used;
- unknown drugs never default silently to a category;
- built-in curated medication mappings take precedence over conflicting learned aliases;
- learned aliases require explicit clinician classification;
- RF application 0..3 NSAID + 0..3 other capacity remains unchanged;
- no RF indication, PDF, imaging, procedure-history or osteoporosis encounter semantics changed.

## Exact verification evidence

Workflow `36222611828` completed SUCCESS at exact implementation head `6ebd1667ce2640076c84a5b81ad031614f654385`.

Proven in that run:

- RF Python + JavaScript syntax PASS;
- authoritative RF PDF template identity/geometry PASS;
- packaged A1/A2 official PDF generation PASS;
- 18 native RF v2 tests PASS;
- 5 RF release-hardening tests PASS;
- 5 unilateral RF tests PASS;
- **9 new learned-medication parser/persistence/API tests PASS**;
- RF v2 UI integrity PASS;
- RF unilateral UI PASS;
- adjacent CU-1 regression suite PASS;
- legacy RF gateway regressions PASS;
- inherited G4/G3/G2/G1/C1 regressions PASS;
- diff hygiene PASS.

New learned-medication tests prove:

- unknown lines remain visible;
- learned alias is recognized on the next parse;
- built-in mappings win over conflicting learned aliases;
- reclassification and deletion work;
- dictionary schema does not store source text/dose/duration;
- dose-bearing alias is rejected;
- dictionary endpoints remain protected;
- end-to-end learn → parse → reclassify → delete works.

## Current release state

```text
IMPLEMENTED: YES
TESTED: YES
PR OPENED: NO
MERGED: NO
DEPLOYED: NO
PRODUCTION-SMOKE-VERIFIED: NO
```

## Exact next action

Verify this checkpoint head with the RF regression gate. Then hold for a separate release decision/review before opening a PR, merge or deploy. The frozen PR-1 transcript branch remains untouched.
