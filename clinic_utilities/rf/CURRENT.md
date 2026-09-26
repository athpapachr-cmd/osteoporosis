# RF CURRENT — learned medication dictionary

> **STATUS:** SECOND INDEPENDENT HOLD_FOR_RF_REMEDIATION / STRUCTURAL ALIAS-IDENTITY HARDENING AUTHORIZED.
> **Workstream:** native RF v2 Clinic Utility.
> **Branch:** `feat/rf-learned-medication-dictionary-2026-09-26`.
> **Base main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Original tested implementation head:** `6ebd1667ce2640076c84a5b81ad031614f654385`.
> **Corrected remediation head:** `f345b0bf557ea5079d49793512cf550ab614988d`.
> **Implementation regression workflow:** `36222611828` — SUCCESS.
> **Checkpoint verification workflow:** `36222684337` — SUCCESS.
> **Independent HOLD checkpoint workflow:** `36228058129` — SUCCESS.
> **Alias-remediation regression workflow:** `36228138003` — SUCCESS.
> **Corrected checkpoint verification workflow:** `36228202897` — SUCCESS.
> **Final corrected sidecar verification:** `36228254994` — SUCCESS.
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

## Independent release review — HOLD_FOR_RF_REMEDIATION

The fresh independent READ-ONLY review verified branch isolation, ancestry, CI evidence, API/auth/privacy behavior and the intended medication-learning design, but found one HIGH release blocker:

```text
generic/non-medication aliases such as:
50
mg
mcg
tablet/formulation markers
can currently be stored as clinician-learned aliases
```

Because learned matching uses whole-token matching, a mistaken alias such as `mg → nsaid` can match an unrelated unknown line such as `Mysteron 50 mg 2 μήνες` and silently move it from UNKNOWN into a learned category.

Independent disposition:

```text
HOLD_FOR_RF_REMEDIATION
```

No other demonstrated finding was reported.

## Alias-validation remediation — COMPLETE deterministically

The independent HIGH finding was remediated without changing parser category fallback or UI learning semantics.

Corrected validator behavior:

```text
pure numeric alias
→ reject

standalone dose-unit alias
→ reject

generic formulation/tablet/capsule/syrup/gel/patch/injection marker
→ reject

valid medication name
→ clinician learning still allowed
```

Examples now rejected include:

```text
50
50.0
mg
mcg
tablet
tablets
δισκίο
χάπια
capsule
syrup
gel
patch
injection
```

Exact corrected implementation head:

```text
f345b0bf557ea5079d49793512cf550ab614988d
```

Workflow `36228138003` completed SUCCESS across the complete RF gate.

The medication-learning suite expanded from 9 to **13 tests** and now proves:

- pure numeric aliases fail closed;
- generic dose/form aliases fail closed;
- API rejects generic aliases;
- attempted `mg → nsaid` learning leaves the dictionary empty;
- after that attempted poisoning, `Mysteron 50 mg 2 μήνες` remains in `unrecognized_candidates`;
- valid medication names such as `Xefo` remain learnable;
- all original learning, reclassification, deletion, privacy and built-in precedence tests continue to pass.

No PR, merge or deploy occurred.

## Second independent release re-review — HOLD_FOR_RF_REMEDIATION

The fresh re-review confirmed the first poisoning example was closed but found the same safety class remained possible through generic regimen/form/route/frequency aliases such as:

```text
XR
SR
MR
PRN
PO
daily
```

A learned alias such as `XR → nsaid` could still match the interior token in `Mysteron XR 50 mg` and silently classify an otherwise unknown medicine.

Independent disposition:

```text
HOLD_FOR_RF_REMEDIATION
```

No other demonstrated finding was reported.

## Second bounded remediation contract

Authorized mutation is limited to the learned-alias identity/matching boundary and focused tests:

1. reject aliases composed only of medication metadata tokens such as release modifiers, route, frequency, dose-unit or formulation markers;
2. learned aliases must match the medication-identity prefix of a line, not an arbitrary interior token;
3. preserve valid multi-word/single-word medication aliases and curated built-in matching;
4. prove `XR → nsaid` and equivalent metadata learning fails closed;
5. prove `Mysteron XR 50 mg` remains unrecognized after attempted invalid learning;
6. preserve all existing RF behavior and storage/privacy rules.

No PR, merge, deploy or unrelated RF change is authorized.

## Exact next action

Verify this second HOLD checkpoint. Then implement only the structural alias-identity hardening and focused regressions, rerun the complete RF gate, checkpoint the corrected head, and require another fresh independent READ-ONLY re-review before any release PR.
