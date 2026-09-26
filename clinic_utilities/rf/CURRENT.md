# RF CURRENT — learned medication dictionary

> **STATUS:** RF RELEASE PR #116 MERGED / RENDER AUTO-DEPLOY LIVE / RELEASE COMPLETE / AUTHENTICATED RF USER-FLOW SMOKE NOT EXECUTED.
> **Workstream:** native RF v2 Clinic Utility.
> **Branch:** `feat/rf-learned-medication-dictionary-2026-09-26`.
> **Base main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Original tested implementation head:** `6ebd1667ce2640076c84a5b81ad031614f654385`.
> **First corrected remediation head:** `f345b0bf557ea5079d49793512cf550ab614988d`.
> **Structural alias-safety head:** `946d464829406718305ade9f4019314e2a64b497`.
> **Orthographic/structured alias-safety head:** `ab21527aee526b43f1907574d8fddcd354c90387`.
> **Implementation regression workflow:** `36222611828` — SUCCESS.
> **Checkpoint verification workflow:** `36222684337` — SUCCESS.
> **Independent HOLD checkpoint workflow:** `36228058129` — SUCCESS.
> **Alias-remediation regression workflow:** `36228138003` — SUCCESS.
> **Corrected checkpoint verification workflow:** `36228202897` — SUCCESS.
> **Final corrected sidecar verification:** `36228254994` — SUCCESS.
> **Structural alias-remediation workflow:** `36229316225` — SUCCESS.
> **Structural remediation checkpoint verification:** `36229374537` — SUCCESS.
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

## Structural alias-identity remediation — COMPLETE deterministically

The second independent HIGH finding was closed with two complementary fail-closed controls:

1. **Identity-bearing alias validation**
   - an alias must contain at least one token that is not numeric and not medication metadata;
   - release, route, frequency, dose-unit and formulation-only aliases are rejected;
   - examples rejected include `XR`, `SR`, `MR`, `PRN`, `PO`, `daily`, `oral`, `IV`, `BID`, `night`, `forte`;
   - valid names such as `Xefo` remain learnable;
   - valid compound aliases such as `Mysteron XR` remain learnable because they contain the identity token `Mysteron`.

2. **Prefix-only learned matching**
   - learned aliases no longer match arbitrary interior tokens in a medication line;
   - generic leading metadata/list markers may be skipped;
   - the learned medication identity must then match at the beginning of the remaining line;
   - unsafe/stale learned entries are ignored defensively by the parser even if they somehow exist outside the normal storage validator.

Exact tested structural-remediation head:

```text
946d464829406718305ade9f4019314e2a64b497
```

Workflow `36229316225` completed SUCCESS across the complete RF gate.

The medication-learning suite now contains **20 PASS** and proves, among other things:

- release/route/frequency metadata-only aliases fail closed;
- API rejects those aliases;
- `XR → nsaid` cannot be learned;
- `Mysteron XR 50 mg` remains unrecognized after that poisoning attempt;
- an unsafe stale `XR` learned entry is ignored by the parser;
- a learned `Mysteron` matches `Mysteron XR 50 mg`;
- that same alias does not match an interior occurrence such as `OtherDrug Mysteron 50 mg`;
- `Tablet Mysteron 50 mg` still recognizes the learned `Mysteron` after stripping leading generic form metadata;
- `Mysteron XR` itself remains a valid learnable medication alias.

All native RF, PDF, UI, CU-1, gateway, inherited G4/G3/G2/G1/C1 and diff-hygiene checks also passed.

No PR, merge or deploy occurred.

## Orthographic / structured alias remediation — COMPLETE deterministically

The third independent HIGH finding identified representation-sensitive metadata aliases such as dotted routes, dotted frequencies, structured regimens and numeric ranges.

The current implementation now canonicalizes medication alias text before both storage validation and learned matching:

```text
I.V.            → iv
P.O.            → po
B.I.D.          → bid
Q.I.D.          → qid
extended-release→ extended release
50-100          → 50 100
```

Medication identity requires an alphabetic non-metadata identity token at the beginning of the alias. Tokens containing digits are treated as regimen/strength metadata rather than medication identity.

The same canonical tokenization is used by parser-side defensive matching, so stale/bypassed metadata-only rows are not trusted.

Exact tested product head:

```text
ab21527aee526b43f1907574d8fddcd354c90387
```

Workflow `36231015466` completed SUCCESS across the complete RF gate.

Checkpoint verification workflow `36231071391` also completed SUCCESS.

The medication-learning suite now reports **28 PASS** and proves, among other things:

- dotted metadata aliases `I.V.`, `P.O.`, `B.I.D.`, `Q.I.D.` are canonicalized and rejected as identities;
- structured regimens `q8h`, `q12h`, `2x`, `2xday` are not medication identities;
- numeric composite/range forms `50-100`, `50/100` are rejected;
- `extended-release` is not a medication identity;
- API rejection mirrors persistence validation;
- attempted `I.V. → nsaid` learning leaves the dictionary empty;
- `I.V. Mysteron 50 mg` remains unrecognized after that poisoning attempt;
- an unsafe stale dotted-route learned entry is ignored by the parser;
- a valid learned `Mysteron` still matches after a dotted route prefix;
- hyphenated valid alias `Mysteron-XR` canonicalizes to `mysteron xr` and remains usable.

All native RF, release-hardening, unilateral, UI, CU-1, gateway, inherited G4/G3/G2/G1/C1 and diff-hygiene checks passed.

No PR, merge or deploy occurred.

## Final independent bounded release re-review — PASS

Fresh independent READ-ONLY re-review verified:

- current `main = 0ab5f9770d220c20e8d94544cb64e93a4aa30d00`;
- current RF branch head `58df7260d405a831426cb75919668e85408f0565`;
- merge-base = current main;
- ahead/behind = 27 / 0;
- no RF release PR existed at review time;
- the prior generic/orthographic learned-alias poisoning HIGH is **CLOSED** at both persistence/API validation and parser-side stale-entry defense;
- no BLOCKER/HIGH/MEDIUM/LOW finding was demonstrated;
- product gate `36231015466` SUCCESS with 28 medication-learning tests;
- checkpoint verification `36231071391` SUCCESS;
- final full verification `36231122209` SUCCESS;
- no product-code mutation occurred after the tested product head `ab21527aee526b43f1907574d8fddcd354c90387`.

Independent disposition:

```text
PASS_TO_RF_RELEASE_PR
```

This authorizes only the controlled release-PR transition. It does not authorize merge, deploy or production smoke claims.

## RF release PR opened

Release PR:

```text
#116
https://github.com/athpapachr-cmd/osteoporosis/pull/116
```

PR transition identity:

```text
base: main
base_sha: 0ab5f9770d220c20e8d94544cb64e93a4aa30d00
head: feat/rf-learned-medication-dictionary-2026-09-26
head_sha at PR creation: c6725a72f3ce1733fe1227cf48378c8ec5194a9f
state: OPEN
draft: NO
merged: NO
deploy: NOT AUTHORIZED
```

The PR body includes the repository-required Canonical Impact Declaration:

```text
release_affecting: yes
checkpoint_stage: release_hold
root_current: none
slice_plan: none
todo: none
clinical_excellence_plan: none
workstream_current: update
workstream_current_path: clinic_utilities/rf/CURRENT.md
changelog: defer_until_completion
```

Opening PR #116 is a release-review transition only. It does not authorize merge, deployment or production smoke.

## Merge and deployment completion

PR #116 completed the controlled release transition.

```text
PR: #116
merge method: squash
merge commit: c0b89f9c49239142e94e0630580771180d6fcadb
merged: YES
merged_at: 2026-09-26T17:18:40Z
```

Render production service:

```text
service: osteoporosis
service_id: srv-d5qfk31r0fns73di596g
branch: main
autoDeploy: yes
deploy_id: dep-darvssavcj2c73adg7vg
deploy_commit: c0b89f9c49239142e94e0630580771180d6fcadb
trigger: new_commit
status: live
finished_at: 2026-09-26T17:20:10Z
```

No manual redeploy was triggered.

Production startup evidence from Render logs:

- build successful;
- Uvicorn process started;
- application startup complete;
- PostgreSQL clinical storage configured online;
- clinical key configured;
- Render reported the service live;
- root/static service request returned successfully after startup.

The RF routes remain protected by the existing clinical-key boundary. This release session did not retrieve or expose the production clinical key, so an authenticated end-to-end RF medication-learning user-flow smoke was not executed.

Release state:

```text
IMPLEMENTED: YES
INDEPENDENT RELEASE REVIEW: PASS
PR MERGED: YES
AUTO-DEPLOY: LIVE
MANUAL REDEPLOY: NO
SERVICE STARTUP: VERIFIED
AUTHENTICATED RF FEATURE SMOKE: NOT EXECUTED
```

## Exact next action

HOLD. No further repository or deployment mutation is required for this release. If an authenticated production RF medication-learning smoke is performed later through the normal clinical UI, record that evidence here; otherwise keep the current release state as deployed/live with authenticated feature smoke not executed.
