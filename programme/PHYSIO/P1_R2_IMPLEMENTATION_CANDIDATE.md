# PHYSIO P1 R2 — bounded implementation candidate

> **Date:** 2026-10-05 Asia/Nicosia
> **Base:** fresh `main` `b9beca7b1e233245f9ea3429a247a1c80a69ac7c`
> **Branch:** `feat/physio-r2-ia-correction-2026-10-05`
> **Authority:** Product Owner-approved `P1_R2A_STRUCTURAL_CORRECTION.md`; independent pre-code PASS in `P1_R2A_CLOSURE_REVIEW_RESULT.md`.
> **Status:** bounded correction after first post-code exact-head BLOCK; local correction tests pass, independent closure and PR CI remain pending. No release or pilot claim.

## Implemented boundary

The physician workflow now presents four direct sections: Κλινική εικόνα, Λειτουργικότητα, Εξέταση and Προτεινόμενο πλάνο. The historical second-tap detail and mixed `Περισσότερα` surfaces are hidden/inert in the routine product. One visible control owns each functional impairment. Routine structured pain locations, tenderness, crepitus and separate effusion controls are absent. Standalone Οίδημα is visible, initially unselected, non-mandatory and never treated as an examined negative when untouched.

Proposed Plan contains its own collapsed additional options. Walking-aid assessment/training appears only there, starts off and is excluded from suggestions. All selected additional directions and adjuncts remain editable inside that disclosure; the immediate plan shows only core choices and non-selecting categorized summaries of extras. Selecting walking aid explicitly retains deterministic text and evidence. Optional chronicity remains directly in Clinical Picture; explicit restrictions and clinical notes remain available in Proposed Plan. An incomplete restriction draft disables export without submitting an invalid payload. Evidence, default active plan, CY_GESY and CU-1 input/safety contracts remain in place.

The already-approved R2-B product-local review cue uses the reviewed combination of explicit observations. Simple swelling or hot/swollen observation alone does not block. A strong acute-joint or alternative-subchondral-pathology pattern requires an explicit Continue or Defer decision before export. Defer withholds text/export; Continue permits the ordinary referral only if the independent CU-1 gate also permits it. A CU-1 unresolved concern remains blocking even after Continue. No diagnosis or imaging is inferred.

## Same Product Owner Cases 1–5 — automated mobile regression

The five exact narratives in `P1_PRODUCT_OWNER_REAL_USE_REGRESSION_CASES.md` were replayed as synthetic browser interactions at 390 px width. Approximate taps below count the scripted decisions needed to reach the tested endpoint; they do not measure clinician reading or typing time.

| Case | Approx. actions | Search/backtracking in scripted route | Manual edit | Automated endpoint |
|---|---:|---|---|---|
| 1 routine right OA | 5 to ready referral; 1 more to verify deselection | none | none | functional choices have one owner; output ready |
| 2 bilateral stiffness/chronicity | 10 including duration entry | none | none | inline stiffness and chair-rise found; output ready |
| 3 objective exam | 10 including opening plan options and explicit aid selection; 1 more to verify in-place deselection | none | none | quadriceps/active flexion visible; retired controls absent; aid stays only in additional options; output ready |
| 4 rapid worsening | 6 | none | none | no SIFK inference or automatic review gate |
| 5 acute joint | 8 through Defer; 1 further Continue and 1 CU-1 check | none | none | pending and Defer prohibit copy; Continue remains CU-1 constrained |

The full five-case automated browser run completed in 4.2 seconds on the local test host. This is machine execution time, **not** a human completion-time estimate. Product Owner referral acceptability, real-device search burden and iPhone Safari/VoiceOver acceptance require a later real-use replay; no new Product Owner acceptance is claimed.

## Local gates

- R2 prototype mobile browser: **6/6 PASS** (Cases 1–5 plus additional-plan/restriction edge).
- Protected Cockpit R2 mobile/CY_GESY browser: **4/4 PASS**.
- PHYSIO semantics, HTTP boundary, CY_GESY, Cockpit integration and adjacent owner: **74/74 PASS**.
- Exact focused CU-1 gate: **55/55 PASS**.
- Evidence, template and interaction validators: **PASS**; interaction parent pins updated to exact revised blobs.
- Adjacent G4 and RF JavaScript checks: **PASS**.
- Standalone synthetic prototype dependency closure: **PASS**.
- Static and synthetic prototype R2 JavaScript sources are identical; JavaScript syntax, workflow YAML parse and `git diff --check` pass.

The old browser suites specifically asserted the retired second-tap, mixed drawer, tenderness and crepitus UI. Their current-workflow gate slots now execute the five-case R2 browser suite with a plan/restriction edge and protected Cockpit R2 browser smoke. Legacy parser and deterministic-output coverage remain in the semantic tests. CI must rerun on the corrected exact head/PR.

## Scope and next gate

No shared CU-1 runtime file, second diagnosis, patient persistence, release setting or deployment changed. The first post-code exact-head review returned **BLOCK** at `c6eb29548a5446289a17a843c650d362b0db3866`; see `P1_R2_POSTCODE_REVIEW_RESULT.md`. The corrected head requires one independent delta + affected-cumulative closure review. A closure PASS permits a bounded PR for review; merge/deploy require separate authority.
