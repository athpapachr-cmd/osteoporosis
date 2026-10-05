# PHYSIO P1 R2-A — frozen structural correction design

> **Task:** `PHYSIO-P1-R2A-STRUCTURAL-IA-CORRECTION-20261004`.
> **Status:** Product Owner decisions 1–3 approved; correction design frozen on 2026-10-05; independent R2-A delta + affected-cumulative closure review pending; runtime implementation not yet started.
> **Base:** fresh `main` `b9beca7b1e233245f9ea3429a247a1c80a69ac7c`.
> **Authority:** Product Owner's 2026-10-04 approval of items 1 and 3 and 2026-10-05 approval of item 2 and bounded implementation request. The earlier coordinator proposals in this file were not authority before those approvals.
> **R2-B:** pre-code semantic PASS in `P1_R2_PRECODE_REVIEW_RESULT.md`; preserve its review-cue design and explicit CU-1 safety precedence.

## 1. Single owner for functional concepts — approved

The routine UI writes `functional_impairments` from **Λειτουργικότητα**. Equivalent pairs:

| Compatibility-only finding | UI-owned functional impairment |
|---|---|
| `walking_limitation` | `walking_tolerance` |
| `stairs_limitation` | `stairs` |
| `sit_to_stand_limitation` | `sit_to_stand` |
| `sport_or_exercise_limitation` | `sport_gym` |

The four finding IDs remain accepted only as legacy machine inputs. They have no selectable control or second owner. Existing deterministic alias/deduplication projects either form to one functional phrase; mixed legacy/new input must never duplicate prose. Other functional impairments also belong in Λειτουργικότητα, not Examination or a generic drawer.

## 2. Routine clinical fields — approved

The Product Owner accepted the simplification after `P1_CLINICAL_FIELD_UTILITY_REVIEW.md`:

| Concept | Routine physician UI disposition | Compatibility / semantic boundary |
|---|---|---|
| Structured pain-location map (`pain_locations[]`) and overlapping `joint_line_pain` / `anterior_peripatellar_pain` selectors | Remove | Existing accepted machine inputs and deterministic projection stay compatible; no new routine pain-location selector. Relevant location can be stated in the existing free note. |
| `tenderness` and `focal_tenderness_locations[]` | Remove | Existing input/output compatibility stays; no routine palpation selector or map. |
| `crepitus` | Remove | Existing qualifier compatibility stays; no routine selector. |
| Separate `effusion` | Remove | Existing machine input/output compatibility stays; do not make an unselected field an examined negative. |
| `swelling` / **Οίδημα** | **Keep visibly in Κλινική εικόνα** as one standalone control | Initially unselected, not required to complete referral. Unselected means not recorded, not proven absent (`missing != negative`). One tap selects or deselects; no grade, effusion subtype, popup or second tap. |
| `hot_swollen_joint` and other R2-B observations | Keep visibly in Clinical Picture / review context where relevant | Explicit observation is distinct from simple Οίδημα and legacy `effusion`; it feeds only the reviewed product-local pattern cue. |

Simple `swelling` **alone never triggers a red-flag or septic-arthritis alert**. The already-PASS R2-B hot/swollen concern requires its reviewed combination of explicit observations and leads to a clinician review cue and Continue/Defer disposition, not diagnosis, automatic imaging or a CU-1 safety flag. Unknown/unselected values remain unknown. The existing explicit CU-1 unresolved-safety block takes precedence independently. R2-B SIFK/alternative-pathology support may still read legacy bony tenderness/effusion when supplied, but neither becomes a routine selector or required trigger merely to preserve that support.

## 3. Walking aid — approved

`walking_aid_assessment_and_training` remains a supported rehabilitation direction with one UI home:

```text
Προτεινόμενο πλάνο → Πρόσθετες / Περισσότερες επιλογές
→ Αξιολόγηση / εκπαίδευση στη χρήση βοηθήματος βάδισης
```

It is default-off and absent from immediate main plan choices. No automatic suggestion, automatic selection or linked goal selection. Explicit selection uses the existing deterministic referral phrase and evidence interaction. The choice never returns to miscellaneous `Περισσότερα`.

## 4. Frozen target structure and interaction

1. **Κλινική εικόνα:** symptoms, straightforward optional qualifiers, visible Οίδημα, and explicit atypical observations/review context. First tap on a parent exposes optional detail immediately. Selection and deselection remain visible without scrolling to the bottom of a sheet.
2. **Λειτουργικότητα:** one owner for walking, stairs, chair rise and other `functional_impairments`.
3. **Εξέταση:** actionable objective weakness, ROM, atrophy/stability and other retained objective findings. Subjective weakness stays in Clinical Picture; objective weakness has one Examination owner. Removed routine tenderness, crepitus and effusion controls cannot reappear through an advanced route.
4. **Προτεινόμενο πλάνο:** reviewed default active-rehabilitation plan, explicit additional rehabilitation options including walking aid, and clearly subordinate adjuncts. Additional options may use a plan-specific disclosure label; no generic mixed-semantic drawer.

Remove hidden same-card second-tap dependency and competing `Λεπτομέρειες` navigation. Retire generic miscellaneous `Περισσότερα` as mixed clinical/exam/function/rehabilitation destination. Evidence links, selected summaries and cue messages remain non-selecting projections; each editable concept has one control owner.

Existing deterministic referral output, evidence classifications, default plan and `CY_GESY` overlay remain unchanged except for deduplicated equivalent function input and visibility of Product Owner-selected controls. No shared CU-1 mutation, second diagnosis, patient/referral persistence, automatic diagnosis or imaging, or new evidence engine is in scope.

## 5. Bounded implementation and regression oracle

After **one independent R2-A delta + affected-cumulative closure PASS**, implement only this owner/control correction and the already-PASS R2-B product-local cue workflow. Preserve legacy transport/projection compatibility while the new UI emits only selected owners. Verify first-tap details, immediate deselection, no second Details route, no generic mixed More, visible Οίδημα with `missing != negative`, walking aid only under Proposed Plan additional options, and unchanged default/evidence/CY_GESY behavior.

Reuse the **same** `P1_PRODUCT_OWNER_REAL_USE_REGRESSION_CASES.md` Cases 1–5; do not replace frozen `P1_SYNTHETIC_CASE_SET.md`. Record completion time, approximate taps, search/backtracking, manual edit, output acceptability and cue/disposition. Case 4 rapid worsening alone must not trigger acute-joint or SIFK cue. Case 5 must allow explicit Defer/reassessment without diagnostic or imaging automation, while a separately selected unresolved CU-1 concern still blocks.

## 6. One closure review, then stop

The independent reviewer decides whether all three original R2-A findings are closed across the changed owner/control graph, legacy projection, swelling versus hot/swollen cue input, walking-aid placement and affected consumers. Check for new material overlap or shared-Core requirement. Reuse existing R2-B PASS; do not restart that review without source-proven new risk.

```text
all three R2-A findings closed
+ affected owner/control graph complete
+ no new material risk
→ PASS → stop review chain → bounded implementation may start
```

If BLOCK or unresolved, checkpoint the finding and stop before runtime work.
