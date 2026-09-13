# CURRENT_OPERATIONAL.md — Physiotherapy Referral Knee-OA V5 integration patch

> **STATUS:** IMPLEMENTED / EXACT-HEAD TESTED — PRODUCT OWNER RELEASE HOLD.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-INTEGRATION-2026-09-13`.
> **Bootstrap main:** `8aeb91ae37b83caaa188054128db98e04b638fd8`.
> **Integration branch:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Exact tested integrated head:** `9a7f360745710deacb0ff82f03249723bdfe87d6`.
> **Integration gate:** run `34736389860` — SUCCESS.
> **Artifact:** `10311108002` / `sha256:c51e3e29c4057b0e90773011d54648703eca5d15c66a9f64e612952f1dd324d6`.
> **Writer:** none — implementation/testing slice closed.
> **Release state:** NOT MERGED / NOT DEPLOYED / NO V5 PRODUCTION-SMOKE CLAIM.
> **Original reviewed V5 candidate:** `a47357c602120d3678e8f2f23b99775e616c79e1`, gate `34693751545` — SUCCESS.
> **Independent review:** `ACCEPT WITH REQUIRED CHANGES` / `PATCH V5 THEN MERGE`.
> **Released jurisdiction runtime:** `e52a4851b504476c1e361575d08664c05467ff53` with production profile `CY_GESY`.
> **Authenticated CY_GESY production smoke evidence:** `34703453615` — SUCCESS.
> **Patient/referral persistence authority:** NONE.

## 1. Product Owner authority exercised

On 2026-09-13 the Product Owner instructed:

`IMPLEMENT V5 INTEGRATION PATCH`

and explicitly accepted the independent-review simplification:

```text
remove duplicated `Ατροφία τετρακεφάλου`
from the second-tap `Αδυναμία` refinement sheet
while retaining it in `Περισσότερα → Εξέταση`
```

That bounded implementation authority has now been exercised and the implementation writer is released.

No authority was created for a second diagnosis, evidence reclassification, patient persistence, autonomous recommendation changes, Greece/England localization, billing/analytics, or unrelated product work.

## 2. Independent-review blocker — resolved

The original V5 head `a47357c...` was correctly tested on older v4 ancestry but predated the released `CY_GESY` jurisdiction runtime and later shared integration changes.

The historical branch was **not** merged or cherry-picked wholesale.

Instead, reviewed V5-owned deltas were reconstructed onto fresh main `8aeb91ae...`; shared files were merged against current-main semantics; the released jurisdiction runtime/data were preserved; and all acceptance layers were run on the same integrated exact head `9a7f3607...`.

Therefore the review's only release blocker, lack of evidence for V5 + current CY_GESY/shared production ancestry, is resolved at the implementation/test level.

## 3. Integrated V5 behavior — tested

The integrated candidate provides:

1. First tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a forced sheet.
2. Second tap on an already-selected symptom opens optional refinement.
3. `Λειτουργικότητα` remains a first-tap chooser.
4. The weakness second-tap exposes only:
   - `Μυϊκή αδυναμία στην εξέταση`
   - `Αδυναμία τετρακεφάλου στην εξέταση`
5. `Ατροφία τετρακεφάλου` is absent from the weakness second-tap and remains available through `Περισσότερα → Εξέταση` as an objective examination finding.
6. Weakness badge/count reflects weakness refinement only; selecting atrophy through Examination does not create a phantom weakness-detail count.
7. Bare/ambiguous `Περιαρθρικά` remains absent from visible routine/advanced UI.
8. Pain qualifier ownership prevents legacy joint-line / pes-anserine duplication.
9. Rich referral separates clinical picture/function from physiotherapy assessment/priorities into distinct paragraphs.
10. Mechanical `Επιπλέον στόχος:` wording is replaced by connected natural prose.
11. Low-information referral remains proportionally compact.

## 4. Hard invariants — preserved on exact integrated head

Run `34736389860` verified that V5 integration does not change:

- international evidence states or source positions;
- `CY_GESY` local-position semantics or explicit-account activation;
- default rehabilitation selections;
- suggestion != selection semantics;
- diagnosis assertion/laterality requirements;
- safety fail-closed behavior;
- deterministic referral ownership;
- manual-edit stale/reconciliation behavior;
- no-patient-persistence / no-browser-storage boundary;
- protected Cockpit authentication boundary;
- clinical guidance vs GeSY admin/reimbursement separation.

Local-only/admin jurisdiction rows still do not become clinical evidence items, and Cyprus context still does not rewrite routine referral prose.

## 5. Exact-head evidence

The final successful gate on `9a7f360745710deacb0ff82f03249723bdfe87d6` passed:

- bounded-scope/no-cache-artifact guard;
- Python and JavaScript syntax checks;
- V5 focused prose/server regressions;
- inherited deterministic projection and qualifier regressions;
- current jurisdiction-overlay unit regressions;
- current protected Cockpit integration regressions;
- V5 Chromium interaction acceptance;
- inherited prototype / qualifier / usability-v2 / More-v3 Chromium acceptance;
- protected Cockpit browser acceptance with active `CY_GESY`;
- explicit atrophy-only-under-Examination regression;
- adjacent-owner isolation smokes;
- package closure.

Two earlier integration runs exposed test/integration harness defects and were not used as acceptance evidence:

- `34736082562`: JavaScript syntax typo in the newly reconstructed V5 clinical-sheet file;
- `34736160407` and `34736298453`: stale/ambiguous inherited browser-test assumptions around the intentionally moved atrophy control.

Those issues were corrected and the full gate was rerun to final SUCCESS.

## 6. Release boundary

The integrated V5 candidate is **implemented and tested**, not released.

Current truthful lifecycle:

```text
reviewed                  yes
fresh-main integrated      yes
exact-head tested          yes
PR / merge                 pending
Render deployment          no V5 deployment claim
V5 authenticated live smoke no
Product Owner device use   pending after any release
```

A release decision must remain explicit. Do not infer merge/deploy authority from implementation success.

## 7. Parallel deferred validation retained

Medical Report V1 remains a separate released/deployed slice whose authenticated production smoke was still pending when this Physio slice was opened. This V5 integration did not mutate Medical Report runtime/configuration and does not silently mark that separate validation complete.

## 8. Exact next action

Open a bounded PR for the exact tested integrated V5 candidate, preserve Product Owner **release HOLD**, and verify exact-head PR gates/review scope. Merge/deploy require explicit release authority.