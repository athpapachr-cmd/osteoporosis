# CURRENT_OPERATIONAL.md — Physiotherapy Referral Knee-OA V5 integration patch

> **STATUS:** IMPLEMENTATION ACTIVE — FRESH-MAIN V5 RE-INTEGRATION AFTER INDEPENDENT REVIEW.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-PHYSIO-KNEE-OA-V5-INTEGRATION-2026-09-13`.
> **Bootstrap main:** `8aeb91ae37b83caaa188054128db98e04b638fd8`.
> **Implementation branch:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Writer:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Mutation scope:** bounded Knee-OA V5 UI/prose integration, exact tests/workflow, and supporting canonicals only.
> **Original exact tested V5 candidate:** `a47357c602120d3678e8f2f23b99775e616c79e1`.
> **Original V5 gate:** `34693751545` — SUCCESS.
> **Independent review:** `ACCEPT WITH REQUIRED CHANGES` / `PATCH V5 THEN MERGE`.
> **Released jurisdiction runtime:** `e52a4851b504476c1e361575d08664c05467ff53` with production profile `CY_GESY`.
> **Authenticated CY_GESY production smoke evidence:** `34703453615` — SUCCESS.
> **Patient/referral persistence authority:** NONE.

## 1. Product Owner authority

On 2026-09-13 the Product Owner explicitly instructed:

`IMPLEMENT V5 INTEGRATION PATCH`

and additionally required the independent-review simplification:

```text
remove duplicated `Ατροφία τετρακεφάλου`
from the second-tap `Αδυναμία` refinement sheet
while retaining it in `Περισσότερα → Εξέταση`
```

This authorizes the bounded fresh-main V5 integration/test slice. It does not authorize a second diagnosis, evidence reclassification, patient persistence, autonomous recommendation changes, Greece/England content, billing/analytics work, or unrelated product mutations.

## 2. Why an integration patch is required

The original V5 candidate was correctly tested at exact head `a47357c...`, but its merge base was the older Knee-OA v4 closeout ancestry `90e4377...`.

Fresh comparison against current `main 8aeb91ae...` shows the candidate and main have diverged. Main has since added the released `CY_GESY` jurisdiction overlay and other shared integrations.

Therefore the old V5 head must not be directly merged/deployed. The approved approach is:

```text
fresh current main
→ re-apply only bounded V5-owned runtime/test deltas
→ apply the approved atrophy simplification
→ preserve current CY_GESY/runtime/shared integrations
→ run V5 + jurisdiction + protected Cockpit regressions on one exact integrated SHA
```

## 3. Bounded V5 behavior to integrate

1. First tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without forcing a popup.
2. Second tap on an already-selected generic symptom opens optional refinement.
3. `Λειτουργικότητα` keeps its first-tap chooser.
4. Weakness refinement keeps only the two explicit weakness examination concepts:
   - `Μυϊκή αδυναμία στην εξέταση`
   - `Αδυναμία τετρακεφάλου στην εξέταση`
5. `Ατροφία τετρακεφάλου` is **not** duplicated in the weakness second-tap sheet; it remains reachable through `Περισσότερα → Εξέταση` as an objective examination finding.
6. Bare/ambiguous `Περιαρθρικά` remains absent from the routine/advanced visible UI while compatibility underneath is preserved where already required.
7. Pain-location prose must not duplicate/conflict with qualifier-owned location detail.
8. Rich referral output separates clinical picture/function from physiotherapy assessment/priorities into distinct paragraphs.
9. Mechanical `Επιπλέον στόχος:` wording is replaced by connected natural prose without changing clinician/treatment authority.

## 4. Hard invariants

The integration must not change:

- international evidence states or source positions;
- `CY_GESY` overlay semantics or explicit-account activation;
- default rehabilitation selections;
- suggestion != selection semantics;
- diagnosis assertion/laterality requirements;
- safety fail-closed behavior;
- deterministic referral ownership;
- manual-edit stale/reconciliation behavior;
- no-patient-persistence boundary;
- protected Cockpit authentication boundary;
- local clinical guidance vs GeSY admin/reimbursement separation.

## 5. Exact implementation rule

Do not cherry-pick or merge the historical V5 branch wholesale.

Generated/cache artifacts such as `__pycache__` / `.pyc` are explicitly forbidden from the integration.

Only reviewed V5 source/test deltas that remain valid on current main may be re-applied. Shared files changed since the V5 merge base must be merged semantically, never overwritten with stale candidate copies.

## 6. Acceptance evidence required on one exact integrated SHA

Required before implementation can be called tested:

- V5 focused server/prose regressions;
- V5 browser interaction regressions;
- current Knee-OA protected Cockpit integration regressions;
- current `CY_GESY` jurisdiction-overlay regressions;
- inherited v4/prototype/CU-1/evidence/safety gates that apply;
- proof `Ατροφία τετρακεφάλου` is absent from weakness second-tap and present under `Περισσότερα → Εξέταση`;
- proof international/local evidence states and referral selection remain unchanged;
- proof no new storage/persistence is introduced;
- branch remains current with `main` and bounded in scope.

## 7. Release boundary

This command authorizes implementation/integration and exact-head testing.

After a clean integrated head exists, the release path must be represented honestly as a separate lifecycle state. No production deployment or production-smoke claim may be inferred merely from implementation success.

## 8. Parallel deferred validation retained

The previously active Medical Report V1 slice is already merged/deployed with AI runtime enabled and **authenticated production smoke still pending**. Activating this Physio writer does not mark that smoke complete and does not mutate Medical Report runtime/configuration.

## 9. Exact next action

Reconstruct the V5 delta from exact candidate `a47357c...`, merge it onto current-main semantics without overwriting the released jurisdiction/shared integration work, apply the atrophy simplification, and run the required exact-head regression set.