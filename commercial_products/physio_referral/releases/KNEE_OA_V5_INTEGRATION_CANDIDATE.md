# Knee OA V5 — fresh-main integration candidate

> **STATUS:** IMPLEMENTED / EXACT-HEAD TESTED / PRODUCT OWNER RELEASE HOLD.
> **Recorded:** 2026-09-13 Asia/Nicosia.
> **Diagnosis:** Knee Osteoarthritis only.
> **Bootstrap main:** `8aeb91ae37b83caaa188054128db98e04b638fd8`.
> **Integration branch:** `feat/physio-knee-oa-v5-integration-2026-09-13`.
> **Exact tested substantive head:** `9a7f360745710deacb0ff82f03249723bdfe87d6`.
> **Final substantive gate:** `34736389860` — SUCCESS.
> **Artifact:** `10311108002`.
> **Artifact digest:** `sha256:c51e3e29c4057b0e90773011d54648703eca5d15c66a9f64e612952f1dd324d6`.
> **Release state:** NOT MERGED / NOT DEPLOYED / NO V5 PRODUCTION-SMOKE CLAIM.

## 1. Authority and reason for this candidate

The historical reviewed V5 candidate was:

```text
head  a47357c602120d3678e8f2f23b99775e616c79e1
gate  34693751545 — SUCCESS
```

Independent review returned:

```text
ACCEPT WITH REQUIRED CHANGES
PATCH V5 THEN MERGE
```

The required change was integration, not clinical redesign. The historical V5 ancestry predated the released `CY_GESY` jurisdiction overlay and later shared Cockpit integration work.

The Product Owner therefore explicitly authorized:

`IMPLEMENT V5 INTEGRATION PATCH`

and also accepted the review's first simplification: remove duplicated `Ατροφία τετρακεφάλου` from the weakness second-tap refinement while retaining that objective finding under `Περισσότερα → Εξέταση`.

## 2. Integration method

The historical V5 branch was not merged or cherry-picked wholesale.

The candidate was reconstructed from fresh main `8aeb91ae...` using only reviewed V5-owned source/test deltas. Shared files were merged semantically against current main rather than overwritten with stale copies.

Generated/cache artifacts (`__pycache__`, `.pyc`) were explicitly excluded.

Current jurisdiction runtime/data and protected production transport were preserved.

## 3. Final V5 interaction behavior

### Pain / Stiffness / Weakness

```text
first tap while inactive
→ select generic symptom
→ no forced popup

second tap while selected
→ open optional focused refinement
```

### Function

`Λειτουργικότητα` remains a first-tap chooser.

### Weakness refinement

The second-tap weakness sheet contains only:

- `Μυϊκή αδυναμία στην εξέταση`
- `Αδυναμία τετρακεφάλου στην εξέταση`

It does **not** contain `Ατροφία τετρακεφάλου`.

Quadriceps atrophy remains available as an objective examination finding through:

```text
Περισσότερα → Εξέταση → Ατροφία τετρακεφάλου
```

The weakness badge/count represents weakness-refinement state only. Selecting quadriceps atrophy through Examination does not create a hidden or phantom weakness-detail count.

## 4. Copy and semantic refinements

The candidate also preserves the reviewed V5 improvements:

- generic weakness remains symptom/context, not objective weakness;
- objective weakness and quadriceps weakness require explicit examination selections;
- bare ambiguous `Περιαρθρικά` is absent from visible routine/advanced UI;
- pain qualifiers own location specificity and suppress redundant legacy location prose;
- clinical picture / functional impact and physiotherapy assessment / priorities are separated by a paragraph boundary in richer referrals;
- mechanical `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` labels are replaced by connected natural prose;
- low-information referral output remains compact;
- no physiotherapy dose, frequency, sequence or protocol is invented.

## 5. Jurisdiction and product-integrity invariants

The same exact integrated candidate preserves:

```text
international evidence state != jurisdiction local position
suggestion != clinician selection
clinical guidance != GeSY admin/reimbursement
manual text != structured state
symptom != objective finding
planned != active
```

Specifically:

- international evidence states/source positions remain unchanged;
- `CY_GESY` remains a separate explicit-account jurisdiction overlay;
- Cyprus context never auto-selects/deselects treatment;
- jurisdiction context never rewrites routine referral prose;
- local-only/admin rows never become clinical evidence items;
- default rehabilitation selections remain unchanged;
- diagnosis/laterality and safety fail-closed semantics remain unchanged;
- manual-edit reconciliation remains fail closed;
- no patient/referral browser persistence is introduced;
- protected Cockpit authentication remains authoritative.

## 6. Exact-head test evidence

Final run `34736389860` on exact substantive head `9a7f360745710deacb0ff82f03249723bdfe87d6` completed `SUCCESS`.

It passed:

1. bounded-scope and no-cache-artifact guard;
2. Python and JavaScript syntax;
3. V5 focused prose/server regressions;
4. inherited server/qualifier regressions;
5. current jurisdiction-overlay unit regressions;
6. current protected Cockpit integration regressions;
7. V5 Chromium first-tap/second-tap acceptance;
8. explicit atrophy-only-under-Examination regression;
9. inherited prototype/qualifier/usability/More Chromium acceptance;
10. protected Cockpit browser acceptance with active `CY_GESY`;
11. manual-edit fail-closed and no-storage boundaries;
12. adjacent-owner isolation smokes;
13. package closure.

Artifact `10311108002` has digest:

`sha256:c51e3e29c4057b0e90773011d54648703eca5d15c66a9f64e612952f1dd324d6`

## 7. Diagnostic history

Earlier integration runs are retained as diagnostic history and are not acceptance evidence:

- `34736082562`: reconstruction syntax typo in the V5 clinical-sheet JavaScript;
- `34736160407`: stale inherited browser test still expected atrophy inside weakness second-tap;
- `34736298453`: protected-browser test used an ambiguous selector matching two legitimate Exam buttons.

Each issue was corrected and the complete gate was rerun. Only final run `34736389860` is the substantive acceptance evidence.

## 8. Release boundary

Truthful state at this candidate record:

```text
independent review             complete
fresh-main integration         complete
exact-head substantive tests   pass
PR review                       next
merge                           not authorized yet
deployment                      not performed
V5 authenticated live smoke     not performed
```

No production release is implied by this document.

A separate explicit Product Owner release decision is required before merge/deploy.

## 9. Explicit exclusions

This candidate does not authorize or implement:

- second diagnosis;
- new evidence classification;
- new jurisdiction/country selector;
- new local-rule automation;
- patient persistence;
- analytics/billing/entitlements;
- formal physiotherapist/receiver validation as a mandatory gate;
- Greece/England localization;
- unrelated Medical Report changes.

## 10. Next action

Open and review a bounded PR containing this fresh-main integration candidate, verify exact PR-head gates and scope, and preserve Product Owner **RELEASE HOLD** until explicit release authority is given.