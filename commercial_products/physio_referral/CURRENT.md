# CURRENT.md — Physio Referral commercial product track

> **STATUS:** KNEE-OA V5 MERGED / DEPLOYED; CY_GESY ACTIVE; AUTHENTICATED V5 PRODUCTION SMOKE PENDING EXECUTION.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **V5 release runtime SHA:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Release PR:** `#101` — squash-merged.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE at exact release runtime SHA.
> **Production jurisdiction profile:** `CY_GESY` via explicit server-side configuration.
> **Prior authenticated CY_GESY smoke:** `34703453615` — SUCCESS before V5 release.
> **Authenticated V5 smoke:** `34737351354` — QUEUED / NOT YET EXECUTED at this reconciliation.
> **Root operational authority:** `CURRENT_OPERATIONAL.md`.

## Product state

Knee-OA V5 is merged and deployed inside the authenticated Clinical Excellence physiotherapy utility.

Current production architecture is:

```text
real CU-1 validation / safety
+
deterministic Knee-OA projection
+
international evidence core
+
reviewed CY_GESY jurisdiction overlay
+
V5 workflow / presentation layer
+
protected Clinical Excellence transport
```

The routine experience remains sparse and clinician-controlled: few meaningful routine decisions, progressive disclosure underneath, deterministic referral output, direct editing with fail-closed reconciliation and no routine LLM-generated referral prose.

## Released V5 behavior

The deployed V5 code provides:

- first tap on inactive `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a forced popup;
- second tap opens optional focused refinement;
- `Λειτουργικότητα` retains its first-tap chooser;
- weakness second-tap contains only `Μυϊκή αδυναμία στην εξέταση` and `Αδυναμία τετρακεφάλου στην εξέταση`;
- `Ατροφία τετρακεφάλου` is not duplicated in weakness second-tap and remains available through `Περισσότερα → Εξέταση` as an objective finding;
- weakness detail count does not count separately selected quadriceps atrophy;
- bare ambiguous `Περιαρθρικά` remains absent from visible routine/advanced UI;
- pain-location duplicate prose is reconciled;
- richer referral output separates clinical picture/function from physiotherapy assessment/priorities;
- mechanical `Επιπλέον στόχος:` wording is replaced by connected prose;
- low-information output remains compact.

## Release evidence

PR #101 was released from exact reviewed head:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

That head passed:

- V5 integration gate `34736919952`;
- jurisdiction overlay v1 gate `34736920005`;
- clinical-sheet v4 gate `34736920059`;
- prototype gate `34736919984`;
- Cockpit integration gate `34736920081`;
- evidence design gate `34736919959`;
- CU-1 focused tests `34736919945`.

All are `SUCCESS`. Final PR-head artifact: `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

Clinical Learning red checks were adjacent-owner scope-only after substantive tests/frozen-owner guards passed and are not V5 regressions.

## Jurisdiction / evidence integrity

The `CY_GESY` layer remains separate from the international evidence core.

```text
local clinical guidance != international evidence state
GeSY admin/reimbursement != clinical efficacy evidence
local difference != silent overwrite
suggestion != clinician selection
planned != active
```

V5 does not reclassify international evidence, auto-select treatment, add local-only interventions to the routine clinical surface or inject Cyprus/GeSY evidence text into the referral.

## Privacy / safety boundaries

Unchanged:

- no patient/referral draft persistence in this product;
- no clinical draft state in `localStorage` / `sessionStorage`;
- no autonomous literature-to-live update;
- diagnosis/laterality requirements remain deterministic;
- inherited safety remains fail closed;
- manual clinician text remains clinician-owned and stale reconciliation blocks export until resolved.

## Production verification state

Render auto-deploy `dep-daj2398u01pc738ojvkg` reached `live` at exact V5 release commit `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.

Authenticated V5 smoke run `34737351354` has been submitted using the protected GitHub Actions secret and only generated/non-identifiable state, but GitHub has not yet assigned a runner. A rerun of the previously successful authenticated jurisdiction smoke is likewise queued.

Therefore the current truthful distinction is:

```text
V5 merged                         yes
V5 deployed                       yes
Render exact-commit deployment    verified
V5 authenticated live smoke       pending execution
real clinical pilot               no
commercial validation             no
second diagnosis                  no
```

A queued smoke is neither a PASS nor evidence of application failure.

## External feedback policy

Formal physiotherapist/receiver evaluation remains optional later external evidence, not a release prerequisite. Receiver utility is not claimed as proven.

## Next product boundary

Complete the queued authenticated production smoke and then finalize the docs-only release closeout. Product Owner real-device/use acceptance may follow as product evidence, not as permission to silently expand scope.

Do not infer authority for a second diagnosis, patient persistence, analytics/billing, Greece/England profiles or new treatment selectors from the V5 release.

Permanent rule:

```text
CLINICALLY INTERESTING != WORKFLOW-USEFUL != RECEIVER-USEFUL != WORTH ADDING
PRODUCT OWNER REQUEST != EVIDENCE != IMPLEMENTATION AUTHORITY
EXTERNAL FEEDBACK != AUTOMATIC IMPLEMENTATION AUTHORITY
```
