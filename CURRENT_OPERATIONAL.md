# CURRENT_OPERATIONAL.md — Clinical Documents Engine Phase 1

> **STATUS:** IMPLEMENTED / TESTED / REVIEWED — RELEASE AUTHORIZED.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh bootstrap main:** `bbfa26f3820f520d7f3ea312e6793fa55f399f01`.
> **Active branch:** `feat/clinic-documents-p1-sick-leave-2026-09-12`.
> **PR:** `#97` — `Clinical Documents Phase 1: Sick Leave V1`.
> **Active slice:** `CU-CLINICAL-DOCUMENTS-P1-SICK-LEAVE-2026-09-12`.
> **Exact reviewed/tested runtime head:** `fd88fa3b7ebce8aa9d97dfea12e2f2667495ed39`.
> **Clinical Documents PR gate:** run `34717351567` — SUCCESS.
> **Writer:** this release conversation, bounded through merge/deploy verification and canonical closeout.
> **Production config/secrets authority:** NONE; no config mutation required.
> **Patient persistence authority:** NONE.
> **Real-patient data in code/tests:** FORBIDDEN / NOT USED.

## 1. Product Owner authority

On 2026-09-12 the Product Owner first authorized bounded implementation of Clinical Documents Phase 1 and later explicitly instructed:

`Merge και deploy`

This authorizes the reviewed Phase-1 PR to follow the normal release path. It does not authorize Phase 2, new production secrets/configuration, patient persistence, medico-legal AI drafting, billing persistence, or unrelated RF/physio changes.

## 2. Released candidate scope

Phase 1 contains only:

```text
Common Clinical Documents Core
+
Sick Leave Certificate V1
+
protected Clinic Utilities navigation/integration
```

Implemented Sick Leave V1 behavior:

- patient full name;
- ID type `ADT` or `ARC` and bounded free-format ID number;
- diagnosis;
- leave from / through inclusive;
- editable issue date;
- deterministic inclusive-duration display;
- Greek A4 PDF preview/download;
- optional bounded PNG/JPEG signature held only in current browser memory;
- explicit re-import of a previously generated V1 PDF through embedded application metadata;
- extension flow retaining identity + diagnosis and starting on prior leave-through + 1 day;
- same-patient/new-leave flow retaining identity only;
- fail-closed rejection of unknown/malformed previous PDFs with no OCR guessing.

## 3. Privacy / persistence boundary

Hard Phase-1 rules remain satisfied:

- no patient PostgreSQL write;
- no patient/document history registry;
- no localStorage/sessionStorage/indexedDB patient state;
- no autosave;
- no automatic prior-document reopening;
- signature never persisted server-side;
- PDF generation/parsing request-scoped/in-memory;
- no patient data in query strings;
- no identifiable patient data or signature assets in repository tests/fixtures.

The package has no SQLAlchemy/database owner. Existing protected clinical authentication remains the access boundary.

## 4. Exact-head review and hardening

Exact PR review identified and corrected two bounded validation gaps before release:

1. imported V1 metadata now rejects `leave_to < leave_from` and rejects a declared relation without `derived_from_document_id`;
2. `draft_json` now has an explicit 16 KiB server-side request bound before JSON parsing, exposed by the contract endpoint and regression-tested.

The exact reviewed/tested runtime head is:

`fd88fa3b7ebce8aa9d97dfea12e2f2667495ed39`

Clinical Documents workflow run `34717351567` completed SUCCESS on that head, including Python syntax, JavaScript syntax, deterministic Sick Leave tests and existing Clinic Utilities navigation regression.

Inherited evidence on the same head:

- CU-1 focused tests: SUCCESS (`34717351536`);
- G3 guidance salience/longitudinal summary: SUCCESS (`34717351537`);
- Clinical Learning L1 substantive runtime/contracts/schema steps: SUCCESS before its expected scope/adjacent-owner guard rejected this non-Learning slice (`34717351566`);
- corresponding red checks from Clinical Learning/physio owner workflows are scope-owner guard failures, not demonstrated runtime regressions in their substantive owners.

## 5. Production release contract

Render service `osteoporosis` remains:

- branch `main`;
- auto-deploy `yes`;
- trigger `commit`;
- Frankfurt runtime;
- no Clinical Documents-specific production config required because Phase 1 can use the already-configured server-side clinician profile fallback.

Therefore the release rule is:

```text
squash merge PR #97 to main
→ Render auto-deploy from merge commit
→ monitor only; DO NOT manually trigger a duplicate deploy
→ verify LIVE release
→ authenticated/product-owner functional smoke remains separately evidenced
```

## 6. Explicit exclusions retained

Not authorized by this release:

- Accident Report intake;
- medico-legal Evidence Ledger;
- AI drafting/provider calls;
- targeted literature retrieval;
- causation/prognosis engine;
- billing/fee-note/receipt persistence;
- persistent Clinical Documents case store;
- reusable uploaded templates;
- Siri/Gemini;
- RF/physio clinical-rule mutation;
- production secret/environment mutation.

## 7. Next legitimate action

Squash merge PR #97, allow the existing Render auto-deploy to run, verify the merge commit reaches LIVE, then write the release closeout canonicals. Production-smoke verification must not be claimed unless actually performed.
