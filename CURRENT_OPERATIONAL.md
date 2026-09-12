# CURRENT_OPERATIONAL.md — Clinical Documents Engine Phase 1

> **STATUS:** MERGED / DEPLOYED — PRODUCTION FUNCTIONAL SMOKE PENDING.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-CLINICAL-DOCUMENTS-P1-SICK-LEAVE-2026-09-12`.
> **PR:** `#97` — SQUASH MERGED.
> **Exact reviewed/tested runtime head:** `fd88fa3b7ebce8aa9d97dfea12e2f2667495ed39`.
> **Clinical Documents PR gate:** run `34717351567` — SUCCESS.
> **Release commit:** `a3ae5dd792a715a8306f5c279cc4bdbec4ca5b5e`.
> **Render deploy:** `dep-dairee0jo6nc73bt03b0` — LIVE.
> **Writer:** none — release mutation complete.
> **Production config/secrets mutation:** NONE.
> **Patient persistence authority:** NONE.
> **Production-smoke label:** NOT YET CLAIMED.

## 1. Product Owner authority and release

On 2026-09-12 the Product Owner explicitly authorized:

`Merge και deploy`

PR #97 was squash merged to `main` as:

`a3ae5dd792a715a8306f5c279cc4bdbec4ca5b5e`

The existing Render `autoDeploy=yes` pipeline then deployed that exact release commit automatically. No duplicate manual deploy was triggered. Deploy `dep-dairee0jo6nc73bt03b0` reached `LIVE` at 2026-09-12T20:35:36Z.

## 2. Released scope

Phase 1 now deployed contains only:

```text
Common Clinical Documents Core
+
Sick Leave Certificate V1
+
protected Clinic Utilities navigation/integration
```

Released Sick Leave V1 behavior:

- patient full name;
- ID type `ADT` or `ARC` with bounded free-format ID number;
- diagnosis;
- leave from / through inclusive;
- editable issue date;
- deterministic inclusive-duration display;
- Greek A4 PDF preview/download;
- optional bounded PNG/JPEG signature retained only in current browser memory;
- explicit re-import of a previously generated V1 PDF using embedded application metadata;
- extension flow retaining identity + diagnosis and setting next start to prior leave-through + 1 day;
- same-patient/new-leave flow retaining identity only;
- fail-closed rejection of unknown/malformed previous PDFs with no OCR guessing.

## 3. Privacy / persistence boundary

The release preserves the Phase-1 hard boundary:

- no patient PostgreSQL write;
- no Clinical Documents patient/document history registry;
- no localStorage/sessionStorage/indexedDB patient state;
- no autosave;
- no automatic prior-document reopening;
- signature never persisted server-side;
- PDF generation/parsing request-scoped/in-memory;
- no patient data in query strings;
- no identifiable patient data or signature assets in repository tests/fixtures.

The package has no SQLAlchemy/database owner. Existing protected clinical authentication remains the access boundary.

## 4. Exact-head review and verification

Before merge, exact review found and corrected two bounded validation gaps:

1. imported V1 metadata rejects `leave_to < leave_from` and rejects a declared relation without `derived_from_document_id`;
2. `draft_json` has an explicit 16 KiB server-side request bound before JSON parsing, exposed by the contract endpoint and regression-tested.

The exact reviewed/tested runtime head was:

`fd88fa3b7ebce8aa9d97dfea12e2f2667495ed39`

Clinical Documents PR workflow `34717351567` completed SUCCESS. On the same runtime head, CU-1 focused tests (`34717351536`) and G3 guidance/longitudinal summary (`34717351537`) also completed SUCCESS. Clinical Learning red checks passed their substantive runtime/contracts/schema steps and failed only their expected scope/adjacent-owner guards for this non-Learning slice.

## 5. Deployment evidence

Production Render service:

- service: `osteoporosis`;
- branch: `main`;
- auto-deploy: `yes`;
- trigger: `new_commit`;
- release commit: `a3ae5dd792a715a8306f5c279cc4bdbec4ca5b5e`;
- deploy: `dep-dairee0jo6nc73bt03b0`;
- status: `LIVE`;
- finished: `2026-09-12T20:35:36.691525Z`.

No Clinical Documents-specific environment variable or secret was added or changed for this release.

## 6. What is not yet proven

`DEPLOYED != PRODUCTION-SMOKE-VERIFIED`.

An authenticated end-to-end production use of the new Sick Leave workflow has not yet been evidenced in this release conversation. Therefore the canonical state intentionally remains **production functional smoke pending** rather than pretending that a green deploy is the same thing as successful clinician use, a distinction software occasionally resents but medicine rather sensibly requires.

A valid production functional smoke should use synthetic/non-identifiable data and verify at minimum:

1. authenticated Sick Leave page opens;
2. contract reports clinician profile configured;
3. PDF preview or download succeeds;
4. generated PDF is readable and contains expected synthetic Greek content;
5. re-import offers extension and same-patient/new-leave modes;
6. no patient persistence is introduced.

## 7. Explicit exclusions retained

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

## 8. Next legitimate action

Writer lock is released. The next release-validation action is an authenticated synthetic/product-owner Sick Leave smoke. Phase 2 requires a fresh bounded slice and fresh Product Owner authority; it is not implied by this deployment.
