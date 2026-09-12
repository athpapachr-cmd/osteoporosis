# CURRENT_OPERATIONAL.md — Clinical Documents Engine Phase 1

> **STATUS:** PRODUCTION-SMOKE-VERIFIED / CLOSED.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CU-CLINICAL-DOCUMENTS-P1-SICK-LEAVE-2026-09-12`.
> **PR:** `#97` — SQUASH MERGED.
> **Exact reviewed/tested runtime head:** `fd88fa3b7ebce8aa9d97dfea12e2f2667495ed39`.
> **Clinical Documents PR gate:** run `34717351567` — SUCCESS.
> **Release commit:** `a3ae5dd792a715a8306f5c279cc4bdbec4ca5b5e`.
> **Render deploy:** `dep-dairee0jo6nc73bt03b0` — LIVE.
> **Production functional smoke:** PASS — Product Owner reported all Sick Leave smoke checks working on 2026-09-12.
> **Writer:** none — Phase 1 slice closed.
> **Production config/secrets mutation:** NONE.
> **Patient persistence authority:** NONE.

## 1. Product Owner authority and release

On 2026-09-12 the Product Owner explicitly authorized:

`Merge και deploy`

PR #97 was squash merged to `main` as:

`a3ae5dd792a715a8306f5c279cc4bdbec4ca5b5e`

The existing Render `autoDeploy=yes` pipeline deployed that exact release commit automatically. No duplicate manual deploy was triggered. Deploy `dep-dairee0jo6nc73bt03b0` reached `LIVE` at 2026-09-12T20:35:36Z.

The Product Owner then performed the agreed production smoke and reported that all smoke checks were working correctly. This closes the production functional validation gate for Sick Leave V1.

## 2. Released scope

Phase 1 contains only:

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

## 6. Production smoke evidence

The Product Owner reports the agreed production Sick Leave smoke as fully successful.

This satisfies the release-validation distinction:

```text
IMPLEMENTED                 YES
TESTED                      YES
EXACT-HEAD REVIEW           PASS
MERGED                      YES
DEPLOYED                    YES
PRODUCTION-SMOKE-VERIFIED   YES
PILOT-VALIDATED             NOT APPLICABLE TO THIS UTILITY RELEASE
```

This smoke verifies the released Sick Leave V1 workflow in production. It does not authorize or validate the later Accident/Medico-Legal phases.

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

Phase 1 is closed and no writer lock is active.

Any move to Accident/Medico-Legal Report implementation, AI-assisted drafting/literature, billing, persistent cases or user-uploaded templates requires a fresh six-canonical bootstrap, a new bounded slice and explicit Product Owner implementation authority.
