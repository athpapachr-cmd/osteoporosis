# Cockpit Medical Reports Tracking v1 — fresh independent PRE-CODE R2

**2026-10-14 · ONE independent READ-ONLY bounded PRE-CODE R2 · STOP.** No implementation, backend/schema writes, merge/deploy, real patient data, tests on production or provider operations.

Repository: `athpapachr-cmd/osteoporosis`.
Activation production main (review base): `244a3264f119252b1bbfc3562a00648cfdf113bd`, PR #152 already squash-merged and Render LIVE.
Design branch: `design/cockpit-medical-report-resumable-tracking-20261014`.
PR: fresh-resolve its DRAFT number, exact HEAD and diff from the activation main at review time.
**Owner design candidate:** `cockpit/COCKPIT_MEDICAL_REPORT_RESUMABLE_TRACKING_V1_DESIGN_2026-10-14.md`; expected design blob `e158510d0c482a1363c389deb836b44bbea4571e`. If the blob is different, reconcile identity before returning a fidelity verdict.

Product Owner **approved the direction**: protected, resumable editable Medical Reports drafts, internal tracking and Home source-backed summary, **not** implementation or deployment. The author cannot independently PASS their own proposal.

## Sources and settled evidence to reuse

* Existing report authority: `clinic_utilities/clinical_documents/{report_api.py,report_models.py}`, `static/clinic-utilities/medical-report/{index.html,app.js,v1-1-refine.js}`, `main.py`, `clinical_auth.py`. The current report contract is **session-only**, `autosave=false` and no patient report database. Examine live source in fresh HEAD, not claims in the design only.
* Prior independently closed Home V4, #151 Clinical Key/global navigation/calendar, #152 surgery/RF real attention summary POST-CODE R2 PASS and verified Render LIVE; RF independent reviews and previously closed Medical Report G3 "no durable ledger" evidence. Reuse these, **do not reopen** unrelated workstreams.
* Existing backend database/Clinical Key integration and canonical impact guard. There is no proven per-user principal behind the shared cookie; test the report persistence authorization claim accordingly.

## Exactly seven finite gates

**G1 — Source identity, governance, bounded scope.** Pin current PR and design blob, ensure branch DRAFT/unmerged, diff **documentation-only** (new design, this request, current checkpoints), no runtime/migration/test/provider change. Verify true baseline: AI analyzes/refines, PDF output, transient case/analysis/refine state; no durable report ledger. Review **one combined** reports UI + one report-owned protected workspace/status store + count-only Home reader, not a parallel patient system or broader cockpit rewrite.

**G2 — Sensitive durable owner and access.** Is the store strictly bounded to explicit saved report attempts with random opaque ID, strict versions/size constraints, protected same-origin CRUD, no patient registry/encounter linkage, no per-doctor "mine" fiction? Does the design truthfully state shared Clinical Key permits access across holders, no verified individual issuer? Assess PHI in `MedicalReportCaseV1` and `MedicalReportAnalysisV1`, file-name metadata and list projections, encryption-at-rest/backup/DB privilege checks, CSRF on writers, no PHI in logs/URL/cache/storage/screenshots. Any material missing deployment/privacy gate = smallest P1/P0 issue; never assume existing auth is individual RBAC.

**G3 — Exact resumability without secret file persistence.** Confirm saved contract includes **DOM-edited sections**, accepted refined analysis, necessary source manifest/evidence/timeline/clinician resolutions, valid research/declaration, and strict reader rehydration of both `app.js` and `v1-1-refine.js` states. Verify **no** raw uploaded sources, extracted pages/transcript texts, signature, PDF, chat, rejected/unapplied proposal or browser persistence. Reopened source manifest must not imply files remain available; new source re-analysis requires explicit upload and does not silently overwrite approved text. On reload clinician confirmation/signature absent; stale research cannot leak into PDF. Verify honest disabled-mode legacy labels and proper conditional enabled-mode wording.

**G4 — Server truth, revisions and state machine.** Check initial `Νέα`/unsaved typing does not create a counted record; first explicit save does. Atomic monotonic revision and `expected_revision` 409, non-destructive two-tab conflict, stale async save/auth/error handling, "saved" only after server acknowledge, fail-closed 401/403/503/no database. Exact states `draft/awaiting_information/completed/void` and auditable explicit transitions. Completion is manual clinician attestation on current saved revision with content, **never** PDF/preview/AI action. Reopen and void do not fabricate immutable signed document/archive; avoid silent overwrite or download-driven status.

**G5 — Home truthful totals and read boundary.** Can `GET /api/tracking/summary` authoritatively count **only reports recorded after activation** with status counts and coverage metadata? Is `pending=draft+awaiting_information`, completed=current manually attested completion, void excluded, counts internally consistent and exhaustive? Does Home receive **only aggregate numbers** from exactly one new guarded reader, invalidate on auth refresh, ignore stale success/failure, reject malformed/incomplete/network/401/403/503 as **unavailable not zero**, show no PHI/IDs? No regression to #152 surgery/RF, Inbox and one-tap report link.

**G6 — Report UI and synthetic evidence contract.** Is the compact list, searchable all-registry pagination (no last-100 truncation), name/status/date filters, one-tap restore, autosave feedback and Save/conflict affordance sufficient at 390/680/1280 without adding a second app? Are concrete future synthetic tests named for version conflicts, logout/re-auth, raw-file exclusions, exact Greek edits/research/refine rehydrate, manual status, false-zero prevention, Home count races, disabled feature mode, source content not persisted, protected search and regression to Medical Reports AI/PDF and Visit Capture Save? Tests must never use real clinical records or provider calls.

**G7 — Explicit release HOLD and proportionality.** Is the feature OFF by default until a documented, approved retention/deletion/backup/access policy and verified storage protections exist? Does the design avoid inventing legally binding retention durations, indefinite-by-accident storage and stealth automatic purge? No production PHI pilot/activation under this design. Confirm separate authorization sequence: independent PRE-CODE R2 → separate Product Owner implementation instruction → synthetic tests + independent affected POST-CODE R2 → separate merge/deploy and real-data/privacy decision. Distinguish true design gaps from already closed unrelated tests and enhancements.

## Required reviewer result — then STOP

Return exactly:
* `PRE-CODE R2 PASS | BLOCK | UNKNOWN`
* `COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`
* `P0:P1:P2 = x:y:z`
* current PR number, exact HEAD, design blob, docs-only diff evidence
* dispositions **G1–G7**, smallest numbered material blockers and the **minimum design-only correction** if BLOCK (no broad redesign)
* **STOP**.

A PASS is **eligibility for separately authorized implementation only**. Reviewer performs no code edits, database operations, draft review writing, merge, deploy, protected data access, provider operations or real patient testing. Author cannot self-PASS.
