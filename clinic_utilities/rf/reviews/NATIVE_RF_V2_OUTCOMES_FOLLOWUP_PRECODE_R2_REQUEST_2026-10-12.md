# Native RF v2 outcomes & follow-up — ONE independent pre-code R2

**MODE: fresh separate READ-ONLY, bounded design fidelity; no code, merge, deploy, clinical records or new patient data.**

Repository: `athpapachr-cmd/osteoporosis`. **Design branch:** `design/rf-v2-outcomes-followup-corrections-20261012`; **original verified main:** `6ee184f9c76bf45a6c43e1f9fae90d43c1dfde00`. Fresh-verify actual main and branch HEAD and pin exact design blob:
`clinic_utilities/rf/RF_V2_OUTCOMES_FOLLOWUP_DESIGN_2026-10-12.md`.

The Product Owner explicitly confirmed: **own RF tracking begins when the clinician marks it Submitted; RF prepared for another doctor does not create a tracking obligation unless explicitly opted in.** Existing RF v2 (#149) is released with issued applications, protected previous-RF reuse and saved drafts without binaries. Do not re-review the completed doctor/JPEG/A2/draft implementation.

**Decision:** Is this bounded design implementable in existing RF with truthful state/provenance, immutable issued applications, properly separated issuer/outcome/follow-up ownership and on-request rejection correction, without a new engine or any automatic external action?

Review only:
- **R1:** Existing RFApplicationORM, RFBase + protected router, existing source lookup/saved drafts, actual current Home boundary; no parallel service/schema owner, no unsafe live-table migration assumptions.
- **R2:** Manual lifecycle vs issued PDF, explicit submission date, decisions/date, rejected reason, scheduled/performed distinctions, unknown/legacy states; colleague-reported rejection *without* fabricated submission date, status amendments and audit/revision; no false approvals in counts.
- **R3:** Precise opt-in/follow-up rule (own submission -> mine; colleagues never automatic), no silent clinician tasks, Home/Calendar writes or provider actions; strong patient/provider identity and historical non-attribution.
- **R4:** Rejection→new distinct linked application only by requested action; reuse existing protected patient search and form; no accidental A2 transition/procedure history, stale doctor, binary-attachment persistence or overwrite of rejected original.
- **R5:** Minimal UI, privacy of reason text and clinical dates, counted metrics with denominators, bounded synthetic regressions and correct stop/release boundary.

Return **PRE-CODE R2 PASS | BLOCK | UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE | PARTIAL**, **P0:P1:P2**, evidence-backed findings and smallest correction (if any), reviewed SHA and **STOP**. No code/release decisions and no repeated review of settled RF issues.
