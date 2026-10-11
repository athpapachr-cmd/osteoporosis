# Native RF v2 — A2 reuse / one-field search / saved drafts, ONE independent post-code R2

**ROLE:** Fresh separate independent READ-ONLY exact-HEAD affected implementation-fidelity R2.
**DECISION:** Can the product owner accept this bounded implementation candidate for a distinct controlled release decision?
**OUTPUT:** IMPLEMENTATION-FIDELITY PASS/BLOCK/UNKNOWN · COMPLETE_FOR_DECLARED_SCOPE/PARTIAL · P0:P1:P2 and STOP.
**PROHIBITED:** No source changes, no merge, no deploy, no clinical patient/provider access, no redundant design review or tests without a specific changed-seam question.

Repository: `athpapachr-cmd/osteoporosis`.
Baseline main at implementation activation: `7186a26105e6edd37ed02b089781ce3909c2eaf3` (fresh verify remote main).
Implementation branch: `feat/rf-v2-a2-continuation-saved-drafts-20261011`.
**Frozen source+test commit:** `ac1dd54ce2f17575e160a025367557fcf8afe598`.
**Frozen source+test blobs:**
- `clinic_utilities/rf/api.py` — `5090438c0e2f7370af51cb0f796514305b77f1b0`
- `clinic_utilities/rf/persistence.py` — `ad40c314bae1008bea6ee5bf5c68d39ae1f693f6`
- `clinic_utilities/rf/prior_pdf.py` — `07ca2ebd8874f8b1aebd803f2f7a95a0539bdc2a`
- `static/clinic-utilities/rf/app.js` — `2199f750c18b89d64f866445679cf76109aeba5a`
- `static/clinic-utilities/rf/index.html` — `f7c6ce09e5409ee4a3311c3eeb0e034dfa187528`
- `static/clinic-utilities/rf/styles.css` — `8796cee034ef3bfc4099db2b3742f42a2674d647`
- `test_rf_v2_native.py` — `d5d0699b58d2ebaf6b5c97c35bbdc0cbfc4494e2`
- `test_rf_v2_ui.js` — `0a409f5551731ba1bf1bd490034eb70369571ae2`
**Existing RF CI:** [workflow 38170026377](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38170026377) SUCCESS on frozen `ac1dd54ce2f17575e160a025367557fcf8afe598`.
RF hotfix workflow reports native 33 tests, browser handlers (Horos, managed doctors and A2/draft), official template A1/A2, adjacent CU-1, legacy gateway and G4–C1 tests PASS.
Original independent PRE-CODE R2 of frozen design `4d9ff0a50f06cd0f4499fe077245422edb017abb` returned PASS/COMPLETE/0:0:0; reuse. Design PR #148 is separate, DRAFT and unmerged. User confirmed saved drafts **do not contain any binary attachments**.

## Scope and finite evidence map

**R1 — Identity/source choice:** Inspect the only source search/retrieval and existing RF application/procedure DB, Greek accent/case and ID/ΓεΣΥ normalization, bounded paging, multiple-name explicit selection, wrong-patient and wrong-side/indication rejection. An issued A1/A2 application must never automatically count as an actual previous RF; only clinician-recorded actual procedure or explicit manual confirmation can satisfy A2. Check that age from older record/PDF is NOT silently carried forward as today's age. No second registry.

**R2 — Old official PDF:** Verify original-template static-word and layout matching with PyMuPDF (existing dependency), extraction from A1 and A2 including printed underscore blanks, failure/partial behavior for unrecognized/scanned PDF, no OCR/guesses, provisional values, historical physician and actual application not auto-authoritative. Uploaded old PDF is not the new imaging attachment. Official RF PDF template and Horos processing unchanged.

**R3 — Saved draft state:** Inspect protected RFBase draft table and existing clinical auth; explicit Save and Resume of incomplete A1/A2 fields (no identity/diagnosis/side prerequisites), revision/stale-update conflicts, list/search/pagination, explicit delete, no localStorage/sessionStorage and NO binary PDF/JPEG or rendered PDFs stored/logged as draft. On resume, file reattachment flagged and no inactive doctor automatically selected. The current RF clinical credential is shared; do not invent per-clinician ownership.

**R4 — Issuance and unchanged authority:** Validate A2 actual prior application event, site/side/identity linkage, current follow-up date/VAS, clinician confirmation when PDF imported, immutable issuing doctor snapshot, unchanged A1 safety rules, zero DB writes for failed PDF/JPEG or cross-person A2, atomic successful created-application + draft deletion, double-finalization conflict, existing official A1/A2 output.

**R5 — UI fidelity/CI/adjacency:** Confirm one query input and one old-PDF upload before any mandatory fields, no original three-field search gate, source choice and conflicts visible, draft Save/Resume/Discard and upload reattach notice, no silent overwrite of typed fields. Inspect affected actual handlers and test oracles (not only static token checks) and observed exact frozen RF workflow. Reuse preceding #146 doctor/JPEG and #147 small UI evidence. Do not expand into other services or require a new implementation framework.

**Reviewer stop rule:** Each R1–R5 gets PASS/BLOCK/UNKNOWN, one concise source-grounded reason and exact correction if material. If no blockers and full declared coverage, RETURN PASS/0:0:0 and STOP. A fresh reviewer does not gain implementation/merge/deploy authority.

Additional bounded pre-review source correction: exact indication_code must match the referenced previous RF application AND actual procedure even when identity/site/laterality match. Synthetic same-site/side/different-indication 422/no-new-application regression passed under RF CI 38170026377. No unrelated runtime changes.
