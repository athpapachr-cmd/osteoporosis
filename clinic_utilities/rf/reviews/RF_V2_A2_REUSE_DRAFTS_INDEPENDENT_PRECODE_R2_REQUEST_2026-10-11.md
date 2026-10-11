# RF v2 — A2 continuation + saved drafts: ONE bounded independent PRE-CODE R2

**MODE:** Fresh separate independent READ-ONLY PRE-CODE R2. Return PASS/BLOCK/UNKNOWN / COMPLETE_FOR_DECLARED_SCOPE|PARTIAL / P0:P1:P2, smallest source-grounded finding, STOP. No code, no merge, no deploy, no patient/provider access or general architecture redesign.

**Repo:** athpapachr-cmd/osteoporosis
**Baseline freshly verify:** main at design activation 7186a26105e6edd37ed02b089781ce3909c2eaf3.
**Design branch:** design/rf-v2-a2-reuse-and-draft-20261011.
**Exact frozen DESIGN checkpoint commit:** 3aa858783ba2cbe599207a41e51abaee9848cb61.
**Frozen design blob:** 4d9ff0a50f06cd0f4499fe077245422edb017abb.
**Read FULL:**
clinic_utilities/rf/RF_V2_A2_CONTINUATION_SEARCH_SAVED_DRAFTS_BOUND_DESIGN_2026-10-11.md.

Product Owner wants three jointly scoped behaviors: (1) Upload previous RF PDF with zero prior mandatory fields, or lookup via single field name/identity/GESY, (2) prefill original patient/indication/side/product and update only current clinical details including *actual* prior RF if it truly occurred, (3) Save/Resume incomplete A1 or A2 draft using the existing protected RF database. The last cosmetic Horos picker fix PR #147 is merged and must not be reopened.

### R1 — Existing mechanism and no duplicate authority
- Inspect only existing native RF protected router/persistence, previous-procedure search, RFApplicationORM JSON, A2 draft builder, official A2 form and currently relevant tests as needed.
- Confirm the single-search and reuse plan fits real data already available, without a parallel patient registry, clinician/issuer side effects or unsupported new provider.

### R2 — Identity and provenance
- Search requires just one term, no side/indication pre-input; it can return multiple patient names safely with explicit selection. No silent wrong-patient match.
- Prior submitted A1/A2 document is never assumed proof of an actual RF procedure. Only verified procedure history or explicit clinician-confirmed actual procedure may satisfy A2.
- Selected current RF doctor remains independent of previous author; previously fixed issuer-switch P1-01 must remain closed.

### R3 — Old PDF import limitations and official form
- Can the proposed text/coordinate extraction be grounded in existing official PDF layout and existing PyMuPDF? Handle scanned/corrupt/mismatched PDFs without guessing, and reject nonofficial false positives; do not require OCR.
- PDF upload is separate from required **new** imaging attachment. Retain existing official template, clinical A2 completion rules, JPEG limits/order and zero-write barriers.

### R4 — Protected saved draft lifecycle
- Review the allowed partial state, new RF-draft-only protected table, version/revision, Save/Resume/list/search/delete, no browser persistence, bounded data/PHI and safe conflict and inactive-doctor handling.
- A successful RF create should close the linked draft atomically; failed create preserves it; no creation of an actual procedure from a draft. No binary media persisted, visible reattach requirement; assess whether this accurately meets the frozen declared first-version scope.
- No hidden data loss or automatic overwrite when importing old PDF after editing current draft.

### R5 — Bounded implementation and evidence
- Check new endpoints and tests are realistically implementable within the existing RF files and whether listed synthetic tests prove A1 unaffected, A2 correct, real prior procedure vs request, wrong-patient rejection, draft restart/revision, and official PDF integrity.
- Only source-proven missing essential requirements may BLOCK. Reuse already successful #146/#147 RF/JPEG/doctor tests and prior independent R2; do not rerun unrelated phases or broaden into Cockpit, commercial products, another service or Clinical Learning.

**Decision:** Is this frozen design ready for one bounded clinical R2 implementation? Exact reviewer handback PASS/BLOCK/UNKNOWN + dispositions R1–R5 + total P0:P1:P2 and STOP. Even PASS is design approval only, not automatic implementation/merge/deploy or real-patient authorization.
