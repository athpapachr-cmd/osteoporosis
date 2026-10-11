# Native RF v2 — Approval, rejection, on-request correction & follow-up ownership

**Date:** 2026-10-12 · **MODE:** DESIGN ONLY / bounded pre-code R2 · **NO implementation, merge, deploy or patient data**  
**Repository:** `athpapachr-cmd/osteoporosis` · **Fresh verified source baseline:** `6ee184f9c76bf45a6c43e1f9fae90d43c1dfde00`  
**Product Owner checkpoint:** Approved in the conversation: “My applications enter tracking when I explicitly mark them Submitted; a colleague’s application creates no personal tracking obligation unless I explicitly take it on.” Also confirmed: if a colleague later reports rejection and requests correction, new linked application should be possible without ever having tracked that application.

## 1. Product intent — reuse the working RF v2

**Today:** Native protected RF v2 creates official A1/A2 PDFs, records issued application rows `clinic_rf_applications`, manages doctors, offers protected previous-RF lookup/reuse, and saves editable *structured-only* A1/A2 drafts; no outcome workflow, automatic GESY submission, approval result, rejection-text register or procedure scheduling exists. `doctor-shell.js` explicitly says concentrated RF tracking is not active.

**After this slice:** From the **same RF screen**, the clinician can find an issued application, mark its **real known submission**, record a subsequently reported approval/rejection, paste a bounded rejection reason, initiate a **new linked corrected application on request**, and, where explicitly followed, enter scheduling and actual-performance status. Distinguish published/created PDF from actual submitted RF. Display true pending/approved/rejected counts and “mine to follow up” separately.

**Product example:** I create an RF for a colleague and hand over the PDF. It does not appear in “My pending approvals.” Weeks later the colleague forwards a rejection and asks for a correction. I find the earlier issued RF, paste the reason, and launch a **new application** prefilled from valid data. The original rejected RF remains historically intact. I need not enter invented submission or follow-up dates or take ownership of the colleague’s remaining workflow.

**Not in scope:** AI or automatic reading of GeSY replies; automatic GeSY submission; Reception/Cal.com appointment booking; automatic clinical procedure-history creation from mere approval/scheduling; storing attachment binaries; redesigning Home; generic clinical task orchestration; any obligations to monitor colleagues.

## 2. Clear state separation (do not conflate)

A. **Document issued**: `RFApplicationORM` row exists because PDF was created. It does **not** prove submission/approval/performance. Older application rows without outcome data have **unknown progression**, never “awaiting approval” retroactively.

B. **External outcome timeline**, with **explicit manual provenance**:
- `issued / submission not recorded` — pre-existing issued application, *not pending*.
- `submitted / awaiting reply` — explicit “Σημειώθηκε ως υποβληθείσα” with actual submission date (default today as editable convenience, confirmation required).
- `approved` or `rejected` — explicit manually recorded decision/date; rejection text pasted verbatim (bounded). Approval is not RF performed.
- `scheduled` — approved and date explicitly entered; may change date with audit.
- `performed` — actual date explicitly confirmed, never inferred from appointment date.

**Exception for colleagues:** A rejection can be recorded directly against an issued application when the colleague reports it *without prior submission tracking*. Preserve submission date as **unknown**; don't invent earlier `submitted` or a tracking obligation. Likewise accept an explicitly reported approval without inventing a submitted date. Label decision date/type of evidence clearly. A rejection is terminal for that attempt; a correction creates a separate new issued application, not an A2 continuation by itself.

C. **Follow-up responsibility**, separate from outcome and issuer:
- On marking **own RF** (issuer key `papachristou` in a new attributable application) as **submitted**, set `mine_to_follow = true` as approved default. Do not activate simply by issuing PDF.
- **Any colleague's RF**: `mine_to_follow = false` even if the user creates PDF or records a reported decision; only explicit “Αναλαμβάνω παρακολούθηση” opts in. The user may later explicitly stop tracking; doing so does not erase objective outcome.
- Historical RF without trustworthy issuer key: **never presume “mine”**. Explicit opt-in available.
- “My attention” counts only real actionable states **with explicit/approved follow-up ownership**; approved but unreported scheduling is not a fabricated booking.
- This system has the existing clinic-only protected authentication; `mine_to_follow` represents this clinician’s declared responsibility, **not** a generalized multi-user authorization/role model.

## 3. Minimal reuse of actual source

**Source-proven (at baseline):**
- `clinic_utilities/rf/persistence.py`: `RFApplicationORM(id, patient_identity_key, pathway, indication_code, laterality, payload_json, created_at)`, `RFSavedDraftORM`, `RFDoctorProfileORM`, `RFProcedureHistoryORM`, `RFBase.metadata.create_all`.
- `clinic_utilities/rf/api.py`: protected `/clinical/clinic-utilities/rf` router; `/api/create`, `/api/reuse/search`, `/api/reuse/{kind}/{source_id}`, structured saved drafts CRUD, authenticated server-authoritative doctor snapshot and official PDF build.
- `static/clinic-utilities/rf/{index.html,app.js,styles.css}`: existing form, searches, saved-draft restoration and doctor selector.
- `static/cockpit/doctor-shell.js`: Home currently links RF but does not have real RF outcomes/attention. **Do not change Home in this implementation; project a compact read-only “mine” count later from RF-owned verified data.**

**Choose EXTEND, not parallel mechanism.** Keep issued RF snapshot immutable. Proposed small **RF-owned progress record keyed by existing application ID**, with revision and status/date/known-provenance fields + `mine_to_follow`; if needed, an append-only RF-specific transition/audit record in the same protected database. Reuse RFBase schema initialization; do not add a second database or generic task engine. Amend `RFApplicationORM` only if a supported migration is justified; a new sidecar table avoids unsafe `create_all` assumptions about altering live tables.

Do **not** modify existing `payload_json` to change the contents of previously issued applications. Keep a link `correction_of_application_id` in the **new** application's issued payload only. If storing it in an RF draft prior to issuance, whitelist/validate it in the existing structured-draft mechanism. Old and new applications remain distinct IDs. Inherit/confirm exact patient identity, indication, side and selected active doctor with existing validation; do not silently substitute or fabricate provider identity. New issuance remains the only source of a new official PDF.

## 4. Small UI and flows

Provide a concise **“Αιτήσεις & εκβάσεις”** view/section within the existing RF, not another app. It lists searchable issued application attempts (date, patient, issuer from immutable snapshot if available, pathway, status, responsibility), filtered by **“Προς παρακολούθηση” / “Όλες”** and optionally clinician/date. Display data completeness explicitly, not an empty approval interpreted as rejected.

One application detail has actions appropriate to present state: “Σημειώθηκε ως υποβληθείσα” (with date), “Καταχώριση έγκρισης”, “Καταχώριση απόρριψης” (paste reason), “Προγραμματίστηκε” (date), “Πραγματοποιήθηκε” (actual date; require explicit real-performance attestation), and ownership toggle. Do not offer misleading state jumps unless the colleague-report exception applies and unknown dates remain explicitly unknown. Edit/correct accidental manual entries with a revision check and audit; no silent backdating. Record both the clinical event date as reported and when entered into Cockpit.

**Rejected → New corrected RF (only on deliberate request):** Button uses existing protected previous-application lookup and form/draft prefill; the rejection reason remains visible as context, **not automatically copied into official RF indication text**. Create separate new application, validate same strongly identified patient, preserve link to rejected source, respect the new active issuer and all A1/A2 existing clinical/PDF/imaging checks. Previously attached imaging PDF/JPEG must be selected again (drafts contain no attachments). Do not interpret resubmission as continuation A2 or as a confirmed previous procedure. Only explicitly marking the **new** attempt Submitted starts follow-up according to issuer/opt-in.

**Counts:** Source-scoped attempt counts: pending (submitted, no decision); approved decisions; rejected decisions; scheduled; performed, plus explicit `mine_to_follow` actionable count. Do not double-count `approved` vs scheduled/performed as separate *approved decisions*; report distinct “decision outcome” and “current stage.” Date denominator/time window and missing/unknown outcomes are visible; re-submissions are distinct **attempts**, with optional patient-level rollup. Never count mere PDF generation or saved drafts as a GeSY submission.

## 5. Invariants / finite acceptance proof

1. Protected API and RF DB only; list/search/pagination of issued RF keyed by authoritative application ID; server checks patient/source association. No PHI in logs/URLs beyond existing auth boundaries; rejection text and dates length/format bounded, authenticated and not automatically sent elsewhere.
2. Original `RFApplicationORM` issued context, six-field issuer snapshot, previous A2 treatment/procedure-history semantics remain unchanged. Historical outcomes and un-attributed issuers remain unknown; no backfill by guessing.
3. Every submitted/outcome/scheduled/performed mutation is explicit, scoped to one application, optimistic revision checked, atomic with an audit event where used; duplicates/replays do not count twice. Rejection reason can be recorded as reported even when submission date was never known.
4. **Personal ownership:** own issuance alone does nothing; own explicit submission starts my tracking; colleague issuance or reported rejection **never** starts tracking unless explicit opt-in; taking/releasing follow-up never alters approval result or prior document. No automatic Home tasks or Calendar writes.
5. Rejected → correction produces a distinct new RF application ID, same strong patient identity and linked original; retains original rejection and issued PDF semantics. New form requires new attachments and existing A1/A2 clinical checks. Do not attach an A2 previous *procedure* merely because a prior application was approved.
6. Status timeline/decisions are manual and provenance-labeled; stage/date errors fail closed; counts match actual records and don't fabricate unresolved responsibilities. Tests use synthetic patients/providers only.
7. Reuse existing focused RF regression CI; targeted new API/persistence and browser-handler tests, plus preserved doctor/JPEG/A1/A2/PDF/drafts regressions. No new general test framework.

## 6. Authority and implementation handoff

**Review tier:** R2 because this extends clinical application lifecycle state, persistent writes, and who receives clinician attention.

**Next:** one separate **independent bounded pre-code R2** against this exact design; return PASS/BLOCK/UNKNOWN. After PASS, claim a clearly bounded implementation writer in the root `CURRENT_OPERATIONAL.md`, implement in existing RF only, run finite synthetic RF CI, one affected exact-head independent post-code R2, and separate release approval. Reuse all prior RF #146–#149 settled evidence without reopening it.

**Release, protected live clinical data, auto-deploy, provider messaging, automatic GESY outcome detection and Home UI integration are NOT authorized by this design. STOP.**
