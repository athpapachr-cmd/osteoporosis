# Native RF v2 — A2 continuation, single-search retrieval and saved drafts (bounded v1)

**DATE:** 2026-10-11 · **STATUS:** FROZEN PRE-CODE DESIGN CANDIDATE · **MODE:** DESIGN ONLY
**Source:** athpapachr-cmd/osteoporosis; source baseline main 7186a26105e6edd37ed02b089781ce3909c2eaf3; branch design/rf-v2-a2-reuse-and-draft-20261011.
**Owning app:** Existing native, authenticated RF v2 utility. **Classification:** R2: protected patient search/linkage plus new persistent clinical drafts. No implementation, deploy, real documents or clinical records authorized by this design checkpoint.

## 1. Clinician checkpoint and outcome

**Before:** Switching to «Συνέχιση θεραπείας» presents almost the same blank form as «Νέα θεραπεία». Previous procedure search refuses to begin until identity, indication and side are supplied. Earlier issued RF applications are stored but not shown as reusable clinical sources. Leaving halfway loses unsaved state.

**After:** Open A2, either upload an old RF PDF immediately or enter **one** search term (name, identity, GESY). Pick the correct result, inspect what was recovered, then update only today's fields and any missing prerequisite. At any stage (A1 or A2), click **Αποθήκευση πρόχειρου**. Later reopen from recent drafts or the same single search, without refilling the form. Example: earlier knee RF in protected history loads patient + site + side + previous product. The clinician supplies only the actual procedure details if not yet documented, today's follow-up and any applicable current notes; then generates the existing official A2 PDF.

**Unchanged:** Existing A1 creation, immutable managed-doctor issuer snapshots, official A1/A2 template, imaging/PDF validation, JPEG processing/order and existing transaction authority. Do not create a second patient registry or storage service.

## 2. Grounded starting point

- clinic_rf_applications already stores submitted RF applications, patient name, identity, GESY, indication, unilateral site, product and an application JSON payload; it has no clinical search/retrieve UI.
- clinic_rf_procedure_history separately stores **actual** RF procedures, and its backend query already supports optional site/side filters; existing POST /api/history instead demands laterality and browser insists on three fields.
- A2 currently accepts previous procedure history or explicit legacy procedure details. An approved/issued application is **not proof of an actual procedure**.
- Currently no server-side saved-draft entity exists. PDF/HP JPEG uploads are transient. Existing RF router is protected by the same clinical authentication boundary. Existing client uses no localStorage/sessionStorage for RF.

## 3. Minimal screen and retrieval interaction

- Keep one RF utility. A2 starts with a compact «Συνέχεια από προηγούμενη αίτηση» choice: **Φόρτωση παλιού PDF**, **Αναζήτηση RF**. Do not require patient identity, indication, side or previous data *before* either action.
- One search box: «Όνομα, ταυτότητα ή αριθμός ΓεΣΥ». Normalize Greek case/accent variants, ordinary spaces and formatted identifiers; show up to 20 records with bounded pagination, most recent first, from existing **RF applications** and actual **RF procedures**. No compulsory diagnosis/side filters; they can appear *after* results as optional filters. A name-only match never silently picks a patient.
- Explicitly select a candidate with origin (application / actual procedure), date, indication/site, side and minimally necessary identifying summary. If multiple patients match, present separate records and request an explicit patient/source choice. Do not auto-populate another patient's draft on a fuzzy match.
- «Τα πρόχειρά μου» / «Πρόχειρες αιτήσεις» gives recent saved drafts with date and minimal identifying label and an explicit **Συνέχεια** action even if the draft has no identity yet; also searchable using the same single field when available. Existing RF access boundary applies; do not assert a per-clinician ownership model unsupported by current auth.
- The A2 visible workflow should be **source → existing details → today's changes → official PDF**, rather than re-entering the A1 cards. Common reusable fields are auto-populated with visible source indicators and remain editable. Only missing/current fields demand attention. User inputs already typed are not silently overwritten when a source is imported.

## 4. Importing a previous RF PDF

- A separate protected upload endpoint accepts an old RF PDF without any preceding patient fields. Validate PDF type, bounded file size/page processing, the supported official RF form structure and recoverable machine-readable text using the *existing PyMuPDF dependency*. Parse only the relevant official pages with known field/checkbox positions from the packaged official form. Do not perform OCR or infer fields from arbitrary radiology prose.
- Treat parsed values as **provisional source candidates**, showing the extracted patient identity, demographics (as historical), indication, site, laterality, product and any recoverable original A1/A2 data with per-field origin. If a field or checkbox cannot be reliably read, leave it empty and mark «Χρειάζεται συμπλήρωση/επιβεβαίωση»; do not guess.
- When an identity is reliably extracted, the server may look up matching existing RF records for **explicit selection** and reconciliation; distinguish persisted record versus uploaded PDF, showing conflicts for clinician resolution. A legacy/scanned/incompatible PDF should yield a clear partial/unreadable state and permit protected single-field search/manual completion, not turn into a false confirmed procedure.
- Uploaded previous RF PDF is used **in memory for extraction only** and is not persisted, logged, or mistaken for the mandatory imaging report attached to the *new* official RF application. The current required imaging PDF and optional Horos JPEG uploads remain subject to existing acceptance rules.
- Prior A2 PDF may mention a historical procedure, but *issuing that A2 request itself* does not establish that a subsequent RF was performed. Source provenance must remain explicit.

## 5. What A2 carries forward versus updates

**Prefill when supported:** patient name, identity, GESY, indication/code, unilateral side, site/exact location, eligible product and relevant clinical background from the selected previous application. Keep the originating source ID and where possible last issued application date.

**Needs present-day check:** age (previously recorded age is not automatically current age), current follow-up date and VAS, clinically relevant changes and any new medication/note information. The current official A2 template does not have A1's medication trial table: new medication changes may be recorded as reviewed additional information only in the official field that actually exists; do not claim a different template field is available.

**Actual procedure:** if there is an identity-linked actual procedure record matching site and laterality, allow explicit selection and populate its actual date, VAS before/after, previous follow-up. Otherwise ask the clinician once for actual date and required VAS/follow-up values. Never fabricate a date from the previous application's date.

**Issuer:** keep the clinician's currently selected active RF issuing physician. The doctor in the older RF is historical context, not an automatic replacement. If a resumed draft references a now-inactive doctor, require deliberate new selection before create/PDF. Preserve the already-fixed no-issuer-switch behavior.

**Final validation:** reuse the existing protected RF create path, same mandatory unilateral side/site/indication rules, previous actual procedure requirement and official imaging attachment assessment. A2 initial search can be easy without diluting creation-time clinical requirements.

## 6. Save and resume a draft (A1 and A2)

- Add one **protected RF saved drafts** persistence table in the existing RF database through the existing RFBase initialization, separate from final clinic_rf_applications and clinic_rf_procedure_history. No mutation of older submitted or historical records.
- A draft can be saved **incomplete**, including without identity, doctor, indication or side. Store an allowlisted, versioned subset of **structured form state** needed to faithfully restore the present RF UI: pathway, patient fields, selected source references, product/site/side, actual procedure entries, relevant pain/follow-up, medication/adverse/physio and notes, A1/A2 visibility-relevant state, selected active doctor key when present. Do not store unrecognized objects or generated PDFs. Bound field sizes and reject secret/token material. Clinical notes are not printed to logs.
- APIs under existing RF protected router: save new draft, list/search recent drafts, get by opaque ID, update with optimistic revision and delete/discard. Use limited sorted result pages; default recent drafts should be usable with **zero** search fields. Revision conflicts must not silently discard the newer stored version. Reopen reconstructs actual visible controls, not just a JSON blob, and handles outdated contract options explicitly.
- Keep original imaging PDFs, imported old PDF bytes and Horos JPEG bytes **out of draft storage in this first version**, consistent with current no-PDF/JPEG persistence boundary. Do **not** claim a draft saved those attachments. Show explicit «Θα χρειαστεί να επισυνάψεις ξανά τα αρχεία πριν την έκδοση» on save and resume, preserving text fields and selected file status as *not attached*. Any later requirement to retain binaries needs a separate storage/retention approval, not an improvised PostgreSQL blob or ephemeral Render filesystem.
- Saving a draft is explicit and does **not** create a submitted RF application, a prior actual procedure, or official PDF. On successful official create, the linked draft is closed/removed from active list **atomically** with the issued application; failed PDF validation must leave the draft intact. If the same client retries, guard against accidental duplicate finalization where feasible within the single-DB transaction. Provide explicit discard control for abandoned drafts, not silent expiry that might lose work.
- Drafts reside only in existing authenticated RF clinical storage, not public browser storage or new apps; no identifiable patient data or real PDFs in tests.

## 7. Narrow implementation seams — no parallel engine

- clinic_utilities/rf/persistence.py: RF draft entity, protected search of already stored RF applications, basic draft CRUD and revision/transaction operations.
- clinic_utilities/rf/api.py: authenticated search/source retrieval, read-only old-PDF parser endpoint (or one adjacent bounded helper module), saved draft CRUD; keep existing /api/create validation and transactional issuer snapshot unchanged except minimal link to finalized draft.
- static/clinic-utilities/rf/index.html + app.js + styles.css: one A2 source/prefill pane, one search field, recent drafts, save/reopen and clear missing/current fields; preserve original A1 and Horos/doctor UI behavior.
- Existing test_rf_v2_native.py and test_rf_v2_ui.js (or one small focused companion) for synthetic exact-field and browser-handler tests. No new service, external AI, provider setting, background worker, cross-system patient registry, general Cockpit task or changed official PDF template.

## 8. Finite acceptance cases

1. Start A2 with all patient fields empty → upload recognized old synthetic official RF PDF or search by **one** input; no three-field or ten-field gate.
2. DB A1 source → correct patient/site/side/product prefill; cannot infer actual RF. Real recorded procedure → explicit matching selection, correct procedure date and VAS; mismatched patient/site/side never linked.
3. Two matching names → two distinct explicit cards, no silent choice or mixed identity. Greek accent/case and formatted identity/ΓεΣΥ match correctly; bounded pagination.
4. Old official A1/A2 synthetic PDF → recover only supported fields; damaged/scanned/unrecognized or contradictory PDF gives partial/clear feedback, no invented performed procedure, no PDF bytes stored.
5. Save an incomplete A1 and A2 draft, restart server/load page → original partial fields and selected source restored. Missing attachments explicitly flagged; no localStorage/sessionStorage or binary persistence. Draft metadata access stays under current RF authentication.
6. Optimistic stale draft update rejected without data loss; explicit delete removes active draft; save never writes to submitted applications/history.
7. Resume a draft with a deactivated doctor → **no automatic issuer substitution**, reselect required; prior RF import cannot change chosen issuer.
8. A2 final PDF uses verified actual procedure, reattached imaging and unchanged official A2 pages; invalid/missing docs and broken image leave zero new application/history writes and active draft intact. Successful finalization retires the active draft transactionally.
9. Preserve native RF A1/A2 current tests, JPEG entropy/EXIF/order, medical history rules, doctor snapshot, official template and unrelated CU-1 unchanged.

## 9. Authority and stop

Product Owner requested start of the design and the additional Save/Resume feature, but **not** automatic production release, real identifiable testing or novel file-storage configuration. This is a single clinically meaningful **R2** (patient identity + new authoritative draft writes), not the prior cosmetic Horos R1. Require exactly **one independent pre-code R2** on this frozen design before runtime implementation, followed by focused CI and one post-code fidelity review of the same affected scope. No separate redundant reviews or broad architecture redesign.

**Open assumption communicated in this design:** drafts persist typed fields but not attached binary PDFs/JPEGs. Reopening tells the clinician clearly to reattach them; modify the design only if the Product Owner explicitly requests binary retention.
