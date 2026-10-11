# Native RF v2 — managed doctor directory: bounded correction design

**2026-10-11 · DESIGN ONLY · pre-code R2 pending · no runtime write.**

## Product Owner correction
The existing native RF v2 works and PR #146 has independently closed its sole JPEG-integrity P1-01 (PASS, 0:0:0). The Product Owner does **not** want doctor details in source code or manually added Render environment variables, and does **not** assume responsibility for tracking approvals, scheduling or procedures on applications prepared for other clinicians.

**Exactly one replacement of the doctor-selection mechanism remains before PR #146 release.** Leave the reviewed Horos JPEG implementation, PDF layout, imaging validation, A.1/A.2 and existing RF clinical logic unchanged.

## Existing mechanisms to extend (verified at PR #146 HEAD cf69e189)
- `clinic_utilities/rf/catalog.py`: existing `DoctorProfile` fields and `doctor_profile_for_key` currently hard-code `papachristou` / `maxoutlou` to Render variables.
- `clinic_utilities/rf/api.py`: existing protected RF router, contract, `doctor_key` in `RFApplicationDraft`, server-generated PDF and `record_application` **after** complete PDF build.
- `clinic_utilities/rf/persistence.py`: existing RFBase, protected PostgreSQL-backed RF application table and `payload_json`; `initialize_rf_tables` uses metadata `create_all`.
- `static/clinic-utilities/rf/{index.html,app.js,styles.css}`: doctor dropdown and existing RF form. No second app, engine or external integration.

## Minimal intended UX
Keep the current compact doctor dropdown in “Ασθενής & προϊόν”, plus one unobtrusive **Διαχείριση ιατρών** control. Within the same RF screen, support:
1. **Προσθήκη ιατρού**: six current official-form fields (name, GeSY code, specialty, medical centre, phone, email).
2. **Επεξεργασία** the six fields of a registered doctor for future applications.
3. **Αφαίρεση από επιλογές** (deactivate/archive); do not hard-delete historical attribution.
4. **Επαναφορά στην επιλογή** when wanted. No technical Render/GitHub access needed for any operation.
5. Keep current user's profile selected by default. Another doctor's profile is selectable only if explicitly active.
6. On deployment the existing user can add ASAN MAXOUTLOU through this protected UI using the supplied information; **do not embed colleague's contact data in repository code, fixtures, docs or env secrets**. No separate prerequisite Render setting for this doctor.

## Data and authority contract
- Extend the *existing* RF persistence owner with a small `clinic_rf_doctor_profiles` table or equivalent existing RF-managed record. No new service or database. Auto-create the minimal table through current RFBase init or narrowly justify a migration.
- Stable server-generated `doctor_key` (up to draft contract limit 40), unique and never reused. Reserve existing default key `papachristou`; preserve missing-key -> legacy default solely for **new legacy-client requests**. Explicit null/blank/unknown/inactive keys fail closed.
- Maintain the existing `RF_DOCTOR_PROFILE_JSON` as the **backwards-compatible default source** on first use / until managed default data exists; do not require ongoing configuration edits. Allow editing that default from protected RF UI thereafter. Handle loss/malformed default config explicitly; do not silently assign a different doctor.
- At `/api/create`, resolve the chosen active doctor **server-side**. Before generating/storing the RF, capture its exact six fields as an immutable server-produced doctor snapshot in the **new** application's existing `payload_json` together with `doctor_key`, preserving issued-document identity when directory rows are later edited/deactivated.
- Previous application rows without `doctor_key` or snapshot remain **historically unattributed**, not retrospectively assigned. A.2's actual prior RF procedure/history is never attributed to the currently selected doctor.
- Read/write doctor profiles only through the existing `/clinical/clinic-utilities/rf` protected router; return the fields needed to edit via authenticated read, not via public/no-auth routes. Do not trust doctor details sent with an RF draft or permit client-side identity substitution. Use narrow field, length, empty-value validation, stable IDs, and explicit status. Include safe failure behavior on concurrent edit/deactivation vs submit.
- Editing/deactivation must not rewrite previously issued application payloads or PDFs. Historic snapshots remain immutable.
- No patient identifiers or attachments added to the doctor directory. No browser localStorage or duplicate owner.
- Prefer compatibility with existing tests/clients and `doctor_key` semantics; the hard-coded second Render variable `RF_DOCTOR_MAXOUTLOU_PROFILE_JSON` is **not** a release prerequisite and must cease being required for the new doctor's UI workflow. Any compatibility alias must not create two competing active sources of truth.

## Responsibility is separate from issuer
- A doctor chosen for an RF is the **issuer**, not automatically the person responsible for monitoring the outcome.
- Preparing an RF for a colleague must **not** create an unresolved “my approval”, scheduling or procedure task, and must not pollute the Home “Χρειάζονται προσοχή” counts.
- **No RF approval/scheduling/tracking implementation now.** On its later milestone, tracking must represent **explicit accepted follow-up responsibility**, not infer it from who typed the application or from provider identity.
- A later **rejection reason → new linked corrected application on request** remains a future feature. Do not erase or overwrite the rejected original; no automatic resubmission or outcome obligations in this slice.

## Affected implementation seams only
`clinic_utilities/rf/catalog.py`, `clinic_utilities/rf/persistence.py`, `clinic_utilities/rf/api.py`, and existing RF `index.html`, `app.js`, `styles.css`, plus focused RF tests/checkpoints. Existing `pdf.py` and JPEG behavior should be reused unchanged; only adjust if source-proven unavoidable.

## Minimal acceptance evidence for future implementation
1. Protected create/list/edit/deactivate/reactivate; doctor defaults and server-side resolution, with real DB roundtrip using **synthetic** physicians.
2. No Render update to add another physician; directory persists across client refresh and server restart; no code change for Nth doctor.
3. Legacy omitted key, explicit invalid key, deactivated key and concurrent change behavior; all invalid submit paths produce no application/history writes.
4. Exact six doctor PDF fields and persisted immutable doctor snapshot, including before/after directory edit; no historical retroactive attribution.
5. A.1/A.2, learned medicines, imaging-PDF mandatory check, Horos JPEG EXIF/order/integrity, protected access, and preexisting RF regression evidence remain valid; rerun only changed affected seams.
6. No new pending approval / surgery / Home state from creating a colleague's RF.

## Review and release boundary
New protected directory write authority plus historic attribution makes this a **focused R2 pre-code design correction**. Perform one independent design PASS/BLOCK/UNKNOWN on this exact doc. Then bounded author changes on existing PR #146, and one focused exact-head affected post-code R2. Reuse prior P1-01 JPEG correction closure and its validated blobs if unchanged.

No approval for merge, Render deployment, real patient/doctor-provider tests or new RF tracking. **STOP after design handoff.**
