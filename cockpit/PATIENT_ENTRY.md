# Shared patient entry — PO request 2026-10-18

Problem: Home has registry search without create; a booking opens a gated Brief without any usable manual chart path on freshness/staff-read failure; the capture URL still serves the obsolete JSON form. The PO explicitly requests creation and opening an existing chart from all three entries. This is the next prerequisite for using installed CP1/CP2 with real clinic patients; CP3–CP5 and unrelated PR1 remain outside.

## Reuse and authority

Use the existing PatientORM/clinical_patients registry and protected /clinical/patients search, existing chart and CP1 natural text capture/context/Save. One shared native-dialog patient selector/creation form serves Home, booking preview and old capture entry. No new registry, account system, data store, model call or migration. Existing permissions equivalent to POST /clinical/patients apply to creation; secretary/unauthenticated sessions do not gain clinical registry access.

A displayed booking name may prefill an editable name/search field even when protected Brief freshness verification fails. It grants no chart GET or historical Brief authority, is not recorded as a booking-patient link, and is not a verified identity. The doctor explicitly selects a protected search result to open that exact patient_id, or explicitly submits a new chart. Existing strict booking-link/read-confirm machinery remains unchanged and available separately. Never auto-match or auto-open by name/UID, and do not silently attach a selected clinical record to a booking.

## Creation contract

Add a strict create-only adapter beside the existing upsert route in clinical_data.py, retaining the same protected dependency and ORM owner. Input: client UUIDv4 creation_id, nonblank full_name, optional DOB (valid nonfuture date), optional phone. No caller-supplied arbitrary demographics, patient_id, linkage, synthetic marker or role. Derive patient_id as patient-{creation_id}. Store only explicitly entered normalized fields. No visit/Pending/provider call on creation.

Existing ID with identical entered demographics is an idempotent replay; changed payload conflicts without overwriting the record. Unique primary key plus IntegrityError rollback/readback closes concurrent identical request races. UI disables form while in flight, retains the same creation_id after uncertain errors/edits in that dialog, and ignores late results after close/switch/logout. Different people with the same name are permitted; show matching existing charts before explicit create, never infer identity or merge people by name. Duplicate-click/request retry creates only one chart. Parse validation privately with bounded input/no source echo and no-store response.

## UI and routing

Home: visible New patient button, plus create affordance on no search results. Shared dialog shows name search results with stored DOB for disambiguation, one explicit Open chart/Start visit action per result, and create form (name required, DOB/phone optional). No extra duplicate identity checkboxes.

Booking preview: always offer manual Search / New chart from the displayed booking name, independent of optional Brief read success. Closing the preview clears its state; shared entry handles protected registry access independently. No booking UID/observation passed to creation/Save or stored as link. Existing verified-link flow remains separate.

Capture URL /static/cockpit/visit-capture/ redirects to the normal Cockpit capture entry, which opens this same selector and starts the existing natural-text composer after selecting/creating a chart. Preserve a bounded explicit patient_id query only as a protected chart selection, never as write authority. Redirect occurs before obsolete UI loads; legacy parser files may remain for existing unit contracts but are not a clinician entry. Home command/library links use the same URL, so they converge automatically. Chart exports a beginVisit action reusing its current button handler.

Auth loss/account switch/close invalidates async search/create/open results and clears hidden DOM/PHI; no browser persistent patient/draft cache. Native dialog has Escape, labelled controls, focus restore and a usable login action when necessary. Draft-sensitive entry/navigation must not silently discard an active natural-text candidate; either keep entry actions unavailable during capture or require one clear discard confirmation for actual nonempty draft navigation.

## Finite evidence and release

API: input/auth role boundaries, same-request replay, changed-payload conflict, no overwrite, concurrent unique-ID behavior, no encounter/Pending created. Browser1280/390: Home create→empty chart→natural composer; appointment prefills search and explicit existing selection works despite Brief failure; namesakes do not auto-open; no-result create; capture URL redirects and supports both existing/new; stale search/create/logout do not reopen another patient; draft guard; refresh/second session find created profile and no duplicate; no document overflow. Reuse existing CP1/CP2 Save/lifecycle proof; run affected gates only.

R2 pre-design review and post-code fidelity, finite correction closure if material findings. Deploy through existing native main service after existing canonical/CI checks; bounded live proof using synthetic data only, no real patient creation/edit or automatic booking linkage. Distinguish local creation proof from actual live navigation/write evidence; do not claim unperformed live actions. Durable checkpoint every material transition, then docs-only closeout.
