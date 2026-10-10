# Cockpit unified Visit V3 — clinician-first integration slice

Date: 2026-10-10. Owner: Cockpit. Product Owner explicitly accepted V3 UX and requested integration/implementation, with **three** rows per quick panel.

## Product behavior (confirmed)

The **existing** `/static/cockpit/` page is the primary entry, not a second standalone product or a separate Render service.

Above the fold:
- primary full-registry search on top, activated by typing; no exposed internal patient ID or JSON;
- two compact panels: **3 actual recently completed clinical encounters** / **3 upcoming appointments today**, ordered meaningfully; each row is one-click to a relevant clinician workspace;
- avoid redundant oversized hero, badges and three instructional tiles;
- contextual patient surface after selection, then an on-demand Dia prompt (visible in the composer with copy) and pasted summary into 3 projections;
- all existing Reception, Surgery, Osteoporosis, Physio, Learning and Utilities links remain available but secondary.

The user may open a *calendar appointment* with one click. Appointment display name is not an authoritative clinical patient link. The appointment card is visible immediately, but patient-specific record content and writing stay unavailable until a protected existing patient is selected/confirmed. A *recent clinical encounter* has an already stored protected patient ID; its row can open that patient's context with one click, using the existing clinical owner. Search results require a clinician click, not auto-linking by name/phone.

## Existing owners / bounded read extensions

REUSE:
- `GET /clinical/calendar/cockpit-context` — already protected and already consuming Reception's bounded real schedule. Add `upcoming_today[]` of at most 3 existing minimized `CockpitAppointment` records without a new calendar fetch or appointment writer. Include only appointments starting later than now and before Cyprus local midnight. No inferred protected patient linkage.
- `clinical_data.py` existing `PatientORM`, `EncounterORM`, `GET /clinical/patients?query=...` from parent PR #140. Add protected `GET /clinical/recent-encounters?limit=3` projecting only completed encounter metadata and resolved protected patient name (if stored), NOT encounter payload, raw Dia text or patient identifiers in display. Order by encounter_date/created_at. This is a read projection, not a new patient registry.
- original Visit Capture `/context`, `/preview`, `/save` as the *only* authoritative protected write path; no alteration of its reviewed contract in this slice.

**No new provider integration, new data store, independent calendar, AI model, or new feature flag.** The Dia plain-text editor is a temporary client-side candidate/review UI; browser memory only until actual vetted VisitCaptureCandidateV1 exists for a protected save. Explicitly do NOT convert three arbitrary prose sections into asserted structured clinical facts. The existing protected Visit Capture Save must not be called from this provisional plaintext preview. The clinician-facing Save action therefore remains disabled/clearly pending typed mapping, rather than falsely claimed working.

## Specific behavior

- Keep exactly 3 (not 4 or 20/100) rows in each quick-view panel. Pagination results can be bounded for technical performance but search coverage is full.
- Empty/unavailable/unauthorized state is truthful, not synthetic filler in the real Cockpit.
- Search and row selection are responsive and keyboard-accessible.
- No PHI in URL query strings, localStorage, analytics or logs. Internal patient identifiers may be held transiently in JS when returned by the protected registry/clinical encounter owner. No raw clinical content is fetched until confirmed patient context.
- Pasted Dia content is rendered via textContent and can be corrected only in browser state; changing patient discards it.
- The clinician-visible Dia prompt stays inside the collapsed-on-demand composer, not a permanent element consuming initial workspace.
- Osteoporosis Module 01 remains owned by its ongoing independent workstream: retain the existing module link, do not change its code. Future deep link waits for the owner's supported entry contract.

## Bounded evidence and admission

This slice includes patient-specific protected read projection and calendar/provider linkage semantics and is conservatively classified **R2**, with one affected independent design/read-authority review and one post-code implementation fidelity review before production merge. The original Visit Capture parent R2 and PR #140 A1–A3 can be reused, not repeated.

Acceptance checks:
1. Three completed visits ordered by actual encounter date; three next-today scheduled appointments, no ambiguous fabricated recent data; unavailable is explicit.
2. One-click opens context. Appointment name alone never grants patient-specific record/write linkage.
3. Search covers the whole existing patient registry, despite finite per-query result limit.
4. Dia prompt visible only when requested; plaintext from Dia creates 3 browser-only views; Save never occurs implicitly.
5. Baseline Home, Surgery queue and learning/tool navigation regression unchanged.
6. Synthetic end-to-end tests only; no identifiable records, external provider reads, merge or deploy by the author.

## Held after this UI implementation

The precise typed Dia-to-VisitCaptureCandidateV1 acceptance/mapping seam and any real identifiable processing approval are separate R2 additions. Neither is falsely solved by this layout work. PR #140 A4 full-registry-search existing review hold remains a prerequisite to release if its implementation is reused.

## Product Owner next visible smoke

Open one *recent encounter* and one *upcoming appointment*, search a patient, request/copy Dia prompt, paste synthetic 3-section text, edit a section, switch patient and verify the draft cleared. Do not enter real patient data in an unqualified Dia flow.
