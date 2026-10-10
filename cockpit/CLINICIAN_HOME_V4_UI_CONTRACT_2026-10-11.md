# Cockpit clinician-first Home V4 — Product Owner implementation contract

Date 2026-10-11. Owner: Cockpit. Product Owner explicitly requested implementing the Apple-inspired workflow proposal after the #142 production UX smoke. Source base: `main=7028c1459ff51c0d6edd0df74298f742f77ccd3f`.

## One screen, four interaction references

- **Things 3 / Tiimo — Today's actual work:** an uncluttered home with source-backed recent clinical encounters and the actionable existing clinic work, not duplicate big appointment cards or decorative teaching boxes.
- **Fantastical — date/day popover:** small top-right date control opens today's appointments in-place. First version may use the existing protected `/clinical/calendar/appointments` source (which intentionally filters *relevant clinical categories*); it MUST label this limitation rather than imply all Reception bookings are available. Full-day/all-specialty projection from the validated Reception source needs a separate R2 protected-read contract; DO NOT invent it in a browser.
- **Linear — Peek:** rail's previous/current/next opens the already proven *appointment-context-only* floating Visit Brief without navigating; never treat an appointment display name as a verified clinical link. Center/recent list opens the clinician workspace with confirmed patient only when protected registry returned it.
- **Raycast — contextual actions:** one small searchable action launcher groups existing surgery, reception, osteoporosis, visit capture, physio, sick leave, medical reports, RF and learning routes. Toolbar remains compact. Original surgery queue and modules survive in a distinct Library/Clinic Work view, not underneath the Home.

## Only source-backed information

- **Recent encounters:** `GET /clinical/recent-encounters?limit=3` presently gives patient name, encounter date, visit_type; it does NOT provide a verified clinical summary. Display a real label when present; show a truthful neutral absence otherwise. A signed clinical summary excerpt (not an invented LLM generation or extra per-row raw visit fetch) is an independent protected-read design/implementation lane.
- **Inbox/Gmail:** no Clinic Inbox ingress or authoritative new-count source yet. The mail popover truthfully says disconnected; NO imaginary 3-new badge, direct Gmail token or email content.
- **Tasks:** reuse existing protected pending surgery queue; a simple category/count may be projected client-side from the already loaded queue, not a new task store or auto-extracted Heidi checklist. Missing RF follow-up/email tracking remains explicitly unavailable.
- **No new protected data query** beyond existing get endpoints; no clinical writes from new surfaces, no source credentials, PHI in URLs/storage, or new provider processor.
- **Responsive:** at half-width Dia/ΓεΣΥ view, hide the narrow rail behind a reachable toggle, preserve the main search and top-right actions, and keep popovers inside viewport. No browser zoom reduction as a layout strategy.
- **Nonduplicative screens:** Home is the default; Clinic Work is a separate in-page view with category navigation to existing surgery/modules/tools, not a giant disclosure consuming Home height. Keep old operational IDs and protected edit handlers unchanged.

## Bounded acceptance evidence and decision

R1 implementation-fidelity review only for the UX/read-only control-plane changes. Tests: focus/escape/outside click, one popover at a time, stale/unavailable source, no fake email count, correct calendar label and local-day boundary, rail Peek vs center open, action/search navigation, no Home footer-bloat at 640–700px, no clinical POST during new UI, old surgery and module DOM still present.

A distinct R2 review is required before expanding the protected Calendar full-day or recent clinical-summary schema and before enabling actual Gmail/Zadarma/ToDos. This PR's code is **not** the full Clinical Inbox, provider integration, longitudinal Visit Brief, or signed Dia Save. No production merge/deploy authority is conferred by implementing the approved UX.
