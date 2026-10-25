# Reception → Cockpit call peek — ONE bounded affected POST-CODE R1 request

Date: 2026-10-25. Independent READ-ONLY reviewer, no code changes, merge, deploy, real credentials or PHI.

Repository: `athpapachr-cmd/osteoporosis`. Baseline `main=3204bc7722b5bb45241c5b23d788f6f51c6fb502`. Verify live main, DRAFT PR head and exact changed files before decision. This is a new independent read-only communication projection and **not** activation of B2 patient-linkage or mailbox adapters.

Four affected questions only:
1. Existing Clinical Key still authorizes the sole clinician's normal protected session; missing/invalid key denies; source OFF by default and missing server-only Reception credentials return 503. No new clinical chart, identity, patient link or task write.
2. Read from the established Reception dashboard only, with authenticated HTTPS service-side request, no redirects or client-stored credentials; bounded, sanitized call identity/time/summary projection, accurate pending narrative, no source completeness or fabricated unread count, no raw call/audit/observability dump.
3. New Home phone icon opens a bounded popover showing latest (max 4) actual call summaries and original dashboard link; existing ✉ synthetic/clinical, ☑ tasks and date panels remain unchanged. 401/403/503/late responses and logout fail closed; XSS-safe DOM, escape/outside/focus behavior and responsive layout.
4. Targeted backend + JS/Chromium regression and CI actually executed on final exact head. No overlap with B0/B1/B2 DRAFT dependency branches and no external provider/PHI config changes.

Return PASS/BLOCK/UNKNOWN / COMPLETE/PARTIAL / P0:P1:P2 and minimum evidence. Review PASS is no merge or PHI/release authorization. STOP READ-ONLY.
