# Native RF ⇄ Reception Cal.com exact-UID association — one independent PRE-CODE R2
**2026-10-15 · READ-ONLY / DESIGN-ONLY · NO IMPLEMENTATION/PRODUCTION DATA.**

Repository `athpapachr-cmd/osteoporosis`; design-only PR (see PR metadata), base/main `686a8f68e3330e467a6422056296faa52a8cacfd`.
Read and inspect frozen:
- `clinic_utilities/rf/RF_RECEPTION_EXACT_UID_LINK_V1_DESIGN_2026-10-15.md` (exact blob provided by PR HEAD).
- Prior [RF Tracking/Calendar approved design](../RF_TRACKING_WEEKLY_CALENDAR_V1_DESIGN_2026-10-14.md) §5; reuse its unaffected R2 findings and released #156/#157.
- Actual `clinic_utilities/rf/{api.py,persistence.py}`, `clinical_calendar.py`, current RF UI and source transport guards.
- Reception source `athpapachr-cmd/ortho-reception-backend-v2` `schedule_projection.py` main `bb591b519074317750940046b37fd7c30d0ef82f` (filter canceled/rejected and UID conflict, coverage, UTC fetch).

**One bounded source-backed PRE-CODE R2:** check (1) manual exact RF ID ↔ unique exact Cal UID/no name-phone-date inference and explicit clinician attestation; (2) source freshness 0..300 sec, complete date/week coverage, source active cancellation filtering, duplicates and interval validity, fail-closed errors; (3) single RF-owned versioned link+austere audit, conflict/locking, unlink, exact-ID deletion, no immutable RF modification; (4) no premature clock unless RF date equals Cyprus-local booking date, 300s absolute expiry without refresh, async/auth invalidation; (5) minimal protected contracts, no provider writes/second registry/Home PHI and scope-appropriate synthetic tests and release gates. Real RF at 27 October is **not** assumed to have a confirmed UID/date; never query or mutate production patient data during this review.

Return `PRE-CODE R2 PASS/BLOCK/UNKNOWN`, `COVERAGE COMPLETE_FOR_DECLARED_SCOPE/PARTIAL`, P0:P1:P2 and only material findings with actual source paths/lines, smallest correction if needed. If PASS, bounded implementation candidate can be requested in a separate authorization. NO code/merge/deploy/database/real patient/Cal writes and no new review loop without new material evidence. STOP.
