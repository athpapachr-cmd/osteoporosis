# Independent pre-code R2 request — iCloud triage preferences

Status: Original review COMPLETE / BLOCK0:0:1; actual disposition is in `ICLOUD_TRIAGE_PREFERENCES_PRECODE_R2_BLOCK_2026-11-01.md`. This original request is retained as evidence, not a pending second review. Requested 2026-11-01 after explicit Product Owner approval of the narrow one-table operational preference exception.

Exact target `65b665d3eccffc31bd48826a2953fd4d30815930`; source main `b8637bcc8af0deebee3716cf7cd98bc99bf2e912`, fresh verified unchanged. Target design: `cockpit/ICLOUD_TRIAGE_PREFERENCES_DESIGN_2026-11-01.md`. Reviewer `icloud_triage_design_r2` is independent of the design author. Subsequent request/status receipts are docs only and do not change the frozen design.

Decision: Is this agreed sorting-preference design safe and sufficiently implementable for a separate later implementation decision?

Six finite questions:

1. Minimum operational store ownership, provisioning and rollback under ONE table in the existing DB; no clinical-owner writes or old two-table action router.
2. Atomic insert/CAS on SQLite/PostgreSQL, bounded exact replay/eviction, current projection, two tabs and conditional undo.
3. Account/alias/folder/UIDVALIDITY/UID qualification and future-only watermark/edit/invalidation.
4. Manual category vs reset, opt-in rules/conflicts, Greek term scope, filtering before page bounds.
5. Existing auth/origin/privacy, source/store failures and visibility, stale responses/cache.
6. Finite acceptance covers the reported clinician-visible correction/undo/rule failures.

Source inspection only. No implementation, edits, live provider/DB/PHI/config, dependency installation, test reruns, publication, merge or deploy. Existing unrelated canonical/release evidence is reused; inspect directly affected seams only.

Return PASS/BLOCK/PARTIAL, P0:P1:P2, COMPLETE_FOR_DECLARED_SCOPE, per-question disposition and exact design/source evidence. For a material defect state the reachable scenario, violated invariant and smallest correction. Stop once the map is complete or a decisive BLOCK plus affected correction path is established. No review-of-review chain.

## One bounded closure requested after original P2-01

Corrected exact target `8c34467681b78a430cdae6742efb013f669e1aae`; same independent reviewer. Design-only delta adds shared source/preference-revision cache contract, doctor-shell seam and one two-cache/delayed-response browser oracle. Original supported questions reused; closure covers P2-01 plus affected cumulative consumers/degraded state/rollback, no new store/auth/clinical/provider scope. Closure COMPLETE / PASS0:0:0; see `ICLOUD_TRIAGE_PREFERENCES_PRECODE_R2_CLOSURE_PASS_2026-11-01.md`. No implementation or release authority.
