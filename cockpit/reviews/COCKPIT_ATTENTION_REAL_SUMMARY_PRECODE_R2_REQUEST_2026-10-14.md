# Cockpit Real Attention Summary — Fresh independent PRE-CODE R2 request

**MODE:** fresh independent **READ-ONLY**, ONE bounded, finite pre-code R2. No implementation, merge, deployment, live PHI or external/provider operation.

Repository: `athpapachr-cmd/osteoporosis`; proposed DRAFT PR on `feat/cockpit-attention-source-summary-20261014`; main verified at design activation `3db1a924f77ad8aae14222e4472206b8b18940fa`. Fresh-resolve PR SHA when reviewing. Owning design: [`cockpit/COCKPIT_ATTENTION_REAL_SUMMARY_DESIGN_2026-10-14.md`](../COCKPIT_ATTENTION_REAL_SUMMARY_DESIGN_2026-10-14.md). Product Owner explicitly approved the compact summary and single-surgery-name rule on 2026-10-14.

Reuse original Home V4 independent R1 and native RF outcomes pre-code/post-code R2 closure, plus PR #151 release/Clinical Key stability. Do NOT reopen these unrelated proven implementations.

## Exactly five finite pre-code gates

1. **Existing source and privacy:** Verify current protected `GET /clinical/surgeries` flows via already loaded `static/cockpit/app.js` into `cockpit:surgery-counts` and current `doctor-shell.js` Home rendering. Is deriving date groups and singleton name/procedure **in memory only**, with zero new clinical ID/matching/source fetch/persistence, safe and bounded? Does a day with 2+ pending surgeries suppress *all* names for that day, and does failed source clear prior PHI? Do not treat past scheduled as performed; no synthetic patient name beyond synthetic tests.
2. **RF source semantics:** Examine `POST /clinical/clinic-utilities/rf/api/outcomes/search` (existing read-only search using Clinical Key), `list_rf_application_progress` and `counts/truncated/mine_actionable`. Does the proposed Home use not conflate `issued_unknown` with submitted/pending, or count RF colleagues as my follow-up, and avoid exhaustive global totals on truncated source? No RF patient info in Home. No new RF creation/events write.
3. **Medical reports:** Verify existing `report_api.py` has no durable draft/completed persistence. For this bounded Home slice, is a clear unavailable-status and link truthful, reserving report-status ledger as a separately accepted R2 owner? Do not claim counters without a supported source.
4. **Protected failure/race/mobile:** Confirm required auth, fail-closed 401/403/503/error states, async revisions, no PHI in public console/screenshot artifacts, 390px/680px/1280px and one-tap routes. No regression to shared Clinical Key, compact Home, Clinical Inbox disabled, Save authority, provider ownership. Define the smallest synthetic tests for these cases.
5. **Owner/scope and proportionality:** Verify no new backend/store/schema, no second surgery reader, no patient linkage or external write. Is this bounded Home presentation/protected existing RF query plus separate future reports state rather than one premature broader workflow? Classify real risks and smallest required design changes, not general review.

Return `PRE-CODE R2 PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, smallest defects or blockers, exact reviewed SHA and **STOP**. PASS authorizes only bounded code implementation under Product Owner's already-recorded approval, not release or real-patient use.
