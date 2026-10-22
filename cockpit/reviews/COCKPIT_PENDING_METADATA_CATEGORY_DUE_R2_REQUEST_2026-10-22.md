# Cockpit — Clinical Obligation Category / Due Date R2

**2026-10-22 Asia/Nicosia · One bounded independent R2 review.**
Do not infer merge, release or real patient permission. Author implementation candidate is prepared on a DRAFT branch; this is NOT an independent review receipt.

Base fresh `main=8708a37c0d4083c3f811ced6b186e9ef7e817355`.
Design: `cockpit/OBLIGATION_CATEGORY_DUE_DATE_EDITOR_2026-10-22.md`. The patient obligations screen #205 is already LIVE. Proposed deltas exclusively touch existing Pending clinical authority, event ledger and existing chart/list UI.

## PRE-CODE design review (R2)
Classify the exact new behavior: patient-level corrected category / date is authoritative clinical metadata, not a cosmetic sort key.

Four questions:
1. **No second writer / source-truthfulness:** `ClinicalPendingORM.item_type` and `.review_date` already persist; mutate only these two columns, preserve signed Encounter and task ID/text/status, original provenance and checkpoint plan. Record before/after with existing PendingCheckpointEventORM, actor and time, not a new store/schema.
2. **Identity/actor/replay/stale:** existing `actor_for`, signed source validation and exact patient-pending ID, same staff-cookie clinician restriction as Pending checkpoint; append-only event and expected global revision unify across corrections/checkpoints/lifecycle. Idempotency same request only; invalid/secretary/cross-patient/stale/deleted/cancelled fail closed; caught conflicts rollback.
3. **Deadline semantics:** `open` ≠ due now; user-chosen date is authoritative. Derivation from task title must be opt-in suggestion only, anchored to exact source encounter_date, with “1,5 μήνα” explicitly approximate (~one month + 15 days) and 1,5 years = 18 calendar months. No silent relative date based on today's date; no auto-completed tasks; missing source date gives no suggestion.
4. **Existing product/UI:** one-tap Edit from main Obligations opens *existing* protected patient task popup with field editor expanded, and popup itself offers editor; one explicit Save, no extra login; existing Clinical Inbox (email) stays entirely separate; avoid 100-row coverage overclaim, preserve clinical drafts and cleanup.

Return PASS/BLOCK/UNKNOWN; COMPLETE_FOR_DECLARED_SCOPE/PARTIAL; P0:P1:P2 and only material defects. *If blocked, correct only affected seam.* No reviewer/agent can grant database/production authority from this alone.

## POST-CODE exact-head fidelity R2

Inspect frozen code and tests on the PR head, including:
`clinical_pending_one_tap.py`,
`static/cockpit/patient-chart.js`,
`static/cockpit/obligations-view.js`,
`static/cockpit/doctor-shell.js`,
`static/cockpit/index.html`,
`static/cockpit/doctor-shell.css`,
`test_cockpit_cp2.py` plus relevant unchanged sources. Reuse earlier CP1/CP2/Home/Visit Brief pass. Inspect actual GitHub CI and real Chromium 1280/390 synthetic evidence. The reviewer may identify new material safety defects, but must not invent a need to re-review the entire Cockpit, perform a second architecture loop or read real patients.

No merge, production deploy, actual patient changes, Gmail read, provider calls, new schema, Aiven changes or secrets before final review and separate explicit clinician go/no-go. Product Owner's real-use correction authorizes draft implementation, not unreviewed clinical writes to live Aiven.
