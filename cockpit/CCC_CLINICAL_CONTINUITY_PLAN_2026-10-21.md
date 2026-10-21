# Cockpit Clinical Continuity — eight sequential milestones

**Product Owner:** approved start 2026-10-21. **Execution:** one milestone at a time; only CCC-01 active. Explicit acceptance required before moving to CCC-02. Source-of-truth patient obligations remain `ClinicalPendingORM` / existing Pending checkpoint event owner; do not fork new task engine.

| Milestone | User-visible result | State |
| --- | --- | --- |
| **CCC-01** | Full obligations page grouped by patient/category/deadline, genuine due-now Tasks icon, deterministic read-only paging/filtering | **ACTIVE implementation; DRAFT / NO RELEASE** |
| CCC-02 | Authorized filtered Gmail arrival in independent Clinical Inbox with durable Reception-owned incremental intake/outbox; Cockpit cold-start safe | NOT STARTED |
| CCC-03 | Clinician-verified source patient↔pending correlation; one incoming-result signal linked without closing awaited clinical task | NOT STARTED |
| CCC-04 | Deliberate clinician review and recipient/text-approved communication instruction, no dispatch | NOT STARTED |
| CCC-05 | Reception-owned actual outbound delivery, idempotent status, failure/retry; existing provider only | NOT STARTED |
| CCC-06 | Reception call/callback outcome + protected verified patient/pending linkage; caller contact != result-review or clinical closure | NOT STARTED |
| CCC-07 | Single source-controlled clinical-attention timeline projected into Home, chart, Visit Brief, Tasks and Reception; no duplicate states | NOT STARTED |
| CCC-08 | Full synthetic→small qualified real clinic pilot including errors/conflicting identities/retries; PO acceptance | NOT STARTED |

## Release rule

For each milestone: bounded design, implementation against existing owners, affected CI, proportional independent review, PO clinical trial/feedback, then accept or correct. No automatic advancement, unapproved external communication, provider secrets or PHI in tests.

**CCC-01 scope precision:** Without an already persisted, trustworthy trigger event for an email/call/next visit, `trigger_text` is NOT a due event. Current-day alert derives only from explicitly recorded valid `review_date`. Future/undated records remain accessible in the full page. Structured event-activation after email/call/appointment belongs to later qualified milestones. No new task state, DB table, identity authority, provider, or clinical write is introduced in CCC-01.

The separate Clinical Inbox remains a source-review domain rather than a view of stored patient Pending. The Tasks icon is selectively actionable attention only, not a global count of active Pending or surgical totals.

## Owner boundaries

- Cockpit/Clinical Excellence: clinician-authored Pending, result review, clinical disposition, patient chart and Visit Brief projections.
- Digital Secretary/Reception: calls, message/callback action and delivery lifecycle, durable incoming adapter; only bounded verified events shared.
- Provider: actual email / SMS / voice transport.
- Strong patient identity evidence and explicit clinician authorization are mandatory before protected clinical linkage or outbound instructions.
