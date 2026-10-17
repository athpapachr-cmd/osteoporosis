# Independent focused PRE-CODE R2 — withdraw four-visit technical-link purge (P1-01)

**Date:** 2026-10-17, Asia/Nicosia. **MODE:** fresh independent READ-ONLY, ONE bounded P1-01 delta + affected S4 cumulative PRE-CODE R2. **Repository:** `athpapachr-cmd/osteoporosis`. **PR:** #161 DRAFT / unmerged; branch `feat/cockpit-verified-appointment-visit-brief-v1-20261016`. **No code, merge, deploy, configuration, actual patient/provider access or database mutation.**

## Pinned correction identity

- **Corrected design-bearing commit:** `2785ea1fa7a1cc873070bcc015cd20a6454e47e0`.
- **Corrected design file:** `cockpit/COCKPIT_VISIT_BRIEF_STAFF_ACCESS_PRIVATE_RETENTION_R2_DESIGN_2026-10-17.md`.
- **Corrected design Git blob:** `76ffbd84077a950a921ebd5aaebc772d6768e58c`.
- Product Owner decision update: `cockpit/COCKPIT_VISIT_BRIEF_ACCESS_RETENTION_POLICY_DECISION_2026-10-17.md`, blob `840a45adc3692c0fc6f4bfde3b3263fdebb2760c`.
- Cockpit CURRENT checkpoint: `7c2c789938d39681fe467d9222ff0a2da6d16c24`.
- Original reviewed design was pinned at commit `0ae3ec832022ada3c0889b1910a3caeb35c5e9de`, blob `4245504692b6b2984b4aecfb6e24bfed3eb34c63`. Do **not** substitute the later docs-only head for either reviewed design.
- Original independent PRE-CODE result reported in conversation: **BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:1:0**: S1 PASS, S2 PASS, S3 PASS, S4 BLOCK, S5 UNKNOWN (unverified live configuration). **P1-01** concerned unreliable `signed_at > source booking start` as proof of four *subsequent* completed visits. This request does not claim that the correction is already independently accepted.

## Product Owner changed requirement

The Product Owner **withdraws** the numerical four-subsequent-visit deletion rule entirely: patients may attend once or return much later. Neither four encounters, elapsed six/24 months, inactivity nor any inferred visit count should automatically delete booking↔patient associations. The medical chart remains separate and protected. A link and staff/link audit still need a documented, purpose-limited lawful retention/review and qualified erasure policy before identifiable activation; absence of an approved number **is not** automatic authorization for indefinite retention.

## Narrow review gates

**D1 / P1-01 closure:** Verify all normative language and synthetic proof in the *corrected* design no longer relies on counting visits, post-booking `signed_at`, or any fixed time/inactivity cutoff. A single-visit and long-return-gap patient must retain chart context. Distinguish booking-specific technical links from the statutory medical chart. Ensure old CURRENT/R2 documentation is clearly historical/superseded rather than remaining active design authority.

**D2 / S4 affected cumulative:** Confirm minimum versioned association and independent per-opening registry verification remain intact (a stored UID/name/association does not prove attendee continuity). Show scoped clinician-governed auditable correction/revocation; state that `unlink`/revocation does not necessarily erase `patient_id`. Actual privacy erasure/de-identification, when separately policy-qualified, must be guarded, transactional/idempotent/CAS, respect active/future bookings, open clinical follow-up and holds, preserve a minimal stale-relink-resistant tombstone, and never delete medical chart, approved visits, medications, pending or required audit. No autonomous purge scheduler is introduced in v1.

**D3 / S5 deployment qualification:** Retain fail-closed production gate for independent clinician+secretary principals, role/audit controls, reviewed migration, protected DB location/least privilege/TLS/backup and separate lawful link/audit retention. Render auto-deploy main; Aiven European region and owner-reported backup/restore or smoke are relevant context but not by themselves proof of private DB or GDPR clearance. If configuration cannot be independently verified without secrets or PHI reads, state **S5 UNKNOWN for real activation and release HOLD**; this alone does not require inventing another deletion rule.

**Reuse:** S1–S3 prior design PASS unless new delta affects a trust boundary; original Visit Brief design and P1/P2 post-code independent PASS remain CLOSED. Home #160, Calendar, Visit Capture Save, RF, Clinical Inbox, Reports, data import and external providers are outside this review.

## Required disposition and STOP

Return **PASS/BLOCK/UNKNOWN; COMPLETE_FOR_DECLARED_SCOPE/PARTIAL; P0:P1:P2; P1-01 CLOSED/OPEN; D1–D3 PASS/BLOCK/UNKNOWN**, with narrowly cited GitHub source lines and minimal affected corrective action. Do not relitigate settled areas. An independent design PASS does **not** grant live identifiable data use; staff auth/role implementation and post-code R2 plus qualified private storage, audit retention and explicit release authority remain necessary.

**STOP — READ-ONLY.**
