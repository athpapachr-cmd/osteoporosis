# Independent pre-code R2 — Visit Brief clinician+secretary access / private retention

**DATE:** 2026-10-17, Asia/Nicosia. **MODE:** fresh independent READ-ONLY, single bounded PRE-CODE R2. **Repository:** `athpapachr-cmd/osteoporosis`. **PR:** #161 DRAFT / open, target main. **New design:** `cockpit/COCKPIT_VISIT_BRIEF_STAFF_ACCESS_PRIVATE_RETENTION_R2_DESIGN_2026-10-17.md`. Do not implement, mutate PR, deploy, query live clinical data or external provider secrets in this review.

**Reused prior acceptance:** Verified Appointment↔Patient Visit Brief original design R2 PASS after P1-01/P1-02; bounded post-code P1/P2 delta independent R2 PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0 at frozen tested `3a98f7c046a83f05e0590661118c92c972483026`. Existing Home #160, Calendar, Visit Capture Save, Clinical Inbox, RF and Reports out of scope. The *new* role/security/retention authority is not established merely by old PASS.

## Bounded gates

**S1 — independent staff authentication and consented scope:** Distinct clinician and secretary actor principals, individually revocable, no shared-key actor attribution, secure cookie/session, server-enforced read-only secretary clinical summary permission; secretary may read bounded previous signed visit/plan/pending but never clinical Save/sign. Show no accidental secretary elevation through existing `ClinicalCookieMiddleware`, `clinical_data.py` global read/write routes or Calendar source. Do not require broad global patient registry access.

**S2 — unqualified booking identity:** Same current Reception `ActualBooking` lacks stable attendee identity. Confirm requires fresh source, protected independently verified registry patient selection and explicit per-opening action; repeated read remains booking-only even if UID/name unchanged, no full chart/history prefetch. Source authority and legacy P1/P2 unchanged.

**S3 — actor audit and secrecy:** minimal auditable actor identity for each PHI brief read/confirm/correction/revocation/denial; avoid staff credential and clinical data leaks to GitHub/logs/browser/URLs. Clarify schema migration from existing `operator_type=shared_clinical_key`; long-lived audit has separate retention/legal-hold logic.

**S4 — four actual *subsequent* completed visits and safe data life cycle:** signed/completed prior chronology rather than appointment count/elapsed time; preserve medical history and pending; active/future booking/hold skip; idempotent CAS cleanup with tombstone privacy/re-link safety. Legal/audit retention must not be invented. Six-month return without four visits should never lose chart context.

**S5 — real production boundary:** verify actual backend runtime model and Render `main` auto-deploy; independent individual access + retention/privacy + private durable database/migrations/config before enabling identifiable links. If real DB destination is unknown, leave release HOLD rather than assuming. No external provider writes/AI/CareTasks.

## Required disposition

Return **PRE-CODE R2 PASS/BLOCK/UNKNOWN; COMPLETE_FOR_DECLARED_SCOPE/PARTIAL; P0:P1:P2; S1–S5 PASS/BLOCK/UNKNOWN**, narrowly cited source lines and minimal affected corrective action if needed. Avoid repeating settled reviews. **STOP: READ-ONLY.** This request itself does not authorize release.
