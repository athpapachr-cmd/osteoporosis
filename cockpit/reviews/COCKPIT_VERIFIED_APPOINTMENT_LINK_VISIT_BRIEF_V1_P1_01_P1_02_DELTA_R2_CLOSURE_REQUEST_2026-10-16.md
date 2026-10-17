# Cockpit Verified Appointment ↔ Patient + Visit Brief v1 — independent two-P1 closure R2

**DATE:** 2026-10-16, Asia/Nicosia. **MODE:** fresh independent READ-ONLY one bounded pre-code R2 **DELTA + AFFECTED CUMULATIVE CLOSURE**. No code, tests, PR mutations, merge, deploy, production/provider or real patient data.

**Repository:** athpapachr-cmd/osteoporosis
**Branch:** design/cockpit-verified-appointment-visit-brief-v1-20261016
**Baseline main at activation:** 4cf219e8f3f254caeb88134577af397afc229410 — freshly re-check remote main.
**Original reviewed HEAD:** dbb55fa1bd5f85b312117da9b360e79a11a51e0b
**Original design blob:** 7cfcdeb58b730ecd7da8441187ac3f2762a522a4
**Frozen corrected design-bearing HEAD:** 199e37c28024391387546fb662350dbabc2b5c1b
**Frozen corrected design blob:** 2015dd1db0e2fd7c12a26664be456da10c77f6d7
**Exact corrected file:** cockpit/COCKPIT_VERIFIED_APPOINTMENT_LINK_VISIT_BRIEF_V1_DESIGN_2026-10-16.md
**Original request:** cockpit/reviews/COCKPIT_VERIFIED_APPOINTMENT_LINK_VISIT_BRIEF_V1_PRECODE_R2_REQUEST_2026-10-16.md

Reverify corrected design blob at the fresh branch HEAD. Later commits updating CURRENT/review request are checkpoints and must not be confused with the design-bearing HEAD.

**Original independent handback:** PRE-CODE R2 BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:2:0; P1-01 and P1-02 only. Original unaffected R1/R4/R5 PASS; reuse previous Visit Brief+Clinical Inbox R2 PASS, Unified Visit V3 identity/Peek R2 PASS and accepted Home #160 without reopening their reviews. Product Owner authorized this two-P1 design correction only. No implementation or release authority.

## Decisive question

Do the **two corrected design contracts** safely close their original design defects without asserting an unproven person/booking guarantee or backdating a prior clinical plan? A fail-closed design can be correct even if **runtime one-click access remains conditional on later source-qualification evidence**. Do not treat unqualified source semantics as automatically proven.

## Finite affected gates

**C1 — P1-01 attendee continuity.** Read corrected design §3.1-3.2 and current clinical_calendar.py ActualBooking / CockpitAppointment / live cockpit-context, and legacy linked_patient_id distinction. UID equality and same name/phone/fingerprint may NOT authorize protected history. Automatic same-UID reopen requires producer-backed unique UID→immutable-attendee semantics (non-reuse/cancellation/reassignment) OR trusted provider-issued opaque stable attendee identity/revision detecting a changed person with unchanged display fields. Verify fallback for unqualified continuity: retain auditable association in needs_reconfirmation; booking-only preview for every new open, no patient pending/history prefetch; independent clinician verification and direct confirmation action can return that opening's bounded brief but grants **no reusable session-wide token or later automatic GET access**. Reschedule, changed attendee, UID collision/reuse, same-name replacement, missing freshness and second UID must be truthfully fail-closed. No new provider writes or identity master. Identify any remaining **one-click release qualification** separately from a design defect.

**C2 — P1-02 previous visit temporal truth.** Read corrected design §4; clinical_data.py EncounterORM and signed Visit Capture Save, NextContact.when/assess and ClinicalPendingORM. Previous encounter must be supported completed/amended and provably available before **selected Reception booking start**, not browser time; earlier date or same-day with trustworthy signed/completed time strictly before target, excluding current/later/backfilled/amended-after-start content where prior version cannot be proven. Important: actual signed_at is serialized from **UTC-naive** utcnow().isoformat() and must NOT be read as Cyprus wall time. Check selected booking local date and Cyprus DST. For booked-date checks, next_contact.when or pending.review_date must **explicitly equal** selected Cyprus-local booking date; undated/different-date intents stay historical/as-recorded and cannot silently become “today.” Pending status is last-recorded-at-read, not reconstructed as-of old booking. Missing evidence remains visibly unknown.

**C3 — affected cumulative / bounded acceptance.** Inspect revised §5 for synthetic oracles covering qualified vs unqualified one-click fallback, unchanged visible attendee details with reassigned actual person, same-day signed before/after, late amendment, different next-contact/pending dates, 1280/680/390 UI stale-response safeguards. Reuse actual Calendar read, PatientORM, signed EncounterORM, ClinicalPendingORM, existing overlay/Home #160/split view and Visit Capture Save untouched. Do not reopen Clinical Inbox, Visit Capture, RF #159, Medical Reports #153 or previous Home reviews. No parallel booking/patient/AI task engine.

## Required answer and stop

Return PRE-CODE R2 **PASS/BLOCK/UNKNOWN**, COMPLETE_FOR_DECLARED_SCOPE/PARTIAL, P0:P1:P2, P1-01 CLOSED/OPEN, P1-02 CLOSED/OPEN and C1/C2/C3 PASS/BLOCK/UNKNOWN with narrow file/line evidence. Explicitly separate conditional provider continuity qualification and implementation/release HOLD. If BLOCK, prescribe only the smallest remaining affected design delta. ONE closure review and STOP — READ-ONLY, no code/merge/deploy/provider/PHI.
