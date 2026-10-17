# Cockpit verified booking/patient link + Visit Brief v1 — one independent affected pre-code R2

**MODE:** fresh separate independent READ-ONLY; one bounded design review only. **No source/test implementation, merge, deploy or real patient/provider data.**

Repository `athpapachr-cmd/osteoporosis`. Design branch `design/cockpit-verified-appointment-visit-brief-v1-20261016`; activation main `4cf219e8f3f254caeb88134577af397afc229410` (fresh-check actual main and design head). Pin exact blob `cockpit/COCKPIT_VERIFIED_APPOINTMENT_LINK_VISIT_BRIEF_V1_DESIGN_2026-10-16.md` from reviewed HEAD.

**Product Owner decision:** prioritize daily clinical visits, patient link and substantive Visit Brief **ahead of** the separately held Medical Reports DRAFT #153. Home's #160 positioning and UX is accepted, released and frozen. Prior broader Visit Brief + Clinical Inbox pre-code R2 **PASS** (receipt `cockpit/reviews/VISIT_BRIEF_CLINICAL_INBOX_PRECODE_R2_PASS_RECEIPT_2026-10-08.md`), prior Unified Visit V3 identity/Peek R2 PASS, and Home #160 accepted review should be reused without re-reviewing them. PR #159 RF↔Reception Cal UID linkage is separate RF-only design.

**One decision:** Is the **bounded first implementation** source-grounded, safely authoritative about exact appointment ↔ protected patient and clinically honest about last signed visit, previous plan and status-as-recorded pending items, using existing sources without a parallel engine?

## Finite affected gates

- **R1 — source identity & reuse:** fresh main/design source, old R2 PASS reuse, actual source `clinical_calendar.py` / live `ActualBooking.uid` / nullable legacy `linked_patient_id`, `clinical_data.py` PatientORM/EncounterORM/ClinicalPendingORM and current `static/cockpit/clinical-workspace.js`. No new booking engine, no second registry.
- **R2 — persistent identity/authorization:** live provider UID scope and freshness, explicitly clinician-confirmed existing patient independent of name/phone; narrow association with version/audit/revoke, source change/conflict fail-closed, no false named doctor audit under shared Clinical Key, zero protected summary before confirmation, repeat same UID vs distinct future UID truthful. Confirm whether an existing calendar persisted linkage owner can be safely extended; sidecar only when justified.
- **R3 — brief evidence & pending:** minimal patient-specific server projection of latest *supported* signed/completed visit, explicit decisions/medications/next contact, ClinicalPending statuses with as-of qualification. No fabricated plan, no legacy visit_type called clinical summary, no status auto-close, no entire encounter payload/browser PHI leakage.
- **R4 — UI concurrency/security:** same overlay and accepted #160 Home, source-only until verified link, one protected check/read per open, patient search and confirmation, 401/403/503/logout/UID change/stale responses fail closed, open chart without changing Save, 1280/680/390 synthetic Chromium oracles.
- **R5 — finite implementation/release:** no external Reception/Cal/Gmail/Zadarma write, no RF/Medical Reports change, no auto AI or CareTask engine; bounded R2 implementation writer after review, synthetic tests and exactly one affected independent post-code R2; separate clinic retention/real-data and production release gates.

**Disposition:** PRE-CODE R2 **PASS / BLOCK / UNKNOWN**, COMPLETE_FOR_DECLARED_SCOPE / PARTIAL, P0:P1:P2, exact reviewed HEAD/design blob and source-backed smallest material findings (if any). Reuse all unaffected earlier PASS. **STOP — READ-ONLY.**
