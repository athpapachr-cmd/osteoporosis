# Unified Visit V3 — independent exact-head post-code R2 (two protected reads)

Date: 2026-10-10. **Fresh separate independent READ-ONLY; ONE finite affected implementation-fidelity review; STOP.** No implementation, merge/deploy, real clinical data/provider calls.

Repository: `athpapachr-cmd/osteoporosis`; [draft PR #142](https://github.com/athpapachr-cmd/osteoporosis/pull/142), stacked on draft PR #140 (separate affected A4 HOLD). At implementation activation production main `1febcfa27096df20f6f4bdeb3ccee15158634fc3`.

**Exact tested implementation + test head:** `46b8f3abf6a4a89dd8b0ffd1e7d31b0ac4f44558`. Later docs-only checkpoint commits must preserve these exact blobs:
- `clinical_calendar.py` blob `8a6a64404a08e9a6cc8c9210cd2d5f028a165386`
- `clinical_data.py` blob `713ce588c2d7061f2f2c35f2a851a2a76b5965a1`
- `test_clinical_calendar_snapshot.py` blob `b715546aef68c91d83a675a25dec7e1c5a2ad134`
- `test_visit_capture.py` blob `b5d1af910946a0387588ff33d6269e5e31b7f2a6`

**Pre-code chain:** original protected-read design `cockpit/UNIFIED_VISIT_V3_INTEGRATION_CONTRACT_2026-10-10.md` blob `9e107112020e2d98d542867c64fbb08d47e278dd`; P1-01 correction `cockpit/UNIFIED_VISIT_V3_READ_COVERAGE_P1_01_DELTA_2026-10-10.md` blob `8a05d13aff22fc54bfa602ffae6ee03db685ce6e`; independent focused P1-01 C1–C3 pre-code closure PASS/COMPLETE/0:0:0 recorded in `reviews/UNIFIED_VISIT_V3_READ_PRECODE_P1_01_CLOSURE_PASS_RECEIPT_2026-10-10.md`. Prior R2–R4 PASS inherited. Stage A R1 earlier PASS and two nonblocking UI corrections are outside this read review.

**Relevant CI on exact tested head:** Clinical Calendar snapshot `38050120250` SUCCESS; Visit Capture `38050120266` SUCCESS; Cockpit Home `38050120253` SUCCESS; canonical impact `38050120258` SUCCESS; Baseline finalization `38050120307` SUCCESS; Learning L1 `38050120257`, L1B `38050120315`, L1C `38050120252` SUCCESS. Legacy L0 design-only scope check failed on implementation PR; do not claim all workflows green or rerun unrelated tests without a material question.

## Finite review questions (R1–R4)

R1 **Snapshot coverage and projection:** New `CockpitContext.upcoming_today` contains at most three minimal `CockpitAppointment` rows from the **same existing** validated Reception source, only `now < start_at < Cyprus local midnight`, sorted. Before any successful today/empty projection enforce known inclusive `ActualSchedule.coverage_start <= Asia/Nicosia today <= coverage_end` plus five-minute freshness; incomplete source returns existing 503 without moving last success. Valid complete empty is true `[]`. Existing cross-day `next`, overlap/Aclasta and previous/current semantics unaffected, no new calendar writer/call.

R2 **Recent encounters:** Protected `GET /clinical/recent-encounters?limit=3` uses existing `PatientORM/EncounterORM`, returns only completed/amended, date/created_at descending with bounded limit, minimal stored patient name/link/date/type; generic fallback for older records. No draft, clinical payload, phone, DOB, raw Dia or broad registry listing, no new store/writes.

R3 **Identity and UI usage:** Existing Home 3+3 consumptions match returned schemas. Calendar name/phone never grants protected patient identity; recent encounters' patient_id links to existing protected patient, not a newly inferred ID. No implicit Save or new side effects; authenticated routes reject unauthenticated calls.

R4 **Exact affected evidence / scope:** Assess four code/test blobs and changed seam, relevant CI; no reopen of passed pre-code R2–R4, prior Visit Capture/Calendar/Module 01 reviews or parent PR #140 A4. Verify synthetic fixtures actually test covered empty vs missing/incomplete coverage, three future rows, cross-day next, completed/amended vs drafts and response field minimization. Distinguish source PASS from production qualification.

Return **IMPLEMENTATION-FIDELITY PASS | BLOCK | UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE | PARTIAL**, R1–R4 dispositions, P0:P1:P2 findings, minimal correction if needed, exact source/CI identities and **STOP**. Review PASS is not merge/deploy authority or approval for real identifiable Dia/Heidi/GESY, Gmail or Zadarma.
