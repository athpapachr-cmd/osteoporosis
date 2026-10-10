# Cockpit Unified Visit V3 — R2 P1-01 coverage correction delta

Date: 2026-10-10 Asia/Nicosia. **MODE: design-only bounded correction for received independent R2 PRE-CODE BLOCK.**
Original reviewed frozen design: `cockpit/UNIFIED_VISIT_V3_INTEGRATION_CONTRACT_2026-10-10.md`, exact immutable Git blob `9e107112020e2d98d542867c64fbb08d47e278dd`; original review checkpoint `36349d826d9d95497b306bc592a71b5995a25e51`. This delta **supplements/supersedes only** the R1 calendar coverage condition. Original contract bytes and original R2 request stay untouched. This addendum is a **candidate pending one independent P1-01 closure**, not a PASS or implementation permission.

## P1-01 — exact necessary correction of read contract

The existing protected `GET /clinical/calendar/cockpit-context` must validate **both source freshness and known complete coverage of today's Cyprus local calendar date** before it can return success for the first three `upcoming_today[]` appointments or a trustworthy empty today.

Use the **same already fetched** `ActualSchedule` object from the Reception-owned schedule endpoint, not a second provider fetch, new calendar, new store, or booking writer.

1. Compute `today_local = now.astimezone(ZoneInfo("Asia/Nicosia")).date()` from the existing timezone-aware request clock. Honor local-day boundaries across Cyprus daylight-saving transitions; do not use UTC's date.
2. After `ActualSchedule.model_validate(...)` and the already required `fetched_at` maximum-five-minute freshness check, enforce **inclusive date-range coverage** using the existing source fields:
   `source.coverage_start is not None AND source.coverage_end is not None AND source.coverage_start <= today_local <= source.coverage_end`.
   A missing, inverted, future-start or past-end date range fails this requirement. Neither successful HTTP 200 nor a recent `fetched_at` establishes coverage.
3. For unknown/incomplete coverage, fail closed via the **existing protected 503 unavailable response** (`status: unavailable` and last successful fetch time only if already known), before computing `today_total`, `previous/current/next`, or new `upcoming_today[]`. Do not advance the last-successful snapshot marker for rejected coverage. Client must show unavailable/stale, **not** `Δεν υπάρχουν άλλες διαθέσιμες σημερινές επισκέψεις` or an apparently authoritative count of zero.
4. Once both freshness and today's coverage are validated, reuse the original minimal verified appointments to produce `upcoming_today[]` (up to 3 start times strictly after now and before the next **Asia/Nicosia** local midnight); if none qualify, `upcoming_today=[]` is an allowed **true zero**. Preserve the existing separate cross-day `next` semantics and unchanged overlap/current/previous rules.
5. Preserve the original R2 design's patient identity rules and the separately PASS R2/R3/R4 review findings. No inferred clinical patient ID, new provider contract, external write or new full patient payload. Avoid unrelated refinements during P1-01 closure.

### Finite focused evidence obligations after future implementation authorization

- **Valid empty:** fresh source, `coverage_start <= today_local <= coverage_end`, `appointments=[]` → protected success and `upcoming_today=[]` (and `today_total=0`). Include a Cyprus-boundary example if needed to establish timezone semantics.
- **Coverage missing/incomplete:** fresh source, `appointments=[]`, missing bounds or dates not encompassing `today_local` → existing 503 unavailable; **not** true empty; last-successful marker unchanged.
- **Unchanged source:** show the existing `/appointments` coverage invariant as reference and confirm no additional Reception fetch/booking mutation, no regression in cross-day `next`.

Only the R1 pre-code correction needs independent closure. Once it independently passes, the original two read additions may proceed through the previously planned bounded code author + focused synthetic tests + one affected post-code R2; **no production release or real patient processing follows automatically**.

## Preserved closed decisions

Original R2 recent-encounters: `EncounterORM.status IN ('completed','amended')`, max 3 minimal metadata rows, ordered by `encounter_date/created_at`; safely generic fallback for old records without `visit_type`; never return `payload_json`, raw text, DOB, phone or unrelated fields.

Original R3: a Cal name/appointment cannot auto-resolve `patient_id`; explicit protected clinician selection or already stored encounter link only. R4: existing Visit Capture Save, Calendar writer, Module 01 and parent PR #140 A4 remain separate; real Dia/Heidi/GESY, privacy, retention and patient/provider activation remain OPEN.

**Stop after focused R2 P1-01 closure request. No actual protected read implementation from this design correction.**
