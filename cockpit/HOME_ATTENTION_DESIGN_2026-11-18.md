# Day Home actionable attention — design candidate, 2026-11-18

Product Owner requests an actually useful Day Home, adopting Claude's ideas and correcting the issues identified in its prototype. Names on Home were explicitly requested, with a concealment option. This step keeps the existing clinician-only Home and links to existing owners; it does not create or modify patient, appointment, surgery, RF, task, call or provider data. Source base: `2d16222369c9ea0cfe94da34e24b95397d30931f`.

## Existing mechanisms and scope

Reuse the already loaded protected surgery queue and its auth-epoch lifecycle; extend its in-memory projection to include ordered source rows for the coming seven days and pending earlier dates, without calling past dates completed. Preserve existing grouped fields for other consumers. Reuse the protected Office Cases RF list with `stage=approved` and explicit `surgery_id`, and label unlinked RFs only when the source explicitly says null; treat incomplete pagination as partial. Reuse Office Cases accidents with its own pagination; distinguish source-proven writing started and not sent, and sent with a positive recorded balance. A visit without a sent date is not proof that a report was requested. Never invent urgency or clinical due dates.

Reuse the first 100 active ClinicalPending items; label that coverage, show dated obligations and never claim the whole set is empty. Reuse Reception peek's qualified unhandled count and coverage; a null count or any failure, including 503, means unknown/unavailable, never zero. Each row opens its existing relevant owner, and opens an exact item only if the existing owner provides a safe documented item-entry path. The Day agenda remains the existing protected source and appointment dialog.

## Privacy, lifecycle, coverage

Patient names appear as requested. A transient per-tab concealment control hides names and free-text descriptions that could contain identifying content; no patient content or privacy preference goes to browser storage or logs. Preserve textContent-only rendering. On logout, clear all identifiable projections, invalidate all in-flight requests and bootstrap responses, reject late events, and make no background protected calls while unauthorized. Refresh after explicit auth, return to visible tab or focus when stale; guard each response and the overall auth epoch. Avoid PHI in URLs or diagnostics.

Loading, unavailable, partial, and genuinely empty states are distinct for each source. An empty result is never a global guarantee when any source is incomplete or unknown. A bounded page of records is identified as such, with navigation to the existing complete owner. Sort date-valid records in Cyprus local calendar time; reject malformed dates and keep undated/invalid as explicitly unknown. Do not duplicate past surgery in future rows. Clinical follow-up timing, appointment classification, automatic clinical decisions and outward messages are out of scope.

## Evidence and release

R2: this changes patient data visibility/privacy and clinical workflow presentation. One independent pre-code source/contract review precedes implementation; one independent exact-code review follows synthetic focused tests, including unauthorized and delayed responses, source errors, partial coverage, no unsupported report inference, date bounds, redaction and real existing navigation. No live patient access or writes for validation. Merge/deploy and production no-PHI smoke follow applicable repository gates and the user's requested implementation authority; preserve existing settings and service identity.
