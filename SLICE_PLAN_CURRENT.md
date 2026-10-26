# SLICE_PLAN_CURRENT.md — Priority 3 Office Cases

> **STATUS (2026-10-26):** Product Owner confirmed Priority 2 works, asked to continue to the next roadmap priority, and chose separate records for each accident payment. Priority 3 design `3405af09006d1616a1a264e5e0d8bb801745a753` passed independent pre-code R2 closure. Implementation at code/SQL/test `54d72e9d815fc0e2d25bd5038f4666e045443575` passed independent post-code R2 0:0:0; final PR #223 CI 19/19 SUCCESS. Reviewed admin migration is installed and metadata-verified in Aiven `osteoporosis`; temporary access IP was removed. PR #223 merged as `bd6133fef96a4e0f4a88e96907ec03ef5a16a263`; native Render `dep-dbfecim7bikc73c4dbj0` was observed building. Live activation, no-write smoke and real clinician acceptance remain pending. No Priority 4.
> **Previous active plan:** archived unchanged at `archive/SLICE_PLAN_PR1_HEIDI_TRANSCRIPT_2026-10-26.md`. PR-1's identifiable-transcript privacy/release HOLD remains unchanged.

## 1. Objective and acceptance

One Cockpit screen, **Περιστατικά ιατρείου**, offers Χειρουργεία, Τροχαία and RF tabs. Reuse the current surgery queue and native RF Tracking. Add an optional, protected Cyprus operating-room entry wall time to surgery rows; an operational accident register with independent payment events; and an explicit exact-RF-to-surgery action without duplicate entry or automatic RF stage change. Scheduled and performed remain distinct.

The clinician must be able to edit every provisional field and confirm each clinical/financial write. Real patient rows are excluded from tests, logs, source and agent release smoke. Priority 4 visit audio is deferred.

## 2. Owning design and seams

`cockpit/OFFICE_CASES_PRIORITY3_DESIGN_2026-10-26.md` at reviewed design commit `3405af0` is the detailed data, API, UI and release contract. The bounded implementation seams are `clinical_surgery_queue.py`, a new private accident API, `main.py` router registration, existing Cockpit library/surgery JS and HTML/CSS, one admin SQL migration, and synthetic focused tests. The RF source/tracking implementation stays its owner; change it only if a concrete integration defect requires a reviewed delta.

The accident report PDF generator is ephemeral and does not declare writing started or report sent. RF source contains no date of birth or phone. Clinician supplies both before a surgery record from RF can be confirmed. No inferred patient linkage from name or phone.

## 3. Safety contracts

- Existing Clinical Key protects all data and writes. Clear in-memory projections on auth loss; no browser storage, public PHI, raw payload logs or live row evidence.
- `clinical_surgery_details` stores local `HH:MM` only alongside a valid surgery date, exact nullable unique RF application ID, and a foreign key that restricts deleting a linked RF. Existing pending/completed/deleted surgery semantics stay unchanged.
- RF transfer re-reads exact application/revision and eligible stage (`approved` or `scheduled`) inside one transaction; one RF application can create at most one surgery, even after soft deletion. RF status is never auto-updated.
- Accident case facts/amount due use revision preconditions and append-only audit. Each separate positive payment has date/payer and a durable request UUID; identical replay returns the same event, conflicting replay is 409. Corrections void with reason and retain the original. Derived balance is unknown when amount due is unknown. No overpayment against a known due amount.
- Admin migration creates the new private tables/FKs/indexes and least-privilege grants **before** runtime deployment. The app role has no public schema CREATE. Startup `create_all` is not an upgrade strategy.

## 4. Finite evidence and release

Independent pre-code R2: initial P1 RF-delete and payment-replay findings, both closed at `3405af0`; final PASS 0:0:0. Implementation requires synthetic API and browser checks for surgery date/time, RF stage/revision/identity/unique/delete, accident revisions/partial payments/replay/void/overpayment, auth revocation and three-tab UI. Then one independent exact-code R2, applicable CI, reviewed admin SQL install/privilege checks, guarded PR merge, native Render auto-deploy and no-write smoke. Mark real clinical acceptance separately from technical release.

If SQL cannot be safely installed or its schema preconditions fail, stop before merging runtime that requires the new tables. Do not self-create clinical tables on startup or simulate production success from local tests.

## 5. Replan triggers and next action

A real RF source/schema mismatch, missing migration privilege, inability to guarantee atomic link/payment replay, or a material change in requested financial/clinical semantics reopens affected design R2. The reviewed implementation and migration are complete and PR #223 is merged. Next: fresh post-merge release gate, observe native Render deployment of the exact merge, then no-write smoke and clinician acceptance handoff with a checkpoint after each material transition. No Priority 4 work.
