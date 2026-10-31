# Follow-through one calm cycle — implementation checkpoint

Date: 2026-10-31, Cyprus. Issue: #266. Status: DRAFT / SYNTHETIC-ONLY / NO MERGE OR DEPLOY.

## What is actually implemented

One deterministic, backend-side, append-only **synthetic event semantics** in
clinical_followthrough_cycle.py. It explicitly models:
- A request from a signed visit (referencing the already-existing signed pending
  identifier) or a minimal provisional visit without a Cockpit patient chart.
- Separately arrived report, images and labs, with qualified source references,
  source version and deduplicated source receipt.
- Clinician attestation scoped to the actual received source version. One decision
  can both mark reviewed and request secretary notification; "no contact" completes
  immediately only when all required materials were reviewed.
- A minimal, PHI-free secretary instruction only after independent contact
  verification and explicit clinician assignment; actual notification versus
  attempted contact are separate events.
- True callback completion by clinician, never inferred from email Seen,
  clicking, sending, a call log or informing the patient.
- Revisions invalidating stale review and communication; conservative explicit
  resumption; source UID collision across two cases rejected.
- Role-separated doctor/secretary/internal-source actions, optimistic version
  checking, actor-scoped idempotency and append-only audit fields.

This is the **first executable domain slice**, not merely another specification.
It is not yet a persistence implementation, production endpoint, live inbox
integration, signed-visit write, patient identity authority, receptionist action
or UI. No clinical PHI, patient names, contact details or Apple credentials
exist in the fixtures. This explicit limitation avoids implying that a synthetic
PASS is clinical acceptance.

## Next bounded technical slices (not implemented yet)

1. Read-only dry-run of actual current Signed Pending and legacy minimal-visit
   identity owners; avoid competing task engines, use canonical pending ID when
   safely linked, otherwise one provisional visit obligation with a reconciliation
   key. Adopt durable reviewed schema and actor-scoped event audit with no startup
   DDL and no real patient data until permitted.
2. Within existing Cockpit Obligations + Visit Brief, one quiet card with one
   dominant next clinician action; separate narrow secretary task projection.
   Must handle no patient chart without fabricated EncounterORM.
3. Qualify incoming iCloud message identity; only exact verified referral/order
   evidence may link automatically. An uncertain result stays unmatched. The
   top ✉ transport repair remains separate DRAFT PR #265.
4. Full synthetic end-to-end source→visit→doctor→secretary→callback evidence,
   exact-head CI and independent security review. Only after product acceptance:
   guarded production DB migration/rollout with explicit approval, no remote
   Apple mailbox writes or auto-communication.

## Demonstrated via the new test suite

Complete signed MRI (report and images arriving separately), clinician review
and instruction, secretary attempt/notification, explicit doctor callback,
no-chart lab case completing in one doctor action, duplicate mail polls, stale
CAS/idempotency, new report version revocation, weak identity rejection and
cross-visit source collision, unauthorized secretary/source transitions,
concurrent duplicate ingestion, and no closure merely on receipt.

## UX acceptance target

Routine receipt: 0 clinician taps.
Review + secretary assignment: 1 clinician tap after inspecting result.
Secretary informed/attempted: 1 actual-outcome tap.
Doctor callback close: only if clinically outstanding.
No separate fake confirmation wizard, chart creation or encounter rewriting.
