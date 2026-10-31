> **2026-10-31 follow-through B2 — exact signed-Pending binding, DRAFT / STACKED / NO RELEASE:** New clinical_followthrough_signed_pending_synthetic.py verifies a real-schema ClinicalPendingORM against an existing PatientORM and the exact signed EncounterORM. It requires **pending_id membership in the signed Visit Capture pending_ids** and the matching signed candidate item type, consistent patient/encounter IDs, valid signature date, active Pending status and synthetic fixture marker. It rejects guessed patient names, unbound legacy Pending, cancelled/reassigned visits, mixed or malformed membership, and unqualified actors. Only then can it register **the same Pending reference** in the restart-safe synthetic durable store, and read an up-to-date projection from its own event history. Signed visit narrative and Pending rows are unchanged. An independent no-chart provisional visit remains supported. The resolver is hard-gated to SQLite + synthetic fixture identities; **it is not a production patient authority** and has NO authenticated real staff endpoints, real iCloud correlation, or provider access. A real follow-through event integration must later respect the existing signed-visit amendment lifecycle and actor/session policy. #267 and #268 independent R1 gates remain outstanding; #265 iCloud unread fix also remains separate. No DB migration, PHI, real staff dispatch, merge or deployment authority.

> **2026-10-31 follow-through durable B1 — DRAFT / STACKED / HOLD:** The second technical slice adds clinical_followthrough_durable_synthetic.py (PostgreSQL-compatible SQLAlchemy schema and exact-source event adapter, but **no production import**, no production DDL/migration, no ClinicalPending mutation and no real identifiers). The ledger persists synthetic visit/request/obligation identity, signed-Pending fixture reference or no-chart provisional visit, source/part claims, actor-attributed event revisions and idempotency across engine restart. Reuses the already-reviewed-by-CI PR #267 domain reducer, not a second competing lifecycle. The isolated SQLite test creates its own tables expressly; it exercises complete MRI and no-chart labs, restart/replay, confirmed notification vs true callback, corrected report revoking old review/secretary instruction, zero patient PHI, malformed event rejection, DB-level uniqueness and concurrent source-UID collision. Initial CI caught a real TOCTOU risk when one UID could be accepted by a second visit after a stale claim check; a second ownership check *after row CAS and before event insertion* now rejects it. **No real staff principal or patient matching is qualified**; this is preparatory offline storage, not yet a production-capable workflow. Source HEAD is frozen only after exact CI + independent code review. Base PR #267's independent R1 remains unsubmitted; no merge/deploy authority implied.

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

This is the **first executable and interactively testable synthetic cycle**,
not merely another specification. The new clinical_followthrough_demo.py is
mounted ONLY under the already protected /clinical/synthetic-161 test router;
a scoped synthetic-only one-card UI in followthrough-synthetic.js/.css shows
two cases and their real backend-driven event progression. The signed synthetic
Pending's visit/chart, Visit Brief and global obligations receive the same
versioned follow-through projection; the legacy Pending completion button is
suppressed for that test case to avoid contradictory actions. The signed visit
narrative itself remains untouched.

IMPORTANT: the synthetic ledger is in-process **memory-only**, and its demo
"doctor", "secretary" and "source" roles are server-side simulations behind
the pre-existing TEST Clinical Key, NOT production staff role grants. This is
NOT persistent across worker restart, a production endpoint, real inbox/mail
integration, GESY identity, independently qualified contact, actual secretary
action or medical chart write. No clinical PHI, real patient names, contact
details, Apple secrets or provider mail exist in the tests. Real data requires
a separate reviewed durable owner and authenticated per-role adapter.

## Next bounded technical slices (not implemented yet)

1. Read-only dry-run of actual current Signed Pending and legacy minimal-visit
   identity owners; avoid competing task engines, use canonical pending ID when
   safely linked, otherwise one provisional visit obligation with a reconciliation
   key. Adopt durable reviewed schema and actor-scoped event audit with no startup
   DDL and no real patient data until permitted.
2. Promote the synthetic one-card flow into a reviewed, protected production
   Cockpit Obligations/Visit Brief surface, with a real least-privilege
   secretary task projection and no double confirmation.
   Must handle no patient chart without fabricated EncounterORM.
3. Qualify incoming iCloud message identity; only exact verified referral/order
   evidence may link automatically. An uncertain result stays unmatched. The
   top ✉ transport repair remains separate DRAFT PR #265.
4. Extend full synthetic end-to-end source→visit→doctor→secretary→callback evidence,
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
