# Cockpit Product Constitution — clinician attention and visit intelligence

> **ROLE:** durable Cockpit product boundary under root `AGENTS.md` and `CLINICAL_EXCELLENCE_PLAN.md` §§33–34.
> **STATE:** product direction; implementation/release truth stays in `cockpit/CURRENT.md` and the applicable workstream `CURRENT.md`.

## Binding current-scope contract — 2026-11-01 Product Owner correction

**ONE real Cockpit operator: the clinician.** Continue using the **existing Cockpit Clinical Key / established protected session only**. No new doctor/secretary account, personal credential, role enrollment, principal registry, alternate login or second authentication system. PR **#270 is explicitly REJECTED FOR CURRENT ROLLOUT / DO NOT MERGE OR DEPLOY**; it is not a prerequisite for iCloud, Clinical Inbox or Follow-through. Multi-user identity may be reconsidered **only after a future, separate and explicit Product Owner commercialization decision**, not by engineering assumption. Keep the existing authentication and privacy checks intact.

**Already working foundations:** the existing, protected, read-only **iCloud IMAP source and full inbox are already integrated and deployed**; do not build a second email connector or wait for another mail integration to do the current work. The clinical source of truth is the **existing ClinicalPendingORM** tied to the real signed visit, with its existing Pending checkpoint/lifecycle history wherever installed; use the established patient/encounter/appointment, Clinical Key and UI. **No new PostgreSQL database, authentication, clinical task/event store or activation of the two proposed `clinical_icloud_message_*` tables.** The Product Owner approved one narrow design exception on 2026-11-01: ONE `icloud_triage_preferences` table in the existing database, solely per-message sorting preferences, explicit personal rules and bounded operation receipts for replay/undo. It contains no patient/visit/Pending linkage, clinical acceptance/review/contact state or message body/attachments. Reviewed table-specific provisioning/minimum DML privileges are part of the design; execution is not authorized by this approval. Never grant the application schema-CREATE privilege or create tables during startup/GET. [Exact design](ICLOUD_TRIAGE_PREFERENCES_DESIGN_2026-11-01.md) requires independent pre-code R2 before separately authorized implementation, with merge/deploy/provisioning separately gated. This exception supersedes earlier blanket no-new-table wording only for this operational preference store. All other missing capabilities still require explicit product-level replan; do not silently manufacture another table or grant CREATE.

**One task, not two:** iCloud messages are evidence/transport, not a second clinical obligation. Existing signed Pending remains the only authoritative clinical follow-through; no duplicate obligation, second status ledger or automatic closure. Unmatched/admin mail remains in the existing mailbox view; no fabricated Patient or Encounter. Review, received report/images, patient contact attempt, confirmed communication and actual closure are distinct facts. The clinician explicitly validates clinical identity and reviews documents before their associated clinical actions. Never send messages or alter mailbox Seen as a side effect.

**Existing PR reconciliation before release:** #267 is a synthetic event/UI example; #268 creates a separate *synthetic* durable event owner; #269 adds valuable signed-Pending/source-invalidation tests on top of both. None is a real clinical deployment-ready chain, and the frozen #267/#268 parent snapshots have defects corrected only downstream in #269. **Do not merge/install them sequentially as-is**; evaluate and reuse only the non-duplicate domain rules/tests against current `main` and existing Pending/checkpoints. #271/#272 are already merged (iCloud reader/UI plus diagnostic), while new email decision writes in #271 are not live-ready (observed SQLSTATE 42501 / optional action-store 503). #273/#274 are unmerged DRAFT safety candidates, **not approval for migrations or two new tables**. No new user-auth or PostgreSQL workstream should be proposed merely to unblock them.

**Release rule:** No merge/deployment of follow-through code before a specific evidence-based reconciliation against actual `main`, no new auth, no schema beyond the explicitly approved and separately provisioned one-table triage exception, no redundant email connection, correct fail-closed behavior, and focused review of any real clinical write. An upcoming commercial product may use a different contract only after the Product Owner explicitly changes this boundary. This active decision supersedes historical design candidates or PR descriptions that request staff-user V2, extra PostgreSQL storage or reintegration of iCloud.

## Purpose and first screen

Cockpit helps the clinician see what matters, prepare the encounter and take the next **clinical** action. Its three concerns are clinician attention, clinical action and visit intelligence. Osteoporosis is Module 01, not the whole Cockpit. The existing Reception/calls system remains separately owned.

Home has two first-class entrances: appointment-driven floating **Visit Brief** and an independent, always-current **Clinical Inbox** for clinical inputs from explicitly approved mailbox accounts (Hotmail/Outlook for GESY notifications and iCloud Mail for patient-supplied results; Gmail is not the clinical-results source). New labs, MRI/X-ray/imaging reports and GESY notifications can be reviewed and lead to clinician-approved patient communication on the day they arrive, independently of appointment timing. Home should normally show only about **3–4 high-attention incoming items** and provide a direct path to the full unresolved queue. Learning, audit, improvement and utilities remain accessible after immediate clinical needs.

## Read projections and boundaries

| Cockpit view | Meaning | Boundary |
|---|---|---|
| **Previous / Current / Next** | Brief attendance context around the visit | Read projection, not a full booking calendar or scheduling writer. |
| **Relevant communication** | Clinically useful clinic → patient and patient → clinic communication around the current/next visit or an unresolved action | Bounded read summary with source/time/status, not an SMS/call inbox or communication orchestration console. |
| **Visit Brief** | Appointment-driven floating preparation view; reuses the linked patient's clinical-attention history | Provenance-preserving read view; unknown, negative, absent and conflict remain distinct; unresolved identity permits appointment context only. |
| **What Changed** | Clinically meaningful changes since the last confirmed point | Show source and comparison basis; do not infer a clinical delta from mere message arrival. |
| **Clinical Inbox** | Independent daily source-review/action surface, including patients with no appointment | Signal/Candidate is not a confirmed clinical fact or automatic task; review, authoritative acceptance and communication are separate states. |

External patient linkage must be verified through an authorized strong mapping before patient-specific content is attached. Name, weak demographic or fuzzy matching may discover **candidates only**. No weak clue, unreviewed message or external Signal may silently write the protected patient record, close an Awaited Item, assert a result or trigger treatment. Clinical confirmation belongs to the clinician and the protected clinical owner.

### D2 — Relevant Communication Context

The read projection is **bidirectional**. Clinic → patient examples include an SMS reminder to complete blood tests before Prolia, a request to bring DXA/imaging/documents, or a clinician-approved result/follow-up message. Patient → clinic examples include confirmation that tests were done, relevant pre-visit information, or a reply to a clinic message. Show only communication relevant to the current/next visit or an unresolved clinically useful action, with direction, source, time, delivery/reply state and any remaining communication obligation. For example, `28/9 · Στείλαμε SMS · εξετάσεις πριν από Prolia · παραδόθηκε` followed by `30/9 · Απάντηση ασθενούς · επιβεβαίωσε ότι έκανε τις εξετάσεις` may end in `καμία εκκρεμής επικοινωνία`. A reply is not proof of a clinical result or authoritative patient fact.

The normalized appointment telephone number may be used as an **operational correlation key** among appointment/contact, Digital Secretary communication and Zadarma message records. Normalize consistently, preferably to international/E.164 form when supported by the source. An exact phone match may establish communication ↔ appointment/contact correlation; it **does not** establish authoritative protected clinical patient identity, and phone is not a primary patient ID. Missing, conflicting, known shared/ambiguous or otherwise insufficient phone evidence fails closed for patient-specific D2 attachment until stronger or human resolution exists. Do not use fuzzy name matching to compensate for a missing phone match. The stronger protected clinical identity boundary above remains in force.

Capture once at the rightful source and reuse everywhere that the contract permits. The clinician should not re-enter a received fact just to make Visit Brief or Home useful. An external communication or an operational task may point to a possible clinical action; a clinical CareTask requires its own deliberate clinical authority.

## Shared clinical-attention boundary

Visit Brief and Clinical Inbox are views/actions over one clinical-attention lifecycle, not duplicate patient histories or work queues. Labs yield structured candidate values; MRI/imaging yield report/document summaries; GESY notifications yield notification/action context. Linking, marking Reviewed or communicating does not silently promote data or close an awaited item/CareTask. A future deliberate acceptance reuses the protected clinical owner.

Always-on intake belongs to Reception-owned orchestration with separately scoped mailbox-account adapters (or a separately reviewed small intake service); the free Cockpit is the clinical consumer and may cold-start. Intake correctness must not depend on opening Cockpit. A durable delivery outbox handles sleep/recovery; its transport state is distinct from the Clinical Excellence review state. Show incomplete/stale/unavailable coverage truthfully.

Cockpit may present clinician-approved communication actions; Reception owns admission, orchestration, action/idempotency/outcome state, and Zadarma owns transport. A disabled voice agent must not block dashboard actions. Recipient/text/item approval is explicit; provider acceptance, delivery, patient understanding and clinical closure stay distinct. Callback windows remain configured working weekdays 10:00–10:20 and 17:40–18:00 (Asia/Nicosia), without a per-patient booked slot. A standalone full D2 communication inbox remains deferred.

The bounded design and integration gates are at `VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`; provider-neutral Phase A account scoping/retention is at `CLINICAL_INBOX_PHASE_A_MULTI_ACCOUNT_CONTRACT_2026-10-23.md`, and design status/next action belongs to `CURRENT.md`. This direction does not establish provider access or runtime authority.

## Existing system ownership

| Responsibility | Owner / Cockpit role |
|---|---|
| Appointment source and ordinary appointment reminders | Existing booking provider owns appointments (the current Calendar reason bridge consumes Cal.com); Setmore owns ordinary reminders where used. Cockpit reads bounded attendance projections. |
| Reception communication workflow and orchestration | Digital Secretary/Reception; Cockpit reads relevant summaries and presents deliberate clinician-approved actions through a bounded contract. Reception owns transport action lifecycle. |
| Messaging transport | Zadarma; Cockpit does not create an SMS transport. |
| Protected clinical facts, review and CareTasks | Clinical Excellence owner and clinician; Cockpit presents authorized clinical views/actions. |
| Longitudinal clinical summary | Existing G3/validated clinical projection; Visit Intelligence consumes rather than duplicates it. |

No duplicate task, calendar or messaging engine belongs in Cockpit. New integrations must pass the existing-mechanism gate in `PROCEDURES.md`, preserve source ownership, and use a separately bounded contract for identity, data minimization, state transitions and any external side effect. This constitution does not activate a UI, database writer, provider integration or deployment by itself.
