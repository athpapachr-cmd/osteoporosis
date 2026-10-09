# Cockpit Product Constitution — clinician attention and visit intelligence

> **ROLE:** durable Cockpit product boundary under root `AGENTS.md` and `CLINICAL_EXCELLENCE_PLAN.md` §§33–34.
> **STATE:** product direction; implementation/release truth stays in `cockpit/CURRENT.md` and the applicable workstream `CURRENT.md`.

## Purpose and first screen

Cockpit helps the clinician see what matters, prepare the encounter and take the next **clinical** action. Its three concerns are clinician attention, clinical action and visit intelligence. Osteoporosis is Module 01, not the whole Cockpit. The existing Reception/calls system remains separately owned.

Home has two first-class entrances: appointment-driven floating **Visit Brief** and an independent, always-current **Clinical Inbox** for Gmail clinical inputs. New labs, MRI/X-ray/imaging reports and GESY notifications can be reviewed and lead to clinician-approved patient communication on the day they arrive, independently of appointment timing. Home should normally show only about **3–4 high-attention incoming items** and provide a direct path to the full unresolved queue. Learning, audit, improvement and utilities remain accessible after immediate clinical needs.

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

Always-on intake belongs to an approved Reception-owned adapter (or a separately reviewed small intake service); the free Cockpit is the clinical consumer and may cold-start. Intake correctness must not depend on opening Cockpit. A durable delivery outbox handles sleep/recovery; its transport state is distinct from the Clinical Excellence review state. Show incomplete/stale/unavailable coverage truthfully.

Cockpit may present clinician-approved communication actions; Reception owns admission, orchestration, action/idempotency/outcome state, and Zadarma owns transport. A disabled voice agent must not block dashboard actions. Recipient/text/item approval is explicit; provider acceptance, delivery, patient understanding and clinical closure stay distinct. Callback windows remain configured working weekdays 10:00–10:20 and 17:40–18:00 (Asia/Nicosia), without a per-patient booked slot. A standalone full D2 communication inbox remains deferred.

The bounded design and integration gates are at `VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`; design status/next action belongs to `CURRENT.md`. This direction does not establish provider access or runtime authority.

## Existing system ownership

| Responsibility | Owner / Cockpit role |
|---|---|
| Appointment source and ordinary appointment reminders | Existing booking provider owns appointments (the current Calendar reason bridge consumes Cal.com); Setmore owns ordinary reminders where used. Cockpit reads bounded attendance projections. |
| Reception communication workflow and orchestration | Digital Secretary/Reception; Cockpit reads relevant summaries and presents deliberate clinician-approved actions through a bounded contract. Reception owns transport action lifecycle. |
| Messaging transport | Zadarma; Cockpit does not create an SMS transport. |
| Protected clinical facts, review and CareTasks | Clinical Excellence owner and clinician; Cockpit presents authorized clinical views/actions. |
| Longitudinal clinical summary | Existing G3/validated clinical projection; Visit Intelligence consumes rather than duplicates it. |

No duplicate task, calendar or messaging engine belongs in Cockpit. New integrations must pass the existing-mechanism gate in `PROCEDURES.md`, preserve source ownership, and use a separately bounded contract for identity, data minimization, state transitions and any external side effect. This constitution does not activate a UI, database writer, provider integration or deployment by itself.
