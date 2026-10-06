# Visit Brief / Lab Results Workflow — Product Design Checkpoint — 2026-10-06

> **STATUS:** PRODUCT OWNER DIRECTION CONFIRMED / DESIGN ONLY / IMPLEMENTATION FOLLOWS CALENDAR UNIFICATION.
> **Scope:** floating Visit Brief plus actionable laboratory-result workflow sourced from the clinician's Gmail, with clinician-approved patient communication through Zadarma.
> **Hard boundary:** incoming email/result is a Signal/Candidate, not an authoritative clinical fact or automatic patient identity match.

## 1. Product-owner intent

Immediately after Calendar Unification, build the Visit Brief / Lab Results workflow.

The clinician wants to:

1. see incoming laboratory-result emails in the Cockpit;
2. open/read the attached analyses from the dashboard;
3. review/understand the results;
4. confirm which protected patient they belong to when identity is not already strong;
5. send a clinician-approved message to the patient that results have arrived and that they may call during the established laboratory-result telephone windows;
6. use Zadarma as the messaging transport.

No secretary is currently active. The workflow must remain usable without the voice agent.

## 2. Visit Brief shape

The Visit Brief remains a floating/overlay view opened from the appointment/patient context rather than expanding full history on Home.

Ordering:

1. main problem/reason today;
2. relevant history;
3. prior visit decision/plan;
4. pending/awaited items;
5. what is expected/checking today;
6. communication only when materially relevant.

A new reviewed/unreviewed laboratory result can surface under pending/awaited items and What Changed when a strong/clinician-confirmed patient link exists.

## 3. Lab-results inbox

Cockpit should expose a small actionable Clinical Inbox area, not a general Gmail mirror.

A laboratory email item may contain:

- source provider/sender;
- received timestamp;
- source email/message ID;
- source-stated patient name / order number;
- attachment metadata;
- state: `unreviewed | reviewed | linked | communicated | dismissed`;
- provenance.

Do not copy the full Gmail inbox into Cockpit.

## 4. Gmail integration boundary

The deployed product cannot assume ChatGPT's Gmail connector is available inside the Cockpit runtime.

Implementation therefore requires a separately authorized Gmail read integration for the application, or another source-proven ingestion mechanism.

Preferred product behavior:

```text
Cockpit opens
→ server performs bounded Gmail read/search for approved lab senders / patterns
→ identifies candidate lab-result messages
→ fetches supported attachment only when needed
→ parses/extracts result candidates
→ clinician reviews
```

Avoid background correctness dependence on the free Cockpit service being continuously awake.

Start with an allow-list of known laboratory senders/patterns rather than unrestricted mailbox scanning.

## 5. Attachment/result handling

For supported laboratory PDF/email attachments:

- fetch on demand;
- do not persist raw attachment bytes by default;
- preserve source message ID, attachment identity and received time;
- extract structured **candidate** laboratory values with provenance;
- distinguish extraction from interpretation;
- never silently write extracted results to the protected clinical record.

If the product later offers `Accept into record`, that is a deliberate clinician action using the existing protected lab persistence owner.

## 6. Patient identity

Email subject/name is not sufficient to establish protected patient identity.

Allowed states:

- **strong already-linked identity** → attach to Visit Brief;
- **candidate match only** → show source-stated name and ask clinician to confirm;
- **ambiguous/no match** → keep as unlinked Clinical Inbox item.

Do not use fuzzy name matching as authority.

A phone number used for outbound messaging must come from an authorized/clinician-confirmed patient/contact context, not be inferred from the laboratory email.

## 7. Review state

Suggested interaction:

```text
Νέα αποτελέσματα
→ Προβολή PDF / αναλύσεων
→ extracted values / concise result summary
→ clinician marks Reviewed
→ choose action:
   [Καμία επικοινωνία]
   [Ενημέρωση ότι ήρθαν]
   [Να με καλέσει]
   [Χρειάζεται ραντεβού]
```

No automated clinical reassurance, urgency statement or treatment advice is sent merely from extracted values.

## 8. Laboratory callback windows

The Product Owner established non-booked callback windows.

Patients are **not assigned a specific appointment time**. The message tells them they may call on any working day during the laboratory-result windows.

Current intended pattern:

- weekdays: approximately **10:00–10:20** and **17:40–18:00**;
- the physical clinic alternates by day, but this is irrelevant for a phone callback;
- these are protected availability windows, not actual patient bookings.

The message should use configured callback-window text rather than hard-code each patient's slot.

## 9. Outbound message

Example default draft:

```text
Έχουν παραληφθεί τα αποτελέσματά σας.
Αν θέλετε να τα συζητήσετε με τον ιατρό, μπορείτε να επικοινωνήσετε
Δευτέρα–Παρασκευή 10:00–10:20 ή 17:40–18:00.
```

The exact wording remains clinician-editable before send.

Sending requires explicit clinician action.

## 10. Zadarma ownership

Per the Cockpit Product Constitution:

- Cockpit owns the clinical action/presentation;
- Reception/Digital Secretary owns communication orchestration;
- Zadarma is transport.

Current repository evidence shows Zadarma configuration variables exist in Reception, but a current production SMS send path has not yet been proven from source. Do not assume one exists merely because environment variables are present.

Before implementation:
- verify the current Zadarma provider/API contract;
- verify whether an existing working send mechanism can be REUSED/REBIND;
- if none exists, add one bounded protected server-to-server send action in Reception rather than implementing transport in Cockpit.

The current voice agent being disabled must not prevent clinician-initiated dashboard messaging.

## 10A. Provider feasibility verified on 2026-10-06

Official Zadarma documentation exposes `POST /v1/sms/send/` for server-side SMS sending and documents API-key/signature authorization plus sender/number/message parameters. Therefore a bounded Reception-owned transport adapter is technically feasible; production SenderID/account permissions still require release-time verification.

Official Gmail documentation confirms the Gmail API supports read-only mailbox access and message search/listing. The general `gmail.readonly` scope is classified as a **restricted** scope. If restricted-scope Gmail data is stored on or transmitted through the application server, Google may require additional verification/security assessment. This is a material implementation constraint.

Therefore the first Gmail design should minimize retained Gmail content aggressively:
- prefer on-demand retrieval;
- retain source IDs/status rather than full email/PDF where possible;
- avoid broad mailbox ingestion;
- assess whether a narrower architecture or user-authorized single-account/internal app posture changes the verification burden before implementation.

## 11. Message state

At minimum preserve:

- draft text;
- clinician approval timestamp;
- recipient number actually used;
- send attempt state;
- provider delivery/reference state where available;
- source clinical inbox/result item association.

Do not treat delivery as patient understanding or clinical closure.

## 12. Privacy

- Gmail credentials/tokens server-side only.
- Zadarma credentials server-side only.
- no secret in browser.
- no raw result attachment in logs.
- no full email body in routine logs.
- no patient-result association without strong/confirmed identity.
- no AI-generated lab value becomes authoritative without clinician review.
- no automatic outbound message.

## 13. First implementation slice

The first usable slice should be deliberately narrow:

1. one approved laboratory sender family (Medicover first);
2. read-only Gmail discovery;
3. one supported PDF/result attachment path;
4. Clinical Inbox item;
5. clinician-confirmed patient link;
6. view extracted candidate results;
7. mark Reviewed;
8. clinician-approved `Να με καλέσει` message;
9. send through the bounded Reception → Zadarma transport;
10. record communication status.

Do not start with broad multi-laboratory mailbox ingestion or automated clinical interpretation.

## 14. Review tier

**R2 design review required** before runtime implementation because this introduces:
- identifiable clinical information from Gmail;
- patient identity/linking;
- laboratory-result extraction;
- new external messaging side effect;
- cross-service authority between Cockpit, Reception and Zadarma.

The R2 should explicitly review identity, privacy/retention, Gmail provider boundary, lab extraction candidate semantics, outbound-message authorization and failure behavior.
