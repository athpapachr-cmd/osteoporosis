# Visit Brief / Lab Results — bounded independent R2 pre-code review request

> **STATUS:** REQUEST ONLY / read-only review / no runtime implementation authority.
> **DATE:** 2026-10-06 Asia/Nicosia.
> **BRANCH:** `docs/cockpit-calendar-unification-visit-brief-2026-10-06`
> **DESIGN:** `cockpit/VISIT_BRIEF_LAB_RESULTS_WORKFLOW_2026-10-06.md`
> **DESIGN BLOB:** `08346fb9d9c479e6f39ffb59068eaa9813703ff5`
> **SEQUENCING:** implementation starts only after Calendar Unification pre-code closure and implementation/release decision.

## Product goal

Build the first actionable Visit Brief / Lab Results slice:

- discover/read laboratory-result emails from the clinician's Gmail;
- open supported lab result attachments in the Cockpit;
- extract candidate laboratory values with provenance;
- require strong or clinician-confirmed patient identity before attaching to Visit Brief;
- allow clinician to mark the result reviewed;
- allow clinician-approved patient notification/callback message;
- send through Reception-owned Zadarma transport;
- preserve communication state.

No automatic patient matching, no automatic clinical write and no automatic message send.

## Known source evidence

- Official provider feasibility already checked on 2026-10-06: Zadarma exposes server-side `POST /v1/sms/send/`; Gmail API supports read-only mailbox access, while `gmail.readonly` is a restricted scope and server-side storage/transmission can trigger Google verification/security-assessment requirements. Review the architecture against that constraint rather than assuming OAuth is trivial.

- Cockpit Product Constitution already assigns:
  - clinical action/presentation to Cockpit/Clinical Excellence;
  - communication orchestration to Digital Secretary/Reception;
  - transport to Zadarma.
- ChatGPT's connected Gmail tools can search/read messages and supported attachments, but that connector is **not** proof that the deployed Cockpit runtime has Gmail access. The product needs its own authorized Gmail integration or another source-proven ingestion mechanism.
- Reception current configuration contains Zadarma-related environment variables, but current repository search has not yet proven a production SMS-send implementation. Do not infer a working send path from configuration alone.
- The voice agent is currently disabled; clinician-initiated dashboard messaging must not depend on it.

## Frozen product rules

### Incoming result
An email/result is a Signal/Candidate, not authoritative patient truth.

### Identity
Name in email/subject is insufficient for protected clinical identity.
Strong existing link or clinician confirmation is required before Visit Brief attachment or clinical action.

### Lab extraction
Extracted values are proposed candidates with source provenance.
No silent write to the protected lab record.

### Outbound communication
Clinician chooses and approves the message.
No AI/autonomous send based only on extracted lab values.

### Callback windows
No per-patient callback booking.
Default message may tell the patient they may call on any weekday at:
- 10:00–10:20
- 17:40–18:00

### Visit Brief
Floating/overlay brief; communication remains supporting context, not a general inbox.

## Finite questions

### Q1 — Gmail integration / data minimization

Assess the smallest safe deployed-app integration.

Check:
- whether on-demand Gmail read when Cockpit opens is preferable to background polling given free Render cold starts;
- allow-listed senders/patterns;
- attachment fetch-on-demand;
- raw email/PDF persistence vs ephemeral handling;
- token/credential storage boundary;
- log minimization;
- what minimal source identifiers/state must persist for review continuity.

Do not assume ChatGPT connector runtime is embeddable in the deployed app.

### Q2 — identity / Visit Brief attachment

Check:
- unlinked inbox item state;
- candidate match vs strong/clinician-confirmed link;
- whether existing protected patient registry can supply the confirmed phone;
- no fuzzy name authority;
- no appointment-phone correlation becoming clinical identity;
- how a reviewed result becomes visible in Visit Brief without silently creating an authoritative lab fact.

### Q3 — lab extraction semantics

Check:
- supported attachment types and parser boundary;
- extraction candidate vs interpretation;
- provenance per extracted value;
- missing/ambiguous values;
- clinician review;
- whether reuse of the existing protected lab persistence owner is sufficient for a later explicit Accept action;
- no automatic diagnosis/treatment message.

### Q4 — Zadarma / messaging owner

Inspect current Reception source and current provider configuration.

Determine:
- whether a working send mechanism already exists and can be REUSED/REBIND;
- if not, the smallest protected Reception server-to-server send endpoint needed;
- explicit clinician approval;
- recipient confirmation;
- failure/retry/idempotency behavior;
- transport/provider reference/delivery status where available;
- communication record ownership;
- voice-agent independence.

If the provider API contract is needed to decide feasibility, identify the exact provider facts that require verification rather than guessing.

### Q5 — retention / privacy / failure

Check:
- raw email/PDF retention;
- extracted candidate retention;
- message content/recipient retention;
- secrets/browser exposure;
- source unavailable behavior;
- Gmail or Zadarma partial failure;
- duplicate email ingestion;
- duplicate SMS send prevention;
- auditability without storing unnecessary PHI.

## Evidence allow-list

Use the minimum needed from:

- `cockpit/VISIT_BRIEF_LAB_RESULTS_WORKFLOW_2026-10-06.md`
- `cockpit/PRODUCT_CONSTITUTION.md`
- `cockpit/CURRENT.md`
- current protected patient/lab persistence contracts in Osteoporosis
- current Reception communication/action-queue code only where directly relevant
- current Reception `render.yaml` Zadarma configuration surface
- current Gmail/provider integration evidence available to the reviewer
- applicable `AGENTS.md` / `PROCEDURES.md`

Do not broaden into voice-agent redesign, booking architecture, general email client, full D2 communication inbox, or automatic clinical decision support.

## Stop rule

Return one of:
- PASS / COMPLETE_FOR_DECLARED_SCOPE
- BLOCK with concrete P0/P1/P2 findings
- UNKNOWN only where a material design decision cannot be disposed from allowed evidence

Once Q1–Q5 are disposed, STOP.

## Requested handback

- exact heads/sources inspected;
- design blob verified;
- P0:P1:P2 counts;
- Q1–Q5 disposition;
- bounded design correction if required;
- explicit statement whether first implementation slice may start after Calendar Unification.

No merge/deploy/patient-data authority is inferred.
