# Visit Brief + Clinical Inbox — two-entry design v2

> **DATE:** 2026-10-07 Asia/Nicosia.
> **STATUS:** PRODUCT DIRECTION CONFIRMED / DESIGN CANDIDATE / INDEPENDENT R2 PRE-CODE PENDING / NO RUNTIME AUTHORITY.
> **BASE:** Osteoporosis `045798dfa28612b268f16a99262c6ecc9ca4829d`.
> **SUPERSEDES:** the 2026-10-06 Visit Brief / Lab Results design and its unused review request. Exact originals are archived under `archive/checkpoints/2026-10-07-pre-clinical-inbox-replan/`.
> **OWNERS:** this is the bounded parallel Cockpit design, subordinate to the root canonicals. `SLICE_PLAN_CURRENT.md` continues to own PR-1; `CURRENT_OPERATIONAL.md` is the only repo-wide writer lock.

## 1. Confirmed product meaning

Calendar Unification is RELEASED / PRODUCT OWNER SMOKE VERIFIED. The Product Owner corrected the next feature: newly received clinical material must be reviewed and, when the clinician decides, communicated **on the day it arrives**, even when the patient has no appointment today or no future appointment.

Two independent entrances share one clinical-attention lifecycle:

```text
Appointment → floating Visit Brief
Gmail clinical input → independent Clinical Inbox

both → verified patient link → source review → clinician disposition
     → pending clinical action / approved communication → longitudinal context
```

The Inbox is a first-class daily work surface. It is not a subsection that waits for Visit Brief, an appointment filter, or a general Gmail client. Same-day review is a workflow objective, dependent on clinician availability; intake cannot itself promise clinical review or patient contact. Received time, first visible/intake time, clinician-review time and communication outcome stay distinct. The system cannot know when an external laboratory released a result unless that date is source-stated.

**Confirmed step example (synthetic):** an MRI report arrives at 09:17 for a patient whose appointment is three weeks away. It appears in the Clinical Inbox now. The clinician opens the report, confirms the patient, records review and chooses communication now. The later Visit Brief reads the same event: report received → reviewed → communication attempted/confirmed → any still-open follow-up. The patient's appointment is never a prerequisite.

The supplied direction authorizes this design/checkpoint update and preparation of the R2 request only. Runtime, OAuth activation, identifiable-data processing, patient messages, booking, deployment and P0-V0 release remain outside this task.

## 2. Appointment entrance — floating Visit Brief

Home keeps compact **Προηγούμενο / Τώρα / Επόμενο**. Clicking the appointment/name opens one overlay without navigating away. Show:

1. why the patient comes today;
2. relevant confirmed clinical history;
3. the previous explicit decision/plan;
4. pending/awaited items;
5. what is expected or needs checking today;
6. new reports/results and communication only when relevant to this visit.

Provide **Άνοιγμα πλήρους φακέλου** when a confirmed clinical link exists. Reuse released G3 longitudinal-summary mechanics and protected clinical reads; do not build another longitudinal truth store. Source/time/review status and unavailable/missing/conflict states remain visible. Current drafts stay separate from completed/amended history; scheduled attendance is never an actual administration.

The live schedule currently supplies appointment identity/context, **not a verified clinical patient link** (`clinical_calendar.py` returns empty phone/null link for its weekly live rows). For an unlinked appointment, the overlay shows appointment context and an explicit patient-selection/confirmation step. It cannot retrieve a candidate patient's history or perform clinical actions merely from the Cal display name.

Closing the overlay, changing appointment/patient, logout and navigation discard transient displayed clinical content. A late response for the previous subject must not populate the new overlay.

## 3. Independent entrance — Clinical Inbox

Expose **Κλινικά εισερχόμενα** directly from Home, available without selecting an appointment. Home surfaces about 3–4 attention items and a path to the full queue. The full Inbox includes today's arrivals and older unresolved/unlinked/retry items; midnight and upcoming-appointment filters must not hide unfinished work.

An item shows type, source/provider, received time, source-stated subject identity (clearly unverified until linked), attachment/report availability, review state and outstanding action/communication. Filtering by type/review state is presentation only. Gmail read/unread state does not mean clinician-reviewed.

Opening an item permits source inspection, identity resolution, candidate review and the clinician's disposition. The patient-specific record and outbound actions remain gated by confirmed identity. A report may be inspected while unlinked without attaching it to any protected patient.

Actions after adequate source review and verified linkage:

- **Καμία επικοινωνία** — deliberate disposition with a reason; not an inferred default.
- **Ενημέρωση ότι ήρθαν** — editable factual receipt message.
- **Να με καλέσει** — editable callback-window message.
- **Χρειάζεται ραντεβού** — clinician follow-up intent/request; no automatic booking.

The clinical review/action state is shared with Visit Brief. Opening either entrance does not duplicate an item, mark it reviewed, clear an awaited investigation, or generate a new communication attempt.

## 4. Always-current intake and the free Cockpit constraint

Replace the old correctness path `Cockpit opens → search Gmail` with:

```text
authorized Gmail read
  → one always-on Reception-owned intake adapter + durable source receipt/outbox
  → protected idempotent Clinical Excellence Inbox ingress
  → one protected clinical-attention record
  → Cockpit Inbox and Visit Brief projections
```

**Proposed hosting choice:** extend the existing paid Reception service with a bounded background intake job, outside voice/booking request handling. Do not create a second Gmail reader in Cockpit, use the Cal heavy-sync job, change the daily Calendar cron or require keeping the free Cockpit awake. A separate small intake service is an alternative only if Reception cannot safely host this adjacent role; that hosting/cost/ownership change requires a bounded replan before activation.

First integration candidate uses periodic read-only incremental discovery (proposed 5-minute cadence, to be qualified before activation), approved source filters and bounded pages/deadlines. Each completed source batch is durably recorded before advancing its cursor. Page/budget exhaustion leaves coverage incomplete and resumes from a durable continuation; it is not empty-success. An expired Gmail history cursor requires a scoped rescan/reconciliation with deduplication, not loss of unresolved items. Initial discovery must include older unresolved eligible material in the approved lookback, not only today's date. The exact source filter and lookback must be frozen per adapter before enabling it.

Gmail watch/Pub/Sub is an optional future trigger, not a required first deployment. If adopted, it needs watch renewal, authenticated notification admission and periodic gap recovery. A push notification is a discovery hint, not the result or clinical review itself.

Reception's source receipt/outbox retains only the minimum approved metadata/source references needed for delivery and recovery. It never owns clinician review, result acceptance or CareTask state. Clinical Excellence owns that state in its existing protected PostgreSQL environment. These are distinct delivery vs clinical responsibilities, not two competing clinical stores.

Delivery to the free Cockpit must tolerate cold start: persist pending delivery before attempting its protected ingress, use a bounded cold-start-aware deadline, retain failures and retry idempotently. The existing Calendar delivery's 90-second allowance is a reuse candidate, not proof that a new Inbox ingress is wired. Clinical acknowledgement is required before retiring the delivery entry; an ingress timeout is reconciled by item key before a duplicate is created.

Inbox reads display last complete intake coverage, last successful delivery and pending/failed intake state separately. On first open/wake, a stale/unavailable/incomplete source shows **Η ενημέρωση εκκρεμεί / Η πηγή δεν είναι διαθέσιμη**, with existing items retained. It must never present **no new clinical items** as current truth without complete coverage. A bounded refresh may accelerate delivery/read; it cannot become the sole intake mechanism. No same-day guarantee is claimed before sleep/recovery/latency evidence exists.

## 5. Source-specific handling

Classification chooses a versioned adapter using approved provider/source and document evidence. Sender/name/filename alone is not patient identity or proof of test completion. All external text, links and documents are untrusted data; they cannot issue instructions to the app, choose target paths or authorize actions.

**Laboratory results:** supported result PDFs/email content produce proposed structured values with source message/attachment/page or text locator, analyte, original value, unit, source reference range/flag, specimen/result date and date precision where actually present. Preserve comparators, missing/ambiguous units and extraction uncertainty. Receipt time is not specimen date. No unsupported normalization, inferred normal value, diagnosis, treatment recommendation or automated reassurance. The source-stated range/flag is not a Cockpit clinical rule.

**MRI reports:** show the report/document and a proposed report summary with study/modality/body site/laterality/date where supported, source findings, source impression and any source-stated recommendation. Keep report findings separate from clinician interpretation. Do not run the lab-value parser, infer a diagnosis absent from the report, or claim interpretation of the MRI images.

**X-rays / other imaging reports:** same document-summary boundary with modality/site/laterality/date/provenance. Preserve comparison language and uncertainty; an imaging report is not an image-reading service. Image-only/DICOM content is unsupported in the first slice and remains visible for manual source review; absence of extracted text is not a normal result.

**GESY notifications:** show notification type, source-stated status, referenced investigation/referral/authorization and explicit action/deadline if stated. A referral issued, approval, expiry or portal notification is **not a completed clinical result**. An external deadline is not invented from receipt time. If a notification actually attaches a lab/imaging report, create typed document children under the same receipt and route each through the corresponding adapter; the notification itself retains notification semantics. No automatic GESY portal access, login, status mutation or link crawling is included.

**Unknown, mixed, encrypted, corrupt or unsupported material:** retain an explicit manual-review/unsupported state and source access. Mixed attachments have independent review/type/subject states. A message or PDF can concern several patients; linkage is to the selected evidence/document scope, never automatically inherited by all children. Material outside the configured source allow-list is outside intake coverage and must not be claimed handled.

All summaries/extracted values remain candidates. Optional AI processing is a separate processor/privacy gate and reuses an existing provider boundary only if that boundary proves suitable; transcript approval is not blanket email/document approval. Deterministic extraction/source viewing must remain possible without an AI provider. No new clinical interpretation engine is part of this design.

## 6. Shared clinical-attention contract and rightful owners

Design envelopes (not implemented schemas):

- **SourceReceipt** (Reception delivery owner): opaque receipt key, mailbox/account scope, Gmail message ID, source attachment/document keys, source timestamps, adapter/type/version, discovery coverage/cursor, content revision/fingerprint where available, delivery state/attempt/acknowledgement. Gmail is the original source owner.
- **ClinicalAttentionItem** (Clinical Excellence): stable item ID; source receipt/document references; type; source-stated subject; patient-link state/evidence/reviewer/time/version; candidate artifact/provenance/version; clinical review state/reviewer/time; clinician disposition; unresolved clinical action references; Reception communication-action references; record version and sanitized audit metadata.
- **CommunicationAction** (Reception): immutable action ID/idempotency key; originating attention item/revision; confirmed patient/contact reference; approved recipient/text snapshot and callback-policy version; clinician approval identity/time; attempt state; provider outcome/reference and delivery evidence when available. Cockpit consumes its status; no second send ledger is created there.

Use separate state axes rather than a single `unreviewed → linked → communicated` enum:

```text
intake:      pending_delivery | delivered | source_unavailable | incomplete
identity:    unlinked | candidate_only | clinician_confirmed | strong_link | conflict
extraction:  not_requested | candidate_ready | unsupported | failed | needs_manual_review
review:      unreviewed | reviewed | needs_re_review | dismissed_with_reason
disposition: undecided | no_communication | notify_receipt | request_patient_call | needs_appointment
transport:   none | draft | approved | submitted | confirmed_accepted | failed | outcome_unknown
delivery:    unknown | delivered | undelivered (only when provider evidence exists)
```

Reviewed does not mean accepted into the record. Dismissal/no communication does not erase source provenance. Provider acceptance does not mean delivery, patient understanding, completed callback or clinical closure. A pending CareTask/awaited item is closed only by its existing clinical owner after deliberate clinician confirmation of the relevant evidence.

Source correction/new report version requires re-review and invalidates unsent approval tied to the earlier revision. Correcting a patient link removes the prior patient's projection, preserves the audit and invalidates pending patient-specific actions; already sent communication remains historical evidence and requires clinician handling. Concurrent edits use version/conflict checks; stale tabs fail closed rather than apply actions to the wrong subject/version.

## 7. Patient identity and recipient authority

`clinical_patients.patient_id` is a generic internal identity, **not automatically ADT/GESY ID**. The current patient registry stores flexible demographics; it does not prove that a phone is available, current or verified. Appointment ID, display name, exact phone correlation, order number and email sender/subject are insufficient alone for protected clinical identity.

An authorized strong mapping must bind source/document subject to the internal clinical patient with verifiable provenance and no unresolved conflict. Otherwise the clinician selects an existing protected patient and explicitly confirms the relevant identifiers against the source. Name/fuzzy/demographic clues may suggest candidates only. No automatic record creation or write from a candidate match.

Persist only the scoped confirmed link and minimal evidence/reviewer/time. An unlinked/ambiguous item remains reviewable in Inbox but cannot load a suggested patient's history, attach to Visit Brief, become an authoritative result, close an awaited item, or send a patient-specific message.

Confirm outbound recipient independently from an authorized current patient/contact source. Missing/shared/conflicting contacts require explicit clinician resolution; do not infer the recipient from email, Cal name or phone similarity. Clinician-entered/confirmed contact may be captured as the action's protected recipient snapshot without silently editing the patient registry or Reception directory. Approval binds exact item/link version + recipient + edited text + policy; changing any of them requires fresh approval.

## 8. Candidate vs authoritative result boundary

Raw arrival, linking, extraction, summary viewing, **Reviewed**, or sending a message cannot write an authoritative encounter/lab record. Visit Brief may show a linked report with its explicit candidate/review status without portraying extracted values as accepted longitudinal facts or a clinical What Changed delta.

The first bounded implementation excludes **Accept into clinical record**. A later explicit acceptance slice must reuse the existing protected lab write owner (`clinical_data.py`, `clinical_data_ext.py`) and add provenance/idempotency/conflict semantics as needed; the current flexible `values_json` alone does not prove those semantics. Do not create a parallel lab store. MRI/report acceptance and notification-derived CareTasks likewise need their own lossless existing-owner contracts. No result-authority inference from notification completion or SMS success.

## 9. Communication and callback rules

The clinician reviews the source, chooses an action, edits the message, confirms recipient and explicitly presses send. No autonomous outbound text, clinical reassurance, urgency label or treatment advice is generated/sent from a candidate. Patient-initiated reply/callback evidence remains separate from clinician instruction.

**Existing Product Owner rule:** patients may call on any working weekday in configured laboratory-result windows, approximately **10:00–10:20** or **17:40–18:00**, **Asia/Nicosia**. There is no per-patient callback slot, appointment booking, clinic-location assignment or promise that the doctor will call the patient. Physical clinic alternation is irrelevant to the phone window. Use the existing approved callback-policy text/version; do not derive a different window from the appointment or an open Cal availability slot. Before send, confirm that the configured working-day/window policy is current; unavailable/conflicting policy blocks that callback draft rather than inventing one.

Illustrative clinician-editable draft:

```text
Έχουν παραληφθεί τα αποτελέσματά σας.
Αν θέλετε να τα συζητήσετε με τον ιατρό, μπορείτε να επικοινωνήσετε
τις εργάσιμες ημέρες Δευτέρα–Παρασκευή, 10:00–10:20 ή 17:40–18:00.
```

These established windows govern the laboratory callback template. MRI/imaging or GESY actions do not silently acquire new call/booking policy; a clinician may deliberately choose an approved callback template when applicable. Receipt-only notification never implies the results are normal or fully discussed.

**REUSE/EXTEND:** Reception already contains `_zadarma_auth_header`, `_zadarma_post` and `_zadarma_sms_to_doctor` (`main.py`, exact Backend base `e6babb5ab758d282166767c36dd7311024406afb`). The last helper is urgent SMS **to the doctor**, not the new patient-result action. Its raw-body exception/detail behavior must not escape a new clinical boundary. Reuse the transport/signing seam with bounded sanitized handling; extend Reception with one protected clinician-authorized patient-send admission and durable action/outcome ledger. Do not repurpose urgency flags as approval, pass arbitrary browser `authorized=true`, duplicate signing/transport in Cockpit, or depend on the disabled voice agent. Existing helper presence is source evidence, not proof of production patient SMS readiness.

Send admission validates the server-authenticated clinician approval/reference and the bound current item/link/recipient/text; a service credential authenticates a caller but does not grant clinical decision authority. Reception persists an action before the provider call, serializes duplicate submissions by idempotency key and returns the existing action on replay. A deterministic rejection can be corrected and deliberately resubmitted; a timeout after submission is **outcome_unknown**, and never automatically re-sent with a new key. Reconcile provider evidence or require a deliberate clinician decision after inspecting the ambiguity. A sender's `success` envelope must be checked for the actual target's rejection/acceptance evidence; do not equate generic HTTP 200 with delivery.

## 10. Privacy, retention and activation gates

All Gmail/Zadarma credentials and tokens stay server-side, encrypted/protected under the integration owner. Source IDs and candidate data are themselves sensitive; private storage/access controls apply even when full raw content is omitted. No names, contact numbers, document text, candidate values, email bodies, attachments, provider raw errors, transcripts or secrets enter public Git/CI/routine logs/browser persistent storage.

Raw email/PDF is fetched on demand by default, processed transiently and cleared after viewing/processing. Durable references support later re-fetch; a deleted/unavailable source yields an explicit unavailable state, not fabricated content. If durable document archiving becomes clinically necessary, it requires a separately approved protected retention/access/deletion contract. Source retention in Gmail is not guaranteed by this design.

For review continuity, only necessary structured candidates, scoped confirmed links, dispositions and communication evidence may persist in protected stores. Before their field-level implementation/activation, freeze the minimal retained fields, owner/access, lifespan/deletion behavior and processor facts. No indefinite candidate retention or implied GDPR/compliance approval. Gmail disconnect stops new reads and marks freshness unavailable; already approved clinical/action history follows its own explicit retention policy, not an automatic erasure guess.

External attachments require allowed MIME/content validation, configured size/page/decompression/time limits, safe non-executable rendering, and sanitized failure states. Do not follow email-provided URLs or execute document instructions. Unsupported/partial extraction remains visible for manual review.

**Required integration gates:** exact approved sender/pattern/label scope and initial lookback; deployable Gmail OAuth/read authorization and applicable Google verification/assessment; privacy/retention/processor contract; supported document limits/fixtures; verified recipient/policy seam; Reception patient-send admission/idempotency contract; Zadarma sender/account permission; paired cold-start/recovery evidence. These gates are OPEN and must not be promoted to implementation/production readiness by an R2 PASS.

## 11. Source evidence and limits

- Osteoporosis base above: `clinical_data.py` patient/encounter/lab contracts and protected endpoints; `clinical_data_ext.py` lab updates; `clinical_calendar.py` actual-schedule projection; `static/baseline-audit/osteoporosis-longitudinal-summary-core.js` G3 reuse owner. P0-V0 is not on this main; its branch-local contract/review is not automatically released.
- Reception Backend base `e6babb5ab758d282166767c36dd7311024406afb`: `main.py` Zadarma helpers and their urgent-outbound consumer; `render.yaml` variable names only. No credentials, live mailbox, patient document, provider request or SMS was inspected/executed.
- Reception Ops inspected read-only at `750c006ba1fb73752a87c181671355a1c69dca16`: governance/OR-SEC/current release pointers; no Ops writer claim, freeze, release or canonical mutation. Its older smoke-pending checkpoint is superseded for this Cockpit design by the direct Product Owner handback; no cross-project lock/state transfer is inferred.
- [Gmail scopes](https://developers.google.com/workspace/gmail/api/auth/scopes), checked 2026-10-07: `gmail.readonly` is restricted; the documented server storage/transmission assessment requirement must be resolved for the actual app posture. Filtering senders does not narrow OAuth authorization.
- [Gmail sync](https://developers.google.com/workspace/gmail/api/guides/sync), checked 2026-10-07: history-based incremental sync and expired-history recovery support the proposed intake recovery boundary.
- [Gmail push](https://developers.google.com/workspace/gmail/api/guides/push), checked 2026-10-07: renewal and periodic recovery remain necessary if push is chosen; it is not guaranteed lossless delivery.
- [Zadarma API](https://zadarma.com/en/support/api/), checked 2026-10-07: SMS send/sender APIs exist and expose target rejection information; exact production eligibility, delivery/reconciliation and provider idempotency support remain unverified. Do not invent them.

## 12. Bounded implementation sequence after review and separate authority

**A — shared core and both entrances:** source/reference, scoped identity/link, orthogonal review/disposition and status contracts; appointment overlay using existing confirmed clinical data; independent Inbox with synthetic typed sources and manual-source viewing. No provider effect or authoritative result acceptance. If released alone, label it accurately as core/manual capability; it does not satisfy live Gmail same-day intake.

**B — first live end-to-end vertical:** one approved laboratory sender family (Medicover first, exact source configuration not yet supplied), one supported PDF adapter, Reception always-on discovery/durable delivery, independent Inbox even without appointments, candidate review and clinician-approved callback/receipt messaging through Reception → Zadarma. Same item visible in later Visit Brief. Keep all source/privacy/send gates above fail-closed; qualify before live activation.

**C — source-specific extensions:** approved MRI and X-ray/imaging report adapters, then GESY notification adapter under the same attention contract. They may share safe document viewing/extraction mechanics but retain their distinct semantics. Unsupported recognized items stay visible for manual handling within enabled source coverage. Broad multi-provider ingestion, image diagnosis, automatic patient matching, authoritative acceptance, automatic tasks/messages/booking and standalone full D2 remain deferred.

This sequence supersedes the previous appointment-dependent lab workflow. Inbox identity/workflow contracts are designed now with Visit Brief; its live intake cannot be declared complete merely because the overlay shipped. Each future implementation-bearing slice must have its exact contract, writer and finite evidence plan under its governing CURRENT before code starts.

## 13. Finite future acceptance oracles and REPLAN

The independent pre-code review checks implementability and these obligations; it does not run or claim runtime tests:

1. Arrival without any appointment becomes a durable Inbox item while Cockpit sleeps; recovery/wake retains backlog and exposes coverage gaps.
2. Opening either entrance shows the same scoped item/review/action references; later Visit Brief reflects review and communication without duplicate facts/effects.
3. Cal name, shared phone, subject name and mixed-patient attachments cannot authorize patient history, authoritative writes or send; correction/stale response invalidates dependent actions.
4. Labs produce proposed typed values; MRI/X-ray reports produce source summaries; GESY referral/approval is not a completed result; unsupported content remains explicit.
5. Reviewed/communicated never accepts clinical values or closes a clinical task; missing, negative, unknown and conflict remain distinct.
6. Exact approved recipient/text/revision is bound to a deliberate send; double-click/replay/timeout does not cause blind duplicate SMS; target rejection and outcome_unknown remain truthful.
7. Callback text retains configured working weekdays/10:00–10:20/17:40–18:00, no booked slot or inferred physician outbound call; voice disabled does not block dashboard action.
8. PHI/tokens/raw provider errors cannot escape to public artifacts, logs or browser persistent storage; document and integration limits fail closed.

REPLAN if identity authority, source type meaning, intake/clinical/transport ownership, result promotion, retention/processor facts, hosting/cost, callback policy or provider feasibility changes materially. Preserve the independently verified Calendar runtime; do not reopen its closed review absent a new source-proven defect. One pre-code R2, bounded correction/closure if needed, and a later exact-head post-code fidelity review are separate gates.

## 14. Product Owner continuation — verified pending follow-through and Daily Brief (2026-10-17, DESIGN ONLY)

> **Adopted future behavior, separate from the reviewed 2026-10-07 R2 and the frozen Verified Appointment ↔ Patient & Visit Brief v1 design/implementation PR #161.** This addendum leaves §§1–13 and their historical review intact; it grants NO implementation, clinical-data processing, schema migration, new review dispatch, provider access, merge or deployment. Morning/evening briefs were previously discussed as short summaries, initially on-demand and later potentially automatic; a separately frozen midday/scheduling contract does not yet exist. Do not imply three production Briefs are running.

### 14.1 Objective and existing mechanisms

**AI recognizes and proposes an obligation; Cockpit tracks it; the clinician certifies what happened; a later separately qualified Secretary can read only authorized, verified status.** A signed Visit Capture stating that a patient should bring an MRI CD, complete tests, return after Prolia or arrange follow-up is evidence of a plan, NOT of fulfillment. Cockpit must preserve explicit clinical commitments and progress evidence without a large manually maintained to-do board.

**REUSE/EXTEND:** the existing protected clinical_data.py patient/encounter database and ClinicalPendingORM, with VisitCaptureCandidateV1.pending and the existing signed Save. Current Save stores a patient- and encounter-scoped ClinicalPendingORM with stable ID, description, status, responsible role, dependency, review date, provenance and optional resolution-event reference. Its current protected pending endpoint is a read-only list; there is **no implemented generic per-stage confirmation/history writer**. A future minimal, versioned, auditable set of checkpoint events belongs to this **same Clinical Pending owner**, not a new task engine, second patient registry or separate database. Historical signed encounters remain immutable; later events are separate clinical pending/longitudinal events. The designed ClinicalAttentionItem/Inbox can refer to those pending IDs but must not be represented as already deployed or create a second independently completable obligation. ClinicalEncounter, ClinicalPendingItem, ClinicalAttentionItem, Reception action and CommunicationAction remain different authorities.

### 14.2 Admission and provenance

- **Visit Capture:** existing Dia/Heidi/GESY candidate processing may propose typed pending entries from explicit source-stated commitments. They become clinician-owned pending items only through the existing protected, independently verified patient selection, clinician review and explicit Save of VisitCaptureCandidateV1.pending. Do NOT assume today's Save extracts structured pending fields from arbitrary unstructured narrative; if missing, present a proposed candidate for deliberate review in a separately approved future UI slice. No model text silently creates an authoritative clinical action.
- **Calls:** use Reception's existing protected calls and action_queue readings with call ID, action type and operational outcome. “Please call me about my test results” may be a proposed pending clinical follow-through; until strong identity/clinician acceptance exists, it remains an operational request with its own Reception owner. Phone, caller name or booking UID do not prove protected clinical patient identity. Reconcile duplicates by stable originating event IDs/revisions; do not create one clinical task per repeated call or another call/booking engine. Reception “done” is NOT evidence the doctor reached the patient.
- **Clinical evidence:** a signed plan proves agreed intent; a patient claim proves only what was said; a received document/report proves only its receipt when independently verified. None automatically proves clinician review, medical findings, communication completion or treatment.

### 14.3 One generic item, independently confirmed ordered checkpoints

One source-grounded pending ID can have conditional, clinician-approved checkpoints, expressed through small kinds/order/condition and separate confirmation events. Examples: material received; images/CD personally reviewed; written report reviewed; telephone contact performed; patient informed, if actually required. Avoid special-purpose logic per CD, labs, referral, postoperative care or Prolia. A minimum future event must retain pending ID, patient+source encounter scope, stage meaning, relevant evidence/confirmation time, version/idempotency, and only truthful operator attribution. The current shared Clinical Key alone cannot establish the identity of a specifically named doctor; never invent it. Genuine protected individual attestation/role must be separately qualified where necessary.

**States cannot collapse:** agreed_to_bring ≠ receipt_confirmed ≠ images_review_confirmed ≠ patient_informed. Viewing a written radiologist report does not prove the CD images were examined, and image review does not imply diagnosis, patient contact, result interpretation or patient understanding. Make communication a stage only when required by the accepted clinical commitment/clinician, not a universally invented obligation.

**Micro-actions:** “Ναι” confirms only the specific checkpoint after deliberate clinician attestation. “Όχι ακόμη” records an explicit timestamped *not yet at the time of checking*, not permanent absence. **No answer = unknown**, never a synthetic negative. Advance the displayed next question only after a successful, version-guarded committed prior confirmation; never mark the whole item resolved while a required approved stage remains unknown/open. Support correction/retraction by separate traceable event, not silent overwriting. A late, repeated or cross-patient response must fail closed.

**MRI-CD synthetic lifecycle:**

| Existing evidence / question | Only justified state |
| --- | --- |
| Approved visit: patient will bring CD | Agreed; receipt UNKNOWN |
| Next morning: “Προσκομίστηκε το CD της μαγνητικής;” | Question shown; unanswered remains UNKNOWN |
| Clinician presses “Όχι ακόμη” | Explicit not-yet observation at that moment; receipt still open |
| Clinician presses “Ναι” after verification | CD receipt confirmed; next proposed question “Εξετάστηκαν οι εικόνες από τον γιατρό;” |
| Clinician confirms image review | Images personally reviewed; report-reading remains separate |
| Patient needs informing and clinician later confirms contact | Patient-informed stage complete; parent closes only when all required milestones confirmed |

### 14.4 One source of truth, several lightweight read surfaces

**Daily Brief is a view over the same Clinical Pending facts/events, not a new stateful task engine.** The protected patient record and Visit Brief show the same actual recorded state and event time after a successful one-tap confirmation. Morning briefly surfaces outstanding clinician-verification questions from prior signed visits (including the next-day CD check); midday focuses on *changes* since the previous Brief and only new actionable call requests; evening differentiates explicitly completed versus unknown/open items and tomorrow's useful follow-ups. “Ask next morning” may follow a still-open signed commitment even without a formal booked-visit due date; do not invent a due date from it.

Preserve the separate exact **selected Cyprus-local appointment date equality rule** of Verified Appointment ↔ Patient & Visit Brief v1: a routine Daily Brief progress question is not automatically a sourced “today's check” for an unrelated selected appointment. No answer leaves status unknown. Deduplicate resurfaced questions, cap visible high-priority items, avoid repeated morning text at midday, and reveal details only on tap. Bookings created are informational aggregates, not action items. No extra manual queue/forms, automatic diagnosis, auto-close or new AI model required merely to display status.

Brief delivery automation, exact morning/midday/evening times and distribution remain separate future decisions. Initial Brief surfaces may be on-demand and read-only except for the deliberate protected confirmation action already belonging to Clinical Pending. Do not claim the three Briefs are currently installed.

### 14.5 Secretary — separately qualified, later read-only bridge

No M8/LiveKit/GPT-Live/telephone tool or Reception provider changes are authorized here. A future **read-only, purpose-limited Clinical Pending status** interface may serve only the minimal clinician-verified stage and its uncertainty/freshness, after exact patient/caller identity qualification, clinical-data authorization and a separate contract. Never authorize protected patient history from phone/name/booking UID alone. A caller asking “Είδε ο γιατρός το CD μου;” may hear only: receipt unconfirmed; received but image review unconfirmed; or clinician-confirmed image review. A written radiology report reviewed is not the same event. A request for an outbound contact uses Reception's existing message/action pathway with explicit recipient/action authorization; sending a message never proves the patient was informed or closes clinical follow-through.

### 14.6 Smallest successor after already active work — NOT activated now

Once PR #161 finishes its own independent R2/release checkpoint, the next separately authorized **Clinical Pending confirmation micro-loop v1** can use synthetic or manually approved structured Visit Capture pending rows: one patient-scoped existing pending ID → one concise Daily Brief/patient-card question → clinician “Ναι / Όχι ακόμη / no response” → transactional, versioned event in the existing clinical owner → next already approved stage, where applicable → same read projection in patient record/Visit Brief. A single MRI-CD receipt → image-review example is sufficient to prove general mechanism without adding another task engine. Further AI free-text extraction, live call-to-patient matching, Reception imports, Gmail, outgoing messages, automatic scheduling and Secretary read access are **separate later gates**, not the micro-loop.

Finite synthetic acceptance: unanswered stays unknown; time-bounded explicit not-yet remains open; receipt does not complete image review; written-report review is not image review; communication and clinician action have separate evidence; pending persists across days without changing signed visit; two views read the same ID/state; repeat/concurrent/stale/mismatched-patient updates cannot double advance or leak PHI; source correction is auditable. Existing Home #160, frozen PR #161 code/design, Save/Calendar behavior and all previously accepted reviews are untouched. A later writer, privacy/retention and clinician-identity qualification, focused test/review, and separate explicit release/live-data approval are still mandatory.
