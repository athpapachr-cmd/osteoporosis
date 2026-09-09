# CLINICAL_LEARNING_CHAT_TRANSPORT_V1.md

## Status

L-1C candidate design for making Cockpit handoff an explicit end-of-Challenge workflow contract rather than a remembered conversational habit.

## Core principle

A Challenge conversation is not complete merely because the final critique has been given.

```text
CHALLENGE COMPLETE
=
DEBRIEF COMPLETE
+ STRUCTURED EPISODE COMPLETE
+ COCKPIT HANDOFF ATTEMPT COMPLETE
+ RECEIPT OR EXPLICIT FAILURE
```

## 1. Durable instruction owner

The durable ChatGPT-side owner is a Project instruction shared by all Osteoporosis Clinical Learning / Challenge chats.

The Project instruction is versioned at:

`clinical_learning/chatgpt_project_instructions_v1.txt`

A conversation inside the Project must follow it regardless of whether that individual thread has previously discussed the Cockpit.

## 2. End-of-Challenge sequence

1. Finish the clinical reasoning/debrief.
2. Separate strengths, reinforcement needs, clear errors, defensible disagreements, evidence gaps/blind spots and clinical insights.
3. Identify deliberate knowledge-island bridge targets when concepts should be applied jointly.
4. Define targeted study actions and repeated consolidation targets.
5. When the challenge calls for current external learning opportunities, search for current high-quality article/guideline/webinar/course resources and mark their freshness/access state when known.
6. Build the structured synthetic learning episode using the observed rich export contract accepted by L-1B.
7. Preserve the clinician's verbatim responses rather than replacing them with AI summaries.
8. Attempt Cockpit handoff using the dedicated learning transport tool when available.
9. Inspect the returned receipt.
10. Report one of the defined completion states.

## 3. Completion states

### `IN_PROGRESS`
The case/debrief is still active.

### `DEBRIEF_COMPLETE_HANDOFF_PENDING`
The learning episode is ready but no trusted write transport is available yet.

### `HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW`
A trusted transport tool returned:

```json
{
  "state": "pending_review",
  "import_id": "uuid",
  "source_event_id": "uuid",
  "source_format": "canonical_challenge_v1 | rich_challenge_export_v1"
}
```

The conversation may say the episode was sent to the Cockpit, but must also say it is pending clinician review.

### `HANDOFF_FAILED_MANUAL_FALLBACK_READY`
A trusted transport exists but failed. The conversation preserves the same structured episode and stable source-event identity for retry, and exposes the Advanced/manual fallback.

## 4. Stable identity / retry

The source event identity belongs to the learning episode, not to a transport attempt.

For the same episode bytes/content:

```text
same episode
→ same source_event_id
→ retry allowed
```

For materially changed episode content:

```text
changed episode
→ new source_event_id or explicit later revision contract
```

Do not silently reuse a source event ID for different normalized content.

## 5. What must cross the boundary

Required:

- synthetic case;
- verbatim clinician reasoning responses;
- progressive disclosures;
- final decision;
- debrief observations;
- evidence/reference snapshots as unverified external references;
- learning gaps/actions;
- Foundation targets;
- bridge targets / consolidation plan where supplied;
- fresh resource recommendations when available.

Forbidden:

- patient name;
- identity/GeSY number;
- phone/email/address/DOB;
- raw transcript;
- clinician-review self-certification;
- reference-verification self-certification;
- Signal authority;
- patient-record authority.

## 6. Transport tool semantics

The native write tool, once available, must expose one narrow operation conceptually equivalent to:

```text
send_learning_episode(episode, source_event_id, resources?, loop_plan?)
```

The operation may only create a Cockpit `pending_review` import. It cannot create an accepted Challenge or mutate Foundation/Signal/patient state.

The Cockpit server remains authoritative for schema validation, PHI guard, normalization and idempotency.

## 7. Conversation wording rules

Allowed after a valid receipt:

> Στάλθηκε στο Cockpit και περιμένει δικό σου review.

Forbidden without receipt:

> Το αποθήκευσα στο Cockpit.
> Το Challenge ενημερώθηκε αυτόματα.
> Το learning record δημιουργήθηκε.

Tool invocation alone is not success evidence.

## 8. Current platform boundary

Project instructions can provide the durable workflow rule now.

A true zero-click write requires an actual write-capable ChatGPT app/MCP action. Until such a trusted tool is connected for the user's supported surface/workspace, the completion protocol stops in `DEBRIEF_COMPLETE_HANDOFF_PENDING` and the existing Advanced/manual fallback remains valid.

No unrelated third-party generic webhook is introduced merely to bypass the platform write boundary.

## 9. Cockpit setup UX

The Learning Hub should expose the exact Project instruction text with one-click copy, together with a truthful transport status:

```text
Project instruction: READY
Native write tool: NOT CONNECTED
Manual fallback: AVAILABLE
```

No secret is exposed to the browser.

## 10. Exit criteria

The initial L-1C runtime/design slice is complete when the versioned Project instruction, setup UX and receipt/failure semantics are implemented and regression-tested. Native write activation remains a separate lifecycle step until an eligible trusted ChatGPT write tool can actually be connected and smoke-tested.