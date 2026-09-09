# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1C Challenge Completion Transport

> **STATUS:** DESIGN + IMPLEMENTATION ACTIVE
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1C-CHALLENGE-COMPLETION-TRANSPORT-2026-09-09`.
> **Base:** `a2af17381eb53de6f1deac4ea1c987e743f6e951`.
> **Branch:** `feat/clinical-learning-l1c-challenge-completion-transport-2026-09-09`.
> **L-1B:** CLOSED / PRODUCTION-SMOKE-VERIFIED PASS.
> **Frozen L-0/L-1 schema owners:** READ-ONLY.

---

# 1. Problem

The Cockpit can already accept structured synthetic learning episodes into `Pending Imports`, but a normal Challenge conversation has no durable rule that says:

> when the challenge is finished, hand the complete learning episode to the Cockpit.

Relying on chat memory, prior wording or the clinician remembering to ask for JSON is not acceptable.

---

# 2. Objective

Create a durable Challenge Completion Protocol that every Challenge conversation can inherit from a ChatGPT Project instruction.

The protocol must make Cockpit handoff part of the challenge's definition of done:

```text
CASE
→ CLINICIAN REASONING
→ PROGRESSIVE DISCLOSURE
→ FOLLOW-UP REASONING
→ FINAL DECISION
→ DEBRIEF
→ LEARNING PLAN
→ FRESH RESOURCE DISCOVERY WHEN APPLICABLE
→ STRUCTURED EPISODE
→ COCKPIT HANDOFF
→ RECEIPT
→ COMPLETE
```

No receipt means no claim that Cockpit was updated.

---

# 3. ChatGPT Project instruction strategy

Use one dedicated ChatGPT Project for Osteoporosis Clinical Learning / Challenges.

All Challenge chats inside that Project inherit one Project instruction. This removes dependence on each individual thread remembering old instructions.

The Project instruction must require:

- high-level case-based challenge behavior;
- ask for clinician answer before critique;
- progressive disclosure;
- explicit strengths / reinforcement / clear errors / evidence gaps / blind spots;
- deliberate knowledge-island bridging;
- learning actions and current high-quality resources where appropriate;
- repeated consolidation targets;
- end-of-challenge structured episode generation;
- Cockpit transport attempt as the final workflow step;
- no success claim without a returned transport receipt.

The Project instruction does **not** create patient-record authority, Foundation-state authority, Signal authority or evidence-verification authority.

---

# 4. Completion states

A Challenge conversation has one of these end states:

```text
IN_PROGRESS
DEBRIEF_COMPLETE_HANDOFF_PENDING
HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW
HANDOFF_FAILED_MANUAL_FALLBACK_READY
```

Only `HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW` permits wording equivalent to “sent to Cockpit”.

Even then the episode remains `pending_review` in the Cockpit until clinician review.

---

# 5. Receipt contract

Minimum receipt:

```json
{
  "state": "pending_review",
  "import_id": "uuid",
  "source_event_id": "uuid",
  "source_format": "rich_challenge_export_v1"
}
```

The Challenge conversation must surface the receipt state concisely and must not reinterpret `pending_review` as accepted/saved/verified.

---

# 6. Failure contract

If transport is unavailable or fails:

```text
DO NOT CLAIM COCKPIT UPDATED
DO NOT DISCARD EPISODE
DO NOT RETRY WITH DIFFERENT CONTENT UNDER SAME source_event_id
DO NOT ROUTE THROUGH UNTRUSTED GENERIC WEBHOOK
```

Instead:

```text
state = DEBRIEF_COMPLETE_HANDOFF_PENDING or HANDOFF_FAILED_MANUAL_FALLBACK_READY
retain structured episode
present existing Advanced/manual import fallback
preserve same stable source_event_id for a same-content retry
```

---

# 7. Platform capability boundary

Current OpenAI product documentation supports Project instructions broadly, so the durable conversation rule can be installed now.

Native zero-click write from a ChatGPT conversation requires a write-capable app/MCP tool. Availability depends on the ChatGPT surface/workspace; current OpenAI documentation limits full MCP write actions to Business / Enterprise / Edu, while Pro custom MCP is read/fetch only.

Therefore this slice must not claim zero-click transport is active until an actual trusted write tool is connected and smoke-tested.

---

# 8. Cockpit setup UX

Add a small protected setup surface in the Learning Hub that shows:

- Challenge Completion Protocol status;
- exact Project instruction text;
- Copy instructions button;
- current transport mode:
  - `project_instruction_ready`
  - `native_write_tool_not_connected`
  - later `native_write_tool_connected`;
- existing manual fallback remains under Advanced.

This setup UX stores no secret and makes no patient/learning write.

---

# 9. Security / privacy

Permanent rules:

```text
synthetic learning only for automatic ingress
verbatim clinician responses required
raw transcript never transported/persisted by this path
no patient identifiers
no imported clinician-review authority
no imported reference-verification authority
no Signal promotion
no Foundation-state mutation
```

Do not configure `CLINICAL_LEARNING_INGEST_KEY` until a trusted write-capable consumer is actually being configured in the same bounded integration lifecycle.

---

# 10. Acceptance evidence

L-1C is implementation-complete when:

- exact Project instruction is versioned in repo;
- completion state machine is documented and testable;
- Cockpit setup UI exposes/copies the exact instruction;
- UI states that native write tool is not connected rather than implying automatic transport;
- manual Advanced fallback remains available;
- no code path can claim handoff success without a receipt object with `state=pending_review` and required IDs;
- no patient/Signal/Foundation/RF/physio/CU-1 mutation;
- inherited L-1B/L-1 tests remain green.

Native zero-click transport activation is a later acceptance criterion only when the account/workspace supports the write-capable app and the app is actually connected.

---

# 11. Release path

```text
project instruction contract
→ Cockpit setup UX
→ focused L-1C tests + inherited L-1B/L-1
→ exact-head review
→ PR / RELEASE HOLD
```

No production secret/config mutation in this initial L-1C implementation.