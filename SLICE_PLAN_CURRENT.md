# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1C Challenge Completion Transport

> **STATUS:** IMPLEMENTED / TESTED / RELEASE HOLD
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1C-CHALLENGE-COMPLETION-TRANSPORT-2026-09-09`.
> **Base:** `a2af17381eb53de6f1deac4ea1c987e743f6e951`.
> **Branch:** `feat/clinical-learning-l1c-challenge-completion-transport-2026-09-09`.
> **Exact tested head before docs-only closeout:** `ffba01a600164122c3ad8dd16d0f48b3e56e1d53`.
> **Gate:** `34399853207` — SUCCESS.
> **L-1B:** CLOSED / PRODUCTION-SMOKE-VERIFIED PASS.
> **Frozen L-0/L-1 schema owners:** READ-ONLY / unchanged.
> **Writer lock:** NONE.

---

# 1. Problem closed by this slice

The Cockpit already accepts structured synthetic episodes, but Challenge conversations previously depended on remembered prompts to know that they must hand off the completed learning episode.

L-1C makes the handoff a durable workflow contract inherited from a ChatGPT Project instruction.

---

# 2. Definition of done

```text
CHALLENGE COMPLETE
=
FINAL DEBRIEF COMPLETE
+ STRUCTURED LEARNING EPISODE COMPLETE
+ COCKPIT HANDOFF ATTEMPT COMPLETE
+ RECEIPT OR EXPLICIT TRANSPORT FAILURE
```

No valid receipt means the conversation must not claim the Cockpit was updated.

---

# 3. Versioned Project instruction

Canonical instruction:

`clinical_learning/chatgpt_project_instructions_v1.txt`

Browser-copy mirror:

`static/clinical-learning/chatgpt-project-instructions-v1.txt`

Focused regression requires them to be byte-identical.

The instruction requires:

- clinician answer before critique;
- progressive disclosure where appropriate;
- strengths / needs reinforcement / clear errors kept distinct;
- evidence gaps, blind spots, reasoning patterns and insights;
- deliberate knowledge-island bridge targets;
- repeated consolidation rather than one-off quiz;
- targeted current resources when appropriate;
- verbatim clinician reasoning in the structured episode;
- no patient identifiers/raw transcript/imported authority;
- Cockpit transport attempt as the final workflow step;
- no success wording without a returned pending-review receipt.

---

# 4. Completion state machine

```text
IN_PROGRESS
DEBRIEF_COMPLETE_HANDOFF_PENDING
HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW
HANDOFF_FAILED_MANUAL_FALLBACK_READY
```

Receipt validator owner:

`clinical_learning/challenge_completion_protocol.py`

Successful transport requires:

```text
state = pending_review
import_id = valid UUID
source_event_id = valid UUID
source_format = canonical_challenge_v1 | rich_challenge_export_v1
```

Tool invocation without receipt is not success evidence.

---

# 5. Cockpit setup surface

Static setup page:

`/static/clinical-learning/project-setup.html`

It exposes:

- exact Project instruction;
- Copy Project Instructions;
- truthful capability status;
- explanation of the completion workflow.

Current status intentionally reads:

```text
Project instruction = READY
Native Cockpit write tool = NOT CONNECTED
Advanced/manual fallback = AVAILABLE
```

No credential is rendered to the browser.

---

# 6. Security / privacy

Preserved:

```text
synthetic learning only for automatic ingress
verbatim clinician responses required
raw transcript never transported/persisted by this path
no patient identifiers
no imported clinician-review authority
no imported reference-verification authority
no Signal promotion
no Foundation-state mutation
no third-party generic webhook bypass
```

`CLINICAL_LEARNING_INGEST_KEY` remains unconfigured because there is not yet a concrete trusted write-capable ChatGPT consumer ready to receive the same credential.

---

# 7. Platform capability boundary

Durable Project instructions are usable now.

Native zero-click write requires an actual write-capable ChatGPT app/MCP action on a supported surface/workspace. Current OpenAI documentation limits full MCP write actions to Business / Enterprise / Edu, while Pro custom MCP is read/fetch only.

Therefore zero-click write activation remains a later integration lifecycle step. L-1C does not mislabel it as connected.

---

# 8. Verification

Exact tested implementation head:

`ffba01a600164122c3ad8dd16d0f48b3e56e1d53`

GitHub Actions:

`Clinical Learning L1C challenge transport gate` run `34399853207` — **SUCCESS**.

Passed:

- focused L-1C completion/receipt/setup regressions;
- inherited L-1B;
- inherited L-1;
- frozen-owner guard;
- bounded scope guard;
- diff hygiene.

---

# 9. Release state

```text
IMPLEMENTED = YES
TESTED = YES
PROJECT PROTOCOL = READY
SETUP SURFACE = READY
NATIVE WRITE TOOL = NOT CONNECTED
PRODUCTION INGEST KEY = NOT CONFIGURED
PR = NEXT / DRAFT
MERGED = NO
DEPLOYED = NO
```

Next allowed action: Draft PR / RELEASE HOLD. Merge/deploy requires separate explicit product-owner authority.