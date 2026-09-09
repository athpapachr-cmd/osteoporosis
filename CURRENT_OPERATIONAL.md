# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1C — IMPLEMENTED / TESTED / RELEASE HOLD
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Base:** `a2af17381eb53de6f1deac4ea1c987e743f6e951`.
> **Branch:** `feat/clinical-learning-l1c-challenge-completion-transport-2026-09-09`.
> **Slice:** `CORE-LEARNING-HUB-L1C-CHALLENGE-COMPLETION-TRANSPORT-2026-09-09`.
> **Exact tested head before closeout:** `ffba01a600164122c3ad8dd16d0f48b3e56e1d53`.
> **Gate:** Clinical Learning L1C challenge transport run `34399853207` — SUCCESS.
> **ACTIVE RUNTIME/DESIGN WRITER:** NONE.
> **Production config/secret authority exercised:** NO.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. What L-1C now solves

A Challenge conversation no longer depends on remembering an old prompt to know what happens at the end. The durable owner is a versioned ChatGPT Project instruction:

`clinical_learning/chatgpt_project_instructions_v1.txt`

Its definition of done is:

```text
final debrief
+ structured learning episode
+ Cockpit handoff attempt
+ returned receipt or explicit transport failure
```

No receipt means no claim that the Cockpit was updated.

---

# 2. Completion states

```text
IN_PROGRESS
DEBRIEF_COMPLETE_HANDOFF_PENDING
HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW
HANDOFF_FAILED_MANUAL_FALLBACK_READY
```

A success state requires a receipt with:

```text
state = pending_review
valid import_id
valid source_event_id
recognized source_format
```

`pending_review` remains pending clinician review; it is not accepted/verified learning authority.

---

# 3. Delivered artifacts

```text
CLINICAL_LEARNING_CHAT_TRANSPORT_V1.md
clinical_learning/chatgpt_project_instructions_v1.txt
clinical_learning/challenge_completion_protocol.py
static/clinical-learning/chatgpt-project-instructions-v1.txt
static/clinical-learning/project-setup.html
test_clinical_learning_l1c_challenge_completion_transport.py
.github/workflows/clinical-learning-l1c-tests.yml
```

The setup page exposes the exact Project instruction with a Copy button and truthfully reports:

```text
Project instruction = READY
Native Cockpit write tool = NOT CONNECTED
Advanced/manual fallback = AVAILABLE
```

No secret is exposed or stored by the setup UI.

---

# 4. Verification

Exact tested implementation head:

`ffba01a600164122c3ad8dd16d0f48b3e56e1d53`

Workflow:

`Clinical Learning L1C challenge transport gate` run `34399853207` — **SUCCESS**.

Passed:

- Python syntax;
- Project-instruction canonical/static byte equality;
- completion protocol wording/required states;
- receipt validation;
- no success without receipt;
- no debrief → IN_PROGRESS;
- no transport → HANDOFF_PENDING;
- invalid/missing receipt → manual fallback state;
- valid pending-review receipt → success-pending-review only;
- setup UI truthful transport status;
- inherited L-1B tests;
- inherited L-1 tests;
- frozen-owner guard;
- scope guard;
- diff hygiene.

---

# 5. Platform boundary / next integration step

Current OpenAI product documentation supports Project instructions, so the conversation-side completion rule can be installed now.

Native zero-click Cockpit write still requires an actual trusted write-capable ChatGPT app/MCP action on a supported workspace/surface. This slice does not pretend that tool is connected.

No unrelated third-party generic webhook is used as a bypass.

---

# 6. Release state

```text
L-1C IMPLEMENTED = YES
L-1C TESTED = YES
PROJECT COMPLETION PROTOCOL = READY
COCKPIT SETUP PAGE = READY
NATIVE WRITE TOOL = NOT CONNECTED
PRODUCTION INGEST KEY = NOT CONFIGURED
PR = NEXT / DRAFT
MERGED = NO
DEPLOYED = NO
WRITER LOCK = NONE
```

Exact next action: open bounded Draft PR / RELEASE HOLD. Merge/deploy requires separate product-owner release authority.