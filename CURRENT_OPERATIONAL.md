# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1C — CHALLENGE COMPLETION TRANSPORT / DESIGN + IMPLEMENTATION ACTIVE
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh implementation base:** `a2af17381eb53de6f1deac4ea1c987e743f6e951`.
> **Branch:** `feat/clinical-learning-l1c-challenge-completion-transport-2026-09-09`.
> **Slice:** `CORE-LEARNING-HUB-L1C-CHALLENGE-COMPLETION-TRANSPORT-2026-09-09`.
> **ACTIVE RUNTIME/DESIGN WRITER:** THIS bounded L-1C slice only.
> **Production config/secret authority:** NOT YET EXERCISED.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Starting point

L-1B is closed and production-smoke-verified. The Cockpit already owns:

```text
POST /clinical/learning/api/ingress/episodes
X-Learning-Ingest-Key
CLINICAL_LEARNING_INGEST_KEY
```

That endpoint accepts explicit synthetic structured learning only, requires verbatim clinician reasoning for automatic rich ingress and creates only `pending_review` candidates.

What remains is the **ChatGPT-side completion/transport contract**: a challenge conversation must know that a completed challenge is not finished until the structured learning episode has been handed off to the Cockpit or a transport failure has been made explicit.

---

# 2. L-1C product rule

```text
CHALLENGE COMPLETE
!=
FINAL ANSWER GIVEN

CHALLENGE COMPLETE
=
FINAL DEBRIEF COMPLETE
+ STRUCTURED LEARNING EPISODE COMPLETE
+ COCKPIT TRANSPORT ATTEMPT COMPLETE
+ RECEIPT OR EXPLICIT TRANSPORT FAILURE
```

The conversation must never rely on remembered chat habits for this. The rule must be installed as a durable Project instruction / reusable workflow instruction.

---

# 3. Required end-of-challenge payload

Before transport, the challenge conversation must preserve:

```text
initial hypothetical scenario
verbatim clinician responses
progressive disclosures
follow-up reasoning and final decision
strengths
needs reinforcement
clear errors
defensible disagreements
evidence gaps / blind spots / reasoning patterns / insights
learning objectives / actions
knowledge-island bridge targets
fresh resource recommendations when current web research was requested/performed
repetition / consolidation candidates
```

No patient identifiers. No raw transcript. No imported clinician-review/reference-verification/Signal authority.

---

# 4. Transport receipt

Successful transport is not inferred from tool invocation. The conversation requires a returned receipt containing at minimum:

```text
transport_state = pending_review
import_id
source_event_id
source_format
```

Only then may it tell the clinician that the episode was sent to the Cockpit.

Failure behavior:

```text
tool unavailable / write action unavailable / endpoint unavailable / auth failure
→ do not claim success
→ retain the structured episode in the conversation
→ label COCKPIT_HANDOFF_PENDING
→ offer the existing Advanced/manual fallback artifact without weakening privacy/review boundaries
```

---

# 5. ChatGPT platform constraint discovered during bootstrap

Current OpenAI product documentation supports durable Project instructions across chats. Write-capable custom MCP actions are surface/plan dependent; current OpenAI documentation states full MCP write actions are available on Business / Enterprise / Edu, while Pro custom MCP access is read/fetch only.

Therefore L-1C separates:

```text
A. durable Challenge Completion Protocol — implementable now
B. native zero-click Cockpit write tool — capability-gated by ChatGPT app/write-action availability
```

No clinical-learning data will be routed through an unrelated generic third-party webhook merely to bypass this platform boundary.

---

# 6. Allowed mutation scope

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
CLINICAL_LEARNING_CHAT_TRANSPORT_V1.md                  # new
clinical_learning/chatgpt_project_instructions_v1.txt  # new
clinical_learning/challenge_completion_protocol.py      # optional deterministic helper
static/clinical-learning/index.html                     # optional setup UX
static/clinical-learning/l1c.js                         # optional setup UX
static/clinical-learning/styles.css                     # optional setup UX
clinical_learning/api.py                                # only if bounded setup/status endpoint needed
test_clinical_learning_l1c_*.py                         # focused regressions
.github/workflows/clinical-learning-l1c-tests.yml       # optional additive gate
TODO.md / CLINICAL_EXCELLENCE_PLAN.md                  # only if durable roadmap architecture changes
osteoporosis-change-log.md                              # append-only at material milestone
```

Frozen L-0/L-1 schemas remain read-only. No patient/Signal/Foundation/RF/physio/CU-1 owner mutation.

---

# 7. Exact next action

```text
freeze Project completion protocol
→ publish exact Project instructions in repo
→ define transport receipt/failure semantics
→ add Cockpit setup surface so clinician can copy/install the Project instruction once
→ regression-test that completion wording cannot claim Cockpit success without receipt
→ capability-gate native write-tool activation
→ exact-head review / RELEASE HOLD
```

Do not configure a production ingest secret until there is a concrete, trusted write-capable consumer ready to receive the same credential. Do not weaken the existing ingress authentication boundary.