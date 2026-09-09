# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1C Challenge Completion Transport

> **STATUS:** MERGED / DEPLOY VERIFICATION PENDING
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1C-CHALLENGE-COMPLETION-TRANSPORT-2026-09-09`.
> **PR #85:** CLOSED / MERGED.
> **Reviewed PR head:** `01522da3985f95729ae693625f28fa5d33f2ead3`.
> **Squash merge SHA:** `14f0eca07133819cde03685d4db8905417a31b6b`.
> **Frozen L-0/L-1 schema owners:** READ-ONLY / unchanged.
> **Writer lock:** NONE.

---

# 1. Problem closed by this slice

Challenge conversations no longer depend on remembered prompts to know that completion requires Cockpit handoff.

The durable owner is the versioned ChatGPT Project instruction:

`clinical_learning/chatgpt_project_instructions_v1.txt`

Definition of done:

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

# 2. Completion state machine

```text
IN_PROGRESS
DEBRIEF_COMPLETE_HANDOFF_PENDING
HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW
HANDOFF_FAILED_MANUAL_FALLBACK_READY
```

Successful transport requires:

```text
state = pending_review
import_id = valid UUID
source_event_id = valid UUID
source_format = canonical_challenge_v1 | rich_challenge_export_v1
```

Tool invocation without a valid receipt is not success evidence.

---

# 3. Learning content contract

The Project instruction requires the final structured episode to preserve:

- initial hypothetical scenario;
- verbatim clinician responses;
- progressive disclosures;
- follow-up reasoning and final decision;
- strengths;
- needs reinforcement;
- clear errors kept distinct from missed opportunities;
- defensible disagreements;
- evidence gaps / blind spots / reasoning patterns / insights;
- learning objectives and study actions;
- knowledge-island bridge targets;
- fresh high-quality resources where appropriate;
- repeated consolidation targets.

It does not grant patient-record, Foundation, Signal, clinician-review or reference-verification authority.

---

# 4. Cockpit setup surface

Merged path:

`/static/clinical-learning/project-setup.html`

It exposes the exact Project instruction, a Copy button and truthful capability state:

```text
Project instruction = READY
Native Cockpit write tool = NOT CONNECTED
Advanced/manual fallback = AVAILABLE
```

No credential is rendered to the browser.

---

# 5. Verification

Exact reviewed release head:

`01522da3985f95729ae693625f28fa5d33f2ead3`

GitHub Actions:

- L1C challenge transport gate `34400362345` — **SUCCESS**.
- inherited L1B gate `34400362395` — **SUCCESS**.
- inherited L1 gate `34400362435` — **SUCCESS**.
- L0 contract validation step — **SUCCESS**; the overall L0 workflow fails only its intentional design-only scope check because L-1C is not an L0 design PR.

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

`CLINICAL_LEARNING_INGEST_KEY` remains unconfigured. Native zero-click transport is not active until an actual trusted write-capable ChatGPT app/action is connected and separately smoke-tested.

---

# 7. Lifecycle state

```text
IMPLEMENTED = YES
TESTED = YES
REVIEWED = PASS
MERGED = YES
DEPLOYED = NOT YET VERIFIED
PROJECT PROTOCOL = READY
SETUP SURFACE = MERGED
NATIVE WRITE TOOL = NOT CONNECTED
PRODUCTION INGEST KEY = NOT CONFIGURED
WRITER LOCK = NONE
```

Next lifecycle actions: verify normal deployment of the merge descendant, then install the Project instruction once in the dedicated ChatGPT Project. Native write activation remains a separate capability-gated integration lifecycle.