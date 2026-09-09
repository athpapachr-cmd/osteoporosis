# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1C — MERGED / DEPLOY VERIFICATION PENDING
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CORE-LEARNING-HUB-L1C-CHALLENGE-COMPLETION-TRANSPORT-2026-09-09`.
> **PR #85:** CLOSED / MERGED.
> **Reviewed PR head:** `01522da3985f95729ae693625f28fa5d33f2ead3`.
> **Squash merge SHA:** `14f0eca07133819cde03685d4db8905417a31b6b`.
> **ACTIVE RUNTIME/DESIGN WRITER:** NONE.
> **Production config/secret authority exercised:** NO.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Released L-1C behavior

Challenge conversations now have a versioned ChatGPT Project completion protocol:

`clinical_learning/chatgpt_project_instructions_v1.txt`

Definition of done:

```text
final debrief
+ structured learning episode
+ Cockpit handoff attempt
+ returned receipt or explicit transport failure
```

No receipt means no claim that the Cockpit was updated.

Completion states:

```text
IN_PROGRESS
DEBRIEF_COMPLETE_HANDOFF_PENDING
HANDOFF_SUCCEEDED_PENDING_COCKPIT_REVIEW
HANDOFF_FAILED_MANUAL_FALLBACK_READY
```

Successful handoff evidence requires a `pending_review` receipt with valid `import_id`, `source_event_id` and recognized `source_format`. `pending_review` is not clinician acceptance or verification authority.

---

# 2. Delivered setup surface

The merged runtime includes:

`/static/clinical-learning/project-setup.html`

It exposes the exact Project instruction with a Copy button and truthfully reports:

```text
Project instruction = READY
Native Cockpit write tool = NOT CONNECTED
Advanced/manual fallback = AVAILABLE
```

No secret is rendered or stored in the browser.

---

# 3. Verification evidence

Exact release head:

`01522da3985f95729ae693625f28fa5d33f2ead3`

- Clinical Learning L1C challenge transport gate `34400362345` — **SUCCESS**.
- Clinical Learning L1B regression gate `34400362395` — **SUCCESS**.
- Clinical Learning L1 regression gate `34400362435` — **SUCCESS**.
- L0 contract validation step — **SUCCESS**; overall L0 workflow failure is the expected design-only scope rejection for a non-L0 runtime/integration PR.

---

# 4. Security / privacy boundary preserved

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

`CLINICAL_LEARNING_INGEST_KEY` remains **NOT CONFIGURED** because no trusted write-capable ChatGPT consumer is connected yet.

---

# 5. Lifecycle state

```text
L-1C IMPLEMENTED = YES
L-1C TESTED = YES
L-1C REVIEWED = PASS
L-1C MERGED = YES
L-1C DEPLOYED = NOT YET VERIFIED
PROJECT COMPLETION PROTOCOL = READY
COCKPIT SETUP PAGE = MERGED
NATIVE WRITE TOOL = NOT CONNECTED
PRODUCTION INGEST KEY = NOT CONFIGURED
WRITER LOCK = NONE
```

Exact next actions are limited to normal deployment verification and one-time ChatGPT Project instruction installation. Native zero-click ChatGPT → Cockpit write remains a separate capability-gated integration step. No manual deploy or production secret mutation is authorized by this reconciliation.