# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1C — MERGED / DEPLOYED
> **Updated:** 2026-09-10 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CORE-LEARNING-HUB-L1C-CHALLENGE-COMPLETION-TRANSPORT-2026-09-09`.
> **PR #85:** CLOSED / MERGED.
> **Reviewed PR head:** `01522da3985f95729ae693625f28fa5d33f2ead3`.
> **Squash merge SHA:** `14f0eca07133819cde03685d4db8905417a31b6b`.
> **Production-verified deploy commit:** `6ad977993fa1832f57d0e59ec41e7cd9d154ead3`.
> **Render deploy:** `dep-dags2tu7bikc73dq15t0` — LIVE.
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

The deployed runtime includes:

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

Production deployment evidence:

```text
Render service = srv-d5qfk31r0fns73di596g
deploy = dep-dags2tu7bikc73dq15t0
commit = 6ad977993fa1832f57d0e59ec41e7cd9d154ead3
trigger = new_commit
status = live
started = 2026-09-09T20:28:39Z
finished = 2026-09-09T20:30:26Z
```

The deployed commit is a docs-only descendant of the L-1C runtime merge SHA and contains the reviewed L-1C runtime unchanged.

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
L-1C DEPLOYED = YES
PROJECT COMPLETION PROTOCOL = READY
COCKPIT SETUP PAGE = DEPLOYED
NATIVE WRITE TOOL = NOT CONNECTED
PRODUCTION INGEST KEY = NOT CONFIGURED
WRITER LOCK = NONE
```

Exact next action: install the Project instruction once in the dedicated ChatGPT Project. Native zero-click ChatGPT → Cockpit write remains a separate capability-gated integration step. No production ingest secret should be configured until a concrete trusted write-capable consumer is available and separately smoke-tested.

Later docs-only canonical descendants may auto-deploy because Render `autoDeploy=yes`; they do not alter the reviewed L-1C runtime and do not require recursive runtime smoke.