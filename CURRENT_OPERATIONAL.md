# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1B — LEARNING LOOP / AUTOMATIC INGESTION IMPLEMENTATION ACTIVE
> **Updated:** 2026-09-08 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh implementation base:** `95629fead4da5aeb1fcc0146296a9ac0f767d7ea`.
> **Active branch:** `feat/clinical-learning-l1b-learning-loop-2026-09-08`.
> **Current slice:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **Exact slice owner:** `SLICE_PLAN_CURRENT.md`.
> **ACTIVE RUNTIME WRITER/LOCK:** THIS bounded L-1B slice only.
> **L-1 production state:** PR #82 merged; reviewed runtime deployed. Product-owner smoke confirms Hub shell, Foundation, History and Due views are visible. Manual import exposed an external-schema integration defect rather than a patient/runtime data-loss defect.
> **Merge/deploy authority:** NONE for L-1B until separate review/release decision.
> **Production config/secret authority:** NONE in this writer cycle.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Production finding that activates L-1B

The clinician used the deployed Learning Hub and confirmed the four L-1 surfaces load. The Challenge Import workflow then received a rich structured challenge export that the source challenge conversation labelled `ClinicalLearningChallengeV1`, but the payload was not the frozen machine contract expected by the Hub.

Observed source/export mismatches include:

```text
schema_version = 1.0                           != clinical_learning_challenge_v1
slug challenge_id                              != UUID
session.title                                  != top-level title
topic_tags                                     != topics
deidentification                               != privacy
initial_case object                            != initial_case string
f01/la01 identifiers                           != UUIDs
clinician_reasoning_responses                  != reasoning_responses
mentor_observations grouped object             != observations[]
evidence                                       != references[]
custom gap objects/classes                     != frozen gap_class enum
spaced_repetition object                       != L-1 due semantics
```

The external export contains clinically useful learning content; the integration boundary is wrong. The remedy is not to weaken `ClinicalLearningChallengeV1` validation. L-1B adds an adapter + pending-import boundary and preserves the frozen accepted Challenge contract.

---

# 2. Product-owner requirements for L-1B

One complete Challenge learning episode must preserve:

```text
initial hypothetical scenario
→ clinician's own responses
→ progressive continuation/disclosures
→ clinician's subsequent responses/final decision
→ strengths
→ needs reinforcement / improvement
→ clear errors
→ evidence gaps / blind spots / insights
→ personalized next-study actions
→ fresh online article/webinar/course/guideline recommendations
→ repeated consolidation testing
→ transfer testing
→ bridging of knowledge islands
```

JSON is an internal contract/debug surface, not the intended clinician workflow.

---

# 3. Learning-loop invariants

```text
LEARNING RECORD != PATIENT RECORD
RAW CHAT/TRANSCRIPT != DURABLE LEARNING RECORD
EXTERNAL ASSISTANT OUTPUT != CLINICIAN REVIEW AUTHORITY
ONLINE RESOURCE SUGGESTION != IMMUTABLE CHALLENGE CONTENT
STRENGTH != ABSENCE OF GAP
NEEDS REINFORCEMENT != CLEAR ERROR
ONE SUCCESSFUL TEST != RETENTION
FOUNDATION NODE KNOWLEDGE != PROVEN BRIDGE BETWEEN NODES
DUE STATE != COMPETENCE EVIDENCE
NO COMPOSITE MASTERY SCORE
```

Frozen L-0/L-1 schema owners remain read-only. L-1B may add new additive contract owners but must not silently mutate `schemas/clinical_learning_core_v1.yaml`, `schemas/clinical_learning_l1_boundary_v1.yaml`, the manifest/design fixtures, or the Foundation Map.

---

# 4. Bounded implementation owners

Authorized mutation scope:

```text
clinical_learning/models.py
clinical_learning/contracts.py
clinical_learning/persistence.py
clinical_learning/service.py
clinical_learning/api.py
clinical_learning/ingress.py          # new
clinical_learning/learning_loop.py    # new
static/clinical-learning/index.html
static/clinical-learning/styles.css
static/clinical-learning/app.js
schemas/clinical_learning_loop_v1.yaml                 # new additive owner
CLINICAL_LEARNING_LOOP_DESIGN_V1.md                     # new design owner
test_clinical_learning_l1b_*.py                         # new focused tests
.github/workflows/clinical-learning-l1b-tests.yml       # optional/additive gate
SLICE_PLAN_CURRENT.md
CURRENT_OPERATIONAL.md
TODO.md / CLINICAL_EXCELLENCE_PLAN.md only for durable roadmap/architecture reconciliation
osteoporosis-change-log.md append-only at material milestones
```

No other runtime owner is authorized.

---

# 5. Exact next action

```text
freeze additive L-1B learning-loop contract
→ implement legacy/rich episode adapter without weakening canonical Challenge validation
→ implement Pending Imports / Inbox
→ implement Learning Loop plan, bridge targets and repeated consolidation occurrences
→ implement mutable fresh-resource recommendation overlay
→ expose narrow external-ingress seam fail-closed unless a dedicated learning-ingest key is configured
→ update clinician UI so JSON becomes Advanced/manual fallback
→ regression-test supplied-export shape using synthetic structural fixture only
→ full inherited L-1 regression + L-1B gate
→ exact-head review
→ PR / RELEASE HOLD
```

Do not merge, deploy, configure a production ingest secret, start Daily Case Review/Practice Review/Signal work, or mutate physiotherapy/CU-1/RF in this writer cycle.