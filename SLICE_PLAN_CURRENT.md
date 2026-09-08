# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1B Learning Loop

> **STATUS:** DESIGN + IMPLEMENTATION ACTIVE
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **Base:** `95629fead4da5aeb1fcc0146296a9ac0f767d7ea`.
> **Branch:** `feat/clinical-learning-l1b-learning-loop-2026-09-08`.
> **Prior L-1 release:** PR #82 merged; runtime deployed.
> **Merge/deploy authority:** NONE for this slice.
> **Frozen L-0/L-1 schema owners:** READ-ONLY.

---

# 1. Problem

The deployed L-1 Hub proves protected Challenge/History/Foundation/Due storage, but the real clinician workflow exposed two product gaps:

1. **Schema bridge gap:** the challenge conversation exports a rich learning artifact in a human-designed shape that is not the frozen `ClinicalLearningChallengeV1` machine contract. The Hub correctly rejects it, producing a useless red/empty import experience.
2. **Learning-loop gap:** a saved Challenge is not enough. The clinician requires deliberate reinforcement, repeated consolidation, transfer, knowledge-island bridging and fresh learning resources.

The new design must preserve L-1 safety semantics rather than relaxing validation.

---

# 2. Objective

Build the first closed Challenge learning loop:

```text
CHALLENGE CONVERSATION / STRUCTURED EXPORT
        ↓
EPHEMERAL ADAPTER + PHI/SCHEMA GUARD
        ↓
PENDING IMPORT / INBOX
        ↓
CLINICIAN PREVIEW + ACCEPT/MODIFY/DISMISS
        ↓
IMMUTABLE ClinicalLearningChallengeV1
        ↓
PERSONALIZED LEARNING LOOP
  strengths / errors / reinforcement
  bridge targets between knowledge islands
  study actions
  fresh online resources
        ↓
REPEATED CONSOLIDATION
  retrieval → discrimination → transfer → bridge transfer
        ↓
RETENTION EVIDENCE / NEEDS REINFORCEMENT
```

Manual raw JSON remains only an Advanced/debug fallback.

---

# 3. In-scope product behavior

## 3.1 Rich episode ingestion

The adapter must accept:

- the canonical `ClinicalLearningChallengeV1` shape; and
- a bounded rich-export shape compatible with the observed challenge export (`schema_type=ClinicalLearningChallengeV1`, `schema_version=1.0`, `session`, `topic_tags`, grouped `mentor_observations`, etc.).

It must convert the rich export into the existing accepted Challenge machine contract by:

- generating deterministic UUIDs for source-local IDs;
- mapping title/session/date/topics/privacy;
- converting initial/disclosure facts into the Fact Ledger;
- converting clinician reasoning into ordered responses;
- flattening strengths, errors, disagreements, evidence gaps, blind spots, reasoning patterns and insights into `LearningObservationV1` candidates;
- mapping evidence entries into `LearningReferenceV1` candidates;
- mapping study actions into canonical `LearningActionV1` candidates;
- mapping known Foundation aliases to the frozen registry;
- never carrying imported review/reference/Signal authority.

Raw external payload is validated and processed ephemerally. Pending persistence stores only normalized structured learning content plus bounded loop/resource metadata.

## 3.2 Pending Imports / Inbox

Add a protected Inbox where external/import candidates appear before accepted Challenge persistence.

Required states:

```text
pending_review
accepted
rejected
invalid
```

A pending import cannot create Foundation state, Signal authority, verified references or a clinician-reviewed Challenge by itself.

## 3.3 Performance debrief

The accepted episode must visibly distinguish:

```text
STRENGTHS
NEEDS REINFORCEMENT / IMPROVEMENT
CLEAR ERRORS
DEFENSIBLE DISAGREEMENTS
EVIDENCE GAPS / BLIND SPOTS
REASONING PATTERNS / CLINICAL INSIGHTS
```

`needs reinforcement` is not silently relabelled `clear_error`.

## 3.4 Knowledge-island bridging

A **knowledge island** is a Foundation concept/domain for which evidence exists in isolation but joint application with another relevant concept has not yet been demonstrated.

A **BridgeTargetV1** identifies 2–3 Foundation nodes that should be tested together, with a human-readable rationale and provenance to the source Challenge/observations/actions.

Bridge evidence requires a later transfer task that actually uses the target nodes together. Merely listing related nodes does not prove a bridge.

No numerical bridge/mastery score is introduced.

## 3.5 Repeated consolidation

One midweek test is insufficient. Every meaningful target may receive multiple scheduled occurrences.

Transparent first-runtime default sequence:

```text
Occurrence 1 — retrieval               ~3 days after acceptance
Occurrence 2 — discrimination/contrast ~7 days after acceptance
Occurrence 3 — transfer                ~14 days after acceptance
Occurrence 4 — bridge/retention        ~30 days after acceptance
```

These are product defaults, not claims of an individually optimal neuroscience cadence. They are configurable in code/contract and must be visible to the clinician.

The first slice does **not** implement an opaque adaptive spacing algorithm. Results may be recorded as:

```text
retained
partially_retained
not_retained
improved_beyond_original
not_assessed
```

Base repetition remains scheduled even after an early successful attempt. This satisfies the product-owner requirement that consolidation itself requires repetition.

## 3.6 Fresh online learning resources

Resource recommendations are a mutable external overlay, never part of immutable Challenge hashing.

Supported kinds:

```text
article
guideline
webinar
course
conference_session
video
podcast
other
```

Each recommendation carries provider/title/URL, why it is relevant to this clinician's gap, target Foundation nodes/gap classes, access state when known, and `checked_at` freshness.

The source assistant may supply fresh recommendations after web research. The server never upgrades a resource to verified evidence merely because it was suggested.

## 3.7 Narrow automatic-ingress seam

Implement a dedicated endpoint for structured learning episodes using a **learning-only key** (`CLINICAL_LEARNING_INGEST_KEY` / `X-Learning-Ingest-Key`).

Rules:

- fail closed with 503 when the dedicated key is absent;
- constant-time key compare;
- no reuse of broad patient/clinical write authority;
- initial external ingress accepts **synthetic Challenge learning only**;
- still runs PHI guard and unknown-field/adapter validation;
- creates only a pending import candidate;
- does not save an accepted Challenge directly.

Production secret configuration and ChatGPT/plugin connection are separate release/integration actions, not authorized in this implementation slice.

---

# 4. Additive data contracts

New owner: `schemas/clinical_learning_loop_v1.yaml`.

Objects:

```text
LearningEpisodeIngressV1
PendingLearningImportV1
LearningLoopPlanV1
LearningObjectiveV1
BridgeTargetV1
ConsolidationOccurrenceV1
ConsolidationAttemptV1
LearningResourceRecommendationV1
```

Existing frozen `ClinicalLearningChallengeV1` remains the immutable accepted Challenge owner.

---

# 5. Persistence

Add dedicated tables only:

```text
clinical_learning_pending_imports
clinical_learning_loop_plans
clinical_learning_consolidation_attempts
clinical_learning_resource_recommendations
```

Reuse `clinical_learning_due_items` for `consolidation_test` occurrences under this additive boundary.

No patient encounter/lab/RF/transcript table is read or written.

---

# 6. Clinician UI

Add/reshape Learning Hub surfaces:

1. **Inbox / Pending Imports** — source episode summary, adapter warnings, review/open/reject.
2. **Challenge Review** — case timeline + own responses + progressive disclosures + debrief dispositions.
3. **Learning Loop** — strengths, reinforcement, errors, bridge targets, study actions, fresh resources, repetition timeline.
4. **Consolidation Test** — prompt/response/result history and next occurrence.
5. **Advanced JSON** — the current raw JSON paste workflow, explicitly secondary.

The default clinician experience must not require understanding the machine schema.

---

# 7. Evidence-informed learning mechanics

The design uses repeated retrieval and distributed practice because health-professions education literature supports benefits of retrieval/distributed practice and spaced digital education. This slice does not claim one universally optimal spacing interval and does not invent a composite mastery score.

The transfer/bridge layer uses novel-case application rather than verbatim replay of the original case so that retention evidence is not merely recognition of a memorized scenario.

---

# 8. Explicit exclusions

```text
NO patient record mutation
NO raw ChatGPT/Heidi transcript persistence
NO Daily Real-Case Review
NO Practice Review AI runtime
NO Signal promotion/backlink authority
NO Foundation state mutation from Challenge result alone
NO composite mastery/excellence score
NO opaque adaptive scheduler
NO production secret/config change
NO automatic direct accepted-Challenge write from external assistant
NO physiotherapy/CU-1/RF mutation
```

---

# 9. Acceptance evidence

Before L-1B can be called TESTED:

- supplied rich-export structural shape converts into valid canonical Challenge without relaxing canonical validation;
- deterministic source-local → UUID mapping is stable/idempotent;
- Foundation aliases resolve or fail closed with explicit warnings;
- raw external unknown fields never persist blindly;
- PHI guard applies before pending persistence;
- imported review/reference/Signal authority is reset;
- Pending Inbox create/list/reject/accept works;
- accepting pending import still requires clinician observation dispositions/confirmation;
- entire scenario/reasoning/disclosure sequence is inspectable;
- strengths/reinforcement/errors remain distinct;
- bridge targets preserve 2–3 Foundation nodes and provenance;
- repeated consolidation creates at least four transparent occurrences when eligible;
- successful early consolidation does not silently cancel all later repetition;
- consolidation attempt history is immutable;
- fresh resource overlay is mutable and excluded from Challenge hash;
- resource freshness/status changes do not create Challenge revisions;
- external ingress fails closed without dedicated learning key and accepts synthetic-only candidates;
- L-1 inherited regression remains green;
- no patient/transcript/Signal/RF/physio mutation;
- browser JavaScript and new clinician UI contract tests pass.

---

# 10. REPLAN triggers

STOP and replan if implementation requires any of:

- weakening the frozen Challenge schema instead of adapting external input;
- storing raw chat/transcript content as durable learning data;
- direct external creation of clinician-reviewed Challenges;
- using `CLINICAL_DATA_KEY` as the external learning-ingest credential;
- patient/encounter linkage in this slice;
- autonomous Foundation mastery/state mutation;
- autonomous Signal promotion;
- hidden/opaque adaptive scheduling;
- production secret/config mutation without separate authority.

---

# 11. Release path

```text
contract/design
→ runtime implementation
→ focused + inherited regression gate
→ exact-head review
→ bounded remediation if needed
→ PR OPEN / RELEASE HOLD
→ separate product-owner merge authority
→ normal Render auto-deploy
→ authenticated production smoke
→ only then production connector/key activation as a separate integration decision
```
