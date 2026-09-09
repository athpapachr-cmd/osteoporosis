# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1B — IMPLEMENTED / TESTED / FOCUSED REVIEW PASS / RELEASE HOLD
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Implementation base:** `95629fead4da5aeb1fcc0146296a9ac0f767d7ea`.
> **Branch:** `feat/clinical-learning-l1b-learning-loop-2026-09-08`.
> **Slice:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **Reviewed runtime candidate:** `54e881cc512b0de9ad1e9d95a8caffbe8c5d8777`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Merge/deploy authority:** NONE until separate product-owner release decision.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Production finding closed by L-1B

The deployed L-1 Hub loaded correctly but exposed a real integration problem: the Challenge conversation produced a clinically rich structured export whose shape did not match the frozen `ClinicalLearningChallengeV1` machine contract. Manual import therefore appeared red/empty despite useful learning content.

L-1B fixes the boundary without weakening the frozen Challenge contract.

---

# 2. Delivered L-1B product behavior

```text
structured synthetic Challenge episode
→ bounded adapter + PHI/schema guard
→ Pending Imports / Inbox
→ clinician review / Accept-Modify-Dismiss
→ immutable ClinicalLearningChallengeV1
→ Learning Loop
   strengths
   needs reinforcement
   clear errors
   evidence gaps / blind spots / insights
   targeted study actions
   knowledge-island bridge targets
   fresh-resource overlay
→ repeated consolidation
   D+3 retrieval
   D+7 discrimination
   D+14 transfer
   D+30 bridge-transfer / retention
```

Manual JSON is now an Advanced/debug fallback rather than the intended default workflow.

---

# 3. Automatic-ingress safety boundary

The code now exposes a narrow synthetic-learning ingress seam using a dedicated learning-only credential:

```text
CLINICAL_LEARNING_INGEST_KEY
X-Learning-Ingest-Key
```

It:

- fails closed when the dedicated key is absent;
- does not reuse `CLINICAL_DATA_KEY`;
- accepts only explicit synthetic learning episodes;
- requires verbatim clinician reasoning for automatic rich-export ingress;
- creates only a pending import candidate;
- cannot directly create a clinician-reviewed Challenge;
- cannot verify references, mutate Foundation state, promote Signals or write patient data.

Production secret configuration and actual ChatGPT/plugin wiring are **not** part of this release candidate and remain a later explicit integration action.

---

# 4. Learning integrity

Permanent L-1B invariants:

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

A bridge is demonstrated only by later joint-application evidence. A consolidation result such as `retained` cannot be recorded unless explicitly clinician-reviewed. Same-payload retry of a completed occurrence is idempotent; a materially different second attempt fails closed.

---

# 5. Verification evidence

Exact reviewed runtime candidate:

`54e881cc512b0de9ad1e9d95a8caffbe8c5d8777`

Exact-head gates:

- Clinical Learning L1B regression gate run `34311161485` — **SUCCESS**.
- Clinical Learning L1 regression gate run `34311161517` — **SUCCESS**.

Passed coverage includes:

- Python/browser syntax;
- rich-export → frozen Challenge adapter;
- deterministic IDs/idempotent pending import;
- raw external payload non-persistence;
- PHI guard;
- dedicated external-ingress key;
- synthetic-only automatic ingress;
- verbatim-reasoning requirement for automatic rich ingress;
- clinician-reviewed retention results;
- retry-safe consolidation attempts;
- knowledge-island bridge provenance/evidence semantics;
- four repeated consolidation occurrences;
- mutable resource overlay outside immutable Challenge hash;
- deletion cleanup;
- inherited L-1/L-0 regression;
- frozen-owner guard;
- adjacent-owner scope guard;
- diff hygiene.

Focused full-diff review against base found no remaining material blocker after the closure regressions.

---

# 6. Release state

```text
IMPLEMENTED = YES
TESTED = YES
FOCUSED REVIEW = PASS
PR = NEXT / DRAFT
MERGED = NO
DEPLOYED = NO
PRODUCTION INGEST KEY = NOT CONFIGURED BY THIS SLICE
CHATGPT AUTOMATIC CONNECTION = NOT ACTIVATED BY THIS SLICE
```

The next allowed lifecycle action is Draft PR / RELEASE HOLD. No merge, deploy or production secret change is authorized by this canonical state.