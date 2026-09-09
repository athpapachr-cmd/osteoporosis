# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1B — MERGED / DEPLOYED / PRODUCTION SMOKE PENDING
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **PR:** #83 — CLOSED / MERGED.
> **Final reviewed PR head:** `4c225784335228ccba6afd705210cd460ab43e28`.
> **Runtime squash-merge SHA:** `f15f854bbfca727356531d7b8ea896e3aedba437`.
> **Verified live deploy SHA:** `0d86cd3bd5b61a23a0d7a73da29559e35cd335e5`.
> **Render deploy:** `dep-daghkeek1f9s73ah0fn0` — LIVE.
> **Writer lock:** NONE.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Released L-1B behavior

```text
structured synthetic Challenge episode
→ adapter + PHI/schema guard
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

Manual JSON is an Advanced/debug fallback rather than the intended default workflow.

---

# 2. Automatic-ingress boundary

Merged code exposes:

```text
POST /clinical/learning/api/ingress/episodes
X-Learning-Ingest-Key
CLINICAL_LEARNING_INGEST_KEY
```

Permanent boundaries:

- explicit synthetic learning only;
- automatic rich ingress requires verbatim clinician reasoning;
- external ingress creates only `pending_review` candidates;
- no direct clinician-reviewed Challenge creation;
- no patient/encounter/lab writes;
- no reference-verification authority;
- no Foundation-state authority;
- no Signal promotion/backlink authority;
- no reuse of `CLINICAL_DATA_KEY`.

**Production ingest key is not configured by this release. ChatGPT/plugin automatic connection is not activated by this release.**

---

# 3. Learning-integrity invariants

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

A bridge is demonstrated only by later joint-application evidence. Any consolidation result other than `not_assessed` requires explicit clinician review. Same-payload retry is idempotent; materially different resubmission fails closed.

---

# 4. Verification evidence

Final reviewed PR head:

`4c225784335228ccba6afd705210cd460ab43e28`

Exact-head gates:

- Clinical Learning L1B regression gate run `34311310276` — **SUCCESS**.
- Clinical Learning L1 regression gate run `34311310300` — **SUCCESS**.

Squash merge:

`f15f854bbfca727356531d7b8ea896e3aedba437`

Verified Render auto-deploy:

```text
deploy_id = dep-daghkeek1f9s73ah0fn0
commit = 0d86cd3bd5b61a23a0d7a73da29559e35cd335e5
status = live
trigger = new_commit
finished_at = 2026-09-09T08:36:42.393912Z
```

The live SHA is a docs-only descendant of the reviewed runtime merge. Runtime code remains the reviewed PR #83 tree.

---

# 5. Lifecycle state

```text
IMPLEMENTED = YES
TESTED = YES
FOCUSED REVIEW = PASS
MERGED = YES
DEPLOYED = YES
PRODUCTION-SMOKE-VERIFIED = NO
PRODUCTION INGEST KEY = NOT CONFIGURED
CHATGPT AUTOMATIC CONNECTION = NOT ACTIVATED
WRITER LOCK = NONE
```

Exact next action:

```text
authenticated production smoke of Inbox / Learning Loop / Due / Advanced fallback
→ STOP
```

Production ingest-key creation and ChatGPT/plugin wiring remain a separate explicit integration decision.