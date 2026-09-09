# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1B — MERGED / DEPLOY VERIFICATION PENDING
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **PR:** #83 — CLOSED / MERGED.
> **Reviewed PR head:** `4c225784335228ccba6afd705210cd460ab43e28`.
> **Runtime squash-merge SHA:** `f15f854bbfca727356531d7b8ea896e3aedba437`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE after this reconciliation commit.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. L-1B product state

The L-1 production finding is closed at code/release level: the source Challenge conversation can now be adapted into the frozen learning contract without weakening `ClinicalLearningChallengeV1`, and the clinician-facing workflow no longer depends on understanding machine JSON.

Delivered flow:

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

Manual JSON remains an Advanced/debug fallback.

---

# 2. Automatic-ingress boundary

The merged code exposes a narrow learning-only ingress seam:

```text
POST /clinical/learning/api/ingress/episodes
X-Learning-Ingest-Key
CLINICAL_LEARNING_INGEST_KEY
```

Permanent boundaries:

- explicit synthetic learning only;
- automatic rich ingress requires verbatim clinician reasoning;
- external ingress creates only `pending_review` candidates;
- no direct creation of clinician-reviewed Challenges;
- no patient/encounter/lab writes;
- no reference verification authority;
- no Foundation-state authority;
- no Signal promotion/backlink authority;
- no reuse of `CLINICAL_DATA_KEY`.

**The production ingest secret is not configured by this merge. The ChatGPT/plugin connection is not activated by this merge.** Those remain separate explicit integration actions.

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

A bridge is demonstrated only by later joint-application evidence. Any consolidation result other than `not_assessed` requires explicit clinician review. Same-payload retry of a completed occurrence is idempotent; materially different resubmission fails closed.

---

# 4. Release evidence

Final reviewed PR head:

`4c225784335228ccba6afd705210cd460ab43e28`

Exact-head gates:

- Clinical Learning L1B regression gate run `34311310276` — **SUCCESS**.
- Clinical Learning L1 regression gate run `34311310300` — **SUCCESS**.

Squash merge:

`f15f854bbfca727356531d7b8ea896e3aedba437`

PR #83 is CLOSED / MERGED. Frozen L-0/L-1 schema owners and adjacent physiotherapy/CU-1/RF owners were not mutated.

---

# 5. Lifecycle state

```text
IMPLEMENTED = YES
TESTED = YES
FOCUSED REVIEW = PASS
MERGED = YES
DEPLOYED = PENDING VERIFICATION
PRODUCTION-SMOKE-VERIFIED = NO
PRODUCTION INGEST KEY = NOT CONFIGURED
CHATGPT AUTOMATIC CONNECTION = NOT ACTIVATED
WRITER LOCK = NONE
```

Exact next action:

```text
verify normal Render auto-deploy of the L-1B runtime tree
→ authenticated production smoke of Inbox / Learning Loop / Due / Advanced fallback
→ STOP
```

Production ingest-key creation and ChatGPT/plugin wiring require a separate explicit integration decision after deployment verification. No manual duplicate deploy is authorized.