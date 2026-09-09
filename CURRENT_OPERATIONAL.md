# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1B — CLOSED / PRODUCTION-SMOKE-VERIFIED PASS
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **PR #83:** CLOSED / MERGED — Learning Loop release.
> **PR #84:** CLOSED / MERGED — bibliographic PHI false-positive hotfix.
> **L-1B runtime merge SHA:** `f15f854bbfca727356531d7b8ea896e3aedba437`.
> **Hotfix merge SHA:** `51f7d225eb960cd3a20d9a8ea7e2121fb853e056`.
> **Verified hotfix Render deploy:** `dep-dagkq56417fc73fl6il0` — LIVE.
> **PRODUCTION-SMOKE-VERIFIED:** YES / PASS.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE after closeout.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Final released learning flow

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

Manual JSON is an Advanced/debug fallback rather than the default clinician workflow.

---

# 2. Production-smoke evidence

The authenticated production smoke used the previously problematic rich Challenge export.

Observed production path:

```text
Advanced rich import
→ rich-export adapter
→ bibliographic references accepted after bounded PHI hotfix
→ Server Preview rendered
→ clinician observation review gate enforced
→ reviewed Challenge save / Learning Loop activation path completed
```

The clinician explicitly confirmed the smoke as **PASS** on 2026-09-09.

The initial smoke correctly exposed a false positive in `references[*].title`; PR #84 fixed only the generic numeric phone heuristic for bibliographic titles while preserving explicit phone/email/identity/DOB/address protections.

---

# 3. Final verification evidence

L-1B final release head before PR #83 merge:

`4c225784335228ccba6afd705210cd460ab43e28`

- L1B gate `34311310276` — **SUCCESS**.
- inherited L1 gate `34311310300` — **SUCCESS**.

Bibliographic PHI hotfix exact tested runtime head:

`94563b24fd4896c1333e0cdf3c75b2586f056adc`

- L1B gate `34349129649` — **SUCCESS**.
- inherited L1 gate `34349129608` — **SUCCESS**.
- L0 contract validation step — **SUCCESS**; its overall workflow failure was the expected design-only scope guard for a runtime hotfix.

Verified hotfix production deploy:

```text
deploy_id = dep-dagkq56417fc73fl6il0
commit = 51f7d225eb960cd3a20d9a8ea7e2121fb853e056
status = live
trigger = new_commit
finished_at = 2026-09-09T12:13:48.830378Z
```

---

# 4. Permanent learning-integrity invariants

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

Automatic rich ingress remains explicit synthetic-only and requires verbatim clinician reasoning. External ingress creates only `pending_review`; it cannot directly create accepted Challenges, verify references, mutate Foundation state, promote Signals or write patient data.

---

# 5. Closed lifecycle state

```text
IMPLEMENTED = YES
TESTED = YES
FOCUSED REVIEW = PASS
MERGED = YES
DEPLOYED = YES
PRODUCTION-SMOKE-VERIFIED = YES / PASS
L-1B = CLOSED
PRODUCTION INGEST KEY = NOT CONFIGURED
CHATGPT AUTOMATIC CONNECTION = NOT ACTIVATED
WRITER LOCK = NONE
```

The automatic ChatGPT → Cockpit transport remains a **separate integration action** because it requires production ingest-key creation / connector wiring. Closing L-1B does not authorize that secret/config change.

Any docs-only closeout descendants may auto-deploy because Render tracks `main`; they do not alter the verified runtime behavior and do not require recursive production smoke.