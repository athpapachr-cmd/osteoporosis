# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1B Learning Loop

> **STATUS:** CLOSED / PRODUCTION-SMOKE-VERIFIED PASS
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **Implementation base:** `95629fead4da5aeb1fcc0146296a9ac0f767d7ea`.
> **PR #83:** CLOSED / MERGED — L-1B Learning Loop.
> **PR #84:** CLOSED / MERGED — bibliographic PHI false-positive hotfix.
> **L-1B runtime merge SHA:** `f15f854bbfca727356531d7b8ea896e3aedba437`.
> **Hotfix merge SHA:** `51f7d225eb960cd3a20d9a8ea7e2121fb853e056`.
> **Verified hotfix Render deploy:** `dep-dagkq56417fc73fl6il0` — LIVE.
> **PRODUCTION-SMOKE-VERIFIED:** YES / PASS.
> **Frozen L-0/L-1 schema owners:** READ-ONLY / unchanged.
> **Writer lock:** NONE.

---

# 1. Closed learning loop

```text
Challenge conversation / structured synthetic export
→ adapter + PHI/schema guard
→ Pending Imports / Inbox
→ clinician review
→ immutable Challenge
→ performance debrief
→ targeted learning objectives
→ knowledge-island bridge targets
→ fresh-resource overlay
→ repeated consolidation
   D+3 retrieval
   D+7 discrimination
   D+14 transfer
   D+30 bridge-transfer / retention
```

One successful test does not cancel later repetition. Manual JSON is Advanced/debug fallback only.

---

# 2. Final ingestion and safety boundary

The released adapter maps the observed rich Challenge export into the frozen `ClinicalLearningChallengeV1` contract without weakening canonical validation.

It preserves case facts, progressive disclosures, clinician reasoning, debrief, evidence/actions and Foundation provenance while resetting imported clinician-review/reference-verification/Signal authority.

Automatic ingress boundary:

```text
POST /clinical/learning/api/ingress/episodes
X-Learning-Ingest-Key
CLINICAL_LEARNING_INGEST_KEY
```

remains:

- explicit synthetic-only;
- verbatim clinician reasoning required for automatic rich ingress;
- pending-import creation only;
- no direct accepted-Challenge write;
- no patient record authority;
- no Foundation state authority;
- no Signal promotion/backlink authority;
- no reference-verification authority.

---

# 3. Knowledge-island bridging and repetition

`BridgeTargetV1` links 2–3 Foundation concepts with provenance to learning observations/actions. A bridge is not demonstrated by co-listing concepts; it requires later joint-application evidence and explicit clinician review.

Transparent consolidation schedule, anchored to Challenge acceptance:

```text
D+3   retrieval
D+7   discrimination / contrast
D+14  novel transfer
D+30  bridge-transfer / retention
```

Any result other than `not_assessed` requires clinician review. Same-payload retry is idempotent; a materially different second submission after completion fails closed.

No opaque adaptive scheduler or composite mastery score is introduced.

---

# 4. Fresh resources

Article/guideline/webinar/course/conference/video/podcast recommendations remain mutable overlays outside immutable Challenge hashing. Resource freshness/status changes do not create Challenge revisions and do not self-certify evidence verification.

---

# 5. Production-smoke defect and closure

The first authenticated production smoke exposed a deterministic false positive:

```text
phone_number_like_sequence_detected @ references[*].title
```

PR #84 narrowly fixed bibliographic reference titles so year/volume/page/PMID-like number patterns are not treated as generic phone sequences, while explicit phone phrases, email, identity/GeSY, DOB and postal-address detection remain active.

Regression coverage proves:

```text
bibliographic numeric citation title -> allowed
explicit phone in reference title -> blocked
email in reference title -> blocked
ordinary non-reference phone-like sequence -> blocked
```

After PR #84 was deployed live, the same authenticated rich-import workflow progressed through conversion, Server Preview, clinician observation review and save/activation flow. The clinician explicitly confirmed **PASS**.

---

# 6. Verification evidence

L-1B final release head before PR #83 merge:

`4c225784335228ccba6afd705210cd460ab43e28`

- L1B run `34311310276` — **SUCCESS**.
- inherited L1 run `34311310300` — **SUCCESS**.

Bibliographic PHI hotfix exact tested runtime head:

`94563b24fd4896c1333e0cdf3c75b2586f056adc`

- L1B run `34349129649` — **SUCCESS**.
- inherited L1 run `34349129608` — **SUCCESS**.
- L0 contract validation step — **SUCCESS**; overall L0 workflow failure was the expected design-only scope rejection of a runtime hotfix.

Verified hotfix production deploy:

```text
deploy_id = dep-dagkq56417fc73fl6il0
commit = 51f7d225eb960cd3a20d9a8ea7e2121fb853e056
status = live
trigger = new_commit
finished_at = 2026-09-09T12:13:48.830378Z
```

---

# 7. Explicit exclusions preserved

```text
NO patient record mutation
NO raw ChatGPT/Heidi transcript persistence
NO Daily Real-Case Review
NO Practice Review AI runtime
NO Signal promotion/backlink authority
NO Foundation state mutation from Challenge result alone
NO composite mastery/excellence score
NO opaque adaptive scheduler
NO direct accepted-Challenge write from external assistant
NO physiotherapy/CU-1/RF mutation
```

---

# 8. Final lifecycle state

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

The next Clinical Learning work, if selected, is not L-1B remediation. Automatic ChatGPT → Cockpit transport activation remains a separate integration slice because it requires production secret/config and connector wiring authority.

Docs-only closeout descendants may auto-deploy under Render `autoDeploy=yes`; they do not alter the verified runtime and do not require recursive smoke.