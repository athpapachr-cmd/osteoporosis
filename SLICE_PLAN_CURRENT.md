# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1B Learning Loop

> **STATUS:** IMPLEMENTED / TESTED / FOCUSED REVIEW PASS / MERGED / DEPLOYED / PRODUCTION SMOKE PENDING
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **Implementation base:** `95629fead4da5aeb1fcc0146296a9ac0f767d7ea`.
> **Final reviewed PR head:** `4c225784335228ccba6afd705210cd460ab43e28`.
> **PR:** #83 — CLOSED / MERGED.
> **Runtime squash-merge SHA:** `f15f854bbfca727356531d7b8ea896e3aedba437`.
> **Verified live deploy SHA:** `0d86cd3bd5b61a23a0d7a73da29559e35cd335e5`.
> **Render deploy:** `dep-daghkeek1f9s73ah0fn0` — LIVE.
> **Frozen L-0/L-1 schema owners:** READ-ONLY / unchanged.
> **Writer lock:** NONE.

---

# 1. Released learning loop

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

# 2. Ingestion + learning integrity

The released adapter maps the observed rich Challenge export into the frozen `ClinicalLearningChallengeV1` contract without weakening canonical validation.

It preserves case facts, progressive disclosures, clinician reasoning, debrief, evidence/actions and Foundation provenance while resetting imported clinician-review/reference-verification/Signal authority.

Automatic ingress:

```text
POST /clinical/learning/api/ingress/episodes
X-Learning-Ingest-Key
CLINICAL_LEARNING_INGEST_KEY
```

remains explicit synthetic-only, requires verbatim clinician reasoning and creates only `pending_review` candidates.

`needs reinforcement` remains distinct from `clear_error`. A knowledge bridge is demonstrated only after later clinician-reviewed joint-application evidence. Any consolidation result other than `not_assessed` requires explicit clinician review.

---

# 3. Repeated consolidation

Transparent schedule anchored to Challenge acceptance:

```text
D+3   retrieval
D+7   discrimination / contrast
D+14  novel transfer
D+30  bridge-transfer / retention
```

Same-payload retry is idempotent. A materially different second submission after completion fails closed. No opaque adaptive scheduler or composite mastery score is introduced.

---

# 4. Fresh resources

Article/guideline/webinar/course/conference/video/podcast recommendations are mutable overlays outside immutable Challenge hashing. Resource freshness/status changes do not create Challenge revisions and do not self-certify evidence verification.

---

# 5. Persistence boundaries

Learning-only persistence owners:

```text
clinical_learning_pending_imports
clinical_learning_loop_plans
clinical_learning_consolidation_attempts
clinical_learning_resource_recommendations
clinical_learning_due_items: consolidation_test occurrences
```

No patient encounter/lab/RF/transcript write path is introduced. Challenge deletion cleans linked loop/attempt/resource/due content.

---

# 6. Verification / release evidence

Final exact-head gates on `4c225784335228ccba6afd705210cd460ab43e28`:

- L1B run `34311310276` — **SUCCESS**.
- inherited L1 run `34311310300` — **SUCCESS**.

Squash merge:

`f15f854bbfca727356531d7b8ea896e3aedba437`

Verified Render auto-deploy:

```text
deploy_id = dep-daghkeek1f9s73ah0fn0
commit = 0d86cd3bd5b61a23a0d7a73da29559e35cd335e5
status = live
trigger = new_commit
```

The deployed SHA is a docs-only descendant of the reviewed runtime merge, so the deployed runtime tree is the reviewed PR #83 runtime.

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
NO production ingest-secret creation by this release
NO direct accepted-Challenge write from external assistant
NO physiotherapy/CU-1/RF mutation
```

---

# 8. Lifecycle state

```text
IMPLEMENTED = YES
TESTED = YES
FOCUSED REVIEW = PASS
MERGED = YES
DEPLOYED = YES
PRODUCTION-SMOKE-VERIFIED = NO
PRODUCTION INGEST KEY = NOT CONFIGURED
CHATGPT AUTOMATIC CONNECTION = NOT ACTIVATED
```

Next lifecycle action: authenticated production smoke of Inbox / Learning Loop / Due / Advanced fallback. Automatic ChatGPT ingress activation remains a separate explicit integration decision.