# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1B Learning Loop

> **STATUS:** IMPLEMENTED / TESTED / FOCUSED REVIEW PASS / MERGED / DEPLOY VERIFICATION PENDING
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **Implementation base:** `95629fead4da5aeb1fcc0146296a9ac0f767d7ea`.
> **Branch:** `feat/clinical-learning-l1b-learning-loop-2026-09-08`.
> **Final reviewed PR head:** `4c225784335228ccba6afd705210cd460ab43e28`.
> **PR:** #83 — CLOSED / MERGED.
> **Runtime squash-merge SHA:** `f15f854bbfca727356531d7b8ea896e3aedba437`.
> **Frozen L-0/L-1 schema owners:** READ-ONLY / unchanged.
> **Writer lock:** NONE.

---

# 1. Problem solved

The deployed L-1 Hub correctly rejected a rich Challenge export because the source conversation was producing a human-oriented learning artifact rather than the frozen `ClinicalLearningChallengeV1` machine contract. That made import appear red/empty and forced manual JSON handling.

L-1B adds the integration layer and closes the first Challenge learning loop without weakening the accepted Challenge contract.

---

# 2. Released L-1B behavior

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

One successful test does not cancel later repetition. Manual JSON is an Advanced/debug fallback rather than the intended default workflow.

---

# 3. Ingestion + Inbox

Implemented support for:

- canonical `ClinicalLearningChallengeV1` input; and
- the observed bounded rich-export shape.

The adapter:

- generates deterministic UUIDs for source-local IDs;
- maps case facts, progressive disclosures, clinician reasoning, debrief, evidence and actions;
- maps known Foundation aliases conservatively;
- resets imported clinician-review/reference-verification/Signal authority;
- discards raw source payload after normalization;
- keeps legacy summary-only exports manually salvageable with warning;
- requires verbatim clinician response text for new automatic rich ingress.

Automatic rich ingestion is explicit synthetic-only and initial-revision-only in L-1B. New external ingestion creates `pending_review` only. Accepted Challenge persistence still goes through the existing clinician review flow.

---

# 4. Learning debrief and knowledge-island bridging

The normalized episode and clinician UI keep these distinct:

```text
STRENGTHS
NEEDS REINFORCEMENT / IMPROVEMENT
CLEAR ERRORS
DEFENSIBLE DISAGREEMENTS
EVIDENCE GAPS / BLIND SPOTS
REASONING PATTERNS / CLINICAL INSIGHTS
```

`BridgeTargetV1` links 2–3 Foundation nodes with provenance to source learning observations/actions. Co-listing concepts does not prove a bridge. Bridge state changes only after later joint-application evidence, and only an explicitly clinician-reviewed bridge-transfer result can mark it demonstrated or needing reinforcement.

---

# 5. Repeated consolidation

Transparent initial sequence anchored to actual Challenge acceptance:

```text
D+3   retrieval
D+7   discrimination / contrast
D+14  novel transfer
D+30  bridge-transfer / retention
```

Any result other than `not_assessed` requires clinician review. Same-payload retry is idempotent; a materially different second submission after completion fails closed.

No opaque adaptive scheduler or composite mastery score is introduced.

---

# 6. Fresh resources

Resource recommendations are mutable overlays outside immutable Challenge hashing. They may represent article, guideline, webinar, course, conference session, video, podcast or other learning opportunities.

Changing recommendation freshness/status does not create a Challenge revision and does not self-certify reference verification.

---

# 7. Automatic-ingress seam

Merged boundary:

```text
POST /clinical/learning/api/ingress/episodes
X-Learning-Ingest-Key
CLINICAL_LEARNING_INGEST_KEY
```

It fails closed when not configured, does not reuse `CLINICAL_DATA_KEY`, accepts explicit synthetic learning only and can create only a pending import.

It cannot directly write patient data, accepted Challenges, Foundation state, Signals or verified evidence.

**Production key creation and ChatGPT/plugin wiring are not activated by PR #83.**

---

# 8. Persistence and deletion hygiene

Added learning-only owners:

```text
clinical_learning_pending_imports
clinical_learning_loop_plans
clinical_learning_consolidation_attempts
clinical_learning_resource_recommendations
```

`clinical_learning_due_items` carries `consolidation_test` occurrences. Challenge deletion cleans linked loop/attempt/resource/due content.

No patient encounter/lab/RF/transcript table is read or written by the L-1B loop runtime.

---

# 9. Verification and review closure

Final reviewed PR head:

`4c225784335228ccba6afd705210cd460ab43e28`

Exact-head gates:

- L1B run `34311310276` — **SUCCESS**.
- inherited L1 run `34311310300` — **SUCCESS**.

The final review closed:

- automatic rich-ingress verbatim-reasoning requirement;
- synthetic-only / initial-revision boundary;
- clinician-review requirement before retention labels;
- idempotent retry and duplicate-safe consolidation attempts;
- bridge provenance back to learning observations.

No material blocker remained in the bounded diff before release.

Squash merge:

`f15f854bbfca727356531d7b8ea896e3aedba437`

---

# 10. Explicit exclusions preserved

```text
NO patient record mutation
NO raw ChatGPT/Heidi transcript persistence
NO Daily Real-Case Review
NO Practice Review AI runtime
NO Signal promotion/backlink authority
NO Foundation state mutation from Challenge result alone
NO composite mastery/excellence score
NO opaque adaptive scheduler
NO production secret/config mutation in this release
NO direct accepted-Challenge write from external assistant
NO physiotherapy/CU-1/RF mutation
```

---

# 11. Lifecycle state

```text
IMPLEMENTED = YES
TESTED = YES
FOCUSED REVIEW = PASS
MERGED = YES
DEPLOYED = PENDING VERIFICATION
PRODUCTION-SMOKE-VERIFIED = NO
PRODUCTION INGEST KEY = NOT CONFIGURED
CHATGPT AUTOMATIC CONNECTION = NOT ACTIVATED
```

Next lifecycle action: verify the normal Render auto-deploy and complete authenticated production smoke. Production automatic-ingress activation remains a separate explicit integration decision.