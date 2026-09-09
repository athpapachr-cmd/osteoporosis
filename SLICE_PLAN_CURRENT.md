# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1B Learning Loop

> **STATUS:** IMPLEMENTED / TESTED / FOCUSED REVIEW PASS / RELEASE HOLD
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1B-LEARNING-LOOP-2026-09-08`.
> **Base:** `95629fead4da5aeb1fcc0146296a9ac0f767d7ea`.
> **Branch:** `feat/clinical-learning-l1b-learning-loop-2026-09-08`.
> **Reviewed runtime candidate:** `54e881cc512b0de9ad1e9d95a8caffbe8c5d8777`.
> **Prior L-1 release:** PR #82 merged/deployed.
> **Merge/deploy authority:** NONE for L-1B until separate product-owner decision.
> **Frozen L-0/L-1 schema owners:** READ-ONLY / unchanged.

---

# 1. Problem solved

L-1 worked as a safe Challenge/Foundation store, but real use exposed that the source Challenge conversation produced a rich human-oriented export rather than the frozen machine schema expected by the Hub. That made import red/empty and forced the clinician to handle JSON manually.

L-1B adds the missing integration and learning loop while preserving the frozen Challenge contract.

---

# 2. Delivered learning loop

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

One successful test does not cancel later repetition.

---

# 3. Rich episode ingestion

Implemented bounded support for:

- canonical `ClinicalLearningChallengeV1` input; and
- the observed rich export shape (`schema_type=ClinicalLearningChallengeV1`, `schema_version=1.0`, `session`, `topic_tags`, grouped mentor observations, source-local IDs, evidence/actions/spaced-repetition candidate).

The adapter:

- generates deterministic UUIDs for source-local IDs;
- maps case facts/disclosures/reasoning/debrief/evidence/actions;
- maps known Foundation aliases conservatively;
- resets imported clinician-review/reference-verification/Signal authority;
- discards the raw source payload after normalization;
- keeps summary-only legacy exports manually salvageable with an explicit warning;
- requires verbatim clinician response text for new **automatic** rich-export ingress.

Rich automatic/manual adapter support is initial-revision only (`revision=1`) in this slice. Later rich revision semantics require a separate contract extension rather than guesswork.

---

# 4. Pending Inbox

Implemented protected states:

```text
pending_review
accepted
rejected
invalid
```

External ingestion can create only `pending_review`. Accepted Challenge persistence still goes through the existing clinician review flow. Browser save then links the accepted Challenge to the pending learning episode and activates the Learning Loop; a failed link remains retryable rather than silently disappearing.

---

# 5. Performance debrief

The UI and normalized artifact distinguish:

```text
STRENGTHS
NEEDS REINFORCEMENT / IMPROVEMENT
CLEAR ERRORS
DEFENSIBLE DISAGREEMENTS
EVIDENCE GAPS / BLIND SPOTS
REASONING PATTERNS / CLINICAL INSIGHTS
```

`needs reinforcement` is not relabelled as `clear_error`.

---

# 6. Knowledge-island bridging

`BridgeTargetV1` links 2–3 Foundation nodes plus provenance to source learning observations/actions.

A bridge remains `planned` until a later bridge-transfer occurrence tests the concepts jointly. Only an explicitly clinician-reviewed later result can move bridge state to `demonstrated` or `needs_reinforcement`.

Graph adjacency or co-listing Foundation nodes is not treated as competence evidence.

---

# 7. Repeated consolidation

Transparent initial schedule anchored to actual Challenge acceptance:

```text
D+3   retrieval
D+7   discrimination / contrast
D+14  novel transfer
D+30  bridge-transfer / retention
```

The schedule is a visible product default, not a claim that one timing algorithm is universally optimal.

Allowed reviewed results:

```text
retained
partially_retained
not_retained
improved_beyond_original
not_assessed
```

Any result other than `not_assessed` requires explicit clinician review.

Same-payload retry is idempotent. A materially different second submission after an occurrence is completed fails closed instead of creating ambiguous duplicate history.

---

# 8. Fresh resources

Resource recommendations remain mutable overlays outside the immutable Challenge content hash.

Supported categories include article, guideline, webinar, course, conference session, video, podcast and other. Recommendation status/freshness changes do not create Challenge revisions and never self-certify reference verification.

---

# 9. Automatic-ingress seam

Dedicated boundary:

```text
POST /clinical/learning/api/ingress/episodes
X-Learning-Ingest-Key
CLINICAL_LEARNING_INGEST_KEY
```

It:

- fails closed when not configured;
- uses constant-time credential comparison;
- does not reuse `CLINICAL_DATA_KEY`;
- accepts explicit synthetic learning only;
- requires verbatim clinician reasoning for automatic rich-export ingress;
- creates only a pending import;
- cannot write patient data, accepted Challenges, Foundation state, Signals or verified evidence directly.

Production key creation and ChatGPT/plugin connection are explicitly **outside this implementation/review slice**.

---

# 10. Persistence

Added learning-only owners:

```text
clinical_learning_pending_imports
clinical_learning_loop_plans
clinical_learning_consolidation_attempts
clinical_learning_resource_recommendations
```

`clinical_learning_due_items` additionally carries `consolidation_test` occurrences.

Challenge deletion cleans linked loop/attempt/resource/due content. No patient encounter/lab/RF/transcript tables are read or written by the new loop runtime.

---

# 11. Clinician UI

Default surfaces:

1. Inbox
2. Challenges
3. Learning Loop
4. Foundation Map
5. Due
6. Advanced

Advanced owns manual JSON fallback. The default learning workflow no longer assumes the clinician understands UUIDs or the machine schema.

---

# 12. Verification evidence

Reviewed runtime candidate:

`54e881cc512b0de9ad1e9d95a8caffbe8c5d8777`

Exact-head gates:

- L1B run `34311161485` — SUCCESS.
- inherited L1 run `34311161517` — SUCCESS.

The L1B run passed focused tests, inherited L1/L0, Python/browser syntax, frozen-owner guard, adjacent-owner scope guard and diff hygiene.

The final focused review also closed:

- automatic rich-ingress verbatim-reasoning requirement;
- synthetic-only/revision boundary;
- clinician-review requirement before retention labels;
- idempotent retry for consolidation attempts;
- bridge provenance back to learning observations.

No remaining material blocker was found in the bounded diff.

---

# 13. Explicit exclusions

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
NO direct accepted-Challenge write from external assistant
NO physiotherapy/CU-1/RF mutation
```

---

# 14. Release state

```text
IMPLEMENTED = YES
TESTED = YES
FOCUSED REVIEW = PASS
PR = NEXT / DRAFT
MERGED = NO
DEPLOYED = NO
PRODUCTION INGEST KEY = NOT CONFIGURED
CHATGPT AUTOMATIC CONNECTION = NOT ACTIVATED
```

Next allowed action: Draft PR / RELEASE HOLD. Merge/deploy requires a separate explicit product-owner decision.