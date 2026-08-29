# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY — CU-1 PRODUCT REPLAN / ROUTE EXPANSION HOLD.
> **Updated:** 2026-08-29 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR #63:** draft / design-only / unmerged.
> **Runtime evidence-aware generation:** NOT AUTHORIZED.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. Current truth

The previous CU-1 route-by-route evidence-hardening direction is **paused by product-owner REPLAN**.

Reviewed evidence/safety work through `posterior_interosseous_nerve_supinator_syndrome` remains valid and preserved. The product defect is not that the evidence layer is too strict; the defect is that the current design has allowed evidence coverage to become the primary success metric while the actual clinic output can remain too generic and clinically thin.

The intended product is a **clinically rich, analytical physiotherapy referral** rather than a safe but minimal evidence summary.

Current architectural correction:

```text
STRICT EVIDENCE / SAFETY ENGINE
+
RICH CLINICAL REHABILITATION MODEL
+
REFERRAL DOCUMENT POLICY
→ clinically useful detailed referral
```

Evidence remains a guardrail and authority layer. It is not the sole author of referral content.

---

# 2. Why the route queue is stopped

The former exact-next action was:

```text
distal_biceps_tendon_disorder_nonoperative
```

That action is now **superseded**.

Continuing route-by-route would risk repeating the same design failure: accumulating correct provenance/evidence-gap states while still producing referrals that do not adequately communicate therapeutic logic, staged rehabilitation orientation and return-to-function priorities.

No repository mutation for distal-biceps route coverage was made before the product-owner stop. Read-only evidence/registry inspection performed after the stop request creates no design authority and must not be treated as route progress.

---

# 3. Preserved proven work

The following remain valid:

- deployed CU-1 runtime v1 and its frozen base contract;
- v1.1 structured-history/evidence-authority semantics already reviewed in PR #63;
- protocol precedence and clinician-instruction separation;
- route/subtype/context applicability;
- no invented universal numeric progression or RTW/RTS thresholds;
- no silent framework hybridization;
- diagnosis/finding and symptom/objective-deficit distinctions;
- reviewed evidence shards, amendments and fixtures through PIN/supinator;
- manifest/matrix state through `cu1_evidence_manifest_v1_19` / `cu1_evidence_coverage_matrix_v1_19` as historical/current evidence-registry state pending the product replan.

These are guardrails to build on, not the final referral product model.

---

# 4. Current design blocker

```text
PRODUCT-UTILITY BLOCK
```

The detailed referral acceptance target is under-specified.

The product must be able to communicate, when supported by actual case inputs:

```text
clinical picture
+ focused history
+ actual findings
+ functional impact
+ therapeutic rationale
+ early rehabilitation priorities
+ progressive rehabilitation orientation
+ functional/work/sport reintegration where relevant
+ selected adjuncts when appropriate
+ reassessment/escalation conditions
```

without converting weak evidence, individual-study protocols or clinical organization into fake guideline certainty.

Therefore:

```text
route evidence profile PASS
!= detailed referral product PASS
```

---

# 5. Exact next authorized action

The next session must **not** start another route review.

It must perform a fresh six-canonical bootstrap and then conduct a pre-code design review of the **Rich Detailed Physiotherapy Referral model** defined in `SLICE_PLAN_CURRENT.md v1.15`.

Required outputs:

1. exact product job/user need for detailed referral;
2. detailed referral information architecture;
3. ownership boundaries between evidence engine, rich clinical rehabilitation model and referral document policy;
4. statement/authority taxonomy distinguishing literature-derived content, patient-specific protocol/clinician instruction, clinical-organizational content and therapist execution detail;
5. goals/progression/function-reintegration representation without false universal thresholds;
6. adjunct representation;
7. reassessment/escalation representation;
8. short-vs-detailed output relationship;
9. representative rendered referrals for already-reviewed routes with different evidence strength;
10. minimal schema/contract delta, if any;
11. usefulness + safety regression fixtures;
12. product-owner approval gate.

Pressure-test at minimum:

```text
lateral_elbow_tendinopathy
medial_elbow_tendinopathy
ulnar_neuropathy_at_elbow or posterior_interosseous_nerve_supinator_syndrome
```

---

# 6. Explicit HOLD / forbidden work

Until the product-model review is accepted:

```text
HOLD distal_biceps_tendon_disorder_nonoperative
HOLD all further route-by-route evidence expansion
HOLD runtime evidence-aware recommendation generation
HOLD formatter/runtime mutation
HOLD persistence/retention changes
HOLD CU-2
HOLD PR-1 restart
HOLD medico-legal runtime implementation
HOLD ClinicalAssertionV1/new claim-state enums
```

Existing evidence guardrails remain mandatory:

```text
missing history != negative history
not_assessed != normal
patient statement != objective finding
diagnosis != single finding/provocation/imaging result
route A evidence != route B evidence
clinician instruction != literature recommendation
patient-specific written protocol/healing restriction > conflicting route default
framework-specific strength != synthetic hybrid strength
therapist execution detail != automatic physician prescription
assessment measure != validated progression threshold
no generic MSK/cervical/elbow/peripheral-nerve evidence fallback
no invented universal numeric progression/RTW/RTS thresholds
```

---

# 7. Future Clinical Documentation direction remains deferred

The approved future direction remains:

```text
ONE reviewed patient-specific clinical-assertion layer
+
SEPARATE literature/evidence layer
+
MANY document policies
```

with:

```text
source/provenance axis != semantic claim-type axis
diagnosis != causation
temporal association != causal relationship
contradiction resolution may preserve unresolved competing interpretations
medico-legal report = future Document Policy, not a separate clinical engine
```

This remains future architecture only. The current CU-1 referral REPLAN does not authorize `ClinicalAssertionV1`, medico-legal schemas/runtime or persistence changes.

---

# 8. Current gates

```text
reviewed evidence/safety work through PIN       PRESERVED / PASS
route expansion                                 HOLD
rich detailed-referral product contract         NOT YET FROZEN
representative usefulness validation             NOT YET DONE
runtime evidence-aware generation               NO
DESIGN-COMPLETE for revised product target       NO
```

PR #63 remains draft and must not be merged merely because the earlier evidence-route gates passed.

---

# 9. Continuity rule

A fresh session must:

1. verify the then-current remote `main` and PR #63 head;
2. read all six canonicals in mandatory order;
3. recognize `SLICE_PLAN_CURRENT.md v1.15` as a **product REPLAN**;
4. preserve the completed evidence work;
5. perform the rich detailed-referral pre-code design review;
6. **not** resume `distal_biceps_tendon_disorder_nonoperative` or another route unless a later canonical decision explicitly reauthorizes route expansion.
