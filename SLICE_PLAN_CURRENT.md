# SLICE_PLAN_CURRENT.md — CU-1 referral product-model REPLAN v1.15

> **STATUS:** ACTIVE PRE-RUNTIME REPLAN — PRODUCT-UTILITY BLOCK.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice:** CU-1 Physiotherapy Referral — rich clinical referral model + evidence guardrails.
> **Authoritative base:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`.
> **Runtime writer:** NONE.
> **Runtime implementation:** NOT AUTHORIZED.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused.

---

# 1. REPLAN trigger

A product-owner review identified that the route-by-route evidence hardening was optimizing the wrong completion target.

The deployed/previous CU-1 utility can generate safe but clinically thin text such as generic requests for individualized exercise, strengthening and assessment. The intended clinic product is materially richer: an **analytical, clinically intelligent physiotherapy referral** that communicates the clinical picture, rehabilitation logic, staged objectives, functional restoration and reassessment priorities to the physiotherapist.

The evidence work remains valuable and is preserved. The design error was allowing the evidence/provenance layer to become the de facto author of the referral rather than a guardrail over a richer clinical rehabilitation/document model.

Therefore:

```text
EVIDENCE / SAFETY ENGINE
!=
REFERRAL AUTHOR
```

and:

```text
route evidence coverage PASS
!=
clinically useful detailed referral PASS
```

This is a material active-slice design change and therefore a REPLAN, not permission to continue mechanically to the next route.

---

# 2. Correct product architecture

The target architecture for the physiotherapy referral utility is now:

```text
STRICT EVIDENCE / SAFETY ENGINE
        ↓
sets boundaries, applicability, strength/certainty, conflicts,
do-not-infer rules, protocol precedence and reassessment triggers

+

RICH CLINICAL REHABILITATION MODEL
        ↓
organizes clinically useful rehabilitation logic without pretending
that every clinically sensible statement is a guideline-derived threshold

+

REFERRAL DOCUMENT POLICY
        ↓
produces a concise or detailed referral useful to the treating physiotherapist
```

The layers remain machine-distinct. Evidence provenance must not be weakened merely to make prose richer.

---

# 3. Preserved evidence/safety work

All reviewed CU-1 v1.1 evidence-governance work through PIN/supinator remains valid unless a later exact review identifies a specific defect.

Preserve in particular:

- diagnosis vs finding separation;
- missing/not-assessed != negative/normal;
- patient statement != objective finding;
- route/subtype/management-context-specific evidence applicability;
- framework-specific grades/certainty without silent hybridization;
- clinician instruction and patient-specific protocol authority distinct from literature authority;
- explicit protocol/healing restriction precedence;
- no invented universal numeric progression/RTW/RTS thresholds;
- no generic cross-route evidence leakage;
- explicit evidence-gap behavior;
- route-specific safety/reassessment boundaries;
- existing reviewed route shards/fixtures and manifest/matrix state through PIN/supinator.

No reviewed route is rolled back merely because the product-output model is being replanned.

---

# 4. Product requirement — Detailed Referral must be clinically rich

The detailed referral should be capable, when supported by the clinician-entered case state, of communicating a coherent sequence such as:

```text
1. clinical diagnosis / impression and relevant context
2. focused history and symptom/load behavior
3. actual examination findings
4. functional impact and patient-priority task
5. therapeutic rationale / rehabilitation priorities
6. early rehabilitation orientation
7. progressive rehabilitation orientation
8. functional / work / sport reintegration orientation where relevant
9. selected adjunct options where appropriate
10. reassessment / escalation conditions
11. evidence basis / authority where useful
```

This is a **clinical communication structure**, not a claim that the literature validates one universal numbered protocol for every route.

The referral must be able to say clinically useful things such as:

- control irritability and modify the specifically aggravating load while maintaining useful activity;
- begin tolerated active loading appropriate to the presentation;
- progressively restore relevant strength, mobility, endurance and load tolerance;
- progressively reintroduce the patient's actual functional, work or sport demands;
- use selected adjuncts when clinically indicated and evidence-compatible;
- reassess when progress is discordant or when defined safety/alternative-owner findings emerge.

The exact content must remain route/context aware and must not invent patient facts.

---

# 5. Evidence does not have to supply every clinical-organization sentence

The old route-hardening logic implicitly pushed too much referral content through `EvidenceClaimV1` and `RehabilitationSequenceV1` as if every clinically useful organizational statement required a route-specific graded recommendation.

The REPLAN must explicitly determine which statements belong to:

```text
A. evidence-derived recommendation / restriction
B. patient-specific protocol or clinician instruction
C. clinically reasonable rehabilitation organization / document structure
D. therapist execution detail
E. clinician-UI-only evidence/context
```

Category C is the missing product layer. It must be clinically responsible and must never be mislabeled as literature authority.

The design may use clinically meaningful stages/orientations without asserting unsupported universal time windows, numeric pain-monitoring thresholds, fixed exercise doses, fixed visit counts or validated transition criteria.

---

# 6. Detailed referral quality target

A route is not product-complete merely because it has an evidence profile or an explicit evidence gap.

For representative routes, the detailed output must demonstrate that a physiotherapist receives enough information to understand:

```text
WHAT the clinician thinks the problem is
WHY physiotherapy is being requested
WHAT was actually found
WHAT function matters
WHAT broad rehabilitation priorities are intended
HOW rehabilitation should evolve conceptually
WHAT should not be assumed/prescribed automatically
WHEN reassessment or escalation is appropriate
```

The referral should avoid both extremes:

```text
unsafe pseudo-protocol specificity
AND
sterile generic wording with little clinical value
```

---

# 7. DIA/reference-output lesson

External/generated richer referral examples may be used as **product-shape pressure tests**, not as canonical evidence authority.

Useful structure observed in such examples:

```text
clinical picture
→ therapeutic logic
→ goals
→ organized rehabilitation
→ adjuncts
→ reassessment
→ return to function
```

However, unsupported or weakly supported specifics must not be promoted into universal defaults, including examples such as:

- fixed pain-monitoring cutoffs/recovery windows;
- mandatory isometric set/hold prescriptions;
- universal week-based phases;
- mandatory heavy-slow-resistance progression;
- fixed session counts/course duration;
- universal symmetric-strength discharge thresholds;
- broad superiority claims unsupported by the applicable evidence.

The product should preserve the useful structure while rejecting false precision.

---

# 8. Existing object model — review before mutation

The following frozen v1.1 objects remain preserved pending audit:

```text
ReferralHistoryV2
HistoryProvenanceEntryV1
RouteHistoryPromptV1
RehabilitationSequenceV1
RehabilitationPhaseV1
InterventionDirectionV1
RehabilitationCriterionV1
GoalPlanV2
ReassessmentPlanV2
AuthorityReferenceV1
ProtocolConstraintV1
ClinicianModificationV1
EvidenceSourceV1
EvidenceClaimV1
RouteEvidenceProfileV1
```

Do **not** create new runtime objects merely to satisfy this REPLAN.

The next design review must determine whether the missing rich-clinical layer can be represented by refining existing referral/goal/rehabilitation/document-policy semantics or requires a narrowly scoped new design object. Object proliferation is not assumed.

---

# 9. Relationship to future Clinical Documentation architecture

The previously accepted future direction remains deferred:

```text
ONE reviewed patient-specific clinical-assertion layer
+
SEPARATE literature/evidence layer
+
MANY document policies
```

This CU-1 REPLAN does not authorize `ClinicalAssertionV1`, medico-legal runtime code, new claim-state enums or persistence changes.

The physiotherapy referral is, however, a concrete pressure test for the broader principle that **document policy and clinical communication logic must not be collapsed into the literature-evidence layer**.

---

# 10. Explicitly out of scope now

```text
NO distal_biceps route review
NO further route-by-route evidence expansion
NO runtime evidence-aware generation
NO formatter/runtime implementation
NO persistence/retention change
NO CU-2
NO PR-1 restart
NO medico-legal implementation
NO ClinicalAssertionV1 creation
NO universal numeric progression thresholds
NO fixed generic MSK protocol
```

Route expansion remains paused until the rich referral product model is reviewed and accepted.

---

# 11. Exact next authorized action

Perform a **fresh pre-code product/architecture review of the Detailed Physiotherapy Referral model** using the real current runtime/contracts and representative already-reviewed routes.

Required review outputs before any further route work:

```text
A. exact user/product job of the detailed referral
B. detailed-referral information architecture
C. evidence-engine vs rich-clinical-model vs document-policy ownership
D. exact statement/authority taxonomy for evidence-derived vs clinical-organizational content
E. how goals / rehabilitation orientation / functional reintegration are represented without false evidence claims
F. how selected adjuncts are represented
G. how reassessment/escalation is represented
H. short-vs-detailed output relationship
I. representative rendered examples from existing reviewed routes
J. exact minimal schema/contract changes, if any
K. acceptance fixtures that test usefulness as well as safety
L. migration impact on existing reviewed evidence shards
```

Use at minimum representative pressure-test routes with different evidence states, including:

- lateral elbow tendinopathy — relatively strong route-specific CPG authority;
- medial elbow tendinopathy — lower-certainty treatment evidence;
- UNE or PIN — context with material evidence gaps/safety ownership.

Do not resume `distal_biceps_tendon_disorder_nonoperative` until this design review reaches an explicit product-owner-approved target and the canonical exact-next action is changed again.

---

# 12. Acceptance gate for the REPLAN

Before route-by-route work can resume:

```text
evidence/safety guardrails preserved                         PASS required
rich detailed-referral product contract defined              PASS required
clinical-organizational vs literature authority separated    PASS required
representative detailed outputs clinically useful            PASS required
no false universal protocol specificity                      PASS required
short/detailed relationship explicit                         PASS required
minimal machine-contract delta identified                    PASS required
product-owner approval                                       PASS required

ROUTE EXPANSION                                               HOLD
RUNTIME AUTHORIZED                                            NO
```

The goal is not more evidence files. The goal is a physiotherapy referral utility that is simultaneously **clinically useful, analytically rich, safe, evidence-aware and honest about uncertainty**.
