# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-0 contract/design freeze

> **STATUS:** PRODUCT-OWNER-APPROVED DIRECTION / L-0 DESIGN FREEZE ACTIVE
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L0-2026-09-07`.
> **Production/runtime base:** `1d26195c77e186cff98086283252af2eb499dd17`.
> **Design branch:** `docs/clinical-learning-hub-l0-main-2026-09-07`.
> **Detailed design authority:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **Scope:** reusable Clinical Excellence Core learning architecture with Osteoporosis as Module 01 content.
> **Design/canonical authority:** GRANTED by product owner.
> **Runtime implementation authority:** NONE in this slice.
> **Patient-data mutation authority:** NONE.
> **Production config/secret authority:** NONE.

---

# 1. Trigger / product problem

The clinician already learns through three distinct evidence sources:

```text
structured theoretical learning
+ deliberately designed Clinical Challenges
+ actual osteoporosis encounters captured through Heidi/approved evidence
```

Today those sources are not joined into one durable, longitudinal learning system. Challenges can remain trapped in chat history, real-case reasoning is not systematically reviewed, and strong clinical intuition may remain disconnected from a formal conceptual model.

The target is not another transcript archive or generic CME tracker. The target is a reusable Clinical Learning Hub that closes the loop:

```text
FOUNDATION
→ CHALLENGE
→ REAL PRACTICE
→ REVIEW
→ SIGNAL
→ TARGETED INTERVENTION
→ SPACED RE-ASSESSMENT
→ REAL-PRACTICE RE-MEASUREMENT
```

---

# 2. Three instruments remain distinct

The Hub must preserve three different instruments because they measure different things:

```text
FOUNDATION MAP
What does the clinician actually understand and how well are concepts connected?

CLINICAL CHALLENGE
Can the clinician transfer that model to a controlled novel problem without hindsight?

DAILY REAL-CASE REVIEW
Was the model used appropriately in an actual encounter under real constraints?
```

They may feed the same Signal engine, but they must never collapse into one opaque score.

---

# 3. Frozen Core object direction

L-0 freezes the object responsibilities and provenance boundaries before runtime implementation.

## 3.1 `ClinicalLearningChallengeV1`

Structured learning artifact for one challenge. Minimum responsibility:

```text
challenge identity/version/module/date/title
challenge mode: synthetic / deidentified_real_case / mixed
topics / learning domains
initial case
Fact Ledger
progressive disclosures
clinician initial reasoning
clinician follow-up reasoning
confidence optional
final decision optional
debrief: strengths / clear errors / defensible disagreements /
         evidence gaps / blind spots / reasoning patterns / insights
references with evidence type / PMID / DOI / URL where available
gap classes
learning actions
next challenge / spaced repetition due
linked Signals
clinician review state
revision
```

Challenge storage is not a chat transcript and not a patient chart.

## 3.2 `LearningFactV1` — mandatory Fact Ledger

Hard invariant:

```text
REAL PATIENT FACT
!= SYNTHETIC / PROGRESSIVE-DISCLOSURE FACT
!= CLINICIAN HYPOTHESIS
!= AI INFERENCE
!= EDUCATIONAL COUNTERFACTUAL
```

Each meaningful fact carries at least:

```text
fact_id
statement
fact_scope
introduced_via
authoritative_for_patient
source
certainty optional
introduced_at_stage
status: active / corrected / withdrawn
supersedes_fact_id optional
```

This is mandatory specifically to prevent educational progressive-disclosure facts from later appearing as real patient facts.

## 3.3 `DailyCaseReviewV1`

Reviewed learning artifact from one actual eligible osteoporosis encounter. Minimum responsibility:

```text
review identity/version/module/date
opaque encounter reference optional
eligibility/source set
clinician self-review
critical decisions / uncertainty / confidence
reviewed decision reconstruction
Practice Observations
strengths / errors / uncertainties / defensible disagreements /
missed opportunities
gap classes
reasoning / communication patterns
evidence references
learning actions
linked Signals
clinician disposition
```

Raw Heidi transcript is not the persisted learning record.

## 3.4 `FoundationDomainStateV1`

Foundation state is evidence-based, not a self-rating or one-off AI score.

Allowed state direction:

```text
FORMAL_SOLID
INTUITIVE_UNSTRUCTURED
FRAGMENTED
UNKNOWN_UNTESTED
```

Evidence may include explanation without notes, mechanism, novel-case transfer, boundary recognition, evidence-directness calibration, later retention and real-case execution where observable.

## 3.5 `LearningDueStateV1`

The Hub needs explicit due-state semantics for:

```text
challenge repetition
foundation reassessment
daily eligible case review
periodic progress review
```

A due state must not fabricate an eligible case or imply that a missing assessment is a failure.

---

# 4. Daily real-case review contract

Target cadence:

```text
once per clinic day
→ if >=1 eligible osteoporosis/metabolic-bone encounter has usable Heidi/approved evidence
→ surface Daily Case Review due
→ clinician chooses one case OR accepts a transparent recommendation
→ self-review BEFORE AI critique
→ Practice Review evaluates reasoning / decision / safety / evidence / communication / close
→ clinician Accept / Modify / Dismiss material observations
→ persist reviewed structured learning artifact
→ feed candidate Signals + Foundation evidence
```

If no eligible case exists:

```text
state = no_eligible_case
```

No case is invented to satisfy cadence.

The first implementation should prefer a deterministic Cockpit due state rather than adding a background cron merely for the sake of scheduling. External notification timing is later scope.

---

# 5. Real-case eligibility and selection

Eligibility requires:

- a real osteoporosis/metabolic-bone encounter;
- usable Heidi transcript or another explicitly approved evidence set;
- protected processing boundary;
- enough encounter evidence to reconstruct at least one material decision/reasoning sequence.

When multiple cases qualify, a recommendation may consider learning value, uncertainty, safety relevance, major decision content or under-sampled domains, but must expose **why** that case was suggested and preserve clinician override.

Selection must not cherry-pick only failures; repeated correct behavior is necessary to identify sustained strengths.

---

# 6. Self-review before AI critique

Before seeing model critique, the clinician records a short self-assessment such as:

```text
What were the 2–3 critical issues?
Which decision was least certain?
Would I change anything now?
Confidence in the main decision(s), where useful
```

Purpose:

- reduce anchoring to AI framing;
- preserve calibration evidence;
- distinguish reasoning gaps from execution/documentation gaps;
- compare clinician uncertainty with later evidence.

---

# 7. Heidi / transcript ownership

Daily Case Review reuses the existing/future PR-1 protected transcript intake/provenance architecture. It must not create a second transcript owner.

Canonical direction:

```text
Heidi transcript
→ protected ephemeral intake
→ semantic evidence/candidates with provenance
→ encounter/decision reconstruction
→ Practice Review
→ reviewed learning artifact
→ raw transcript discarded by default
```

Hard rules:

- no raw transcript in public repo, localStorage, routine logs or learning tables by default;
- AI inference is not patient truth;
- transcript absence is not a negative clinical finding;
- no invented date, diagnosis, medication exposure, outcome or preference;
- learning review does not autonomously write patient data.

---

# 8. Module 01 initial Foundation Map

Initial osteoporosis/metabolic-bone nodes:

```text
1. Bone remodeling / cell biology / physiology
2. DXA physics, BMD/T-score, artifacts, precision and LSC
3. VFA / vertebral-fracture identification and interpretation
4. Fracture epidemiology, absolute risk and FRAX/FRAXplus reasoning
5. Secondary osteoporosis
6. Glucocorticoid-induced osteoporosis
7. Antiresorptive pharmacology
8. Anabolic pharmacology
9. Sequential / combination treatment
10. Denosumab discontinuation / rebound physiology and management
11. Bone-turnover markers and treatment-response interpretation
12. Treatment safety / contraindications / transitions
13. Evidence appraisal: directness, surrogate endpoints, guideline disagreement and uncertainty
14. Communication / shared decision making / continuity
```

This is a dependency map, not a reading list. L-0 may split/combine nodes only if diagnostic-assessment design demonstrates a better representation.

---

# 9. Diagnostic mapping before teaching

The default response to a learning gap is not `read more`.

Assessment should establish whether knowledge is:

```text
formal/solid
intuitive but under-structured
fragmented
unknown/untested
```

Assessment methods may include:

```text
explain mechanism without notes
explain why a clinical rule works
apply to a novel case
identify where the rule/evidence fails
compare competing explanations/strategies
separate direct evidence from inference
connect mechanism → evidence → clinical decision
```

Structural gaps lead to bounded Foundation prescriptions:

```text
prerequisite concepts
→ formal/mechanistic model
→ 1–2 authoritative sources
→ short retrieval test
→ transfer case
→ retention check
```

---

# 10. Challenge ingestion MVP boundary

First practical external-ChatGPT path:

```text
assistant natural-language debrief
+
validated ClinicalLearningChallengeV1 JSON
→ Cockpit Import Challenge
→ schema validation
→ PHI/identifier guard
→ duplicate detection by challenge_id
→ preview
→ clinician edit/confirm
→ protected persistence
```

Required record operations for L-1:

```text
view
edit
approve/save
delete
export JSON / Markdown
```

Direct external write is later scope.

---

# 11. Later narrow external ingestion

A later API/connector may use narrowly scoped learning authority such as:

```text
learning.challenge.write
learning.case_review.write
learning.read optional
```

It must not inherit:

```text
patient.read
patient.write
encounter.write
rf.write
admin/config
```

Server-side schema, PHI guard, duplicate/revision handling and persistence remain authoritative.

---

# 12. PHI firewall / learning-patient boundary

Hard boundary:

```text
LEARNING RECORD
!= PATIENT RECORD
```

Challenge imports should reject obvious direct identifiers including patient name, identity/GeSY number, phone, email, address and full DOB.

A real-case learning artifact may hold an internal opaque encounter reference in protected storage, but it is not an alternate patient chart. Hypothetical, counterfactual or AI-inferred facts can never write into authoritative patient storage.

---

# 13. Signals and intervention selection

Challenges, Foundation assessments, Daily Case Reviews and later Audit/Practice Review feed the existing Signal engine.

Negative Signals retain the existing root-cause classes:

```text
KNOWLEDGE GAP
REASONING GAP
EXECUTION GAP
COMMUNICATION / SYSTEM GAP
```

Typical interventions remain root-cause specific:

```text
knowledge → targeted foundation/reading/testing/spaced repetition
reasoning → cases/challenge/red-team/deliberate practice
execution → workflow/interface/task redesign
communication/system → wording/teach-back/handoff/process redesign
```

A single error does not automatically establish a stable weakness. Recurrent patterns require denominator/reliability context. Repeated positive observations may mature into sustained strengths.

---

# 14. Cadence

Target learning cadence:

```text
DAILY
one eligible real-case review when a suitable transcript-backed case exists

WEEKLY
one deliberately designed Clinical Challenge with progressive disclosure

EVERY 3–4 WEEKS OR AFTER A SMALL BLOCK OF COMPLETED RECORDS
Foundation/progress review

SPACED REPETITION
per node/topic based on prior evidence
```

Cadence is a learning target, never a quota permitting fabricated evidence.

---

# 15. Baseline-methodology protection

Visible systematic Daily Case Review coaching is an **intervention** and can change clinician behavior.

Default policy:

```text
before scored baseline:
  design/test learning machinery
  record learning exposure explicitly

during 30-case system-assisted scored baseline:
  Practice Review / Daily Case Review may run in shadow
  routine AI critique/coaching hidden by default
  safety-critical feedback remains allowed

after baseline lock:
  activate clinician-facing Daily Case Review as a formal intervention
```

If visible daily coaching is intentionally used during baseline, methodology must be explicitly REPLANned and the cohort relabelled before interpretation.

---

# 16. Implementation sequence after L-0

```text
L-0 — contract/design freeze
  object/provenance/privacy/due-state/revision/Signal contracts

L-1 — Challenge + Foundation MVP
  protected learning persistence
  JSON import/preview/edit/delete
  challenge history/filtering
  Foundation skeleton
  spaced-repetition due state
  NO new transcript stack

PR-1 / PR-2
  protected transcript extraction + inline clinician review/population

PR-3 / Practice Review shadow
  reviewed encounter observations / evidence / clinician disposition

L-2 — Daily Case Review
  reuse transcript owner
  self-review-before-AI
  reviewed learning artifact
  Signal/Foundation integration
  daily due state

post-baseline
  clinician-facing daily coaching
  adaptive foundation prescriptions
  Signal-targeted interventions
  re-measurement
```

---

# 17. Explicit exclusions in L-0

```text
NO runtime learning database implementation
NO new API endpoint
NO external connector credential
NO raw Heidi persistence
NO patient-record write
NO background cron
NO composite knowledge/excellence score
NO signature asset or RF change
NO production config mutation
```

---

# 18. L-0 exit criteria

L-0 is complete only when:

- object ownership and field-level contracts are explicit enough to implement without semantic invention;
- Fact Ledger provenance cannot confuse synthetic/progressive-disclosure facts with real patient facts;
- challenge revision/duplicate semantics are frozen;
- Foundation state/evidence transition semantics are frozen;
- Daily Case Review eligibility/self-review/clinician-disposition semantics are frozen;
- PHI firewall and learning-vs-patient boundary are explicit;
- due-state and baseline-intervention behavior are explicit;
- exact L-1 persistence/API/UI owners are identified;
- independent design review finds no material unresolved owner/privacy/methodology contradiction.

Until those criteria pass, `CLINICAL_LEARNING_HUB_DESIGN_V1.md` is canonical design direction, not runtime authority.

---

# 19. Exact next action

After this design is merged to `main`:

```text
perform L-0 exact contract review
→ freeze implementable schemas / owners / revision semantics
→ HOLD for separate L-1 implementation authority
```

No learning runtime implementation is authorized merely by merging this design.
