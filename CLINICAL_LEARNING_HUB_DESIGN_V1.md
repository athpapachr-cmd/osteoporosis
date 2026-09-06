# Clinical Learning Hub — Design v1

> **STATUS:** PRODUCT-OWNER-APPROVED DIRECTION / PRE-IMPLEMENTATION DESIGN INPUT
> **Canonical home:** `athpapachr-cmd/osteoporosis`
> **Core scope:** reusable Clinical Excellence learning mechanics
> **Module 01 content:** Osteoporosis / metabolic bone disease
> **Runtime authority:** NONE — this document does not start implementation

---

# 1. Product problem

The clinician already learns through three different evidence sources:

1. structured theoretical learning;
2. deliberately designed Clinical Challenges;
3. actual clinical encounters recorded through Heidi or another approved evidence source.

Today these are not joined into one durable learning system. Challenge discussions can disappear inside chat history, real-case reasoning is not systematically reviewed each clinic day, and strong clinical intuition may remain disconnected from a formal conceptual map.

The Clinical Learning Hub closes that loop without turning the Cockpit into a transcript archive or generic CME tracker.

Canonical intent:

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

# 2. Core architecture

```text
CLINICAL LEARNING HUB
│
├── Foundation Map
│   ├── dependency/concept graph
│   ├── diagnostic assessment
│   ├── mastery evidence
│   └── spaced retention state
│
├── Clinical Challenges
│   ├── synthetic/de-identified cases
│   ├── progressive disclosure
│   ├── clinician reasoning before debrief
│   └── evidence-calibrated learning artifact
│
├── Daily Case Review
│   ├── real eligible osteoporosis encounter
│   ├── Heidi transcript as ephemeral evidence
│   ├── clinician self-review before AI critique
│   ├── Practice Review / evidence review
│   └── reviewed learning artifact
│
├── Signals
│   ├── recurrent gaps
│   ├── sustained strengths
│   ├── uncertainty/calibration patterns
│   └── transfer / prompt-dependence evidence where valid
│
└── Learning Plan
    ├── foundation blocks
    ├── papers/guidelines
    ├── deliberate-practice cases
    ├── spaced repetition
    └── re-assessment
```

Core owns reusable objects/mechanics. Module 01 owns osteoporosis-specific domains, cases, evidence and clinical content.

---

# 3. Three instruments must remain distinct

```text
FOUNDATION MAP
What do I actually understand and how well is it connected?

CLINICAL CHALLENGE
Can I transfer that model to a controlled novel problem without hindsight?

DAILY REAL-CASE REVIEW
Did I use it correctly in a real encounter under real constraints?
```

All three may feed the same Signal engine, but they do not measure the same thing and must not be collapsed into a single score.

---

# 4. `ClinicalLearningChallengeV1`

A challenge is a structured learning artifact, not a transcript and not a patient chart.

Candidate contract:

```text
challenge_id
schema_version
module
created_at
title
challenge_mode: synthetic / deidentified_real_case / mixed
topics[]
learning_domains[]
difficulty_optional

initial_case
fact_ledger[]
progressive_disclosures[]
clinician_initial_response
clinician_followup_responses[]
clinician_confidence_optional
final_clinician_decision_optional

debrief:
  strengths[]
  clear_errors[]
  defensible_disagreements[]
  evidence_gaps[]
  blind_spots[]
  reasoning_patterns[]
  clinical_insights[]

references[]:
  title
  evidence_type
  framework_or_guideline_optional
  PMID_optional
  DOI_optional
  url_optional
  supports_or_challenges

gap_classes[]
learning_actions[]
next_challenge_topic_optional
spaced_repetition_due_optional
linked_signal_ids[]
clinician_review_state
revision
```

Duplicate detection is by stable `challenge_id`. Revision must not silently overwrite the provenance of earlier accepted learning evidence.

---

# 5. `LearningFactV1` — mandatory Fact Ledger

A progressive-disclosure teaching fact must never later be summarized as a real patient fact.

Candidate contract:

```text
fact_id
statement
fact_scope:
  real_deidentified_case_fact
  synthetic_case_fact
  clinician_hypothesis
  ai_inference
  counterfactual_teaching_point
introduced_via:
  initial_case
  progressive_disclosure
  heidi_transcript
  clinician_entry
  evidence_review
authoritative_for_patient: true / false
source
certainty_optional
introduced_at_stage
status: active / corrected / withdrawn
supersedes_fact_id_optional
```

Hard invariant:

```text
REAL PATIENT FACT
!= SYNTHETIC / PROGRESSIVE-DISCLOSURE FACT
!= CLINICIAN HYPOTHESIS
!= AI INFERENCE
!= EDUCATIONAL COUNTERFACTUAL
```

A learning artifact cannot write a hypothetical, AI-inferred or counterfactual fact into authoritative patient storage.

---

# 6. `DailyCaseReviewV1`

A Daily Case Review is a Practice-Review-derived learning artifact from one actual eligible osteoporosis encounter.

Candidate contract:

```text
review_id
schema_version
module
review_date
opaque_encounter_reference_optional
eligibility_state
source_set

clinician_self_review:
  key_decisions[]
  uncertainty_points[]
  what_might_be_done_differently_optional
  confidence_by_decision_optional

decision_reconstruction
practice_observation_ids[]
strengths[]
clear_errors[]
uncertainties[]
evidence_based_disagreements[]
missed_opportunities[]
gap_classes[]
reasoning_patterns[]
communication_patterns[]
evidence_refs[]
learning_actions[]
linked_signal_ids[]
clinician_disposition
created_at
```

The persisted object stores structured, clinician-reviewed learning evidence. Raw Heidi transcript is not persisted by default.

---

# 7. Daily scheduled real-case review

Target operating cadence:

```text
once per clinic day
→ if >=1 eligible osteoporosis encounter has a usable Heidi transcript
→ surface `Daily Case Review due`
→ clinician chooses one case OR accepts a transparent system recommendation
→ transcript uses the existing protected ephemeral transcript boundary
→ reconstruct encounter facts / decisions with provenance
→ clinician self-review BEFORE AI critique
→ Practice Review evaluates reasoning, decision, safety, evidence, communication and close
→ material claims link to evidence/standards where applicable
→ clinician Accept / Modify / Dismiss important observations
→ persist structured reviewed learning artifact only
→ feed candidate Signals and Foundation evidence
```

If no eligible real case exists:

```text
state = no_eligible_case
```

The system must never fabricate a case to satisfy the daily cadence.

The first implementation may use a deterministic Cockpit due-state rather than a background cron. Notification channels and exact clock time are separate later choices.

---

# 8. Real-case eligibility and selection

Eligibility requires:

- a real osteoporosis/metabolic-bone encounter;
- a usable Heidi transcript or another explicitly approved evidence set;
- protected processing boundary;
- sufficient encounter evidence to reconstruct at least one material clinical decision/reasoning sequence.

If multiple cases qualify, the system may recommend one based on learning value, uncertainty, safety relevance, major decision content or under-sampled domains, but must show **why** it recommends that case and preserve clinician override.

The selector must not cherry-pick only failures. Repeated correct behavior is necessary to identify sustained strengths.

---

# 9. Self-review before AI critique

Before seeing model critique, the clinician records a short self-assessment:

```text
What were the 2–3 critical issues in this case?
Which decision was least certain?
Would I change anything now?
Confidence in the main decision(s), where useful
```

Purpose:

- reduce anchoring to the AI framing;
- preserve calibration evidence;
- compare clinician uncertainty with later evidence;
- distinguish true reasoning gaps from documentation/execution gaps.

---

# 10. Heidi / transcript boundary

Daily Case Review reuses the same transcript intake/provenance architecture as PR-1/PR-2. It must not create a second competing transcript owner.

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

- no raw transcript in repository, localStorage, routine logs or learning tables by default;
- identifiable transcript use requires the approved provider/privacy/data-control gate;
- learning review does not autonomously write patient data;
- AI inference is not patient truth;
- transcript absence is not a negative finding;
- no invented date, diagnosis, medication exposure, outcome or preference.

---

# 11. `FoundationDomainStateV1`

The Foundation Map addresses a specific pattern: reasoning may be strong while the explicit formal/theoretical model is incomplete or fragmented.

Candidate states:

```text
FORMAL_SOLID
  can explain from first principles, apply to a novel case,
  identify exceptions/boundaries and calibrate evidence

INTUITIVE_UNSTRUCTURED
  usually uses the concept correctly but cannot yet articulate
  a complete formal model or dependency chain

FRAGMENTED
  knows components but dependencies/causal model are incomplete,
  weakly connected or internally inconsistent

UNKNOWN_UNTESTED
  insufficient current evidence of understanding
```

Candidate fields:

```text
foundation_node_id
module
domain
concept
prerequisite_node_ids[]
state
evidence_attempt_ids[]
last_assessed_at
assessment_methods[]
confidence_calibration_optional
strength_or_gap_signal_ids[]
next_review_due_optional
clinician_note_optional
```

State transitions require assessment evidence; no one-off AI judgment or self-rating is sufficient.

---

# 12. Module-01 initial Foundation Map

Initial osteoporosis/metabolic-bone domains:

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

This is a dependency map, not a reading list. L-0 may split or combine nodes if diagnostic assessment demonstrates a better representation.

---

# 13. Diagnostic mapping before teaching

For each foundation domain, first establish what is already known.

Assessment methods include:

```text
explain mechanism without notes
explain why the clinical rule works
apply it to a novel case
identify where the rule/evidence fails
compare competing explanations or strategies
separate direct evidence from inference
connect mechanism → evidence → clinical decision
```

Output:

```text
formal/solid
intuitive but under-structured
fragmented
unknown/untested
```

The default response to a gap is not "read more".

---

# 14. Foundation prescription

When a gap is structural, prescribe a bounded foundation block:

```text
prerequisite concept(s)
→ formal/mechanistic model
→ 1–2 authoritative sources
→ short retrieval test
→ transfer case
→ retention check
```

Example:

```text
denosumab clinical management = strong
denosumab rebound recognition = strong
osteoclast/RANKL/reversible-suppression model = fragmented
↓
target mechanism block
↓
return to timing / CTX / zoledronate clinical reasoning
```

This connects existing knowledge islands rather than accumulating unrelated papers.

---

# 15. Mastery evidence

`MASTERED` does not mean "read" or "one answer correct".

Evidence should include several of:

```text
accurate explanation without prompts
mechanistic explanation
novel-case transfer
exception/boundary recognition
evidence-directness calibration
retention on later assessment
consistent real-case execution where observable
```

No fixed numerical mastery threshold is frozen before denominators/reliability are established.

---

# 16. Signals from multiple evidence sources

Challenges, Daily Case Reviews, Foundation assessments and later Audit/Practice Review all feed the existing Signal engine.

Example:

```text
Challenge → overstates indirect evidence
Daily Case Review → same pattern
Later Challenge → same pattern
↓
Candidate recurrent Signal:
EVIDENCE DIRECTNESS / CALIBRATION
↓
root cause = knowledge and/or reasoning
↓
foundation intervention + contrasting cases
↓
spaced reassessment
```

A single error may be important but does not automatically define a stable weakness. Recurrent patterns require denominator/reliability context. Repeated positive observations may mature into `SUSTAINED_STRENGTH`.

---

# 17. Target cadence

```text
DAILY
one real osteoporosis case review when an eligible transcript-backed case exists

WEEKLY
one deliberately designed Clinical Challenge with progressive disclosure

EVERY 3–4 WEEKS OR AFTER A SMALL BLOCK OF COMPLETED RECORDS
Foundation / progress review:
  what stabilized
  what remains intuitive-unstructured
  what remains fragmented
  recurrent gap/strength signals
  next foundation target

SPACED REPETITION
per-node/per-topic due dates based on prior evidence
```

Cadence is a learning target, not a quota that permits fabricated cases.

---

# 18. Baseline-methodology protection

Systematic visible Daily Case Review is an **intervention** and can alter clinician behavior. It must not silently contaminate the 30-case system-assisted baseline.

Default policy:

```text
before scored baseline:
  design/test learning machinery
  record learning exposure explicitly

during 30-case scored baseline:
  Practice Review / Daily Case Review may run in shadow
  routine AI critique/coaching hidden by default
  no visible daily coaching is called baseline-neutral

after baseline lock:
  activate clinician-facing Daily Case Review as a formal intervention
```

If visible daily AI coaching is desired during the scored baseline, that is a methodology **REPLAN** and the cohort must be redefined/relabelled before collection continues.

---

# 19. Challenge ingestion MVP

First practical external-ChatGPT path:

```text
assistant natural-language debrief
+
validated ClinicalLearningChallengeV1 JSON
→ Cockpit `Import Challenge`
→ schema validation
→ PHI/identifier guard
→ duplicate detection by challenge_id
→ preview
→ clinician confirm/edit
→ protected persistence
```

Required record actions:

```text
view
edit
approve/save
delete
export JSON / Markdown
```

PDF export is optional later and is never patient documentation.

---

# 20. Later direct authenticated ingestion

A later connector/API may write learning records directly using a **narrowly scoped** capability rather than reusing the broad browser clinical session as an external bearer credential.

Candidate scopes:

```text
learning.challenge.write
learning.case_review.write
learning.read_optional
```

Explicitly not implied:

```text
patient.read
patient.write
encounter.write
rf.write
admin/config
```

The server retains schema validation, duplicate handling, PHI checks and persistence ownership.

---

# 21. PHI firewall / patient-learning separation

Hard boundary:

```text
LEARNING RECORD
!= PATIENT RECORD
```

Challenge imports should reject obvious direct identifiers such as:

```text
patient name
identity / GeSY number
phone / email / address
full DOB
```

De-identified descriptors such as age/age band, sex, diagnosis, medication exposure and fracture phenotype may be retained where necessary for learning and permitted by the privacy policy.

A real-case learning artifact may contain an internal opaque encounter reference in protected storage, but the learning record is never an alternate patient chart or unreviewed source of clinical truth.

---

# 22. Clinician-facing progress view

The Hub should answer:

```text
What have I worked on?
What repeatedly fails?
Where are disagreements evidence-defensible?
Which strengths are stable?
What remains intuitive rather than formally structured?
What is due for repetition?
What should the next challenge/foundation block be?
```

Useful displays:

- topic/domain attempt counts;
- Foundation state/evidence history;
- recurrent Signals;
- retention-due items;
- strengths;
- calibration patterns;
- learning actions due.

Do not introduce a stable-looking composite "Clinical Knowledge" or "Clinical Excellence" score without adequate denominator, reliability and validated interpretation.

---

# 23. Implementation sequence

```text
L-0 — Learning contract / design freeze
  ClinicalLearningChallengeV1
  LearningFactV1
  DailyCaseReviewV1
  FoundationDomainStateV1
  LearningDueStateV1
  persistence/privacy/import contracts

L-1 — Challenge + Foundation MVP
  protected learning persistence
  JSON import / preview / edit / delete
  challenge history/filtering
  Foundation Map skeleton
  spaced-repetition due state
  NO new transcript ingestion stack

PR-1 / PR-2
  existing transcript intake/extraction + clinician review
  remains authoritative transcript/candidate pathway

PR-3 / Practice Review shadow
  reviewed encounter observations
  evidence/provenance
  clinician disposition

L-2 — Daily Case Review
  reuse PR-1 transcript boundary
  self-review-before-AI
  decision reconstruction / Practice Review
  reviewed learning artifact
  Signal/Foundation integration
  daily due-state

post-baseline intervention phase
  clinician-facing daily coaching
  adaptive foundation prescriptions
  Signal-targeted interventions
  re-measurement
```

---

# 24. L-0 acceptance requirements

Before runtime implementation, the bounded L-0 slice must freeze at least:

1. normative object schemas and versioning;
2. Challenge fact/provenance semantics;
3. Daily Case Review eligibility and self-review ordering;
4. Foundation state transition evidence rules;
5. learning-record persistence ownership;
6. PHI guard and deletion/export behavior;
7. duplicate/revision semantics;
8. spaced-repetition due-state semantics without invented clinical cadence;
9. Signal integration contract;
10. external ingestion authentication boundary;
11. baseline-intervention/shadow-mode rule;
12. regression-threat map against PR-1/PR-2/Practice Review/patient-record owners.

---

# 25. REPLAN triggers

STOP and REPLAN if implementation requires any of the following:

- a second transcript ingestion owner competing with PR-1;
- raw transcript persistence by default;
- learning artifacts becoming authoritative patient truth;
- patient identifiers in public/source fixtures;
- AI findings becoming accepted Signals without clinician disposition where material;
- one opaque composite mastery score replacing domain evidence;
- visible daily coaching during the scored baseline without methodology revision;
- external learning API requiring broad patient/encounter/RF credentials;
- autonomous clinical-record mutation from a learning review.

---

# 26. Current status

```text
PRODUCT DIRECTION              APPROVED
DETAILED DESIGN ARTIFACT       V1 RECORDED
L-0 CONTRACT FREEZE            NOT STARTED
L-1 IMPLEMENTATION             NO
DAILY CASE REVIEW RUNTIME      NO
DIRECT EXTERNAL INGESTION      NO
```

The next learning-runtime action is a separately bounded **L-0 design slice**, only after the current RF production-smoke/canonical closeout gate is completed.