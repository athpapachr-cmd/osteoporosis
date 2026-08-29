# AGENTS.md — permanent operating rules for Clinical Excellence / Module 01 Osteoporosis

> **STATUS:** permanent project operating authority.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **SCOPE:** reusable Clinical Excellence Core Engine + Module 01 Osteoporosis + every AI/Codex/ChatGPT session working on this repository.
> **PRODUCT OWNER:** the clinician using the system.

---

# 1. Mandatory bootstrap — SIX CANONICALS before project work

A fresh conversation must not reconstruct the project from chat memory, a prior assistant summary, README, an old PR body or a remembered implementation.

Before substantial planning, implementation, review, branch/PR creation, merge, deploy, smoke or canonical mutation:

1. verify the fresh remote GitHub SHA of `athpapachr-cmd/osteoporosis/main`;
2. read the six active canonical authorities in this order:
   1. `AGENTS.md`;
   2. `TODO.md`;
   3. `CLINICAL_EXCELLENCE_PLAN.md`;
   4. `SLICE_PLAN_CURRENT.md`;
   5. `CURRENT_OPERATIONAL.md`;
   6. `osteoporosis-change-log.md`;
3. build an internal **Canonical Bootstrap Manifest** containing at minimum:
   - verified `main` SHA;
   - current major phase;
   - active slice and its design status;
   - active writer/lock and allowed mutation scope;
   - current branch/PR/deploy/smoke state when applicable;
   - permanent safety/privacy/baseline invariants relevant to the work;
   - exact next authorized action;
   - explicit deferred/forbidden actions;
4. inspect the exact runtime/schema/evidence needed for the requested task only after the bootstrap.

Hard rules:

```text
CHAT MEMORY != CANONICAL TRUTH
OLD HANDOFF != CURRENT STATE
PARTIAL CANONICAL READING != BOOTSTRAP
FRESH CONVERSATION != NEW MUTATION AUTHORITY
```

If current truth, writer scope or next action cannot be determined, stop mutation and reconcile the canonicals first.

---

# 2. Canonical architecture — progressive refinement

There are exactly six active canonical authorities:

```text
AGENTS.md
→ TODO.md
→ CLINICAL_EXCELLENCE_PLAN.md
→ SLICE_PLAN_CURRENT.md
→ CURRENT_OPERATIONAL.md
→ runtime/code execution

osteoporosis-change-log.md = durable history
```

Their roles are deliberately different.

### `AGENTS.md`
Permanent operating rules: how all sessions must work.

### `TODO.md`
Long-range product roadmap/checklist: where the system is going and broad completion state.

### `CLINICAL_EXCELLENCE_PLAN.md`
Active detailed phase architecture: system design, object model, major implementation stages, dependencies, methodological constraints and phase exit criteria.

### `SLICE_PLAN_CURRENT.md`
One approved current slice: exact problem, scope/out-of-scope, design, object/data contract, implementation seams, acceptance evidence, rollback and REPLAN triggers.

### `CURRENT_OPERATIONAL.md`
The sole operational **NOW** and active-work lock: current source identity, writer, branch/PR, status, blockers, deploy/smoke state and exact next action.

### `osteoporosis-change-log.md`
Append-only durable history of completed decisions, releases and validated milestones. Historical NEXT/HOLD language has no operational authority.

`HANDOFF_CURRENT.md` is retained only as a compatibility redirect and is not an active canonical authority.

`README.md` is navigation only.

## 2.1 Contradiction rule

A new accepted rule must not coexist with an older active instruction that directs materially incompatible behavior.

```text
identify contradiction
→ determine correct canonical owner
→ replace/remove/supersede stale active instruction
→ preserve historical fact in changelog/archive if useful
→ scan remaining canonicals for drift
```

Do not duplicate the same rule into every file merely to make it visible.

---

# 3. One active writer for overlapping mutation scope

`CURRENT_OPERATIONAL.md` is the sole writer lock.

Before modifying runtime or canonicals, determine:

- active task;
- active writer/branch/scope;
- HOLD/review/merge/deploy authority;
- exact next action.

Rules:

```text
active overlapping writer exists
→ other sessions may inspect/review
→ other sessions MUST NOT mutate overlapping scope

review-only state
→ inspection allowed
→ mutation not implied

new conversation
→ does not override an existing lock
```

A meaningful implementation session should claim its scope in `CURRENT_OPERATIONAL.md` before overlapping mutation and release/update that lock when the slice closes or pauses.

---

# 4. Canonical update protocol — continuity after every meaningful step

The documentation system exists so a conversation can end at any point and the next fresh conversation can resume without reconstructing hidden context.

After every material state transition, update the correct canonical owner before moving on.

### Operational state changes → `CURRENT_OPERATIONAL.md`
Update when any of these change:

- active writer/scope;
- branch/PR/head/base;
- implementation started/completed;
- test/smoke evidence;
- blocker/HOLD;
- merge/deploy state;
- exact next action.

It must always distinguish:

```text
DESIGNED
!= IMPLEMENTED
!= TESTED
!= MERGED
!= DEPLOYED
!= PRODUCTION-SMOKE-VERIFIED
!= PILOT-VALIDATED
```

### Active design changes → `SLICE_PLAN_CURRENT.md`
Update only when the design of the active slice changes materially. A discovery that invalidates an owner, data contract, safety assumption, workflow state or implementation boundary is a **REPLAN trigger**, not permission to patch around the plan.

### Roadmap/completion changes → `TODO.md`
Update when a durable roadmap item changes status, priority or sequence.

### Phase architecture changes → `CLINICAL_EXCELLENCE_PLAN.md`
Update when object architecture, methodological rules, major stage sequence or system boundaries change. Do not use it as a PR log.

### Completed historical event → `osteoporosis-change-log.md`
Append after meaningful completion/merge/deploy/validated decision. Do not rewrite old entries to match new terminology.

At the end of a substantial session, `CURRENT_OPERATIONAL.md` must be sufficient for a new session to answer:

1. Where are we?
2. What is already proven?
3. What is not yet proven?
4. What is actively being changed?
5. What is deferred?
6. What exactly happens next?

---

# 5. Product purpose

This repository is not only an Osteoporosis Cockpit. It is the proving ground for a reusable **Personal Clinical Excellence System** whose purpose is to improve the clinician's real practice over time.

Canonical learning-health loop:

```text
STANDARD
→ LEARN
→ TEST / MASTER
→ APPLY IN REAL PRACTICE
→ MEASURE
→ AUDIT
→ PRACTICE REVIEW
→ GAP OR STRENGTH
→ INTERVENE / REINFORCE
→ RE-MEASURE
→ SYSTEM LEARNS
```

Module 01 is Osteoporosis. Reusable Core mechanics must later support other clinical domains without rebuilding the engine.

Every substantial design decision asks:

> Is this reusable Core behavior, or Osteoporosis-specific clinical content?

---

# 6. Audit and Practice Review are different instruments

The system must not collapse measurement and coaching into one black box.

### Audit
Answers primarily:

> Did the applicable clinical process occur according to the defined metric/standard?

Audit should be deterministic where possible, denominator-aware, transparent and neutral during baseline collection.

### Clinical Practice Review
Answers primarily:

> How well was the consultation conducted, reasoned, communicated and closed, and what should change next time?

Practice Review may use AI-assisted interpretation but every important observation must preserve:

- encounter evidence/provenance;
- linked standard/evidence when applicable;
- confidence;
- clinical importance;
- whether it is a strength, gap, safety concern or uncertainty;
- proposed action;
- clinician accept/modify/dismiss state.

Practice Review never silently rewrites clinical truth or substitutes for clinician judgment.

---

# 7. Four gap classes — prescribe the right intervention

Negative Signals must be classified before intervention:

1. **Knowledge gap** — relevant knowledge missing/not retained.
2. **Reasoning gap** — facts known but interpretation/decision process weak.
3. **Execution gap** — clinician knows what should happen but action is unreliable.
4. **Communication/system gap** — reasoning may be sound but communication, workflow, documentation or continuity fails.

Typical response:

```text
knowledge → targeted reading/testing/spaced repetition
reasoning → cases/challenge/red-team/deliberate practice
execution → workflow/checklist/task redesign
communication/system → teach-back/template/handoff/process redesign
```

Do not answer every gap with more reading.

---

# 8. Strengths are active Signals

Repeated good performance may mature into `SUSTAINED_STRENGTH` only after adequate evidence and stability over time.

A sustained strength should trigger preservation of the successful workflow, less basic repetition, more advanced challenge and periodic surveillance rather than endless drilling.

---

# 9. Transparent measurement — no black-box scores

For meaningful metrics preserve where applicable:

```text
current
baseline
change
trend
denominator/sample size
reliability
target/standard
benchmark + comparability
data completeness
```

No stable-looking composite Clinical Excellence score before an adequate baseline exists. Outcomes such as fractures must not be naively converted into clinician-competence penalties without context/risk adjustment.

---

# 10. Baseline integrity and intervention exposure

The approved osteoporosis baseline sequence remains:

```text
5-case usability/capture pilot
→ one deliberate refinement
→ freeze Baseline Form + KPI applicability/calculation contract
→ 30 consecutive unique scored baseline cases
→ baseline lock
→ systematic improvement interventions + re-audit
```

During the scored baseline, no live KPI coaching/red-green feedback or routine Practice Review intervention should alter behavior. Safety-critical alerts remain an exception.

Clinical Practice Review infrastructure may be developed and validated in **shadow mode** before baseline lock. If the product owner chooses to expose systematic coaching before the scored baseline, that is a methodological change and must be explicitly recorded; the resulting cohort must not be mislabeled as an untouched pre-intervention baseline.

---

# 11. AI/transcript governance

Heidi or other transcripts are supplementary sources, not unreviewed clinical truth.

For transcript-assisted capture/review:

```text
raw transcript
→ structured candidate extraction
→ clinician review/edit/accept/reject
→ accepted structured data
```

Default rules:

- raw transcript is ephemeral and is not persisted in PostgreSQL/localStorage/logs by default;
- do not commit transcript content or identifiable patient information to the public repository;
- do not invent absent values, dates, diagnoses, treatment exposure or patient preferences;
- preserve negation, temporality, speaker/source and uncertainty;
- distinguish patient statement, objective result, clinician interpretation, option discussed, recommendation, preference, final decision and follow-up task;
- accepted data retains provenance such as `source=heidi_transcript` and clinician-review state;
- AI suggestions require clinician review before becoming authoritative patient data.

---

# 12. Evidence governance

Clinical standards/rules must be explicit, versioned and non-hybrid. Important rules should eventually carry:

```text
rule_id
module/domain
framework/guideline
version/year
recommendation/criterion
strength/certainty when available
reviewed_on
status
```

If frameworks differ, show them separately rather than manufacturing a silent hybrid threshold.

New evidence should be classified as confirming, interesting/no change, potentially practice-changing, practice-changing or conflicting/insufficient before changing workflow or standards.

## 12.1 Patient-specific clinical assertion governance — permanent future-safe invariants

The reusable Core must preserve a strict boundary between patient-specific assertions and literature/evidence authority.

```text
SOURCE ASSERTION != REVIEWED ACTIVE CLINICAL ASSERTION
PATIENT-SPECIFIC CLINICAL ASSERTION != LITERATURE / GUIDELINE EVIDENCE CLAIM
DOCUMENT WORDING != CLINICAL FACT
GUIDELINE RECOMMENDATION != PATIENT FACT
```

For patient-specific clinical information, **source/provenance is one axis and semantic claim type is a separate axis**. A radiology report, clinician note, transcript, prior document, AI draft or patient statement identifies where an assertion came from; it does not by itself determine whether that assertion is an objective finding, diagnosis, interpretation, recommendation, causation opinion or another semantic type, nor whether the assertion is correct.

Permanent reasoning boundaries:

```text
DIAGNOSIS != CAUSATION
TEMPORAL ASSOCIATION != CAUSAL RELATIONSHIP
PATIENT-REPORTED != OBJECTIVELY CONFIRMED
SUPPORTED INTERPRETATION != CONFIRMED FACT
NOT DOCUMENTED != NEGATIVE
```

Material contradictions must not be silently propagated. Contradiction review may resolve one assertion as contradicted/superseded when justified, but the system must also be able to preserve **unresolved competing interpretations** when the available evidence does not support a defensible single winner. Historical provenance is preserved; two mutually incompatible assertions must not be silently presented downstream as one settled active truth.

The clinician retains final clinical authority. Material override of a surfaced evidence/provenance conflict should remain explicit and attributable when the future data contract supports it.

These invariants define future Core behavior only. They do **not** by themselves authorize a new `ClinicalAssertionV1`, new enums, medico-legal implementation, persistence change or expansion of an active slice. Exact object/schema ownership must be decided by a fresh owner/schema audit in an explicitly authorized future design boundary.

---

# 13. Patient Voice

Patient feedback is a learning input, not merely satisfaction scoring. It should be capable of generating Signals and later re-measurement around understanding, plan/rationale, questions/preferences and communication failures.

Clinician impression of understanding remains distinct from the patient's own report.

---

# 14. Privacy and production safety

The repository is public.

**Never commit identifiable patient data, transcripts, clinical exports, names, patient identifiers, phone/email/address, unredacted documents or secrets.**

Production clinical data belongs only in authenticated/private storage. Authentication, authorization, audit logging, retention/data-minimization and GDPR/privacy requirements remain explicit production-readiness concerns.

Do not claim whole-service privacy/GDPR compliance merely because one clinical route is protected.

---

# 15. Repository/release discipline

Prefer feature branch → PR → focused review/evidence → squash merge.

For the current Render service, auto-deploy follows `main`; do not manually trigger a second deploy after a normal merged code/doc commit unless auto-deploy actually failed or an explicit redeploy is required.

Do not expand scope for cosmetic cleanup while a safety/data-integrity objective is unresolved.

Public fixtures/tests must be synthetic or fully anonymized.

---

# 16. Current product boundary

The legacy `index.html`/legacy Cockpit remains historical point-of-care material, not the final Clinical Excellence Home.

The Baseline Audit/patient-centric clinical layer is the current production proving ground. Calendar/Setmore/Zadarma integration can be paused independently and must not block Clinical Practice Review, Core Engine, standards, audit or learning work.

Patient leaflets/posters remain downstream unless the product owner explicitly changes priority.

---

# 17. Stop rule

The system exists to improve real clinical practice, not to maximize documentation volume or engineering polish.

Every change should be classified as:

```text
critical safety/data-integrity defect
clinically meaningful improvement
useful operational refinement
optional/cosmetic refinement
```

When the approved objective is adequately evidenced, close the slice and move on rather than extending scope without a new reason.
