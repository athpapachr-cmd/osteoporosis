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

# 5. Product purpose — improve the visit and improve the clinician

The repository is the proving ground for a reusable **Personal Clinical Excellence System**.

The primary product outcome is not data entry, documentation completeness or audit-score production. The system exists to improve:

1. the **current clinical encounter** while it is happening; and
2. the **clinician's future practice** by reviewing what was said, reasoned and decided and whether it was appropriate.

The canonical learning-health loop remains:

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

# 6. Clinical Guidance, Audit and Practice Review are different instruments

The system must not collapse live guidance, measurement and coaching into one black box.

## 6.1 Clinical Guidance

Answers primarily:

> Given this patient's longitudinal state and today's purpose, what should be surfaced, checked, resolved or closed now?

Clinical Guidance may:

- order the visible consultation flow;
- surface due, milestone, event-triggered or unresolved items;
- show why an item is relevant now;
- use prior structured data to avoid repeated questioning;
- identify unresolved prerequisites before downstream decisions;
- remain agent-/treatment-/archetype-aware.

Clinical Guidance must not silently choose the clinician's treatment decision or present an unsupported rule as guideline truth.

## 6.2 Audit

Answers primarily:

> Did the applicable clinical process occur according to the defined metric/standard?

Audit should be deterministic where possible, denominator-aware, transparent and suitable for longitudinal measurement.

## 6.3 Clinical Practice Review

Answers primarily:

> How well was the consultation conducted, reasoned, communicated and closed, and what should change next time?

Practice Review may use AI-assisted interpretation but every important observation must preserve:

- encounter evidence/provenance;
- linked standard/evidence when applicable;
- confidence;
- clinical importance;
- strength/gap/safety/uncertainty direction;
- proposed action;
- clinician accept/modify/dismiss state.

Practice Review never silently rewrites clinical truth or substitutes for clinician judgment.

Hard distinction:

```text
LIVE CLINICAL GUIDANCE
!= KPI/PERFORMANCE FEEDBACK
!= POST-VISIT PRACTICE REVIEW
```

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
execution → workflow/interface/task redesign
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

# 10. Revised baseline integrity — pilot the real product, not a known-unusable manual form

The former sequence that placed a five-case manual usability pilot before transcript-assisted capture and adaptive visit guidance is superseded.

The approved sequence is now:

```text
close/merge/deploy/smoke critical finalization integrity
→ implement minimum dynamic guided-visit engine
→ implement transcript candidate extraction
→ implement inline clinician review / provisional population
→ run 5 consecutive eligible real system-assisted pilot encounters
→ one deliberate refinement
→ freeze guided-visit + capture + KPI/applicability contracts
→ 30 consecutive unique scored system-assisted baseline encounters
→ baseline lock
→ systematic Practice Review / improvement interventions
→ re-measure
```

The five-case pilot tests the **workflow intended for actual use**, not a deliberately manual predecessor already known to impose unacceptable duplicate-entry burden.

During the 30-case scored baseline:

- the stabilized **Clinical Guidance** layer remains active because it is part of the product being evaluated;
- routine KPI score feedback, red/green performance coaching and routine clinician-facing Practice Review remain hidden by default;
- safety-critical alerts remain allowed;
- guidance exposure must be traceable where technically feasible;
- the cohort must be described as a **system-assisted baseline**, not an untouched/unassisted clinician baseline.

The architecture should preserve, where feasible, whether clinically relevant content was already present before a system cue versus entered/resolved after a cue. This enables later measurement of prompt dependence and internalization rather than forcing the clinician to work without useful support merely to preserve a theoretical unassisted baseline.

Any later methodological change must be recorded explicitly before relabelling a cohort.

---

# 11. Dynamic encounter principle — archetype plus longitudinal triggers, not one checklist per visit number

Osteoporosis visits are not interchangeable. The visible workflow must be derived from multiple context layers rather than from one static form.

At minimum the visit-plan engine may use:

```text
encounter archetype / visit intent
patient relationship / prior encounter state
active treatment agent and treatment episode
actual administration history
elapsed treatment exposure
next-due status / delay state
monitoring due state
new fracture / adverse event / other safety trigger
unresolved prior tasks or prerequisites
patient-specific modifiers
```

Do not create a separate hard-coded form for every ordinal treatment visit.

For repeated therapy such as denosumab, ordinal administration count may be an input, but clinical behavior should be driven by versioned evidence/clinic-policy **milestone rules** using actual administration history and elapsed exposure. Exact visit-number rules must not be invented merely because examples such as early doses, periodic reviews or a long-duration review were discussed.

Priority layering:

```text
critical safety/event override
→ unresolved prior critical item
→ treatment/agent-specific requirement
→ evidence-defined milestone/due item
→ archetype base flow
→ contextual/optional item
```

A higher-priority trigger must never be hidden by a lower-priority default.

---

# 12. AI/transcript governance

Heidi or other transcripts are supplementary sources, not unreviewed clinical truth.

For transcript-assisted capture/review:

```text
raw transcript
→ structured candidate extraction
→ provisional in-place population / review state
→ clinician edit/accept/reject
→ accepted structured data
```

Default rules:

- raw transcript is ephemeral and is not persisted in PostgreSQL/localStorage/logs by default;
- do not commit transcript content or identifiable patient information to the public repository;
- do not invent absent values, dates, diagnoses, treatment exposure or patient preferences;
- preserve negation, temporality, speaker/source and uncertainty;
- distinguish patient statement, objective result, clinician interpretation, option discussed, recommendation, preference, final decision and follow-up task;
- accepted data retains provenance such as `source=heidi_transcript` and clinician-review state;
- AI suggestions require clinician review before becoming authoritative patient data;
- a blank field after extraction means "not captured / not established", not an inferred negative;
- existing authoritative patient data must not be silently overwritten by transcript extraction;
- conflict between transcript candidates and authoritative longitudinal data must surface for clinician resolution.

The preferred UX is not a second disconnected candidate list that recreates data-entry burden. Candidates should be able to appear **in the clinical cards they belong to** as clearly provisional values until reviewed.

---

# 13. Evidence governance

Clinical standards, guidance rules and milestone rules must be explicit, versioned and non-hybrid.

Important rules should eventually carry:

```text
rule_id
module/domain
framework/guideline or approved clinic policy
version/year
recommendation/criterion
applicability / trigger
strength/certainty when available
reviewed_on
status
```

If frameworks differ, show them separately rather than manufacturing a silent hybrid threshold.

New evidence should be classified as confirming, interesting/no change, potentially practice-changing, practice-changing or conflicting/insufficient before changing workflow or standards.

No exact therapy milestone, monitoring cadence or visit-number behavior becomes canonical clinical guidance without reviewed evidence/policy provenance.

---

# 14. Patient Voice

Patient feedback is a learning input, not merely satisfaction scoring. It should be capable of generating Signals and later re-measurement around understanding, plan/rationale, questions/preferences and communication failures.

Clinician impression of understanding remains distinct from the patient's own report.

---

# 15. Privacy and production safety

The repository is public.

**Never commit identifiable patient data, transcripts, clinical exports, names, patient identifiers, phone/email/address, unredacted documents or secrets.**

Production clinical data belongs only in authenticated/private storage. Authentication, authorization, audit logging, retention/data-minimization and GDPR/privacy requirements remain explicit production-readiness concerns.

Do not claim whole-service privacy/GDPR compliance merely because one clinical route is protected.

---

# 16. Repository/release discipline

Prefer feature branch → PR → focused review/evidence → squash merge.

For the current Render service, auto-deploy follows `main`; do not manually trigger a second deploy after a normal merged code/doc commit unless auto-deploy actually failed or an explicit redeploy is required.

Do not expand scope for cosmetic cleanup while a safety/data-integrity objective is unresolved.

Public fixtures/tests must be synthetic or fully anonymized.

---

# 17. Current product boundary and stop rule

The legacy `index.html`/legacy Cockpit remains historical point-of-care material, not the final Clinical Excellence Home.

The patient-centric clinical layer is the current production proving ground. Calendar/Setmore/Zadarma integration can be paused independently and must not block Clinical Guidance, transcript-assisted capture, Practice Review, Core Engine, standards, audit or learning work.

Patient leaflets/posters remain downstream unless the product owner explicitly changes priority.

The system exists to improve real clinical practice, not to maximize documentation volume or engineering polish.

Every change should be classified as:

```text
critical safety/data-integrity defect
clinically meaningful improvement
useful operational refinement
optional/cosmetic refinement
```

When the approved objective is adequately evidenced, close the slice and move on rather than extending scope without a new reason.
