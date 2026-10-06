# Clinical Excellence Cockpit — Product Constitution v0.2

> **STATUS:** CONCEPTUALLY APPROVED / FROZEN FOR ORGANISATIONAL BOOTSTRAP.
> **Date:** 2026-09-27.
> **Role:** supporting product-intent/design artifact.
> **Authority boundary:** this file does **not** replace or extend the repository's six active canonical authorities. Runtime, roadmap and active-slice authority remain with the existing canonical owners.
> **Product Owner:** clinician using the system.

## 1. Product identity — clinical care trajectory, not a visit form

The Clinical Excellence Cockpit is not primarily:

- a guideline viewer;
- a calculator;
- an electronic osteoporosis form;
- a note generator;
- a collection of unrelated clinic utilities;
- or an AI that independently chooses treatment.

Its core purpose is to maintain and understand the patient's **clinical care trajectory** across time:

```text
PAST
How did we get here?
        ↓
PRESENT
What matters now and why?
        ↓
FUTURE
Where are we trying to go, and what must happen next?
```

For Osteoporosis Module 01, the disease course may span decades. The encounter is therefore an important event in the trajectory, not the centre of the data model.

## 2. Patient-above-module principle

There is one patient, not one patient per module.

A person may attend for osteoporosis, knee pain, spinal stenosis, an ankle injury, a distal-radius fracture or another problem. Clinical modules are independent domain interpreters of one shared patient reality.

```text
PATIENT
├── shared facts / events / investigations
├── Osteoporosis module
├── Knee / MSK module
├── Spine module
├── Trauma module
└── future modules
```

A shared event is recorded once. Relevant modules may consume it and add their own provenance-preserving interpretation.

Example:

```text
shared fact:
distal-radius fracture after fall

Trauma interpretation:
fracture-management problem

Osteoporosis interpretation:
possible fragility event requiring osteoporosis review

Rehabilitation interpretation:
functional / rehabilitation consequence
```

**One canonical factual reality does not imply one universal interpretation.**

## 3. One actual worldline

The Cockpit maintains one actual longitudinal history of what really happened.

It must distinguish:

```text
ACTUAL
what happened

CURRENT
what is true / unresolved now

PLANNED / INTENDED
what has been agreed or scheduled

POSSIBLE
options under consideration
```

A recommendation, option or planned future event does not become patient truth merely because it was proposed.

If guidance recommends IV zoledronate but the patient declines and oral therapy is chosen, the single actual trajectory records:

- guidance/recommendation;
- patient preference/constraint;
- actual decision;
- actual subsequent outcome.

It does not create parallel patient universes.

## 4. Core trajectory concepts

The long-term model should be able to represent, without prematurely fixing implementation details:

- **Event** — something that happened at a time;
- **State** — the current derived clinical condition;
- **Goal / Target State** — where care is trying to go;
- **Interpretation** — clinician/module meaning assigned to facts;
- **Decision** — what was actually chosen;
- **Treatment / Intervention Epoch** — a clinically coherent period of therapy or intervention;
- **Obligation** — a future care commitment created by prior facts/decisions;
- **Outcome** — what subsequently happened;
- **Preference / Constraint** — patient or system factor materially affecting the trajectory;
- **Evidence Context** — the evidence/guideline context relevant at a particular decision time.

This Constitution does not yet decide the technical storage model for these concepts.

## 5. Events first; state must be explainable

Important clinical states must be traceable to the events and decisions that produced them.

It is insufficient to know only:

```text
current treatment = denosumab
```

The system should be able to reconstruct the clinically relevant course:

```text
why started
→ actual administrations
→ delays / interruptions
→ response
→ fractures / adverse events
→ reason for continuation
→ reason for stopping or changing
→ future obligations created
```

Product invariant:

> For an important present state, the clinician should be able to ask: **How did we get here?**

## 6. Clinical Life Course Timeline

The timeline is the primary human projection of the trajectory, not the canonical database itself.

The main view should favour **clinical meaning over event density**:

```text
PAST                         TODAY                         FUTURE
●────■────·────·────◆────■────│────○────○────⚠
Dx   Tx   visit visit fracture       DXA   dose  unresolved
```

Major milestones may include:

- diagnosis;
- important fracture;
- major risk-state transition;
- treatment start/change/exit;
- important DXA or laboratory transition;
- major adverse event;
- clinically important patient decision.

Minor encounters remain accessible as smaller visit points.

The system may propose salience, but the clinician must be able to promote a seemingly minor encounter to an important milestone.

Detailed timeline-density and encounter-display rules are intentionally deferred.

## 7. Clinical Episode — context, not cold values

A major milestone should be explorable as a clinical episode containing, where applicable:

- what happened;
- state before;
- new information;
- what changed;
- options considered;
- clinician recommendation;
- patient preference/constraint;
- final decision;
- rationale;
- evidence/guideline context at that time;
- expected next steps;
- what actually happened afterwards;
- outcome;
- future obligations created.

The system should preserve temporal/clinical relationships without inventing causality.

```text
temporal or clinical association
!=
proven causal relationship
```

## 8. Treatment and intervention epochs may overlap

A treatment is not merely a medication name plus active/inactive.

A clinically important therapy is a longitudinal epoch with start, reason, exposure, response, deviations, outcomes and transition logic.

More than one intervention trajectory may coexist:

- anti-osteoporosis pharmacotherapy;
- vitamin-D replacement;
- exercise programme;
- glucocorticoid exposure;
- falls intervention;
- rehabilitation.

Therefore "epoch" must not be implemented as a single mutually exclusive global phase.

## 9. Past decisions create future obligations

A clinical decision may create future care commitments.

Examples:

```text
denosumab administration
→ next administration review/due state

planned denosumab discontinuation
→ successor-therapy decision
→ timing
→ monitoring

anabolic treatment
→ treatment-completion milestone
→ post-anabolic antiresorptive plan

new fracture
→ reassessment obligation
```

### Obligations require disposition, not obedience

The system is advisory. It should not force the clinician to execute an old plan when new information changes the situation.

A future item may resolve as:

```text
COMPLETED
PLANNED / RESCHEDULED
DEFERRED FOR CLINICAL REASON
NOT NEEDED + reason
PATIENT DECLINED
ALTERNATIVE PLAN CHOSEN
SUPERSEDED
UNKNOWN / NEEDS FOLLOW-UP
```

Example:

```text
expected next Prolia dose
→ patient decides not to continue
→ old obligation is not merely "overdue"
→ new decision point is created
→ clinician reassesses the trajectory
```

If a dental extraction or another relevant clinical issue arises, the system should surface the changed decision context rather than mechanically demand completion.

## 10. Care Communication Layer

Future reminders may be sent to the clinician and, when appropriate and consented, to the patient through a provider-agnostic communication layer.

Possible surfaces:

- messaging/SMS carrier such as Zadarma or another provider;
- email;
- reception/phone workflow;
- future patient application.

Rules:

- provider integration must be replaceable;
- message content should be privacy-minimal by default;
- sent/delivered message does not mean the clinical obligation was completed;
- patient responses are patient-supplied information, not automatically authoritative clinician-confirmed facts;
- a patient response may create a new review/decision event.

A future patient application is valuable but **parked**; it is not required for current Cockpit development.

## 11. What Matters Today

"What Matters Today" is not an independent manually maintained checklist.

It is a projection of the trajectory:

```text
past trajectory
+ current patient state
+ goals / desired direction
+ new events
+ open obligations
+ unresolved conflicts / uncertainty
+ current evidence
+ patient preferences / constraints
=
WHAT MATTERS TODAY
```

It must answer:

- what matters now;
- why it matters now;
- what from the past explains it;
- what future state we are trying to achieve.

## 12. Goals / desired future state

The Cockpit must represent direction of care, not only history.

A goal/target may include, for example:

- maintain fracture protection;
- complete a safe treatment transition;
- restore vitamin-D sufficiency;
- reduce falls risk;
- achieve a realistic monitoring/reassessment target.

Goals are time- and context-bound clinical intentions, not immutable patient facts.

## 13. Progressive complexity — Quick Clinical Tools

A clinician must not be forced to navigate the complete longitudinal workspace to solve one focused problem.

The same clinical/evidence engine should support progressively contextualised tools:

### A. General

Example:

> "I want to give zoledronic acid. What should I check?"

The tool provides general current evidence-backed prerequisites, cautions, monitoring and missing-information requirements.

### B. Patient-aware

Opened inside a patient record, the same tool reuses known relevant information:

- history;
- comorbidities;
- latest relevant laboratories;
- previous treatment/adverse reactions;
- current medication/treatment context;
- fractures and risk state.

It asks only for clinically necessary missing information.

### C. Decision-aware

The tool also understands the reason for the decision, for example:

> "I am considering zoledronic acid because this patient is stopping denosumab."

It then contextualises guidance to the specific decision state.

Hard rule:

> **One clinical/evidence engine, many interfaces.**

A Quick Tool must not create a second conflicting rule set.

Quick Tool guidance is **not** automatically written as patient truth. Only actual facts and clinician-confirmed decisions enter the clinical trajectory.

## 14. Decision support remains clinician-governed

Patient-specific guidance should:

- expose relevant patient facts;
- identify missing information;
- show options;
- explain evidence and uncertainty;
- identify factors that materially favour or weaken an option.

The final clinical decision remains the clinician's responsibility.

The preferred product language is **patient-specific clinical guidance / contextualised assessment**, not an autonomous "AI opinion".

## 15. Three distinct kinds of uncertainty

The system should distinguish:

### Data uncertainty
Example: exact last dose date unknown.

### Evidence uncertainty
Example: reasonable competing strategies with no decisive evidence.

### Clinician decision uncertainty
Example: clinician is unsure how to proceed.

For clinician decision uncertainty, a focused capability such as **"Help me decide"** may summarise:

- what is known;
- what is missing;
- reasonable options;
- evidence supporting each;
- patient-specific factors;
- residual uncertainty;
- useful next checks.

## 16. Patient preference and constraints are trajectory-relevant

Patient preference need not become a questionnaire.

It should be captured when it materially affects care.

Example:

```text
recommended:
IV zoledronate

patient:
declines infusion / prefers oral treatment

actual decision:
oral treatment
```

This distinction is clinically and medico-legally important for later interpretation of the trajectory.

Patient preference must remain distinct from:

- contraindication;
- access limitation;
- adverse effect;
- clinician choice;
- system availability constraint.

## 17. Strict semantic separation

At minimum the system must preserve distinctions among:

```text
FACT
OBJECTIVE RESULT
INTERPRETATION
OPTION CONSIDERED
CLINICIAN RECOMMENDATION
PATIENT PREFERENCE / CONSTRAINT
FINAL DECISION
TASK / OBLIGATION
OUTCOME
```

This is a foundational requirement for transcript/AI capture.

## 18. Cross-module relevance, not global broadcast

Shared patient events should not be broadcast as noise to every module.

Future cross-module mechanics should follow:

```text
shared patient event
→ typed relevance / provenance
→ interested modules become aware
→ each module decides whether the event is clinically actionable
```

A fracture may be highly relevant to Osteoporosis; an unrelated minor event may not be.

No module may silently transform another module's fact into a new diagnosis.

## 19. Evidence has time

The system needs two evidence clocks:

### What was known then?
Used to review a historical decision fairly.

### What is known now?
Used to guide today's decision.

New evidence may change current guidance without rewriting the historical context of an older decision.

## 20. Decision Review

Important decisions may later be examined through two lenses:

### Decision-time review
Was the decision reasonable given facts, options and evidence available then?

### Current review
How would the equivalent situation be approached with current evidence?

Avoid simplistic retrospective labels such as "correct/wrong" when uncertainty or legitimate alternatives existed.

## 21. Provider-agnostic capture

The Cockpit must not make Heidi the owner of clinical truth.

Possible capture sources may include:

- pasted Heidi transcript;
- future Heidi Enterprise/API;
- another scribe;
- manual transcript;
- future live speech layer.

All must feed a common semantic boundary before clinician confirmation and authoritative clinical state changes.

```text
capture provider
→ semantic candidate layer
→ clinician review where required
→ canonical patient facts/decisions
```

## 22. Future Live Clinical Copilot

The future Live Copilot is a planned capability, not a current implementation target.

It should reuse:

- the same patient state;
- the same semantic distinctions;
- the same evidence/rule engine;
- the same obligations;
- the same module logic.

The main future change is the capture mode from batch to live/streaming.

Real-time intervention should be selective; unnecessary interruption is a product failure.

## 23. Practice Review and learning

Practice Review should emerge from real trajectory evidence:

```text
actual decision
+ why it was taken
+ evidence available then
+ what happened afterwards
→ review / learning
```

Learning should distinguish knowledge, reasoning, execution and communication/system gaps.

## 24. Practice Cohort Intelligence

The system may aggregate multiple patient trajectories for:

- clinical audit;
- descriptive cohort analysis;
- pattern discovery;
- quality improvement;
- hypothesis generation;
- future research with appropriate governance/methodology.

Examples may include treatment exposure, fractures, DXA trajectory, BTMs, exercise/nutrition exposures, adherence and outcomes.

Hard safety principle:

```text
LOCAL OBSERVATIONAL ASSOCIATION
!=
CAUSAL CLAIM
!=
AUTOMATIC CLINICAL RULE
```

Cohort observations must not automatically rewrite patient-care guidance.

## 25. Complexity belongs under the surface

The underlying system may be sophisticated.

The clinician-facing product should answer simply:

- What do we know?
- What changed?
- What matters now?
- What are my reasonable options?
- What is missing?
- What should happen next?
- Why?

The clinician should be able either to:

> "Show me the whole trajectory"

or:

> "Help me with this one clinical question"

without feeling that these are unrelated products.

## 26. Fifteen-year product test

A central commercial/product test is:

> If the same patient has been followed for 15 years, can a clinician opening the record today rapidly understand what happened, why major decisions were made, what worked or failed, where the patient is now, and what should happen next?

If not, the longitudinal product architecture is inadequate.

## 27. Core vs module

Reusable Core should own generic mechanics where appropriate.

Modules own domain-specific clinical meaning, evidence, pathways and interpretation.

Every major design decision must ask:

> Is this shared patient/Core behaviour, or Osteoporosis-specific domain logic?

## 28. Current implementation boundary

This Constitution does not alter the currently authorised PR-1 transcript-capture slice.

PR-1 remains:

```text
ephemeral transcript
→ semantic candidates
→ deterministic Module-01 mapping
→ transient preview
→ no authoritative write
```

No Product Constitution principle grants authority to expand PR-1, implement PR-2, mutate the production UI, migrate schemas or activate real-patient transcript processing.

## 29. Twelve parked downstream design questions

The following remain explicitly **PARKED** and are not prerequisites for organisational bootstrap:

1. Which exact event types become canonical?
2. How are events versus derived state technically stored?
3. Which state belongs to reusable Core versus an individual clinical module?
4. How are historical patients/backfill handled?
5. How are guidelines/evidence versioned and renewed technically?
6. What exact confirmation flow promotes AI-derived candidates to authoritative state?
7. Which obligations are deterministic versus clinician-defined?
8. What exact information-density and interaction model should the timeline use?
9. Which decision reviews may be automated versus requiring explicit human review?
10. What regulatory boundary will govern future in-session decision support?
11. How is the existing Step-based Cockpit migrated without losing useful structure?
12. How do we prove the redesigned product is simpler and more useful in real clinical use?

They should be activated only through bounded downstream design work when their dependencies are ready.

## 30. Programme principle

One product vision may be advanced through multiple independent workstreams, but no workstream may create a parallel patient truth or duplicate an existing mechanism without an explicit reuse-before-new-path review.

---

# Core statement

> **The Clinical Excellence Cockpit maintains one factual clinical reality for each person across time. Each clinical module interprets the relevant parts of that reality through its own evidence and domain logic. Osteoporosis Module 01 must explain how the patient arrived at the present, support the clinician's decision now, and make the intended future care trajectory and obligations visible — while preserving clinical judgment and avoiding coercive automation.**

At practice level:

> **Multiple longitudinal patient trajectories may support safe cohort intelligence for audit, learning and research, without confusing association with causation or allowing local observations to silently become clinical rules.**
