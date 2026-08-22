# HANDOFF_CURRENT.md — current operational handoff

> **Updated:** 2026-08-22 14:38 Asia/Nicosia
> **Canonical repository:** `athpapachr-cmd/osteoporosis`
> **Current major phase:** Clinical Excellence Blueprint + Baseline/Audit foundation
> **Active detailed plan:** `CLINICAL_EXCELLENCE_PLAN.md`
> **Current module:** Module 01 — Osteoporosis

This file contains only current operational truth. Permanent rules belong in `AGENTS.md`; roadmap in `TODO.md`; completed history in `osteoporosis-change-log.md`.

---

## 1. Current project definition

The project is now defined as a **Personal Clinical Excellence System** with a reusable Core Engine.

The existing Osteoporosis Cockpit is one component of Module 01, specifically the point-of-care Clinical Practice / Encounter Execution layer. It is not the whole improvement system.

Core loop:

```text
STANDARD
→ LEARN
→ TEST / MASTER
→ APPLY
→ MEASURE
→ AUDIT
→ GAP OR STRENGTH
→ INTERVENE / REINFORCE
→ RE-MEASURE
→ SYSTEM LEARNS
```

Future clinical modules may include low-back pain, neck pain, knee pain, hip pain and shoulder pain, but they are intentionally deferred until the reusable engine is proven with Osteoporosis.

---

## 2. Canonical project control plane

The Osteoporosis repository now uses the same canonical-document discipline as the digital-secretary project.

Active canonical set:

```text
AGENTS.md
TODO.md
CLINICAL_EXCELLENCE_PLAN.md
HANDOFF_CURRENT.md
osteoporosis-change-log.md
```

`README.md` is navigation only.

No chat thread is allowed to become the only place where a durable architectural decision lives.

---

## 3. Existing repository/runtime truth

Current root application assets present before this control-plane bootstrap:

```text
Dockerfile
README.md
index.html
main.py
osteoporosis-qa-handout.html
requirements.txt
```

The existing Cockpit already contains structured osteoporosis inputs, risk logic, evidence-assisted suggestions, AI elaboration, follow-up, agreed-plan controls and longitudinal history/trend functionality.

Known architectural limitations to address in later implementation slices include:

- guideline frameworks are partly hybridized;
- a custom internal risk index should not carry treatment-decision authority;
- DXA longitudinal analysis is T-score-centric rather than BMD/LSC/scanner-centric;
- fracture and treatment history need more structured event/timeline objects;
- follow-up needs task/due-date lifecycle;
- post-visit audit is not yet a dedicated object;
- evidence linking is partly keyword/string driven;
- identifiable-patient production use needs stronger privacy/security architecture.

No runtime code was changed during the current documentation/bootstrap step.

---

## 4. Current architecture — preserve

Reusable Core Engine:

```text
Standards
Evidence / Guidelines
Learning
Testing / Mastery / Calibration
Clinical Practice
Patient Voice
Audit / Measurement
Safety
Benchmarking
Improvement
Signal Engine
Personal Adaptation
```

Module 01 — Osteoporosis supplies domain-specific content.

The central adaptive object is `Signal`.

Signal sources include:

- clinical practice;
- patient feedback;
- audit;
- learning/testing;
- evidence;
- safety;
- benchmarks;
- sustained strengths.

Negative signals must be root-cause classified before intervention:

```text
KNOWLEDGE GAP
REASONING GAP
EXECUTION GAP
COMMUNICATION / SYSTEM GAP
```

Positive signals can mature into `SUSTAINED STRENGTH` and should reduce unnecessary basic repetition while increasing advanced challenge and benchmarking.

---

## 5. Personalization principles currently approved

The system should be demanding but not mechanically adversarial.

Approved modes:

```text
STANDARD
CHALLENGE
RED TEAM
LEARNING
```

Approved personalized behaviors:

- expose reasoning and source/framework rather than only `correct/incorrect`;
- allow clinician ACCEPT / MODIFY / REJECT with reason;
- prioritize high-confidence errors;
- track calibration where useful;
- compare strengths and gaps externally when methodologically valid;
- distinguish critical defect from meaningful improvement from cosmetic refinement;
- stop expanding scope when the approved objective has sufficient evidence of completion.

---

## 6. Patient Voice — current approved role

Patient feedback is a clinical-learning input, not merely satisfaction data.

Initial concepts to preserve:

```text
understanding of condition
understanding of plan
understanding of rationale/duration/risks
whether questions/preferences were addressed
free-text confusion/praise/suggestion
```

Repeated feedback can generate Signals, trigger learning/workflow change and require re-measurement.

---

## 7. Measurement principles currently approved

Progress bars alone are insufficient.

For important metrics preserve:

```text
Current
Baseline
Change
Trend
Sample size / denominator
Reliability
Target / standard
External benchmark + comparability
Data completeness
```

Progress bar = current state.
Run chart = trajectory.

No overall Clinical Excellence score should be shown as meaningful until a real baseline is established.

---

## 8. Current priority

**NEXT DESIGN ACTION:** create **Baseline Osteoporosis Audit v1 + KPI Dictionary v1**.

This should define:

- case sampling method;
- inclusion/exclusion rules;
- first audit domains;
- exact KPI numerators/denominators/exclusions;
- data-completeness rules;
- `not applicable` handling;
- minimum sample/reliability display rules;
- baseline lock criteria;
- re-audit timing;
- mapping from each KPI to competency/standard/signal/intervention.

The audit should be designed before any dashboard score is treated as real.

---

## 9. Current stop boundary

Do not yet:

- perform a major rewrite of `main.py` / `index.html`;
- create a composite excellence score from invented data;
- expand into the next musculoskeletal module;
- prioritize patient leaflets over the current system architecture;
- commit identifiable patient information to this public repository.

The next conversation/session can bootstrap by reading, in order:

```text
AGENTS.md
HANDOFF_CURRENT.md
TODO.md — section 0/1
CLINICAL_EXCELLENCE_PLAN.md
```
