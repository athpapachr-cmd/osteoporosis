# AGENTS.md — permanent operating rules for Clinical Excellence / Module 01 Osteoporosis

> **STATUS:** permanent project operating authority.
> **CANONICAL HOME:** `athpapachr-cmd/osteoporosis`.
> **SCOPE:** the reusable Clinical Excellence Core Engine and Module 01 — Osteoporosis, including the existing Osteoporosis Cockpit, future learning/audit layers, and every AI/Codex/ChatGPT session working on this repository.
> **PRODUCT OWNER:** the clinician using the system.

---

## 1. Canonical documentation architecture — CLOSED ACTIVE SET

The active canonical project set contains exactly five documents:

1. `AGENTS.md` — permanent operating rules.
2. `TODO.md` — long-range compass across phases.
3. `CLINICAL_EXCELLENCE_PLAN.md` — the one active detailed phase/architecture plan.
4. `HANDOFF_CURRENT.md` — exact current operational handoff.
5. `osteoporosis-change-log.md` — append-only historical logbook.

`README.md` is navigation only and is not a sixth source of architecture truth.

Permanent rule:

> **ONE CANONICAL PROJECT TRUTH.**

Do not create a second active roadmap, TODO, handoff, architecture contract or overlapping plan for a subject already governed by these files. If a future major phase needs a new detailed plan, archive the completed active plan unchanged and replace it with one new active phase plan only after explicit product-owner approval.

Historical material is preserved, not silently rewritten. Superseded plans belong under `archive/` with provenance.

---

## 2. Product purpose

This repository is no longer only an Osteoporosis Cockpit. The Cockpit is one point-of-care component of a broader **Personal Clinical Excellence System**.

The reusable system must connect:

```text
STANDARD
→ LEARN
→ TEST / MASTER
→ APPLY IN REAL CLINICAL PRACTICE
→ MEASURE
→ AUDIT
→ IDENTIFY GAP OR STRENGTH
→ TARGETED CHANGE / REINFORCEMENT
→ RE-MEASURE
→ SYSTEM LEARNS
```

Module 01 is **Osteoporosis**. The architecture must be reusable later for other clinical domains such as low-back pain, neck pain, knee pain, hip pain and shoulder pain without rebuilding the Core Engine from scratch.

Every substantial design decision must ask:

> **Is this reusable Core Engine behavior, or Osteoporosis-specific domain content?**

---

## 3. The system is a Learning Health System, not a static dashboard

The system must behave as a dynamic feedback organism.

Signals may originate from:

- real clinical encounters;
- patient feedback;
- audit findings;
- tests and learning performance;
- errors and near misses;
- new guidelines or evidence;
- courses, seminars and congresses;
- external benchmarks;
- sustained strengths;
- clinical outcomes and follow-up failures.

Canonical signal lifecycle:

```text
DETECT
→ CLASSIFY
→ ASSESS IMPORTANCE / RELIABILITY
→ LINK TO DOMAIN / STANDARD / COMPETENCY
→ DECIDE ACTION
→ IMPLEMENT
→ REASSESS
→ CLOSE OR CONTINUE
```

Nothing important should end as a passive note if it implies a reasonable action, learning need, safety intervention, benchmark comparison or re-audit.

---

## 4. Four gap classes — do not prescribe the wrong remedy

When performance is suboptimal, classify the cause before acting:

1. **Knowledge gap** — the relevant knowledge is missing or not retained.
2. **Reasoning gap** — the facts are known but the clinical decision process is weak or inconsistent.
3. **Execution gap** — the clinician knows what should happen, but the action did not reliably occur.
4. **Communication / system gap** — clinical reasoning may be sound, but workflow, documentation, handoff or patient understanding failed.

Do not respond to every gap with more reading or more courses.

Typical interventions:

- knowledge gap → targeted articles/courses/testing/spaced repetition;
- reasoning gap → cases, challenge mode, red-team review, peer/guideline comparison;
- execution gap → Cockpit/workflow/checklist/task changes;
- communication/system gap → teach-back, template/handoff changes, patient-feedback loop, workflow redesign.

---

## 5. Strengths are active signals too

The system must identify, validate and reinforce sustained strengths.

A positive result becomes a **SUSTAINED STRENGTH** only after sufficient repeated evidence, stable performance over time and appropriate audit/sample-size context.

A sustained strength should trigger:

- less basic repetition;
- more advanced cases;
- external comparison where valid;
- preservation of the workflow that produces the result;
- periodic surveillance rather than abandonment.

Do not waste learning time repeatedly drilling a domain already demonstrated to be stable and strong.

---

## 6. Transparent measurement — no black-box scores

Every progress bar or Clinical Excellence score must be explainable.

For each metric display or preserve, where applicable:

- current performance;
- baseline;
- absolute change / percentage-point change;
- trend over time;
- denominator and sample size;
- reliability/confidence of the estimate;
- target/standard;
- external benchmark if available;
- comparability of the benchmark;
- data completeness.

Never display a stable-looking percentage from an inadequate sample without a warning.

An overall score may summarize the system, but it must never hide constituent domain scores or safety-critical deficits.

Clinical outcomes such as fracture occurrence must not be naively converted into a clinician-competence penalty without appropriate risk/context adjustment.

---

## 7. Baseline before claims of improvement

Improvement must be measured against a real baseline, not an arbitrary starting score.

Until an adequate baseline audit is complete, the dashboard should state that baseline assessment is in progress rather than inventing a score.

The first formal Osteoporosis baseline audit will define initial practice measurements and later re-audit comparators.

---

## 8. Evidence governance — explicit, versioned, non-hybrid

Clinical rules must not silently blend incompatible guideline frameworks.

Every important rule should eventually carry explicit metadata:

```text
rule_id
clinical_domain
framework / guideline
version / year
recommendation or criterion
strength / certainty when available
linked evidence IDs
reviewed_on
status: current / review_due / superseded
```

If multiple frameworks differ, show them separately and explain the difference. Do not manufacture a synthetic threshold without explicit product-owner approval and clear labeling.

A new paper or conference item does not automatically change practice. Classify evidence impact as one of:

- practice confirming;
- interesting / no change;
- potentially practice changing;
- practice changing;
- insufficient / conflicting evidence.

Then identify which standards, Cockpit rules, learning material, patient information or audit KPIs are actually affected.

---

## 9. AI is support, not an untraceable clinical authority

AI-generated clinical support must distinguish:

- structured source data;
- deterministic rules;
- external evidence;
- inference;
- uncertainty;
- clinician override.

AI must not invent missing diagnoses, medication history, fracture history, patient preferences, laboratory values or treatment decisions.

Where a clinician accepts, modifies or rejects a recommendation, preserve the decision and rationale when clinically useful.

The clinician must always remain able to disagree with a guideline or AI recommendation. A reasoned override is not automatically an error.

---

## 10. Personalized challenge behavior

The system should support four explicit working modes:

1. **STANDARD** — ordinary clinical support.
2. **CHALLENGE** — identify important alternatives, omissions and weak assumptions.
3. **RED TEAM** — assume the current decision may be wrong and construct the strongest evidence-based counter-case.
4. **LEARNING** — convert the current case into a structured educational exercise.

High-confidence errors are higher-priority learning signals than low-confidence errors.

Track calibration where useful:

```text
accuracy vs stated confidence
```

The system should also protect against perfectionism-driven overwork by distinguishing:

- critical flaw;
- clinically meaningful improvement;
- optional refinement;
- cosmetic change.

When evidence is sufficient and the current objective is met, stop expanding scope without a new reason.

---

## 11. Patient Voice is part of the learning loop

Patient feedback is not only a satisfaction survey.

It should capture, where useful:

- understanding of the condition;
- understanding of the plan;
- understanding of treatment rationale/duration/risks;
- whether questions and preferences were addressed;
- free-text confusion, concern, praise or suggestion.

Repeated feedback patterns can create Signals and Improvement Projects.

A patient statement that reveals systematic misunderstanding must be capable of changing communication, handoff, education or workflow and must later be re-measured.

---

## 12. External benchmarking requires comparability

External data may be used to learn what high-quality practice looks like, but comparisons must be methodologically honest.

Each benchmark should eventually record:

```text
metric
source
country
population
clinical setting
year
definition
value
comparability: high / moderate / low / context only
```

Do not claim superiority or inferiority from a benchmark whose population, denominator or clinical setting is materially different.

---

## 13. Safety and privacy — public repository rule

This repository is public. Therefore:

> **NO IDENTIFIABLE PATIENT DATA MAY BE COMMITTED.**

Never commit:

- names;
- GeSY/EMR identifiers;
- phone numbers/emails/addresses;
- exact identifiable patient timelines;
- unredacted clinical documents;
- transcripts containing patient identifiers;
- secrets, API keys or credentials.

Use synthetic or fully anonymized fixtures/examples only.

Before any future production use with identifiable patient data, authentication, authorization, audit logging, secure storage, data minimization and applicable GDPR/privacy controls must be addressed outside public source files.

---

## 14. Repository/workflow discipline

Before substantial work:

1. read `AGENTS.md`;
2. read `HANDOFF_CURRENT.md`;
3. read the current section of `TODO.md`;
4. read `CLINICAL_EXCELLENCE_PLAN.md` for the active architecture/phase;
5. inspect relevant code/evidence before proposing implementation.

After meaningful work:

- update `HANDOFF_CURRENT.md` if current operational truth changed;
- update `TODO.md` if roadmap status changed;
- append to `osteoporosis-change-log.md` for completed historical events;
- update the active plan only when architecture/phase truth genuinely changes.

Do not rely on chat history as the sole project memory.

---

## 15. Current implementation boundary

The existing `index.html` and `main.py` represent the current Osteoporosis Cockpit/application baseline. They are not the complete Clinical Excellence System.

Until the Blueprint/Baseline phase is approved, prioritize architecture, data definitions and audit design over large runtime rewrites.

Patient handouts and educational leaflets remain downstream assets; do not let them displace current Core Engine / Module 01 design work unless the product owner explicitly changes priority.
