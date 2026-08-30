# SLICE_PLAN_CURRENT.md — Osteoporosis Module 01 Closure Program v1

> **STATUS:** CLOSURE DESIGN FROZEN / C1 PILOT READINESS BLOCKED.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-CLOSE-v1.
> **Authoritative remote main at slice start:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Writer:** `design/module01-closure-program-2026-08-30`.
> **Runtime writer:** NONE.
> **Runtime implementation in this slice:** NOT AUTHORIZED.
> **Merge/deploy/preview:** NOT AUTHORIZED.

---

# 1. Problem

The project contains a large long-range Clinical Excellence roadmap, but the product owner now wants to stop expanding Clinic Utilities and finish the Osteoporosis proving module.

The risk is to confuse:

```text
"finish Module 01"
with
"implement every roadmap idea"
```

or, conversely, to label Module 01 closed while essential real-practice validation has never occurred.

This slice therefore defines a strict closure boundary and the shortest evidence-valid path to it.

---

# 2. Product decision

Effective now:

```text
physiotherapy expansion → PARKED
Osteoporosis Module 01 closure → PRIMARY PRIORITY
```

Existing CU-1 work is preserved. No further musculoskeletal disease-route rollout belongs in the Module 01 closure program.

Future non-osteoporosis disease work should use a condition-centered vertical slice:

```text
one canonical condition model
→ clinical assessment projection
→ physiotherapy referral projection
→ later reusable outputs
```

That future architecture does not need implementation in this slice.

---

# 3. What "Module 01 closed" means

Module 01 closure is a product/validation state, not a code-completeness label.

Closure requires evidence that Osteoporosis has actually proven the minimum reusable learning-health loop:

```text
CAPTURE
→ REVIEW
→ MEASURE
→ IDENTIFY GAP/STRENGTH
→ INTERVENE
→ RE-MEASURE
```

The module is not closed merely because forms, APIs or AI components exist.

---

# 4. Closure-critical gates

## Gate A — Baseline capture usability

Required evidence:

```text
5 consecutive eligible real osteoporosis encounters
```

For each case capture at minimum:

- post-visit completion time;
- friction/extra clicks or duplicated entry;
- ambiguous or missing fields;
- branching/applicability problems;
- persistence/save/reload defects;
- any safety/data-integrity issue;
- whether the encounter could be completed without changing the form mid-pilot.

Hard rule:

> Do not redesign after each case unless a safety, data-loss or persistence defect requires immediate correction.

After all five cases, make one deliberate refinement.

## Gate B — Freeze measurement contract

After the pilot refinement:

- freeze Baseline Form v1;
- freeze KPI applicability/calculation contract;
- freeze denominator/exclusion/missing-data semantics;
- confirm scored-baseline exposure policy;
- record any unavoidable pre-baseline intervention exposure.

## Gate C — Transcript-assisted capture

Minimum engineering capability:

```text
paste transcript
→ ephemeral processing
→ structured candidates
→ deterministic target mapping
→ no raw transcript persistence/logging by default
```

Must preserve negation, temporality, source/speaker, uncertainty and the distinction between discussion/recommendation/preference/final decision.

Provider output must not author application storage paths.

## Gate D — Clinician acceptance boundary

Before candidate data become authoritative:

```text
Accept / Reject / Edit
```

with conflict handling, clinician-review state and provenance.

No candidate may silently overwrite authoritative clinical truth.

## Gate E — Minimum viable Practice Review

Quick Practice Review must be capable of producing structured, evidence-traceable `PracticeObservation` objects across the agreed dimensions with:

```text
direction
importance
confidence
encounter provenance
linked standard/evidence when material
suggested change
clinician disposition
```

It may operate in shadow mode before baseline lock.

## Gate F — Scored baseline

Required default evidence:

```text
30 consecutive unique eligible osteoporosis encounters
```

During the scored baseline:

- routine coaching remains hidden;
- no red/green KPI behavior-shaping prompts;
- safety-critical exception paths remain allowed;
- exposure contamination is recorded.

Alternative: the 30-case baseline may be replaced only by an explicit product-owner-approved methodological revision recorded canonically before the resulting cohort is described.

## Gate G — Closed improvement loop

After baseline policy permits clinician-facing intervention, Module 01 must demonstrate at least one complete real-practice loop:

```text
repeated observation / measured gap or strength
→ denominator-aware Signal
→ root-cause classification
→ appropriate intervention
→ later re-measurement
→ explicit result: improved / unchanged / worsened / insufficient evidence
```

A one-off AI critique is insufficient.

## Gate H — Adaptive workflow informed by real evidence

The osteoporosis visible workflow/Close behavior must incorporate the highest-value changes supported by pilot/review evidence while retaining the canonical storage/audit schema underneath.

No universal workflow redesign from one transcript or one preference.

## Gate I — Closure review

Final review must verify:

- all above gates have evidence or explicit approved exception/revision;
- unresolved critical safety/data-integrity defect = none;
- fresh canonical bootstrap reconstructs the state without chat history;
- Module 01-specific domain content can now be distinguished from reusable Core mechanics;
- deferred roadmap items are explicitly non-blocking rather than silently forgotten.

Only then may `MODULE 01 CLOSED` be written into the canonicals/changelog.

---

# 5. Explicitly NOT required for Module 01 closure

Unless later evidence elevates one to a safety/data-integrity dependency, the following remain post-closure or independent work:

- full Deep Review feature parity;
- full Red Team productization;
- exhaustive Decision Reconstruction UI;
- comprehensive Patient Voice program;
- external Benchmark Registry;
- complete Clinical Excellence Home/analytics polish;
- all possible standards/competencies/learning resources across every subdomain;
- Calendar/Setmore/Zadarma/CareTask integration;
- Radiofrequency utility implementation;
- new physiotherapy disease routes;
- Module 02 selection/generalization.

The minimum evidence/standards needed to support material Practice Review claims remains closure-critical; exhaustive content coverage does not.

---

# 6. Sequence and dependencies

```text
C0 Closure program + canonical priority rebase                  COMPLETE
↓
C1 pilot finalization-integrity readiness correction            BLOCKED / NEXT RUNTIME SLICE
↓
C1b 5-case real pilot                                           NOT STARTED
↓
C2 one post-pilot refinement                                    NOT STARTED
↓
C2b freeze Baseline Form/KPI contract                           NOT STARTED
↓
C3 PR-1 transcript extraction                                   NOT STARTED
↓
C4 PR-2 clinician candidate review/merge                        NOT STARTED
↓
C5 Quick Practice Review shadow mode                            NOT STARTED
↓
C6 30-case scored baseline + lock                               NOT STARTED
↓
C7 expose reviewed Signals/interventions                        NOT STARTED
↓
C8 adaptive workflow refinement from accumulated evidence       NOT STARTED
↓
C9 re-measure at least one intervention/pattern                 NOT STARTED
↓
C10 final Module 01 closure review                              NOT STARTED
```

PR-1/PR-2/Practice Review engineering may be designed around pilot findings, but no systematic coaching may contaminate the scored baseline without explicit methodology revision.

---

# 7. Current preserved assets

Already proven and reusable for closure:

- Baseline Audit Steps 1–6;
- pre-pilot hardening and 14-scenario synthetic smoke;
- PostgreSQL patient/encounter/lab persistence;
- browser-session clinical authentication;
- patient load/save/reload and longitudinal lab smoke;
- server-side encounter finalization state-machine integrity + 3/3 live synthetic validation;
- baseline methodology and KPI draft contracts;
- Practice Review / Signal / gap-class architecture;
- corrected PR-1 v3 pre-code design archived for restart;
- adaptive consultation-flow candidate architecture;
- CU-1 production baseline as an independent Clinic Utility;
- later CU-1 rich-referral enhancement branch preserved but unmerged.

Do not rebuild these from scratch merely because the active priority changed.

---

# 8. Pilot-readiness result — BLOCKED

Read-only verification confirms that current `main` contains the expected Steps 1–6 pilot flow, protected patient/encounter persistence and the existing eligibility/baseline methodology. No later unrelated branch is required for pilot use.

However, the integrated Step-6 finalization path has a concrete ownership defect.

## 8.1 Current browser event wiring

`pilot-completion.js` owns a capture-phase listener on `#finishVisitBtn` and performs:

```text
preventDefault()
stopImmediatePropagation()
click Save
mark local pilot_completion=complete
```

`patient-registry.js` separately binds:

```text
Save → syncActiveEncounter("draft")
Finish → syncActiveEncounter("completed")
```

Because the capture-phase Finish handler stops immediate propagation, the later `Finish → completed` server synchronization is not a reliable executed path. The Save action triggered inside the pilot-completion handler schedules a `draft` synchronization.

The resulting possible state is internally contradictory:

```text
browser/local pilot completion: complete
protected server encounter status: draft
```

This violates pilot persistence/finalization integrity and blocks use of real pilot cases until corrected.

## 8.2 Existing evidence does not close this exact gap

`test_encounter_finalization.py` validates `resolve_encounter_status()` after the requested status reaches the server. It proves completed/amended transition semantics but does not drive the browser Finish event chain.

The prior 3/3 live synthetic finalization smoke remains valid for the tested completed/amended scenarios, but it did not establish an integrated invariant that Step-6 Finish necessarily delivers `requested status=completed` through the current capture/bubble listener composition.

---

# 9. Bounded C1 correction design

The next implementation must create one authoritative finalization operation rather than two competing click-listener owners.

Required semantic result:

```text
Step-6 Finish
→ snapshot/save all current module state
→ mark pilot completion locally
→ synchronize the same final encounter payload to the protected server as completed
→ await/verify success
→ only then present successful completion state
```

Implementation may refactor event ownership as needed, but must NOT change clinical fields, KPI meaning, pilot eligibility, scored-baseline methodology or unrelated registry behavior.

### Acceptance fixtures / integrated regression

Required:

```text
1. new/draft encounter + Step-6 Finish
   local pilot_completion.status == complete
   server status == completed
   server payload contains final Steps 1–6 state

2. reload/reopen
   completed encounter remains loadable and completed

3. no-op Save after completion
   completed remains completed

4. material edit + Save
   completed becomes amended

5. Finish without an active protected patient/server link
   must not falsely imply protected-server completion;
   behavior/error state must be explicit
```

A browser/integration-level test is required in addition to the existing server transition unit test.

---

# 10. Pilot protocol after C1 passes

The five pilot cases are:

```text
5 consecutive eligible real osteoporosis encounters
```

Eligibility follows the existing baseline schema:

- adult encounter;
- osteoporosis, osteopenia, fragility-fracture risk or osteoporosis treatment materially assessed or managed;
- includes new assessment, follow-up, post-fracture review, treatment review or transition;
- excludes purely administrative contacts and encounters outside osteoporosis clinical scope.

Unlike the later core scored baseline, pilot design does not require 30 unique patients; its purpose is usability/capture validation. Pilot cases are not included in the locked scored baseline because the form may change.

For each pilot case record:

- completion time;
- friction/duplicated work;
- ambiguous/missing fields;
- branching/applicability issues;
- persistence/load/reload behavior;
- safety/data-integrity issue if any.

Do not alter the form after individual pilot cases unless safety/data-loss/persistence requires immediate correction. After all five, make one deliberate refinement and freeze the measurement contract before scored-baseline accrual.

---

# 11. REPLAN triggers

Stop and replan if:

- the C1 correction reveals broader ownership conflict across Save/Finish/patient sync rather than the bounded seam above;
- real pilot shows the current form cannot capture core eligible osteoporosis encounters reliably;
- persistence or finalization integrity fails in real use;
- current KPI applicability cannot be frozen without material new clinical modelling;
- transcript workflow cannot remain ephemeral/PHI-safe at the required provider/API boundary;
- Practice Review requires hidden model opinion without traceable evidence for material clinical claims;
- the scored baseline cannot be conducted without unavoidable systematic coaching exposure;
- a closure-critical requirement expands into a nonessential platform build.

---

# 12. Acceptance of this planning slice

This closure-planning slice is now complete at design/readiness level:

```text
closure-critical boundary frozen                    YES
deferred/non-blocking boundary frozen               YES
physiotherapy parked/preserved by exact branch+SHA  YES
TODO priority/order reconciled                      YES
pilot-readiness inspected                           YES
pilot readiness                                     BLOCKED
exact blocker defined                               YES
bounded next implementation scope defined           YES
runtime mutation in this slice                      NO
```

STOP. The next action is a separately authorized bounded C1 pilot-finalization-integrity implementation slice followed by readiness re-verification.
