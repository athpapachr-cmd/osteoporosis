# PROCEDURES.md — bounded operating procedures

> **ROLE:** procedural authority for work in `athpapachr-cmd/osteoporosis`; not a seventh project-state canonical.
> **SOURCE PATTERN:** the task routing, reuse, evidence, closure and plan-impact safeguards in `ortho-reception-ops/AGENTS.md` and `PROCEDURES.md`. This is a lean adaptation for Clinical Excellence, Cockpit and PHYSIO.

`AGENTS.md` owns permanent invariants. The six root canonicals own project state and design. `CURRENT_OPERATIONAL.md` alone owns the repo-wide writer lock. A workstream `CURRENT.md` records its own sidecar state, never a second root lock. This file says **how** to perform a task; it cannot grant mutation, release, deployment or patient-data authority.

## P0. Source and task-mode router

1. Fresh-verify remote `main` and read the six root canonicals in the order required by `AGENTS.md` before substantial work. Then read the applicable workstream `CURRENT.md` and product/technical owners. If remote verification is unavailable, label the limitation and do not assert a fresh-main identity.
2. Record the exact base/head, active slice, writer scope, existing evidence, open blockers and next lawful action. A previous chat, PR body or old handoff is a pointer, not current truth.
3. Select the mode below. Load only the sources needed for that mode; recheck affected current truth when the head, lock, evidence or decision changes. Reuse settled unrelated facts.

| Task | Mode | Required action |
|---|---|---|
| Explain a failure or discrepancy | INVESTIGATE | Reconstruct source and first divergence; distinguish observed fact from inference. |
| Define or materially change a slice | DESIGN | Apply P1–P3; update the owning design/plan. |
| Implement an approved slice | IMPLEMENT | Check writer scope and frozen behavior; change only the approved seam. |
| Select or run evidence | EVIDENCE | Name the question and smallest adequate check; apply P3. |
| Assess a design or code candidate | REVIEW | Apply P4–P5 on the exact affected surface; read existing contradictory evidence. |
| Edit governance or canonicals | GOVERNANCE | Apply P7 and one bounded consistency audit. |
| Merge, deploy or smoke | RELEASE | Use the existing release/approval/checkpoint rules in `AGENTS.md`; review PASS is not release authority. |
| Close a slice or select the next | CLOSURE | Apply P6 and checkpoint the owning `CURRENT`. |

Investigation and independent review are read-only by default. A fresh conversation never acquires an overlapping writer lock by inference.

## P1. Product Owner plain-language checkpoint

Before activating each **new implementation-bearing slice** or material behavior change, present one short step card in the Product Owner's language:

```text
Problem now; what the clinician/patient sees today; what changes after this step;
one concrete example; what stays outside this step; why this step precedes the next;
the next milestone.
```

Record the Product Owner's confirmation or correction in the owning slice/current before implementation authority is claimed. Do not repeat this checkpoint for every code edit, docs sync, mechanical repair or bounded correction inside already confirmed behavior. A new behavior, owner or side effect needs a new checkpoint.

## P2. Existing-mechanism reuse gate

Before a new owner, engine, store, integration path or model call, identify the closest working mechanism and its responsibility. Choose `REUSE`, `EXTEND`, `REBIND`, `REPLACE` or a genuinely distinct adjacent role; state why the existing mechanism cannot meet the missing need and whether two paths could create the same effect.

Full overlap without a source-proven reason blocks a parallel mechanism. Check G3 longitudinal projections, the protected patient record, Clinical Calendar, Setmore, Digital Secretary and Zadarma where relevant. Their ownership is defined by the phase plan and `cockpit/PRODUCT_CONSTITUTION.md`; this procedure adds no product authority. A broken path requires evidence before replacement, and a bypassed path should be considered for rebinding before compensating duplication.

## P3. Finite evidence and reuse

Each test or check must answer a named material question about the changed seam, preserved behavior, safety, release or current-state identity. Use the smallest adequate evidence path and inspect already-produced applicable CI/checks before requesting a rerun.

```text
same bytes + same harness/state + same evidence question → reuse evidence
new bytes, changed seam, contradictory evidence or new material risk → targeted new evidence
```

Do not rerun tests merely because a reviewer is reading the work or to certify a prior review. Do not ignore an established targeted regression for a changed shared seam. Stop adding slice-local tests when its questions are answered; a separate release gate may still need applicable existing checks.

## P4. Review classification and proportionality

Classify the **actual changed behavior and affected authority**, not the file extension or workstream name. Record the tier and reason in the owning checkpoint.

| Tier | Typical change | Default independent review |
|---|---|---|
| **R0** | Documentation, evidence capture, checkpoint or canonical reconciliation with no runtime or material design-semantic delta | None; author performs one bounded consistency/self-audit. |
| **R1** | Bounded UI or deterministic read-only projection using unchanged proven Core and no new clinical/identity/write authority | One post-code exact-head review of implementation fidelity. |
| **R2** | Patient identity/linkage, clinical state or semantics, authoritative writes, shared Core, safety/privacy, new external effects or changed ownership | One pre-code design review and one post-code exact-head fidelity review. |

An unchanged proven Core second-diagnosis vertical is normally R1; changing that Core or clinical safety/authority is normally R2. Existing accepted pre-code evidence can be reused when the design question and semantics remain unchanged. A material design change reopens only its affected gate. Release, deploy and smoke are separate decisions, not extra architecture reviews.

For each material gate, use **one independent review by default**. A finding must identify a reachable behavior or genuine safety/authority risk, the violated invariant, evidence and smallest correction. `UNKNOWN` is not `PASS`; no reviewer has to invent a defect to prove independence.

## P5. Correction closure and stop

```text
BLOCK → bounded correction → ONE independent delta + affected cumulative closure review
```

The closure reviewer verifies each original material finding, the affected consumers/invariants and any genuinely new material risk. Reuse settled evidence; do not restart the entire architecture or test matrix for reassurance. If the correction changes the product behavior or owner outside the confirmed slice, return to P1/design and classify that new risk. An author cannot independently certify closure of their own material correction.

```text
ALL MATERIAL FINDINGS CLOSED
+ NO NEW MATERIAL RISK
+ AFFECTED SURFACE COVERED
→ STOP REVIEW CHAIN
```

A new source-proven material defect is a **new finding** with its own bounded disposition, not a reason to review whether the previous review was correct. Canonical sync, release-note publication, SHA restatement or reconciliation without runtime/design delta does not create another product review. A docs-only governance patch receives the R0 consistency audit in P7 and then stops.

## P6. Plan impact and slice closure

For a material finding, first ask whether it is already in `TODO.md`, the phase plan, current slice or a future workstream plan. Classify `ALREADY PLANNED`, `CURRENT BLOCKER`, `REPLAN`, `FUTURE CHANGE`, `PHASE ORDER CHANGE`, `NO PLAN CHANGE` or `UNKNOWN`. **ALREADY PLANNED → no duplicate task.** Update only the owning canonical if priority, dependency, behavior or completion state actually changes.

Close a slice only after its applicable evidence and release/smoke state are accurately distinguished. Checkpoint the owning `CURRENT`, update roadmap/phase/history only for their own facts, release the writer scope and name one next bounded action. An implementation PASS is not a claim of merge, deployment, real-use validation or permission to start a dependent slice.

## P7. Governance/canonical mutation

Choose the owner before editing: permanent invariant → `AGENTS.md`; reusable method → this file; broad roadmap → `TODO.md`; phase architecture → `CLINICAL_EXCELLENCE_PLAN.md`; active design → `SLICE_PLAN_CURRENT.md`; root NOW/lock → `CURRENT_OPERATIONAL.md`; local NOW → workstream `CURRENT.md`; closed history → changelog; navigation → registry/index. Do not duplicate authority across them.

Identify incompatible active statements, then replace, remove or explicitly mark them historical. Preserve factual history. Check changed-file scope, `git diff --check`, the repository's canonical-impact guard where applicable, and cross-references/contradictions among affected owners. For an R0 change, this **one bounded canonical consistency/self-audit** is the closure gate; stop unless it finds a material new risk.
