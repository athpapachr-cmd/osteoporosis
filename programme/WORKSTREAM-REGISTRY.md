# Clinical Excellence programme — workstream registry

> **ROLE:** navigation and coordination only.
> **NOT AN ACTIVE CANONICAL AUTHORITY:** the repository's six canonical authorities remain unchanged.
> **Root operational owner:** `CURRENT_OPERATIONAL.md`.

This registry exists to let multiple bounded workstreams proceed in parallel without turning the repository into one undifferentiated project.

## Workstreams

| Workstream | Purpose | Current state | Authority boundary |
|---|---|---|---|
| **OST-CAPTURE** | Heidi/transcript → semantic clinical candidates | **ACTIVE under root PR-1 lifecycle** | Root `CURRENT_OPERATIONAL.md` / `SLICE_PLAN_CURRENT.md`; not mutated by this programme bootstrap |
| **OST-LIFECOURSE** | Shared longitudinal patient truth, care trajectory, events/state/goals/decisions/obligations | **P1 CORRECTED OWNERSHIP MAP / INDEPENDENT DELTA+CUMULATIVE PASS / R2 INPUT READY** | Corrected exact target `b4f917b161f33b0bfb3507328df07cec2d7bd2b6`; DXA/task residuals closed; Q1–Q12 remain parked; no P2/runtime authority |
| **OST-CLINICAL** | Osteoporosis evidence, pathways, guidance rules and treatment logic | **S1 P6B PASS / PR #123 MERGED / AUTO-DEPLOY LIVE / SMOKE NOT RUN** | Independently reviewed substantive target `ba2f8635372f85cd94409fa66e08c3d8b428bcba`; squash merge `dd16a505662606f588a2adc8b1ae0ddddd884d19`; merge checkpoint `3c7e502e29bfbc103dc7838a902f3e7adcfca36f`; Render `dep-daumvhu417fc73ffcgn0` LIVE; production smoke requires separate authority |
| **OST-UI** | Osteoporosis product reconstruction: longitudinal product model, workflow/UX, information architecture and reuse | **R2 COMPLETE / COORDINATOR ACCEPTED / R3 AUTHORIZED** | R2 exact reviewed head `75a18ae7bfad1b812f73fbce9e4f82bb56c49dde`; longitudinal substrate reusable but incomplete for guaranteed 10–15-year reconstruction; R3 point-of-care interaction is next; no root-writer or clinical-semantic authority |
| **OST-REVIEW** | Practice Review, decision audit, Signals and improvement loop | **QUEUED** | Existing Practice Review architecture preserved; no new mutation authority |
| **OST-LEARNING** | Clinical Learning Hub / challenges / longitudinal clinician learning | **EXISTING PARALLEL TRACK** | Existing learning contracts and release state remain separately authoritative |
| **OST-SAFETY** | Privacy, provenance, CDS/regulatory and data-governance guardrails | **CROSS-CUTTING / NOT YET A STANDALONE WRITER** | Must be consulted when relevant; does not own product truth |
| **OST-LIVE** | Future live in-consultation Clinical Copilot | **PARKED / FUTURE** | Architecture constraint only; no runtime implementation |
| **PHYSIO** | Physiotherapy product evolution using Knee OA as reference implementation | **READY FOR SEPARATE COORDINATOR** | May run in parallel; must not mutate Osteoporosis workstreams unless shared-Core change is explicitly coordinated |
| **COCKPIT HOME** | Global navigation/home and shared-surface presentation | **RELEASED / separate sidecar state** | `cockpit/CURRENT.md` |

## Coordination rules

1. **One patient truth, no duplicated owner.**
2. **Reuse before new path.** Before proposing a new mechanism, inspect whether current runtime/Core already owns the responsibility.
3. **Coordinator != implementation author != independent reviewer.**
4. A workstream may inspect another workstream's outputs, but may not silently mutate its owned scope.
5. Cross-cutting changes that alter shared patient/Core semantics return to programme-level synthesis before implementation.
6. Workstream CURRENT files are sidecars for local continuity; they do not replace the root writer lock.
7. Product Constitution principles are product intent, not permission to bypass current runtime/safety gates.

## Programme sequencing — current

```text
Product Constitution v0.2
        ↓
OST-LIFECOURSE Phase 0 reuse/gap inventory — COMPLETE
        ↓
programme synthesis — COMPLETE
        ↓
OST-LIFECOURSE-P1 current-owner reconciliation — CORRECTED AFTER REVIEW BLOCK
        ↓
independent P1 delta+cumulative review — PASS
        ├── OST-LIFECOURSE successor work remains gated on P1 disposition
        ├── OST-CLINICAL S1 merged / auto-deploy live; production smoke not run
        └── OST-UI R1 current-product audit — COMPLETE / ACCEPTED
                ↓
           OST-UI R2 longitudinal trajectory review — COMPLETE / ACCEPTED
                ↓
           OST-UI R3 point-of-care interaction — AUTHORIZED
                ↓
           R4 → one synthesis → Product Owner decisions
        ↓
target architecture
        ↓
bounded implementation slices
```

In parallel, PHYSIO may proceed under a separate coordinator and return shared-Core findings to programme level. `OST-UI` is not an additional owner beside `OST-PRODUCT`; it is the activated name/scope of that previously queued placeholder.

## Explicit current non-goals

This registry does not:

- change the root PR-1 writer lock;
- authorize schema/runtime/UI mutation;
- resolve the 12 parked Product Constitution questions;
- activate Live Copilot;
- activate a patient app;
- create a new clinical recommendation engine;
- merge physiotherapy and osteoporosis into one module.
