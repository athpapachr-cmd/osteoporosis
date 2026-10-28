# SLICE_PLAN_CURRENT.md — web audio semantic completeness and next-visit suggestions

> **STATUS (2026-10-28): PRODUCT OWNER REPLAN / PRE-CODE R2 HOLD.** A successful fictional web recording produced a compact but clinically incomplete draft. The Product Owner's Dia comparison shows missing symptoms, treatment, plan and obligations and requests source-informed next-visit suggestions. The prior native recording/ASR qualification is preserved as deferred at `cockpit/archive/SLICE_PLAN_NATIVE_AUDIO_ASR_QUALIFICATION_DEFERRED_2026-10-28.md`. New design owner: `cockpit/WEB_VISIT_AUDIO_SEMANTIC_COMPLETENESS_2026-10-28.md`.

## Objective

Preserve exact-source clinical truth while allowing multiple exact excerpts from one compound transcript unit, populate the existing structured treatment/plan/pending/follow-up sections, and add a strictly separate transient AI suggestion layer for next-visit checks.

## Frozen boundary

- Existing `gpt-transcribe`, `gpt-6-luna`, standard-retention disclosure/gate, patient context, temporary content lifecycle, one final confirmation, Preview and explicit Save remain.
- Source-backed candidate text must be an exact validated transcript substring. No generated clinical prose, medication structure, dates, diagnoses or roles enter the record.
- AI suggestions may infer later questions/checks from cited source units but remain outside the candidate until one bulk clinician adoption into follow-up.
- Dia output is comparison evidence only, not source truth. Synthetic data only in implementation/release evidence.

## Gates and next action

R2 because classification semantics, pending obligations and generated suggestions change. Writer `codex/web-visit-semantic-completeness` owns only the design, bounded provider contract/builder/UI/tests and related checkpoints. One independent pre-code R2 PASS is required before runtime mutation; then focused implementation/tests, one exact-code R2, exact-head CI and guarded release. Native real-chart work remains deferred and no real patient audio/provider request/clinical write is authorized.
