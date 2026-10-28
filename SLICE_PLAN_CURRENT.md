# SLICE_PLAN_CURRENT.md — web audio semantic completeness and next-visit suggestions

> **STATUS (2026-10-28): CI MANUAL-LABEL ISOLATION CORRECTION / CLOSURE HOLD.** Corrected clinical semantics passed R2, but PR #245 Chromium proved audio-specific wording had leaked into shared manual/amendment labels. The bounded delta restores shared labels, scopes enhanced wording to compact audio only, and corrects the new Save-payload oracle. One focused independent closure and fresh exact-head CI are required. The prior native recording/ASR qualification remains deferred at `cockpit/archive/SLICE_PLAN_NATIVE_AUDIO_ASR_QUALIFICATION_DEFERRED_2026-10-28.md`.

## Objective

Preserve exact-source clinical truth while allowing multiple exact excerpts from one compound transcript unit, populate the existing structured treatment/plan/pending/follow-up sections, and add a strictly separate transient AI suggestion layer for next-visit checks.

## Frozen boundary

- Existing `gpt-transcribe`, `gpt-6-luna`, standard-retention disclosure/gate, patient context, temporary content lifecycle, one final confirmation, Preview and explicit Save remain.
- Source-backed candidate text must be an exact validated transcript substring. No generated clinical prose, medication structure, dates, diagnoses or roles enter the record.
- AI suggestions may infer later questions/checks from cited source units but remain outside the candidate until one bulk clinician adoption into follow-up.
- Dia output is comparison evidence only, not source truth. Synthetic data only in implementation/release evidence.

## Gates and next action

R2 because classification semantics, pending obligations and generated suggestions change. Pre-code correction closure passed 0:0:0. Writer `codex/web-visit-semantic-completeness` owns only the bounded provider contract/builder/UI/tests and linked checkpoints. One independent post-code exact-head R2 is now required; then canonical-impact PR, exact-head CI with Chromium and guarded release only if all remain PASS. Native real-chart work remains deferred and no real patient audio/provider request/clinical write is authorized.
