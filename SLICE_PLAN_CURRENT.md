# SLICE_PLAN_CURRENT.md — contextual AI clinical considerations from visit audio

> **STATUS (2026-10-28): DESIGN WRITER / PRE-CODE R2 HOLD.** Product Owner explicitly requested implementation after confirming the desired behavior: the AI should suggest what may merit checking next from the combined clinical context, not merely repeat spoken follow-up. Fresh `main=0e4230425f5010167ea449a4b80b58008697ed79`; branch `codex/web-visit-contextual-clinical-suggestions` owns the bounded provider response, transient compact UI, focused synthetic tests and linked checkpoints. No runtime mutation before one independent pre-code R2 PASS.

## Objective

Extend the released exact-source audio analysis with a visibly separate, source-cited AI reasoning layer that can propose bounded questions or clinical considerations inferred from the combined explicit facts. Keep every consideration transient and outside the clinical candidate, Preview and Save.

## Frozen behavior

- Existing `gpt-transcribe`, `gpt-6-luna`, standard-retention disclosure/gate, provider headers, `store:false`, no-tools call, temporary-content lifecycle, exact-source candidate, one final confirmation, Preview and explicit Save remain unchanged.
- Released typed `suggested_next_checks` remain the only suggestions eligible for the existing one bulk add-to-plan action.
- New free-text contextual considerations are display-only. They have no adoption button and never enter candidate, Preview, Save, pending obligations or the chart automatically.
- Manual Dia/amendment and local/native audio paths remain unchanged. Synthetic data only for implementation and release evidence.

## Provider and validation contract

Extend the strict analysis response with up to four `contextual_clinical_considerations` per batch. Each item contains only:

```text
category: history_question | focused_examination | investigation_consideration |
          referral_consideration | management_review | safety_check
text: concise Greek question/consideration, 1..240 characters
rationale: concise Greek explanation, 1..320 characters
source_ids: 1..3 unique IDs from the current batch
confidence: low | moderate | high
```

Server validation rejects unknown/additional keys, unknown or duplicate source IDs, IDs without a validated extracted fact, blank/oversize/control-character text, more than four batch items and more than four merged items. Merge is deterministic and deduplicated by normalized text.

The model instruction requires a question/consideration rather than diagnosis or order; forbids declaring a diagnosis, prescribing/starting/stopping/changing treatment or dose, claiming a test is required, inventing urgency/timing, or treating silence as a negative. These semantic instructions reduce unsafe output but are not treated as deterministic proof. Therefore the display-only authority boundary is mandatory.

## UI contract

- Separate block after the released typed next-visit checks: `Κλινικές σκέψεις AI από τα συμφραζόμενα`.
- Visible warning: inferred, may be wrong, was not said, is not guideline verification and is never recorded automatically.
- Each item shows category, text and confidence. A closed `Γιατί προτείνεται` disclosure shows the model rationale.
- No add/adopt/confirm action exists inside this block. The clinician may independently write an accepted idea into the editable plan and then use the existing final confirmation/Preview/Save.

## Evidence questions

1. A synthetic compound osteoporosis scenario can produce a new investigation consideration grounded in multiple validated source IDs while the exact-source candidate remains unchanged.
2. Unknown/unvalidated/duplicate source IDs, additional keys, invalid confidence/category and oversize/control-character prose fail closed.
3. Contextual consideration text is absent from candidate and from protected Preview/Save payloads.
4. Browser UI labels the inference clearly, shows rationale/confidence, exposes no adoption button and retains one final patient/content confirmation.
5. Existing typed next-check adoption, manual Dia/amendment labels, local audio and standard-retention behavior do not regress.

## Review and release

R2: free model-authored clinical reasoning is visible to the clinician even though it cannot write. One independent pre-code design review and one independent exact-code fidelity review are required. Exact-head CI and guarded no-PHI release smoke remain separate gates. No agent recording, provider request or clinical write.
