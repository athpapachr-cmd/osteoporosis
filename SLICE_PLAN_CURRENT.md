# SLICE_PLAN_CURRENT.md — contextual AI clinical considerations from visit audio

> **STATUS (2026-10-28): PRE-CODE R2 P1 CORRECTION / CLOSURE HOLD.** Independent review of design `a8389e59` BLOCKED P0:P1:P2=0:1:0: per-batch generation could not combine facts split across the existing 8KB classification boundary. The corrected design keeps batch classification unchanged and adds one bounded whole-visit `gpt-6-luna` consideration pass over all validated facts with global source IDs. Oversize evidence skips that optional pass and returns an explicit unavailable status so absence cannot masquerade as clearance. One focused independent finding/delta/affected-cumulative closure is next; no runtime mutation before PASS.

## Objective

Extend the released exact-source audio analysis with a visibly separate, source-cited AI reasoning layer that can propose bounded questions or clinical considerations inferred from the combined explicit facts. Keep every consideration transient and outside the clinical candidate, Preview and Save.

## Frozen behavior

- Existing `gpt-transcribe`, `gpt-6-luna`, standard-retention disclosure/gate, provider headers, `store:false`, no-tools call, temporary-content lifecycle, exact-source candidate, one final confirmation, Preview and explicit Save remain unchanged.
- Released typed `suggested_next_checks` remain the only suggestions eligible for the existing one bulk add-to-plan action.
- New free-text contextual considerations are display-only. They have no adoption button and never enter candidate, Preview, Save, pending obligations or the chart automatically.
- Manual Dia/amendment and local/native audio paths remain unchanged. Synthetic data only for implementation and release evidence.

## Provider and validation contract

Keep the existing strict per-batch `visit_audio_analysis_v2` response unchanged. After all batches pass validation and the exact-source candidate is built, make one additional bounded `gpt-6-luna` Responses call using only the merged validated facts (`source_id`, exact `excerpt`, `kind`) from the whole visit. It uses the same project/org headers, `store:false`, no tools and existing timeout/model settings.

The whole-visit input is permitted only when its serialized UTF-8 JSON is at most 60,000 bytes. Otherwise do not truncate or sample it: skip the optional consideration call, return `contextual_considerations_status=input_limit`, and show that unavailability visibly. A provider/network/invalid-output failure in this optional layer does not discard the already validated clinical draft; it returns `contextual_considerations_status=unavailable`, logs only fixed stage/kind/status metadata and no clinical content. `complete` means the whole validated fact set was considered, including a valid empty list; it is not clinical clearance.

The second strict response contains up to four `contextual_clinical_considerations`. Each item contains only:

```text
category: history_question | focused_examination | investigation_consideration |
          referral_consideration | management_review | safety_check
text: concise Greek question/consideration, 1..240 characters
rationale: concise Greek explanation, 1..320 characters
source_ids: 1..3 unique IDs from the current batch
confidence: low | moderate | high
```

Server validation rejects unknown/additional keys, unknown or duplicate global source IDs, IDs without a validated extracted fact, blank/oversize/control-character text and more than four items. Normalized text must also be unique. There is no per-batch merge or partial whole-visit evidence set.

The model instruction requires a question/consideration rather than diagnosis or order; forbids declaring a diagnosis, prescribing/starting/stopping/changing treatment or dose, claiming a test is required, inventing urgency/timing, or treating silence as a negative. These semantic instructions reduce unsafe output but are not treated as deterministic proof. Therefore the display-only authority boundary is mandatory.

## UI contract

- Separate block after the released typed next-visit checks: `Κλινικές σκέψεις AI από τα συμφραζόμενα`.
- Visible warning: inferred, may be wrong, was not said, is not guideline verification, an empty result is not clinical clearance, and nothing is recorded automatically.
- Each item shows category, text and confidence. A closed `Γιατί προτείνεται` disclosure shows the model rationale.
- No add/adopt/confirm action exists inside this block. The clinician may independently write an accepted idea into the editable plan and then use the existing final confirmation/Preview/Save.

## Evidence questions

1. A synthetic compound osteoporosis scenario can produce a new investigation consideration grounded in multiple validated source IDs while the exact-source candidate remains unchanged.
2. Unknown/unvalidated/duplicate source IDs, additional keys, invalid confidence/category and oversize/control-character prose fail closed.
3. Contextual consideration text is absent from candidate and from protected Preview/Save payloads.
4. Browser UI labels the inference clearly, shows rationale/confidence, exposes no adoption button and retains one final patient/content confirmation.
5. Existing typed next-check adoption, manual Dia/amendment labels, local audio and standard-retention behavior do not regress.
6. Facts placed on opposite sides of the 8KB classification boundary can support one whole-visit consideration; oversize and failed optional passes visibly return unavailable status without losing the exact-source draft.

## Review and release

R2: free model-authored clinical reasoning is visible to the clinician even though it cannot write. One independent pre-code design review and one independent exact-code fidelity review are required. Exact-head CI and guarded no-PHI release smoke remain separate gates. No agent recording, provider request or clinical write.
