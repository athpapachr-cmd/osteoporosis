# SLICE_PLAN_CURRENT.md — contextual AI clinical considerations from visit audio

> **STATUS (2026-10-28): PRODUCTION CLASSIFICATION FAILURE / PRE-CODE R2 HOLD.** Two fictional Product Owner recordings reached transcription but failed strict classification with `invalid_output`; shorter speech failed identically. PR #247 contextual reasoning is therefore not being released even though its prior head passed 15/15 CI. The bounded correction preserves the validator and adds at most one same-batch retry only for allowlisted structural codes plus content-free `validation_code` diagnostics; identity uncertainty/conflict never retries. Design: `cockpit/WEB_VISIT_AUDIO_CLASSIFICATION_RETRY_2026-10-28.md`. Independent pre-code R2 is required before runtime mutation.

## Objective

Extend the released exact-source audio analysis with a visibly separate, source-cited AI reasoning layer that can propose bounded questions or clinical considerations inferred from the combined explicit facts. Keep every consideration transient and outside the clinical candidate, Preview and Save.

## Frozen behavior

- Existing `gpt-transcribe`, `gpt-6-luna`, standard-retention disclosure/gate, provider headers, `store:false`, no-tools call, temporary-content lifecycle, exact-source candidate, one final confirmation, Preview and explicit Save remain unchanged.
- Released typed `suggested_next_checks` remain the only suggestions eligible for the existing one bulk add-to-plan action.
- New free-text contextual considerations are display-only. They have no adoption button and never enter candidate, Preview, Save, pending obligations or the chart automatically.
- Manual Dia/amendment and local/native audio paths remain unchanged. Synthetic data only for implementation and release evidence.

## Provider and validation contract

Keep the existing strict per-batch `visit_audio_analysis_v2` response unchanged. After all batches pass validation and the exact-source candidate is built, make one additional bounded `gpt-6-luna` Responses call using only the merged validated facts (`source_id`, exact `excerpt`, `kind`) from the whole visit. It uses the same project/org headers, `store:false`, no tools and existing timeout/model settings.

The second pass does not receive a separate `selected_identity` object or the complete `source_units` transcript. It does receive exact verbatim fact excerpts. Those excerpts can contain identifiers spoken in the visit; they are therefore covered by the same approved standard-retention disclosure and gate as the full transcript and selected-chart name/DOB. The model instruction forbids reproducing identity in its consideration prose, but that instruction is not presented as deterministic de-identification.

The whole-visit input is permitted only when its serialized UTF-8 JSON is at most 60,000 bytes. Otherwise do not truncate or sample it: skip the optional consideration call, return `contextual_considerations_status=input_limit`, and show that unavailability visibly.

The route creates one monotonic `pipeline_deadline` and a `core_deadline = pipeline_deadline - 5 seconds`. Transcription, every classification batch and candidate construction must complete before `core_deadline`. The implementation checks the monotonic clock before and after the synchronous candidate builder as well as bounding awaited core stages; crossing this cutoff is a pre-candidate fail-closed 502.

Only after the candidate passes the post-build clock check may the optional layer start. It receives at most `min(summarize_timeout, pipeline_deadline - now - 5 seconds)`. If this budget is below one second, it is not started and status is `unavailable`. Its own timeout, provider/network/invalid-output failure returns `unavailable` without escaping into the outer pipeline timeout. Thus every successfully established candidate retains the full five seconds for final patient-context verification and response construction. Optional failure logs only fixed stage/kind/status metadata and no clinical content. `complete` means the whole validated fact set was considered, including a valid empty list; it is not clinical clearance.

The second strict response contains up to four `contextual_clinical_considerations`. Each item contains only:

```text
category: history_question | focused_examination | investigation_consideration |
          referral_consideration | management_review | safety_check
text: concise Greek question/consideration, 1..240 characters
rationale: concise Greek explanation, 1..320 characters
source_ids: 1..3 unique global IDs from the current visit
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
6. Facts placed on opposite sides of the 8KB classification boundary can support one whole-visit consideration; oversize, deadline-exhausted and failed optional passes visibly return unavailable status without losing the exact-source draft. Core crossing its earlier cutoff returns 502 before candidate authority; a candidate built before cutoff survives optional deadline exhaustion.

## Review and release

R2: free model-authored clinical reasoning is visible to the clinician even though it cannot write. One independent pre-code design review and one independent exact-code fidelity review are required. Exact-head CI and guarded no-PHI release smoke remain separate gates. No agent recording, provider request or clinical write.

## Implementation evidence

- One additional strict `visit_audio_contextual_considerations_v1` request uses only the fully validated fact set and the existing `gpt-6-luna` settings, project headers, `store:false` and no tools.
- Server validation enforces shape, category/confidence enums, length/control bounds and global validated source IDs. Optional failure returns only `complete | input_limit | unavailable` plus a bounded list and never mutates the exact-source candidate.
- The compact browser renders a visibly separate block with category, confidence, a closed rationale disclosure and no adoption control. Its warning states that the content is inferred, may be wrong, was not spoken, is not guideline verification and is never saved automatically.
- Synthetic route tests cover facts across the 8KB batch boundary, invalid citations/shapes, the 60KB no-truncation gate and optional deadline exhaustion. The browser oracle proves one final confirmation remains and the AI prose is absent from the Save request.
- Local affected result after the deadline correction: `35 passed`; Python compile, JavaScript syntax and `git diff --check` pass. No provider request, real audio or clinical write occurred.
