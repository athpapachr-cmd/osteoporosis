# Web Visit Audio — contextual AI clinical considerations

> **Implementation checkpoint (2026-10-28): POST-CODE P1 CORRECTED / CLOSURE HOLD.** Review of `e593052e` found post-candidate merge/context work could consume the five-second reserve. The corrected candidate keeps merging, evidence serialization and the pre-optional patient-context read under `core_deadline`; route tests now exercise both near-cutoff optional skip and an actual optional timeout. All 35 affected tests pass, including Chromium via Dia. Focused independent closure is next; no provider request, real audio or clinical write occurred.

## Product Owner checkpoint

**Problem now:** the released system can preserve explicit symptoms, therapy, plan and obligations and can generate nine bounded follow-up intents. It still cannot say that a combination of facts may merit an additional clinical question, focused examination or possible investigation unless that step was already spoken.

**After this step:** `gpt-6-luna` may return up to four concise considerations inferred from validated explicit facts, each with category, rationale, confidence and one to three cited source IDs. They appear in a separate display-only block and never enter the patient record automatically.

**Concrete example:** explicit thoracic pain plus reported height loss may produce `Να εξεταστεί αν χρειάζεται έλεγχος για σπονδυλικό κάταγμα`, with the two source IDs and a rationale. It is labelled as an AI consideration, not a diagnosis or order.

**Outside this step:** no automatic diagnosis, prescribing, dose change, test order, referral, urgency claim, guideline-authority claim, candidate mutation, obligation creation, one-click adoption, new provider or model, retention/configuration change or real-patient agent test. One additional same-model call is required so the whole validated visit context is considered once.

**Why now / next milestone:** the Product Owner explicitly asked for reasoning from context after the exact-source completeness correction went live. The next milestone is synthetic implementation evidence and one independent exact-code R2 before any PR/release decision.

## Authority boundary

Contextual consideration prose is provider-authored and therefore cannot inherit the exact-source authority of the clinical candidate. Prompt and schema constraints are useful but cannot deterministically prove clinical correctness. The safe boundary is:

```text
validated exact-source facts
→ transient model consideration + rationale + confidence + cited source IDs
→ clinician reads it as decision support
→ no direct candidate/Preview/Save path
→ clinician independently documents any accepted plan
```

The released typed next-visit checks remain separate: fixed server wording and one bulk adoption into follow-up. Contextual considerations do not reuse that adoption path.

## Strict response shape

The existing strict per-batch `visit_audio_analysis_v2` response remains unchanged. After all facts are validated and merged, one strict `visit_audio_contextual_considerations_v1` call receives the whole list of validated facts with their global source IDs. Its response contains `contextual_clinical_considerations[]`, maximum four:

- `category`: one of six fixed consideration categories;
- `text`: 1..240 characters;
- `rationale`: 1..320 characters;
- `source_ids`: one to three unique global visit source IDs, all with at least one validated extracted fact;
- `confidence`: `low`, `moderate` or `high`.

No extra keys. Text is normalized, must not contain control characters and must be unique. Every cited ID must belong to at least one validated fact in the whole visit. The prose is returned only in the transient audio-draft response.

## Whole-visit completeness and availability

The second-pass input is serialized validated facts only and must be at most 60,000 UTF-8 bytes. The server never truncates or samples it because that would silently weaken the meaning of whole-visit reasoning.

The audio-draft response includes:

```text
contextual_considerations_status: complete | input_limit | unavailable
contextual_clinical_considerations: [] | validated items
```

- `complete`: the model saw the complete validated fact set; the list may be empty, but that is not clinical clearance;
- `input_limit`: no consideration call ran because the full fact set exceeded the bound;
- `unavailable`: the optional call failed or its response failed validation; the exact-source draft remains available.

The optional pass reuses `gpt-6-luna`, project/org headers, `store:false`, no tools and current reasoning settings. The route creates one monotonic pipeline deadline and a core cutoff five seconds earlier. Transcription/classification/candidate construction is bounded by the earlier cutoff, with explicit clock checks immediately before and after the synchronous candidate builder. Crossing it is a pre-candidate fail-closed 502.

After a candidate passes the cutoff, the optional pass may use only `min(summarize_timeout, pipeline_deadline - now - 5 seconds)`; if less than one second remains, it is skipped as `unavailable`. An optional timeout is caught inside the whole-visit layer. Therefore every successfully established candidate keeps five seconds for final patient-context verification and response construction and cannot be discarded by optional reasoning. Failure logging contains only fixed stage/kind/provider-status metadata. No raw transcript or patient identity is sent in this second pass.

## Prompt semantics

The model may identify a question or consideration that follows from the combination of explicit facts. It must not:

- declare a new diagnosis;
- prescribe, start, stop or change treatment/dose;
- state that a new test/referral is required;
- invent urgency, timing or a follow-up interval;
- infer a negative from silence;
- claim guideline/evidence authority;
- reproduce patient identity in the suggestion.

Because semantic prompt compliance is not a deterministic safety control, violations remain possible model error. The response is therefore labelled potentially wrong and has no write/adoption control.

## Browser behavior

The block is rendered when considerations exist. Each card shows a Greek category label, the consideration and confidence. Rationales sit inside one closed disclosure. The warning explicitly distinguishes inferred advice from what was said and from evidence-backed guidance, says that an empty result is not clinical clearance, and disappears with the transient draft. When status is `input_limit` or `unavailable`, a compact visible note explains that contextual suggestions were not generated; it never hides that state behind an empty block. All status/prose stays outside the candidate object.

Any clinician who accepts the idea must independently enter the appropriate wording into the existing editable plan. That ordinary edit invalidates the final confirmation and prior Preview under the existing contract.

## R2 questions

1. Does display-only separation fully prevent provider prose from entering candidate/Preview/Save?
2. Are cited IDs constrained to validated whole-visit facts and deduplicated safely?
3. Does the UI avoid implying diagnosis, order or guideline authority?
4. Do existing typed suggestions and one-confirmation behavior remain unchanged?
5. Does the bounded whole-visit pass actually see validated facts across classification batches, with no silent truncation?
6. Do a near-deadline core overrun fail before candidate authority, while optional deadline exhaustion after a timely candidate remains visible without discarding the draft or leaking content to logs?
