# Web Visit Audio — contextual AI clinical considerations

## Product Owner checkpoint

**Problem now:** the released system can preserve explicit symptoms, therapy, plan and obligations and can generate nine bounded follow-up intents. It still cannot say that a combination of facts may merit an additional clinical question, focused examination or possible investigation unless that step was already spoken.

**After this step:** `gpt-6-luna` may return up to four concise considerations inferred from validated explicit facts, each with category, rationale, confidence and one to three cited source IDs. They appear in a separate display-only block and never enter the patient record automatically.

**Concrete example:** explicit thoracic pain plus reported height loss may produce `Να εξεταστεί αν χρειάζεται έλεγχος για σπονδυλικό κάταγμα`, with the two source IDs and a rationale. It is labelled as an AI consideration, not a diagnosis or order.

**Outside this step:** no automatic diagnosis, prescribing, dose change, test order, referral, urgency claim, guideline-authority claim, candidate mutation, obligation creation, one-click adoption, new provider/model call, retention/configuration change or real-patient agent test.

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

The existing strict `visit_audio_analysis_v2` response gains `contextual_clinical_considerations[]`, maximum four per batch:

- `category`: one of six fixed consideration categories;
- `text`: 1..240 characters;
- `rationale`: 1..320 characters;
- `source_ids`: one to three unique current-batch IDs, all with at least one validated extracted fact;
- `confidence`: `low`, `moderate` or `high`.

No extra keys. Text is normalized, must not contain control characters and is returned only in the transient audio-draft response. Across batches the server deduplicates normalized text and returns at most four.

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

The block is rendered only when considerations exist. Each card shows a Greek category label, the consideration and confidence. Rationales sit inside one closed disclosure. The warning explicitly distinguishes inferred advice from what was said and from evidence-backed guidance. It disappears with the transient draft and is excluded from the candidate object.

Any clinician who accepts the idea must independently enter the appropriate wording into the existing editable plan. That ordinary edit invalidates the final confirmation and prior Preview under the existing contract.

## R2 questions

1. Does display-only separation fully prevent provider prose from entering candidate/Preview/Save?
2. Are cited IDs constrained to validated current-batch facts and deduplicated safely?
3. Does the UI avoid implying diagnosis, order or guideline authority?
4. Do existing typed suggestions and one-confirmation behavior remain unchanged?
5. Are multi-batch limits, failure behavior and temporary-content boundaries explicit?
