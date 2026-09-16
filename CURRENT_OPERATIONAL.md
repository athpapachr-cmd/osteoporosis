# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — IMPLEMENTED / DETERMINISTIC-TESTED; LIVE SYNTHETIC PROVIDER-EVAL HOLD — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Activation PR:** #115 — MERGED.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact tested runtime head:** `15d109c4550719e52f0a1bcb4bf11b4cfedba6ac`.
> **Final deterministic/inherited gate:** `35057309343` — SUCCESS.
> **Focused PR-1 tests:** 24 PASS.
> **Synthetic provider-eval probe:** `35056606836` — HOLD; GitHub Actions `OPENAI_API_KEY` unavailable, no provider call made.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner explicitly instructed `Ok. Go. Ξεκίνησε` after fresh closeout identified Heidi-first capture / PR-1 as the next primary roadmap target. This authorizes bounded implementation of the frozen PR-1 extraction slice, including branch creation, code, tests/evals and implementation-candidate preparation.

This does **not** automatically authorize merge of the runtime implementation, production deployment, enabling identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutations.

## Implemented PR-1 boundary

The exact tested runtime head implements:

```text
protected POST /clinical/transcript/extract
→ 512 KiB body ceiling + 120k transcript character ceiling
→ sanitized strict request/response contracts
→ provider-neutral semantic extraction
→ server-owned deterministic osteoporosis target mapping
→ mapped / ambiguous / unmapped candidate preview
→ no authoritative write
```

Core semantics preserve speaker, polarity, temporality, certainty and the required distinctions between patient/history fact, objective result, clinician interpretation, option discussed, recommendation, preference, final decision and patient disposition. Relative/vague timing cannot be normalized into an invented exact date.

The OpenAI adapter is isolated behind transcript-specific configuration gates and uses structured Responses parsing, `store=False`, no tools, `max_retries=0` and a bounded timeout. Provider output cannot supply application target paths. Structured Pydantic/schema failures are classified as `PROVIDER_INVALID_OUTPUT`; refusal remains `PROVIDER_REFUSAL`; transport/rate-limit/timeout/upstream failures remain `PROVIDER_UNAVAILABLE`.

The browser Heidi panel is isolated from `currentCase`: transcript/candidates live only in textarea/JS/DOM memory, use no localStorage/sessionStorage/indexedDB, have no Accept/Edit/Reject-to-record control, clear on close/pagehide/pageshow, and render provider-derived text with `textContent`.

## Deterministic runtime hardening completed

Static review against the actual current Step-3/Step-4 UI contracts found and corrected one class of mapper defect: a provider concept key with a real target must **not** be treated as mapped unless its value also satisfies that runtime target's exact enum/type contract.

The final mapper therefore fails closed for:

- treatment episode status outside `planned / active / completed / stopped / holiday / unknown`;
- administration status outside `done / due / overdue / missed / planned / not_applicable`;
- fracture sites outside the current runtime set;
- FRAX tool and resulting-risk category outside current runtime enums;
- VFA indication/action/modality outside current runtime enums;
- treatment duration that is non-numeric or outside the runtime 0–50-year range;
- unsupported units and legacy-negative/default semantics already identified by the frozen design.

Invalid or non-lossless provider values become `ambiguous` or `unmapped`; they are never silently coerced into authoritative-looking runtime values.

## Final exact-head gate evidence

Workflow `35057309343` passed on exact runtime head `15d109c4550719e52f0a1bcb4bf11b4cfedba6ac`.

Evidence includes:

- Python and browser syntax — PASS;
- **24 PR-1 focused privacy/contract/mapping/UI/eval-contract tests — PASS**;
- reusable cookie-session authentication of the protected transcript endpoint;
- 512 KiB body and 120k-character fail-closed limits;
- sanitized error responses with no sentinel PHI echo/logging;
- one provider call per extraction;
- provider target-path injection rejection;
- relative/vague timing cannot acquire an invented exact date;
- adjusted FRAX cannot overwrite original formal FRAX;
- unsupported units/legacy-negative semantics become ambiguous rather than silently authoritative;
- option-discussed cannot populate final treatment decision;
- fixed runtime enums and treatment-duration type/range are locally validated;
- structured-output validation is classified separately from provider unavailability;
- OpenAI adapter retry/timeout/structured-output contract;
- inherited protected-clinical regressions — 6 PASS;
- inherited Medical Report regressions — 24 PASS;
- inherited workspace navigation regression — PASS;
- bounded PR-1 scope guard — PASS;
- no mutation of `app-core.js`, `step3.js`, `step4.js`, `clinical_data.py` or `clinical_data_ext.py`.

## Synthetic provider-eval checkpoint

Temporary GitHub Actions run `35056606836` attempted to start the synthetic/de-identified provider eval. The runner established that the repository has **no usable `OPENAI_API_KEY` Actions secret** for this path. The workflow recorded:

```text
PR1_SYNTHETIC_PROVIDER_EVAL_HOLD
No production configuration was changed and no transcript was sent.
```

This is not a provider-model PASS and not a provider-model FAIL. The deterministic provider adapter contract is tested, but the selected live GPT-5.6 semantic extraction behavior has not yet been evaluated by this branch.

Do not substitute the assistant model, the Medical Report provider path, a fake provider, or production secret/config mutation for this missing live adapter evidence and label it equivalent.

## Privacy boundary remains fail-closed

```text
raw transcript: ephemeral only
candidate preview: ephemeral only
DB write: none
browser persistence: none
provider retries: disabled
identifiable transcript use: BLOCKED
```

`CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED` remains a separate transcript-specific gate. PR-1 testing does not treat `store=False` as Zero Data Retention or as provider/privacy approval.

## Exact next action

Preserve **LIVE SYNTHETIC PROVIDER-EVAL HOLD** until a safe credential path is explicitly available without exposing/copying a secret and without mutating production configuration.

Until then:

- the deterministic implementation may be inspected/reviewed;
- the provider/model Definition-of-Done item remains unsatisfied;
- do **not** open the runtime release PR;
- do **not** merge/deploy PR-1;
- do **not** enable identifiable transcript processing;
- do **not** advance to PR-2 or real-patient pilot use.

## Explicitly deferred / forbidden in PR-1

- Accept/Edit/Reject-to-record or any authoritative patient write;
- PR-2 inline population/persistence;
- real identifiable Heidi transcript transmission before separate privacy/data-control approval;
- persistence of transcript, evidence snippets or candidates;
- provider-authored application target paths;
- exact-date invention from vague timing;
- collapsing option/recommendation/preference/final decision semantics;
- Practice Review coaching, KPI changes, treatment recommendation, pilot activation;
- unrelated Medical Report, Physio, RF, Calendar or Clinical Learning mutation.
