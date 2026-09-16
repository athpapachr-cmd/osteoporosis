# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — IMPLEMENTED / EXACT-RUNTIME-HEAD TESTED; SYNTHETIC PROVIDER EVAL PENDING.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Activation PR:** #115 — MERGED.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact tested runtime head:** `b62a55d9a8d06580cdea6ec42767a94bd1b7aee4`.
> **PR-1 focused/inherited gate:** `35056516818` — SUCCESS.
> **Provider eval:** pending — synthetic/de-identified only; identifiable transcript processing remains blocked.
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

The OpenAI adapter is isolated behind transcript-specific configuration gates and uses structured Responses parsing, `store=False`, no tools, `max_retries=0` and a bounded timeout. Provider output cannot supply application target paths.

The browser Heidi panel is isolated from `currentCase`: transcript/candidates live only in textarea/JS/DOM memory, use no localStorage/sessionStorage/indexedDB, have no Accept/Edit/Reject-to-record control, clear on close/pagehide/pageshow, and render provider-derived text with `textContent`.

## Exact-head gate evidence

Workflow `35056516818` passed on runtime head `b62a55d9a8d06580cdea6ec42767a94bd1b7aee4`, including:

- Python and browser syntax;
- 18 PR-1 focused privacy/contract/mapping/UI/eval-contract tests;
- reusable cookie-session authentication of the protected transcript endpoint;
- 512 KiB body and 120k-character fail-closed limits;
- sanitized error responses with no sentinel PHI echo/logging;
- one provider call per extraction;
- provider target-path injection rejection;
- adjusted FRAX cannot overwrite original formal FRAX;
- unsupported units/legacy-negative semantics become ambiguous rather than silently authoritative;
- option-discussed cannot populate final treatment decision;
- OpenAI adapter retry/timeout/structured-output contract;
- inherited protected-clinical regressions;
- inherited Medical Report regressions;
- inherited workspace navigation regression;
- bounded PR-1 scope guard with no mutation of `app-core.js`, `step3.js`, `step4.js`, `clinical_data.py` or `clinical_data_ext.py`.

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

Run the synthetic/de-identified provider eval harness against the configured OpenAI provider **only if** a usable protected `OPENAI_API_KEY` is available to GitHub Actions. The eval must not print transcript/candidate content. If the secret is unavailable, checkpoint `LIVE PROVIDER EVAL HOLD`; do not fake success or change production configuration.

After provider-eval evidence is durably checkpointed, prepare the implementation candidate/release hold. Do not merge or deploy the runtime without separate Product Owner release authority.

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
