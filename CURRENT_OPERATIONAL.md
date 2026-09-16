# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — IMPLEMENTED / DETERMINISTIC-TESTED / HARDENED MINIMUM EVAL SUITE COMPLETE; LIVE SYNTHETIC PROVIDER-EVAL HOLD — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Activation PR:** #115 — MERGED.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact tested runtime head:** `a79d68915bde230a53bb7b5fd31a4104a491b058`.
> **Latest deterministic/inherited gate:** `35084122094` — SUCCESS.
> **Latest slice-canonical reconciliation gate:** `35084479897` — SUCCESS.
> **Focused PR-1 tests:** 26 PASS.
> **Synthetic provider minimum suite:** 13 synthetic/de-identified scenarios with structured safety assertions.
> **Synthetic provider-eval probe:** `35056606836` — HOLD reconfirmed on rerun; GitHub Actions `OPENAI_API_KEY` unavailable, no provider call made.
> **Credential-path verification:** current connected GitHub/Render tooling exposes no safe non-production secret-inheriting execution path; production config remains untouched.
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

## Independent implementation review hardening — 2026-09-16

A fresh review after the previous chat boundary found two additional deterministic weaknesses and closed them without expanding PR-1 scope.

### Exact-date contract

The previous date validator verified only the textual shape of `YYYY-MM-DD`, `YYYY-MM` or `YYYY`. It could therefore accept impossible calendar values such as `2026-02-31`, `2026-13` or year `0000`.

The current exact tested head validates normalized day/month/year values as real calendar dates while preserving the existing hard rule that relative/vague timing cannot contain an invented normalized exact date. Deterministic tests include impossible dates plus a valid leap-day control.

### Synthetic provider-eval gate

The live synthetic eval runner previously treated a case as PASS when the expected semantic types and concept keys appeared somewhere in the model output. That was insufficient for the frozen v3 safety invariants: a model could theoretically emit those expected sets while also inventing an exact date, adding a wrong final decision, misassigning speaker/polarity, or producing an unsafe target mapping.

The eval harness is now fail-closed on structured case invariants. Fixtures can require and forbid specific assertions and concepts, constrain source semantics and component values, verify deterministic mapping state/target/reason, forbid invented exact dates, constrain semantic counts, reject unverifiable evidence warnings, and assert that the returned response remains ephemeral/non-authoritative. Output remains PHI-safe: it reports only case IDs, coded failed checks and candidate counts, never transcript or candidate content.

## Frozen-v3 minimum synthetic suite — COMPLETE

Head `a79d68915bde230a53bb7b5fd31a4104a491b058` completes the representative minimum synthetic/de-identified suite with 13 explicit scenarios:

1. positive fracture history with vague relative timing;
2. explicit negative smoking history;
3. exact DXA objective result;
4. laboratory objective results;
5. multiple options + one recommendation + patient acceptance + exactly one final decision;
6. patient preference without accidental decision/recommendation;
7. vague follow-up timeframe without fabricated due date;
8. garbled/uncertain speech without guessed administration truth;
9. original formal FRAX versus clinician-adjusted interpretation;
10. ambiguous speaker/treatment history;
11. explicit negative history versus negative objective investigation, with reciprocal semantic-collapse checks;
12. exact follow-up date when the source really supports an exact day;
13. unrelated general musculoskeletal text without osteoporosis-target hallucination.

This closes the deterministic **suite-definition** gap. It does **not** constitute live GPT-5.6 provider performance evidence; that still requires execution of all 13 scenarios through the selected adapter/model.

## Latest exact-head gate evidence

Workflow `35084122094` passed on exact runtime head `a79d68915bde230a53bb7b5fd31a4104a491b058`.

Evidence includes:

- Python and browser syntax — PASS;
- **26 PR-1 focused privacy/contract/mapping/UI/eval-contract tests — PASS**;
- all 13 minimum synthetic fixtures validated as de-identified structured gate cases;
- reusable cookie-session authentication of the protected transcript endpoint;
- 512 KiB body and 120k-character fail-closed limits;
- sanitized error responses with no sentinel PHI echo/logging;
- one provider call per extraction;
- provider target-path injection rejection;
- relative/vague timing cannot acquire an invented exact date;
- impossible normalized calendar dates are rejected;
- adjusted FRAX cannot overwrite original formal FRAX;
- unsupported units/legacy-negative semantics become ambiguous rather than silently authoritative;
- option-discussed cannot populate final treatment decision;
- fixed runtime enums and treatment-duration type/range are locally validated;
- strengthened eval contract rejects dangerous extra assertions even when the older semantic/concept set checks would otherwise pass;
- negative history and negative objective investigation are represented as distinct assertions in the frozen eval gate;
- unrelated clinical narrative has explicit forbidden osteoporosis-target assertions;
- structured-output validation is classified separately from provider unavailability;
- OpenAI adapter retry/timeout/structured-output contract;
- inherited protected-clinical regressions — 6 PASS;
- inherited Medical Report regressions — 24 PASS;
- inherited workspace navigation regression — PASS;
- bounded PR-1 scope guard — PASS at the exact runtime head.

The immediately preceding canonical hardening checkpoint was independently verified by workflow `35083865144` — SUCCESS. The subsequent `SLICE_PLAN_CURRENT.md` evidence reconciliation was independently verified by workflow `35084479897` — SUCCESS.

## Earlier canonical-checkpoint guard verification

Post-canonical run `35057529756` re-executed all substantive legs successfully but exposed one workflow-only defect: `SLICE_PLAN_CURRENT.md` was absent from the PR-1 scope allowlist. That conflicted with the permanent atomic-canonical protocol, which requires the active slice canonical to be checkpointed when material implementation state changes.

The PR-1 workflow was corrected to trigger on and allow `SLICE_PLAN_CURRENT.md`. Clean verification run `35057627424` then passed **all** steps. The harness-only finding is closed. No runtime or clinical behavior changed as part of that correction.

## Synthetic provider-eval checkpoint

Temporary GitHub Actions run `35056606836` attempted to start the synthetic/de-identified provider eval. The runner established that the repository has **no usable `OPENAI_API_KEY` Actions secret** for this path. A safe rerun on 2026-09-16 reconfirmed the same condition. The workflow recorded:

```text
PR1_SYNTHETIC_PROVIDER_EVAL_HOLD
No production configuration was changed and no transcript was sent.
```

This is not a provider-model PASS and not a provider-model FAIL. The deterministic provider adapter contract and the full 13-case minimum suite are tested, but the selected live GPT-5.6 semantic extraction behavior has not yet been evaluated by this branch.

Do not substitute the assistant model, the Medical Report provider path, a fake provider, or production secret/config mutation for this missing live adapter evidence and label it equivalent.

## Safe credential-path verification — 2026-09-16

A read-only inspection was performed after completion of the hardened 13-case suite to determine whether the missing provider evidence could be executed without exposing/copying a secret or mutating production configuration.

Current connected tooling establishes:

- the GitHub Actions environment used by the probe has no usable `OPENAI_API_KEY` for this path;
- the connected GitHub interface does not expose repository/organization secrets APIs, so it cannot safely read, copy or create the missing secret from this workflow;
- the connected Render interface can inspect the deployed service/deploy/log state but exposes no one-off shell/job action that inherits the existing service environment;
- creating a new Render service/cron or updating environment variables would require an explicit configuration mutation and/or supplying the secret value, which is outside the authorized path;
- the production Render service remains untouched and continues to track `main` with auto-deploy; no PR-1 branch deployment or configuration mutation occurred.

Therefore there is **no safe non-production secret-inheriting execution path available from the current connected tool surface**. The HOLD is an external credential prerequisite, not an unresolved code/test harness problem.

The provider eval entrypoint remains ready in `evals/transcript_v1/run_provider_eval.py`; the temporary Actions provider-eval workflow was deliberately removed after the original HOLD checkpoint. Reintroducing an execution wrapper is permitted only when a safe non-production credential is actually available, so that the wrapper executes the frozen synthetic suite rather than manufacturing a different form of evidence.

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

The exact external prerequisite is a dedicated safe non-production credential path available to a synthetic-only execution job without revealing the credential value in chat, repository content or logs.

Once that prerequisite exists, the next authorized execution action is:

```text
safe non-production credential becomes available
→ attach it only to a synthetic-only provider-eval execution wrapper
→ run the hardened 13-case synthetic/de-identified provider eval through the selected OpenAI adapter/model
→ require zero failed cases under the frozen promotion invariants
→ checkpoint exact provider/model/eval evidence
```

Until that provider HOLD is resolved:

- the implementation may be inspected/reviewed and a safe credential path may be investigated read-only;
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
