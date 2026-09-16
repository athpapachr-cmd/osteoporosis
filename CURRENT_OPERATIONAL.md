# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — H-01..H-06 DETERMINISTIC CLOSURE PROVEN / SECOND LIVE 22-CASE SYNTHETIC QUALIFICATION AUTHORIZED — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-06 deterministic runtime/eval head:** `d1470ce1c10cd69f6f9e4fe72096527157f084e9`.
> **H-06 deterministic/inherited gate:** `35139716235` — SUCCESS.
> **Focused PR-1 tests:** 52 PASS.
> **Expanded synthetic qualification suite:** 22 synthetic/de-identified cases.
> **First live provider run:** `35138694138` — provider schema rejection before case 1; no semantic qualification occurred.
> **Synthetic credential boundary:** CLOSED; Actions secret available/masked, PHI approval false.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner explicitly authorized bounded PR-1 implementation and then confirmed that a repository Actions `OPENAI_API_KEY` had been added. That authorizes the designed synthetic-only qualification path, not production/PHI use.

This does **not** authorize runtime release PR merge/deploy, identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutation.

## H-01 through H-04 — CLOSED deterministically

Independent review findings remain closed:

- **H-01:** promotion evaluator is default-deny for unexpected/hallucinated assertions;
- **H-02:** duplicate candidate-local concept keys are rejected and repeated-event grouping is evaluated without cross-wiring;
- **H-03:** synthetic-eval authorization is separate from identifiable-PHI approval;
- **H-04:** reviewed runtime range/type/semantic/source guards fail closed.

The adversarial promotion suite remains frozen at 22 synthetic/de-identified cases.

## Safe synthetic credential path — CLOSED

The bounded workflow:

```text
.github/workflows/pr1-transcript-live-provider-eval.yml
```

proved in run `35138694138` that:

- `OPENAI_API_KEY` is available from the GitHub Actions secret store and masked in logs;
- `CLINICAL_TRANSCRIPT_AI_ENABLED=true`;
- `CLINICAL_TRANSCRIPT_SYNTHETIC_EVAL_ENABLED=true`;
- `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false`;
- model is `gpt-5.6`;
- purpose is `synthetic_eval`;
- fixture count is 22 with unique IDs;
- production configuration was not changed.

Therefore the previous credential HOLD is closed.

## First live provider attempt — schema failure evidence

Run `35138694138` reached the OpenAI API but failed before case 1 with HTTP 400 `invalid_json_schema` because the Pydantic discriminated `CandidateValueV1` generated a `oneOf` response-format schema. The adapter also incorrectly classified the deterministic HTTP 400 contract failure as `ProviderUnavailable`.

This was adapter/schema evidence only, **not** model semantic-performance evidence.

## H-06 — OpenAI Structured Outputs schema compatibility — CLOSED deterministically

Exact remediation head:

```text
d1470ce1c10cd69f6f9e4fe72096527157f084e9
```

Exact verification run:

```text
35139716235 — SUCCESS
```

The bounded remediation:

1. removes the Pydantic discriminator that generated unsupported `oneOf` for `CandidateValueV1`;
2. preserves the seven strict local value variants through their mutually exclusive literal `kind` values:
   - text;
   - code;
   - number;
   - integer;
   - boolean;
   - quantity;
   - date;
3. preserves strict `extra=forbid`, calendar-valid exact dates, vague/relative-date fail-closed behavior and duplicate concept-key rejection;
4. adds a deterministic schema regression proving `ProviderTranscriptExtractionV1.model_json_schema()` contains no `oneOf` keyword;
5. adds direct typed-validation regression coverage for all seven variants plus invalid kind/shape and impossible-date controls;
6. classifies deterministic HTTP 400/provider request-schema failures as `ProviderInvalidOutput`, not retryable `ProviderUnavailable`.

The provider path still uses `responses.parse(..., text_format=ProviderTranscriptExtractionV1)` and `store=False`; no permissive JSON-mode/free-text fallback was introduced.

## Exact deterministic evidence after H-06

Workflow `35139716235` passed on exact head `d1470ce1c10cd69f6f9e4fe72096527157f084e9`:

- Python syntax — PASS;
- browser syntax — PASS;
- **52 focused PR-1 tests — PASS**;
- inherited protected-clinical regressions — 6 PASS;
- inherited Medical Report regressions — 24 PASS;
- inherited workspace/navigation regression — PASS;
- bounded PR-1 scope verification — PASS.

The immediately preceding harness-only scope fix was verified by run `35139241635` — SUCCESS. It only allowed the explicitly authorized live-eval workflow filename inside the PR-1 scope guard; it did not change runtime behavior.

## Preserved PR-1 invariants

- raw transcript remains ephemeral and non-authoritative;
- no transcript/candidate DB, encounter, browser-storage or log persistence;
- provider emits semantic assertions, never application/storage paths;
- deterministic Module-01 code owns runtime mapping;
- candidates remain `proposed` and require clinician review;
- speaker/source, polarity, temporality, certainty and semantic distinctions are preserved;
- vague/relative timing cannot become an invented exact date;
- no authoritative patient/encounter/lab/task write exists in PR-1;
- identifiable transcript use remains blocked behind its separate privacy/provider approval gate.

## Exact next action

After this H-06 checkpoint is durably verified, deliberately retrigger the bounded synthetic-only workflow against the H-06-compatible branch state and run all 22 frozen synthetic/de-identified cases through `gpt-5.6`.

Promotion rule:

```text
22 cases
→ default-deny oracle
→ failed=0 required
→ checkpoint exact live provider/model/run evidence
→ fresh independent READ-ONLY evidence review
```

If the provider rejects a remaining schema construct or one or more cases fail, checkpoint the exact failure evidence before any further remediation/tuning.

## Explicitly blocked

Until a live 22-case selected-model qualification passes and is independently reviewed:

- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no production credential/config mutation;
- no claim that deterministic H-06 closure is equivalent to model semantic qualification.

## Separate production-release blockers/debt

**H-05 remains OPEN:** the async clinical route still invokes a synchronous provider client in the current single-worker web process. It does not block isolated CLI synthetic qualification, but it must be remediated and verified before production enablement/deploy.

Stronger executable browser lifecycle/BFCache/logout evidence also remains production-release debt.
