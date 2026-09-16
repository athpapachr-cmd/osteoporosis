# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — LIVE GPT-5.6 22-CASE QUALIFICATION EXECUTED: 7 PASS / 15 FAIL — PROMOTION HOLD / READ-ONLY FAILURE TRIAGE REQUIRED — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-06 runtime/eval head:** `d1470ce1c10cd69f6f9e4fe72096527157f084e9`.
> **H-06 deterministic/inherited gate:** `35139716235` — SUCCESS.
> **H-06 canonical checkpoint:** `0167a28df071d5c2714800e8f740d34c24a08b9f`; verification `35140063802` — SUCCESS.
> **Second live execution wrapper head:** `2714715ee2c14c6a477abd78562685ae1e7ecf0b`.
> **Wrapper-head deterministic gate:** `35140165777` — SUCCESS.
> **Live GPT-5.6 qualification:** `35140165842` — 22 executed / 7 PASS / 15 FAIL.
> **Synthetic credential + schema boundary:** CLOSED; Actions secret available/masked, PHI approval false, H-06 schema accepted by live API.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

On 2026-09-16 the Product Owner authorized bounded PR-1 implementation and confirmed that a repository Actions `OPENAI_API_KEY` had been added. This authorizes the designed synthetic-only qualification path, not production/PHI use.

This does **not** authorize runtime release PR merge/deploy, identifiable transcript processing, PR-2 authoritative writes, real-patient pilot use or unrelated product mutation.

## Deterministic hardening status

H-01 through H-06 remain closed deterministically:

- **H-01:** default-deny unexpected/hallucinated assertion oracle;
- **H-02:** unique candidate-local concept identity + repeated-event grouping protection;
- **H-03:** synthetic qualification separated from identifiable-PHI approval;
- **H-04:** reviewed runtime ranges/types/semantic/source guards;
- **H-06:** OpenAI Structured Outputs schema compatibility + deterministic HTTP 400 contract-error classification.

Exact H-06 runtime head `d1470ce1c10cd69f6f9e4fe72096527157f084e9` passed run `35139716235` with 52 focused PR-1 tests, 6 protected-clinical regressions, 24 Medical Report regressions, workspace navigation and bounded scope all green.

Canonical H-06 checkpoint `0167a28df071d5c2714800e8f740d34c24a08b9f` passed verification `35140063802`. The workflow-only trigger head `2714715ee2c14c6a477abd78562685ae1e7ecf0b` then passed full deterministic gate `35140165777`, so the live run did not execute on an unverified runtime mutation.

## Live qualification boundary — proven

Run `35140165842` proved the selected live path can execute the frozen promotion suite:

```text
provider=openai
model=gpt-5.6
purpose=synthetic_eval
OPENAI_API_KEY=Actions secret / masked
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false
fixture count=22
fixture IDs unique
Structured Outputs schema accepted
all 22 provider calls reached the qualification loop
```

No production configuration changed. No identifiable transcript/PHI authorization was enabled.

This closes the earlier credential/schema execution uncertainty. The remaining failure is now qualification behavior/oracle alignment, not infrastructure setup.

## Exact live result — 7 PASS / 15 FAIL

### PASS cases

1. `explicit_negative_smoking`
2. `dxa_objective`
3. `labs_objective`
4. `preference_only`
5. `garbled_speech`
6. `negative_history_vs_negative_investigation`
7. `embedded_instruction_untrusted`

### FAIL cases and coded checks

- `fracture_relative_time`: `required_assertion_0_missing`, `unexpected_assertion_fracture.site`.
- `options_one_final`: required assertions 0..4 missing; unexpected `treatment.agent`, `patient.acceptance`, `decision.type`, `treatment.status`.
- `followup_vague`: unexpected `followup.task_type`.
- `frax_original_adjusted`: unexpected `frax.tool_name`.
- `speaker_ambiguity`: missing expected concept / required assertion; forbidden `administration.status`; unexpected `administration.agent`, `administration.status`.
- `followup_exact`: unexpected `followup.task_type`.
- `unrelated_general_clinical_text`: unexpected `followup.task_type`.
- `repeated_fracture_event_grouping`: required assertions 0 and 2 missing; both required candidate groups missing; unexpected `fracture.site`.
- `referral_not_completed_result`: required assertion 0 missing; unexpected `followup.task_type`, `clinical.unmapped_narrative`.
- `prescription_not_administration`: forbidden `administration.agent` and `administration.status`; unexpected `treatment.status`, `administration.agent`, `administration.status`.
- `self_correction_date`: required assertion 0 and candidate group 0 missing; unexpected `fracture.site`.
- `third_party_treatment_history`: required assertions 0 and 1 missing; unexpected `treatment.agent`, `treatment.status`.
- `out_of_range_runtime_values`: unexpected `frax.tool_name`.
- `planned_not_done_administration`: required assertion 1 missing; unexpected `administration.agent`, `administration.status`.
- `negated_treatment_exposure`: required assertion 0 missing; unexpected `treatment.agent`, `treatment.status`.

Exact summary emitted by the hardened runner:

```text
provider-eval summary: total=22 failed=15
```

## H-07 — live qualification mismatch / triage boundary

**OPEN.**

The 15 failures must not be treated as one undifferentiated “model failure”. The coded evidence shows at least two possible classes that require read-only reconciliation against the frozen fixtures/provider profile/mapper before mutation:

1. **likely semantic/provider failures**, where required distinctions are missing or explicitly forbidden treatment/administration truth appears;
2. **possible fixture/oracle overconstraint**, where the model emits an additional assertion that may be source-supported but was omitted from the fixture allowlist, e.g. `followup.task_type` or `frax.tool_name`.

No prompt, fixture, ontology, mapper or evaluator mutation is authorized until each failed case is classified from repository evidence as one of:

```text
A. provider semantic failure against intended contract
B. fixture/default-deny allowlist overconstraint
C. ontology/provider-profile ambiguity
D. deterministic mapping/evaluator contract defect
```

The triage must preserve H-01 default-deny behavior. A case must never be made green by blanket-permitting extras. Any allowlist change requires evidence that the additional assertion is source-supported and semantically acceptable under the product contract.

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

Perform a **READ-ONLY H-07 triage** of all 15 failed live cases against:

- `evals/transcript_v1/cases.json`;
- `evals/transcript_v1/run_provider_eval.py`;
- `clinical_excellence/modules/osteoporosis/transcript_profile.py`;
- deterministic mapper/guard contracts.

Freeze the per-case classification and bounded remediation contract canonically before any prompt/fixture/code mutation or third live run.

## Explicitly blocked

Until H-07 is reconciled, a revised qualification passes with zero failures, and that evidence receives fresh independent READ-ONLY review:

- no third live provider run;
- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no blanket weakening of the default-deny oracle;
- no claim that 7/22 constitutes promotion readiness.

## Separate production-release blockers/debt

**H-05 remains OPEN:** the async clinical route still invokes a synchronous provider client in the current single-worker web process. It does not block isolated CLI synthetic qualification, but it must be remediated and verified before production enablement/deploy.

Stronger executable browser lifecycle/BFCache/logout evidence also remains production-release debt.
