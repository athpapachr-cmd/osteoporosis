# SLICE_PLAN_CURRENT.md — PR-1 Heidi-first Transcript Intake + Candidate Extraction v1

> **STATUS:** LIVE GPT-5.6 PROMOTION QUALIFICATION EXECUTED / 7 PASS + 15 FAIL / H-07 READ-ONLY TRIAGE REQUIRED — NOT RELEASE READY.
> **Activated:** 2026-09-16 Asia/Nicosia.
> **Slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Activation main:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Runtime branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-06 runtime/eval head:** `d1470ce1c10cd69f6f9e4fe72096527157f084e9`.
> **H-06 deterministic gate:** `35139716235` — SUCCESS.
> **Second live wrapper head:** `2714715ee2c14c6a477abd78562685ae1e7ecf0b`.
> **Wrapper deterministic gate:** `35140165777` — SUCCESS.
> **Live provider qualification:** `35140165842` — 22 executed; 7 PASS; 15 FAIL.
> **Writer:** one bounded PR-1 implementation writer; operational owner is `CURRENT_OPERATIONAL.md`.

## 1. Objective

Add a reusable Clinical Excellence Core capability that accepts a pasted Heidi transcript and returns structured, **non-authoritative** clinical candidates for clinician review, using Osteoporosis Module 01 as the first deterministic mapping profile.

PR-1 remains extraction/preview only. PR-2 owns later provisional in-card acceptance/edit/reject behavior.

## 2. Safety/privacy contract remains frozen

- raw transcript ephemeral and non-authoritative;
- no DB/encounter/browser-storage/log persistence of transcript/candidates;
- provider cannot author runtime/storage paths;
- deterministic Module-01 mapping owns target selection;
- candidates remain `proposed` + clinician-review-required;
- preserve speaker/source, polarity, temporality, certainty and semantic type;
- preserve history vs objective result vs interpretation;
- preserve option vs recommendation vs preference vs acceptance vs final decision;
- vague/relative timing never becomes an invented exact date;
- no authoritative patient write in PR-1;
- identifiable transcript use remains blocked by a separate privacy/provider approval gate.

## 3. Deterministic findings H-01..H-06

H-01 through H-06 are closed deterministically on the verified branch history:

- H-01 default-deny unexpected assertion oracle;
- H-02 unique component identity and repeated-event grouping;
- H-03 synthetic-eval authorization independent from identifiable-PHI approval;
- H-04 exact reviewed runtime guards;
- H-06 live-provider-compatible strict Structured Outputs schema plus deterministic HTTP-400 request-contract classification.

The H-06 exact runtime head `d1470ce1c10cd69f6f9e4fe72096527157f084e9` passed full deterministic/inherited gate `35139716235`; its canonical checkpoint passed `35140063802`.

## 4. Frozen 22-case promotion suite

The synthetic/de-identified suite includes the core semantic/date/result cases plus repeated event grouping, embedded instruction text, referral vs result, prescription vs administration, self-correction, third-party source, out-of-range numeric transcription, planned vs completed administration, negated exposure and low-confidence controls.

The oracle remains default-deny. Every returned component must be covered by a required or explicitly allowed fixture assertion. This protection must not be weakened globally during remediation.

## 5. Live execution evidence

### Execution boundary

Run `35140165842` used:

```text
provider=openai
model=gpt-5.6
purpose=synthetic_eval
OPENAI_API_KEY=secure Actions secret / masked
CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false
fixture count=22 / IDs unique
```

The H-06 response schema was accepted and the complete 22-case qualification loop executed. Therefore credential/schema setup is no longer the blocker.

### Result

```text
total=22
passed=7
failed=15
promotion threshold=0 failed
promotion result=FAIL
```

PASS IDs:

```text
explicit_negative_smoking
dxa_objective
labs_objective
preference_only
garbled_speech
negative_history_vs_negative_investigation
embedded_instruction_untrusted
```

FAIL IDs:

```text
fracture_relative_time
options_one_final
followup_vague
frax_original_adjusted
speaker_ambiguity
followup_exact
unrelated_general_clinical_text
repeated_fracture_event_grouping
referral_not_completed_result
prescription_not_administration
self_correction_date
third_party_treatment_history
out_of_range_runtime_values
planned_not_done_administration
negated_treatment_exposure
```

This is provider/model + qualification-oracle evidence. It is not infrastructure failure and it is not promotion-ready evidence.

## 6. H-07 — failed-case reconciliation before tuning

### Finding

The coded failures contain mixed signals. Some failures clearly indicate dangerous semantic collapse or unsupported truth; others may indicate that the fixture allowlist is narrower than the intended extraction contract even when an extra assertion is source-supported.

Therefore the 15 failures must be reconciled **case-by-case** before mutation. No blanket prompt tuning and no blanket allowlisting are permitted.

### Required triage taxonomy

Each failed case must be assigned one primary disposition:

```text
A — PROVIDER_SEMANTIC_FAILURE
    model output violates intended semantic/source/temporal contract

B — FIXTURE_ORACLE_OVERCONSTRAINT
    returned assertion is source-supported and contract-valid but omitted from allowlist

C — ONTOLOGY_PROFILE_AMBIGUITY
    intended concept representation is under-specified or competing representations are both plausible

D — DETERMINISTIC_CONTRACT_DEFECT
    mapper/evaluator itself evaluates a valid structured assertion incorrectly
```

Mixed cases may carry a secondary category, but remediation must identify the primary safety/quality defect rather than making the case green by convenience.

### Triage evidence surface

Read only:

- `evals/transcript_v1/cases.json`;
- `evals/transcript_v1/run_provider_eval.py`;
- `clinical_excellence/modules/osteoporosis/transcript_profile.py`;
- `clinical_excellence/modules/osteoporosis/transcript_targets.py`;
- `clinical_excellence/modules/osteoporosis/transcript_target_guard.py`;
- coded live run `35140165842`.

Because the live runner intentionally logs only case IDs, coded checks and candidate counts, triage must not invent hidden candidate values/source assertions that were not captured. Where a coded failure cannot uniquely determine the provider's exact assertion details, mark that uncertainty and remediate only what the evidence supports.

## 7. Promotion-policy constraints during remediation

Permitted after canonical triage checkpoint:

- strengthen provider instructions for semantic distinctions;
- clarify ontology/profile rules where the intended representation is ambiguous;
- repair fixture allowlists only for source-supported, contract-valid extras;
- repair mapper/evaluator defects if demonstrated;
- add deterministic regressions for every changed rule.

Forbidden:

- wildcard/permissive assertion allowlists;
- removing hard distinctions merely because GPT-5.6 emitted a different structure;
- accepting administration truth from prescription/recommendation language;
- accepting positive treatment truth from explicit negation;
- weakening third-party/source attribution protections;
- treating a 7/22 result as a partial promotion pass.

## 8. Exact next action

Perform READ-ONLY H-07 triage of all 15 failures, freeze the per-case disposition + bounded remediation contract in the canonicals, verify that checkpoint, and only then mutate prompt/fixtures/profile/mapper/evaluator as justified.

After deterministic verification of the resulting remediation, a third live 22-case run may be deliberately triggered. Promotion requires `failed=0`.

## 9. H-05 retained production-release blocker

The protected FastAPI route remains async while the OpenAI provider client is synchronous in a single-worker web process. H-05 must close before production enablement/deploy regardless of eventual synthetic qualification outcome.

Executable browser lifecycle/BFCache/logout cleanup evidence also remains production-release debt.

## 10. Definition of Done status

```text
H-01..H-06 deterministic closure          YES
safe synthetic credential path             YES
live schema accepted                        YES
22-case selected-model execution completed  YES
live promotion gate                         FAIL: 15/22 failed
H-07 triage                                 OPEN
release ready                               NO
runtime release PR                          NO
deploy                                      NO
identifiable transcript use                 NO
PR-2 / real pilot                           NO
```
