# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — LIVE QUALIFICATION 7/22 PASS; H-07 READ-ONLY TRIAGE COMPLETE / BOUNDED REMEDIATION AUTHORIZED — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-06 runtime/eval head:** `d1470ce1c10cd69f6f9e4fe72096527157f084e9`.
> **Live execution wrapper head:** `2714715ee2c14c6a477abd78562685ae1e7ecf0b`.
> **Live qualification:** `35140165842` — 22 executed / 7 PASS / 15 FAIL.
> **Live-failure canonical checkpoint:** `5498f88ce8e9a67090863a7a5c9c272d3cced23e`; verification `35140838915` — SUCCESS.
> **H-07 triage:** COMPLETE read-only; 5 A / 4 B / 6 C / 0 demonstrated D.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

The Product Owner authorized bounded PR-1 implementation and supplied the safe Actions credential prerequisite. Authority remains limited to synthetic qualification, bounded remediation and implementation-candidate preparation. It does not authorize production/PHI use, merge/deploy, PR-2 or real-patient pilot activity.

## Proven baseline retained

H-01 through H-06 remain deterministically closed. The H-06 runtime head `d1470ce1c10cd69f6f9e4fe72096527157f084e9` passed gate `35139716235`; its canonical checkpoint passed `35140063802`. The workflow-only live trigger head passed deterministic gate `35140165777`.

Live run `35140165842` then proved:

```text
provider=openai
model=gpt-5.6
purpose=synthetic_eval
secure Actions key available/masked
PHI approval=false
strict Structured Outputs schema accepted
22/22 cases executed
7 PASS
15 FAIL
```

No production configuration changed and no identifiable transcript use was enabled.

## H-07 read-only triage — COMPLETE

The 15 failures were reconciled against the frozen fixtures, provider profile, evaluator and deterministic mapper/guard without changing code, prompt or fixtures.

### A — PROVIDER_SEMANTIC_FAILURE (5)

These cases violate an intended semantic/source/scope distinction already present in the product contract:

1. `speaker_ambiguity`
   - provider emitted administration truth where the source explicitly says actual administration is uncertain;
   - live evidence includes forbidden `administration.status` plus unexpected `administration.agent/status`.

2. `unrelated_general_clinical_text`
   - generic shoulder narrative was correctly represented as unmapped narrative but additionally produced a mapped `followup.task_type` in the osteoporosis module;
   - module-scoped unrelated narrative must not silently create an osteoporosis Step-4 task.

3. `prescription_not_administration`
   - the required clinician-recommendation treatment agent was present, but provider also emitted forbidden administration agent/status despite explicit statement that administration was not documented.

4. `third_party_treatment_history`
   - two patient-history candidates were produced, but neither satisfied the required third-party-positive / patient-negative source-polarity contracts and extra treatment status was inferred;
   - the profile already requires third-party and negation preservation.

5. `negated_treatment_exposure`
   - treatment-agent concept was emitted but did not satisfy the explicit patient-negative contract, and extra treatment status was inferred from a statement of never having received treatment.

Primary remediation: strengthen explicit semantic/scope rules in the provider profile. Do not weaken source/negation/administration guards.

### B — FIXTURE_ORACLE_OVERCONSTRAINT (4)

These failures consist only of additional concepts that are directly supported by the synthetic source and are valid under the intended ontology, while the required assertions otherwise passed:

1. `followup_vague` — extra `followup.task_type`; a future review in ~6 months can validly carry a constrained follow-up-visit task type in addition to timeframe text.
2. `followup_exact` — extra `followup.task_type`; an explicitly dated return visit can validly carry `followup_visit` in addition to due date.
3. `frax_original_adjusted` — extra `frax.tool_name`; the source explicitly says the formal tool is FRAX.
4. `out_of_range_runtime_values` — extra `frax.tool_name`; the source explicitly identifies the formal FRAX result while the deliberately implausible numeric values are correctly left to deterministic range validation.

Primary remediation: add narrowly constrained `allowed_assertions` rules only for the source-supported value. H-01 default deny remains intact; no wildcard permission is allowed.

### C — ONTOLOGY / PROVIDER-PROFILE AMBIGUITY (6)

The provider returned the relevant concept family but did not satisfy the exact required value/type/mapping contract. The live runner intentionally did not log candidate payloads, so the exact mismatch is not recoverable after the run. The repository profile currently lists concept keys but does not fully specify all provider-facing value kinds/code sets or preferred concept usage by semantic role.

1. `fracture_relative_time` — `fracture.site` exists but does not match exact expected hip/source/mapping rule; date + low-trauma assertions passed.
2. `options_one_final` — all expected semantic classes exist, but options/recommendation/acceptance/decision components are represented using competing `treatment.*` / `decision.*` structures; only the final selected-agent assertion clearly satisfied its frozen rule.
3. `repeated_fracture_event_grouping` — two event candidates and dates exist, but both site assertions fail exact expected site contracts, preventing same-event grouping proof.
4. `referral_not_completed_result` — provider avoided false DXA result fields but its `followup.task_type` did not satisfy the exact `DXA` task contract and it also emitted redundant unmapped narrative.
5. `self_correction_date` — corrected May date appears to have been preserved, but `fracture.site` fails the exact site rule, so the grouped event cannot pass.
6. `planned_not_done_administration` — scheduled date passes, but administration agent/status do not satisfy the exact provider-facing code contract despite the candidate remaining a future follow-up task.

Primary remediation: make the provider profile explicit about value kinds, allowed code sets and concept ownership by semantic role. Do not infer hidden live values that were not logged.

### D — DETERMINISTIC_CONTRACT_DEFECT (0 demonstrated)

No current live evidence proves that the deterministic mapper/evaluator incorrectly handled a provider assertion that exactly satisfied the frozen contract. The duplicate `unexpected_assertion_*` signal can accompany a required-rule mismatch because the default-deny authorization rule deliberately includes semantic/source/value/mapping constraints; this is noisy but not, by itself, a correctness defect.

## Bounded remediation contract

The next mutation may change only the PR-1 provider profile, the four demonstrated overconstrained fixtures, and focused deterministic tests unless a new deterministic finding proves another file is required.

### Provider-profile hardening

Add explicit provider-facing value contracts for at least:

- `fracture.site` code set: `vertebral / hip / distal_radius / proximal_humerus / pelvis / other`;
- treatment/administration agent codes;
- treatment status and administration status code sets;
- decision type codes;
- patient acceptance codes;
- follow-up task type codes including exact `DXA` casing;
- required value `kind` for date, code, boolean, number/integer and quantity concepts used by the 22-case suite.

Clarify semantic ownership:

```text
actual/historical treatment episode → treatment.*
option / recommendation / final selected plan → decision.selected_agent with its semantic_type preserved
administration.* → actual or explicitly planned administration event only
prescription/recommendation alone → never administration.*
uncertain administration occurrence → uncertain_needs_review; no administration status/actual truth
unrelated non-osteoporosis narrative → clinical.unmapped_narrative; no osteoporosis follow-up task unless explicitly osteoporosis-related
third-party fact → speaker=third_party
explicit never/not received → polarity=negative; no positive treatment status
```

### Fixture corrections

Narrowly allow only:

- `followup.task_type=followup_visit` for `followup_vague` and `followup_exact`;
- `frax.tool_name=frax` for `frax_original_adjusted` and `out_of_range_runtime_values`.

No other live failure is authorized to be made green by fixture relaxation at this stage.

### Deterministic evidence before next live run

Add focused tests that prove:

- provider profile exposes the exact required code/value contracts and semantic ownership rules;
- the four B fixture additions are narrow and value-constrained;
- hard A safety distinctions remain forbidden;
- all H-01..H-06 tests remain green;
- all 22 fixtures remain valid under default deny;
- inherited protected-clinical, Medical Report, navigation and scope checks remain green.

Then checkpoint the exact remediation SHA/run before any third live evaluation.

## Exact next action

Implement the bounded H-07 remediation above, run deterministic/inherited CI, checkpoint exact evidence, then deliberately execute a third frozen 22-case GPT-5.6 qualification. Promotion still requires `failed=0`.

## Explicitly blocked

- no third live run before deterministic + canonical checkpoint;
- no wildcard fixture permissions;
- no weakening of administration/source/negation safety rules;
- no runtime release PR;
- no merge/deploy;
- no identifiable transcript use;
- no PR-2 or real-patient pilot.

## Separate production-release blockers/debt

**H-05 remains OPEN:** synchronous provider execution inside the async single-worker web process must be remediated before production enablement/deploy.

Executable browser lifecycle/BFCache/logout cleanup evidence also remains release debt.
