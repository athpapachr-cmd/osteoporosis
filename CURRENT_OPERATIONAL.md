# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 HEIDI-FIRST TRANSCRIPT CAPTURE — H-11 SOURCE-SEMANTIC NORMALIZATION DETERMINISTICALLY PROVEN / NEW 22-CASE GPT-5.6 QUALIFICATION AUTHORIZED — NOT RELEASE READY.
> **Updated:** 2026-09-16 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `0ab5f9770d220c20e8d94544cb64e93a4aa30d00`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** this bounded PR-1 implementation lifecycle only.
> **Runtime implementation branch:** `feat/pr1-transcript-capture-v1-2026-09-16`.
> **Exact H-08 deterministic head:** `5eeddeba664814b38d55bad8231afb5b33448eae`.
> **H-08 deterministic/inherited gate:** `35446869081` — SUCCESS.
> **H-09 exact deterministic head:** `00985996fe7c32a9596c5440d27223552d290d06`.
> **H-09 deterministic/inherited gate:** `35448153136` — SUCCESS.
> **H-10 exact deterministic head:** `eee45f9bab1fa0eba9fc36dea8fdcbf61dd89d5a`.
> **H-10 deterministic/inherited gate:** `35455400599` — SUCCESS.
> **H-11 exact deterministic head:** `a6bb0b2a8897f83c6584106e03c26b59aa9c266b`.
> **H-11 deterministic/inherited gate:** `35455876663` — SUCCESS.
> **Focused PR-1 tests:** 75 PASS.
> **Frozen synthetic qualification suite:** 22 synthetic/de-identified cases.
> **Pre-H09 live qualification:** `35446962808` — 22 PASS / 0 FAIL.
> **Post-H09 live qualification:** `35448273993` — 19 PASS / 3 FAIL.
> **Post-H10 live qualification:** `35455512992` — **20 PASS / 2 FAIL**; secure synthetic-only boundary and 22-case fixture count passed.
> **Safe credential/schema boundary:** CLOSED; Actions secret available/masked, PHI approval false, strict Structured Outputs accepted.
> **Medical Report V1.1:** CLOSED; do not reopen without separate authority.

## Product-owner authority

The Product Owner authorized bounded PR-1 implementation and supplied the safe repository Actions credential prerequisite. Authority remains limited to synthetic qualification, bounded remediation and implementation-candidate preparation. It does not authorize production/PHI use, merge/deploy, PR-2 or real-patient pilot activity.

## Proven engineering baseline

H-01 through H-06 remain closed deterministically. Live run `35140165842` then executed all 22 cases and returned 7 PASS / 15 FAIL. That result was checkpointed at `5498f88ce8e9a67090863a7a5c9c272d3cced23e` and verified by `35140838915`.

A read-only H-07 triage classified those failures as:

```text
A provider semantic failures       5
B fixture/oracle overconstraints   4
C ontology/profile ambiguities     6
D demonstrated deterministic bugs  0
```

The triage contract was frozen at `71cb10a21551ad2da356990cc74a02153b7e3fb2` and verified by `35141347844`.

## H-07 bounded remediation — COMPLETE deterministically

The remediation is intentionally narrow. It changes provider-facing ontology/profile guidance, only four source-supported fixture permissions, and focused deterministic tests. It does not change endpoint behavior, mapper/evaluator logic, authoritative-write boundaries or production configuration.

### Provider profile hardening

The osteoporosis provider profile now explicitly defines exact provider-facing value kinds and code sets for the mapped concepts exercised by the promotion suite, including:

- fracture-site codes;
- treatment/administration/decision agent codes;
- treatment and administration statuses;
- decision types;
- patient acceptance codes;
- follow-up task codes with exact `DXA` casing and `followup_visit`;
- boolean/integer/number/quantity/date expectations.

It also freezes semantic ownership:

```text
actual/historical treatment → treatment.*
option discussed → decision.selected_agent + option_discussed
clinician recommendation → decision.selected_agent + clinician_recommendation
final selected plan → decision.selected_agent + final_decision
administration.* → only explicit actual/planned administration event
prescription/recommendation alone → never administration truth
uncertain administration occurrence → uncertain_needs_review, no administration status
third-party fact → speaker=third_party
explicit never/not received → polarity=negative, no positive treatment status
unrelated non-osteoporosis narrative → no osteoporosis task creation
```

### Four narrow fixture corrections only

The default-deny oracle remains intact. Only these source-supported extras are now allowed, with exact semantic/source/value/mapping constraints:

- `followup_vague`: `followup.task_type=followup_visit`;
- `followup_exact`: `followup.task_type=followup_visit`;
- `frax_original_adjusted`: `frax.tool_name=frax`;
- `out_of_range_runtime_values`: `frax.tool_name=frax`.

No wildcard permission and no blanket relaxation were introduced. Hard administration/source/negation cases remain fail-closed.

## Exact deterministic evidence

The H-07 remediation lifecycle contained two harness-only wording corrections before the gate could close. Neither changed provider profile, fixtures or runtime behavior. The final exact head is:

```text
e066c87bf6b42c2c56c80b2b71b43bc909510d39
```

Workflow `35142115023` passed:

- Python syntax — PASS;
- browser syntax — PASS;
- **56 focused PR-1 tests — PASS**;
- inherited protected-clinical regressions — 6 PASS;
- inherited Medical Report regressions — 24 PASS;
- inherited workspace/navigation regression — PASS;
- bounded PR-1 scope verification — PASS.

Therefore the bounded H-07 remediation is deterministically proven and the canonical barrier may advance to the next live qualification attempt.

## H-08 third live qualification — CHECKPOINTED

Workflow `35142415250` executed the same frozen 22-case synthetic/de-identified suite through `gpt-5.6`.

Boundary checks passed before provider execution:

- Actions `OPENAI_API_KEY` available and masked;
- `CLINICAL_TRANSCRIPT_PHI_PROVIDER_APPROVED=false`;
- frozen fixture count = 22;
- strict Structured Outputs/provider schema path accepted.

Live result:

```text
18 PASS
4 FAIL
```

Exact coded failures:

```text
negative_history_vs_negative_investigation
  unexpected_assertion_vfa.modality

referral_not_completed_result
  unexpected_assertion_clinical.unmapped_narrative

prescription_not_administration
  missing_expected_concepts
  required_assertion_0_missing
  unexpected_assertion_decision.selected_agent

planned_not_done_administration
  unexpected_assertion_followup.task_type
```

Read-only triage against the frozen H-07 profile/fixtures found:

- three source-supported extras are fixture/oracle overconstraints: `vfa.modality=VFA`, `clinical.unmapped_narrative` for the explicit absence-of-result statement, and `followup.task_type=administration` for an explicitly scheduled administration;
- `prescription_not_administration` exposes an ontology/profile ambiguity: the profile generally assigns clinician recommendation to `decision.selected_agent`, while a later special clause also permits `treatment.agent`. Because `treatment.agent` maps to a treatment episode and `decision.selected_agent` remains ambiguous unless final_decision, the safer canonical recommendation representation is `clinician_recommendation + decision.selected_agent`.

No mapper/evaluator defect is demonstrated by these four coded failures.

## H-08 bounded remediation — COMPLETE deterministically

The bounded H-08 patch made only the authorized changes:

- exact source-supported permission for `vfa.modality=VFA` in `negative_history_vs_negative_investigation`;
- exact source-supported `clinical.unmapped_narrative` permission for the explicit no-result statement in `referral_not_completed_result`;
- exact `followup.task_type=administration` permission for the explicitly scheduled administration task;
- prescription/recommendation semantics canonicalized to `clinician_recommendation + decision.selected_agent`, with `treatment.agent` forbidden for prescription-only wording;
- focused H-08 regressions added; default-deny remains intact.

An initial deterministic run `35446828051` reached 59 PASS / 1 FAIL in the focused suite. The sole failure was a brittle pre-existing H-07 string assertion (`must NEVER` vs `and NEVER`) after the safety sentence was reworded without semantic relaxation. A harness-only assertion correction produced exact head:

```text
5eeddeba664814b38d55bad8231afb5b33448eae
```

Workflow `35446869081` completed SUCCESS with:

- Python syntax PASS;
- browser syntax PASS;
- **60 focused PR-1 tests PASS**;
- inherited protected-clinical regressions PASS;
- inherited Clinical Documents regressions PASS;
- inherited workspace/navigation regression PASS;
- bounded PR-1 scope PASS.

No mapper/evaluator relaxation, production configuration change or PHI processing occurred.

## Fourth live GPT-5.6 qualification — PASS

The workflow-only trigger head is:

```text
85d3a11ddff676a4569d1951af79aa2c28175e5d
```

The trigger-head deterministic gate `35446962951` completed SUCCESS across syntax, focused PR-1 tests, inherited clinical regressions, Clinical Documents regressions, navigation and bounded scope.

Live synthetic provider workflow `35446962808` then passed:

- synthetic-only execution boundary — PASS;
- provider/model declaration — `openai / gpt-5.6 / purpose=synthetic_eval / phi_approval=false`;
- frozen promotion fixtures — `total=22 ids_unique=true`;
- all 22 case-level oracle checks — PASS;
- final summary — `total=22 failed=0`.

Therefore the selected-model synthetic promotion gate is satisfied for the frozen 22-case suite.

This does **not** authorize production, PHI use, release PR, merge/deploy or PR-2.

## Independent promotion review — HOLD_FOR_SEMANTIC_REMEDIATION

A fresh independent READ-ONLY review verified the exact ancestry, deterministic chain and live GPT-5.6 evidence. It independently confirmed that workflow `35446962808` is genuine selected-model evidence with all 22 frozen cases PASS and `failed=0`.

The review did **not** invalidate that live result. It found one material deterministic-boundary defect before PR-1 semantic promotion can close:

- schema-valid provider output may still map `treatment.*` / `administration.*` into runtime targets under semantically incompatible states such as `clinician_recommendation`;
- a negated fracture/event presence assertion such as `polarity=negative + fracture.site=hip` can still become positive mapped runtime truth;
- the Module-01 target guard therefore does not yet fully enforce the provider-as-untrusted architecture.

Independent disposition:

```text
HOLD_FOR_SEMANTIC_REMEDIATION
```

Separate non-semantic release blockers remain unchanged: H-05 async/provider execution, executable browser lifecycle/BFCache/logout evidence, and the identifiable-transcript privacy/provider gate.

The review also noted two medium evaluator-hardening debts (unconstrained authorized free-text content and duplicate authorized candidate cardinality) and one low adapter diagnostic-classification weakness. These are recorded but are not the smallest blocking remediation action.

## H-09 bounded remediation contract

Authorized H-09 mutation is limited to the deterministic Module-01 target boundary plus focused adversarial tests:

1. `treatment.*` runtime mapping must fail closed when the candidate semantic type is not an actual/history treatment state;
2. `administration.*` runtime mapping must fail closed outside actual/history/objective/follow-up administration-event semantics;
3. negated fracture/treatment/administration presence/event assertions must not become positive mapped runtime truth;
4. existing valid scheduled-administration follow-up behavior must remain mapped;
5. no provider prompt/profile relaxation, evaluator/default-deny relaxation, endpoint behavior change, persistence change or production configuration change is authorized.

Use deterministic `ambiguous` mappings with explicit reason codes rather than silently discarding provider assertions.

## H-09 deterministic semantic boundary — COMPLETE deterministically

The bounded H-09 patch now enforces the previously missing local semantic boundary:

- all `treatment.*` episode concepts are mapped only for `patient_history_fact`; incompatible option/recommendation/decision semantics become `ambiguous` with `SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE`;
- all `administration.*` event concepts are mapped only for `patient_history_fact`, `objective_result` or `followup_task`; recommendation/option/decision semantics become `ambiguous` with `SEMANTIC_TYPE_NOT_ALLOWED_FOR_ADMINISTRATION_EVENT`;
- negated fracture presence, treatment episode and administration event concepts become `ambiguous` with `NEGATED_ASSERTION_NOT_POSITIVE_RUNTIME_VALUE`;
- legitimate actual treatment history, actual administration and scheduled follow-up administration remain mapped.

The first H-09 gate `35448114022` reached 64 focused PASS / 1 FAIL. The sole failure was a test-harness double-mapping error in the newly added scheduled-administration regression; runtime code was unchanged by the correction.

Final exact H-09 head:

```text
00985996fe7c32a9596c5440d27223552d290d06
```

Workflow `35448153136` completed SUCCESS:

- Python syntax PASS;
- browser syntax PASS;
- **65 focused PR-1 tests PASS**;
- 6 inherited protected-clinical tests PASS;
- 24 inherited Clinical Documents / Medical Report tests PASS;
- workspace/navigation PASS;
- bounded PR-1 scope PASS.

The independent HIGH deterministic semantic-boundary finding is therefore closed deterministically. This does not yet restore promotion PASS because runtime semantic code changed after the previously recorded 22/22 provider qualification.

## Post-H09 live GPT-5.6 qualification — FAIL CHECKPOINT

Workflow-only trigger head:

```text
d2c878cc036b656b273a3684d3726c2fb8ae8353
```

Trigger-head deterministic workflow `35448274111` completed SUCCESS with the H-09 65-test gate and all inherited checks.

Live synthetic qualification workflow `35448273993` passed:

- synthetic-only execution boundary;
- `provider=openai / model=gpt-5.6 / purpose=synthetic_eval / phi_approval=false`;
- frozen fixture count `total=22 ids_unique=true`.

The 22-case semantic qualification then returned:

```text
19 PASS
3 FAIL
provider-eval summary: total=22 failed=3
```

Exact coded failures:

```text
speaker_ambiguity
  required_assertion_0_missing
  unexpected_assertion_treatment.agent

unrelated_general_clinical_text
  unexpected_assertion_clinical.unmapped_narrative
  semantic_count_clinician_recommendation_mismatch

referral_not_completed_result
  unexpected_assertion_clinical.unmapped_narrative
```

No provider/schema/credential/PHI boundary failure occurred. No runtime or fixture mutation is authorized from this result until read-only triage determines whether each failure represents provider variability, a demonstrated oracle/fixture defect, or an additional semantic contract issue.

## Read-only three-case triage — COMPLETE

The post-H09 failure checkpoint `d85ca94e4fa86f7ea4c98761e2451910b81f3109` was verified by workflow `35448460778` before triage.

### 1. speaker_ambiguity — deterministic fixture regression

The frozen fixture still expects:

```text
uncertain_needs_review + treatment.agent
→ mapped step4.treatment_episodes[].agent
```

H-09 intentionally changed that semantic combination to fail closed:

```text
uncertain_needs_review + treatment.agent
→ ambiguous
→ SEMANTIC_TYPE_NOT_ALLOWED_FOR_TREATMENT_EPISODE
```

Therefore `required_assertion_0_missing` plus `unexpected_assertion_treatment.agent` is caused by a stale fixture mapping expectation, not by a new provider semantic defect.

### 2. unrelated_general_clinical_text — provider/profile semantic ambiguity

The required patient shoulder complaint remains `patient_history_fact + clinical.unmapped_narrative`.

The new live run also emitted a second `clinical.unmapped_narrative` under `clinician_recommendation`, producing both the unexpected assertion and the recommendation-count mismatch.

The source phrase “Θα το εξετάσουμε ξεχωριστά” is a generic cross-domain deferral, not a concrete osteoporosis recommendation/task. The provider profile currently says to retain clinically meaningful unrelated content as `clinical.unmapped_narrative` but does not explicitly forbid turning such a generic deferral into `clinician_recommendation`.

The fixture's no-recommendation safety intent is retained. Provider guidance must be made explicit rather than loosening the fixture to accept an osteoporosis recommendation semantic.

### 3. referral_not_completed_result — provider/profile metadata ambiguity

The required DXA follow-up task passed. No forbidden DXA result concept was emitted.

Only an extra `clinical.unmapped_narrative` failed authorization. The frozen optional rule currently accepts it only as `clinician_interpretation` from the clinician. The coded logs do not retain provider payload metadata, so the exact mismatching semantic/source field cannot be reconstructed without a new provider call.

The safe contract is:

```text
explicit referral/request
→ followup_task

explicit “no result yet”
→ may be retained only as non-runtime clinical.unmapped_narrative
→ clinician_interpretation
→ source-anchored evidence
→ never DXA objective-result fields
```

### Independent MEDIUM evaluator findings now in scope

Because both failed cases exercise `clinical.unmapped_narrative`, H-10 also closes the two previously demonstrated evaluator debts directly relevant to promotion confidence:

- allowed free-text assertions must be source-bound through an explicit evidence rule; blank evidence must not authorize arbitrary text metadata;
- exact duplicate authorized candidates must produce a deterministic promotion failure.

## H-10 bounded remediation contract

Authorized mutation is limited to:

1. update `speaker_ambiguity` fixture mapping expectation from mapped treatment truth to H-09 ambiguous/reason-code truth;
2. harden provider profile so a generic unrelated “we will examine it separately” statement does not become `clinician_recommendation` or an osteoporosis follow-up task;
3. harden referral/no-result profile semantics so optional no-result narrative is `clinician_interpretation + clinical.unmapped_narrative`, never objective result/task truth;
4. add evaluator support for explicit source-bound `evidence_contains` rules and apply it to optional free-text narrative allowances where needed;
5. fail exact structurally duplicate provider candidates in the promotion evaluator;
6. add focused deterministic regressions.

No H-09 target-guard relaxation, no general default-deny relaxation, no new wildcard fixture permission, no endpoint/persistence/production configuration change.

## H-10 semantic + promotion-oracle stabilization — COMPLETE deterministically

H-10 implemented only the frozen triage contract:

- `speaker_ambiguity` now expects H-09 ambiguous treatment mapping rather than mapped episode truth;
- generic unrelated clinician deferral is explicitly not `clinician_recommendation` and not an osteoporosis follow-up task;
- referral + explicit no-result wording may retain only `clinician_interpretation + clinical.unmapped_narrative`, never DXA result truth;
- optional free-text narrative allowances are source-bound through `evidence_contains`;
- exact structurally duplicate authorized candidates produce `duplicate_candidate` promotion failure;
- focused H-10 regressions cover the new contracts.

The first H-10 deterministic attempt `35455364544` reached 69 focused PASS / 1 FAIL. The single failure was a stale H-08 exact-dict test that did not include the new stricter `evidence_contains` field. The correction changed only the test expectation.

Final exact H-10 head:

```text
eee45f9bab1fa0eba9fc36dea8fdcbf61dd89d5a
```

Workflow `35455400599` completed SUCCESS:

- Python syntax PASS;
- browser syntax PASS;
- **70 focused PR-1 tests PASS**;
- 6 inherited protected-clinical tests PASS;
- 24 inherited Clinical Documents / Medical Report tests PASS;
- workspace/navigation PASS;
- bounded PR-1 scope PASS.

No H-09 target-guard relaxation, endpoint/persistence change, production config mutation or PHI processing occurred.

## Post-H10 live GPT-5.6 qualification — FAIL CHECKPOINT

Workflow-only trigger head:

```text
9d77df8569a0037ec51204e6724a0c62c4bcc552
```

Trigger-head deterministic gate `35455512998` completed SUCCESS with the H-10 70-test gate and all inherited checks.

Live synthetic workflow `35455512992` passed:

- synthetic-only boundary;
- `provider=openai / model=gpt-5.6 / purpose=synthetic_eval / phi_approval=false`;
- fixture count `total=22 ids_unique=true`.

Live semantic result:

```text
20 PASS
2 FAIL
provider-eval summary: total=22 failed=2
```

Exact coded failures:

```text
negative_history_vs_negative_investigation
  unexpected_assertion_vfa.action

prescription_not_administration
  unexpected_assertion_clinical.unmapped_narrative
```

No provider/schema/credential failure occurred. H-09 runtime hardening and H-10 stricter evaluator remained active.

## Post-H10 two-case read-only triage — COMPLETE

The 20/22 failure checkpoint `c54a43bdf1075f4243f9c7cb809be0606b8de64c` was verified by workflow `35455697350` before triage.

### negative_history_vs_negative_investigation

Source:

```text
Ιατρός: Στη VFA δεν βρέθηκε σπονδυλικό κάταγμα.
```

The required negative vertebral-fracture result passed. The only extra was `vfa.action`.

The source supports that VFA evidence exists and is being referenced/reviewed. It does **not** establish that the VFA was performed during the current encounter. Therefore the safest canonical action code is:

```text
vfa.action = already_available_reviewed
semantic_type = objective_result
speaker = clinician
evidence anchored to "Στη VFA"
```

Do not infer `performed` unless the transcript explicitly states performance.

### prescription_not_administration

Source:

```text
Συνταγογραφήθηκε denosumab ως προτεινόμενη θεραπεία.
Δεν αναφέρεται ότι έγινε χορήγηση.
```

The required `clinician_recommendation + decision.selected_agent=denosumab` passed. No forbidden administration concept was emitted. The only extra was `clinical.unmapped_narrative`.

“Δεν αναφέρεται ότι έγινε χορήγηση” is absence-of-documentation wording. It is neither evidence of a completed administration nor a negative administration event. If retained, it may exist only as:

```text
clinician_interpretation
+ clinical.unmapped_narrative
+ source-bound evidence
+ unmapped / NO_CURRENT_RUNTIME_TARGET
```

It must not create `administration.*`, `treatment.agent`, or positive/negative administration truth.

## H-11 bounded remediation contract

Authorized H-11 changes are limited to provider profile, these two exact fixture permissions and focused regressions:

1. normalize referenced VFA-result wording to `vfa.action=already_available_reviewed` unless explicit performance is stated;
2. permit that exact source-supported VFA action in `negative_history_vs_negative_investigation`;
3. normalize absence-of-administration-documentation wording to source-bound `clinician_interpretation + clinical.unmapped_narrative`;
4. permit that exact non-runtime narrative in `prescription_not_administration`;
5. preserve all H-09 target guards and H-10 evidence/duplicate/default-deny hardening.

No wildcard permission, no evaluator relaxation, no runtime-target relaxation and no production/PHI changes.

## H-11 source-semantic normalization — COMPLETE deterministically

H-11 implemented only the frozen two-case triage contract:

- referenced VFA-result wording without explicit current performance now instructs `vfa.action=already_available_reviewed`;
- the VFA fixture allows that exact objective-result action only when source evidence contains `Στη VFA`;
- `vfa.action=performed` is explicitly forbidden for that source because current-encounter performance is not stated;
- “Δεν αναφέρεται ότι έγινε χορήγηση” is classified as absence-of-documentation, not administration truth or negative administration truth;
- the prescription fixture permits only source-bound `clinician_interpretation + clinical.unmapped_narrative` for that sentence;
- all H-09 target guards and H-10 evidence/default-deny/duplicate protections remain unchanged.

Exact H-11 head:

```text
a6bb0b2a8897f83c6584106e03c26b59aa9c266b
```

Workflow `35455876663` completed SUCCESS:

- Python syntax PASS;
- browser syntax PASS;
- **75 focused PR-1 tests PASS**;
- 6 inherited protected-clinical tests PASS;
- 24 inherited Clinical Documents / Medical Report tests PASS;
- workspace/navigation PASS;
- bounded scope PASS.

## Exact next action

After this canonical checkpoint itself verifies, trigger the synthetic-only workflow against the verified H-11 branch state and execute the same 22 transcript inputs through `gpt-5.6`.

Promotion still requires `failed=0`, a durable live-evidence checkpoint and a fresh independent READ-ONLY promotion review.

H-05 and browser lifecycle remediation remain blocked pending that review.

## Explicitly blocked

Until a live 22-case qualification passes and its evidence receives fresh independent review:

- no runtime release PR;
- no merge/deploy of PR-1;
- no identifiable transcript processing;
- no PR-2 or real-patient pilot;
- no production credential/config mutation;
- no claim that deterministic H-07 closure equals live promotion PASS.

## Separate production-release blockers/debt

**H-05 remains OPEN:** synchronous provider execution inside the async single-worker web process must be remediated before production enablement/deploy.

Executable browser lifecycle/BFCache/logout cleanup evidence also remains release debt.
