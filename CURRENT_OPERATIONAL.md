# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PR-1 H13 DETERMINISTIC GATE PASS / LIVE QUALIFICATION PENDING; NOT RELEASE READY.
> **Updated:** 2026-10-01 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `63e903e05c1bfe22ca925374b8994355f6c92baf`.
> **Active slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`.
> **Active writer:** fresh separate H13 correction author; only the promotion evaluator, exact FRAX fixture authorization, focused H13 tests and branch-local checkpoints.
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
> **Post-H10 live qualification:** `35455512992` — 20 PASS / 2 FAIL.
> **Post-H11 live qualification:** `35455986630` — **22 PASS / 0 FAIL**.
> **H-12 exact deterministic head:** `b1d57e1c6e9feb6968592f700da7de8113badf96`.
> **H-12 deterministic/inherited gate:** `36821972100` — SUCCESS (82 focused, 6 protected clinical, 24 Clinical Documents).
> **First post-H12 live qualification:** `36822187245` — 21 PASS / 1 FAIL at trigger head `1c201203a60d645595d04613727ff329271b8a96`.
> **H-12 corrected exact deterministic head:** `864b5a0b389c165f7dfc2a0a4461147d7ffb548d`; gate `36822645627` — SUCCESS (83 focused).
> **Second post-H12 live qualification:** `36822789808` — 20 PASS / 2 FAIL at trigger head `750338794020c92f2f0bcd18a556231506b0f902`.
> **Content-free diagnostic head/gate:** `423b53c983d45c43f2c61e98ad4e437001923faa` / `36823179758` — SUCCESS (83 focused).
> **Third post-H12 live qualification:** `36823333376` — 19 PASS / 3 FAIL at trigger head `2ccd5a333e64abdb1568ba712f0513d9991143e8`.
> **Paraphrase-rule exact deterministic head/gate:** `473ada825cd79e69145f01038480be3d111f7d5a` / `36823978604` — SUCCESS (83 focused).
> **Fourth post-H12 live qualification:** `36824150686` — 21 PASS / 1 FAIL at trigger head `51490713a4a57820fa42d1407d411da149d17876`.
> **H-12 hardened exact deterministic head/gate:** `7f97359fc8860f209e18c07e5b76c2ac26f0f100` / `36824814247` — SUCCESS (84 focused).
> **Fifth post-H12 live qualification:** `36824997276` — 21 PASS / 1 FAIL at trigger head `c43ce7ab7a45995f4127f69d4c2fe219f15badb4`; `garbled_speech` emitted zero candidates.
> **Exact H-12 qualified head:** `43a501f8c51854e69062c66d9855fbee7cb47259`; complete deterministic gate `36825475834` — SUCCESS (84 focused, 6 protected clinical, 24 Clinical Documents, syntax, navigation and scope).
> **Sixth post-H12 live qualification:** `36825475846` — **22 PASS / 0 FAIL**, `openai / gpt-5.6 / synthetic_eval / phi_approval=false`, 22 unique frozen cases.
> **H13 exact implementation head:** `89a3b3d52fa4a2dd59d3b2d30c405555c6d501da`.
> **H13 complete deterministic gate:** `36844603110` — SUCCESS (88 focused, 6 protected clinical, 24 Clinical Documents, syntax, navigation, effective scope).
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

- secure Actions credential available and masked;
- identifiable-transcript provider approval remained false;
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

## Post-H11 live qualification — PASS

Exact trigger head: `e0522220caf54857d6ac14e91c0aa255619f3b62`.

Deterministic trigger gate `35455986818` — SUCCESS.

Live selected-model qualification `35455986630`:

```text
gpt-5.6 synthetic evaluation
identifiable-transcript approval=false
22 fixtures / unique IDs
22 PASS / 0 FAIL
provider-eval summary: total=22 failed=0
```

This evidence includes the H-09 semantic/polarity guard, H-10 source-bound and duplicate-candidate oracle hardening, and H-11 source-semantic normalization.

## Exact next action

The source-vocabulary narrative-value hardening is exact-head tested at `7f97359fc8860f209e18c07e5b76c2ac26f0f100` / gate `36824814247`. Verify this checkpoint, rerun the unchanged frozen 22-case GPT-5.6 suite with PHI approval false, and require 22/22 before handback. The VFA case's extra falls assertion/missing required assertion remains unsupported and must not be authorized. Do not change transcript inputs, rebase or perform release engineering.

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

## H-12 deterministic semantic-safety checkpoint — PASS

Base branch head `d0844ef4f7bb80c812d7c7856955b7d8694f8553` was fresh-verified with no runtime/semantic delta after H-11; remote `main` was fresh-verified at `63e903e05c1bfe22ca925374b8994355f6c92baf`. No rebase occurred.

- H12-A: future/planned treatment episodes cannot become current/historical exposure. Future/planned administration assertions containing `status=done` or `actual_date` fail closed as an entire administration candidate, with deterministic temporal reason codes. Valid future follow-up `planned` mapping and past actual administration remain mapped. H-09 semantic and negation guards retain priority.
- H12-B initial correction required a fixture-specific source phrase in each narrative value and a contiguous normalized source transcript span. Later live evidence showed source-supported paraphrase variation in FRAX/shoulder cases; the current narrower per-fixture contract is recorded below. Authentic evidence-snippet anchoring, default-deny and duplicate rejection remain active. Frozen transcript inputs did not change.
- Before correction: focused adversarial tests reproduced 5 failures, including future `done`/`actual_date`, future active treatment and authentic DXA evidence paired with invented denosumab narrative. After correction: all 7 H-12 tests pass, including appended invented text rejection and valid planned/past controls.

Exact implementation/tested head: `b1d57e1c6e9feb6968592f700da7de8113badf96`. Complete GitHub Actions gate `36821972100` SUCCESS: Python/browser syntax, 82 focused PR-1 tests, 6 protected-clinical tests, 24 Clinical Documents tests, workspace navigation and a repaired effective branch-scope check all PASS. The first run `36821826827` passed tests but exposed a pre-existing shallow-fetch false PASS in its scope step; the bounded workflow correction at `b1d57e1` made a real merge base mandatory. No production, PHI, PR-2, H-05 or browser-lifecycle changes were made.

First post-H12 live trigger head `1c201203a60d645595d04613727ff329271b8a96` passed the full deterministic gate `36822187284`. Live synthetic run `36822187245` verified `provider=openai / model=gpt-5.6 / purpose=synthetic_eval / phi_approval=false`, 22 unique frozen inputs, then returned 21 PASS / 1 FAIL. The sole coded failure was `frax_original_adjusted: unexpected_assertion_clinical.unmapped_narrative`; no provider, schema, credential or PHI-boundary failure was reported. This is a failure checkpoint, not promotion PASS. Raw candidate values were not logged.

The sole narrowly justified fixture correction replaces the FRAX narrative value's full-sentence `value_text_contains` anchor with source-derived `κίνδυν`, allowing a shorter quote of the same source-stated higher-risk interpretation. The separate exact normalized source-span requirement and full-sentence authentic `evidence_contains` remain mandatory, so an invented extension still fails. No frozen transcript input changed. New focused regression proves the shorter genuine quote passes and the invented denosumab extension fails. Corrected exact head `864b5a0b389c165f7dfc2a0a4461147d7ffb548d`; complete deterministic gate `36822645627` SUCCESS with 83 focused, 6 protected-clinical, 24 Clinical Documents, syntax, navigation and effective scope PASS. Live requalification remains pending.

Second trigger head `750338794020c92f2f0bcd18a556231506b0f902` passed complete deterministic gate `36822789835`. Live run `36822789808` passed the same `openai / gpt-5.6 / synthetic_eval / phi_approval=false` and 22-unique-fixture boundaries, then returned 20 PASS / 2 FAIL. `frax_original_adjusted` again had only `unexpected_assertion_clinical.unmapped_narrative`; current coded output cannot distinguish which new narrative value check failed. `negative_history_vs_negative_investigation` had `required_assertion_0_missing` and `unexpected_assertion_risk.falls_last_12_months`, a separate provider-output failure absent in the first H-12 run. There is no source basis to permit the falls assertion. Candidate text was not logged. Neither result is promotion PASS.

The evaluator now emits only additional reason codes for unmatched narrative rules: evidence anchor, value phrase anchor, or whole-value source-span mismatch. It never prints candidate text or evidence. Existing default-deny failure remains. Exact diagnostic head `423b53c983d45c43f2c61e98ad4e437001923faa` passed gate `36823179758`: 83 focused, 6 protected-clinical, 24 Clinical Documents, syntax, navigation and effective scope PASS. The next frozen synthetic run is diagnostic evidence, not permission to weaken unrelated source semantics.

Third trigger head `2ccd5a333e64abdb1568ba712f0513d9991143e8` passed deterministic gate `36823333355`. Live run `36823333376` passed synthetic-only/22-unique-case boundaries but returned 19 PASS / 3 FAIL: `frax_original_adjusted` reported `narrative_value_source_span_mismatch` only (its value anchor passed); `unrelated_general_clinical_text` reported a missing required assertion plus an optional narrative that missed both value and source-span anchors; `negative_history_vs_negative_investigation` again missed its required assertion and emitted unsupported `risk.falls_last_12_months`. These are coded classifications without candidate text. The FRAX code is consistent with paraphrase but does not expose the full claim; a future rule must still reject unsupported additions. The VFA falls claim has no source authorization and remains default-denied.

Narrow fixture-level correction at `473ada825cd79e69145f01038480be3d111f7d5a`: FRAX narrative authorization keeps the full source-evidence anchor and source-derived risk-value anchor `κίνδυν`, while no longer requiring verbatim whole-value text; shoulder patient/clinician narrative rules use source-derived anatomical/separate-review stems to permit inflection/paraphrase. Exact whole-value source-span binding remains mandatory on the reviewer-reproduced DXA no-result and H-11 prescription/no-administration cases. All optional narrative permissions retain explicit value anchors and authentic `evidence_contains`; default-deny and duplicate rejection remain active. The new local test still rejects an unrelated invented denosumab value, while the DXA adversarial test rejects both invented-only and appended denosumab claims. No frozen transcript input changed. Complete gate `36823978604` SUCCESS: 83 focused, 6 protected-clinical, 24 Clinical Documents, syntax, navigation and effective scope PASS. This is deterministic evidence only; live 22/22 remains unproven.

Fourth trigger head `51490713a4a57820fa42d1407d411da149d17876` passed complete deterministic gate `36824150739`. Live synthetic run `36824150686` returned 21 PASS / 1 FAIL after all privacy/model/fixture-count boundaries passed. FRAX and shoulder narratives passed; `negative_history_vs_negative_investigation` again had `required_assertion_0_missing` and unsupported `risk.falls_last_12_months`. No oracle permission was added for this fabricated falls claim. The current FRAX/shoulder value stem rule can still authorize an unrelated clause appended to a source-related narrative, so H12-B is not yet considered deterministically closed despite the 21/22 live result.

H12-B hardening at exact head `7f97359fc8860f209e18c07e5b76c2ac26f0f100` adds a deterministic source-vocabulary check to the paraphrase-capable FRAX and shoulder rules. Every substantive value token must occur in the frozen source transcript or a short, explicit fixture list of source-supported inflection/paraphrase terms; source-specific value anchors and authentic `evidence_contains` remain mandatory. Reviewer-class DXA and H-11 prescription narratives retain the stricter whole-value source-span rule. Focused regressions prove legitimate paraphrases pass and appended unrelated denosumab administration claims fail in FRAX and shoulder, as well as DXA. Complete gate `36824814247` SUCCESS: 84 focused PR-1, 6 protected-clinical, 24 Clinical Documents, syntax, navigation and effective scope PASS. This is deterministic evidence; post-hardening live 22/22 remains pending. No frozen transcript input, provider target path or authoritative-write boundary changed.

Fifth trigger head `c43ce7ab7a45995f4127f69d4c2fe219f15badb4` passed complete deterministic gate `36824997233`. Live synthetic run `36824997276` verified `openai / gpt-5.6 / synthetic_eval / phi_approval=false` and 22 unique frozen inputs, then returned 21 PASS / 1 FAIL. All H-12 narrative and temporal cases passed; `garbled_speech` emitted zero candidates and missed its required ambiguity assertion. This is a provider-output failure, not a reason to relax the frozen oracle. The same frozen suite must pass 22/22 before independent review readiness.

Sixth trigger at exact head `43a501f8c51854e69062c66d9855fbee7cb47259` changed only the qualification workflow comment after the fifth-result checkpoint; implementation, evaluator, fixtures and transcript inputs were unchanged. Complete deterministic gate `36825475834` SUCCESS: syntax, 84 focused PR-1, 6 protected clinical, 24 Clinical Documents, navigation and effective scope. Live run `36825475846` verified the synthetic-only provider boundary and 22 unique frozen fixtures, then returned 22 PASS / 0 FAIL, including all H-09/H-10/H-11 and H-12 cases. Earlier live variation (VFA/falls and garbled speech) remains a residual reliability observation; no oracle, transcript or provider profile was relaxed. The bounded H-12 implementation is ready for a separate independent delta+cumulative promotion review; this author has not performed that review or any release action.


## Independent H-12 delta+cumulative review — HOLD

Fresh separate independent review verified the exact H-12 lineage through qualified head `43a501f8c51854e69062c66d9855fbee7cb47259` and final checkpoint `53ad7d6babb9eee9f0c642ac804cc65769c19dfc`.

Disposition:

```text
H12-A TEMPORALITY: CLOSED
H12-B FREE-TEXT SOURCE BINDING: PARTIALLY CLOSED
H09: PRESERVED
H10: PRESERVED
H11: PRESERVED
FROZEN INPUTS: UNCHANGED
PRIVACY / AUTHORITY BOUNDARY: PASS
VERDICT: HOLD_FOR_SEMANTIC_REMEDIATION
```

The remaining material residual is narrow:

- H-12 source-vocabulary / anchor checks can still accept a clinically false relation among authentic source values.
- Independent adversarial reproduction used the real FRAX source meaning `MOF 18%, hip 4%` and a narrative that preserved source vocabulary/numbers but swapped the relationships: `hip 18%, MOF 4%`.
- The final H-12 evaluator returned a false PASS because token/stem membership does not prove that each numeric value remains bound to the correct clinical measure.

This is not a provider-output finding from the final 22/22 run; it is a deterministic oracle-integrity finding.

### H13 bounded correction objective

Close only the remaining relational-binding residual:

```text
AUTHENTIC SOURCE TOKENS / NUMBERS
!=
AUTHORIZED CLINICAL RELATIONSHIP

MEASURE ↔ VALUE associations
must remain source-grounded.
```

H13 may change only the existing promotion evaluator, exact affected fixture authorization, focused tests, and these branch-local checkpoints unless source evidence proves another bounded owner is necessary.

H13 must not:
- relax default-deny;
- alter the 22 transcript inputs;
- add a second model/judge;
- reopen H12-A;
- weaken H09/H10/H11;
- start H-05 or browser-lifecycle work;
- rebase to current main;
- open release PR / merge / deploy / enable PHI.

Exact next sequence:

```text
H13 bounded relational-binding correction
→ deterministic adversarial PASS
→ full inherited gate PASS
→ rerun unchanged frozen 22-case GPT-5.6 qualification
→ fresh independent delta+cumulative review
→ only PASS_TO_RELEASE_ENGINEERING may advance
```

## H13 implementation start — 2026-10-01

Fresh remote verification found `main` at `63e903e05c1bfe22ca925374b8994355f6c92baf` and PR-1 at the expected reconciled `b3e8680ba1080e28eb7c51e48975852dd5a103eb`. The two commits since the local H12 checkpoint changed only this operational record and `SLICE_PLAN_CURRENT.md`; no semantic/runtime drift occurred. H13 is active within the bounded evaluator/fixture/test/checkpoint scope above. The frozen 22 transcript inputs remain immutable.

## H13 candidate — local checks complete, CI and live qualification pending

The promotion evaluator now checks the exact FRAX source pairings `MOF ↔ 18` and `hip ↔ 4` whenever the optional interpretation narrative restates numeric risk values. Candidate pairings may be reordered or omit the percent glyph, but each numeric value needs an explicit matching measure. Numeric claims without a measure, swapped values and contradictory source pairings fail closed. Existing H12 evidence, value anchor, source vocabulary, duplicate and default-deny checks remain in force. No runtime mapper, provider profile, transcript input or authoritative-write path changed.

Local evidence: 88 focused PR-1 tests PASS, 7 runnable inherited clinical/SDK tests PASS, Python/browser syntax PASS, workspace navigation PASS, and `git diff --check` PASS. The two local Clinical Documents PDF modules could not collect because this shell lacks the declared `PyMuPDF` dependency; the complete Actions gate installs `requirements.txt` and proves those tests below. All 22 frozen transcript strings and IDs match `b3e8680` byte-for-byte; ordered transcript-list SHA-256 is `6df23521cf7280b82491982fabbdc4e56ff70f219c2991e43f7a98149322b694`.

## H13 exact-head deterministic checkpoint — PASS

At implementation head `89a3b3d52fa4a2dd59d3b2d30c405555c6d501da`, workflow [`36844603110`](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/36844603110) completed SUCCESS: 88 focused PR-1 tests, 6 protected-clinical tests, 24 Clinical Documents tests, Python/browser syntax, workspace navigation and effective branch-scope verification all passed. This resolves the local missing-PDF-dependency limitation. The next material step is to run the same frozen 22-case GPT-5.6 synthetic qualification, with `purpose=synthetic_eval` and PHI approval false, and require `failed=0` without relaxing the H13 oracle.
