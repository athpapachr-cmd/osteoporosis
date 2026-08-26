# SLICE_PLAN_CURRENT.md — PR-1 Transcript Intake + Candidate Extraction v1

> **STATUS:** ACTIVE REPLAN-CORRECTED DESIGN v3 — implementation not yet authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** reusable Personal Clinical Excellence System.
> **Proving module:** Module 01 — Osteoporosis.
> **Parent phase:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **Slice ID:** PR-1.
> **Product-owner direction:** enable paste of a Heidi transcript so useful encounter data can be extracted automatically, while preserving clinician control, keeping raw transcript ephemeral, and building the capability as reusable Clinical Excellence Core rather than osteoporosis-only code.

This file is the exact active PR-1 design. Runtime branch/PR/deploy/smoke state belongs in `CURRENT_OPERATIONAL.md`.

The v3 design supersedes v2 after a read-only pre-code review of the actual runtime and persisted encounter seams identified three material defects:

1. YAML/schema wording and actual browser-persisted runtime paths are not identical everywhere;
2. a singular `value + target_mapping` candidate is insufficient for composite clinical facts such as fracture events, treatment episodes and final decisions;
3. standard request-validation/error handling is not sufficient for a request body that may contain transcript PHI unless the endpoint owns a sanitized validation boundary.

These findings trigger an in-slice **REPLAN**, not a new roadmap phase.

---

# 1. Product outcome

After PR-1, the clinician can paste a Heidi consultation transcript and receive a structured, non-authoritative candidate preview.

```text
PASTE TRANSCRIPT
→ protected Clinical Excellence Core endpoint
→ provider extraction
→ strict structured-output validation
→ Core semantic candidates
→ deterministic Module 01 mapping
→ clinically readable ephemeral preview
→ discard raw transcript
→ NO authoritative patient-data write
```

PR-1 is capture infrastructure for the wider Personal Clinical Excellence System before, during and after clinic. Osteoporosis is the first domain profile and test case; it must not define the reusable Core architecture.

---

# 2. Hard scope boundary

PR-1 includes:

- Heidi transcript paste;
- protected `/clinical/transcript/extract` endpoint;
- reusable Core request/response/candidate contracts;
- provider-neutral extraction interface;
- first OpenAI adapter if the final provider verification passes;
- strict structured response validation;
- deterministic osteoporosis concept/target mapping;
- clinically readable preview;
- synthetic/de-identified tests/evals;
- explicit privacy/logging/failure controls.

PR-1 does **not**:

- Accept/Reject/Edit values into the patient record;
- mutate patients, encounters, labs, treatment history or tasks;
- persist transcript or candidate preview;
- alter KPI calculation;
- generate Practice Review coaching;
- redesign the osteoporosis consultation flow;
- modify Calendar/Setmore/Digital Secretary;
- implement unrelated Clinic Utilities work;
- use identifiable real transcripts before the separate provider/privacy gate is closed.

PR-2 remains the first slice that may authorize clinician-reviewed candidate data to become record truth.

---

# 3. Core-vs-Module ownership

## 3.1 Clinical Excellence Core owns

```text
/clinical/transcript/extract transport
request body/size validation
sanitized validation/error boundary
source_type / encounter_phase / language
semantic candidate envelope
composite component mechanism
speaker / polarity / temporality / certainty
source evidence-snippet lifecycle
provider protocol
provider timeout/retry/error normalization
privacy / logging rules
strict local validation
candidate transport to UI
module registry/dispatch interface
```

## 3.2 Provider adapter owns

```text
provider SDK/API invocation
structured-output request
provider-specific error classification
provider storage/retry/timeout settings
```

It contains **no osteoporosis clinical logic**.

## 3.3 Module 01 — Osteoporosis owns

```text
osteoporosis concept vocabulary
allowed component keys per concept
drug/site/lab normalization rules
runtime target registry
lossless/lossy/unmapped mapping decisions
fracture / FRAX / DXA / VFA / lab / treatment / task semantics
```

## 3.4 Critical rule

The model/provider must **never manufacture application target paths**.

```text
provider extracts concept + components + source semantics
→ deterministic Module01 code maps to runtime targets
```

The Core must not know T-scores, FRAX, Prolia/Aclasta, VFA indications or osteoporosis treatment categories.

---

# 4. Exact request contract v1

Route:

```text
POST /clinical/transcript/extract
```

Request:

```json
{
  "schema_version": "clinical_transcript_extract_request_v1",
  "source_type": "heidi_transcript",
  "module": "osteoporosis",
  "encounter_phase": "during_visit",
  "language": "el",
  "transcript": "<string>",
  "context": {
    "encounter_archetype": null
  }
}
```

Constraints:

- request body ceiling: **512 KiB**;
- `transcript`: stripped non-empty, maximum **120,000 Unicode characters**;
- `schema_version`: exact literal v1;
- `source_type`: v1 accepts `heidi_transcript`;
- `module`: resolved through the generic module registry; v1 registers `osteoporosis`;
- `encounter_phase`: v1 accepts `during_visit`;
- initial evaluated language: Greek (`el`), while the Core field remains generic;
- `context.encounter_archetype`: optional/null, bounded string;
- unknown request keys rejected;
- no patient name, DOB, phone, GeSY/EMR identifier, `patient_id`, encounter ID or authoritative encounter payload is required or sent merely to perform extraction.

A future provider call should receive only the minimum context required for extraction.

---

# 5. Sanitized request-validation boundary

Transcript input may contain PHI. The transcript endpoint must not delegate sensitive error representation to a default path that can echo invalid input.

Required behavior:

```text
raw request body
→ explicit size guard
→ controlled JSON parse
→ controlled local request validation
→ sanitized public error
```

Public errors must never contain:

- transcript text;
- Pydantic `input` payloads;
- provider prompt/output;
- raw provider exception strings/bodies;
- candidate clinical values.

This is a merge blocker, not optional hardening.

---

# 6. Exact success/error response contract

Success:

```json
{
  "schema_version": "clinical_transcript_candidates_v1",
  "request_id": "<uuid>",
  "source_type": "heidi_transcript",
  "module": "osteoporosis",
  "encounter_phase": "during_visit",
  "language": "el",
  "candidates": [],
  "warnings": [],
  "meta": {
    "processing_mode": "ephemeral_preview",
    "candidate_count": 0,
    "raw_persisted": false,
    "candidates_persisted": false,
    "authoritative_write": false
  }
}
```

Warnings are server-owned structured codes, not free provider prose. Initial examples:

```text
LOW_SOURCE_CLARITY
UNMAPPED_CANDIDATE
AMBIGUOUS_TARGET
EVIDENCE_NOT_VERIFIABLE
PARTIAL_EXTRACTION
```

Sanitized error:

```json
{
  "request_id": "<uuid>",
  "error": {
    "code": "TRANSCRIPT_TOO_LARGE",
    "message": "Transcript exceeds the supported size."
  }
}
```

No full transcript appears in any response.

---

# 7. Core candidate model — composite semantic assertion

The candidate unit is **one semantic assertion containing one or more related components**.

```text
ClinicalCandidate
  candidate_id              server-generated UUID
  module                    server/module-registry supplied
  domain                    module vocabulary
  concept_key               module vocabulary
  semantic_type             Core enum
  components[]              one or more typed components
  source_assertion          Core semantics
  evidence                  ephemeral
  extraction_assessment     non-authoritative
  target_mappings[]         deterministic module output
  warnings[]                coded only
  status                    always proposed
```

The provider does not supply `candidate_id`, `module`, target paths, mapping status or authoritative state.

---

# 8. Candidate components

Each component contains:

```text
component_key
value
unit_optional
```

Allowed typed value kinds:

```text
text
code
number
integer
boolean
quantity
date
```

Example fracture candidate:

```json
{
  "concept_key": "fracture_event",
  "semantic_type": "patient_history_fact",
  "components": [
    {"component_key": "site", "value": {"kind": "code", "code": "vertebral"}},
    {"component_key": "event_time", "value": {"kind": "date", "date": "2025-03", "precision": "month"}},
    {"component_key": "low_trauma", "value": {"kind": "code", "code": "yes"}}
  ]
}
```

This keeps related facts together and prevents loss of co-reference.

---

# 9. Source assertion and temporality

```text
speaker:
  patient | clinician | third_party | unclear

polarity:
  positive | negative | not_applicable | unclear

temporality:
  current | past | planned | future | relative | unclear

normalized_date:
  YYYY-MM-DD | YYYY-MM | YYYY | null

date_precision:
  day | month | year | null

date_text:
  original temporal wording | null

certainty:
  explicit | probable | uncertain
```

Rules:

- exact day only when the source supports a day;
- explicit month may normalize to `YYYY-MM` with `date_precision=month`;
- explicit year may normalize to `YYYY`;
- vague relative wording remains `date_text` with no fabricated exact date.

Example:

```text
"πριν περίπου δύο χρόνια"
→ temporality = relative
→ normalized_date = null
→ date_precision = null
→ date_text = "πριν περίπου δύο χρόνια"
```

---

# 10. Evidence snippet contract

`evidence_snippet` is a short, ephemeral verbatim substring used only to support immediate clinician review.

Rules:

- bounded length, target maximum about 320 characters;
- Core server verifies deterministically that the snippet exists in the supplied transcript after safe whitespace normalization only;
- if not verifiable, do not rewrite the quote; add `EVIDENCE_NOT_VERIFIABLE`;
- evidence snippets are never persisted/logged.

---

# 11. Extraction assessment

```text
confidence_band: high | medium | low
requires_clinician_review: true
uncertainty_reason_optional
```

`requires_clinician_review=true` is imposed by server code for every PR-1 candidate.

Confidence is an extraction aid, not a calibrated clinical probability and never grants write authority.

---

# 12. Generic semantic types

```text
patient_history_fact
objective_result
clinician_interpretation
option_discussed
clinician_recommendation
patient_preference
final_decision
patient_accepted
patient_declined
patient_undecided
followup_task
uncertain_needs_review
```

Hard distinctions:

```text
option_discussed != clinician_recommendation != final_decision
patient_preference != patient_accepted
negative history != normal investigation
objective result != clinician interpretation
```

---

# 13. Deterministic normalization vs model extraction

The model extracts what was stated. Deterministic application/module code computes only approved derivations.

Examples:

```text
model extracts weight + height
→ existing deterministic code may calculate BMI later

model extracts fracture date/mechanism wording
→ do not silently infer fragility classification unless explicitly stated or an approved deterministic rule exists

model extracts lab value + unit
→ do not silently convert unsupported units
```

No LLM reimplementation of deterministic clinical calculations already owned by the application.

---

# 14. Actual Module 01 runtime target namespace

The current persistence seam is:

```text
browser currentCase
→ patient-registry.js
→ POST/PUT encounter payload
→ clinical_encounters.payload_json
```

Therefore PR-1 freezes the mapper against the **actual current persisted browser payload**, versioned as:

```text
osteoporosis_runtime_targets_v1
```

YAML documentation alone is not an acceptable target registry when it differs from runtime.

Initial mapping table:

| Extracted concept | Actual current runtime target | v1 mapping decision |
|---|---|---|
| encounter archetype | `encounter_archetype` | mapped |
| weight | `anthropometrics.weight_kg` | mapped |
| current height | `anthropometrics.current_height_cm` | mapped |
| fracture event | `fracture_history.events[]` | composite mapped |
| fracture site | `fracture_history.events[].site` | mapped |
| fracture month | `fracture_history.events[].month` | mapped when month known |
| low-trauma status | `fracture_history.events[].low_trauma` | mapped |
| fracture on treatment | `fracture_history.events[].occurred_on_treatment` | mapped |
| vertebral level/type | `fracture_history.events[].vertebral_level` | mapped |
| current smoking | `risk_assessment.current_smoking` | mapped |
| alcohol ≥3 units/day | `risk_assessment.high_alcohol_3_units_day` | mapped |
| rheumatoid arthritis | `risk_assessment.rheumatoid_arthritis` | mapped |
| glucocorticoid positive | `risk_context.glucocorticoids` | mapped to true |
| glucocorticoid explicit negative | same boolean | **ambiguous/lossy** because false may also mean unchecked/default |
| GC daily dose | `risk_context.glucocorticoid_prednisolone_mg_day` | mapped if context/unit compatible |
| GC duration | `risk_context.glucocorticoid_duration_months` | mapped if explicit |
| falls last 12 months | `risk_context.falls_last_12_months` | mapped; explicit 0 preserved |
| frailty/immobility positive | `risk_context.frailty_or_immobility` | mapped |
| explicit negative frailty | same boolean | **ambiguous/lossy** |
| FRAX tool | `risk_assessment.tool_name` | mapped |
| FRAX country/model | `risk_assessment.country_or_surrogate_model` | mapped |
| FN BMD used | `risk_assessment.femoral_neck_bmd_used` | mapped |
| original FRAX MOF | `risk_assessment.frax_mof_percent` | mapped only when source explicitly supports original/formal output |
| original FRAX hip | `risk_assessment.frax_hip_percent` | same |
| adjusted FRAX/FRAXplus-like value | — | **unmapped**; never overwrite original FRAX |
| resulting risk category | `risk_assessment.resulting_risk_category` | mapped only as stated clinician interpretation |
| DXA date | `step3.dxa.date` | mapped |
| spine BMD/T | `step3.dxa.spine_bmd`, `.spine_t` | mapped with validation |
| total-hip BMD/T | `step3.dxa.total_hip_bmd`, `.total_hip_t` | mapped |
| femoral-neck BMD/T | `step3.dxa.femoral_neck_bmd`, `.femoral_neck_t` | mapped |
| VFA indication | `step3.vfa.indicated` | only if explicitly stated |
| VFA/imaging action | `step3.vfa.action` | mapped |
| imaging modality | `step3.vfa.modality` | mapped |
| vertebral fracture found | `step3.vfa.vertebral_found` | objective investigation only |
| Ca | `step3.labs.ca` | mapped only with compatible unit |
| phosphate | `step3.labs.phosphate` | same |
| vitamin D | `step3.labs.vitamin_d` | same |
| PTH | `step3.labs.pth` | same |
| CTX | `step3.labs.ctx` | same |
| P1NP | `step3.labs.p1np` | same |
| treatment episode | `step4.treatment_episodes[]` | composite mapped |
| treatment agent | `step4.treatment_episodes[].agent` | mapped |
| episode status | `step4.treatment_episodes[].status` | mapped |
| episode start/end | `.start_date`, `.end_date` | only with supported precision |
| approximate duration | `.duration_years` | mapped only without date invention |
| route | — | **unmapped in current runtime** |
| administration event | `step4.administrations[]` | mapped |
| actual administration date | `step4.administrations[].actual_date` | mapped |
| scheduled/next due | `.scheduled_date`, `.next_due_date` | mapped if explicit |
| option discussed | — | **unmapped in current runtime** |
| clinician recommendation | — | **unmapped in current runtime** |
| final treatment decision | `step4.decision.type` + `.selected_agent` | mapped |
| patient preference documented | `step4.decision.preference_documented` | presence metadata only |
| actual preference content | — | **unmapped** |
| patient accepted/declined/undecided | `step4.decision.patient_accepted` | mapped to yes/no/undecided |
| follow-up task type | `step4.tasks[].type` | mapped |
| exact task due date | `step4.tasks[].due_date` | mapped |
| vague timeframe | `step4.tasks[].timeframe_text` | mapped without exact-date invention |
| task-specific narrative/detail | — | **unmapped** |

If an actual runtime path differs from this table during implementation inspection, stop and update the registry/design rather than guessing.

Unmapped clinically meaningful concepts remain visible candidates in PR-1. PR-1 does not expand Step 4 merely to make extraction easier.

---

# 15. Existing-value conflict behavior

PR-1 never overwrites.

A mapped candidate may be compared with current active-form state only when the target is reliably addressable.

```text
same value → no material conflict
reliably different value → conflict
lossy/ambiguous target or comparison → conflict_status=unknown
```

Conflict resolution belongs to PR-2.

---

# 16. Provider/model/API pattern

The Core exposes a provider-neutral protocol.

Preferred first implementation path, subject to fresh official-provider verification immediately before coding:

```text
OpenAI Responses API
non-streaming
single structured extraction pass
strict JSON Schema Structured Output
store=false
tools=none
truncation disabled
explicit bounded timeout
SDK automatic retries disabled (max_retries=0)
```

Why zero automatic SDK retries is frozen: the current official Python SDK retries several transient/5xx/rate-limit classes twice by default. For a privacy-sensitive transcript slice, duplicate external transmission must not happen implicitly.

The exact provider model name, SDK version/pin, output-token ceiling and reasoning setting are **implementation-time verified configuration**, not permanent product invariants. They must be reviewed against current official documentation and the synthetic extraction eval suite before implementation authority is granted.

Provider output contains only extraction semantics such as:

```text
concept_key
domain
semantic_type
components
source_assertion
evidence snippet
source/extraction clarity
```

Target mappings, candidate IDs, module ownership and authoritative state are deterministic server additions.

---

# 17. Privacy / retention posture

Local application guarantees for PR-1:

```text
raw transcript → no PostgreSQL
raw transcript → no encounter payload
raw transcript → no LabSnapshot
raw transcript → no localStorage
raw transcript → no sessionStorage
raw transcript → no logs
provider response → no server persistence
candidate preview → no server persistence
patient identifier → not sent merely for extraction
```

`store=false` does **not** equal Zero Data Retention.

Before identifiable clinical transcript use is allowed, verify the actual provider/project posture for:

- Zero Data Retention / Modified Abuse Monitoring or another accepted data-control arrangement;
- endpoint/model compatibility;
- regional processing/storage where applicable;
- GDPR processor/subprocessor posture;
- application retention/deletion policy.

Until that gate closes, development, CI, provider eval and production engineering smoke use synthetic/de-identified transcripts only.

---

# 18. Logging / observability

Allowed non-content telemetry:

```text
event name
request_id
module
source_type
language
transcript_character_count
candidate_count
mapped_count
ambiguous_count
unmapped_count
latency_ms
provider/model identifier when useful
sanitized error category
exception class optional
```

Forbidden in server/browser logs:

```text
transcript text
evidence snippet
candidate value
patient identifier
provider prompt
provider output
provider exception body
HTTP provider response body
Pydantic input payload
```

Do not use logging patterns that stringify raw provider exceptions or whole response objects.

---

# 19. Browser ephemeral-state lifecycle

The transcript component must be isolated from the current `currentCase` serialization path.

```text
OPEN PANEL
  textarea empty
  candidates empty

PASTE
  raw transcript exists only in textarea DOM value

SUBMIT
  in-memory request only
  no currentCase mutation
  no local/session storage

SUCCESS
  blank textarea
  release raw transcript reference
  retain only candidates in transient JS/DOM

PREVIEW
  candidate values/evidence snippets remain transient only

CLOSE / DISCARD
  blank textarea
  remove candidate DOM
  clear candidate references

PAGEHIDE / LOGOUT / NAVIGATION
  explicit clear

PAGESHOW / BFCache restore
  defensive clean-state reset
```

On network/provider failure the textarea may remain temporarily for explicit user retry, but closing/navigating clears it. No autosave, drafts or restore-transcript behavior.

The UI must state clearly:

```text
AI-extracted candidate != clinician-confirmed clinical record
```

---

# 20. Candidate preview

Preview is clinically grouped, not raw JSON.

Use semantic badges such as:

```text
FACT / HISTORY
OBJECTIVE RESULT
INTERPRETATION
OPTION DISCUSSED
RECOMMENDATION
PATIENT PREFERENCE
FINAL DECISION CANDIDATE
FOLLOW-UP TASK
UNCERTAIN
```

No green/red audit-performance styling.

---

# 21. Semantic safety invariants

Merge blockers if violated:

1. negation preserved;
2. vague timing never becomes fabricated exact date;
3. speaker/source preserved;
4. patient report does not become clinician-confirmed administration;
5. negative history does not become normal investigation;
6. discussion/recommendation/final decision remain distinct;
7. objective result remains distinct from interpretation;
8. original FRAX and adjusted/contextual values remain distinct;
9. ambiguous/garbled source yields uncertainty, not guessed certainty;
10. unsupported/lossy runtime target remains ambiguous/unmapped rather than being squeezed into a convenient field.

---

# 22. Failure semantics

| Condition | HTTP | Public code | Behavior |
|---|---:|---|---|
| missing/invalid clinical auth | 401 | `UNAUTHORIZED` | no provider call |
| provider feature/config absent | 503 | `PROVIDER_NOT_CONFIGURED` | no provider call |
| malformed JSON | 422 | `INVALID_REQUEST` | sanitized |
| empty transcript | 422 | `EMPTY_TRANSCRIPT` | sanitized |
| body >512 KiB | 413 | `REQUEST_TOO_LARGE` | no provider call |
| transcript >120k chars | 413 | `TRANSCRIPT_TOO_LARGE` | no provider call |
| unsupported module | 422 | `UNSUPPORTED_MODULE` | no provider call |
| upstream timeout/rate limit/5xx | 503 | `PROVIDER_UNAVAILABLE` | no automatic retry |
| provider refusal | 422 | `PROVIDER_REFUSAL` | no free-form fallback |
| malformed structured provider output | 502 | `PROVIDER_INVALID_OUTPUT` | discard malformed output |
| local candidate validation failure | 502 | `PROVIDER_INVALID_OUTPUT` | no free-form salvage |
| deterministic mapper bug | 500 | `INTERNAL_PROCESSING_ERROR` | no candidates returned |
| schema-valid ambiguous/unmapped candidates | 200 | warning codes | normal preview |

A partial result is permitted only when the provider result is structurally valid and individual candidates are uncertain/unmapped. Do not permissively salvage half of malformed provider output.

---

# 23. Test architecture

## 23.1 Deterministic CI — no live provider required

Test at minimum:

- request/body/character size guards;
- unknown-field rejection;
- sanitized errors do not echo sentinel PHI;
- candidate discriminated value validation;
- composite candidate validation;
- date precision and relative-date behavior;
- semantic enum/source assertion validation;
- evidence substring verification;
- server forces `requires_clinician_review=true`;
- provider cannot inject target mappings;
- module registry dispatch;
- actual osteoporosis target mapping;
- mapped/ambiguous/unmapped behavior;
- adjusted risk never overwrites original FRAX;
- unsupported units remain unnormalized;
- explicit-negative checkbox-like runtime states remain lossy/ambiguous where runtime cannot distinguish unchecked/default;
- endpoint auth;
- fake provider called exactly once;
- no patient/encounter/lab database mutation;
- no transcript/candidate PHI in logs.

## 23.2 Provider adapter contract test

With fake SDK/client assert:

```text
structured output requested
store=false
tools disabled
truncation disabled
max_retries=0
bounded timeout/configured model used
```

## 23.3 Synthetic provider eval suite

At least representative Greek scenarios for:

1. positive fracture history;
2. explicit negative history;
3. exact vs vague timing;
4. DXA numeric results;
5. laboratory results;
6. negative history vs negative investigation;
7. several treatment options with only one final decision;
8. patient preference;
9. follow-up with vague/exact timeframe;
10. garbled/uncertain speech;
11. original vs adjusted formal risk;
12. speaker ambiguity + unrelated general clinical text.

Promotion gate for a selected provider/model:

```text
100% schema-valid minimum-suite outputs
0 invented exact dates
0 option/recommendation → final-decision collapses
0 history → investigation collapses
0 original/adjusted FRAX overwrites
0 authoritative writes/claims
0 transcript/provider content in logs
all expected critical facts represented in the minimum suite
```

Live-provider eval is explicit and synthetic-only, not mandatory on every deterministic CI run. Any material model/provider change reruns the eval suite.

---

# 24. Red-line merge blockers

PR-1 must not merge if any of the following occurs:

1. raw transcript or candidates are persisted in DB/localStorage/sessionStorage;
2. transcript/candidate PHI appears in server/browser logs;
3. endpoint mutates patient/encounter/lab state;
4. provider generates/selects application target paths;
5. free-form output bypasses structured validation;
6. validation/error handling can echo the raw transcript;
7. SDK automatic retries remain enabled;
8. provider/model changes without the required eval;
9. vague timing becomes an invented exact date;
10. negative history becomes a negative investigation;
11. option/recommendation becomes final decision without explicit source support;
12. adjusted risk overwrites original formal FRAX;
13. currently unmapped Step-4 concepts are forced into unrelated fields;
14. a lossy checkbox false is treated as reliable explicit negative when runtime cannot distinguish default/unreviewed;
15. osteoporosis field names leak into Core contracts/provider abstraction;
16. identifiable transcript is used before the provider data-control/privacy gate is closed;
17. mapping registry is based on YAML-only fields not present in the actual persisted runtime;
18. PR-1 acquires any Accept/Reject/Edit-to-record or authoritative write behavior.

---

# 25. Exact implementation seams after approval

Preferred package boundary:

```text
clinical_excellence/
  __init__.py

  core/
    __init__.py
    transcript_contracts.py
    transcript_provider.py
    transcript_service.py
    transcript_router.py

    providers/
      __init__.py
      openai_transcript.py

  modules/
    __init__.py
    registry.py

    osteoporosis/
      __init__.py
      transcript_profile.py
      transcript_targets.py
```

Responsibilities:

### `core/transcript_contracts.py`

```text
TranscriptExtractionRequest
TranscriptExtractionResponse
ClinicalCandidate
CandidateComponent
CandidateValue discriminated union
SourceAssertion
ExtractionAssessment
TargetMapping
ExtractionWarning
SafeErrorResponse
```

### `core/transcript_provider.py`

```text
TranscriptExtractionProvider protocol
ProviderExtractionPayload
ProviderCandidatePayload
provider error classes
```

No provider SDK import here.

### `core/transcript_service.py`

```text
extract_transcript_candidates()
validate_provider_payload()
verify_evidence()
enrich_candidate_ids()
apply_module_mapping()
build_response()
```

### `core/transcript_router.py`

```text
build_transcript_router()
parse_transcript_request_safely()
classify_public_failure()
```

### `core/providers/openai_transcript.py`

```text
build_openai_transcript_client()
OpenAITranscriptProvider.extract()
build_structured_output_schema()
classify_openai_error()
```

### `modules/registry.py`

Generic profile registration/lookup only.

### `modules/osteoporosis/transcript_profile.py`

Osteoporosis concept vocabulary, allowed components, extraction instructions and normalization enums.

### `modules/osteoporosis/transcript_targets.py`

```text
OSTEOPOROSIS_RUNTIME_TARGETS_V1
map_osteoporosis_candidate()
map_component()
```

Only here should fracture/risk/step3/step4/FRAX/DXA/drug target logic appear.

Existing files expected to change minimally:

- `clinical_auth.py`: expose/reuse a protected dependency for the transcript router without broad auth refactor;
- `main.py`: compose the transcript router;
- `requirements.txt`: replace the unbounded OpenAI SDK floor with a reviewed implementation-time dependency choice after regression verification;
- `static/baseline-audit/app.js`: load capture asset;
- add `static/baseline-audit/transcript-capture.js`;
- add `static/baseline-audit/transcript-capture.css`.

Do not modify merely for convenience:

```text
app-core.js clinical state schema
step3.js
step4.js
clinical_data.py persistence model
clinical_data_ext.py
existing Step YAML schemas
```

Tests/evals expected:

```text
test_transcript_contracts.py
test_transcript_endpoint.py
test_transcript_privacy.py
test_transcript_provider_openai.py
test_osteoporosis_transcript_mapping.py
test_transcript_ui_contract.py

evals/transcript_v1/cases.json
evals/transcript_v1/run_provider_eval.py
```

---

# 26. Browser/UI conflict with current audit methodology

PR-1 is capture engineering, not clinician coaching.

During pilot/scored baseline:

- transcript extraction use may be recorded as capture-source exposure;
- extraction is not a KPI success;
- no Practice Review critique or red/green performance verdict is generated;
- no automatic treatment recommendation is generated;
- if systematic use materially changes baseline methodology, exposure/methodology must be recorded explicitly.

---

# 27. Definition of Done

PR-1 is complete only when:

- protected transcript-paste UI exists;
- reusable Core contracts implemented;
- provider adapter isolated from module logic;
- strict structured provider output and local validation active;
- composite candidates preserve source semantics;
- actual Module 01 runtime target registry is implemented;
- mapped/ambiguous/unmapped behavior is explicit;
- preview clearly says AI candidate, not clinician truth;
- no authoritative write path exists;
- raw transcript/candidates are not persisted;
- content is not logged;
- browser state clears including BFCache paths;
- deterministic tests pass;
- synthetic provider eval gate passes for the selected provider/model;
- identifiable transcript use remains blocked until privacy/data-control readiness is separately verified;
- operational canonicals record exact test/merge/deploy/smoke truth.

---

# 28. Rollback boundary

PR-1 remains additive. Rollback removes/disables:

```text
transcript UI
Core transcript router/service
provider adapter wiring
osteoporosis transcript profile/target mapper
```

No patient-data migration is required because PR-1 does not persist candidate data as authoritative clinical state.

---

# 29. REPLAN triggers

Stop implementation and return to design if:

- current auth cannot safely protect the endpoint;
- logging/validation middleware cannot prevent sensitive text exposure without broader work;
- provider retention/data-control requirements are incompatible with intended use;
- structured-output reliability is inadequate;
- existing runtime cannot represent useful first-slice mappings without major schema redesign;
- a second competing source of truth would be introduced;
- transcript size requires chunking/multi-pass architecture that materially changes privacy/cost/reconciliation;
- Core/module separation proves artificial or highly duplicative;
- implementation appears to require auto-write to be useful;
- baseline methodology would be materially altered by routine use.

A REPLAN trigger is not permission to patch around the frozen design.

---

# 30. Final pre-code approval gate

This v3 design is the corrected canonical design after the runtime review. It still does **not** authorize implementation.

The next fresh design-review conversation must:

1. bootstrap from fresh `main` and all six canonicals;
2. verify the v3 corrections against actual runtime seams;
3. verify current official provider/API/SDK facts without freezing stale model names;
4. identify any remaining contradiction or REPLAN trigger;
5. present the final compact implementation contract to the product owner.

Only explicit product-owner instruction:

```text
IMPLEMENT
```

may transition PR-1 from design to runtime implementation and allow a runtime writer lock/implementation branch.

---

# 31. Next slice after PR-1

If extraction quality and privacy invariants are demonstrated:

```text
PR-2 — Clinician Review / Accept / Reject / Edit + authoritative merge
```

PR-2 owns conflict resolution, accepted-data merge, persistent provenance and clinician-reviewed state.
