# SLICE_PLAN_CURRENT.md — PR-1 Transcript Intake + Candidate Extraction v1

> **STATUS:** ACTIVE PRE-CODE DESIGN v2 — ready for product-owner review; implementation not yet authorized.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** reusable Personal Clinical Excellence System.
> **Proving module:** Module 01 — Osteoporosis.
> **Parent phase:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **Slice ID:** PR-1.
> **Product-owner direction:** enable paste of a Heidi transcript so useful encounter data can be extracted automatically, while preserving clinician control, keeping raw transcript ephemeral, and building the capability as reusable Clinical Excellence Core rather than osteoporosis-only code.

This file defines the exact current slice design. Runtime branch/PR/deploy/smoke state belongs in `CURRENT_OPERATIONAL.md`.

---

# 1. Product outcome

After PR-1, the clinician can paste a Heidi consultation transcript and receive a **structured, non-authoritative candidate preview**.

The first increment ends deliberately before any candidate can mutate the patient record.

```text
PASTE TRANSCRIPT
→ protected Core endpoint
→ model/provider extraction
→ strict local validation
→ reusable candidate envelope
→ Module 01 mapping where possible
→ clinically readable preview
→ discard raw transcript
→ NO automatic clinical write
```

PR-1 is a capture foundation for the wider Clinical Excellence system before, during and after clinic. Osteoporosis supplies the first real domain schema and test cases; it must not define the reusable Core architecture.

---

# 2. Why PR-1 is first

The current system already has useful structured encounter, fracture-risk, DXA/VFA, laboratory, treatment and follow-up schemas. The major friction is duplicate manual entry after a real consultation.

A generic Heidi summary is not sufficient because it can:

- collapse “option discussed” into “final treatment decision”;
- convert negative history into a negative investigation result;
- lose negation, temporality or speaker;
- invent precise dates from vague timing;
- merge objective data with clinician interpretation;
- miss or hide uncertainty;
- produce prose that cannot safely populate longitudinal clinical objects.

Therefore PR-1 produces **typed candidate observations**, not a summary and not chart truth.

---

# 3. Core-vs-Module boundary — permanent design rule

PR-1 must be reusable for later modules and other clinical text sources.

## 3.1 Clinical Excellence Core owns

```text
source intake contract
request-size and failure handling
provider/model abstraction
semantic candidate envelope
speaker / polarity / temporality / uncertainty
source evidence snippet lifecycle
local schema validation
privacy / logging / retention safeguards
candidate transport to UI
mapping interface
module registry interface
```

## 3.2 Module 01 — Osteoporosis owns

```text
osteoporosis concept vocabulary
fracture event mapping
FRAX/risk concept mapping
DXA/VFA mapping
osteoporosis laboratory mapping
treatment episode / administration / decision mapping
osteoporosis follow-up task mapping
existing Step 1–5 / longitudinal target paths
```

## 3.3 Core must not hard-code

The Core must not contain osteoporosis-specific assumptions such as:

```text
T-score fields
FRAX semantics
Prolia / Aclasta rules
VFA indication logic
osteoporosis treatment categories
```

Those belong to the osteoporosis module adapter/profile.

---

# 4. Position in the wider Clinical Excellence lifecycle

The reusable engine should eventually support text captured at different points in care:

```text
PRE-VISIT
referral / prior note / patient questionnaire

DURING VISIT
Heidi or other transcript

POST-VISIT
clinician note / discharge letter / follow-up summary
```

PR-1 implements only:

```text
source_type = heidi_transcript
encounter_phase = during_visit
processing_time = usually post-visit
module = osteoporosis
```

The request/response contracts should nevertheless be generic enough that later source types can be added without replacing the Core candidate model.

---

# 5. UI host and mental model

The current Baseline/Clinical Excellence workspace is the first host because it already owns the active encounter and protected clinical session.

Add an additive entry point such as:

```text
Clinical Capture
└── Εισαγωγή από Heidi
```

Do **not** present it as an “Audit” function. It is a Clinical Excellence capture capability that currently lives in the Module 01 workspace.

Expected PR-1 interaction:

1. open transcript panel/modal;
2. paste transcript;
3. show explicit privacy/experimental notice;
4. choose `Analyze / Extract`;
5. receive candidate groups;
6. automatically clear or explicitly discard raw transcript from the textarea after successful extraction;
7. review preview only;
8. close/discard candidates.

No Accept/Reject/Edit-to-record action exists until PR-2.

---

# 6. Protected endpoint contract

Candidate route:

```text
POST /clinical/transcript/extract
```

It must use the existing protected `/clinical/*` browser-session boundary.

PR-1 request concept:

```json
{
  "source_type": "heidi_transcript",
  "module": "osteoporosis",
  "encounter_phase": "during_visit",
  "language": "el",
  "transcript": "...",
  "context": {
    "encounter_archetype": "optional"
  }
}
```

Rules:

- do not require patient name or patient identifier;
- do not send `patient_id` to the model merely to perform extraction;
- `encounter_archetype` may be used as non-identifying context when available;
- no authoritative encounter payload is required for extraction;
- current-form conflict checking may be performed client-side later from mapped target paths rather than coupling the Core endpoint to patient persistence.

Initial size guard:

```text
non-empty transcript
reasonable consultation-size ceiling
sanitized 413/422-style failure when exceeded
```

The implementation should choose and document a concrete ceiling sufficient for a typical ~60-minute consultation, rather than accepting unbounded request bodies.

---

# 7. Response contract — reusable candidate envelope

Response concept:

```json
{
  "schema_version": "clinical_transcript_candidates_v1",
  "source_type": "heidi_transcript",
  "module": "osteoporosis",
  "candidates": [],
  "warnings": [],
  "meta": {
    "raw_persisted": false,
    "authoritative_write": false
  }
}
```

The response must not include the full transcript.

Provider/model metadata may be returned for engineering traceability but must not become a substitute for clinical provenance and must not expose secrets.

---

# 8. Core candidate contract

Each candidate should be structurally explicit.

Suggested v1 envelope:

```text
candidate_id
module
domain
concept_key
semantic_type
value
unit_optional
target_mapping
source_assertion
extraction_assessment
warnings
status = proposed
```

## 8.1 `target_mapping`

```text
mapping_status: mapped / ambiguous / unmapped
canonical_target_path_optional
mapping_note_optional
```

A mapped path is a **candidate destination**, not permission to write.

## 8.2 `source_assertion`

```text
speaker: patient / clinician / third_party / unclear
polarity: positive / negative / not_applicable / unclear
temporality: current / past / planned / future / relative / unclear
date_exact_optional
date_text_optional
certainty: explicit / probable / uncertain
evidence_snippet_ephemeral
```

If timing is vague, preserve the original relative wording and leave exact date null.

## 8.3 `extraction_assessment`

```text
confidence_band: high / medium / low
requires_clinician_review: true
uncertainty_reason_optional
```

`confidence_band` is an extraction aid, **not calibrated clinical probability** and never authorizes auto-write.

---

# 9. Semantic types — generic across modules

PR-1 must support at least:

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

The distinction between these types is a hard safety/data-quality requirement.

Examples:

```text
“Συζητήσαμε Aclasta”
→ option_discussed
!= final_decision

“Προτιμώ χάπι”
→ patient_preference

“Ξεκινάμε Binosto”
→ final_decision candidate, only if the transcript actually supports a final decision
```

---

# 10. Deterministic normalization vs model extraction

The model should extract what was stated. Existing deterministic code should compute what can be computed safely.

Examples:

```text
model extracts weight + height
→ existing deterministic code calculates BMI

model extracts exact lab value + unit
→ deterministic normalization layer may later convert units if an approved rule exists

model extracts fracture date/mechanism text
→ do not let the model silently decide fragility status unless explicitly stated or the module has a separately reviewed derivation rule
```

Do not use the LLM to reproduce deterministic calculations already owned by the application.

---

# 11. Module 01 mapping profile — first implementation

The osteoporosis adapter should map candidate concepts to the **existing canonical clinical model**, not invent a second patient model.

Initial high-yield target families:

```text
encounter context / reason
anthropometrics
fracture events
smoking / alcohol / glucocorticoid / secondary-risk context
falls / frailty / function when explicit
formal-risk facts actually stated
DXA / VFA / imaging
laboratory results with units/dates when explicit
treatment episodes / administrations
treatment decision components
follow-up tasks
```

Examples:

```text
weight / current height
→ Step 1 anthropometrics

fracture event
→ Step 2 fracture_history.events[]

FRAX original MOF / hip output
→ Step 2 formal_risk_assessment

DXA site / BMD / T-score
→ Step 3 dxa

PTH / Ca / CTX etc.
→ Step 3 laboratory candidate snapshot

treatment option / recommendation / preference / final decision
→ Step 4 current_decision components

follow-up lab / referral / review
→ Step 4 followup_tasks
```

The mapping registry must be built from the actual current schema/runtime paths before coding. If the runtime path is ambiguous or differs from prose documentation:

```text
mark unmapped/ambiguous
→ record design finding
→ do not guess a write target
```

---

# 12. Semantic extraction invariants — merge blockers if violated

## 12.1 Negation

“Δεν είχε πτώσεις” must not become positive falls history.

## 12.2 Temporality

“Πριν περίπου έξι μήνες έκανε Prolia” must not become an invented exact administration date.

## 12.3 Speaker/source

Patient-reported medication exposure remains distinguishable from clinician-confirmed administration.

## 12.4 History vs investigation

“Δεν έχει πρόβλημα με παραθυρεοειδή” must not become “PTH normal” unless a PTH result is explicitly present.

## 12.5 Discussion vs recommendation vs final decision

Mentioning teriparatide, zoledronate and alendronate does not mean all are current plan.

## 12.6 Objective result vs interpretation

DXA BMD/T-score is separate from the clinician’s risk interpretation.

## 12.7 Original vs adjusted risk output

Original FRAX and FRAXplus/contextual-adjusted estimates remain separate facts with provenance.

## 12.8 Uncertainty

Garbled, contradictory or ambiguous text must emit `uncertain_needs_review` or a warning rather than a guessed value.

## 12.9 No silent clinical inference

The extractor does not convert ambiguous wording into a diagnosis, contraindication, fragility classification, treatment acceptance or treatment failure merely because that conclusion seems clinically plausible.

---

# 13. Provider/model boundary

The Core should expose a provider-neutral adapter boundary.

Conceptual architecture:

```text
browser
→ protected clinical transcript router
→ Core extraction service
→ provider adapter
→ provider structured output
→ local Pydantic/schema validation
→ module mapper
→ candidate response
```

Implementation should not wire the UI directly to legacy `openai_client` behavior.

Provider requirements:

- supports Greek clinical text adequately;
- can produce strict structured output or equivalent schema-constrained response;
- request explicitly disables provider-side application storage where supported (`store=false` or equivalent);
- local validation remains mandatory even when provider structured outputs are used;
- provider refusal / malformed response / timeout fails safely;
- model selection is configuration, not clinical business logic.

The existing repository already uses an OpenAI client, but PR-1 should keep provider code isolated so another approved provider or local model could later replace it without changing candidate semantics.

---

# 14. Privacy and retention design

The raw transcript may contain identifiable and highly sensitive clinical information.

Required local invariants:

```text
raw transcript is request-scoped / ephemeral
raw transcript is not written to PostgreSQL
raw transcript is not written to localStorage/sessionStorage
raw transcript is not included in normal application logs
raw transcript is not committed to GitHub
candidate clinical values are not dumped to application logs
synthetic/de-identified transcripts only in automated tests
```

Provider-side retention is a separate issue from local persistence.

Before real identifiable transcript use is approved, explicitly verify:

```text
provider/project data-retention mode
whether Zero Data Retention or equivalent is active/required
endpoint/model compatibility
store=false behavior
applicable GDPR / processor / regional requirements
```

Until that verification is complete:

> PR-1 development, automated tests and production engineering smoke use synthetic/de-identified transcripts only.

Do not claim that `store=false` alone means Zero Data Retention.

---

# 15. Logging / observability contract

Useful engineering telemetry may include only non-content metadata such as:

```text
request_id
module
source_type
transcript_character_count
candidate_count
latency_ms
success / error category
provider/model identifier if appropriate
```

Do not log:

```text
transcript text
evidence snippets
candidate values
patient identifiers
provider error payloads that echo prompt content
```

Sanitize provider exceptions before returning/logging them.

Pydantic/FastAPI validation behavior must be reviewed so an oversized/invalid transcript is not echoed into error logs or error payloads unnecessarily.

---

# 16. Ephemeral browser-state lifecycle

PR-1 must not create hidden browser persistence.

Expected lifecycle:

```text
paste into textarea
→ submit
→ receive candidates
→ clear/discard raw textarea content
→ keep candidate preview only in memory/DOM
→ close page/modal
→ candidates disappear unless PR-2 later creates an explicit accepted-data workflow
```

No transcript or candidate cache in localStorage/sessionStorage during PR-1.

The UI must clearly label:

```text
AI-extracted candidate
!= clinician-confirmed clinical record
```

---

# 17. Existing-value conflict behavior

PR-1 never overwrites existing data.

When a mapped candidate points to a field that already contains a different current value, the preview should eventually be capable of showing:

```text
EXISTING VALUE
vs
EXTRACTED CANDIDATE
```

For PR-1, conflict comparison may be performed client-side using the active form state. If reliable comparison is not available for a target, use:

```text
conflict_status = unknown
```

Conflict resolution/merge authorization belongs to PR-2.

---

# 18. UI candidate preview

Preview should be clinically readable and grouped by domain.

Example:

```text
Fractures
• May 2026 — toe fracture
  mechanism: unclear
  source: patient
  confidence: medium
  target: fracture event candidate

DXA
• Total hip T-score -2.8
• BMD 0.533 g/cm²

Treatment discussion
• Teriparatide — OPTION DISCUSSED
• Oral bisphosphonate — PATIENT PREFERENCE / FINAL DECISION NEEDS REVIEW
```

Use visible semantic badges such as:

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

Do not use green/red “correctness” styling; this is extraction review, not audit coaching.

---

# 19. Baseline/audit boundary

PR-1 is capture infrastructure, not a coaching intervention.

During pilot/scored baseline:

- extraction may be tested for capture engineering;
- Heidi/transcript use remains a capture-source exposure, not a KPI success;
- no Practice Review critique is shown by PR-1;
- no live red/green KPI verdict is shown;
- no treatment recommendation is generated.

If transcript capture materially changes the baseline methodology, record the exposure rather than silently treating the cohort as untouched.

---

# 20. Error and failure contract

Expected safe failure categories:

```text
401/403 — clinical session/auth failure
413/422 — empty/oversized/invalid request
503 — provider not configured/unavailable
502/422 — provider response invalid/refused/unparseable
200 + warnings — partial but valid candidate set
```

Failure responses must not echo the full transcript.

If strict structured output fails validation:

```text
no free-form fallback becomes clinical data
→ return safe extraction failure/partial warning
```

PR-1 should prefer one well-specified structured extraction pass over repeated unconstrained repair calls. Any retry policy must be bounded and must not create uncontrolled duplicate external exposure.

---

# 21. Test architecture

Testing must separate deterministic application correctness from model quality.

## 21.1 Deterministic/unit tests — no external API required

Test:

- request validation and size guards;
- candidate Pydantic/schema validation;
- semantic enum validation;
- module registry dispatch;
- osteoporosis target mapping;
- unknown/ambiguous mapping behavior;
- conflict flagging helper where applicable;
- sanitized error handling;
- no persistence path called;
- no transcript content emitted to application logs.

Use a fake provider adapter returning synthetic structured objects.

## 21.2 Provider integration test — synthetic only

Use a small synthetic/de-identified Greek transcript and verify:

- structured response parses;
- Greek clinical text is extracted plausibly;
- no full transcript appears in response;
- provider failure is sanitized.

Do not make live-provider calls part of every deterministic CI run unless explicitly justified.

## 21.3 Clinical extraction scenario set

Minimum scenario families:

1. positive and negative history;
2. vague vs exact dates;
3. objective DXA and laboratory values;
4. negative history vs absent investigation;
5. several treatment options with only one final decision;
6. explicit patient preference affecting the plan;
7. follow-up task and timeframe;
8. garbled transcript → uncertainty;
9. original vs adjusted fracture-risk outputs;
10. speaker ambiguity;
11. existing-value conflict → flagged/not overwritten;
12. unrelated/general clinical text must not be forced into osteoporosis-specific concepts.

Prefer representative scenarios over phrase-list overfitting.

---

# 22. Red-line merge blockers

PR-1 must not merge if any of these are demonstrated:

```text
raw transcript persisted in DB/browser storage
raw transcript or candidate PHI printed to logs
authoritative encounter write occurs
provider free-form output bypasses validation
exact date invented from vague timing
negative history converted to normal investigation
option discussed collapsed into final plan
original and adjusted risk outputs overwrite each other
ambiguous text converted to high-certainty clinical fact
Core implementation hard-codes osteoporosis semantics that prevent module reuse
real identifiable transcript used for acceptance before retention/privacy readiness is verified
```

---

# 23. Definition of Done

PR-1 is complete when:

- protected transcript-paste UI exists in the current clinical workspace;
- reusable Core request/response/candidate contracts are implemented;
- provider adapter is isolated from module-specific logic;
- strict structured response + local validation are active;
- Module 01 mapping profile maps initial high-value concepts to existing schema targets where safe;
- uncertain/unmapped concepts remain explicit rather than guessed;
- candidate preview clearly distinguishes AI extraction from clinician truth;
- no authoritative encounter write occurs;
- raw transcript is not persisted locally or server-side;
- content is not logged;
- synthetic deterministic and provider/scenario tests pass;
- privacy limitation for real identifiable transcripts remains explicit until verified;
- `CURRENT_OPERATIONAL.md` records exact merge/deploy/smoke state;
- changelog/TODO are updated at appropriate completion gates.

---

# 24. Rollback boundary

PR-1 is additive.

Rollback consists of disabling/removing:

```text
transcript UI
transcript Core router/service
provider adapter wiring
osteoporosis transcript mapping profile
```

No patient-data migration is required because PR-1 does not persist extracted candidates as authoritative encounter data.

---

# 25. REPLAN triggers

Stop mutation and return to design if source inspection or implementation discovers:

- current auth boundary cannot safely protect the endpoint;
- logging/validation middleware cannot prevent sensitive text exposure without broader changes;
- provider retention/privacy requirements are materially incompatible with intended real use;
- provider lacks reliable structured-output support for the chosen path;
- existing schemas cannot represent initial candidates without a major data-model redesign;
- module mapping requires a second competing source of truth;
- auto-writing candidates is required to make the feature usable;
- transcript size requires chunking/multi-pass architecture that materially changes privacy, cost or candidate reconciliation;
- Core/module separation proves artificial or introduces substantial duplication;
- baseline methodology would be materially altered by routine use.

A REPLAN trigger is not permission to patch around the frozen design.

---

# 26. Pre-code design approval gate

Before any PR-1 runtime branch is created, the implementation conversation must present a final compact design review covering:

```text
A. Core-vs-Module ownership
B. exact request/response schema
C. candidate semantic model
D. actual osteoporosis target-path mapping table
E. provider/model/API call pattern
F. retention/privacy posture
G. logging/observability behavior
H. browser ephemeral-state lifecycle
I. deterministic + provider/eval tests
J. failure semantics
K. red-line merge blockers
L. exact files/functions to add or modify
```

The product owner must explicitly approve **IMPLEMENT** after this review.

Until then:

```text
DESIGN ≠ IMPLEMENTATION AUTHORITY
```

---

# 27. Next slice after PR-1

If extraction quality and privacy invariants are demonstrated:

```text
PR-2 — Clinician Review / Accept / Reject / Edit + authoritative merge
```

PR-2 will own:

- per-candidate Accept/Reject/Edit;
- conflict resolution;
- accepted-data merge;
- persistent provenance;
- clinician-reviewed state.

Only PR-2 may authorize accepted candidate values to become clinical record truth.
