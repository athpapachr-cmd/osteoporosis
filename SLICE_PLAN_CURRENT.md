# SLICE_PLAN_CURRENT.md — PR-1 Transcript Intake + Candidate Extraction v1

> **STATUS:** ACTIVE APPROVED SLICE DESIGN — implementation not started.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** 01 — Osteoporosis.
> **Parent phase:** `CLINICAL_EXCELLENCE_PLAN.md`.
> **Slice ID:** PR-1.
> **Product-owner direction:** enable paste of a Heidi transcript so useful encounter data can be extracted automatically, while preserving clinician review and keeping raw transcript out of persistent clinical storage.

This file defines the one current implementation slice. Operational branch/PR/deploy state belongs in `CURRENT_OPERATIONAL.md`.

---

# 1. Product outcome

After this slice, the clinician can paste a Heidi transcript and ask the system to analyze it into **structured candidate clinical data** aligned to the existing Baseline/longitudinal schema.

The slice ends before authoritative patient-data mutation.

Canonical first increment:

```text
PASTE TRANSCRIPT
→ server-side structured extraction
→ candidate objects
→ review preview
→ NO automatic database write yet
```

This deliberately separates **extraction quality** from **write authorization**.

---

# 2. Why this slice is first

The current system already has structured encounter, DXA/lab/treatment and audit schemas. The main friction is duplicate manual capture after a real consultation.

A generic Heidi summary is insufficient because it can:

- collapse “option discussed” into “final plan”;
- convert negative history into negative investigation;
- lose negation/temporality;
- invent certainty around vague dates;
- merge clinician interpretation with objective result;
- omit clinically important uncertainty.

Therefore the first implementation must produce typed **candidates**, not an authoritative summary.

---

# 3. In scope

## 3.1 UI

Add a clear entry point in the Baseline/clinical workspace, tentatively:

```text
Εισαγωγή από Heidi
```

Expected interaction:

1. open modal/panel;
2. paste transcript;
3. click Analyze/Extract;
4. see structured candidate preview grouped by clinical domain;
5. close/discard safely.

No accepted candidate writes to the encounter in PR-1.

## 3.2 Protected backend extraction endpoint

Candidate route:

```text
POST /clinical/transcript/extract
```

It must use the existing clinical browser-session protection rather than a new public endpoint.

Request concept:

```json
{
  "source": "heidi",
  "transcript": "...",
  "encounter_id": "optional existing encounter reference"
}
```

Response concept:

```json
{
  "schema_version": "transcript_extraction_v1",
  "candidates": [],
  "warnings": [],
  "meta": {
    "source": "heidi",
    "raw_persisted": false
  }
}
```

Exact implementation types may be refined during source inspection without changing the canonical semantics.

## 3.3 Extraction candidate model

Each candidate should preserve at least:

```text
candidate_id
domain
target_path_or_object_type
semantic_type
candidate_value
unit_optional
date_or_time_context_optional
confidence
uncertainty_reason_optional
source_speaker_optional
short_evidence_snippet_ephemeral
conflicts_with_existing_value: yes/no/unknown
status: proposed
```

`semantic_type` must support:

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

## 3.4 Initial clinical domains

Prefer objective/high-yield candidates first:

- anthropometrics;
- fracture history/events;
- smoking/alcohol/glucocorticoid/risk factors;
- falls/frailty/function where explicit;
- DXA/VFA/imaging;
- laboratory values with explicit units/dates when available;
- treatment episodes/administrations;
- treatment decision components;
- follow-up tasks.

The extractor may identify communication/preference text, but PR-1 does not yet score consultation quality.

---

# 4. Explicitly out of scope

PR-1 does **not**:

- write extracted candidates into the authoritative encounter;
- persist raw transcripts;
- store raw transcript in localStorage;
- log transcript text;
- generate audit/KPI verdicts for the user;
- generate Practice Review coaching;
- change the Baseline form/KPI calculation contract;
- redesign visible consultation flow;
- modify Calendar/Setmore/Digital Secretary integration;
- send patient communications;
- make treatment recommendations autonomously.

These boundaries keep the first slice reversible and testable.

---

# 5. Privacy / data-minimization contract

The raw transcript may contain identifiable and sensitive clinical information.

Required invariants:

```text
raw transcript is request-scoped / ephemeral
raw transcript is not written to PostgreSQL
raw transcript is not written to localStorage/sessionStorage
raw transcript is not printed to application logs
raw transcript is not committed to GitHub
synthetic transcripts only in tests
```

The backend/model adapter should return only structured candidates and temporary short evidence snippets needed for immediate clinician review.

PR-2 will decide what provenance is persisted after clinician acceptance; default intent is structured source metadata, not raw transcript quotations.

---

# 6. Semantic extraction invariants

## 6.1 Negation

“Δεν είχε πτώσεις” must not become a positive falls history.

## 6.2 Temporality

“Πριν περίπου έξι μήνες έκανε Prolia” must not become an invented exact administration date.

## 6.3 Speaker/source

Patient-reported medication use must remain distinguishable from a clinician-confirmed administration record.

## 6.4 History vs investigation

“Δεν έχει πρόβλημα με παραθυρεοειδή” must not become “PTH normal” unless a PTH result is explicitly present.

## 6.5 Discussion vs final decision

Mentioning teriparatide, zoledronate and alendronate in one consultation does not mean all three are in the final treatment plan.

## 6.6 Objective result vs interpretation

A DXA T-score/BMD value is separate from the clinician’s interpretation of risk/category.

## 6.7 Original vs adjusted risk outputs

Original FRAX and contextual/FRAXplus-adjusted estimates remain separate candidates with provenance. One must not overwrite the other during extraction.

## 6.8 Uncertainty

When the transcript is garbled, contradictory or ambiguous, emit `uncertain_needs_review` rather than guessing.

---

# 7. Mapping strategy

The extractor should map to the **existing canonical clinical model** rather than create a parallel transcript-specific patient schema.

Examples:

```text
age/height/weight
→ existing encounter core fields

fracture details
→ fracture_history.events[]

DXA data
→ step3.dxa / longitudinal model

labs
→ step3.labs candidate snapshot

treatment
→ step4.treatment_episodes / administrations / decision

follow-up
→ step4.tasks
```

Where the existing schema is too ambiguous, the candidate should remain typed but unmapped and generate a design finding for a later schema revision; PR-1 should not silently invent a new authoritative field.

---

# 8. Existing-value conflict behavior

If an encounter already contains a value and extraction suggests a different value:

```text
DO NOT overwrite
→ mark candidate conflict
→ show existing vs extracted candidate in preview
```

Conflict resolution and merge authorization belong to PR-2.

---

# 9. Model/provider boundary

The model is a structured extraction component, not a clinical-data writer or decision authority.

Required architecture:

```text
browser
→ protected backend endpoint
→ transcript extraction adapter
→ strict structured response validation
→ candidate preview
```

The exact model/provider and schema-validation implementation should reuse available project infrastructure where sensible and must be inspected before coding. A model response that fails structural validation returns a safe extraction failure/partial result; it must not become free-form authoritative state.

No secret/provider credential is exposed to the browser.

---

# 10. UI candidate preview

Candidate groups should be clinically readable, not raw JSON.

Possible structure:

```text
Fractures
  • May 2026 — toe fracture — mechanism unclear
    Confidence: medium
    Needs review: fragility status not established

DXA
  • Hip T-score -2.8
  • BMD 0.533 g/cm²

Treatment discussion
  • Teriparatide — option discussed
  • Binosto — possible/final decision candidate, needs review
```

Important visual distinction:

```text
FACT / RESULT
OPTION DISCUSSED
FINAL DECISION CANDIDATE
UNCERTAIN
```

The preview must make it difficult to mistake “AI extracted” for “clinician confirmed”.

---

# 11. Baseline/audit boundary

PR-1 is a capture-engineering slice.

During pilot/scored baseline:

- extraction may be tested without showing KPI coaching;
- using transcript extraction should be recordable as capture-source exposure later;
- extraction itself does not count as a KPI success;
- no Practice Review intervention is activated by this slice.

---

# 12. Acceptance evidence

Use only synthetic/de-identified fixtures.

Minimum scenario families:

1. explicit positive/negative history extraction;
2. vague vs exact treatment dates;
3. objective DXA/lab values;
4. negative history vs absent lab result;
5. multiple treatment options with one final decision;
6. patient preference affecting plan;
7. follow-up task/timeframe extraction;
8. garbled transcript segment → uncertainty rather than invention;
9. existing structured value conflict → flagged, not overwritten;
10. request/log inspection confirms raw transcript is not persisted/logged.

Prefer a small set of representative synthetic transcripts over phrase-list overfitting.

---

# 13. Definition of Done

PR-1 is complete when:

- protected transcript-paste UI exists;
- structured extraction endpoint returns validated candidate objects;
- initial clinical domains map to existing schema paths where possible;
- semantic categories distinguish facts/results/discussion/recommendation/preference/final decision/tasks;
- no authoritative encounter write occurs;
- raw transcript is not persisted or logged;
- uncertainty/conflicts fail safely;
- synthetic scenario families pass;
- `CURRENT_OPERATIONAL.md` records merge/deploy/smoke truth;
- completed history is appended to `osteoporosis-change-log.md`.

---

# 14. Rollback boundary

PR-1 is additive. Rollback consists of removing/disabling the transcript UI/endpoint/adapter without patient-data migration because no extracted candidate is authoritative or persisted as encounter data in this slice.

---

# 15. REPLAN triggers

Stop and replan before further mutation if source inspection shows any of the following:

- existing logging middleware cannot prevent transcript text from being logged without broader security work;
- current auth boundary cannot safely protect the endpoint;
- model/provider requires a materially different data-retention/privacy contract;
- existing schemas cannot represent the initial candidate domains without major redesign;
- the implementation would need to auto-write candidates to be useful;
- a second competing clinical-data owner would be introduced;
- baseline methodology would be materially altered by the implementation.

---

# 16. Next slice after PR-1

If extraction quality and privacy invariants are demonstrated:

```text
PR-2 — Clinician Review / Accept / Reject / Edit + authoritative merge
```

Only PR-2 may authorize accepted candidate data to become encounter truth.
