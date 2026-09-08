# CLINICAL_LEARNING_LOOP_DESIGN_V1.md

## Status

L-1B candidate design for the first closed Challenge learning loop. This design is additive to the frozen L-0/L-1 Challenge/Foundation contracts.

## Product thesis

The Learning Hub must not be a JSON archive. It should retain the clinician's reasoning trajectory, identify what was strong versus wrong versus merely underdeveloped, prescribe focused next learning, repeatedly test retention, and deliberately connect knowledge that is currently used in isolated islands.

The clinician-facing loop is:

```text
CHALLENGE
→ REASONING TRAJECTORY
→ DEBRIEF
→ TARGETED STUDY
→ REPEATED RETRIEVAL
→ DISCRIMINATION
→ TRANSFER
→ BRIDGE TRANSFER
→ RETENTION EVIDENCE
→ FUTURE CHALLENGE
```

## 1. Episode anatomy

Every imported Challenge episode should preserve these distinct layers:

1. **Initial scenario** — the hypothetical case and information available at time zero.
2. **Clinician response** — the clinician's own reasoning, not an AI rewrite standing in for it.
3. **Progressive disclosure** — consequential new information with ordering/provenance.
4. **Follow-up reasoning** — how the clinician updated or failed to update the model.
5. **Final decision** — where the clinician ultimately landed, including uncertainty.
6. **Strengths** — demonstrated good reasoning/knowledge/clinical calibration.
7. **Needs reinforcement** — incomplete, weak or poorly structured areas that are not necessarily wrong.
8. **Clear errors** — materially incorrect statements/decisions with correction and significance.
9. **Defensible disagreements** — choices with more than one reasonable view.
10. **Evidence gaps/blind spots** — what is uncertain, indirect or unfamiliar.
11. **Clinical insights/reasoning patterns** — useful higher-order patterns worth reinforcing.
12. **Learning actions** — what to read, retrieve, practise, discuss or transfer next.
13. **Fresh resources** — current online opportunities tied to actual gaps.
14. **Repeated consolidation plan** — multiple later tests, not a single quiz.
15. **Bridge targets** — concepts that should be applied jointly rather than remaining islands.

The immutable Challenge remains the historical episode. Fresh resources and later consolidation evidence are mutable/append-only overlays linked to it.

## 2. Why the current external export fails

The observed source conversation produced a content-rich export but not the frozen machine contract. This is expected to be solved by an adapter, not by relaxing canonical validation.

The adapter recognizes a bounded `rich_challenge_export_v1` shape whose fields include examples such as:

```text
schema_type / schema_version
session
topic_tags
deidentification
initial_case.prompt + facts
fact_ledger with source-local IDs
progressive_disclosures
clinician_reasoning_responses
mentor_observations
decision_framework
evidence
gap_classes
learning_actions
spaced_repetition
clinician_review
```

The raw shape is never itself accepted as an immutable Challenge.

## 3. Adapter rules

### Stable identifiers

Source-local IDs (`f01`, `la01`, etc.) are converted with UUIDv5 under a fixed Learning Ingress namespace derived from `source_event_id + source-local-id`. The same external episode therefore normalizes idempotently.

### Foundation aliases

The adapter owns a conservative alias table mapping only known labels/slugs to the frozen registry. Unknown aliases produce warnings or fail closed when they would otherwise create unresolved structural references.

Example aliases:

```text
bone-remodeling-physiology → ost.foundation.bone_remodeling
secondary-osteoporosis → ost.foundation.secondary_osteoporosis
antiresorptive-pharmacology → ost.foundation.antiresorptive_pharmacology
anabolic-pharmacology → ost.foundation.anabolic_pharmacology
safety-contraindications-transitions → ost.foundation.treatment_safety_transitions
evidence-appraisal-directness-guideline-disagreement → ost.foundation.evidence_appraisal
```

### Facts

All substantive scenario/disclosure facts are represented in the Fact Ledger. Imported facts remain `authoritative_for_patient=false`.

### Reasoning

The source conversation should ideally send the clinician's verbatim response text. If a legacy export contains only structured summaries, the adapter may preserve those summaries but emits an explicit `source_reasoning_was_summary_not_verbatim` warning. It must not pretend a summary is verbatim.

### Debrief

Grouped source observations are flattened:

```text
strengths → strength
clear_errors → clear_error
defensible_disagreements → defensible_disagreement
evidence_gaps → evidence_gap
blind_spots → blind_spot
reasoning_patterns → reasoning_pattern
clinical_insight → clinical_insight
```

Imported dispositions always become `pending`.

### Evidence

Evidence/references become canonical `LearningReferenceV1` snapshots with `verification_state=unverified`. Fresh resource recommendations are not treated as evidence verification.

## 4. Pending Inbox

External automation writes to **Pending Imports**, never directly to immutable Challenge history.

Inbox card shows:

```text
source/title/date
mode/topics/Foundation nodes
adapter warnings
case + reasoning completeness
strength/error/reinforcement counts
learning actions
fresh-resource count
consolidation plan count
```

Clinician actions:

```text
Open review
Reject
```

Opening a pending candidate uses the existing Challenge preview/disposition workflow. Final accepted persistence still requires explicit clinician confirmation.

## 5. Knowledge islands and bridges

### Knowledge island

A knowledge island is not merely a low Foundation state. It is a concept the clinician can use/explain in one context without evidence that it is integrated with another concept that must often be reasoned about jointly.

Examples in osteoporosis can include:

```text
bone turnover markers understood alone
+
denosumab rebound understood alone
but weak joint use for transition timing

renal/mineral physiology understood alone
+
antiresorptive pharmacology understood alone
but weak joint drug-selection/safety reasoning in CKD-MBD

evidence hierarchy understood alone
+
treatment sequencing understood alone
but weak directness calibration when evidence is subgroup/extrapolated
```

### Bridge target

A bridge target has 2–3 Foundation nodes, source observations/actions and a rationale. It stays `planned` until a later test requires joint application and reviewed evidence supports `demonstrated`.

The Foundation Map's graph provides structural context, but graph adjacency is not evidence that the clinician has integrated the concepts.

### Bridge test

A bridge test must use a **new case** or contrastive task. Repeating the exact original scenario is insufficient because recognition can masquerade as integration.

## 6. Repeated consolidation

The system guarantees repetition instead of treating one successful test as permanent learning.

Initial transparent schedule after accepted Challenge:

```text
D+3   retrieval
D+7   discrimination / contrast
D+14  novel transfer
D+30  bridge-transfer / retention
```

These intervals are a transparent product default, not an assertion that one spacing schedule is uniquely optimal. Health-professions education evidence supports distributed/retrieval practice broadly; the optimal interval depends on content and retention horizon.

Every occurrence remains visible and addressable. Completing D+3 does not erase D+7/D+14/D+30.

Results:

```text
retained
partially_retained
not_retained
improved_beyond_original
not_assessed
```

The first runtime records these results without an opaque adaptive algorithm. A future adaptive scheduler requires separate design/review authority.

## 7. Consolidation test structure

Each occurrence carries:

- one prompt;
- target objectives;
- target Foundation nodes;
- optional target bridge(s);
- expected points/rubric;
- clinician response;
- reviewed result/evaluator note.

Preferred progression:

### Retrieval
Explain the principle unaided.

### Discrimination
Distinguish two close clinical situations where the correct reasoning differs.

### Transfer
Apply the principle to a new case with altered surface details.

### Bridge transfer
Apply two or more previously separate concepts jointly and explain why one changes the use of the other.

## 8. Learning resource recommendations

Fresh resources are selected because of a specific observed target, not because they happen to mention osteoporosis.

Desired ranking inputs:

```text
specific learning objective
Foundation node(s)
gap class
clinical importance
source authority/directness
freshness
accessibility
format fit (article/webinar/course/etc.)
```

Every recommendation must answer:

- What is it?
- Why this clinician needs it now?
- Which gap/bridge does it target?
- What should be learned from it?
- Who provides it?
- When was it last checked?
- Is it free/paid/registration-required when known?

Recommendations remain a mutable overlay because availability and web content change.

## 9. Automatic ChatGPT → Cockpit seam

The server exposes one narrow synthetic-learning ingress endpoint. Its sole capability is:

```text
validated structured synthetic learning episode
→ pending import candidate
```

It cannot:

- read/write patient records;
- create accepted Challenges;
- verify references;
- mutate Foundation state;
- promote Signals;
- mutate RF/physio/CU-1.

Authentication is a dedicated `X-Learning-Ingest-Key`, never the broad `CLINICAL_DATA_KEY`.

The code may ship with the endpoint fail-closed. Production key configuration and connection of a ChatGPT custom integration/plugin are separate lifecycle actions after code review and merge.

## 10. Privacy

External source payloads are untrusted. Before pending persistence:

1. reject forbidden structured patient-identifier keys;
2. scan persistable strings with the existing deterministic PHI guard;
3. normalize only known fields;
4. discard the raw source payload;
5. persist normalized learning objects only.

Initial automatic external ingress is synthetic-only. Real-case learning remains owned by future PR-1/PR-2/Daily Case Review boundaries.

## 11. UX

Default Learning Hub navigation becomes:

```text
Inbox
Challenges
Learning Loop
Foundation Map
Due
Advanced
```

`Advanced` owns manual JSON import/export. A clinician should not need to know field names such as `record_review_state` or manufacture UUIDs.

A Learning Loop card should read clinically, e.g.:

```text
CKD-MBD + fragility fracture
Strengths        4
Reinforce        3
Clear errors     1
Bridge targets   2
Study actions    3
Fresh resources  2
Next test        Wed · retrieval
Later repeats    3
```

## 12. Evidence basis for the learning mechanics

This design is intentionally modest about what the literature establishes. Systematic reviews in health-professions education support distributed practice/retrieval practice and spaced digital education for learning/retention, but interventions and timing are heterogeneous. Therefore the product guarantees repeated retrieval and transfer while keeping the first spacing schedule transparent and configurable rather than claiming a hidden optimal algorithm.

## 13. Exit criteria

L-1B is design/runtime complete only when:

- the observed rich challenge-export structure is adapted successfully into the frozen Challenge contract using a synthetic structural test fixture;
- Inbox eliminates machine-JSON knowledge from the default clinician workflow;
- accepted Challenge detail shows the full reasoning trajectory/debrief;
- strengths/reinforcement/errors remain distinct;
- bridge targets exist and require later joint transfer evidence;
- at least four explicit consolidation occurrences can be materialized;
- attempt history is append-only;
- fresh resources are mutable and hash-external;
- external ingress is synthetic-only and fail-closed without a dedicated key;
- inherited L-1 safety/privacy/concurrency tests remain green;
- an independent exact-head review passes before merge authority is considered.
