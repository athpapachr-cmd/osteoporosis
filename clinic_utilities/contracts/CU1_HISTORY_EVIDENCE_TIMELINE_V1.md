# CU-1 History + Evidence + Rehabilitation Sequence Contract v1.1 — DESIGN HARDENING

> **STATUS:** DESIGN-FROZEN OBJECT SEMANTICS / ROUTE COVERAGE INCOMPLETE.
> **Slice:** CU-1 clinician-quality completion.
> **Runtime evidence-aware generation:** NOT AUTHORIZED.
> **Clinical taxonomy:** existing frozen CU-1 routes remain unchanged unless a specific evidence conflict requires a narrow reviewed correction.
> **Normative machine schema:** `clinic_utilities/contracts/cu1_history_timeline_schema_v1.yaml`.
> **Evidence work queue:** `clinic_utilities/contracts/cu1_evidence_coverage_matrix_v1.yaml`.

---

# 1. Why this contract exists

Clinician review identified three structural deficits that cannot be solved by formatter polish:

```text
1. generated referrals lack a coherent HISTORY section
2. goals are flat and do not express a safe criteria-based rehabilitation progression
3. rehabilitation suggestions are disconnected from explicit route/subtype-specific literature provenance
```

The target is a condition-specific, subtype-aware, evidence-linked rehabilitation prescription framework that produces natural Greek referral prose while preserving clinician judgment and physiotherapist autonomy.

---

# 2. Permanent semantic invariants

```text
history_fact != diagnosis
history_not_recorded != negative_history
patient_statement != objective_finding
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective_deficit != subjective_symptom
provocation_test != diagnosis
imaging_finding != automatically_symptomatic_diagnosis
not_assessed != normal
adjunct != core_rehabilitation
clinician_entered_diagnosis_may_be_carried_but_not_inferred
rehabilitation_phase_order != calendar_duration_prescription
progression_criterion_met != elapsed_time_only
clinician_selected_intervention != evidence_recommended_intervention
one_evidence_source != universal_guideline_consensus
low_certainty_evidence != strong_recommendation
absence_of_route_specific_evidence != permission_to_invent_generic_care
route_A_evidence != route_B_evidence
```

No generated statement may be labelled evidence-based unless its exact rendered clinical element resolves to current applicable evidence for the selected route/subtype, except that explicit patient-specific written postoperative/fracture restrictions may override route evidence and must be labelled as protocol authority rather than literature authority.

---

# 3. ReferralHistoryV2 — coherent history with provenance

`ReferralDraft` gains a structured `history` object. No patient identifiers are added.

It can carry only when explicitly supplied:

```text
onset date or approximate duration
onset pattern
mechanism / trigger
symptom course
prior episodes
prior treatment + response
relevant investigations
aggravating factors
easing factors
work / sport / activity context
patient-priority activity
route-specific history items
```

Every non-empty history value carries provenance through `HistoryProvenanceEntryV1` or the route-history item itself.

Allowed history sources:

```text
patient_reported
clinician_entered_history_summary
documented_record
```

History provenance cannot be used to encode an objective examination finding. Negative history requires an explicit negative statement; omission is not a negative.

Hard distinctions:

```text
approximate duration != inferred exact onset date
mechanism != causal tissue diagnosis
patient report != objective finding
missing != negative
route-specific prompt != auto-selected answer
```

The Detailed formatter generates a natural `ΙΣΤΟΡΙΚΟ` section, never database serialization.

---

# 4. RouteHistoryPromptV1 — dynamic history without diagnosis inference

Route/subtype-specific history questions are first-class design objects.

Each prompt defines:

```text
prompt_id
applicable_route_ids[]
applicable_subtype_ids[] when material
Greek label
response type / enum / unit where relevant
conditional applicability
optional evidence-claim linkage
```

Rules:

```text
show only for applicable route/subtype
never preselect
never turn a response into a diagnosis automatically
preserve patient-report / record / clinician-summary source
```

Examples of materially different prompts include sitting intolerance for a deep-gluteal presentation, high-demand work/sport load for lateral elbow tendinopathy, and midportion-vs-insertional symptom location/load context for Achilles tendinopathy.

---

# 5. RehabilitationSequenceV1 — criteria-based, not calendar-based

The prior generic timeline concept is superseded.

```text
RehabilitationSequenceV1
  sequence_id
  route_id
  subtype scope when required
  route_evidence_profile_id
  phases[]
  clinician_modifications[]
  protocol_overrides[]
```

The intended meaning is:

```text
phase objective
→ evidence/protocol-supported intervention directions
→ criteria for progression
→ precautions / do-not-progress criteria
→ next objective when criteria are met
```

There is no universal:

```text
analgesia → ROM → strengthening
```

or any other common MSK sequence.

Each route/subtype gets only the phases supported by its own evidence and clinical applicability.

---

# 6. Element-level provenance is mandatory

A phase-level list of citations is insufficient. The design must be able to prove authority for every substantive rendered element.

`RehabilitationPhaseV1` therefore contains typed:

```text
clinical objective + authority
InterventionDirectionV1[]
RehabilitationCriterionV1[] progression
RehabilitationCriterionV1[] precautions / do-not-progress
separate strength and certainty fields when applicable
```

Every evidence-derived rendered:

```text
phase objective
intervention direction
progression criterion
precaution / do-not-progress criterion
reassessment / escalation criterion
```

must resolve to at least one active route/subtype/applicability-compatible `EvidenceClaim`.

A broad `phase.evidence_claim_ids[]` bucket is not enough because it cannot prove which source authorizes which intervention or threshold.

---

# 7. AuthorityReferenceV1 — evidence, protocol, or clinician instruction

Every rendered rehabilitation element has one explicit authority class:

```text
evidence_claim
patient_specific_protocol
clinician_instruction
```

### Evidence authority

Requires one or more active applicable `EvidenceClaim` IDs.

### Patient-specific protocol authority

Used for explicit postoperative, fracture-healing or orthopaedic written restrictions. It may override conflicting route defaults but must never be presented as a literature recommendation.

### Clinician instruction

Allows the referring clinician to accept, modify, omit or add an instruction. It remains visibly distinct from guideline authority.

This preserves:

```text
evidence != automatic treatment selection
clinician preference != guideline recommendation
physiotherapist execution autonomy != absence of referral direction
```

---

# 8. ProtocolConstraintV1 — machine-readable override precedence

Text saying “protocol overrides evidence” is not enough. The machine contract explicitly represents patient-specific constraints:

```text
protocol_constraint_id
protocol_type
source description
constraint statement
affected phase/element IDs
effective dates when explicitly supplied
```

Precedence:

```text
explicit patient-specific postoperative/fracture/orthopaedic restriction
>
route/subtype default rehabilitation element
```

Conflicting default text is suppressed rather than blended.

No protocol restriction is inferred from a generic procedure label or from elapsed time alone.

---

# 9. GoalPlanV2

Goals are attached to phases rather than maintained as an unrelated flat checklist.

A goal identifies its source:

```text
route_evidence
clinician_selected
patient_priority
```

Only an evidence-sourced goal is presented as literature-supported. A patient-priority or clinician-selected goal is clinically valid but is not silently relabelled as evidence authority.

---

# 10. ReassessmentPlanV2

Reassessment is primarily criteria-triggered:

```text
phase progression checks
failure-to-progress criteria
safety escalation criteria
explicit clinician medical-review request when used
```

The referring physician is not required to prescribe routine PT frequency or total duration.

Calendar timing appears only when an authoritative postoperative/healing protocol uses time, or when the clinician explicitly requests a medical review date/window.

---

# 11. EvidenceSourceV1 — source identity + freshness

An `EvidenceSource` is bibliographic/source identity, not a recommendation.

Required machine-readable fields include:

```text
evidence_id
source_type
title
authors_or_organization
year_or_version
reference
reviewed_on
next_review_due
freshness_state
status
```

and, where applicable:

```text
published_on
DOI / stable URL
framework
population scope
setting scope
subtype scope
supersedes[]
superseded_by[]
```

Freshness states:

```text
current
review_due
stale
superseded
```

Source status and freshness are separate concepts.

---

# 12. EvidenceClaimV1 — exact clinical authority

A source becomes usable only through explicit claims.

Each `EvidenceClaim` includes:

```text
claim_id
evidence_ids[]
applicable_route_evidence_profile_ids[]
applicable_route_ids[]
applicable_subtype_ids[] when material
applicability conditions[]
domain
claim summary
recommendation direction
output scope
strength / certainty when available
conflicts_with_claim_ids[]
reviewed_on
```

Output scopes:

```text
referral_core
therapist_execution_detail
clinician_ui_only
```

Meaning:

- `referral_core`: disease-specific strategy, staged objectives, materially useful progression criteria and precautions suitable for clinician-to-physiotherapist communication.
- `therapist_execution_detail`: implementation details such as exercise dosing/frequency that should not automatically be dictated in the referral.
- `clinician_ui_only`: uncertainty, conflicts, evidence context or diagnostic boundaries useful to the clinician but not routine referral prose.

---

# 13. RouteEvidenceProfileV1 — unique route/subtype authority

`profile_id = lumbar` or `profile_id = elbow` is too broad for evidence authority. The hardened design uses:

```text
route_evidence_profile_id = unique route/subtype evidence identity
region_profile_id = frozen regional profile identity
route_id
subtype scope
applicability conditions
claim groups
rehabilitation_sequence_id
evidence gaps
primary sources
freshness
```

Examples:

```text
rep_lateral_elbow_tendinopathy_v1
rep_achilles_midportion_v1
rep_achilles_insertional_v1
```

This prevents a claim attached to a regional container from silently leaking across unrelated routes.

---

# 14. Disease/subtype specificity

The 2024 JOSPT/APTA Achilles CPG is explicitly for **midportion** Achilles tendinopathy. It does not authorize insertional wording.

Hard regression rule:

```text
midportion-only source
!= insertional authority
```

Insertional Achilles receives its own evidence profile. Route-specific trials/reviews may be used when a current subtype-specific CPG does not exist, but their population and certainty limits must be preserved.

Likewise:

```text
lateral elbow tendinopathy
!= Achilles tendinopathy
```

Even if both include loading, they require independent disease-specific claims and progression semantics.

---

# 15. Evidence strength / conflicts

Rendering preserves evidence strength:

```text
strong/core recommendation
→ direct recommendation wording

conditional / low-certainty
→ cautious wording such as «μπορεί να εξεταστεί» / «επικουρικά»

conflicting evidence
→ conflict remains visible; no silent hybrid

insufficient route-specific evidence
→ explicit evidence gap; no invented recommendation or threshold
```

A trial in a narrow population is not generalized into a universal guideline recommendation.

---

# 16. Referral output contract

Detailed referral order:

```text
ΔΙΑΓΝΩΣΗ / ΚΛΙΝΙΚΗ ΕΝΤΥΠΩΣΗ
ΙΣΤΟΡΙΚΟ
ΚΛΙΝΙΚΑ ΕΥΡΗΜΑΤΑ
ΛΕΙΤΟΥΡΓΙΚΗ ΕΠΙΒΑΡΥΝΣΗ
ΑΙΤΗΜΑ
ΣΤΑΔΙΑΚΟΙ ΣΤΟΧΟΙ ΚΑΙ ΚΡΙΤΗΡΙΑ ΠΡΟΟΔΟΥ
ΠΡΟΤΕΙΝΟΜΕΝΟΣ ΠΡΟΣΑΝΑΤΟΛΙΣΜΟΣ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΠΡΟΫΠΟΘΕΣΕΙΣ ΕΠΑΝΕΚΤΙΜΗΣΗΣ / ΚΛΙΜΑΚΩΣΗΣ
ΒΙΒΛΙΟΓΡΑΦΙΚΗ ΒΑΣΗ
```

Short output uses the same route/subtype evidence authority, a compressed disease-specific sequence and a compact disease-specific source footer. Short must not become generic.

Default bibliography shows 1–3 highest-authority applicable sources.

---

# 17. Evidence freshness lifecycle

```text
new guideline / review / material trial detected
→ classify impact:
   confirming
   no_change
   potentially_practice_changing
   practice_changing
   conflicting
→ clinician/reviewer approval
→ update EvidenceClaim
→ update affected RehabilitationSequence only if warranted
→ regression fixtures
→ version/changelog
```

No silent autonomous evidence update.

---

# 18. Deep-gluteal evidence boundary

Current DGS literature supports a diagnostic construct and history/examination pathway, but comparative conservative-treatment evidence remains low quality and does not establish a superior specific conservative protocol.

Therefore:

```text
broad active physiotherapy / self-management may be carried cautiously where applicable
specific invented phase thresholds are forbidden
explicit evidence gap is preferable to false precision
```

The route must never infer DGS or piriformis syndrome from buttock pain alone.

---

# 19. Coverage gate

Before runtime evidence-aware generation:

```text
EVERY routine route
→ own unique RouteEvidenceProfile

EVERY clinically material subtype
→ separate applicability/evidence handling

EVERY nonblocked routine route/subtype
→ complete RehabilitationSequenceV1

EVERY evidence-derived rendered element
→ >=1 active applicable EvidenceClaim

patient-specific protocol override
→ explicit ProtocolConstraint authority
→ not mislabeled as literature

evidence gaps
→ explicit

conflicts
→ explicit

freshness
→ current or explicitly reviewed
```

Generic fallback is forbidden.

---

# 20. Required regression fixtures

At minimum:

```text
1. chronic deep-gluteal pain with 8-month history + uncertain diagnosis
2. lateral epicondylalgia with elbow-specific sequence
3. midportion Achilles with midportion-specific sequence
4. insertional Achilles evidence-boundary / low-compression evidence case
5. same generic loading label cannot authorize identical elbow and Achilles semantics
6. postoperative case where written protocol overrides route evidence
7. fracture case where healing/loading restriction overrides route evidence
8. conflicting adjunct guidance
9. weak-evidence route with broad/noninvented output
10. missing history without generated reassuring negatives
11. stale/superseded evidence case
12. therapist-execution-detail claim that must not appear automatically in referral core
13. clinician-modified evidence proposal that remains visibly clinician-authored
```

---

# 21. Stop rule

The object semantics in this document and the normative schema are frozen for the current design-hardening pass. Route-by-route evidence curation, dynamic history-prompt population, complete rehabilitation sequences and fixtures remain required.

```text
complete route-specific evidence coverage
→ exact synthetic fixtures
→ exact design-completeness review
→ DESIGN-COMPLETE or BLOCK
→ only then consider separate runtime authorization
```

No runtime evidence-recommendation engine is authorized by this document.
