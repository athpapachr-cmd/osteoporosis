# CU-1 History + Evidence + Timeline Contract v1 — DESIGN CANDIDATE

> **STATUS:** PRE-RUNTIME DESIGN CANDIDATE — requires exact review before implementation.
> **Slice:** CU-1 clinician-quality completion.
> **Purpose:** add clinically useful history, evidence provenance and goal/reassessment timing without inventing facts, overstating evidence or forcing generic treatment protocols.
> **Clinical taxonomy:** existing frozen CU-1 routes remain unchanged unless a specific evidence conflict requires a separately reviewed correction.

---

# 1. Why this contract exists

Product-owner review identified three structural deficits that are not solved by language polish or progressive disclosure:

```text
1. generated referrals lack a coherent HISTORY section
2. goals have no explicit time horizon / reassessment plan
3. rehabilitation suggestions are not connected at runtime to explicit literature provenance
```

The target is a referral that reads like a clinician wrote it while preserving the existing CU-1 safety invariants.

---

# 2. Core invariants

```text
history_fact != diagnosis
history_not_recorded != negative_history
selected_goal != guaranteed_outcome
reassessment_window != promised_goal_achievement
clinician_selected_intervention != evidence_recommended_intervention
one_evidence_source != universal_guideline_consensus
low_certainty_evidence != strong_recommendation
absence_of_route_specific_evidence != evidence_of_no_effect
```

No referral generator may describe a recommendation as evidence-based unless a current machine-readable evidence claim supports that wording.

---

# 3. ReferralHistoryV2

`ReferralDraft` gains a structured `history` object. No patient identifiers are added.

```text
ReferralHistoryV2
  onset_date_optional: YYYY-MM-DD|null
  duration_value_optional: number|null
  duration_unit_optional: days|weeks|months|years|null
  onset_pattern: sudden|gradual|post_traumatic|postoperative|post_immobilization|recurrent_episode|other|not_stated
  mechanism_or_trigger_optional: string|null
  course: improving|stable|fluctuating|worsening|recurrent|not_stated
  prior_episode_state: yes|no|not_stated
  prior_treatment_summary_optional: string|null
  prior_treatment_response_optional: improved|partial_response|no_change|worse|mixed|not_stated|null
  relevant_investigation_summary_optional: string|null
  aggravating_factors_optional[]: string
  easing_factors_optional[]: string
  work_sport_activity_context_optional: string|null
  patient_priority_activity_optional: string|null
  route_history_items[]: RouteHistorySelection
```

`RouteHistorySelection`:

```text
history_item_id
state_or_value
unit_optional
free_text_optional
source: clinician_entered|patient_reported|documented_record
```

Rules:

```text
- duration may be represented by exact onset date OR approximate duration; exact date is never inferred from approximate wording
- mechanism/trigger remains history, not proof of tissue diagnosis
- prior treatment and response are carried only when explicitly entered
- route-specific history prompts are dynamically scoped and never auto-selected
- negative history statements require an explicit negative entry; omission is not a negative
```

---

# 4. History composition

Detailed referral order becomes:

```text
CLINICAL IMPRESSION / PRESENTATION
HISTORY
RELEVANT EXAMINATION / FINDINGS
FUNCTIONAL IMPACT
REFERRAL REQUEST
GOALS + TIME HORIZON
REHABILITATION DIRECTIONS
REASSESSMENT PLAN
```

Short referral may compress history to one sentence, but must retain duration/mechanism when supplied and clinically useful.

Example shape only:

> Συμπτωματολογία διάρκειας περίπου 8 μηνών, με έναρξη μετά από κάμψη/άρση φορτίου, με επίμονη πορεία χωρίς πλήρη ύφεση.

The formatter must not convert this into a causal diagnosis.

---

# 5. GoalTimelineV1

Goals may carry timing, but timing is explicitly typed to avoid fake prognostic certainty.

```text
GoalPlanV1
  goal_id
  timing:
    timing_type: reassessment_window|expected_progress_window|goal_achievement_target
    min_value_optional
    max_value_optional
    unit: days|weeks|months
    provenance: clinician_entered|evidence_supported_default|evidence_informed_suggestion
    evidence_claim_ids[]
    certainty: high|moderate|low|very_low|not_graded|not_applicable
    wording_strength: target|expected_progress|reassess_by|no_default
  clinician_override_optional
```

Hard rules:

```text
- evidence default may populate only when a route evidence claim explicitly supports that time window
- low/very-low certainty must not become a promise
- if evidence supports only reassessment timing, output says reassessment, not expected recovery
- clinician may enter an individualized target; it is labelled clinician-entered, not evidence-derived
- no universal 6–8 week default exists
```

---

# 6. ReassessmentPlanV1

```text
ReassessmentPlanV1
  suggested_window_optional
  clinician_selected_window_optional
  trigger_criteria_optional[]
  escalation_criteria_optional[]
  evidence_claim_ids[]
```

The plan distinguishes:

```text
scheduled reassessment
vs
expected clinical progress
vs
full goal achievement
vs
safety-triggered earlier reassessment
```

---

# 7. Evidence architecture

CU-1 reuses the Clinical Excellence evidence-governance model rather than creating an unrelated citation list.

Two layers are required.

## 7.1 EvidenceSource

```text
evidence_id
source_type: guideline|systematic_review|clinical_practice_guideline|consensus|randomized_trial|cohort|narrative_review|other
title
authors_or_organization
year_or_version
reference
doi_optional
url_optional
framework_optional
reviewed_on
status: active|superseded|context_only
```

## 7.2 EvidenceClaim

```text
claim_id
evidence_ids[]
applicable_profile_ids[]
applicable_route_ids[]
domain: diagnostic_definition|history|examination|core_rehabilitation|adjunct|timeline|reassessment|safety|differential
claim_summary
recommendation_direction: recommend|consider|may_consider|do_not_offer|insufficient_evidence|context_only
strength_optional
certainty_optional
conditions_optional[]
conflicts_with_claim_ids[]
reviewed_on
```

A source is not itself a recommendation. Generated behavior is driven by `EvidenceClaim` objects.

---

# 8. RouteEvidenceMap

Every routine route must resolve to a route evidence profile before evidence-aware runtime generation is authorized.

```text
RouteEvidenceProfile
  profile_id
  route_id
  diagnostic_claim_ids[]
  history_claim_ids[]
  core_rehabilitation_claim_ids[]
  adjunct_claim_ids[]
  timeline_claim_ids[]
  reassessment_claim_ids[]
  evidence_gaps[]
  last_reviewed_on
```

Coverage rule:

```text
routine route
→ must have route evidence profile
→ every evidence-labelled rehab/timeline statement must resolve to >=1 active claim
→ missing evidence is represented as evidence_gap, never silently filled by model preference
```

Rare/advanced routes may initially carry explicit `evidence_gap` states, but the UI must not present unsupported recommendations as guideline-backed.

---

# 9. Evidence-aware intervention behavior

For each selectable rehabilitation direction the UI may show one of:

```text
SUPPORTED / RECOMMENDED
CONDITIONAL / MAY CONSIDER
CONFLICTING FRAMEWORKS
INSUFFICIENT ROUTE-SPECIFIC EVIDENCE
CLINICIAN-SELECTED — NO EVIDENCE CLAIM ATTACHED
DO NOT OFFER / NOT ROUTINE
```

Rules:

```text
- clinician choice is preserved
- evidence status never auto-selects a treatment
- contradictory guidelines remain visible separately; no silent hybridization
- adjuncts cannot be presented with the same evidentiary status as core active rehabilitation unless the evidence claims justify it
- evidence caveats belong primarily in clinician UI/evidence panel; referral prose stays concise unless bibliography/evidence appendix is explicitly enabled
```

---

# 10. Evidence display / output

The clinician-facing tool gains an `Evidence` panel after route selection.

It should display compactly:

```text
core evidence posture
recommended/considered/discouraged directions
certainty/strength when available
last reviewed date
source links
known evidence gaps
```

Detailed referral may optionally include a short `Βιβλιογραφική βάση` appendix when the clinician enables it. Routine recipient-facing referral text does not need inline citations by default.

---

# 11. Deep-gluteal example — evidence behavior

For `deep_gluteal_piriformis_presentation`:

```text
- classical DGS definition = non-discogenic sciatic nerve disorder/entrapment in deep gluteal space
- history/exam are central to diagnostic pathway
- route must not infer DGS or piriformis syndrome from buttock pain alone
- conservative-treatment comparative evidence is low quality; no single conservative technique is established as superior
- first-line physiotherapy/activity-oriented care may be referenced through broader back/sciatica guidance when clinically applicable
- a fixed 6–8 week evidence-based recovery claim is NOT supported by the reviewed DGS evidence
```

Therefore the tool may offer clinician-entered or evidence-informed reassessment timing, but must not label a universal DGS recovery window as guideline-derived.

---

# 12. Implementation boundary

Before runtime implementation:

```text
1. freeze ReferralHistoryV2 schema
2. freeze GoalTimelineV1 / ReassessmentPlanV1
3. create machine-readable evidence registry
4. create route evidence map
5. curate all routine routes to the defined coverage gate
6. review evidence freshness/conflicts
7. add synthetic composition fixtures
8. repeat exact design review
```

No runtime recommendation engine may be written before this gate passes.

---

# 13. Acceptance fixtures

At minimum:

```text
A. chronic deep-gluteal pain with 8-month history and uncertain diagnosis
B. routine lateral epicondylalgia with onset/duration and evidence-linked loading plan
C. knee OA with evidence-linked exercise and reassessment window
D. postoperative shoulder where protocol overrides generic evidence suggestions
E. fracture where healing/loading restrictions override route evidence
F. route with conflicting guideline positions on adjunct treatment
G. route with no supported timeline → no invented recovery window
H. clinician-entered timeline overrides evidence suggestion without being mislabeled as evidence-derived
```

---

# 14. Stop rule

```text
design freeze
→ evidence coverage review
→ exact synthetic fixtures
→ DESIGN-COMPLETE or BLOCK
→ only then runtime implementation
```
