# CU-1 History + Evidence + Rehabilitation Sequence Contract v1 — DESIGN CANDIDATE

> **STATUS:** PRE-RUNTIME DESIGN CANDIDATE — requires exact review before implementation.
> **Slice:** CU-1 clinician-quality completion.
> **Purpose:** add clinically useful history, disease-specific evidence provenance, and criteria-based rehabilitation sequencing without inventing facts, overstating evidence or forcing generic treatment protocols.
> **Clinical taxonomy:** existing frozen CU-1 routes remain unchanged unless a specific evidence conflict requires a separately reviewed correction.

---

# 1. Why this contract exists

Product-owner review identified three structural deficits that are not solved by language polish or progressive disclosure:

```text
1. generated referrals lack a coherent HISTORY section
2. goals are a flat list and do not express a safe staged rehabilitation sequence
3. rehabilitation suggestions are not connected at runtime to disease-specific literature provenance
```

The target is a referral that reads like a clinician wrote it and gives the physiotherapist a safe, complete, evidence-linked rehabilitation direction while preserving physiotherapist autonomy and the existing CU-1 safety invariants.

---

# 2. Core invariants

```text
history_fact != diagnosis
history_not_recorded != negative_history
selected_goal != guaranteed_outcome
rehabilitation_phase_order != calendar-duration prescription
progression_criterion_met != elapsed_time_only
clinician_selected_intervention != evidence_recommended_intervention
one_evidence_source != universal_guideline_consensus
low_certainty_evidence != strong_recommendation
absence_of_route_specific_evidence != evidence_of_no_effect
route_A_evidence != route_B_evidence
```

No referral generator may describe a recommendation as evidence-based unless a current machine-readable evidence claim supports that wording for the selected route.

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

Rules:

```text
exact onset date is never inferred from approximate duration
mechanism/trigger remains history, not proof of tissue diagnosis
prior treatment/response are carried only when explicitly entered
route-specific history prompts are dynamically scoped and never auto-selected
negative history statements require explicit negative entry; omission is not a negative
```

---

# 4. RehabilitationSequenceV1 — criteria-based, not calendar-based

The prior `GoalTimelineV1` concept is superseded for routine CU-1 rehabilitation planning.

The product owner's intended meaning is **clinical progression from one therapeutic objective to the next after the previous objective is functionally adequate**, not session frequency or total treatment duration.

```text
RehabilitationSequenceV1
  route_id
  sequence_source: route_evidence_profile
  phases[]: RehabilitationPhaseV1
  clinician_modifications_optional[]
```

`RehabilitationPhaseV1`:

```text
phase_id
order
clinical_objective
intervention_direction_ids[]
progression_criteria[]
precautions_or_do_not_progress_criteria[]
evidence_claim_ids[]
strength_or_certainty_optional
required_for_route: boolean
```

Examples of phase concepts that may exist **only when supported for that route**:

```text
symptom control / irritability reduction
protected or pain-limited mobility
passive ROM restoration
active-assisted ROM
active ROM / motor control
isometric loading
isotonic loading
progressive resistance / strengthening
energy-storage / plyometric loading
functional task retraining
return-to-work / return-to-sport exposure
self-management / recurrence prevention
```

These are a vocabulary, not a universal sequence. Each route selects and orders only the phases supported by its own evidence profile.

Hard rules:

```text
- progression is criterion-based whenever evidence or protocol supports criteria
- elapsed time alone never advances a phase unless an authoritative postoperative/fracture protocol explicitly uses time
- a phase may be omitted entirely when not relevant to the selected disease
- postoperative and fracture written protocols override generic route evidence
- the engine must not force passive ROM before active ROM, analgesia before loading, or any other universal sequence across all diagnoses
- clinician may modify or remove a proposed phase, but the generated referral must preserve that it was clinician-modified when evidence metadata are shown
```

---

# 5. GoalPlanV2

Goals are attached to phases rather than existing as an unrelated flat checklist.

```text
GoalPlanV2
  phase_id
  goal_ids[]
  patient_priority_goal_optional
  progression_criteria[]
  evidence_claim_ids[]
```

The referral may say, for example:

```text
Αρχικά: έλεγχος συμπτωμάτων και αποκατάσταση ανεκτής κινητικότητας.
Μετά την επίτευξη των σχετικών κριτηρίων: προοδευτική ενδυνάμωση / φόρτιση.
Τελικό λειτουργικό στάδιο: επάνοδος στις επιλεγμένες δραστηριότητες με επαρκή ανοχή φορτίου.
```

Only route-supported phases are rendered.

---

# 6. ReassessmentPlanV2

Reassessment is primarily **criteria-triggered**, not a fixed calendar recommendation from the physician.

```text
ReassessmentPlanV2
  phase_progression_checks[]
  failure_to_progress_criteria[]
  safety_escalation_criteria[]
  clinician_requested_review_optional
  evidence_claim_ids[]
```

The physician is not required to prescribe physiotherapy frequency or total duration.

Calendar timing may still exist when:

```text
exact postoperative protocol requires it
fracture/healing follow-up requires it
an evidence source explicitly provides a clinically meaningful reassessment window
clinician explicitly requests a medical review date
```

Otherwise the referral focuses on **what must be achieved before progression**, not how many sessions or weeks must elapse.

---

# 7. Evidence architecture

CU-1 reuses the Clinical Excellence evidence-governance model rather than creating an unrelated citation list.

## 7.1 EvidenceSource

```text
evidence_id
source_type: guideline|clinical_practice_guideline|systematic_review|consensus|randomized_trial|cohort|narrative_review|other
title
authors_or_organization
year_or_version
published_on_optional
reference
doi_optional
url_optional
framework_optional
reviewed_on
next_review_due
status: active|superseded|context_only
supersedes_optional[]
superseded_by_optional[]
```

## 7.2 EvidenceClaim

```text
claim_id
evidence_ids[]
applicable_profile_ids[]
applicable_route_ids[]
domain: diagnostic_definition|history|examination|core_rehabilitation|rehab_phase|progression_criteria|adjunct|safety|differential
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

# 8. RouteEvidenceProfile — disease-specific authority

Every routine route must resolve to its **own** route evidence profile before evidence-aware runtime generation is authorized.

```text
RouteEvidenceProfile
  profile_id
  route_id
  diagnostic_claim_ids[]
  history_claim_ids[]
  examination_claim_ids[]
  rehabilitation_sequence_id
  adjunct_claim_ids[]
  safety_claim_ids[]
  evidence_gaps[]
  primary_source_ids[]
  last_reviewed_on
  next_review_due
  freshness_state: current|review_due|stale|superseded
```

Coverage rules:

```text
routine route
→ must have its own RouteEvidenceProfile
→ must have an explicit RehabilitationSequenceV1
→ every phase/intervention/progression statement must resolve to >=1 active route-applicable evidence claim
→ missing evidence is represented as evidence_gap, never silently filled by generic recommendations
```

**No cross-route generic treatment template is allowed.** Shared rehabilitation concepts may be reused only when each route independently maps to supporting evidence claims.

---

# 9. Disease-specific recommendation behavior

The selected diagnosis/pathway determines the initial evidence-backed rehabilitation proposal.

Examples of architecture, not final recommendations:

```text
lateral elbow tendinopathy
→ use the lateral-elbow RouteEvidenceProfile and its own phase/intervention/progression claims

midportion Achilles tendinopathy
→ use the Achilles RouteEvidenceProfile and its own loading/progression claims
```

The two routes may share words such as `progressive loading`, but they do not share authority, sequence, dose logic, progression criteria or adjunct posture unless their route-specific evidence independently supports them.

The clinician may modify the proposal, but the default must be disease-specific and literature-backed.

---

# 10. What must appear in the referral

The evidence-informed rehabilitation approach is not merely a hidden clinician-side tooltip. **What the literature supports for the selected condition must be visible in the generated referral.**

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

Short referral still uses the same route-specific evidence profile but compresses the output. It must include a compact evidence-based rehabilitation direction and a concise source footer; it must not fall back to generic goals merely to stay short.

---

# 11. Evidence presentation rules

The referral should distinguish recommendation strength without becoming a literature review.

Examples:

```text
core recommendation → direct wording
conditional/low-certainty recommendation → "μπορεί να εξεταστεί" / "επικουρικά"
conflicting guidance → do not hide the conflict; use cautious wording or omit from default
insufficient evidence → never present as routine recommended care
```

Bibliographic section:

```text
- 1–3 highest-authority route-specific sources by default
- source title/organization + year/version
- stable link/DOI where available
- no unrelated general-MSK source when a route-specific CPG exists
```

---

# 12. Evidence freshness / renewal

The route evidence profile is versioned and actively maintained.

```text
new guideline/systematic review detected
→ evidence candidate
→ classify impact: confirming | no_change | potentially_practice_changing | practice_changing | conflicting
→ clinician/reviewer approval
→ update EvidenceClaim(s)
→ update affected RehabilitationSequence only if warranted
→ regression fixtures
→ version bump / changelog
```

Freshness rules:

```text
next_review_due reached → freshness_state=review_due
known superseding guideline → freshness_state=superseded
stale/superseded route profile → UI warns clinician and blocks claims labelled "current guideline recommendation" until reviewed
```

The system must eventually support scheduled evidence surveillance, but surveillance itself never silently changes clinical recommendations.

---

# 13. Deep-gluteal example — evidence behavior

For `deep_gluteal_piriformis_presentation`:

```text
- classical DGS definition = non-discogenic sciatic nerve disorder/entrapment in deep gluteal space
- history/exam are central to diagnostic pathway
- route must not infer DGS or piriformis syndrome from buttock pain alone
- conservative-treatment comparative evidence is low quality; no single conservative technique is established as superior
```

Therefore the route-specific sequence may remain deliberately broad where evidence is weak. The engine must not invent a detailed staged protocol merely to make the referral look comprehensive.

---

# 14. Implementation boundary

Before runtime implementation:

```text
1. freeze ReferralHistoryV2
2. freeze RehabilitationSequenceV1 / GoalPlanV2 / ReassessmentPlanV2
3. create machine-readable evidence registry
4. create RouteEvidenceProfile for every routine route
5. curate a route-specific RehabilitationSequence for every routine route
6. verify evidence freshness/conflicts
7. add synthetic composition fixtures
8. repeat exact design-completeness review
```

No evidence-aware runtime recommendation engine may be written before this gate passes.

---

# 15. Acceptance fixtures

At minimum:

```text
A. chronic deep-gluteal pain with 8-month history and uncertain diagnosis
B. lateral epicondylalgia → elbow-specific evidence profile + elbow-specific staged rehab sequence
C. midportion Achilles tendinopathy → Achilles-specific evidence profile + Achilles-specific staged loading/progression sequence
D. same generic option cannot render identically for B and C unless route-specific evidence independently maps to it
E. postoperative shoulder → exact protocol overrides generic route evidence
F. fracture → healing/loading restrictions override route evidence
G. conflicting guideline positions on adjunct treatment remain explicit
H. unsupported route → no invented staged protocol
I. stale/superseded evidence profile cannot be labelled current guideline-based
```

---

# 16. Stop rule

```text
design freeze
→ complete route-specific evidence coverage
→ exact synthetic fixtures
→ DESIGN-COMPLETE or BLOCK
→ only then runtime implementation
```
