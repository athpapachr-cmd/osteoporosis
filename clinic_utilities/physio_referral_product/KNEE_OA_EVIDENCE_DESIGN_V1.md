# KNEE_OA_EVIDENCE_DESIGN_V1.md — Physio Referral Step 2

> **STATUS:** REVIEW-HARDENED DESIGN CANDIDATE — 2026-09-11.
> **Scope:** Knee Osteoarthritis only.
> **Parent route:** `profile_id=knee`, `route_id=knee_osteoarthritis`.
> **Parent runtime:** existing CU-1 Physiotherapy Referral v2 — read-only in this slice.
> **Machine companion:** `contracts/knee_oa_evidence_contract_v1.yaml`.
> **Runtime implementation:** NOT AUTHORIZED.

---

# 1. Purpose

Step 2 defines the evidence layer beneath the minimal Knee-OA referral UX. For each material option the product must be able to answer, without cluttering the routine screen:

```text
what is supported
what is context-dependent
what is uncertain
what is advised against
where guidelines materially disagree
why the item is suggested/cautioned
which source/year supports the statement
when the product last reviewed that evidence
```

The product must not manufacture consensus by averaging incompatible frameworks.

---

# 2. Reviewed source set

Six major guideline/CPG authorities were reviewed:

1. **VA/DoD Hip & Knee OA CPG v3.0, 2026** — evidence through July 2025; GRADE; current structured PT, weight, bracing and complementary-intervention positions.
2. **Singapore ACE Knee OA ACG, 24 Apr 2026** — knee-specific; education, exercise and weight management as mainstays; allied-health/walking-aid support; selected adjunct acupuncture.
3. **EULAR non-pharmacological core management, 2023 update (published 2024)** — individualized multicomponent plan; education/self-management; exercise (strength/aerobic/flexibility/neuromotor); delivery mode; weight; walking aids.
4. **NICE NG226, 2022** — tailored exercise, weight management, selected walking aids; manual therapy only alongside exercise; no acupuncture/dry needling; supports not routine without specific context.
5. **AAOS OAK3, 2021** — strong exercise/self-management/education; moderate neuromuscular training and weight loss; limited manual therapy/acupuncture; unclear dry needling.
6. **ACR/AF 2019 guideline (published 2020)** — strong exercise, weight loss, self-management, cane/tibiofemoral brace; conditional balance/taping/acupuncture; conditional-against manual therapy over exercise alone and massage; TENS strongly against.

Every source retains its own recommendation direction and native strength/certainty. A later app summary is only a projection over those source positions.

---

# 3. Step-1 REPLAN — add an explicit guideline-conflict state

The original five evidence states cannot honestly represent material framework disagreement. Acupuncture proves the point:

```text
NICE 2022      → do not offer
AAOS 2021      → limited recommendation in favour
ACR/AF 2019    → conditional recommendation in favour
ACE 2026       → selected adjunct use
VA/DoD 2026    → insufficient evidence for or against
```

Therefore the Step-1 REPLAN trigger is activated and the app evidence model becomes:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

Greek surface semantic for the new state:

```text
Οι οδηγίες διαφέρουν
```

This state is not a vote, meta-analysis or arithmetic majority. It means that reviewed credible frameworks differ materially enough that hiding the disagreement could alter a clinician's choice.

Hard distinctions:

```text
INSUFFICIENT EVIDENCE != EVIDENCE AGAINST
GUIDELINE CONFLICT != CONSENSUS
SOURCE YEAR != PRODUCT REVIEW DATE
```

---

# 4. Source-claim scope — prevent evidence laundering

A second review finding is that a strong recommendation for a broad category must not silently become a strong recommendation for every narrower component.

Each source position therefore records one of:

```text
direct_item_recommendation
named_component_of_broader_recommendation
broader_recommendation_only
contextual_clinical_mapping
```

Example:

- ACR/AF strongly recommends **exercise** and explicitly lists strengthening among exercise modes.
- The product may therefore support strengthening as part of the exercise programme.
- It must not claim that ACR/AF issued a separate strong recommendation for one exact progressive-strengthening protocol.

Likewise, task-specific retraining may be clinically sensible within individualized PT, but the product must not pretend a guideline separately mandates stair retraining merely because stairs are the patient's limitation.

---

# 5. Default Knee-OA starting plan

The default should remain small and defensible.

Implicit:

```text
physiotherapy_assessment_and_individualized_active_rehabilitation
```

Visible preselected core:

```text
therapeutic_exercise
progressive_strengthening
education_and_self_management
```

These three are broadly supported across current reviewed frameworks.

`graded_activity_exposure` is **not** a default core selection after review. Physical activity/progression is supported, but the CU-1 ID is more specific than the broad evidence statement. It becomes a context-driven option when walking, exercise tolerance or a patient-priority activity is actually relevant.

---

# 6. Context-driven active rehabilitation

The following remain available without appearing as universal defaults:

| CU-1 item | App state | Trigger / interpretation |
|---|---|---|
| `graded_activity_exposure` | conditional/context-dependent | walking, exercise or patient-priority limitation |
| `progressive_endurance_or_capacity_work` | conditional/context-dependent | reduced walking/activity capacity |
| `mobility_exercise_when_restricted` | conditional/context-dependent | actual ROM restriction |
| `neuromuscular_proprioceptive_training` | conditional/context-dependent | balance/control deficit, relevant giving-way/instability symptom |
| `balance_stepping_recovery_training` | conditional/context-dependent | assessed balance deficit |
| `gait_walking_practice` | conditional/context-dependent | walking limitation/tolerance |
| `functional_task_retraining` | conditional/context-dependent | stairs, sit-to-stand, squat, kneeling or patient-priority task |
| `home_exercise_programme` | conditional/context-dependent | delivery choice based on needs/preferences/access |

Important interpretation rules:

- subjective giving-way does not become objective structural instability;
- missing finding is not a negative finding;
- no exact exercise dose/protocol is invented by the referral tool;
- gait/task retraining is described as individualized clinical mapping, not as a stand-alone guideline mandate.

---

# 7. Supports / power-user options

## Walking aid

`walking_aid_assessment_and_training` is supported for selected patients and is `conditional_or_context_dependent`.

Evidence includes EULAR, NICE, ACE and ACR/AF. ACR/AF strongly recommends cane use when impact on ambulation/stability/pain warrants it.

**Existing integration seam:** the canonical CU-1 ID already exists but is not currently exposed in the Knee UI relevance scope. Step 2 records this only. A later prototype may use a bounded presentation-scope amendment; no new ID is needed.

## Brace / orthosis

`orthosis_or_brace_context` is `conditional_or_context_dependent`, not a default.

VA/DoD supports selected bracing; ACR/AF supports tibiofemoral bracing strongly and patellofemoral bracing conditionally in appropriate cases; NICE explicitly advises against routine supports without the relevant instability/loading/function criteria.

## Taping

`taping` remains `conditional_or_context_dependent`, not core. ACR/AF conditionally supports kinesiotaping; NICE restricts tape/support use to selected biomechanical/functional contexts.

---

# 8. Manual therapy and soft tissue — show disagreement

`manual_therapy` and `soft_tissue_techniques` are power-user adjuncts with:

```text
app_state = guideline_conflict_or_mixed
```

Source logic:

- NICE: manual therapy only alongside therapeutic exercise; insufficient evidence for manual therapy alone;
- AAOS: limited support for manual therapy in addition to exercise / massage as adjunct;
- ACR/AF: manual therapy with exercise conditionally against **over exercise alone**; massage conditionally against for OA symptom reduction.

Selected-state bubble:

```text
Οι οδηγίες διαφέρουν
```

They are never allowed to replace active rehabilitation.

---

# 9. Acupuncture — flagship conflict example

`acupuncture` remains selectable only as a non-default power-user adjunct.

```text
app_state = guideline_conflict_or_mixed
selected bubble = Οι οδηγίες διαφέρουν
```

The first prototype must not auto-promote it.

The `i` layer should concisely state that major guidelines differ: NICE recommends against, AAOS/ACR/ACE allow selected use with different strengths, and VA/DoD 2026 considers evidence insufficient for or against.

The referral must never present acupuncture as a substitute for exercise/self-management.

---

# 10. Dry needling — excluded and not mislabeled

The existing frozen Knee v1.1 profile excludes dry needling from the Knee-OA selectable surface, and Step 2 preserves that decision.

Reviewed positions:

```text
NICE 2022     → against
AAOS 2021     → efficacy unclear / further evidence
VA/DoD 2026   → insufficient for or against
```

No reviewed Step-2 source provides a positive recommendation. Therefore the machine contract classifies the evidence state as:

```text
recommendation_against_routine_use
```

while retaining:

```text
product_selectable = false
```

This does **not** mean the product claims dry needling has been proven harmful or universally ineffective; the detailed evidence sheet preserves the against-vs-insufficient distinction.

---

# 11. Weight management — strong evidence, current machine seam

Weight management is strongly and consistently supported when overweight/obesity applies across VA/DoD, EULAR, NICE, ACR/AF, AAOS and ACE.

Clinical evidence state:

```text
recommended_or_supported
```

But the current CU-1 option catalog has no dedicated selectable weight-management ID, despite the frozen Knee clinical profile already mentioning support/referral when relevant.

Step-2 decision:

```text
role = product_advisory_only_for_now
product_selectable = false
auto_trigger = false
requires future explicit overweight/obesity context
NO new CU-1 ID in Step 2
```

The system must not infer overweight/obesity from missing data, appearance, age or indirect context.

---

# 12. Suggestion policy

## Core omission suggestions

If the clinician removes one of:

```text
therapeutic_exercise
progressive_strengthening
education_and_self_management
```

the product may show a compact evidence-backed one-tap suggestion. It never blocks the referral.

## Phenotype/function suggestions

Suggestions are allowed only from explicit structured context, for example:

```text
ROM restriction       → mobility
quadriceps weakness   → strengthening emphasis
balance deficit       → neuromuscular / balance work
walking limitation    → graded activity/endurance/gait
stairs/sit-to-stand   → task-specific functional retraining
```

## No automatic adjunct promotion

Do not auto-suggest in the first prototype:

```text
manual therapy
soft-tissue techniques
acupuncture
taping
brace/orthosis
walking aid
weight management
```

A later rule may add context-sensitive promotion only after separate review.

---

# 13. Evidence bubbles and `i` layer

Routine semantic bubbles:

```text
limited_or_insufficient_evidence   → Περιορισμένη τεκμηρίωση
guideline_conflict_or_mixed        → Οι οδηγίες διαφέρουν
recommendation_against_routine_use → Δεν συνιστάται για συνήθη χρήση
```

The `i` layer shows:

```text
plain-language rationale
source-specific positions
source guideline/version year
native source strength/certainty where supplied
product Reviewed date
```

No warning modal and no fake “confidence percentage”.

---

# 14. Update governance

There is no autonomous literature-to-live-rule pipeline.

```text
surveillance
→ candidate evidence change
→ source review
→ impact classification
→ clinician/product-owner approval
→ versioned contract update
→ tests/review
→ separate runtime release
```

Candidate impact classes:

```text
confirming_no_change
clarifying_no_state_change
potential_state_change
state_change
source_conflict_changed
source_withdrawn_or_superseded
```

A newer source does not silently erase an older still-relevant framework.

---

# 15. CU-1 compatibility result

No broad Knee route/taxonomy rewrite is required.

Existing IDs cover the main evidence-aware product surface, including exercise, strengthening, mobility, graded activity, education/self-management, neuromuscular/balance, gait, function, manual/soft tissue, acupuncture, taping and brace.

Two bounded seams remain:

```text
walking aid
→ canonical ID already exists
→ absent from current Knee UI relevance scope

weight management
→ clinically supported and present in Knee profile prose
→ no dedicated selectable machine ID
→ advisory-only for Step 2
```

Existing dry-needling exclusion is preserved.

---

# 16. Step-2 acceptance boundary

Step 2 is eligible to freeze only when machine and human contracts agree on:

- six evidence states including explicit guideline conflict;
- source-specific positions and source-claim scope;
- three-item visible core default plus implicit individualized PT;
- context-driven suggestions without inference from missing data;
- adjunct conflict semantics;
- walking-aid and weight-management integration seams;
- evidence freshness/provenance;
- no runtime, UI, billing or patient-data mutation.

A separate exact design review remains required before declaring Step 2 frozen.
