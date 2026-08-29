# CU-1 Lateral Elbow Tendinopathy — Exact Route Review — 2026-08-29

## Verdict

```text
ROUTE: lateral_elbow_tendinopathy
RESULT: PASS AS SINGLE-PHASE EVIDENCE-BOUNDED ROUTE
PROFILE: existing rep_lateral_elbow_tendinopathy_v1 retained
ACTIVE SEQUENCE AFTER ACTIVATION: seq_lateral_elbow_evidence_bounded_v1
RUNTIME AUTHORIZED: NO
```

This PASS is conditional on applying the reviewed activation amendment:

```text
clinic_utilities/contracts/cu1_evidence_route_coverage_lateral_elbow_amendment_v1.yaml
```

The amendment is part of the reviewed route package. It must be activated only after the route-extension shard is merged into the logical registry. It removes a synthetic outcome-measure strength, restores three exact Grade-B recommendation directions under selection/applicability gates, and redirects the existing core profile away from the incomplete seed sequence.

---

## 1. Frozen route reviewed

Frozen owner:

```text
clinic_utilities/physio_profiles/elbow_v1_1.md
```

Frozen route:

```text
E1
lateral_elbow_tendinopathy
```

The frozen design already requires:

```text
lateral elbow pain != automatic LET diagnosis
provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
subjective symptom != objective neurological deficit
not assessed != normal
adjunct != core rehabilitation
```

The route must preserve radial/PIN, cervical, intra-articular, instability and traumatic alternatives when the presentation is atypical.

---

## 2. Existing active core authority retained

The core evidence registry already contains:

```text
source:
lateral_elbow_cpg_2022

claims:
lateral_elbow_resisted_wrist_extensor_exercise
lateral_elbow_high_demand_phased_reintroduction

profile:
rep_lateral_elbow_tendinopathy_v1

seed sequence:
seq_lateral_elbow_seed_v1
```

The source is the 2022 APTA/JOSPT clinical practice guideline:

```text
Lucado AM, Day JM, Vincent JI, MacDermid JC, Fedorczyk J, Grewal R, Martin RL.
Lateral Elbow Pain and Muscle Function Impairments.
J Orthop Sports Phys Ther. 2022;52(12):CPG1-CPG111.
DOI 10.2519/jospt.2022.0302
```

APTA continues to list this as the current lateral-elbow CPG at the 2026-08-29 review date.

The original seed had two phases but no evidence-linked transition criterion. That incompleteness is resolved by replacing the two-phase seed as the profile target with one evidence-bounded phase and keeping high-demand reintroduction as a conditional direction rather than inventing a phase-transition rule.

---

## 3. Current evidence freshness review

### A. APTA/JOSPT 2022 CPG — current route-specific guideline

Key exact route authorities retained:

```text
subacute/chronic resisted wrist-extensor exercise
→ Grade B
→ isometric, concentric and/or eccentric

high-demand reintroduction of stress/strength/endurance/motor control
→ Grade F

shoulder/scapular stabilizer training only when impairment identified
→ Grade C

local elbow mobilization/manipulation for short-term outcomes
→ Grade B

dry needling
→ Grade B

rigid taping for irritable LET short-term relief
→ Grade B

counterforce/wrist-support orthosis during aggravating activity for immediate benefit
→ Grade F

education/behavioral modification/ergonomic or workstation adjustment
→ Grade E
```

The same guideline explicitly identifies unresolved knowledge gaps around optimal strengthening dosage and high-demand exercise progressions.

### B. Cochrane 2024 manual therapy/exercise review

```text
Wallis JA et al.
Manual therapy and exercise for lateral elbow pain.
Cochrane Database Syst Rev. 2024;5(5):CD013042.
DOI 10.1002/14651858.CD013042.pub2
```

The review searched through 2024-01-31. Its central design consequence is not a treatment prohibition. It found low-certainty evidence that manual therapy, exercise, or both may produce only small improvements in pain/disability at end of treatment, with effects generally not sustained and limited or no meaningful differences for several longer-term outcomes.

Therefore:

```text
CPG treatment direction may remain active
!=
claim of large, durable or guaranteed effect
```

The Cochrane review does not justify converting exercise into `do_not_offer`.

### C. Campos et al. 2024 non-invasive therapy systematic review/meta-analysis

```text
Campos MGM et al.
Braz J Phys Ther. 2024;28(2):100596.
DOI 10.1016/j.bjpt.2024.100596
```

Across 47 randomized trials and 22 non-invasive therapy categories, most estimates were small-to-no effect with evidence commonly low or very low certainty. This reinforces the rule that no conservative modality should be labelled universally superior and symptom resolution must not be promised.

---

## 4. Diagnosis and differential boundary — PASS

The CPG describes LET diagnosis as history-plus-examination synthesis and notes weak diagnostic usefulness for common special tests when considered alone.

CU-1 therefore preserves:

```text
local lateral tenderness
+ painful resisted wrist/digit extension
+ reduced pain-free grip
+ Cozen/Mill/Maudsley-type finding
+ imaging abnormality

!= autonomous software-generated LET diagnosis
```

Atypical or materially discordant findings require consideration of alternative owners such as:

```text
cervical radicular source
radial tunnel presentation
posterior interosseous motor neuropathy
intra-articular pathology
radiocapitellar disorder
posterolateral rotatory instability
material traumatic pathology
```

The route is not a differential-diagnosis engine; it preserves clinician reassessment/owner-switch semantics.

Gate: **PASS**.

---

## 5. History prompt coverage — PASS for this route

New prompts:

```text
let_symptom_course_and_irritability
let_grip_wrist_extension_symptom_pattern
let_high_demand_function_context
let_prior_treatment_and_response
let_atypical_neural_cervical_joint_context
let_patient_priority_grip_or_load_task
```

Existing core prompt retained:

```text
lateral_elbow_load_context
```

Hard rules:

```text
missing history != negative history
irritability is not inferred from one pain score
not-stated duration != inferred acuity class
work/sport exposure != proven causation
```

Gate: **PASS**.

---

## 6. Rehabilitation sequence decision — PASS

Reviewed replacement sequence:

```text
seq_lateral_elbow_evidence_bounded_v1
```

It contains one required phase:

```text
let_irritability_guided_active_rehabilitation
```

Broad route objective:

```text
support function
+ individualized education/self-management
+ relevant load/activity modification
+ progressive restoration of wrist-extensor/grip load tolerance
```

Conditional intervention directions:

```text
subacute/chronic LET
→ Grade-B resisted wrist-extensor loading
→ isometric and/or concentric and/or eccentric
→ no universal dose

actual proximal impairment identified
→ Grade-C shoulder/scapular stabilizer work may be added

high-demand occupation/sport/hobby/performing-arts context
→ Grade-F gradual high-demand reintroduction may be used
→ no numeric clearance threshold
```

Acute/highly irritable LET does **not** inherit the subacute/chronic Grade-B resisted-loading claim merely because E1 is selected. The route can still carry education/load-management and irritability-guided rehabilitation context while specific subacute/chronic loading authority remains condition-gated.

Gate: **PASS**.

---

## 7. Why there is no second high-demand phase

The prior seed sequence had:

```text
phase 1 = resisted loading
phase 2 = high-demand function
```

but no evidence-based transition rule from phase 1 to phase 2.

The 2022 CPG itself gives high-demand progression only Grade F and states that research on these progressions is needed. A second normative phase would therefore invite a fabricated transition criterion.

The reviewed design instead uses:

```text
one evidence-bounded phase
+ high-demand conditional direction when relevant
+ progression_criteria: []
```

This is more faithful to the available evidence than forcing a staged protocol.

Gate: **PASS**.

---

## 8. Adjunct governance — PASS

Adjuncts are never auto-selected.

Reviewed route-evidence adjunct claims include:

```text
local manual therapy
→ Grade B when selected and tolerated

dry needling
→ Grade B when selected and competence/availability confirmed

rigid taping
→ Grade B for selected irritable short-term context

counterforce/wrist-support orthosis
→ Grade F for selected aggravating activity / immediate context
```

The activation amendment restores the exact `recommend` direction for the three Grade-B `should use` recommendations while keeping them optional through explicit selection/applicability conditions.

This distinction is mandatory:

```text
evidence direction
!=
automatic treatment selection
```

The newer Cochrane synthesis remains visible as effect-magnitude/durability uncertainty and prevents wording that manual therapy is required or durably superior.

ESWT is intentionally not promoted to route-evidence authority in this review. The frozen UI may retain clinician selection, but absent a reviewed LET-specific ESWT claim it must remain clinician instruction rather than automatically labelled evidence recommendation.

Gate: **PASS**.

---

## 9. Outcome measures are not progression criteria — PASS after mandatory amendment

The CPG separates:

```text
PRTEE and/or DASH + PSFS/high-demand activity-specific scale
→ Grade A

ROM + pressure pain threshold + pain-free grip + maximum grip
→ Grade B
```

The initial route-extension draft mistakenly represented those two measure families with one synthetic `A_B_by_measure_family` strength string.

Mandatory activation amendment:

```text
let_outcome_tracking_not_progression_threshold_2022
→ strength_optional: not_graded
→ exact A vs B distinction retained in claim text and fixture
```

These measures are recommended for assessment/follow-up. None is converted into an automatic phase-transition, return-to-work, return-to-sport or discharge threshold.

Gate: **PASS WITH MANDATORY AMENDMENT**.

---

## 10. Progression / return-to-function evidence gap — explicit and acceptable

Current evidence does not validate:

```text
one optimal exercise dose
one numeric loading progression threshold
one universal pain rule for progression
one universal pain-free-grip clearance value
one fixed PT visit frequency
one total PT course duration
one universal RTW/RTS threshold
```

Therefore the sequence deliberately contains:

```text
progression_criteria: []
```

This is an explicit evidence gap, not missing design work.

Gate: **PASS**.

---

## 11. Safety / owner boundary — PASS

Routine LET rehabilitation must not absorb:

```text
progressive objective PIN/radial motor deficit
cervical/radicular presentation
major mechanical block / true locked elbow
material traumatic/instability concern
other clinician concern suggesting a different elbow owner
```

The regression fixture corpus verifies fail-closed behavior and forbids generic elbow/MSK fallback.

Gate: **PASS**.

---

## 12. Regression fixture gate — PASS

Normative fixture extension:

```text
clinic_utilities/contracts/cu1_lateral_elbow_fixtures_v1.yaml
```

It covers:

```text
typical subacute/chronic loading
acute/high-irritability scope boundary
high-demand Grade-F reintroduction without numeric threshold
single test/imaging != diagnosis
PIN motor deficit safety/owner switch
cervical pattern not silently absorbed
manual therapy selected vs mandatory
manual therapy CPG strength + Cochrane uncertainty
dry needling selected not core
orthosis immediate-context only
A-vs-B outcome-measure grading without synthetic hybrid
2024 low certainty != do-not-offer exercise
ESWT not auto-evidence-authorized in this review
missing history remains missing
```

Gate: **PASS**.

---

## 13. Activation package

After exact-head focused CI succeeds, activation must be atomic in logical meaning:

```text
1. activate cu1_evidence_route_coverage_lateral_elbow_v1.yaml in manifest
2. add cu1_evidence_route_coverage_lateral_elbow_amendment_v1.yaml to logical amendments
3. resolve rep_lateral_elbow_tendinopathy_v1 to seq_lateral_elbow_evidence_bounded_v1
4. add cu1_lateral_elbow_fixtures_v1.yaml to matrix fixture extensions
5. mark lateral_elbow_tendinopathy sequence_complete in matrix
6. reconcile CURRENT_OPERATIONAL / SLICE_PLAN_CURRENT / changelog / PR body
7. final exact-head focused CI
```

No runtime implementation is authorized by this activation.

---

## 14. Final route verdict

```text
route identity                         PASS
diagnostic/finding separation         PASS
differential/owner boundary           PASS
history prompt coverage               PASS
current source freshness              PASS
CPG framework grades preserved        PASS after mandatory amendment
newer synthesis uncertainty preserved PASS
single-phase sequence                 PASS
acute/subacute-chronic applicability  PASS
high-demand behavior                  PASS
adjunct selection semantics           PASS
progression evidence gap explicit     PASS
matching fixtures                     PASS
runtime authorization                 NO

FINAL ROUTE RESULT                     PASS
```

The next route may begin only after manifest/matrix/canonical reconciliation and final-head CI confirm this activation package.
