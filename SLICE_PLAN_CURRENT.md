# SLICE_PLAN_CURRENT.md — CU-1 rich referral lateral-elbow prototype v1.16

> **STATUS:** ACTIVE BOUNDED RUNTIME PROTOTYPE — PRODUCT-SHAPE VALIDATION.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice:** CU-1 Physiotherapy Referral — rich clinical rehabilitation document model.
> **Authoritative remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Design parent:** `design/cu1-history-evidence-timeline-2026-08-28` @ `cc479f4a1d818481a886916e3f0f05dc56c623b3`.
> **Writer/runtime writer:** `feat/cu1-rich-referral-lateral-elbow-2026-08-29`.
> **Product-owner authorization:** YES for bounded `lateral_elbow_tendinopathy` runtime prototype + tests.
> **Merge/deploy:** NOT AUTHORIZED until generated text is reviewed.
> **Global rollout:** NEXT after LET wording approval; must be horizontal/shared, not route-by-route product coding.
> **Further evidence-route expansion:** HOLD.

---

# 1. Locked product job

Generate a physiotherapy referral that is clinically useful enough to guide the **direction and expected progression of rehabilitation** while preserving the treating physiotherapist's responsibility for exact exercise selection and dosing.

The output has two audiences:

```text
physiotherapist → understands clinical priorities, rehabilitation progression and expected functional endpoint
patient          → understands what meaningful rehabilitation should contain and can distinguish it from passive-only care
```

The product must therefore sit between two unacceptable extremes:

```text
TOO THIN
"reduce pain / improve strength / improve function"

TOO PRESCRIPTIVE
named exact exercises + sets + repetitions + kilograms + fixed weeks + fabricated clearance cutoffs
```

---

# 2. Locked rehabilitation-document grammar

Every rehabilitation stage is represented as:

```text
STAGE
├── Goal(s): what clinical/functional capability should improve
├── Intervention directions: how those goals are pursued
├── Progress markers: what improvement should be observed/measured
└── Next-stage orientation: what the rehabilitation evolves toward
```

This grammar is a clinical/document organization layer. It is not itself a literature authority and must never be labelled as a universal evidence-validated protocol.

Intervention directions may name broad treatment categories such as:

```text
active ROM / mobility work
isometric activation/loading
concentric/eccentric resisted loading
progressive strengthening/endurance work
load/activity modification
manual therapy where applicable
cryotherapy/TENS or other evidence-compatible symptom modulation
functional graded exposure / work or sport reintegration
```

They must not define universal dosage.

---

# 3. Non-negotiable product invariants

1. **Goal without method is insufficient.** "Μείωση πόνου" must be linked to how pain/irritability will be managed. "Αποκατάσταση ROM" must be linked to mobility treatment. "Βελτίωση load tolerance" must be linked to progressive loading.
2. **Passive-only care is not complete rehabilitation.** TENS, laser, ultrasound, cryotherapy, taping, manual therapy or similar modalities may be adjunctive when appropriate but cannot satisfy the rehabilitation plan by themselves.
3. **No exercise micromanagement.** No universal sets/reps/kg/hold times or exact named exercise menu in the physician-generated referral.
4. **No false precision.** No universal fixed week phases, pain cutoff, strength-symmetry threshold, PRTEE/DASH percentage or return-to-work threshold unless a patient-specific protocol/clinician instruction or appropriately reviewed authority actually supplies it.
5. **No invented patient facts.** Occupation or sport may inform reintegration only from explicit patient context; it must not create causal mechanism or unreported task limitations.
6. **Evidence direction != automatic selection.** Evidence-supported adjuncts are not automatically required.
7. **Therapist execution remains therapist-owned.** The referral specifies destination and treatment direction; exact technique/exercise/dose remains clinical execution detail.

---

# 4. Evidence/safety architecture preserved

The accepted architecture remains:

```text
PATIENT CLINICAL STATE
+
STRICT EVIDENCE / SAFETY ENGINE
+
REFERENCE REHABILITATION PATHWAYS / CLINICAL ORGANIZATION
+
REFERRAL DOCUMENT POLICY
→ SHORT or DETAILED REFERRAL
```

The evidence layer remains responsible for applicability, strength/certainty, conflicts, do-not-infer rules, protocol precedence and reassessment/safety boundaries. It is not the sole prose author.

Useful reference rehabilitation pathways may inform practical stage organization without promoting their exact protocol thresholds to guideline authority.

---

# 5. Lateral elbow prototype — expected clinical shape

## Stage 1 — symptom/irritability control, mobility and initial loading

### Goals

- reduce pain/irritability enough to support active rehabilitation;
- maintain or restore functionally adequate elbow/forearm/wrist mobility where impaired;
- begin restoration of wrist-extensor load tolerance;
- identify and modify relevant aggravating load while maintaining useful activity.

### Intervention directions

- education, activity/load modification and ergonomic adaptation as relevant;
- active mobility/ROM and flexibility work for identified mobility restriction;
- low-demand active wrist-extensor recruitment with isometric loading as an early option when tolerated;
- short-term symptom-modulation adjuncts such as cryotherapy/TENS in appropriate contexts;
- selected manual therapy, taping or other evidence-compatible adjuncts where indicated;
- passive modalities must serve active progression rather than replace it.

### Progress markers

- irritability and activity-related pain are improving rather than progressively worsening;
- functional ROM is adequate or improving when an impairment was present;
- initial active/isometric loading is tolerated without clinically important prolonged exacerbation;
- basic grip/use tasks are becoming better tolerated.

### Next-stage orientation

Progress toward more demanding resisted loading and restoration of strength/endurance when the clinical response permits.

---

## Stage 2 — strength, endurance and load-capacity restoration

### Goals

- restore progressive wrist-extensor loading capacity;
- improve grip strength and repeated-use tolerance;
- improve upper-limb endurance for relevant daily/work/sport demands;
- address proximal shoulder/scapular impairment only when actually identified.

### Intervention directions

- progressive resisted wrist-extensor loading, evolving from initial activation/isometric work toward concentric and eccentric loading as tolerated;
- progressive grip and upper-limb endurance work;
- progressive mechanical loading based on response rather than a universal dose;
- shoulder/scapular stabilizer work only when examination identifies a relevant impairment;
- adjunct symptom-modulation treatment may continue selectively but must not displace active progression.

### Progress markers

- objective and/or functional grip capacity improves relative to baseline;
- progressive resisted loading is increasingly tolerated;
- repeated use produces less limitation and no clinically important prolonged deterioration;
- patient-reported function and priority tasks are improving.

### Next-stage orientation

Progress toward higher-demand, longer-duration and task-specific functional loading.

---

## Stage 3 — functional / occupational / sport reintegration

### Goals

- restore capacity for the patient's explicitly recorded high-demand activities;
- restore tolerance to repeated and sustained upper-limb use;
- support self-management, load control and recurrence-risk reduction.

### Intervention directions

- progressively higher mechanical demand and longer-duration/repeated loading;
- task-specific grip/upper-limb functional conditioning;
- graded exposure to actual work, sport or hobby demands that were explicitly recorded as limited;
- ergonomic/activity strategy and independent self-management plan.

### Progress markers

- meaningful return toward the patient's actual priority activities;
- improved function/outcome measures relative to baseline;
- sufficient strength/endurance/load tolerance for the patient's real demands;
- no material or disproportionate deterioration with ordinary functional loading;
- patient demonstrates practical self-management of load and recurrence symptoms.

### Completion/reassessment orientation

No universal numeric discharge rule is generated. Reassess diagnosis/owner/plan when progress is discordant or when neurological, cervical/radicular, mechanical-block, traumatic/instability or other atypical findings emerge.

---

# 6. LET evidence boundary for the prototype

Use the already-reviewed lateral-elbow package and preserve its exact limitations:

- JOSPT 2022: subacute/chronic resisted wrist-extensor exercise, isometric/concentric/eccentric, Grade B, no universal dose;
- high-demand reintegration Grade F and conditional;
- shoulder/scapular work Grade C only if impairment exists;
- local mobilization/manipulation Grade B when selected/applicable;
- dry needling Grade B when selected/applicable;
- rigid taping Grade B for selected irritable short-term context;
- counterforce/wrist support Grade F for selected aggravating-activity/immediate context;
- education/behavioral/ergonomic intervention Grade E;
- cryotherapy + burst TENS may be used for short-term pain reduction in the CPG-defined context, cryotherapy may be used for irritable LET, TENS may be used for short-term pain relief, and laser may be used as an adjunctive option;
- ultrasound as stand-alone treatment has conflicting evidence and must not be presented as a core treatment;
- PRTEE/DASH/PSFS and grip/ROM measures are follow-up measures, not automatic phase-transition or discharge thresholds;
- the Day/Lucado/Uhl 2019 three-phase program is a Level-5 clinical rehabilitation commentary/pathway: its useful organization may inform this document model, while its exact repetitions/loads/pain cutoffs are not universalized.

---

# 7. Short vs Detailed document policy

Both modes consume the same patient/evidence state.

**Short** must still be clinically directive, not a compressed checkbox list. It should include:

```text
diagnosis/context
+ core active rehabilitation progression
+ key functional endpoint
+ passive-adjunct boundary
+ reassessment trigger when relevant
```

**Detailed** should expose the full stage grammar:

```text
clinical picture
+ Stage 1 goals/methods/progress
+ Stage 2 goals/methods/progress
+ Stage 3 goals/methods/progress when relevant
+ monitoring
+ adjunct boundary
+ reassessment
```

Short is a compression of the same plan, not a separate clinical truth.

---

# 8. Acceptance fixtures for the prototype

The prototype must pass all of the following:

```text
PRODUCT-SHAPE
- every rendered stage contains both goals and intervention directions
- progress markers are visible
- detailed output contains meaningful evolution across stages
- short output preserves the active-rehab direction

PASSIVE-ONLY FAILURE
- output cannot be interpreted as complete care through TENS/laser/ultrasound/manual modalities alone
- active loading/progression and functional reintegration remain explicit

NO FALSE PRECISION
- no universal sets/reps/kg/hold times
- no fixed week ranges
- no universal pain cutoff
- no invented PRTEE/DASH/grip discharge percentage

CONDITIONALITY
- proximal/scapular work is conditional on actual impairment
- work/sport reintegration does not invent unreported tasks
- adjunct evidence does not become mandatory selection

SAFETY / INTEGRITY
- diagnosis is not generated from findings alone
- missing/not-assessed is not rendered as normal
- atypical PIN/cervical/mechanical/traumatic context preserves reassessment ownership
- existing non-LET formatter behavior remains unchanged in this prototype
```

---

# 9. Exact next action

Implement only the LET product-shape override at the existing Greek formatter seam and focused tests. Generate actual short and detailed outputs from a synthetic but contract-valid LET case and present them to the product owner.

After wording review:

```text
IF product owner requests corrections
→ correct LET product shape and re-render

IF product owner approves LET output
→ freeze shared Rich Rehabilitation Document Model
→ implement horizontal/global generation across all conditions
→ use route clinical/evidence content as data/configuration
→ DO NOT manually hand-code one disease after another
```

Further route-by-route evidence expansion remains on hold while this product work is active.
