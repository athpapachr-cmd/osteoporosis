# PHYSIO P1 — Clinical Field Utility Review before R2-A Decision 2

> **DATE:** 2026-10-04 Asia/Nicosia.
> **MODE:** evidence/product utility review; no runtime mutation.
> **QUESTION:** before assigning UI ownership to pain-location, tenderness, swelling/effusion and other examination concepts, does each concept deserve a structured place in a fast physician-to-physiotherapist Knee-OA referral at all?

## 1. Existing internal review signal

The existing product reviews already established a strong utility principle:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

Relevant prior decisions/findings:
- Step 6A added optional pain location because the early prototype was considered too summary-level; this proved semantic feasibility, not receiver value.
- Cross-review synthesis explicitly said not to expand the pain map further.
- PT-05 / review disposition identified anatomical granularity as a candidate for reduction pending clinical/receiver value.
- FFD was kept advanced/optional rather than promoted to routine collection.
- PT-03 / GEN-F09 explicitly deferred new functional-baseline fields until receiver value was demonstrated.
- Post-review amendment preserved multiselect pain localization, but did not establish that it changes the rehabilitation plan.

## 2. External evidence signal

### Management is driven mainly by symptoms + function

NICE NG226 states that OA management should be guided by symptoms and physical function. Therapeutic exercise should be tailored to individual needs; NICE does not use pain location, focal tenderness or crepitus to choose the core exercise plan.

EULAR 2023/2024 similarly recommends an individualised multicomponent plan and exercise with dosage/progression tailored to physical function, preferences and available services.

The 2026 ACE knee-OA evidence-to-recommendation framework places particular emphasis on pain level, functional capacity, QoL and prior response. It notes that physiotherapy is particularly relevant when functional limitations, muscle weakness or limited ROM are present.

### Full examination != referral payload

The 2026 VA/DoD guideline says a full symptomatic-knee OA examination may include crepitus, joint-line tenderness, ROM, strength, effusion and laxity.

That supports those findings as legitimate examination concepts. It does **not** establish that every one of them should be captured by the referring physician and transmitted as a structured field before physiotherapy, especially when the physiotherapist will reassess them.

### Pain location

Pain location can describe the presentation and may help focus subsequent assessment, but the reviewed major OA management guidelines do not use pain location to select the core rehabilitation programme.

No evidence found in this bounded review shows that medial/anterior/posterior pain location reliably changes first-line Knee-OA exercise treatment.

Therefore pain location currently has:
- clinical descriptive value: YES;
- proven incremental rehab-plan value in this referral workflow: NO;
- proven receiver value: NO.

### Tenderness / crepitus

Joint-line tenderness and crepitus are accepted clinical examination findings and may support diagnostic characterisation.

In the reviewed guidance, neither is used as a main determinant of first-line OA rehabilitation selection.

Therefore their incremental value in a fast physician referral appears low unless an external receiver demonstrates a specific use.

### ROM and objective weakness

These are more actionable.

ROM limitation can justify mobility/flexibility emphasis; objective quadriceps/lower-limb weakness can justify targeted strengthening emphasis. Both are specifically consistent with individualised PT planning and with the current deterministic projection.

These remain stronger candidates for structured physician-entered findings when known.

### Swelling / effusion

Swelling and effusion may reflect symptom irritability or an inflammatory component and can be clinically important.

However:
- clinical effusion assessment has imperfect reliability;
- simple swelling/effusion does not by itself establish a different diagnosis;
- evidence does not support using isolated swelling/effusion as a stand-alone determinant of the core OA rehabilitation plan.

The stronger product value is likely:
1. as contextual severity/irritability information when clearly present; and/or
2. as one component of the already-reviewed atypical-pattern review cue when combined with other concerning observations.

### Instability / major weight-bearing difficulty

These are more actionable than detailed pain maps or crepitus because they can affect safety, gait, balance, assistive-device consideration and the content/progression of rehabilitation.

### Walking aids

NICE explicitly recommends considering walking aids for lower-limb OA. EULAR and other reviewed sources also support walking aids/assistive devices in selected patients.

This supports the Product Owner decision to expose `walking_aid_assessment_and_training` as a default-off additional plan option without automatic selection.

## 3. Provisional utility classification

| Concept | Structured referral value | Reason |
|---|---|---|
| Functional limitations | HIGH | directly changes goals, exercise/function focus and receiver actionability |
| Objective weakness | HIGH | directly supports targeted strengthening |
| Meaningful ROM restriction | HIGH | directly supports mobility/flexibility emphasis |
| Instability / giving-way when clinically established | MODERATE-HIGH | may alter balance/neuromuscular/gait focus and safety |
| Major weight-bearing difficulty | HIGH for review/safety context | affects appropriateness and progression |
| Pain severity / irritability | MODERATE-HIGH | may affect dosing/progression and urgency |
| Simple swelling | CONDITIONAL | context/irritability; low value as isolated treatment selector |
| Objective effusion | CONDITIONAL-LOW | legitimate exam finding but low proven incremental referral value |
| Detailed pain location | LOW-CONDITIONAL | descriptive; no proven change to core OA rehab plan |
| Focal tenderness location | LOW | mainly examination/diagnostic detail; PT will reassess |
| Crepitus | LOW | descriptive/diagnostic; little evidence of rehab-plan impact |
| Detailed atrophy location | LOW-CONDITIONAL | weakness is more directly actionable; prior reviews questioned anatomical granularity |
| Hot/swollen atypical pattern | HIGH as review cue | relevant to alternative pathology review, not routine rehab content |
| Walking-aid need/assessment | MODERATE-HIGH when relevant | evidence-supported device strategy and functional/safety relevance |

## 4. Coordinator recommendation for Product Owner consideration

Do **not** decide the final owner of every current clinical field yet.

First simplify the active structured set.

Candidate direction:

### Keep as structured, prominent or clearly accessible
- Functionality;
- objective weakness;
- meaningful ROM restriction;
- instability/giving-way where established;
- pain severity/irritability if the Product Owner wants a simple severity control;
- major weight-bearing difficulty;
- review-cue observations needed for R2-B;
- walking-aid assessment/training as default-off plan option.

### Strong candidates to remove from the routine physician referral UI
- detailed multi-site pain map;
- focal tenderness map;
- crepitus;
- detailed atrophy localization.

They may remain:
- compatible machine concepts where needed;
- physiotherapist assessment items;
- free-text/manual note when materially relevant.

### Needs Product Owner decision after this review
- whether to retain one simple optional pain-location field at all;
- whether simple swelling deserves one optional clinical-picture control or should only enter through review-cue/context logic;
- whether objective effusion should remain a structured physician field or be left to physiotherapist examination.

## 5. Current decision state

Product Owner has approved:
- R2-A decision 1: one Functionality owner for duplicate functional concepts.
- R2-A decision 3: expose `walking_aid_assessment_and_training` under Proposed Plan → Additional options, default off, no automatic suggestion.

R2-A decision 2 remains **OPEN** pending Product Owner simplification choice informed by this utility review.

No closure review or runtime implementation is authorized yet.
