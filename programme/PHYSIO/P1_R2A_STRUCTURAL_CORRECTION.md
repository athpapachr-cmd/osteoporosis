# PHYSIO P1 R2-A Structural IA — Bounded Correction Design

> **TASK:** `PHYSIO-P1-R2A-STRUCTURAL-IA-CORRECTION-20261004`
> **STATUS:** DESIGN CORRECTION ONLY / NO RUNTIME IMPLEMENTATION.
> **BASE:** `2ce4f3a9f2fd6535bf9da2b06c9c37dff1ebbda6`.
> **PURPOSE:** close only the three material R2-A findings.
> **R2-B:** already PASS; safety semantics are preserved and are not redesigned here.

---

# 1. Finding R2-A-1 — duplicated functional concepts

Current contract accepts four function-like findings:

```text
finding.walking_limitation
finding.stairs_limitation
finding.sit_to_stand_limitation
finding.sport_or_exercise_limitation
```

and also the canonical functional-impairment concepts:

```text
functional_impairment.walking_tolerance
functional_impairment.stairs
functional_impairment.sit_to_stand
functional_impairment.sport_gym
```

The current template already aliases the finding forms into the functional forms and deduplicates after aliasing.

## Correction decision

**Single UI owner = Functionality / `functional_impairments`.**

Target UI/state writes:
- `walking_tolerance`;
- `stairs`;
- `sit_to_stand`;
- `sport_gym`.

The four `finding.*_limitation` IDs:
- remain accepted only for backward compatibility / legacy transport;
- remain server-side aliases for deterministic projection;
- are **not** separate selectable controls in the corrected UI;
- must not appear as a second clinical/examination owner.

No CU-1 taxonomy change is required.

Regression requirement:
- legacy finding input and canonical functional input produce one functional output phrase, never duplicate prose;
- new UI writes only canonical `functional_impairments`.

---

# 2. Finding R2-A-2 — pain-location overlap and swelling semantics

## 2.1 Pain location

Current product contains both:
- product-local `pain_locations[]`;
- legacy/specific findings such as `joint_line_pain` and `anterior_peripatellar_pain`.

## Correction decision

**Single UI owner for patient-reported pain location = Clinical Picture → Pain → `pain_locations[]`.**

Rules:
- do not expose `joint_line_pain` or `anterior_peripatellar_pain` as separate competing UI controls;
- retain them only as compatible machine/projection inputs where required;
- multiple genuinely distinct pain locations may remain selectable;
- `diffuse` remains mutually exclusive with focal locations;
- duplicate semantic representations must collapse to one referral phrase.

Objective tenderness remains separate:
```text
reported pain location
!= focal tenderness on examination
```

## 2.2 Swelling / effusion / hot-swollen-joint context

These are three different semantics and must remain three different owners:

### `swelling`
Owner: **Clinical Picture**
Meaning: reported/observed nonspecific swelling.
It is not automatically an intra-articular effusion and is not itself a septic-joint cue.

### `effusion`
Owner: **Examination**
Meaning: clinician-recorded intra-articular effusion / objective examination finding.

### `hot_swollen_joint`
Owner: **Clinical Review Cue input**
Meaning: explicit atypical hot/swollen-joint observation used only by the product-local review-cue layer.

It must not be merged with simple swelling or effusion.

Permanent boundary:
```text
SWELLING != EFFUSION != HOT/SWOLLEN REVIEW OBSERVATION
HOT/SWOLLEN OBSERVATION != SEPTIC ARTHRITIS DIAGNOSIS
```

R2-B trigger/disposition semantics remain authoritative for the review cue and require no CU-1 mutation.

---

# 3. Finding R2-A-3 — walking aid assessment/training

Current state:
- `walking_aid_assessment_and_training` is already a canonical rehab-direction ID;
- evidence contract classifies it `conditional_or_context_dependent`;
- reviewed sources directly support walking aids in selected lower-limb OA patients;
- it is default-off;
- it is currently hidden from the Knee UI;
- evidence contract itself records that presentation scope amendment is the missing step.

## Correction decision

**EXPOSE, do not remove.**

Target owner:
```text
Προτεινόμενο πλάνο
→ Πρόσθετες επιλογές
→ Αξιολόγηση / εκπαίδευση στη χρήση βοηθήματος βάδισης
```

Rules:
- default **off**;
- no automatic selection;
- no automatic suggestion in this correction;
- evidence cue available under the existing evidence interaction model;
- remains subordinate to the active rehabilitation plan;
- selecting it must render the existing deterministic referral phrase;
- no duplicate separate goal is auto-selected;
- it must not return to a generic `Περισσότερα` drawer.

Rationale:
- existing machine ID and deterministic output already exist;
- evidence is direct and contextual;
- there is now a coherent semantic home in the corrected Proposed Plan;
- deleting it from active scope would discard an evidence-supported, receiver-relevant option without evidence that it lacks utility.

---

# 4. Resulting single-owner IA

```text
CLINICAL PICTURE
  Pain
    reported pain + pain_locations
  Stiffness
    symptom pattern/duration
  Weakness
    subjective weakness only
  Swelling
    nonspecific reported/observed swelling

FUNCTIONALITY
  walking_tolerance
  stairs
  sit_to_stand
  sport_gym
  other functional_impairments

EXAMINATION
  objective weakness
  quadriceps weakness
  atrophy
  ROM findings
  crepitus
  focal tenderness
  effusion
  stability

CLINICAL REVIEW CUES
  product-local atypical observations
  including hot_swollen_joint and the R2-B-approved inputs
  → clinician continue/defer disposition
  != diagnosis
  != automatic imaging
  != CU-1 safety block

PROPOSED PLAN
  reviewed default active-rehab plan
  + additional rehab directions
  + walking_aid_assessment_and_training
  + adjuncts as a clearly separate subordinate group

NO GENERIC MISCELLANEOUS MORE
NO HIDDEN SECOND-TAP DETAIL OWNER
NO PARALLEL ΛΕΠΤΟΜΕΡΕΙΕΣ ROUTE
```

---

# 5. Implementation boundary after closure PASS

If and only if one independent R2-A closure review returns PASS, the subsequent bounded implementation may:

- reorganize the existing Knee-OA UI to the single-owner map above;
- remove hidden second-tap detail dependency;
- remove the competing `Λεπτομέρειες` route;
- retire generic `Περισσότερα` as miscellaneous IA;
- write functional concepts only through canonical `functional_impairments`;
- keep legacy finding aliases accepted server-side;
- expose `walking_aid_assessment_and_training` under Proposed Plan / Additional options;
- implement only the already-reviewed R2-B product-local cue semantics;
- preserve current deterministic referral output/evidence/default plan except where the reviewed semantic owner requires deterministic de-duplication.

It may **not**:
- change shared CU-1 taxonomy or safety rules;
- add a second diagnosis;
- add autonomous diagnostic inference;
- auto-order imaging;
- change evidence states/default plan;
- add patient persistence;
- redesign the referral generator.

---

# 6. Required closure review

One independent review only:

`R2-A delta + affected cumulative closure`

Questions:
1. Are all function duplicates now assigned to one UI owner with safe legacy alias handling?
2. Does pain location have one UI owner without losing distinct locations or confusing tenderness?
3. Are swelling, effusion and hot/swollen review context semantically distinct and uniquely owned?
4. Is walking-aid support given one coherent disposition and owner?
5. Did the correction create any new material overlap or shared-Core requirement?

Terminal rule:
```text
ALL THREE ORIGINAL R2-A FINDINGS CLOSED
+ NO NEW MATERIAL RISK
+ AFFECTED OWNER MAP COMPLETE
→ PASS
→ STOP REVIEW CHAIN
```
