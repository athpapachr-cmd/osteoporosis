# SLICE_PLAN_CURRENT.md — Step 6A Knee-OA qualifier refinement

> **STATUS:** PRODUCT-OWNER FEEDBACK REFINEMENT / ACTIVE.
> **Slice:** `CU1-PRODUCT-KNEE-OA-PROTOTYPE-V1-1-QUALIFIERS-20260911`.
> **Branch:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11`.
> **Parent closeout:** `c3f79a192a3fcac9fb11a5245df4e93312c4522a`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Writer:** ACTIVE, bounded to synthetic prototype + supporting canonicals/tests.
> **Release / real clinical use:** NOT AUTHORIZED.

## 1. Problem

The Step-5 routine phenotype was too coarse. `Pain`, `stiffness` and `weakness` were quick but clinically under-specified. The correction must add discriminating information without returning to a conventional long medical form.

## 2. UX rule

```text
broad first tap
→ only then reveal the clinically meaningful qualifier
→ after selection, collapse qualifier controls into a one-line clinical summary
```

A qualifier is justified only when it changes the referral wording, clinical review state, evidence/suggestion interpretation, or physiotherapy usefulness.

## 3. Pain refinement

Routine row remains `Πόνος`.

When selected, reveal multi-select location:

```text
medial_joint_line
lateral_joint_line
anterior_peripatellar
pes_anserine_region
posterior
 diffuse
```

The collapsed summary should read naturally, e.g. `Έσω μεσάρθρια · χήνειος πόδας`.

Hard rule:

```text
pes-anserine pain/tenderness location != autonomous pes-anserine bursitis diagnosis
```

## 4. Weakness refinement

Routine row remains `Αδυναμία`.

Qualifiers:

```text
subjective_or_generic
objective
quadriceps
visible_atrophy
```

If `visible_atrophy` is selected, optionally specify `quadriceps` or `peri_knee_general`.

Existing CU-1 finding IDs are used when they already represent the exact clinician-selected meaning (`objective_weakness`, `quadriceps_weakness`). Generic weakness remains the product phenotype and is never upgraded silently.

## 5. Stiffness refinement

Routine row remains `Δυσκαμψία`.

Qualifiers:

```text
morning
after_inactivity
```

If `morning` is selected, reveal:

```text
≤30 min
>30 min
```

`>30 min` creates a non-blocking clinical-review clue because it is outside the typical NICE NG226 OA diagnostic pattern. It does not infer inflammatory arthritis or any other alternative diagnosis and does not automatically select/alter treatment.

## 6. Examination / power-user layer

Keep these out of the default surface:

```text
extension_lag            → existing CU-1 finding
fixed_flexion_deformity  → product-local examination qualifier + optional degrees
effusion                 → existing CU-1 finding
focal_tenderness         → optional location, including pes-anserine region
```

Stiffness symptom remains distinct from passive/fixed extension deficit.

## 7. Referral projection

The existing frozen Step-3 renderer remains the base. A bounded prototype qualifier overlay may make symptom/examination prose more specific but must not:

- create treatment selections;
- create a diagnosis beyond clinician assertion;
- change evidence state;
- bypass inherited CU-1 validation/safety;
- overwrite manual text;
- introduce patient persistence.

Examples:

```text
pain + pes_anserine_region
→ pain in the pes-anserine region

weakness + quadriceps + visible atrophy
→ quadriceps weakness with visible quadriceps atrophy

stiffness + morning + ≤30 min + after inactivity
→ morning stiffness up to 30 minutes and stiffness after inactivity

fixed flexion deformity 10°
→ explicit examination phrase, separate from stiffness
```

## 8. Suggestion boundary

Symptoms/findings do not automatically select rehabilitation. Existing suggestions remain explicit clinician actions. A more specific selected finding may make an existing suggestion eligible only where the mapping is clinically explicit and traceable.

## 9. Acceptance

Focused tests must prove at minimum:

- qualifiers remain hidden until parent symptom selected;
- collapsed summary updates and survives routine interaction;
- pes-anserine location does not create a bursitis diagnosis;
- generic weakness does not become objective weakness;
- quadriceps/objective options map only when explicitly selected;
- stiffness >30 min creates review note but does not block export by itself;
- fixed flexion deformity stays distinct from stiffness and can carry degrees;
- existing Step-5 exact outputs remain unchanged when no new qualifier is selected;
- actual CU-1 safety block still outranks all qualifier UI;
- manual edit/stale/network/export guards remain intact;
- mobile/forced-colour/focus tests remain green.

## 10. Hold

No second diagnosis, production integration, patient persistence, public hosting, billing/auth, autonomous evidence update, PR/merge/deploy/production smoke or real-patient use.
