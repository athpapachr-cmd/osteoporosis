# CURRENT_OPERATIONAL.md — Step 6A Knee-OA prototype product-owner refinement

> **STATUS:** STEP 6A PRODUCT-OWNER QUALIFIER REFINEMENT — ACTIVE.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-4 parent:** `4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6`.
> **Step-5 tested substantive head:** `6595bf4cc41388dbd796f9ba5b53ae7c49bafdee`.
> **Step-5 closeout head:** `c3f79a192a3fcac9fb11a5245df4e93312c4522a`.
> **Branch:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11`.
> **Active slice:** `CU1-PRODUCT-KNEE-OA-PROTOTYPE-V1-1-QUALIFIERS-20260911`.
> **ACTIVE RUNTIME / DESIGN / CANONICAL WRITER:** this bounded Step-6A refinement session.
> **PR / merge / deploy / production smoke authority:** NONE.
> **Production API/UI/database/config/secrets/patient-data mutation:** NONE.

## 1. Trigger

Product-owner testing found that the first prototype was too summary-level in three clinically important places:

```text
generic pain without location
generic weakness without objective / quadriceps / atrophy refinement
stiffness without pattern and without a distinct fixed-flexion-deformity examination finding
```

The product owner approved a bounded refinement that increases clinical meaning without converting the routine screen into a long form.

## 2. Exact authorized refinement

Synthetic Knee-OA prototype only:

- pain location progressive disclosure, including medial/lateral joint line, anterior/peripatellar, posterior, diffuse and pes-anserine region;
- symptom location remains symptom/location and must not autonomously diagnose pes-anserine bursitis;
- weakness refinement for subjective/generic, objective, quadriceps and visible atrophy context, preserving `generic weakness != objective weakness`;
- stiffness refinement for morning / after inactivity, with morning-duration qualifier `≤30 min | >30 min`;
- `>30 min` becomes a non-blocking review clue, not a treatment command;
- fixed flexion deformity remains an examination finding, separate from stiffness, with optional degrees;
- compact examination qualifiers for extension lag, effusion and focal tenderness, including pes-anserine tenderness;
- completed qualifiers collapse into one-line clinical summaries rather than remaining as permanent fields;
- deterministic referral wording may become more specific, but clinician-selected treatment remains separate from symptom/exam detail;
- existing Step-2 evidence states, Step-3 treatment-selection rules and inherited CU-1 safety/validation authority remain unchanged.

## 3. Evidence boundary for this refinement

NICE NG226 (2022) defines the typical clinical OA diagnostic pattern as activity-related pain with no morning stiffness or morning stiffness lasting no more than 30 minutes in people 45 or older. The prototype may therefore surface `>30 min` as an atypical-feature review clue; it must not infer an alternative diagnosis or block export solely from that clue.

Published knee-OA literature supports attention to anserine-region tenderness/pain as a potentially relevant extra-articular finding, but the prototype must not convert a location/tenderness tap into a diagnosis of pes-anserine bursitis.

This is a bounded clinical-UX refinement, not a full renewed Knee-OA evidence review.

## 4. Preserved Step-5 state

The existing prototype remains loopback-only, synthetic-only and not registered with production. The actual CU-1 engine remains validation/safety authority. No patient persistence, analytics, AI request, production router, second diagnosis, account/billing work or public hosting is introduced.

## 5. Exact next action

Implement the approved qualifiers, update focused adapter/browser tests, then run the same `Physio Knee OA prototype gate` at the exact refined head. If clean, close the writer and return to product-owner testing before independent multi-axis review.
