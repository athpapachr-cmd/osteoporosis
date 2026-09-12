# Knee OA Step 6A qualifier refinement result

> **Date:** 2026-09-11 Asia/Nicosia  
> **Scope:** product-owner feedback refinement of the synthetic Knee-OA prototype only  
> **Tested substantive head:** `243095ca9545bd2f96be8986520aeae8c3551c27`  
> **Workflow:** `Physio Knee OA prototype gate`  
> **Run:** `34627436841`  
> **Result:** **PASS**  
> **Independent review:** NOT PERFORMED

## Product-owner problem addressed

The first functional prototype was judged too summary-level in three areas: pain without useful location, weakness without clinically meaningful specificity/atrophy context, and stiffness without separation from fixed loss of extension.

The refinement preserves the small routine surface and implements progressive disclosure:

```text
broad first tap
→ clinically useful qualifier on demand
→ compact summary in the main flow
```

## Implemented refinements

### Pain

Optional location now includes medial/lateral joint line, anterior/peripatellar, posterior, diffuse and **pes-anserine region / χήνειος πόδας**. Multiple focal locations can coexist; diffuse is exclusive. Location refines the referral prose but never creates an autonomous diagnosis of pes-anserine bursitis.

### Weakness

Generic weakness remains a subjective/product phenotype unless the clinician explicitly selects `objective` or `quadriceps`. Visible atrophy can be added and optionally localized to quadriceps or peri-knee region. Explicit quadriceps weakness can refine already-selected strengthening toward quadriceps emphasis; it does not select strengthening on its own.

### Stiffness

Stiffness can be refined to morning and/or after inactivity. Morning stiffness can be further labelled `≤30′` or `>30′`. `>30′` creates a non-blocking review clue linked to NICE NG226 rather than a treatment command or alternative diagnosis.

### Examination / power-user layer

`Extension lag`, `effusion`, fixed flexion deformity with optional degrees, and focal tenderness remain in the advanced examination layer. Fixed flexion deformity is explicitly separate from the stiffness symptom. A clinician-selected FFD can make the existing mobility suggestion eligible through the documented clinical mapping but does not select treatment.

## Safety / semantic invariants preserved

- symptom location != diagnosis;
- pes-anserine pain/tenderness != automatic bursitis;
- generic weakness != objective weakness;
- stiffness != fixed flexion deformity;
- review clue != treatment selection;
- suggestion != selection;
- clinician selection remains required for referral-plan wording;
- inherited CU-1 validation/safety remains authoritative;
- no patient persistence / analytics / AI request introduced;
- no production route, DB, config or secret changed.

## Exact technical evidence

The successful run passed:

```text
scope guard                                      PASS
Python + JS syntax                               PASS
existing real-CU1/HTTP tests                     15 / 15 PASS
existing exact frozen Greek output fixtures      15 PASS inside existing suite
new qualifier projection tests                    8 / 8 PASS
existing real Chromium tests                     12 / 12 PASS
new real Chromium qualifier tests                 6 / 6 PASS
Greek source-summary display coverage             54 positions
packaged real-CU1 dependency closure              PASS
```

Two useful failures occurred during refinement and were corrected rather than hidden:

1. the first qualifier mapping revision removed legacy explicit `objective_weakness`, `quadriceps_weakness` and `tenderness` when no new qualifier owned that semantic group; compatibility tests caught it and the mapping was corrected to preserve prior findings unless an explicit qualifier replaces them;
2. the first new browser test used string-eval polling, which the prototype's strict CSP correctly rejected. The test harness was corrected to use DOM locator assertions; CSP was not weakened.

## Remaining limits

This PASS proves the bounded synthetic implementation and its tested browser behavior in Chromium. It does not establish clinical validation of the entire product, Safari/VoiceOver conformance, real-patient usability, commercial readiness or independent review.

## Next boundary

Return to **Step 6 product-owner synthetic usability / clinical-copy review** of this refined prototype. The next planned engineering action should be driven by observed usability/clinical-copy findings, not by adding more fields pre-emptively. Independent clinical / physiotherapy / UX / commercial review remains after product-owner acceptance of the functional candidate.
