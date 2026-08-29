# CU-1 Thumb CMC-1 Osteoarthritis Route Coverage Review — 2026-08-29

> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime mutation:** not authorized and not performed.  
> **Route shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_thumb_cmc1_oa_v1.yaml`  
> **Review mode:** exact source / route specificity / orthosis / exercise / output-scope / progression review.

## Decision

```text
canonical route identity                         PASS
frozen wrist-hand profile consistency            PASS
current guideline status check                   PASS
thumb-base-specific orthosis authority           PASS
hand/thumb exercise authority                    PASS
orthosis-type uncertainty preserved              PASS
orthosis wear schedule kept execution-level      PASS
exercise programme/dose uncertainty preserved    PASS
assessment != progression conversion              PASS after correction
CMC1 vs interphalangeal route boundary            PASS
source / claim / profile references               PASS
explicit payload IDs                              PASS
required profile / sequence fields                PASS
output-scope compatibility                        PASS
no generic hand-OA sequence fallback              PASS
route-specific history prompts                    PASS

THUMB CMC1 OA PROFILE                             PASS
REHABILITATION SEQUENCE                           COMPLETE — SINGLE-PHASE EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                                NO
CU-1 DESIGN-COMPLETE                              NO
```

## Frozen route boundary

Canonical route:

```text
thumb_cmc1_osteoarthritis
```

The frozen wrist/hand profile allows use when thumb CMC-1 OA is clinician-established or carried as the clinician's working diagnosis. It preserves:

```text
positive grind finding alone != OA diagnosis
radiographic OA alone != proof that all current symptoms arise from CMC-1
orthosis != automatically mandatory
thumb CMC-1 OA != interphalangeal/generalized hand OA
```

Frozen rehabilitation directions already include CMC-support orthosis when appropriate, thumb/hand exercise, pinch/grip strategy, joint-protection education, adaptive strategies and load/activity modification. The evidence review does not broaden the diagnosis or route taxonomy.

## Current sources reviewed

1. Kloppenburg et al. / EULAR. **2018 update of the EULAR recommendations for the management of hand osteoarthritis.** Ann Rheum Dis. 2019;78(1):16-24. DOI `10.1136/annrheumdis-2018-213826`.
2. Kolasinski et al. / ACR-Arthritis Foundation. **2019 Guideline for the Management of Osteoarthritis of the Hand, Hip, and Knee.** Arthritis Rheumatol. 2020;72(2):220-233. DOI `10.1002/art.41142`.
3. Thakker et al. **What Are the Most Clinically Effective Nonoperative Interventions for Thumb Carpometacarpal Osteoarthritis? An Up-to-date Systematic Review and Network Meta-analysis.** Clin Orthop Relat Res. 2025;483(4):719-736. DOI `10.1097/CORR.0000000000003300`.
4. Karanasios et al. **Exercise-Based Interventions Are Effective in the Management of Patients with Thumb Carpometacarpal Osteoarthritis: A Systematic Review and Meta-Analysis of Randomised Controlled Trials.** Healthcare. 2024;12(8):823. DOI `10.3390/healthcare12080823`.
5. Tossini et al. **Effect of physical therapy interventions in individuals with primary thumb carpometacarpal osteoarthritis: a systematic review and meta-analysis.** Disabil Rehabil. 2024;46(26):6251-6265. DOI `10.1080/09638288.2024.2325652`.
6. Algar et al. / American Society of Hand Therapists. **Assessment and treatment of nonsurgical thumb carpometacarpal joint osteoarthritis: A modified Delphi-based consensus paper.** J Hand Ther. 2023;36(4):982-999. DOI `10.1016/j.jht.2023.08.008`.

Freshness check:

```text
EULAR recommendations index
→ still lists 2018 hand-OA management update as current hand-OA recommendation

ACR OA guideline status page
→ still lists 2019 ACR/AF OA guideline as current OA guideline
```

A newer generic 2025 EULAR physical-activity update published in 2026 does not replace the thumb-base-specific hand-OA management recommendations reviewed here.

## Guideline mapping

### Education / joint protection / adaptive strategies

EULAR:

```text
education + ergonomic principles + pacing + assistive devices
→ Grade A
→ referral_core
```

This supports physician referral wording requesting joint-protection/activity-modification/adaptive strategies without specifying a therapist's exact teaching script.

### Exercise

EULAR:

```text
exercise for function/strength/pain
→ Grade A
```

ACR/AF:

```text
exercise for hand OA
→ strong recommendation
```

ACR explicitly notes that available evidence does not support one best exercise type or ideal duration/intensity/frequency. This is aligned with CU-1's prohibition on converting therapist execution details into physician prescriptions.

The 2024 exercise meta-analysis included 14 RCTs / 1280 participants and found low-to-moderate-certainty short-term benefit for pain/disability versus no treatment, without sustained comparator differences at mid/long term and without establishing an optimal programme.

The 2025 NMA found clinically important short-term pain reductions with hand exercise and multimodal treatment and short-term grip-strength benefit from hand exercise with moderate/high-confidence network evidence for the specific comparisons used by that analysis.

These syntheses are represented without manufacturing a single combined certainty grade.

## Orthosis mapping

EULAR:

```text
orthosis for thumb-base OA symptom relief
→ Grade A
→ longer-term use advocated
```

ACR/AF:

```text
first-CMC hand orthosis
→ strong recommendation
```

The existence of a strong orthosis recommendation does **not** mean the generator must mandate an orthosis for every referral or specify one universal design.

The 2024 physiotherapy meta-analysis found orthosis benefit versus passive control for pain, grip and pinch with **very-low-certainty** evidence, and found no clear difference between neoprene and thermoplastic orthoses or between short and long thermoplastic orthoses for the reported outcomes.

The newer comprehensive NMA found a clinically important medium-term pain/function signal for a rigid CMC-MCP splint versus control with moderate/high-confidence network evidence for those comparisons.

The route therefore preserves both findings:

```text
orthosis may be a meaningful CMC1 intervention
BUT
exact rigid/soft + short/long design is not a universal physician prescription
```

The rigid-CMC-MCP NMA signal remains therapist execution detail rather than auto-rendered mandatory orthosis design.

Likewise, EULAR's statement advocating longer-term use is not converted into a fixed routine rehabilitation duration or wear schedule. Exact activity/night wear instructions remain individualized execution detail.

## ASHT consensus role

The modified Delphi consensus supports assessment of pain, thumb ROM, grip/tripod pinch, region-specific PROMs, environmental factors and expectations and supports orthosis during painful activities as needed, dynamic stability, education, joint protection, adaptive equipment and functional intervention.

Its role is explicitly **expert consensus**.

A pre-activation audit identified an unsafe possible inference:

```text
recommended assessment measures
!= validated progression criteria
```

The shard was corrected before activation. These measures remain examination/assessment context; no evidence-derived numeric or qualitative transition criterion is created from them.

## Rehabilitation sequence

The route is complete as a one-phase evidence-bounded sequence:

```text
education / pacing / joint protection / adaptive strategies
+
individualized hand-thumb exercise
+
consider CMC-support orthosis when appropriate
```

No second phase is required merely for structural symmetry.

The sequence intentionally contains:

```text
progression_criteria: []
```

because the reviewed literature does not establish a validated universal CMC1 OA rehabilitation progression threshold.

## History prompts

The route adds non-inferential prompts for:

```text
pinch/grip/jar/key/task provocation
dominance and work/home/sport demands
prior orthosis + response
prior exercise/joint-protection/adaptive strategy + response
known adduction contracture / MCP compensation / opposition deficit
imaging context without symptom-attribution inference
```

## Cross-route boundary

Thumb-base-specific evidence must not silently authorize:

```text
interphalangeal_hand_osteoarthritis
or
generalized/multijoint hand-OA-specific output
```

Broad EULAR/ACR hand-OA recommendations can be separately reviewed for those routes, but the CMC1 orthosis profile and thumb-specific intervention evidence do not become a generic hand-OA rehabilitation sequence.

## Final route state

```text
rep_thumb_cmc1_oa_v1
→ PASS

seq_thumb_cmc1_oa_v1
→ COMPLETE — SINGLE-PHASE EVIDENCE-BOUNDED

orthosis
→ referral_core option/support direction
→ exact type, fitting, wear schedule = therapist_execution_detail

exercise
→ referral_core broad direction
→ exact programme/dose = not fixed

progression
→ no evidence-derived universal threshold rendered
```

The next queue item after matching fixtures and manifest/matrix reconciliation is `cervical_routes`.
