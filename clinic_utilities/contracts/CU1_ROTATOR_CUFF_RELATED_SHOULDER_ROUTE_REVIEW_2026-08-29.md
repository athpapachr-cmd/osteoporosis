# CU-1 Rotator-Cuff-Related Shoulder Pain Route Review — 2026-08-29

## Scope

Route: `shoulder.rotator_cuff_related_shoulder_pain`

Primary current authority: Desmeules F, Roy JS, Lafrance S, et al. *Rotator Cuff Tendinopathy Diagnosis, Nonsurgical Medical Care, and Rehabilitation: A Clinical Practice Guideline.* J Orthop Sports Phys Ther. 2025;55(4):235-274. doi:10.2519/jospt.2025.13182.

The reviewed CPG covers adults with shoulder pain with suspected rotator-cuff tendinopathy and adults with established rotator-cuff tendinopathy undergoing nonsurgical management and rehabilitation. Its scope includes rotator-cuff tendinopathy with or without calcification and partial-thickness rotator-cuff tears. It explicitly excludes full-thickness rotator-cuff tears.

## Referral-core findings accepted for CU-1

1. Patient-centred individualized education is appropriate and may address the condition, pain-management options, activity modification and self-management while taking account of goals, beliefs, health literacy and relevant psychosocial context.
2. Active rehabilitation is the initial rehabilitation core. It may include motor-control and/or resistance exercise using individualized loading.
3. Current evidence does not establish one universally superior exercise package or one universally superior high- versus low-load prescription. The CPG identifies uncertainty in the optimal frequency, intensity, type and time parameters.
4. Return to sport, when it is an actual patient goal, may be guided by rotator-cuff/shoulder capacity and load tolerance together with patient-rated readiness and functional-performance measures. This is not converted into a universal numeric clearance threshold or a fixed time-based rule.
5. Manual therapy and taping may be used selectively for short-term symptom benefit but do not replace the active rehabilitation core.

## Clinical-organization decision

CU-1 renders this route using two document/clinical-organization stages:

- active rehabilitation / self-management;
- functional reintegration / maintenance, with sport-specific return content only when relevant.

These stages are **not** represented as an evidence-validated fixed multiphase protocol. The reviewed evidence does not define a universal phase-transition threshold. Progression remains response-, function- and load-tolerance-informed, with execution and dosing retained by the treating physiotherapist.

For CU-1 rollout purposes, the evidence-bounded sequence is therefore considered complete when the above scope and limits are preserved. `sequence_complete` in the route coverage matrix means complete for this constrained referral model; it does not mean that the literature establishes one mandatory protocol.

## Boundaries

- `confirmed_full_thickness_rotator_cuff_tear_nonoperative` remains a separate route and does not inherit this rich plan.
- Calcific-tendinopathy-specific interventions are not borrowed automatically into the generic rotator-cuff-related shoulder-pain referral; the dedicated calcific route remains the correct owner for that treatment-specific context.
- Postoperative rehabilitation remains owned by `postoperative_shoulder_rehabilitation` and patient-/procedure-specific restrictions take precedence.
- Suspected major structural injury, material trauma or a changed diagnostic context requires reassessment/correct route ownership rather than generic RCRSP continuation.

## Reassessment / clinician-only evidence

The 2025 CPG recommends specialist reassessment for severe persistent pain and/or disability despite a maximum of 12 weeks of appropriate nonsurgical care. CU-1 keeps that maximum-12-week recommendation in the clinician evidence layer. It is **not** copied into routine referral prose as a fixed physiotherapy course length or an automatic progression rule.

## Rich-referral output constraints

- no universal exact sets/repetitions/load prescription;
- no fixed treatment-course duration;
- no invented numeric progression or discharge criterion;
- no passive-only plan;
- no automatic calcific-specific modality;
- no full-thickness-tear evidence borrowing;
- Detailed output remains within the CU-1 standard target and the hard GeSY 2000-character ceiling;
- Short and Detailed express the same clinical truth.

## Decision

Accepted for `rich_ready` rollout after exact route review, subject to CI proving:

- evidence-profile identity is `rep_rotator_cuff_related_pain_v1`;
- the route remains isolated from full-thickness tear and calcific-specific treatment authority;
- clinician-only 12-week reassessment evidence does not leak into routine referral text;
- all generated outputs remain within character limits;
- no generic cross-route fallback is introduced.

This review authorizes only the CU-1 route-specific rich-referral projection on the current feature branch. It does not authorize merge or deployment.
