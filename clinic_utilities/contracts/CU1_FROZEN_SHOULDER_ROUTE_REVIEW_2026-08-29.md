# CU-1 Primary Frozen Shoulder Route Review — 2026-08-29

## Scope

Route: `shoulder.adhesive_capsulitis_frozen_shoulder`

Rich-referral authority is limited to a clinician-established **primary frozen shoulder / adhesive capsulitis** context. Presentation-only shoulder stiffness, secondary frozen shoulder, post-traumatic stiffness, postoperative stiffness, glenohumeral osteoarthritis and other structural causes remain outside this disease-specific rich projection.

Primary current authorities reviewed:

1. Lee BC, et al. *Clinical Practice Guidelines for Diagnosis and Non-Surgical Treatment of Primary Frozen Shoulder.* Ann Rehabil Med. 2025;49(3):113-138. doi:10.5535/arm.250057.
2. British Elbow and Shoulder Society. *Patient care pathway: Frozen shoulder.* Shoulder Elbow. 2025;17(4):351-363.
3. Lee JH, Jeon HG, Yoon YJ. *Effects of Exercise Intervention with and without Joint Mobilization in Patients with Adhesive Capsulitis: A Systematic Review and Meta-Analysis.* Healthcare. 2023;11:1504.

## Referral-core findings accepted for CU-1

- For **primary** frozen shoulder, manual therapy including range-of-motion exercise may be considered for upper-limb function and shoulder ROM; the Korean 2025 CPG grades this B with low certainty.
- Exercise and exercise combined with joint mobilization may improve ROM and subjective outcomes, but treatment intensity should be individualized to symptoms and ROM rather than tied to a fixed disease stage or timeline.
- The route may therefore support a single evidence-bounded rehabilitation organization focused on symptom-tolerated mobility, function and self-management.

## Evidence that is deliberately not converted into routine referral prescriptions

- Self-stretching is supported only as an adjunct under expert consensus and remains therapist execution detail; CU-1 does not prescribe a universal stretch technique, frequency or duration.
- Shoulder strengthening has very low / insufficient evidence for a routine recommendation. It may be individualized by the treating clinician/physiotherapist but is not made a mandatory referral-core direction.
- BESS 2025 does not establish that supervised physiotherapy is superior to the natural history of frozen shoulder. CU-1 therefore does not promise superiority or imply that one supervised programme is mandatory.
- BESS conditionally supports physiotherapy following injection therapy. This context-specific recommendation remains visible in the clinician evidence panel and is not auto-rendered unless a future explicitly reviewed intervention-context branch is added.

## Clinical-organization decision

The rich Detailed referral uses **one clinical-organization stage**, not a validated freezing/frozen/thawing protocol:

- individualized mobility / ROM / function / self-management.

Progress is described by change in symptoms, functional use and shoulder mobility. There is no universal numeric ROM threshold, fixed phase-transition rule or fixed treatment-course duration.

## Boundaries

- `primary_frozen_shoulder` + `formal_diagnosis` is required for the disease-specific rich plan.
- `presentation` wording does not receive primary-frozen-shoulder treatment authority.
- `secondary_or_other_stiff_shoulder` and `not_stated` fail closed to the pre-existing non-rich formatter path.
- Post-traumatic and postoperative stiffness retain their separate semantic owners.
- Imaging is not used by CU-1 to autonomously establish the diagnosis; the diagnosis remains clinician-entered.

## Decision

Accepted as a **context-gated, evidence-bounded single-phase rich referral** for clinician-established primary frozen shoulder only. `sequence_complete` for this context means complete for the constrained CU-1 referral projection; it does not mean that the literature establishes a universal staged frozen-shoulder protocol.

No merge or deployment is authorized by this review.
