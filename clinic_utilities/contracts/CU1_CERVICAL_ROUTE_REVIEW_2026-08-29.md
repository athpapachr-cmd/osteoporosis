# CU-1 Cervical Route Review — 2026-08-29

> **Writer:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime mutation:** not authorized and not performed.  
> **Cervical evidence shard:** `clinic_utilities/contracts/cu1_evidence_route_coverage_cervical_v1.yaml`  
> **Review mode:** exact route/reference/applicability/output-scope/freshness review.

---

## C1 — Nonspecific / mechanical neck pain

Canonical route:

```text
nonspecific_neck_pain
```

Frozen cervical design uses this route for axial/mechanical neck pain without a more specific neurological, headache, dizziness or traumatic pathway. The utility does not infer a structural diagnosis and does not convert missing safety information into reassuring negatives.

### Current sources reviewed

1. El-Allawy A, Hecht N, Luedtke K, Schleicher P, Weidner N, Koetter T / DEGAM-AWMF. **Clinical Practice Guideline: Nonspecific Neck Pain.** Dtsch Arztebl Int. 2025;122(20):552-557. DOI `10.3238/arztebl.m2025.0119`. S3 guideline version 3.0; current guideline validity extends to 2030.
2. Blanpied PR et al. / APTA Orthopedics. **Neck Pain: Revision 2017.** J Orthop Sports Phys Ther. 2017;47(7):A1-A83. DOI `10.2519/jospt.2017.0302`. APTA Orthopedics still lists this as its published Neck Pain CPG at the review date.
3. Chacko N et al. **Manual therapy with exercise for neck pain.** Cochrane Database Syst Rev. 2025;12:CD011225. DOI `10.1002/14651858.CD011225.pub2`. Evidence search current to March 2025.

### Freshness / framework decision

The 2025 DEGAM/AWMF S3 guideline is the primary generic-route authority because its population and route definition directly match nonspecific neck pain.

The 2017 APTA CPG remains relevant but is classification based. Its chronic mobility-deficit recommendations must not be silently promoted into a universal C1 bundle. In particular, older classification-specific support for selected passive modalities does not override newer negative generic recommendations in the 2025 S3 framework.

Therefore:

```text
DEGAM/AWMF 2025 generic nonspecific-neck route authority
!=
APTA 2017 chronic mobility-deficit classification bundle
```

The frameworks remain distinguishable rather than being flattened into a synthetic recommendation.

### Exact recommendation mapping

```text
physical activity / activation
→ strong current DEGAM/AWMF recommendation
→ referral_core

patient education / self-management
→ current DEGAM/AWMF recommendation
→ referral_core

exercise therapy — chronic nonspecific neck pain
→ strong current DEGAM/AWMF recommendation
→ referral_core
→ no single exercise type or physician-prescribed dose frozen

exercise therapy — acute nonspecific neck pain
→ may be offered as activating treatment
→ source position is consensus rather than a comparative effect estimate
→ referral_core

manual mobilization / manipulation
→ optional adjunct only
→ must not replace activation/self-management
→ referral_core only when selected/applicable

soft-tissue treatment
→ chronic selected adjunct only with activating methods
→ not core rehabilitation

mechanical traction — generic nonspecific neck pain
→ negative 2025 generic-route recommendation
→ clinician_ui_only exclusion authority

laser / electrotherapy / ultrasound / kinesiotaping
→ negative 2025 generic-route posture
→ clinician_ui_only exclusion authority

routine immobilization
→ not recommended
→ clinician_ui_only safety/exclusion authority

persistent activity-limiting or progressive symptoms
→ medical reassessment rather than indefinite routine progression
→ referral_core reassessment criterion
```

### Cochrane 2025 manual-therapy interpretation

The 2025 Cochrane review does not justify a mandatory manual-therapy-plus-exercise package for C1. Compared with placebo, manual therapy plus exercise may improve function with little/no pain reduction; evidence is low certainty and studies are predominantly chronic. This supports keeping manual therapy optional rather than making it the physician-prescribed core.

### Rehabilitation-sequence decision

A one-phase evidence-bounded sequence is appropriate:

```text
activation + education/self-management
+ individualized exercise when appropriate
+ optional selected adjuncts
→ no universal numeric progression threshold
→ no fixed visit frequency or total course duration
→ persistent/progressive functional limitation triggers reassessment
```

`progression_criteria` remains intentionally empty because the reviewed sources do not define a universal evidence-based C1 transition threshold.

The guideline's medical imaging reconsideration window is retained as clinician context only and must not be converted into a physiotherapy course-duration rule.

### Route-boundary decision

The following features require a different cervical route or separate assessment rather than generic C1 fallback:

```text
radiating upper-limb / radicular-type symptoms
headache-dominant cervical presentation
cervical/dizziness presentation
post-traumatic / whiplash context
progressive objective neurological deficit
possible myelopathic / cord feature
other structural/red-flag concern
```

### Route-specific history prompts

The C1 shard captures:

```text
duration and course
activity / posture / work-load relationship
prior episodes and treatment response
activity avoidance / self-management context
features suggesting another cervical route
```

### Exact gate

```text
canonical route identity                         PASS
current route-specific source identified         PASS
source freshness                                 PASS
explicit payload IDs                             PASS
source/claim/profile/sequence references         PASS
required profile fields                          PASS
required sequence fields                         PASS
route applicability                              PASS
other-cervical-route leakage                     PASS
output-scope compatibility                       PASS
2017-vs-2025 framework distinction               PASS
passive-modality generic override prevention     PASS
manual-therapy mandatory-bundle prevention       PASS
no generic MSK fallback                          PASS
no invented progression threshold                PASS
no fixed PT frequency/course duration            PASS
route-specific history prompts                   PASS
matching regression fixtures                     PASS after fixture artifact activation

C1 ROUTE PROFILE                                 PASS
C1 REHABILITATION SEQUENCE                       COMPLETE — SINGLE-PHASE EVIDENCE-BOUNDED
RUNTIME AUTHORIZED                               NO
```

### C1 promotion decision

C1 is eligible for active design authority only after the matching cervical fixture artifact is present and focused CI passes on the resulting exact head.

It does not authorize C2-C5 cervical routes.
