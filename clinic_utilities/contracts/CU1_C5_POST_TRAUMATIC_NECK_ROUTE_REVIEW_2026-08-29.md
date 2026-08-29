# CU-1 C5 post-traumatic neck route review — 2026-08-29

> **Route:** `post_traumatic_neck_pain`
> **Frozen region profile:** `cervical`
> **Review type:** exact evidence/applicability/sequence/history/fixture review before manifest activation
> **Runtime authorization:** NO
> **Result:** **PASS AS TEMPORAL + MECHANISM + SAFETY CONTEXT SPLIT**

---

## 1. Exact design question

The frozen C5 label is intentionally broad:

```text
Μετατραυματική / whiplash-associated αυχεναλγία
```

That label cannot be treated as one homogeneous evidence population.

The exact question for this review is therefore:

> Can CU-1 provide evidence-bounded rehabilitation authority for the material C5 contexts without
> converting every cervical trauma into WAD, without importing generic C1 authority, without
> inventing structural clearance, and without mixing recent and persistent WAD evidence?

Answer after exact review:

```text
YES — only as an explicit context split.
```

---

# 2. Frozen semantic boundary preserved

The existing cervical v1.1 profile already requires C5 to preserve:

```text
injury date / phase
known structural status when relevant
current restrictions / precautions when relevant
```

and forbids silent claims that fracture, instability or other significant structural injury was excluded.

This review strengthens, rather than changes, that frozen taxonomy.

Hard C5 invariants are:

```text
post-traumatic neck pain != generic C1 nonspecific neck pain

whiplash / acceleration-deceleration context
!= every cervical trauma

recent / acute WAD
!= persistent WAD

approximate duration
!= machine-inferred WAD temporal phase

C5 selection
!= fracture/dislocation/instability excluded

WAD grade
!= CU-1 inferred classification

associated post-traumatic headache
!= formal CGH

associated post-traumatic dizziness
!= formal cervical dizziness

associated post-traumatic radiating arm symptoms
!= formal cervical radiculopathy

patient-specific structural/healing restriction
> conflicting generic WAD route default
```

No generic cervical fallback is permitted.

---

# 3. Freshness review

## 3.1 SIRA acute WAD guideline

The current SIRA acute-whiplash resource page, reviewed 2026-08-29, continues to identify the **third edition, December 2014** as the acute-WAD guideline for adults in the first 12 weeks.

The proposed Australian fourth edition remains a **draft**. SIRA's published governance record states that NHMRC did not approve the submitted draft and that further work was required. The draft is therefore **not** used as current normative C5 authority.

Current active C5 acute authority remains:

```text
sira_acute_whiplash_guideline_2014_current
```

with a shortened next-review interval because a replacement guideline remains under development.

## 3.2 Current synthesis context

The review additionally uses:

```text
Bussières et al. 2016 WAD/NAD CPG
Côté et al. 2016 OPTIMa NAD CPG — safety / persistent neurological-sign context
Chrcanovic et al. 2022 WAD exercise systematic review
Muñoz Lazcano et al. 2024 guided neck-specific exercise meta-analysis
Muñoz-Bustos et al. 2025 education + exercise GRADE meta-analysis
Chen et al. 2025 chronic WAD vs nontraumatic neck-pain systematic review
APTA/JOSPT Neck Pain CPG 2017 — WAD classification context
```

No later finalized WAD guideline was found that supersedes the active SIRA third edition at this review date.

---

# 4. Context split required by evidence

The native route shard is:

```text
clinic_utilities/contracts/cu1_evidence_route_coverage_cervical_posttraumatic_v1.yaml
```

After pre-PASS self-review it was corrected from one cross-phase WAD profile into five machine-distinct contexts.

## 4.1 Recent / acute explicit WAD — PASS

```text
rep_c5_recent_whiplash_wad_v1
→ seq_c5_recent_whiplash_wad_v1
→ sequence_complete / single-phase evidence-bounded
```

Required context includes:

```text
explicit whiplash / acceleration-deceleration context
recent_or_acute_whiplash_within_12_weeks
no unresolved material structural injury
no unresolved material neurological/safety concern
physiotherapy considered appropriate by clinician
```

### Active rehabilitation authority

SIRA provides two separate Level-B recommendations:

```text
stay active / maintain usual activity and function
neck-specific exercise
```

The exercise examples include ROM, low-load isometric work, postural endurance and strengthening.

CU-1 may preserve those as broad rehabilitation directions but must not convert the examples into a fixed physician-prescribed exercise dose or mandatory sequence.

### Manual therapy

SIRA places manual therapy among limited-evidence treatments that are not routinely recommended:

```text
manual therapy → Level C
```

Therefore CU-1 represents manual therapy only when selected as an adjunct and does not auto-render a technique or dose.

### Activity restriction and collar recommendations remain distinct

Pre-PASS review corrected a potential evidence-strength hybrid:

```text
reduction of usual activities >4 days
→ SIRA consensus clinical-practice point

immobilisation collar in uncomplicated acute WAD
→ SIRA Level A do-not-use recommendation
```

These remain separate claims. Neither overrides a patient-specific written structural/healing restriction.

---

## 4.2 Persistent explicit WAD — PASS

```text
rep_c5_persistent_whiplash_wad_v1
→ seq_c5_persistent_whiplash_wad_v1
→ sequence_complete / single-phase evidence-bounded
```

Required context includes:

```text
explicit whiplash / acceleration-deceleration context
persistent_whiplash_over_3_months
no unresolved material structural injury
no unresolved material neurological/safety concern
physiotherapy considered appropriate by clinician
```

Current evidence supports an active exercise/self-management direction but not a universal protocol.

The 2022 WAD exercise systematic review found some short- and medium-term signals but described the overall evidence base as weak.

The 2024 guided neck-specific exercise meta-analysis found modest short-term advantages for pain and disability. Its observation that positive study results were more common in programmes longer than six weeks and with at least two sessions per week is **study-context information**, not a validated universal minimum treatment frequency or duration.

The 2025 GRADE meta-analysis of education plus exercise versus either component alone found no important clinical superiority and downgraded all pooled comparisons to:

```text
very_low certainty
```

Therefore CU-1 must not freeze a mandatory education-plus-exercise bundle or declare education alone/exercise alone inferior.

---

# 5. Temporal phase is material and fail-closed

An earlier draft used one WAD profile/sequence for both recent and persistent presentations.

Exact self-review rejected that design because the phase objective could reference both recent and persistent claims while only one set was applicable in a given case.

The corrected model now has separate recent and persistent profiles/sequences.

If whiplash context is explicit but temporal phase is `other_or_unclear_phase` or `not_stated`:

```text
rep_c5_whiplash_phase_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap
```

CU-1 must not infer the phase from vague or approximate duration.

This preserves the global rule:

```text
approximate duration != inferred exact date or exact evidence phase
```

---

# 6. Other post-traumatic cervical pain is not silently WAD

If C5 is selected but explicit whiplash/acceleration-deceleration context is absent and no separate safety block applies:

```text
rep_c5_other_posttraumatic_neck_pain_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap
```

Reason:

The reviewed treatment evidence is predominantly WAD-specific. It cannot safely authorize every other cervical traumatic mechanism.

Forbidden fallbacks:

```text
other cervical trauma → WAD sequence      FORBIDDEN
other cervical trauma → generic C1        FORBIDDEN
```

The route may still carry the clinician's history/findings/goals and explicit instructions, but evidence-derived disease-specific rehabilitation content remains blocked until a matching evidence owner is curated.

---

# 7. Structural / neurological safety block

When there is unresolved concern for:

```text
fracture
dislocation
instability
other significant structural injury
new/progressive objective neurological deficit
possible cord/myelopathic feature
other material post-traumatic safety concern
```

resolution is:

```text
rep_c5_unresolved_posttraumatic_safety_v1
→ rehabilitation_sequence_id: null
→ routine C5 progression blocked
```

The utility does not autonomously decide urgency, imaging or structural clearance.

A clinician-entered/documented structural status is required before uncomplicated WAD rehabilitation authority can resolve.

WAD/QTF grade IV, or equivalent known fracture/dislocation context, cannot use the routine WAD sequence.

---

# 8. Persistent objective neurological signs — source correction

Pre-PASS evidence audit found that the statement:

> persistent neurological signs with disability beyond three months warrant physician investigation/management

is directly supported by the OPTIMa/Côté 2016 grade-III NAD guidance.

The first shard draft had attributed this claim to the Bussières CPG without sufficient exact-source confirmation.

That attribution was corrected before review:

```text
c5_persistent_neurological_signs_require_medical_review
→ evidence_id: optima_nad_cpg_2016
```

The claim is used only as safety/reassessment context and not as generic treatment authority.

---

# 9. Associated symptoms do not import other cervical-route authority

Whiplash commonly coexists with headache, dizziness/imbalance and upper-limb symptoms.

C5 may preserve those symptoms in history, but route selection does not automatically authorize:

```text
C3 formal-CGH synthesis
C4 cervical-dizziness treatment claims
C2 cervical-radiculopathy-specific component NMA
```

Those disease-specific claims require their own explicit matching diagnosis/context.

This prevents post-traumatic symptom overlap from becoming cross-route evidence leakage.

---

# 10. No autonomous WAD grading or referral triage

The SIRA acute framework contains WAD grading, VAS/NDI/recovery-expectation risk stratification and specific review timepoints.

CU-1 is not converted into an autonomous whiplash diagnostic or referral-triage engine.

Therefore:

```text
WAD grade
→ only when clinician entered/documented

low/high recovery-risk category
→ not inferred from missing or partial VAS/NDI/expectation data

SIRA review timepoints / improvement thresholds
→ not automatic PT course duration or phase-transition rules

C5 route selection
→ not proof that PT referral is evidence-indicated for every acute WAD case
```

These framework details may remain clinician context for future separately authorized decision-support design, but are not required physician-generated referral prose in this slice.

---

# 11. Rehabilitation progression model

Both nonblocked C5 sequences are deliberately single-phase and contain:

```text
progression_criteria: []
```

Current reviewed evidence does not establish one universal numeric criteria-based progression threshold for all WAD.

No routine output may manufacture:

```text
fixed visit frequency
fixed six-week or other course duration
mandatory minimum number of supervised sessions
numeric pain/NDI transition threshold
universal return-to-work or return-to-driving threshold
elapsed-time-only progression
```

Reassessment remains function/recovery/safety based, with clinician review for nonprogression or materially concerning change.

---

# 12. Patient-specific protocol precedence

SIRA's uncomplicated-WAD stay-active and no-collar recommendations apply only when no conflicting patient-specific structural/healing restriction exists.

Permanent precedence remains:

```text
explicit patient-specific written structural / healing / orthopaedic restriction
>
conflicting generic WAD route default
```

The route default is suppressed, not blended, and the patient-specific restriction is not relabelled as literature authority.

---

# 13. Regression fixtures

Dedicated oracle:

```text
clinic_utilities/contracts/cu1_c5_post_traumatic_neck_fixtures_v1.yaml
```

Required PASS cases include:

```text
recent explicit WAD → recent profile/sequence + SIRA Level-B claims
persistent explicit WAD → persistent profile/sequence
recent != persistent cross-resolution
unknown WAD phase → blocked evidence gap, no inferred phase
other cervical trauma → no WAD/C1 fallback
structural status not stated → safety block
known structural restriction → patient-specific protocol precedence
WAD IV / equivalent structural injury → no routine WAD sequence
progressive objective neurological/cord concern → block
persistent objective neurological signs + disability → OPTIMa medical-review claim
2024 study frequency/duration != universal prescription
2025 education+exercise very-low certainty != mandatory superior bundle
manual therapy → selected Level-C adjunct only
activity-restriction consensus != collar Level-A evidence
post-traumatic headache/dizziness/arm symptoms != automatic C3/C4/C2 import
route selection != autonomous PT-referral eligibility decision
```

The fixture corpus tests design semantics only and does not authorize runtime selection.

---

# 14. Exact source-scope findings

### SIRA 2014 / current resource

Appropriate for:

```text
acute WAD first 12 weeks
stay-active advice
neck exercise
manual-therapy limited-evidence context
activity-restriction/collar recommendations
acute assessment/recovery monitoring
```

Not appropriate for:

```text
generic non-WAD cervical trauma
persistent-WAD treatment authority beyond its acute scope
patient-specific structural restriction override
```

### Bussières 2016

Appropriate as condition/grade-specific recent/persistent WAD/NAD treatment context. It is not used to manufacture one universal multimodal WAD bundle.

### OPTIMa/Côté 2016

Used for major-pathology safety semantics and persistent grade-III/equivalent objective-neurological-sign review. It is not used as generic WAD exercise authority.

### Chrcanovic 2022

Supports possible exercise benefit while explicitly preserving a weak evidence base.

### Muñoz Lazcano 2024

Supports a modest guided neck-specific exercise signal, but observed study duration/frequency is execution context, not protocol authority.

### Muñoz-Bustos 2025

Preserves very-low-certainty evidence and no clinically important superiority of education+exercise over either component alone.

### Chen 2025

Supports the boundary that chronic WAD and nontraumatic neck pain are clinically distinguishable populations; it does not itself prescribe treatment.

---

# 15. Exact review gate

```text
explicit payload IDs / route identities                    PASS
source identity / freshness                                PASS
SIRA active-vs-draft-fourth-edition status                 PASS
recent-vs-persistent WAD applicability split               PASS
other-trauma evidence-gap behavior                         PASS
structural / neurological fail-closed behavior             PASS
WAD-grade non-inference                                    PASS
patient-specific protocol precedence                       PASS
framework-specific SIRA strength preservation              PASS
persistent-neuro source attribution correction             PASS
no generic C1 fallback                                     PASS
no C2/C3/C4 cross-route evidence leakage                   PASS
no fixed visit-frequency/course-duration invention         PASS
no numeric progression-threshold invention                 PASS
route-specific history prompts                             PASS
matching dedicated fixtures                                PASS
runtime authorization                                      NO
```

---

# 16. Formal result

```text
C5 post_traumatic_neck_pain
→ PASS AS TEMPORAL + MECHANISM + SAFETY CONTEXT SPLIT
```

Normative contexts after activation:

```text
recent explicit uncomplicated WAD
→ rep_c5_recent_whiplash_wad_v1
→ seq_c5_recent_whiplash_wad_v1
→ sequence_complete

persistent explicit WAD
→ rep_c5_persistent_whiplash_wad_v1
→ seq_c5_persistent_whiplash_wad_v1
→ sequence_complete

explicit WAD but temporal phase unclear
→ rep_c5_whiplash_phase_unresolved_v1
→ blocked_evidence_gap

other post-traumatic cervical pain without explicit WAD context
→ rep_c5_other_posttraumatic_neck_pain_v1
→ blocked_evidence_gap

unresolved structural / neurological safety context
→ rep_c5_unresolved_posttraumatic_safety_v1
→ routine sequence blocked
```

This PASS authorizes **manifest activation of the reviewed design objects only** after focused CI succeeds.

It does **not** authorize runtime evidence-aware generation, persistence changes, PR merge, CU-2 or PR-1.

Exact next route after canonical C5 activation/reconciliation:

```text
remaining wrist/hand and elbow routes
```
