# CU-1 Medial Elbow Tendinopathy Route Review — 2026-08-29

## Decision

```text
ROUTE: medial_elbow_tendinopathy
RESULT: PASS — SINGLE-PHASE LOW-CERTAINTY EVIDENCE-BOUNDED ROUTE
RUNTIME AUTHORIZED: NO
```

This review activates a medial-specific evidence profile only if the accompanying route shard and regression fixtures pass the repository focused tests. It does not authorize runtime evidence-aware recommendation generation.

## Frozen route reviewed

```text
clinic_utilities/physio_profiles/elbow_v1_1.md
route_id: medial_elbow_tendinopathy
```

Frozen semantic boundaries remain authoritative:

```text
medial elbow pain != automatic medial elbow tendinopathy
subjective ulnar paresthesia != objective ulnar neurological deficit
provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not_assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried but not inferred
```

## Current evidence review

### 1. Medial-specific rehabilitation synthesis

See ZH, Loo CE, Jaafar Z. *Eccentric exercise therapy for medial epicondylitis: A systematic review of clinical outcomes.* Complement Ther Med. 2026;98:103364. DOI `10.1016/j.ctim.2026.103364`.

The review included five clinical studies totaling 143 patients. Eccentric flexor-pronator exercise was associated with within-group pain/function improvement, with between-group superiority in only one randomized study. Protocol heterogeneity and small samples prevented meta-analysis. The authors explicitly characterize the overall certainty as low and call for larger high-quality trials.

Normative interpretation:

```text
eccentric loading may be considered
!= mandatory eccentric-only programme
!= high-certainty efficacy claim
!= universal superiority over other loading modes
!= universal dose/frequency/duration
```

### 2. Clinical-context rehabilitation and diagnostic review

Konarski et al. *Current concepts of natural course and in management of medial epicondylitis: a clinical overview.* Orthop Rev (Pavia). 2023;15:84275. DOI `10.52965/001c.84275`.

This review supports the clinical context of activity/load modification and rehabilitation but repeatedly acknowledges that medial epicondylitis is less studied than lateral epicondylitis and that physiotherapy evidence is heterogeneous/inconclusive.

Normative interpretation:

```text
narrative clinical management context
!= comparative treatment-effect estimate
!= validated staged rehabilitation protocol
!= authority to borrow lateral-elbow CPG grades
```

### 3. Differential diagnosis / medial-owner boundary

Cho et al. *Ultrasonographic differential diagnosis of medial elbow pain.* Ultrasonography. 2024;43(5):299-313. DOI `10.14366/usg.24102`.

The review preserves alternative medial-elbow diagnoses including ulnar neuropathy, snapping triceps, UCL injury, medial antebrachial cutaneous neuropathy and joint disease.

Normative interpretation:

```text
medial pain + flexor-pronator findings
!= proof that every medial symptom is tendon-only
```

## Search/freshness result

At this review date, no current medial-specific clinical-practice guideline with graded rehabilitation recommendations equivalent to the 2022 APTA/JOSPT lateral-elbow CPG was identified. This absence is treated as an evidence limitation, not permission to import lateral-elbow recommendation grades by analogy.

The 2026 medial-specific systematic review is the primary treatment-effect authority for exercise in this route.

## Exact gate review

### A. Route identity / frozen taxonomy — PASS

No route ID or frozen elbow taxonomy was changed.

### B. Diagnosis-vs-finding separation — PASS

`met_diagnostic_boundary_not_single_finding` prevents local tenderness, resisted wrist-flexion/pronation pain, gripping pain or imaging abnormality from autonomously creating a formal diagnosis.

### C. History specificity — PASS

Dedicated prompts cover:

```text
symptom duration/course
flexor-pronator / grip load behavior
work/sport/manual/throwing-valgus context
ulnar-neural symptoms
prior treatment/response
patient-priority task
```

Missing history remains missing.

### D. Treatment-effect certainty — PASS

The 2026 systematic review remains `low` certainty. The route does not upgrade the evidence because all included studies improved within groups or because one randomized comparison favored an eccentric-containing intervention.

### E. Lateral-to-medial evidence leakage — PASS

The following lateral-elbow objects are explicitly forbidden as medial authority unless independently reviewed for medial applicability:

```text
lateral_elbow_resisted_wrist_extensor_exercise
lateral_elbow_high_demand_phased_reintroduction
lateral Grade-B manual-therapy authority
lateral Grade-B dry-needling authority
lateral Grade-B rigid-taping authority
lateral Grade-F orthosis authority
```

No lateral CPG grade is imported into this route.

### F. Rehabilitation sequence — PASS

The route uses one evidence-bounded phase:

```text
seq_medial_elbow_evidence_bounded_v1
→ activity/load modification as narrative clinical context
→ consider eccentric flexor-pronator loading with low-certainty authority
→ no mandatory loading mode
→ no universal dose
→ progression_criteria: []
```

A multi-phase scheme from narrative reviews is not treated as validated progression evidence.

### G. Progression / return thresholds — PASS

No evidence source reviewed here establishes a validated numeric transition criterion, universal return-to-work/sport threshold or fixed course duration. None is manufactured.

### H. Adjunct governance — PASS

Manual therapy, dry needling, taping, orthosis and ESWT remain clinician-selectable only through a separate reviewed authority or clinician-instruction path. They are not automatically labelled medial-route evidence by analogy with lateral epicondylalgia.

### I. Ulnar-neural / structural differential — PASS

Subjective ring/small-finger paresthesia remains a history symptom and does not create objective deficit or formal ulnar-neuropathy diagnosis. Progressive objective motor deficit, material valgus/UCL instability, substantial mechanical block, major trauma or other discordant findings require reassessment/correct-owner behavior rather than routine tendon reassurance.

### J. Fixture coverage — PASS pending CI

`cu1_medial_elbow_fixtures_v1.yaml` covers:

```text
medial-specific profile resolution
single finding/imaging != diagnosis
low-certainty eccentric evidence preservation
no lateral CPG grade borrowing
subjective ulnar symptoms != objective neuropathy
progressive ulnar motor deficit -> reassessment
valgus/UCL context -> reassessment/correct owner
activity modification != fixed rest protocol
narrative phases != validated progression model
adjuncts not auto-evidence-authorized
missing history remains missing
```

## Evidence gaps preserved

```text
no current medial-specific graded rehabilitation CPG identified
2026 eccentric evidence low certainty / small / heterogeneous
no universal superior loading mode or dose
no validated numeric progression criterion
no universal RTW/RTS clearance threshold
no fixed PT course duration
no automatic medial authority for lateral-elbow adjunct grades
```

## Activation condition

Activate `cu1_evidence_route_coverage_medial_elbow_v1.yaml` only after focused CI succeeds on this exact review head. Then reconcile manifest, coverage matrix, `CURRENT_OPERATIONAL.md`, `SLICE_PLAN_CURRENT.md`, append-only changelog and PR #63 metadata.

## Final review result

```text
PASS
```

The route is suitable for **single-phase, low-certainty, evidence-bounded design authority**. This PASS does not claim that eccentric loading is proven superior, does not authorize a universal medial-elbow rehabilitation protocol, and does not authorize runtime generation.
