# CU-1 ulnar-neuropathy-at-elbow route review — 2026-08-29

> **Route:** `ulnar_neuropathy_at_elbow`
> **Review result:** **PASS AS MILD-CONSERVATIVE + NONMILD/SAFETY CONTEXT SPLIT, PENDING EXACT-HEAD FOCUSED CI**
> **Runtime authorization:** NO
> **Manifest activation:** NOT YET at review-write time

## 1. Review question

Can the frozen elbow route `ulnar_neuropathy_at_elbow` receive a current evidence-bounded rehabilitation design without:

- diagnosing cubital tunnel syndrome from symptoms, Tinel sign, elbow-flexion provocation or imaging alone;
- treating subjective paresthesia as objective neurological deficit;
- treating `not_assessed` motor status as normal;
- expanding a small mild-case conservative RCT across moderate/severe or unknown-severity cases;
- turning very-low-certainty night-splint evidence into a default orthosis protocol;
- turning heterogeneous low-quality physiotherapy literature into a mandatory nerve-gliding/manual/electrical programme;
- creating autonomous surgical criteria or procedure choice;
- or silently absorbing cervical, plexus, wrist-level ulnar, structural or traumatic alternative owners?

**Answer:** yes, but only with a narrow context split. A route-wide generic cubital-tunnel physiotherapy sequence would not pass.

---

## 2. Frozen product semantics preserved

The existing frozen elbow profile already distinguishes:

```text
subjective ulnar symptoms
!= objective sensory deficit
!= objective motor deficit
!= formal ulnar neuropathy/cubital tunnel diagnosis
```

Default wording without an explicit clinician diagnosis remains symptom-presentation wording. Positive Tinel/elbow-flexion/neurodynamic findings remain findings, not diagnosis. Progressive motor weakness, intrinsic atrophy or materially worsening objective deficit already require reassessment semantics.

The evidence layer does not reopen that taxonomy.

---

## 3. Current evidence reviewed

### 3.1 Caliandro et al. — Cochrane 2025

**Treatment for ulnar neuropathy at the elbow**

```text
Cochrane Database Syst Rev. 2025;4(4):CD006839
DOI 10.1002/14651858.CD006839.pub5
```

This is the current high-level treatment synthesis. It included 15 randomized/quasi-randomized trials overall, but conservative treatment evidence remains sparse.

For conservative treatment, the review identifies single small trials of education, night splinting, nerve gliding and injections. The clinically relevant conservative signal for this CU-1 rehabilitation design is narrow:

```text
mild UNE
+ information about movements/positions to avoid
→ may reduce subjective discomfort
```

This does **not** establish a universal splint, exercise, nerve-gliding, visit-frequency or total-course protocol.

The review also describes common practice in which severe neurological signs such as objective motor weakness or muscular atrophy move management away from routine conservative-only care. CU-1 uses that only as a safety/reassessment context, not as an autonomous surgical threshold.

### 3.2 Bateman et al. — night-splint systematic review 2025

**Effectiveness of night splints for cubital tunnel syndrome — A systematic review**

```text
Hand Ther. 2025;30(3):105-112
DOI 10.1177/17589983251336157
```

The review found only one controlled RCT of splint vs advice, underpowered and at high risk of bias. Overall evidence certainty was **very low**. The authors concluded that current evidence is insufficient to determine whether night splints should be recommended.

Therefore:

```text
night splint
!= route core
!= validated device type
!= validated elbow angle
!= validated nightly duration
!= validated total course
```

A clinician may separately select an orthosis as clinician instruction, but CU-1 must not relabel that as strong route evidence.

### 3.3 Wolny et al. — physiotherapy systematic review 2022

**The Effects of Physiotherapy in the Treatment of Cubital Tunnel Syndrome: A Systematic Review**

```text
J Clin Med. 2022;11(14):4247
DOI 10.3390/jcm11144247
```

Eleven heterogeneous studies / 187 participants were included, with substantial risk-of-bias limitations. The review could not recommend one best physiotherapy method.

Therefore nerve gliding/neurodynamic techniques, manual therapy and electrical modalities remain non-core unless a future route-specific reviewed claim supports them. Positive case-series signals are not promoted into mandatory protocol authority.

### 3.4 AANEM neuromuscular-ultrasound guideline — published 2022

**Evidence-based guideline: Neuromuscular ultrasound for the diagnosis of ulnar neuropathy at the elbow**

```text
Muscle Nerve. 2022;65(3):255-263
DOI 10.1002/mus.27460
```

For patients with symptoms and signs suggestive of UNE, the guideline gives a **Level B** recommendation to offer ultrasound measurement of ulnar-nerve CSA or diameter to help confirm diagnosis and localize compression.

Critical boundary:

```text
ultrasound = diagnostic adjunct
ultrasound != replacement for clinical/EDX evaluation
imaging abnormality != automatically symptomatic diagnosis
```

This is clinician-UI diagnostic authority, not rehabilitation-treatment authority.

### 3.5 Collins et al. — diagnostic Delphi, published 2025

**Cubital Tunnel Syndrome: Does a Consensus Exist for Diagnosis?**

```text
J Hand Surg Am. 2025;50(2):230.e1-230.e7
DOI 10.1016/j.jhsa.2023.05.014
```

Expert consensus identified clinically relevant candidate criteria including ulnar-distribution paresthesia, flexion-provoked symptoms, Tinel sign, late motor findings and sensory loss. The authors explicitly state that weighting and validation remain necessary before a formal diagnostic scale exists.

Therefore the Delphi cannot become an autonomous CU-1 diagnostic engine.

---

## 4. Final route architecture

### Context A — explicitly mild, sensory-predominant conservative candidate

```text
rep_une_mild_sensory_predominant_v1
→ seq_une_mild_conservative_v1
→ sequence_complete_evidence_bounded
```

Applicability requires all of the following to be explicit/documented:

```text
explicit mild clinical context
objective ulnar motor status actually assessed without material deficit
no intrinsic atrophy or clawing
no unresolved alternative localization or structural owner
```

The sequence contains one required phase:

```text
education
+ individualized reduction/modification of documented provoking positions/movements
```

The sequence deliberately contains:

```text
progression_criteria: []
```

No splint protocol, nerve-gliding protocol, exercise dose, visit frequency, total duration, numeric progression criterion or discharge threshold is manufactured.

### Context B — severity not explicitly mild / motor status unresolved / explicit nonmild without safety trigger

```text
rep_une_nonmild_or_severity_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked_evidence_gap
```

This includes the important regression:

```text
objective motor status = not_assessed
!= normal motor examination
!= mild conservative eligibility
```

An explicitly moderate presentation does not inherit the small mild-case RCT as a complete criteria-based PT sequence. Night-splint uncertainty and heterogeneous physiotherapy literature are not used as gap fillers.

### Context C — progressive motor / atrophy / structural / localization safety context

```text
rep_une_progressive_motor_or_structural_safety_v1
→ rehabilitation_sequence_id: null
→ routine sequence blocked
```

Examples:

```text
progressive objective intrinsic weakness
intrinsic-hand atrophy
ulnar clawing / established late motor sign
materially worsening objective neurological deficit
material trauma or structural compression concern
observed nerve instability/subluxation requiring explicit evaluation
cervical / lower-trunk / other localization concern
```

CU-1 triggers reassessment/correct ownership. It does not select a surgical procedure or manufacture a surgical threshold.

---

## 5. Diagnosis / wording boundary

The route preserves the frozen wording model:

```text
formal_ulnar_neuropathy_cubital_tunnel_diagnosis != yes
→ symptom-presentation wording only

formal_ulnar_neuropathy_cubital_tunnel_diagnosis = yes
→ clinician-entered diagnostic wording may be carried
```

Neither formal diagnosis nor positive EDX/ultrasound automatically resolves rehabilitation severity. A formal diagnosis with severity not stated and motor status not assessed remains in the blocked/unresolved profile.

---

## 6. Evidence-strength governance

PASS requires the following exact distinctions:

```text
AANEM Level B diagnostic ultrasound recommendation
!= treatment-effect certainty

2025 night-splint GRADE very low
!= recommend splint
!= do-not-offer splint

small mild-case education RCT signal
!= moderate/severe authority
!= universal conservative-care bundle

heterogeneous physiotherapy case/RCT literature
!= best physiotherapy method

current management practice for severe neurological signs
!= autonomous evidence-derived surgical threshold
```

No cross-framework synthetic strength is created.

---

## 7. Regression fixtures

Normative route fixture extension:

```text
clinic_utilities/contracts/cu1_ulnar_elbow_fixtures_v1.yaml
```

It verifies:

- explicit mild + assessed motor status resolves the narrow sequence;
- `not_assessed` motor status blocks mild-profile resolution;
- explicit moderate does not inherit mild-case education authority as a complete sequence;
- subjective paresthesia does not become objective deficit or formal diagnosis;
- positive Tinel/flexion findings do not create formal diagnosis;
- Delphi consensus does not become a validated diagnostic scale;
- ultrasound remains Level-B diagnostic adjunct, not autonomous diagnosis;
- progressive motor weakness/atrophy/clawing blocks routine sequence;
- night-splint evidence remains very low and nondefault;
- nerve gliding/manual/electrical modalities are not auto-selected;
- cervical/plexus/wrist localization and trauma/instability contexts fail closed;
- formal diagnosis does not bypass severity/motor-status gates;
- missing history remains missing.

---

## 8. Review gates

```text
frozen route taxonomy preserved                         PASS
diagnosis-vs-finding semantics                         PASS
subjective-vs-objective neurological semantics         PASS
not_assessed != normal                                 PASS
current evidence freshness                             PASS
AANEM diagnostic grade preserved                       PASS
night-splint very-low certainty preserved              PASS
mild evidence not expanded across severity             PASS
no best physiotherapy modality invented                PASS
no generic peripheral-nerve fallback                   PASS
structural/localization owner boundaries               PASS
no numeric progression/discharge threshold             PASS
route-history prompts                                  PASS
matching regression fixtures                           PASS
runtime authorization                                  NO
```

## 9. Result

**PASS AS MILD-CONSERVATIVE + NONMILD/SAFETY CONTEXT SPLIT, pending exact-head focused CI.**

If focused CI passes, the shard and fixture extension may be activated in the CU-1 manifest/coverage matrix. Activation does not authorize runtime evidence-aware generation and does not authorize PR #63 merge.
