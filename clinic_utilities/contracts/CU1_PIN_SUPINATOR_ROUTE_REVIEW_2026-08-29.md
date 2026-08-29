# CU-1 PIN / supinator route exact evidence review — 2026-08-29

> **Route:** `posterior_interosseous_nerve_supinator_syndrome`  
> **Branch:** `design/cu1-history-evidence-timeline-2026-08-28`  
> **Runtime authorization:** NO  
> **Review classification:** **PASS PENDING EXACT-HEAD CI — explicit evidence-gap + safety/context split; no route-specific rehabilitation sequence authorized**

---

## 1. Review question

Can the frozen elbow route `posterior_interosseous_nerve_supinator_syndrome` receive a current, route-specific, evidence-bounded physiotherapy sequence without conflating:

- pain-predominant radial tunnel syndrome with motor PIN neuropathy/palsy;
- spontaneous non-traumatic/non-compressive palsy with demonstrable entrapment/compression;
- motor-pattern localization with autonomous diagnosis;
- conservative management with a specific physiotherapy programme;
- source-specific recovery/surgical-review timing with a universal rehabilitation threshold?

Answer after exact review: **No complete route-specific PT sequence is supported. The route can nevertheless close its evidence-coverage obligation through explicit context-specific evidence-gap and safety behavior.**

---

## 2. Frozen taxonomy preserved

The existing `elbow_v1_1.md` route deliberately separates motor PIN/supinator syndrome from a pain-predominant radial-tunnel presentation.

Frozen invariants retained:

```text
lateral forearm pain alone != PIN/supinator syndrome
radial-tunnel provocation alone != motor neuropathy
objective radial/PIN motor deficit != routine epicondylalgia
new or progressive motor weakness -> medical/specialist reassessment semantics
```

A 2025 nomenclature commentary proposes that deep radial-branch compression may be understood as a pain-to-palsy spectrum. This is useful terminology context but is not a treatment-effect guideline and is not sufficient reason to rewrite the frozen CU-1 route taxonomy during this evidence-hardening slice. The 2024 radial-tunnel diagnostic systematic review also documents substantial heterogeneity in how pain-predominant RTS has been diagnosed. The safe CU-1 rule therefore remains: pain-only/radial-tunnel findings do not silently become motor PIN syndrome.

---

## 3. Current evidence reviewed

### 3.1 Kato et al. 2026 — prospective multicenter spontaneous PIN palsy cohort

`Clinical characteristics and results after conservative treatment or interfascicular neurolysis of 58 limbs with spontaneous posterior interosseous nerve palsy: A prospective Japanese multicenter study`  
Journal of Orthopaedic Science, online 2026-03-07. DOI `10.1016/j.jos.2026.02.010`.

Important population boundary:

```text
spontaneous PIN palsy
= non-traumatic
+ non-compressive / non-entrapment
+ no space-occupying lesion
```

The study followed 58 limbs. Good recovery occurred in 31/34 conservatively managed limbs and 19/24 limbs treated with interfascicular neurolysis. The authors concluded that conservative management is advisable initially in this spontaneous population and used motor-recovery trajectory to inform later neurolysis consideration.

Normative interpretation:

```text
supports conservative-management course context
!= specific physiotherapy intervention evidence
!= nerve-gliding protocol
!= splint protocol
!= exercise prescription
!= fixed PT frequency or duration
!= universal six-month surgical threshold
```

The six-month recovery signal is specific to the studied spontaneous PIN palsy cohort and is not converted into a universal CU-1 elapsed-time progression rule, PT course duration or autonomous surgery decision.

### 3.2 Huisstede et al. 2006 — intervention systematic review

`Interventions for treating the posterior interosseus nerve syndrome: a systematic review of observational studies`  
J Peripher Nerv Syst. 2006;11(2):101-110. DOI `10.1111/j.1085-9489.2006.00074.x`.

The review found no randomized or controlled clinical trials of PINS treatment. Only case series were available; the effectiveness of conservative treatment was unknown because no higher-quality studies were available.

Despite its age, this remains important route-specific evidence-gap authority because the contemporary search did not identify a later high-quality controlled rehabilitation evidence base that establishes a validated PIN physiotherapy sequence.

### 3.3 McGraw 2019 — spontaneous PIN palsy management review

`Isolated spontaneous posterior interosseous nerve palsy: a review of aetiology and management`  
J Hand Surg Eur Vol. 2019;44(3):310-316. DOI `10.1177/1753193418813788`.

The review separates compressive and non-compressive palsy. Demonstrable compression changes management ownership; non-compressive spontaneous pathology may receive conservative management first. It does not supply a validated physiotherapy progression protocol.

### 3.4 Hones et al. 2024 — radial-tunnel diagnostic systematic review

`Establishing the diagnosis of radial tunnel syndrome: a systematic review of published clinical series`  
Eur J Orthop Surg Traumatol. 2024;34(6):2813-2821. DOI `10.1007/s00590-024-04003-8`.

The literature uses heterogeneous RTS definitions, and abnormal EMG/NCS was uncommon in the reviewed pain-predominant RTS series. This supports preserving the pain-predominant RTS vs motor PIN boundary rather than creating diagnostic or treatment leakage.

### 3.5 Braun et al. 2025 — radial-nerve nomenclature commentary

`Nomenclature of the radial nerve: distinguishing between the deep branch of the radial nerve and the posterior interosseous nerve`  
J Hand Surg Eur Vol. 2025;50(1):122-123. DOI `10.1177/17531934241254706`.

The authors advocate a single compression syndrome spanning pain to palsy. This is current terminology context, not a graded clinical-practice guideline. CU-1 records the controversy but does not use it to erase the frozen clinical distinction or to import pain-only evidence into motor PIN output.

### 3.6 Electrodiagnostic / differential context

Bevelaqua et al. (`HSS J. 2012;8(2):184-189`, DOI `10.1007/s11420-011-9238-8`) describes the characteristic finger/thumb-extension weakness with preserved wrist extension and the role of electrodiagnostic testing for localization and exclusion of cervical, plexus or other peripheral lesions.

Baima & Heise 2025 specifically illustrates that selective radial-pattern weakness can also be encountered in multifocal motor neuropathy and that electrodiagnostic localization matters. These sources are diagnostic/differential context only, not treatment-effect authority.

### 3.7 Bartoletti et al. 2025 — broad elbow peripheral-nerve review

The review discusses physical therapy, splinting, nerve glides and other conservative modalities across elbow nerve syndromes. Its broad narrative recommendations do not establish route-specific comparative effectiveness for motor PIN palsy. It therefore cannot upgrade those modalities into mandatory or superior PIN referral core.

---

## 4. Exact context model

### A. Pain-predominant radial-tunnel / lateral-forearm presentation without material PIN-pattern motor deficit

```text
rep_pin_pain_only_or_radial_tunnel_mismatch_v1
→ rehabilitation_sequence_id: null
→ explicit route mismatch / blocked evidence gap
```

The frozen PIN route does not autonomously absorb pain-only RTS.

### B. PIN-pattern motor presentation with unresolved diagnosis, etiology or localization

```text
rep_pin_motor_presentation_unresolved_v1
→ rehabilitation_sequence_id: null
→ blocked evidence gap
```

Motor pattern may support localization but does not determine etiology, compression site or formal diagnosis.

### C. Explicit clinician-established spontaneous non-traumatic/non-compressive PIN palsy without space-occupying lesion

```text
rep_pin_spontaneous_noncompressive_established_v1
→ rehabilitation_sequence_id: null
→ conservative-management context supported
→ specific PT sequence remains evidence gap
```

The 2026 cohort supports an initial conservative-management strategy but does not define an evidence-bounded physiotherapy programme that satisfies CU-1 element-level provenance requirements.

### D. Demonstrable compression/entrapment, space-occupying lesion, trauma, iatrogenic/structural cause, or progressive/materially worsening motor deficit

```text
rep_pin_compressive_structural_or_progressive_v1
→ rehabilitation_sequence_id: null
→ safety / specialist / correct-owner behavior
```

CU-1 does not generate surgical timing or procedure choice. Patient-specific structural/postoperative restrictions retain precedence.

---

## 5. Diagnostic and semantic boundaries

PASS requires all of the following:

```text
finger/thumb extension weakness with preserved wrist extension
!= autonomous formal PIN diagnosis

formal PIN diagnosis
!= spontaneous etiology
!= supinator/arcade-of-Frohse compression

lateral forearm pain
!= PIN motor neuropathy

radial-tunnel provocation
!= motor PIN diagnosis

pain-predominant RTS
!= motor PIN rehabilitation authority

route selection
!= cervical / plexus / proximal radial / MMN exclusion

not assessed motor status
!= normal motor status

investigation finding
!= rehabilitation protocol
```

No evidence source is allowed to silently convert one of these concepts into another.

---

## 6. Rehabilitation evidence decision

The route intentionally contains:

```text
rehabilitation_sequences: {}
```

This is a reviewed evidence decision, not missing work.

Current evidence does not justify automatic literature-derived PIN referral directions for:

- nerve gliding/neurodynamic treatment;
- a splint type or wear schedule;
- manual therapy;
- electrical stimulation;
- a strengthening dose/protocol;
- a staged motor-recovery programme;
- visit frequency;
- total PT course;
- numeric progression/discharge thresholds.

A clinician may still enter an explicit clinician instruction under the existing authority model, but it must not be relabelled as route-specific literature recommendation.

---

## 7. Recovery/surgical-review timing

The 2026 spontaneous PIN cohort provides clinically useful course information, including the association between motor improvement within six months and later good recovery and consideration of neurolysis when recovery is absent.

CU-1 interpretation is deliberately narrower:

```text
source-specific six-month decision point
!= universal PT course duration
!= automatic surgery trigger
!= universal progression criterion
```

New/progressive objective weakness or material motor worsening triggers reassessment independent of an invented calendar threshold.

---

## 8. Source-metadata and applicability correction before PASS

Pre-PASS audit found three curation issues and corrected them through the pending route activation amendment:

1. the 2006 systematic-review journal/pages were initially transcribed incorrectly and are normatively corrected to `J Peripher Nerv Syst. 2006;11(2):101-110`, DOI `10.1111/j.1085-9489.2006.00074.x`;
2. the electrodiagnostic article issue year is 2012 rather than 2013 and its exact HSS Journal citation/DOI are normatively corrected while retaining the pre-activation evidence ID to avoid identity churn;
3. the pain-only/radial-tunnel mismatch profile no longer references a motor-deficit-only localization safety claim.

These changes correct identity/applicability only and do not upgrade clinical evidence.

---

## 9. Regression fixture requirements

`cu1_pin_supinator_fixtures_v1.yaml` must prove at minimum:

- pain-only RTS does not become motor PIN;
- motor pattern does not create formal diagnosis;
- `not_assessed` motor status does not become normal;
- formal diagnosis does not infer etiology or entrapment site;
- spontaneous 2026 conservative-management evidence does not become a PT protocol;
- the six-month cohort signal is not universalized;
- compressive/mass/traumatic/iatrogenic contexts fail closed;
- progressive motor deficit triggers reassessment;
- sensory/proximal/MMN differentials remain open when relevant;
- diagnostic tests do not generate a rehabilitation programme;
- nerve gliding/splint/manual/electrical modalities are not auto-rendered;
- no LET/generic elbow/peripheral-nerve fallback;
- missing history remains missing.

---

## 10. Review verdict

```text
route taxonomy preserved                         PASS
pain-only vs motor PIN boundary                  PASS
spontaneous vs compressive/structural split      PASS
motor-pattern vs diagnosis separation            PASS
alternative-localization semantics               PASS
2026 cohort scope preserved                      PASS
specific PT evidence sufficiency                 INSUFFICIENT — explicit reviewed gap
route-specific rehab sequence                    NOT AUTHORIZED
progression thresholds                           NONE INVENTED
route-history prompts                            PASS
matching fixtures                                PASS
runtime authorization                            NO
```

### Final route classification

**PASS PENDING EXACT-HEAD CI** as a **profile-curated explicit evidence-gap + safety/context split**.

If focused CI passes at this exact review head, activation may add the shard, its mandatory narrow amendment and fixtures to the normative manifest/coverage matrix. Activation must record all PIN contexts as `blocked_evidence_gap` or safety/correct-owner states; it must **not** report `sequence_complete` and must not create a rehabilitation sequence merely to satisfy coverage bookkeeping.
