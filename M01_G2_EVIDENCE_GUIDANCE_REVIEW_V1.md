# M01 G-2 Evidence-backed Osteoporosis Guidance — Design Review v1

> **Review date:** 2026-08-31 Asia/Nicosia  
> **Module:** 01 — Osteoporosis  
> **Slice:** `M01-G2-EVIDENCE-GUIDANCE-CONTENT-v1`  
> **Primary framework:** NOGG 2024  
> **Regulatory context:** current EU/EMA product information where applicable  
> **Review classification:** DESIGN REVIEW — runtime activation classification included; no merge/deploy authority.

---

## 1. Review question

Does the G-2 contract define a minimum clinically useful, evidence-backed osteoporosis guidance layer that can be deterministically activated over G-1 without:

- turning a guideline into an automatic prescribing engine;
- inventing thresholds or visit-number milestones;
- treating missing information as a negative finding;
- silently mixing incompatible frameworks;
- deriving exact timing from scheduled rather than actual treatment events;
- or claiming safety clearance when the current runtime only has enough data to surface a checklist?

**Conclusion:** after the corrections described below, the contract is suitable to freeze as the design authority for a bounded runtime implementation. Runtime activation must remain narrower than the full evidence registry where structured inputs/linkage are not yet reliable.

---

## 2. Evidence hierarchy

G-2 uses:

1. **NOGG 2024** as the primary clinical framework for the first implementation;
2. **EMA/EU product information** for current regulatory contraindication/safety facts;
3. **Endocrine Society 2020** and **ECTS 2020** as explicitly separate supporting frameworks for denosumab;
4. **ASBMR/BHOF 2024** as contextual goal-directed treatment reasoning only;
5. recent denosumab-discontinuation studies as uncertainty/context evidence, not automatic treatment rules.

Hard rule:

```text
PRIMARY FRAMEWORK + EXPLICIT SUPPORTING SOURCE
!=
SILENT HYBRID THRESHOLD
```

The runtime must preserve provenance and must not apply NOGG-specific thresholds when another framework is explicitly being used unless the same criterion is independently represented under that framework.

---

## 3. Clinical fidelity corrections made during final review

### G2-RV1 — denosumab >7-month warning requires prior exposure of at least two doses

The NOGG FAQ wording is not a generic statement after the first injection. It specifies substantial rebound high-turnover/bone-loss risk when a patient who has received **at least two denosumab doses** misses/delays the next injection beyond 7 months from the prior dose.

Therefore `OST_G2_R24_DENOSUMAB_GT7M_REBOUND_ESCALATION` and `DENOSUMAB_GT7_MONTH_REBOUND_ESCALATION` now require:

```text
reliable actual denosumab administration count >= 2
+ reliable last actual administration date
+ encounter date
+ interval > 7 calendar months
```

A patient >7 months after a **first** documented dose can still receive general time-critical/due guidance from the six-month denosumab rules, but the specific NOGG ≥2-dose rebound escalation is not asserted.

### G2-RV2 — formal FRAX evidence rule is conditional on actual indication

NOGG recommends FRAX for postmenopausal women and men age ≥50 **with a clinical risk factor for fragility fracture**. The initial-visit archetype alone is not evidence that every patient automatically requires FRAX.

The product-flow profile can still surface the Formal Risk card during an initial assessment. The evidence-specific rule `OST_G2_R01_INITIAL_FORMAL_RISK` now requires:

```text
NOGG scope eligible
+ risk_assessment.formal_indicated == "yes"
```

This cleanly separates:

```text
PRODUCT FLOW: determine whether formal risk assessment is indicated
from
EVIDENCE RULE: formal assessment is explicitly indicated today
```

### G2-RV3 — very-high-risk indicators verified against NOGG Section 4

The exact candidate indicators represented in `OST_G2_R08_EXPLICIT_VERY_HIGH_RISK_REVIEW` were verified against NOGG 2024 Section 4:

- recent vertebral fracture within 2 years;
- at least 2 vertebral fractures;
- BMD T-score ≤ -3.5;
- high-dose glucocorticoids ≥7.5 mg/day prednisolone equivalent over 3 months;
- or other/multiple risk factors producing very-high-risk status including FRAX-defined very high risk.

The rule remains a **specialist/parenteral/anabolic consideration** rule, not an automatic anabolic-treatment command.

### G2-RV4 — YAML enum semantics made explicit

Persisted application enums use strings such as `"yes"` and `"no"`. Bare YAML `yes/no` can be parsed as booleans by common YAML parsers. Predicate values that refer to persisted enum strings are therefore explicitly quoted in the machine contract.

This is a machine-semantics correction, not a clinical change.

---

## 4. Runtime activation classification

| Rule | Clinical purpose | Runtime v1 classification | Rationale |
|---|---|---|---|
| R01 | Formal risk when explicitly indicated | `activate_v1` | Existing explicit `formal_indicated`; NOGG scope guard required. |
| R02 | VFA structured indications | `activate_v1` | Height loss, GC context, T-score and structured vertebral fracture can be represented. Kyphosis/acute back pain remain known structured-data gaps. |
| R03 | Secondary-cause review | `activate_v1` | May surface review; prior adequate work-up should prevent unnecessary full repeat. |
| R04 | Falls/function | `activate_v1` | Current fall count and fracture context exist. |
| R05–R06 | New fragility-fracture override | `activate_v1` | Existing structured fracture state; no automatic drug choice. |
| R07 | Fracture on treatment | `activate_v1` | Existing event/treatment context; explicitly not automatic treatment failure. |
| R08 | Very-high-risk review | `activate_v1_with_NOGG_guard` | Exact criteria verified; only apply NOGG threshold semantics under NOGG framework/scope. |
| R09 | Shared treatment-decision factors | `activate_v1` | Treatment-decision visit intent is explicit; final choice remains clinician-owned. |
| R10 | Vitamin-D repletion before parenteral therapy | `checklist_only` | Surface “verify/address”; do not infer adequacy from absent/stale lab data. |
| R11 | Denosumab long-term plan at start | `activate_v1` | Selected agent and start decision are explicit. |
| R12 | Denosumab six-month due | `activate_v1` | Derive only from reliable exact actual date; derived due date remains ephemeral. |
| R13 | Denosumab pre-dose calcium | `checklist_only` | Surface pre-dose safety requirement; do not auto-clear from unverified lab freshness. |
| R14 | Denosumab exit/sequential plan | `activate_v1` | Explicit transition/stop semantics + reliable last actual dose can surface time-locked transition guidance. No automatic selected agent write. |
| R15 | CTX after post-exit zoledronate | `blocked_missing_reliable_linkage` | Evidence is valid, but current projection does not yet prove that a specific zoledronate actual event is the denosumab-exit sequential infusion across visits. |
| R16 | No-CTX fallback | `blocked_missing_structured_input` | Current runtime has no authoritative `CTX monitoring unavailable` field; blank CTX is not equivalent to unavailable monitoring. |
| R17 | Zoledronate renal/mineral safety | `checklist_only` | Surface prerequisites. Do not infer “safe” without explicit freshness/context. |
| R18 | Romosozumab CV/hypocalcaemia safety | `checklist_only` | Current form does not comprehensively represent prior MI/stroke/CV risk. Surface verification; never auto-pass. |
| R19 | Teriparatide metabolic/contraindication safety | `checklist_only` | Not every contraindication is fully structured; surface verification, not clearance. |
| R20 | Post-anabolic/romosozumab antiresorptive follow-on | `activate_v1_when_completion_or_transition_explicit` | Requires explicit completion/transition or reliable exact course exposure. |
| R21 | Oral BP ≥5-year reassessment | `activate_v1_when_exact_start_reliable` | Exact exposure only from reliable start date/timeline; no automatic holiday. |
| R22 | Zoledronate ≥3-year reassessment | `activate_v1_when_exact_start_reliable` | Exact exposure only from reliable start date/timeline; no automatic holiday. |
| R23 | Targeted lifestyle communication | `activate_v1` | Initial/post-fracture only; avoid repetitive generic counselling at every visit. |
| R24 | Denosumab >7-month rebound escalation | `activate_v1_with_two_dose_guard` | Requires reliable count ≥2 + actual last-dose date + >7 calendar months. |
| R25 | Oral BP start safety/use | `checklist_only` | Oesophageal/upright/renal-mineral suitability and instructions are review items, not automatic clearance. |
| R26 | Oral BP 12–16-week review | `activate_v1_when_exact_start_reliable` | Exact start date required; approximate duration must not be treated as exact timing. |

Blocked rules remain part of the evidence contract so the system can activate them later when safe structured inputs/linkage exist. They must not be approximated from blank fields or free text in G-2 runtime v1.

---

## 5. Denosumab/time-critical semantics

G-2 freezes four distinct concepts:

```text
EXPECTED DUE AT 6 MONTHS
!=
>7-MONTH REBOUND-RISK ESCALATION AFTER >=2 DOSES
!=
PLANNED DENOSUMAB DISCONTINUATION / EXIT
!=
POST-EXIT ZOLEDRONATE + CTX FOLLOW-UP
```

Rules:

- use exact actual administration dates, never nominal appointment labels;
- do not reconstruct missing doses from cadence;
- administration count and elapsed exposure remain separate;
- an evidence-derived six-month due date is ephemeral and never overwrites a clinician-recorded due date;
- a scheduled/planned administration without `actual_date` does not count as a dose;
- conflicting administration history suppresses exact milestone derivation;
- CTX may guide further management after denosumab exit, but G-2 v1 has **no automatic CTX 280/300 ng/L retreatment rule**;
- no 4th/8th/10th Prolia milestone is encoded without separate reviewed evidence/policy.

---

## 6. Safety rules are guidance, not clearance engines

For zoledronate, denosumab, romosozumab, teriparatide and oral bisphosphonates, G-2 may surface medication-specific prerequisites at the point of use.

Where the current form lacks complete/current structured data, the output must be phrased as:

```text
verify / review / confirm / address before treatment
```

and not:

```text
safe / cleared / no contraindication
```

Particularly:

- romosozumab: prior MI/stroke and broader CV risk must be actively checked; no mandatory cardiology/vascular referral is encoded without approved clinic policy;
- zoledronate: renal/mineral values must be clinically current enough for the planned infusion before any clearance inference;
- teriparatide: major contraindications are not all safely inferable from the existing data model;
- oral bisphosphonate: correct administration and oesophageal/upright suitability are checklist items unless explicitly recorded.

---

## 7. G-1 integration architecture

Do not rewrite the working G-1 engine into a monolithic osteoporosis rule engine.

Preferred bounded implementation:

```text
existing G-1 LongitudinalGuidanceProjection
+ current live encounter snapshot
        ↓
G2 OsteoporosisEvidenceContext
        ↓
pure deterministic G2 rule evaluator
        ↓
evidence contributions per current domain/card
        ↓
merge with existing G-1 Visit Plan using manifest precedence
        ↓
existing `Σημερινή ροή` + `Γιατί τώρα:` UI
```

A G-2 contribution should retain at least:

```text
rule_id
rule_class
priority
domain
why_now
guidance_objective
source_refs[]
strength
activation_mode
```

G-1 product-flow reasons remain valid and distinct from evidence-backed reasons.

---

## 8. Live-state ownership requirement

G-1 already established:

```text
live control value, including blank
>
persisted browser cache
```

G-2 must preserve this for every new field it reads. The implementation must not read a stale persisted Step 3/4 value when a corresponding live control exists and has been cleared/changed.

The G-2 runtime tests must include a live-blank-over-persisted-value regression for evidence-trigger fields.

---

## 9. Required implementation tests

At minimum:

1. initial visit can show product-flow Formal Risk without asserting R01 until formal risk is explicitly indicated;
2. NOGG-scope + `formal_indicated="yes"` activates R01;
3. VFA trigger from ≥4 cm height loss;
4. new fracture overrides stable flow;
5. fracture-on-treatment surfaces reassessment but never auto-failure/switch;
6. denosumab exact six-month due derives from reliable actual date only;
7. scheduled-only denosumab event does not count as actual;
8. >7 months after **first** reliable denosumab dose does **not** activate R24;
9. >7 months after ≥2 reliable denosumab doses activates R24;
10. conflicting denosumab history suppresses exact milestone derivation;
11. denosumab exit guidance never writes an automatic selected drug;
12. R15/R16 remain inactive when reliable linkage/CTX-availability input is absent;
13. romosozumab/zoledronate/teriparatide/oral-BP safety rules surface as checklist guidance, not automatic clearance;
14. oral BP 12–16-week review requires reliable exact start date;
15. oral BP 5-year and zoledronate 3-year reassessment require reliable exact exposure;
16. NOGG-specific very-high-risk criteria do not activate under an explicitly non-NOGG framework merely from NOGG threshold semantics;
17. no CTX 280/300 automatic retreatment rule exists;
18. no generic Prolia 4th/8th/10th milestone exists;
19. live blank/changed controls override persisted cache;
20. inherited G-1/C1 regression suites remain green.

---

## 10. Design conclusion

With G2-RV1 through G2-RV4 incorporated and machine-contract CI passing at the final design head, classify the evidence/content contract as:

```text
DESIGN-COMPLETE
```

This classification means:

```text
EVIDENCE CONTRACT REVIEWED
+ RULE/PROFILE/MILESTONE SEMANTICS FROZEN FOR BOUNDED IMPLEMENTATION
```

It does **not** mean:

```text
IMPLEMENTED
TESTED IN RUNTIME
MERGED
DEPLOYED
PRODUCTION-SMOKE-VERIFIED
PILOT-VALIDATED
```

The next authorized action under the product-owner instruction is a separate bounded G-2 runtime implementation branch from the exact design-complete head. No merge/deploy is authorized by this design review.
