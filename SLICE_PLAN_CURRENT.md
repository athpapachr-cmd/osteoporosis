# SLICE_PLAN_CURRENT.md — CU-1 rich referral clinical-context composition v1.20

> **STATUS:** ACTIVE RUNTIME IMPLEMENTATION — CLINICAL-CONTEXT COMPOSITION.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice:** CU-1 Physiotherapy Referral — shared clinical-context composer for rich referrals.
> **Authoritative remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Writer/runtime writer:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Product-owner authorization:** YES — update canonicals and implement the shared composition correction.
> **Merge/deploy/preview:** NOT AUTHORIZED.

---

# 1. Product objective

Preserve the existing rich rehabilitation/evidence architecture while correcting the remaining checklist-like opening of the generated referral.

Canonical distinction:

```text
structured clinical selection
!=
physician referral prose
```

The referral is a projection of reviewed structured facts. It must not mutate the underlying clinical record and must not infer facts that were not selected or explicitly supplied.

---

# 2. Existing seam

Current runtime seam:

```text
ReferralDraftV1
→ route/context validation
→ CU1GreekReferralFormatter v2
→ _rich_clinical_context()
→ CU1RichReferralRenderer
→ Short / Detailed text
```

The rich renderer is not the defect. It already accepts composed `clinical_context` sentences and handles route-specific rehabilitation content correctly.

The defect is `_rich_clinical_context()`, which currently converts IDs to labels and joins them. The correction belongs immediately before the rich renderer.

Target seam:

```text
validated normalized ReferralDraftV1
→ shared CU1ClinicalContextComposer
→ composed clinical-context sentences
→ existing CU1RichReferralRenderer
```

No frozen-shoulder-specific Python formatter branch is permitted.

---

# 3. Minimal reusable object model

Introduce one versioned composition artifact:

```text
cu1_clinical_composition_el_v1.yaml
```

It owns only referral-language composition mechanics, not clinical truth.

Required concepts:

```text
problem phrase template
laterality suffix
finding fusion rules
functional fusion rules
explicit subsumption rules
priority for overlapping rules
```

Each fusion rule contains at minimum:

```text
rule_id
priority
require_all[]
consume[]
phrase_el
suppress_if_matched[] optional
```

Semantics:

- `require_all`: every listed selected fact must be present before the rule may fire;
- `consume`: selected facts represented by the fused phrase and therefore not rendered again;
- `suppress_if_matched`: selected broader facts explicitly subsumed by the fused phrase;
- `priority`: deterministic resolution of overlapping rules, with more-specific rules first;
- `phrase_el`: reviewed Greek phrase representing only the declared selected facts.

Route-specific grammar may be supplied as structured route data, for example a grammar-ready accusative problem phrase. That is data, not route-specific formatter logic.

---

# 4. Composition algorithm

For each rich referral:

```text
1. read validated primary problem + selected findings/functions
2. resolve applicable rich route/variant
3. resolve optional grammar-ready route problem phrase
4. evaluate fusion rules by descending priority
5. emit each matched fused phrase once
6. remove only declared consumed/subsumed selected IDs
7. render residual selected IDs with existing clinician-facing labels
8. compose natural clinical sentences
9. append explicit work/sport context, restrictions/precautions and existing detailed context
10. hand sentences to the unchanged rich renderer
```

No rule may trigger from diagnosis alone unless the data contract explicitly defines a non-patient-fact language template. Findings/functions remain selection-driven.

---

# 5. No-invention / contradiction behavior

Hard invariants:

```text
UNSELECTED PATIENT FACT → NEVER EMITTED
FUSION WITHOUT ALL REQUIRED FACTS → NEVER EMITTED
ROUTE-SPECIFIC PYTHON BRANCH → FORBIDDEN
UNKNOWN MACHINE ID → EXISTING VALIDATION RULES APPLY
UNSAFE/AMBIGUOUS COMPOSITION → DO NOT GUESS
```

A selected broad fact may be omitted from literal output only when an explicit subsumption rule states that a more-specific selected phrase already represents it. This is semantic compression, not deletion of evidence.

Residual facts are retained rather than silently discarded. If a combination has no reviewed fusion rule, the composer falls back to existing clinician-facing labels for those remaining selected facts.

The composer does not infer irritability, phase, diagnosis, severity, occupation, activity demand, ROM direction or functional limitation.

---

# 6. Frozen-shoulder first integration

Route:

```text
shoulder.adhesive_capsulitis_frozen_shoulder
```

Rich generation remains allowed only for clinician-established primary frozen shoulder. Missing/unresolved/secondary context remains blocked with `formatter_blocked=true` and `text=null`.

The route may supply a grammar-ready problem phrase for formal diagnosis:

```text
συμφυτική θυλακίτιδα / παγωμένο ώμο
```

The generic composer may then render:

```text
Ασθενής με {problem phrase} {laterality}, με {reviewed fused findings}.
```

Initial reviewed global fusion rules are deliberately narrow:

- painful + restricted active/passive ROM quartet;
- restricted active/passive ROM pair;
- overhead + lifting/carrying + driving functional trio;
- overhead + lifting/carrying functional pair.

They are not frozen-shoulder-only rules; any future rich route selecting the same canonical facts can reuse them.

---

# 7. Output rules

Short and Detailed use the same composed clinical truth.

Short remains flowing prose followed by the route-specific rich rehabilitation direction.

Detailed remains route-owned. For frozen shoulder the accepted structure stays:

```text
ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ
ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ
ΕΠΑΝΕΚΤΙΜΗΣΗ
```

No artificial `ΣΤΑΔΙΟ 1` is reintroduced.

Output remains Greek-only and under the existing GeSY 2000-character ceiling.

---

# 8. Acceptance evidence

Required focused tests:

```text
A. realistic frozen case produces exact natural Short output
B. same case produces exact natural Detailed output
C. ROM quartet fuses once; individual ROM labels do not leak redundantly
D. selected generic pain is subsumed only when the reviewed painful-ROM fusion matched
E. partial ROM selection does not invent passive/painful ROM
F. overhead-only function does not invent lifting/driving
G. three-item functional fusion resolves before two-item overlapping rule
H. uncertain/not-assessed irritability remains omitted
I. composer source contains no frozen route-id branch
J. existing context-gated block behavior remains intact
```

Workflow must compile the new composer module and execute the new test file.

---

# 9. Out of scope / HOLD

- new disease rollout;
- evidence expansion unrelated to this clinical opening;
- generic goals/rehab checkbox redesign;
- persistence or clinical-record write-back;
- preview service;
- merge;
- deployment;
- CU-2 / PR-1.

---

# 10. REPLAN triggers

Stop and replan if:

- natural composition requires route-specific Python branches;
- a fusion rule cannot preserve selected-facts-only semantics;
- fixing the opening requires changing underlying clinical data meaning;
- existing rich-route safety/context gating regresses;
- exact-output tests require invented patient facts.

---

# 11. Completion boundary

The slice is IMPLEMENTED when the shared composer and declarative rules are committed.

It is TESTED only after exact-head GitHub Actions passes.

It is not MERGED, DEPLOYED or PREVIEWED unless separately authorized by the product owner.
