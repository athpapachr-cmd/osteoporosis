# SLICE_PLAN_CURRENT.md — CU-1 rich referral clinical-context composition v1.21

> **STATUS:** IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED — SLICE CLOSED ON FEATURE BRANCH.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice:** CU-1 Physiotherapy Referral — shared clinical-context composer for rich referrals.
> **Authoritative remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Writer/runtime writer:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Product-owner authorization used:** bounded wording correction + lock reviewed Detailed referral.
> **Implementation commit:** `38f7977811d50636a1585225c74306bef496601c`.
> **Accepted product-review correction head:** `e0f690818c63c146a08a5e508a8123b9059b6b33`.
> **CI:** `CU-1 focused tests` run `33311066018` / #398 — compile PASS; browser-JS syntax PASS; Python focused suite PASS.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product objective — CLOSED AT IMPLEMENTATION/TEST/PRODUCT-REVIEW LEVEL

The checklist-like opening of rich physiotherapy referrals has been replaced by deterministic clinical composition while preserving the existing evidence/rehabilitation architecture.

Canonical distinction:

```text
structured clinical selection
!=
physician referral prose
```

The referral remains a projection of reviewed structured facts. It does not mutate the underlying clinical record and does not infer patient facts that were not selected or explicitly supplied.

---

# 2. Final runtime seam

```text
ReferralDraftV1
→ route/context validation
→ CU1GreekReferralFormatter v2
→ shared CU1ClinicalContextComposer
→ selected-fact composition
→ CU1RichReferralRenderer
→ Short / Detailed text
```

`CU1RichReferralRenderer` remains responsible for rich route rehabilitation/content structure. `CU1ClinicalContextComposer` owns only the natural composition of selected clinical facts before that renderer.

There is no frozen-shoulder-specific Python formatter branch.

---

# 3. Implemented reusable model

Normative artifact:

```text
clinic_utilities/contracts/cu1_clinical_composition_el_v1.yaml
```

Implemented rule fields:

```text
rule_id
priority
require_all[]
consume[]
suppress_if_matched[]
phrase_el
```

Semantics:

- all `require_all` IDs must be selected before a rule fires;
- `consume` declares selected IDs represented by the fused phrase;
- `suppress_if_matched` declares broader selected facts explicitly subsumed by that exact fusion;
- higher-priority more-specific rules resolve before overlapping lower-priority rules;
- residual selected facts remain rendered with existing clinician-facing Greek labels rather than being silently dropped or guessed.

Route-specific grammar is data. The frozen route provides the reviewed formal-diagnosis problem phrase:

```text
συμφυτική θυλακίτιδα / παγωμένο ώμο
```

The shared Python composer does not know the frozen route ID.

---

# 4. Product-owner reviewed Detailed wording

The product owner reviewed the actual generated referral and locked the following corrections on 2026-08-30:

## 4.1 Physiotherapy direction

The rendered Detailed referral must NOT include:

```text
Η επιλογή και δοσολογία των επιμέρους ασκήσεων και τεχνικών εξατομικεύονται από τον φυσιοθεραπευτή σύμφωνα με την κλινική ανταπόκριση.
```

Rationale: this is an unnecessary execution/disclaimer statement in the physician-to-physiotherapist referral. The underlying execution boundary may remain represented in route/evidence governance; it does not need to be rendered as referral prose.

## 4.2 Reassessment / escalation handoff

The Detailed `ΕΠΑΝΕΚΤΙΜΗΣΗ` section is locked to:

```text
Παρότι έχει προγραμματιστεί ιατρική επανεκτίμηση, συνιστάται επικοινωνία με τον θεράποντα ιατρό για νωρίτερη επανεκτίμηση σε περίπτωση επιδείνωσης, εμφάνισης νέων κλινικών ή τραυματικών στοιχείων ή άλλης ουσιώδους μεταβολής της κλινικής εικόνας.
```

This intentionally does NOT specify a universal routine medical follow-up interval. The purpose of the referral sentence is to tell the physiotherapist when earlier medical reassessment should be activated.

The rendered Detailed section and route-level `reassessment_el` authority carry the same wording.

---

# 5. No-invention / contradiction behavior — TESTED

Hard invariants remain:

```text
UNSELECTED PATIENT FACT → NEVER EMITTED
FUSION WITHOUT ALL REQUIRED FACTS → NEVER EMITTED
ROUTE-SPECIFIC PYTHON BRANCH → FORBIDDEN
UNKNOWN MACHINE ID → EXISTING VALIDATION RULES APPLY
UNSAFE/AMBIGUOUS COMPOSITION → DO NOT GUESS
```

Regression coverage proves that partial ROM selection does not invent passive or painful ROM and overhead-only function does not invent lifting/carrying or driving.

`frozen_shoulder_irritability` remains explicit clinician-entered context and is never inferred from findings.

---

# 6. Frozen-shoulder output contract — TESTED / REVIEWED

Rich generation remains limited to clinician-established primary frozen shoulder. Missing/unresolved/secondary scope remains blocked with `formatter_blocked=true` and `text=null`.

Short and Detailed share the same composed clinical truth. Detailed retains:

```text
ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ
ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ
ΕΠΑΝΕΚΤΙΜΗΣΗ
```

No artificial `ΣΤΑΔΙΟ 1` was reintroduced.

---

# 7. Acceptance evidence

The dedicated clinical-composition and frozen-shoulder tests lock:

```text
A. exact natural Short output
B. exact natural Detailed output
C. ROM quartet fusion without duplicate raw labels
D. explicit generic-pain subsumption only on matching rule
E. partial ROM no-invention
F. functional no-invention
G. priority resolution for overlapping function rules
H. unresolved irritability omission
I. no frozen route branch in composer source
J. existing context-gated fail-closed behavior
K. removed physiotherapist dosage/selection sentence stays absent from Detailed
L. earlier-reassessment communication sentence stays present in Detailed
```

Accepted corrected head:

```text
head: e0f690818c63c146a08a5e508a8123b9059b6b33
workflow: CU-1 focused tests
run id: 33311066018
run number: 398
compile: PASS
browser JavaScript syntax: PASS
Python focused suite: PASS
```

---

# 8. Current HOLD

Still out of scope / not authorized:

- new disease rollout;
- unrelated evidence expansion;
- generic goals/rehab checkbox redesign;
- persistence or clinical-record write-back;
- preview service;
- merge;
- deployment;
- CU-2 / PR-1.

A future delivery-model decision may choose disease-centered slices that pair clinical assessment and physiotherapy projection from one reviewed condition model. That is a sequencing/product decision, not authorization to alter the current runtime in this slice.

---

# 9. Completion boundary

```text
DESIGNED                YES
IMPLEMENTED             YES
TESTED                  YES
PRODUCT-OWNER REVIEWED  YES
MERGED                  NO
DEPLOYED                NO
PREVIEWED               NO
```

This slice is closed on the feature branch. Merge/deploy remain separate explicit decisions.
