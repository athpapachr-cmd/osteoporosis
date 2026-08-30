# SLICE_PLAN_CURRENT.md — CU-1 rich referral clinical-context composition v1.20

> **STATUS:** IMPLEMENTED / TESTED — PRODUCT-OWNER REVIEW GATE.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice:** CU-1 Physiotherapy Referral — shared clinical-context composer for rich referrals.
> **Authoritative remote main:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Writer/runtime writer:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Product-owner authorization used:** update canonicals + implement the shared composition correction.
> **Implementation commit:** `38f7977811d50636a1585225c74306bef496601c`.
> **Accepted test head:** `9b46623b2c991df631698bf018749550dd843f87`.
> **CI:** `CU-1 focused tests` run `33306399908` / #394 — 127/127 Python tests PASS; compile PASS; browser-JS syntax PASS.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product objective — CLOSED AT IMPLEMENTATION/TEST LEVEL

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
- residual selected facts remain rendered with existing clinician-facing labels rather than being silently dropped or guessed.

Route-specific grammar is data. The frozen route provides the reviewed formal-diagnosis problem phrase:

```text
συμφυτική θυλακίτιδα / παγωμένο ώμο
```

The shared Python composer does not know the frozen route ID.

---

# 4. Implemented initial global fusion rules

Finding composition:

```text
active ROM restricted
+ passive ROM restricted
+ painful active ROM
+ painful passive ROM
(+ selected generic pain may be explicitly subsumed)
→ επώδυνο και περιορισμένο ενεργητικό και παθητικό εύρος κίνησης
```

and:

```text
active ROM restricted
+ passive ROM restricted
→ περιορισμένο ενεργητικό και παθητικό εύρος κίνησης
```

Functional composition:

```text
overhead activity
+ lifting/carrying
+ driving
→ δυσκολία σε δραστηριότητες πάνω από το ύψος του ώμου,
  στην άρση ή μεταφορά φορτίου και στην οδήγηση
```

and the narrower overhead + lifting/carrying pair.

These are canonical-fact rules, not frozen-only rules.

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

# 6. Frozen-shoulder output contract — TESTED

Rich generation remains limited to clinician-established primary frozen shoulder. Missing/unresolved/secondary scope remains blocked with `formatter_blocked=true` and `text=null`.

Short and Detailed share the same composed clinical truth. Detailed retains:

```text
ΚΛΙΝΙΚΗ ΕΙΚΟΝΑ
ΣΤΟΧΟΙ ΑΠΟΚΑΤΑΣΤΑΣΗΣ
ΚΑΤΕΥΘΥΝΣΗ ΦΥΣΙΟΘΕΡΑΠΕΙΑΣ
ΕΠΑΝΕΚΤΙΜΗΣΗ
```

No artificial `ΣΤΑΔΙΟ 1` was reintroduced. Existing accepted frozen-shoulder rehabilitation wording remains route-owned and unchanged apart from the clinical-context grammar hook.

---

# 7. Acceptance evidence

The dedicated clinical-composition tests lock:

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
```

The full focused suite passed at exact head `9b46623b2c991df631698bf018749550dd843f87`:

```text
127 tests
0 failures
compile PASS
browser JavaScript syntax PASS
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

---

# 9. Completion boundary

```text
DESIGNED       YES
IMPLEMENTED    YES
TESTED         YES
MERGED         NO
DEPLOYED       NO
PREVIEWED      NO
```

The implementation/test slice is complete. The next gate is product-owner review of the actual generated Short and Detailed referral text. Any further code mutation requires a specific product defect or an explicit next authorization.
