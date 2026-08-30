# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE — CU-1 CLINICAL-CONTEXT COMPOSITION.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Branch head at composition-slice claim:** `0a17856ae2d0d7737ad3748711676cdc34297b2d`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **Runtime authorization:** shared/data-driven clinical-context composition + exact frozen-shoulder integration + deterministic tests only.
> **Latest previously accepted runtime head:** `917f38704745aeec48d8e332bdf5f1d23c82a26d`.
> **Latest previously accepted CI evidence:** workflow run `33303230721` / run #389 — compile PASS, browser JavaScript syntax PASS, focused Python suite PASS.
> **Merge/deploy authorization:** NO.
> **Preview deployment:** NOT REQUESTED / NOT AUTHORIZED.
> **Further route-by-route rollout:** HOLD until the clinical-context composer is product-reviewed and exact-head tested.
> **CU-2 / PR-1:** HOLD.

---

# 1. Current product problem

The frozen-shoulder rehabilitation content and section-based rich renderer are already accepted, but the opening clinical context still serializes selected findings/functions too literally.

Canonical product rule:

```text
SELECTION INPUT != REFERRAL TEXT
```

Current defect:

```text
validated structured selections
→ label lookup
→ joined phrases
→ checklist-like clinical opening
```

Required behavior:

```text
validated structured selections
→ deterministic clinical composition
→ natural physician-to-physiotherapist clinical prose
```

No patient fact may be invented. Composition may fuse or subsume selected facts only when a reviewed declarative rule explicitly supports that transformation.

---

# 2. Proven state preserved

Already IMPLEMENTED / TESTED before this slice:

- rich referral renderer with staged and section-based Detailed layouts;
- context-gated fail-closed behavior (`formatter_blocked=true`, `text=null` when required rich context is unresolved);
- primary frozen shoulder rich generation only for clinician-established primary frozen shoulder;
- optional clinician-entered `frozen_shoulder_irritability` with no inference from findings;
- frozen-shoulder Greek physician-referral wording and section structure;
- hierarchical UI relevance (`profile → route → subtype → explicit context`);
- clinician-only evidence panel and route-specific rich content architecture.

Not yet proven:

- natural shared composition of the initial clinical context.

---

# 3. Exact authorized implementation

Authorized now:

1. introduce one shared deterministic clinical-context composer;
2. add a versioned declarative Greek composition contract;
3. allow route data to provide a grammar-ready problem phrase for the shared composer;
4. add reviewed fusion/subsumption rules for compatible selected facts;
5. integrate the composer at the existing formatter → rich-renderer seam;
6. preserve selected-facts-only / no-invention behavior;
7. add focused exact-output and partial-selection regression tests;
8. update workflow coverage and canonical state.

The implementation must remain generic. A Python branch such as:

```text
if route_id == adhesive_capsulitis_frozen_shoulder
```

is forbidden.

Route-specific language data is allowed because clinical grammar/content belongs in reviewed data, not in disease-specific formatter code.

---

# 4. Composition safety contract

The shared composer must obey:

```text
selected fact → may appear directly
selected compatible fact set → may be fused by an explicit reviewed rule
selected more-general fact → may be suppressed only by explicit reviewed subsumption
unselected fact → must not appear as patient-specific clinical truth
ambiguous/unsupported combination → no invented resolution
```

Fusion rules must declare their required IDs, consumed IDs and any explicitly subsumed IDs. More-specific rules must resolve before less-specific overlapping rules.

Residual facts that are not safely composable remain represented using existing clinician-facing labels rather than being guessed into new prose.

`frozen_shoulder_irritability` remains explicit context and is never inferred from pain/ROM selections.

---

# 5. Frozen-shoulder acceptance target

For the realistic selected case:

```text
formal diagnosis
primary frozen shoulder
right
pain
active ROM restricted
passive ROM restricted
painful active ROM
painful passive ROM
overhead activity difficulty
lifting/carrying difficulty
driving difficulty
high irritability
```

the clinical opening should read naturally, beginning approximately:

```text
Ασθενής με συμφυτική θυλακίτιδα / παγωμένο ώμο δεξιά,
με επώδυνο και περιορισμένο ενεργητικό και παθητικό εύρος κίνησης.
Λειτουργικά αναφέρεται δυσκολία σε δραστηριότητες πάνω από το ύψος του ώμου,
στην άρση ή μεταφορά φορτίου και στην οδήγηση.
```

The exact Short/Detailed full outputs must be locked by deterministic tests.

---

# 6. Explicit HOLD

Do not:

- merge to `main`;
- deploy;
- create preview service;
- resume new disease-route rollout;
- change persistence;
- reopen CU-2 or PR-1;
- alter underlying clinical-record semantics;
- add inferred diagnosis/severity/phase/function.

---

# 7. Exact next action

```text
canonical claim
→ implement shared composer + declarative rules
→ exact-output/partial-selection tests
→ exact-head CI
→ update CURRENT_OPERATIONAL + append changelog
→ STOP for product-owner review
```
