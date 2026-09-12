# KNEE_OA_TEMPLATE_DESIGN_REVIEW_V1.md — Step 3 exact design review

> **REVIEW TYPE:** active-writer exact-head design review.
> **DATE:** 2026-09-11 Asia/Nicosia.
> **SLICE:** `CU1-PRODUCT-KNEE-OA-TEMPLATE-V1-2026-09-11`.
> **FROZEN STEP-2 PARENT:** `ab4b349223cd4c461837ab3125967a06d169a7e1`.
> **REVIEWED SUBSTANTIVE HEAD:** `cd4a42b4582921df7eb64d6ff3fb7c718a141c3a`.
> **RUNTIME AUTHORITY:** NONE.

---

# 1. Review objective

Review the complete Step-3 branch against the frozen Step-2 ancestry and determine whether the Knee-OA dynamic referral/template design is sufficiently explicit, deterministic, clinically honest and bounded to freeze before Step 4.

The review did not treat a passing YAML parser or earlier machine run as proof of design correctness.

---

# 2. Scope reviewed

Reviewed:

```text
root Step-3 writer/slice canonicals
human Knee-OA template design
machine template contract
primary deterministic fixtures
power-user / preservation / blocking edge fixtures
contract validator
Step-3 GitHub Actions gate
frozen CU-1 state/language/route-requirement/rule seams
frozen Step-2 evidence contract compatibility
full branch-vs-Step-2 diff hygiene
```

Explicitly not reviewed as implemented behavior because no runtime implementation exists:

```text
production visual UX
actual browser live projection
actual Copy control
manual-edit interaction details
billing/account/entitlements
external-clinician usability
commercial willingness-to-pay
```

---

# 3. Material findings found and corrected before PASS

## F1 — fixed diagnosis screen risked autonomous diagnosis semantics

The first candidate could visually open directly into Knee OA while final prose stated OA without an explicit clinician assertion boundary.

Correction:

```text
Copy readiness requires existing CU-1 formal diagnosis semantics
formal_assertion_state = yes
right | left | bilateral laterality
```

Negative Copy-readiness fixtures now verify that missing/non-yes diagnosis assertion and non-copy-ready laterality block Copy.

## F2 — exact Greek opening and task grammar required polish

The initial phrase contained an awkward repeated `για`, and functional-task output could produce phrases such as `για σκάλες`.

Correction:

```text
... φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας δεξιού/αριστερού γόνατος

λειτουργική επανεκπαίδευση για τις σκάλες
```

Exact fixture strings own the reviewed wording.

## F3 — power-user selections could have been silently omitted

The early template mapped the routine surface well but did not prove that every supported `Περισσότερα` selection had output semantics.

Correction:

- explicit product-supported Knee-OA subset;
- unsupported selections fail closed rather than disappear;
- power-user findings/functions have deterministic phrases;
- restrictions and clinician notes are preserved;
- non-redundant goals are preserved;
- machine validator proves supported-input coverage.

## F4 — `true locking` cannot be safely exposed under current inherited rule semantics

CU-1 contains `true_locking_or_major_mechanical_rom_block` as a finding, but the frozen rule catalog does not itself infer a safety concern from that finding; safety input flags remain explicit clinician-confirmed concerns.

Correction:

```text
true locking remains in CU-1 taxonomy
but is excluded from the first Knee-OA product surface
until a bounded safety/reassessment mapping is designed
```

A blocking edge fixture proves the product projection fails closed if that unsupported finding is supplied.

## F5 — YAML `yes` ambiguity could corrupt formal assertion fixture semantics

Unquoted YAML `yes` may parse as boolean under PyYAML rather than the canonical string enum.

Correction:

All copy-ready fixtures use:

```yaml
formal_assertion_state: "yes"
```

and the validator requires the exact string.

## F6 — absolute “render or block” invariant conflicted with intentional de-duplication

Generic goals and specific-over-generic finding precedence intentionally avoid repetitive prose while preserving structured state.

Correction:

```text
selected product item
→ renders
OR blocks
OR is covered by an explicit semantic de-duplication rule
```

The validator proves the complete supported goal set is partitioned into explicit output-redundant vs non-redundant-rendered groups and verifies finding/function/plan/adjunct phrase coverage.

---

# 4. Clinical/semantic review disposition

PASS.

The reviewed design preserves these distinctions:

```text
stiffness symptom != ROM restriction
generic weakness != objective weakness
specific quadriceps weakness > generic weakness prose
functional limitation != automatic rehabilitation selection
suggestion != clinician selection
adjunct != core rehabilitation
evidence UI != copied referral prose
missing != negative
manual edited text != structured clinical state
```

No exact exercise dosage is invented.

---

# 5. Evidence compatibility review

PASS.

Step 3 inherits Step-2 evidence identity/state rather than reinterpreting it.

The copied referral remains a clinical handoff, not a literature report:

```text
evidence colours/bubbles/source/year/i-sheet
→ clinician-facing UI

selected intervention phrase
→ copied referral
```

Mixed-guideline states such as acupuncture do not leak `Οι οδηγίες διαφέρουν`, NICE/AAOS/ACR/VA-DoD labels or citations into copied prose.

---

# 6. Determinism / fixture review

PASS.

At the reviewed head, the design gate covers:

```text
12 primary deterministic render fixtures
3 additional power-user/preservation render fixtures
2 fail-closed unsupported-selection fixtures
2 negative Copy-readiness fixtures
```

The validator constructs exact Greek output from the machine contract and compares it byte-for-byte with expected synthetic output.

It also validates supported IDs against frozen CU-1 catalogs/UI scope, plan/adjunct IDs against the frozen Step-2 evidence contract, existing formal-diagnosis route requirements and the current true-locking safety-rule seam.

---

# 7. Diff / ownership review

PASS.

Exact compare:

```text
base / merge base = ab4b349223cd4c461837ab3125967a06d169a7e1
reviewed head     = cd4a42b4582921df7eb64d6ff3fb7c718a141c3a
behind            = 0
```

Changed scope is limited to:

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
Step-3 human design
Step-3 machine template contract
primary + edge synthetic fixtures
Step-3 validator
Step-3 design workflow
```

No modification exists to:

```text
clinic_utilities/physio_referral_runtime.py
clinic_utilities/physio_referral_api.py
existing CU-1 formatters
existing production static Physio UI
patient/database schemas
RF
Clinical Learning runtime
production configuration/secrets
```

---

# 8. Machine evidence

Latest reviewed exact-head machine gate:

```text
workflow: Physio Knee OA template design gate
run:      34561575795
head:     cd4a42b4582921df7eb64d6ff3fb7c718a141c3a
result:   SUCCESS
```

---

# 9. Final disposition

```text
STEP 3 DESIGN                         PASS
MATERIAL OPEN FINDING                 NONE
HUMAN/MACHINE CONTRACT CONSISTENCY    PASS
STEP-2 EVIDENCE COMPATIBILITY         PASS
POWER-USER LOSS-OF-SELECTION GUARD    PASS
COPY-READINESS AUTHORITY              PASS
TRUE-LOCKING SAFETY SEAM              EXPLICITLY DEFERRED / FAIL-CLOSED
RUNTIME IMPLEMENTATION                NOT AUTHORIZED
MERGE / DEPLOY                        NOT AUTHORIZED
```

Step 3 is eligible for design freeze/closeout.

Exact next product-design boundary after closeout:

```text
STEP 4 — Evidence Interaction / Traceability Layer
```

Step 4 may define the runtime-facing UI semantics for colour/state cues, compact bubbles, `i` evidence sheet, source/year/review-date display and conflict-state presentation. It must not begin runtime implementation merely because Step 3 passed design review.
