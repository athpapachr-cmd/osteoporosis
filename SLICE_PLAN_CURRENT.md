# SLICE_PLAN_CURRENT.md — CU-1 Greek human referral formatting maintenance v1

> **STATUS:** ACTIVE MAINTENANCE IMPLEMENTATION.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Area:** Clinic Utilities / Clinical Operations.
> **Slice ID:** CU-1 formatter-quality maintenance v1.
> **Maintenance base:** `d1716f8ea889a9369367c3bb18e469e9bbfef9f0`.
> **Writer:** `fix/cu1-greek-human-referral-formatting-2026-08-28`.
> **Clinical taxonomy:** frozen and unchanged.
> **Machine contract entrypoint:** `clinic_utilities/contracts/cu1_contract_manifest_v1.yaml`.
> **Formatter amendment:** `clinic_utilities/contracts/CU1_FORMATTER_LANGUAGE_EL_V1.md`.
> **Prior runtime slice:** CU-1 runtime v1 remains technically implemented/deployed but clinician-facing formatter acceptance failed.
> **CU-2:** not authorized.
> **PR-1:** remains paused.

---

# 1. Problem

The deployed CU-1 runtime produces technically valid referrals but the actual prose is not acceptable as a clinician-authored referral.

Observed defects:

```text
machine-like field serialization
English/machine-derived phrases in generated text
insufficient semantic/structural difference between Short and Detailed
```

The defect is in the formatter/display layer. It does not demonstrate a taxonomy, routing, safety, validation or persistence defect.

---

# 2. Objective

Deliver clinician-facing Greek referral text that is immediately usable after generation.

```text
validated ReferralDraftV1
→ unchanged clinical/safety semantics
→ Greek clinician-facing phrase resolution
→ natural Short or Detailed referral composition
→ copy / print
```

No raw machine ID may appear in final referral prose.

---

# 3. Short formatter contract

Short output should read like a routine referral written by a clinician.

Target:

```text
2–4 compact sentences
problem/presentation + laterality
most actionable findings/function
request for selected rehabilitation goals/directions
material restrictions if present
optional clinician note
```

It should not be a list of labeled database fields.

---

# 4. Detailed formatter contract

Detailed output should have a clearly different structure and information density.

Default shape:

```text
ΠΑΡΑΠΟΜΠΗ ΓΙΑ ΦΥΣΙΟΘΕΡΑΠΕΙΑ

Κλινική εικόνα
<problem + selected findings + functional impact + relevant secondary/context>

Περιορισμοί / προφυλάξεις
<only if present>

Στόχοι και κατευθύνσεις αποκατάστασης
<selected goals + core rehab directions + adjuncts only when explicitly selected>

Πρόσθετα κλινικά στοιχεία
<measurements, structural/postoperative context, clinician free text when useful>
```

Detailed output must contain materially more selected information than Short when such information exists, while retaining natural medical prose.

---

# 5. Greek phrase authority

Create a versioned machine-readable Greek phrase catalog for every selectable ID that can be rendered:

```text
findings
functional impairments
goals
rehab directions
adjuncts
restrictions
measurements
relevant context values
safety disposition wording if rendered
```

Rules:

```text
known renderable id + Greek label → render Greek phrase
known renderable id + missing Greek label → formatter contract error / fail closed
unknown id → existing validation already blocks
_humanize_id() must never be the final generated-referral fallback
```

Profile route display labels may be sourced from frozen profile display text when already Greek.

---

# 6. Invariants preserved

```text
suggested != examined
suggested != selected
selected != mandatory
symptom != diagnosis
objective deficit != subjective symptom
provocation/test finding != diagnosis
imaging finding != automatically symptomatic diagnosis
not_assessed != normal
adjunct != core rehabilitation
clinician-entered diagnosis may be carried faithfully but must not be inferred
explicit restrictions retain precedence
safety generation blocks remain unchanged
no persistence
```

---

# 7. Acceptance evidence

Executable tests must prove:

```text
1. representative outputs are Greek and natural
2. machine IDs/underscores do not leak into final prose
3. Short and Detailed are both deterministic
4. Short and Detailed are materially different for a rich draft
5. Detailed carries extra context/measurements/secondary problem when supplied
6. Short remains compact while retaining material restrictions
7. route labels/laterality are rendered naturally in Greek
8. not_assessed/unselected never generate reassuring negatives
9. existing gateway/safety/no-persistence suites remain green
```

Representative output tests:

```text
knee OA
cervical nonspecific pain
lumbar nonspecific pain
shared fracture + restriction
shared muscle injury
postoperative rehabilitation
```

Product-owner browser smoke after deploy must include visual inspection of the actual generated prose, not merely successful button execution.

---

# 8. REPLAN triggers

STOP and replan rather than silently expanding scope if:

```text
natural Greek prose requires changing clinical taxonomy
formatter needs to infer unselected findings/diagnoses
clinical-profile wording contradicts the machine contract
required Greek phrasing introduces a new clinical recommendation
```

---

# 9. Stop rule

```text
implementation
→ focused exact-head tests
→ independent exact-head review
→ MERGE-READY or BLOCK
```

Merge/deploy and final product-owner prose acceptance occur only after that gate.
