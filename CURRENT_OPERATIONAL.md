# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-28 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified current main:** `58f41b5b29ee61280f1557df480989c5830465ba`.
> **Prior CU-1 formatter-quality fix:** PR #61 squash-merged as `58f41b5b29ee61280f1557df480989c5830465ba`.
> **New product-quality defects:** routine referral generation can be blocked by avoidable field friction, and the form exposes global unrelated option catalogs instead of dynamically relevant controls.
> **Current major phase:** bounded CU-1 dynamic-form / generation-friction maintenance.
> **CU-1 status:** REOPENED FOR UX/VALIDATION-PRESENTATION CORRECTION.
> **ACTIVE CANONICAL WRITER/LOCK:** `fix/cu1-dynamic-relevant-fields-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** `fix/cu1-dynamic-relevant-fields-2026-08-28`.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Defect statement

Product-owner testing exposed two clinically meaningful usability defects:

```text
1. routine generation can fail because the UI leaves avoidable contract fields unresolved instead of choosing a safe/default presentation path or explaining the one truly required field in context
2. findings/functions/goals/rehab/adjuncts/restrictions/safety controls are rendered from global catalogs, so unrelated options appear for the selected region/pathway
```

Example: selecting Elbow must not expose walking/stair/weight-bearing/cervical-only controls.

This is an execution/workflow defect. It does not by itself reopen the frozen clinical taxonomy.

---

# 2. Authorized maintenance boundary

Authorized changes:

```text
CU-1 browser interaction model
contract-driven UI option scoping / progressive disclosure
auto-selection of the safest allowed wording mode when this does not create a diagnosis assertion
contextual display of route-required structural/postoperative fields
routine-generation friction reduction without weakening genuine safety gates
human-readable inline validation guidance
focused synthetic UX/contract tests
canonical/changelog reconciliation after verified fix
```

Explicitly out of scope:

```text
new diagnoses or clinical pathways
new evidence-sensitive rehabilitation recommendations
silent inference of diagnosis/findings/normal examination
removal of genuine structural/postoperative safety requirements
referral persistence / patient-registry linkage
CU-2 work
PR-1 work
```

---

# 3. UX acceptance contract

```text
region not selected → no downstream clinical option catalogs shown
region selected + route not selected → only route choice shown
route selected → only options relevant to that profile/route are shown
routine presentation route → generation possible without findings/goals/rehab selections
multiple wording modes → safest non-assertive presentation mode selected automatically when available
formal diagnosis assertion → explicit clinician confirmation only when actually required
structural/postoperative route → only its required context fields appear, with plain-Greek explanation
advanced safety/restriction controls → collapsed or surfaced only when relevant
```

No irrelevant global option should remain visible simply because it exists in the machine catalog.

---

# 4. Generation blocking policy

Blocking remains appropriate for:

```text
missing primary region/route
required formal-diagnosis assertion when no non-assertive mode applies
required structural disposition/source where the frozen route contract requires it
postoperative protocol/restriction state required for safe generic rehabilitation
fracture loading/use status required by anatomical site
active safety rule whose frozen severity requires acknowledgement or disposition
```

Blocking is NOT appropriate merely because optional findings, functional impairments, goals, rehab directions, adjuncts, measurements or clinician notes were not selected.

---

# 5. Exact next action

```text
1. add explicit contract-driven UI relevance scope
2. implement progressive-disclosure rendering and safe wording defaults
3. improve contextual required-field guidance
4. add focused tests proving unrelated controls are absent and routine generation remains possible
5. run exact-head CI
6. independent exact-head review
7. STOP at MERGE-READY or BLOCK
```
