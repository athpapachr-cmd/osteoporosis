# SLICE_PLAN_CURRENT.md — G-4 Workspace Ergonomics + RF Utility Integration v1

> **STATUS:** DESIGN-COMPLETE / IMPLEMENTATION ACTIVE.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** Clinical Excellence workspace + cross-module Clinic Utilities.
> **Slice ID:** `M01-G4-WORKSPACE-ERGONOMICS-RF-UTILITY-v1`.
> **Production base:** `ab94c6286bdc49cb8304b072e557c5eb0a96b0c6`.
> **Branch:** `feat/module01-g4-collapsible-sticky-summary-rf-utility-2026-09-02`.
> **Runtime writer:** ACTIVE — bounded files listed below.

---

# 1. Trigger / evidence from use

After the G-3 hotfix was deployed, the product owner reported that all intended G-3 elements were visible and working well. Real production interaction then produced three ergonomics/integration needs:

1. `Σύνοψη ασθενούς` occupies too much vertical space and must be collapsible;
2. `Σημερινή ροή` must be collapsible for the same reason;
3. the patient summary should remain available at the top while scrolling;
4. the previously built radiofrequency-treatment PDF page should be accessible from the Cockpit.

This slice changes presentation/navigation only. It does not add or alter osteoporosis clinical rules.

---

# 2. G4-A — collapsible summary and current flow

Both dynamic top surfaces remain owned by the existing G-3 guidance UI.

Required behavior:

```text
Σύνοψη ασθενούς
→ accessible expand/collapse control
→ body hidden when collapsed
→ heading remains visible
→ no loss/recomputation of clinical state caused by collapse

Σημερινή ροή
→ accessible expand/collapse control
→ body hidden when collapsed
→ heading remains visible
→ underlying VisitPlan continues updating while collapsed
```

Accessibility contract:

- native button control;
- `aria-expanded` reflects state;
- `aria-controls` targets the collapsible body;
- keyboard activation works through native button behavior;
- collapse state is UI-only and never patient data.

Default v1 state: expanded on first load. Per-browser UI preference may be retained locally but is not authoritative clinical persistence.

---

# 3. G4-B — sticky patient summary

`Σύνοψη ασθενούς` remains the single existing summary surface and becomes sticky within the main encounter scroll context.

```text
scroll down encounter
→ patient summary header/surface remains available at top
→ collapse control remains reachable
```

Constraints:

- no duplicate/floating second summary renderer;
- sticky surface must preserve background, z-index and readability over underlying cards;
- responsive layout must not obscure the full viewport on smaller screens;
- collapsed sticky state is the compact fallback if the user wants maximal workspace.

---

# 4. G4-C — Radiofrequency PDF Clinic Utility integration

The prior RF workflow is a Clinic Utilities / Clinical Operations tool, not osteoporosis encounter state.

Recovered prior source evidence establishes an existing protected utility with:

```text
/rf
/rf/create
/rf/pdf/{application_id}
/rf/debug-grid/{template_key}
```

and official Medikey / DIROS / Thermedico PDF templates. The original implementation generated PDFs with its calibrated source/templates and maintained its own request/history semantics.

G4 v1 therefore integrates by navigation rather than duplicating the RF engine:

```text
Cockpit sidebar
→ Clinic Utilities section
→ Φυσιοθεραπεία
→ Ραδιοκύματα — PDF
```

RF target for v1:

```text
https://ortho-reception-backend-v2.onrender.com/rf
```

The RF link opens the existing protected utility in a new tab/window. This avoids mixing external RF authentication, PDF templates, request persistence or patient-history ownership into the osteoporosis encounter payload.

A later migration into this repository is explicitly out of scope unless the complete RF source/templates/auth/storage contract is recovered and reviewed.

---

# 5. Runtime files allowed

Expected bounded mutation:

```text
static/baseline-audit/progressive-guidance-ui.js
static/baseline-audit/progressive-guidance.css
static/baseline-audit/index.html
optional small UI helper only if needed
test_g4_workspace_ergonomics.js
existing G3 workflow extended to run G4 regression
canonicals for slice state
```

No clinical API or DB mutation is required.

---

# 6. Acceptance tests

Must prove at minimum:

1. patient summary has accessible collapse/expand control;
2. current flow has accessible collapse/expand control;
3. collapse changes visibility only, not VisitPlan/summary calculation state;
4. patient summary has sticky presentation contract;
5. RF utility navigation exists under Clinic Utilities and points exactly to the existing protected `/rf` page;
6. no RF URL/content is written into encounter state;
7. G-3 `Νέο`, patient summary and production visibility/cache regressions remain green;
8. G-2/G-1/C1 inherited regressions remain green.

---

# 7. Explicit exclusions

```text
NO NEW OSTEOPOROSIS CLINICAL RULES
NO RF PDF ENGINE REIMPLEMENTATION
NO RF TEMPLATE COPYING FROM MEMORY
NO RF DATA IN OSTEOPOROSIS ENCOUNTER PAYLOAD
NO C2 RELEASE / REBASE IN THIS SLICE
NO PR-1 / PR-2
NO MERGE / DEPLOY WITHOUT SEPARATE AUTHORITY
```

---

# 8. Stop gate

Implementation may proceed on the active branch through focused/full regression and exact-head review.

After `IMPLEMENTED / TESTED` closeout:

```text
release writer lock
→ STOP before PR/merge/deploy unless separately authorized
```
