# SLICE_PLAN_CURRENT.md — G-4 Workspace Ergonomics + RF Utility Integration v1

> **STATUS:** IMPLEMENTED / TESTED / RELEASE REVIEW PASS — RELEASE PR ALLOWED; MERGE HOLD.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** Clinical Excellence workspace + cross-module Clinic Utilities.
> **Slice ID:** `M01-G4-WORKSPACE-ERGONOMICS-RF-UTILITY-v1`.
> **Production base:** `ab94c6286bdc49cb8304b072e557c5eb0a96b0c6`.
> **Branch:** `feat/module01-g4-collapsible-sticky-summary-rf-utility-2026-09-02`.
> **Exact tested runtime head:** `942d4e06944ebd6de97891cb8e2739c88ba85a38`.
> **Test workflow:** `G3 guidance salience longitudinal summary` run `33599860151` — SUCCESS.
> **Runtime writer:** NONE.

---

# 1. Trigger / evidence from use

After the G-3 hotfix was deployed, the product owner reported that all intended G-3 elements were visible and working well. Production interaction then produced the following ergonomics/integration needs:

1. `Σύνοψη ασθενούς` occupies too much vertical space and should be collapsible;
2. `Σημερινή ροή` should also be collapsible;
3. the patient summary should remain available at the top while scrolling;
4. the previously built radiofrequency-treatment PDF page should be accessible from the Cockpit.

This slice changes presentation/navigation only. It does not add or alter osteoporosis clinical rules.

---

# 2. Implemented G4-A — collapsible summary and current flow

A small UI-only module decorates the existing G-3 surfaces rather than replacing their renderer:

```text
static/baseline-audit/g4-workspace-ergonomics.js
```

Implemented behavior:

```text
Σύνοψη ασθενούς
→ native Σύμπτυξη / Ανάπτυξη button
→ heading remains visible when collapsed
→ summary calculation continues unchanged

Σημερινή ροή
→ independent native Σύμπτυξη / Ανάπτυξη button
→ heading remains visible when collapsed
→ VisitPlan calculation continues unchanged
```

Accessibility:

- native `button`;
- `aria-expanded` reflects current state;
- `aria-controls` targets the dynamic root;
- keyboard activation is native;
- default first-load state is expanded.

Per-browser collapse preference uses `sessionStorage` under `osteoporosis.workspace.ui.v1.*`. It is explicitly UI-only state, not clinical or patient persistence.

The helper observes G-3 DOM replacement and reapplies the controls when the existing renderer refreshes the summary/flow. It does not render clinical content itself.

---

# 3. Implemented G4-B — sticky patient summary

`Σύνοψη ασθενούς` remains the single existing summary surface and now uses sticky positioning:

```text
position: sticky
top: 8px
z-index: 30
max-height: calc(100vh - 16px)
overflow: auto
```

The existing summary background and a light shadow preserve readability over underlying content. On narrower screens the sticky surface uses a bounded viewport height.

Collapsed sticky state provides a compact header-only surface when maximal vertical workspace is desired.

No duplicate/floating summary owner was created.

---

# 4. Implemented G4-C — Radiofrequency PDF Clinic Utility navigation

Recovered prior implementation evidence identifies the original RF utility as a protected Clinic Operations workflow with:

```text
/rf
/rf/create
/rf/pdf/{application_id}
/rf/debug-grid/{template_key}
```

and official Medikey / DIROS / Thermedico PDF templates.

G4 intentionally does not clone that PDF engine. Instead the Cockpit sidebar receives a `Clinic Utilities` group with:

```text
Φυσιοθεραπεία
→ /clinical/clinic-utilities/physio-referral

Ραδιοκύματα — PDF
→ https://ortho-reception-backend-v2.onrender.com/rf
```

The RF utility opens in a new tab with `noopener noreferrer`, preserving the active clinical encounter in the Cockpit. The original service remains the RF PDF/request source of truth.

No RF URL, template content, request state or patient-history content enters the osteoporosis encounter payload.

A later full migration of the RF engine into this repository remains out of scope unless its complete source/templates/auth/storage contract is deliberately recovered and reviewed.

---

# 5. Runtime boundary

Changed runtime/test files:

```text
static/baseline-audit/app.js
static/baseline-audit/g4-workspace-ergonomics.js
static/baseline-audit/progressive-guidance.css
test_g4_workspace_ergonomics.js
.github/workflows/g3-guidance-summary-tests.yml
```

Canonical state files changed separately for the slice lifecycle.

No changes to:

```text
G-2 evidence rules or thresholds
G-3 clinical summary derivation
G-1 VisitPlan semantics
C1 Finish/finalization
patient APIs / DB schema
C2 persistence runtime
PR-1 / PR-2
RF PDF engine/templates
```

---

# 6. Acceptance evidence

Exact tested runtime head:

```text
942d4e06944ebd6de97891cb8e2739c88ba85a38
```

Workflow:

```text
G3 guidance salience longitudinal summary
run 33599860151
SUCCESS
```

Passed G-4 assertions:

- summary root has accessible collapse-control wiring;
- current-flow root has independent accessible collapse-control wiring;
- collapse state is session UI preference only;
- helper owns no clinical fetch or encounter write;
- patient summary has sticky CSS contract;
- collapsed CSS retains the relevant heading;
- `Clinic Utilities` navigation includes the existing CU-1 physiotherapy utility;
- RF target is exactly the recovered protected `/rf` utility URL;
- RF opens with safe new-tab isolation;
- G-4 does not implement a second clinical summary/guidance owner.

Inherited gates also passed:

- original G-3 salience/summary;
- G-3 same-card/new-evidence salience;
- G-3 production visibility/cache;
- frozen G-2 evidence contract/runtime;
- G-1 core/wiring/UI/WHY-NOW;
- C1 authoritative Finish browser;
- server finalization lifecycle.

---

# 7. Release-readiness review

Fresh production base remained:

```text
ab94c6286bdc49cb8304b072e557c5eb0a96b0c6
```

The code/test review passed with no runtime defect or scope leakage. The first review identified stale G-3 roadmap/history state in `TODO.md` / changelog; those were reconciled docs-only. The changelog reconciliation was append-only.

Post-reconciliation exact-head checks must satisfy:

```text
branch behind production main = 0
merge base = exact production main
post-tested-runtime changes = canonical docs only
no runtime/test/workflow drift after 942d4e06...
```

The external RF host could not be independently HTTP-probed from the assistant execution environment during review, so successful navigation to the protected RF page remains an explicit production-smoke criterion rather than an assumed fact.

---

# 8. Completion matrix

```text
G-4 design                                  COMPLETE
G-4 implementation                          YES
G-4 focused regression                      PASS
G-3 inherited regression                    PASS
G-2 inherited contract/runtime              PASS
G-1 inherited regressions                   PASS
C1 inherited finalization                   PASS
Exact-head delta review                     PASS
Canonical reconciliation                    PASS
Product-owner release review                PASS
PR opened                                   NO
Merged                                      NO
Deployed                                    NO
Production-smoke-verified                   NO
```

---

# 9. Stop gate

G-4 is release-ready for a bounded PR, with no active writer.

Next allowed action:

```text
open G-4 release PR
→ verify exact PR-head checks
→ STOP before merge unless product owner explicitly authorizes merge
```

Until explicit merge authority:

```text
NO MERGE
NO DEPLOY
```
