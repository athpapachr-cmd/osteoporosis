# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 EVIDENCE-BACKED GUIDANCE RUNTIME IMPLEMENTATION ACTIVE.
> **Updated:** 2026-08-31 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design-complete ancestry:** `design/module01-g2-evidence-backed-guidance-2026-08-31 @ 0395e52ed75f835d49713504df3df4ce51183edf`.
> **Implementation branch:** `feat/module01-g2-evidence-guidance-runtime-2026-08-31`.
> **Design contract CI:** run `33358433732` — SUCCESS on the reviewed contract ancestry.
> **ACTIVE CANONICAL WRITER/LOCK:** THIS SESSION — G-2 runtime slice canonicals only.
> **ACTIVE RUNTIME WRITER/LOCK:** THIS SESSION — G-2 evidence-guidance runtime integration only.

---

# 1. Production base retained

C1 authoritative Finish and G-1 progressive guidance remain:

```text
IMPLEMENTED
TESTED
MERGED
DEPLOYED
PRODUCTION-SMOKE-VERIFIED
```

The G-2 implementation must preserve C1/G-1 ownership, history-availability semantics, live-DOM-over-cache semantics, and existing `Γιατί τώρα:` rendering.

---

# 2. Active slice

```text
M01-G2-EVIDENCE-GUIDANCE-RUNTIME-v1
```

Objective:

```text
existing G-1 longitudinal projection
+ live current encounter snapshot
→ G-2 osteoporosis evidence context
→ pure deterministic evidence evaluator
→ evidence contributions
→ merge with existing G-1 Visit Plan
→ existing Σημερινή ροή / Γιατί τώρα UI
```

This is runtime activation of the already reviewed evidence contract, not a new clinical-content design exercise.

---

# 3. Allowed mutation scope

Allowed:

- bounded browser-side G-2 evidence context/evaluator;
- minimal G-1 integration needed to merge evidence contributions;
- live-state snapshot extensions for fields required by reviewed rules;
- focused synthetic/unit/wiring regressions;
- CI workflow for G-2 runtime tests;
- operational/canonical state updates for this slice.

Not allowed:

- new treatment recommendation semantics;
- new evidence thresholds or clinic policy;
- R15/R16 activation;
- CTX 280/300 automatic retreatment;
- generic Prolia 4th/8th/10th milestones;
- PR-1 transcript runtime;
- PR-2 inline population;
- database/schema migration unless a REPLAN trigger proves it necessary;
- physiotherapy/RF mutation;
- release PR, merge or deploy.

---

# 4. Frozen clinical/runtime safeguards

```text
GUIDANCE != AUTOMATIC TREATMENT DECISION
CHECKLIST GUIDANCE != SAFETY CLEARANCE
MISSING / UNKNOWN != NEGATIVE
SCHEDULED / PLANNED DOSE != ACTUAL DOSE
ADMINISTRATION COUNT != ELAPSED EXPOSURE
LIVE CURRENT CONTROL, INCLUDING BLANK > PERSISTED CACHE
```

Additional first-runtime rules:

- NOGG threshold semantics require NOGG framework/scope guard;
- R01 formal-risk evidence rule requires explicit indication;
- denosumab six-month due uses reliable exact actual date and is ephemeral;
- R24 >7-month rebound escalation requires reliable actual denosumab count ≥2;
- approximate treatment duration must not generate exact timing milestones;
- safety checklist rules may say verify/review/confirm but must not infer `safe/cleared` from absent or stale data;
- no automatic selected-agent write or treatment-failure label.

Blocked in runtime v1:

```text
OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP
OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION
```

---

# 5. Current status

```text
G-2 DESIGN                               COMPLETE / REVIEWED
G-2 MACHINE CONTRACT                     PASS
G-2 RUNTIME IMPLEMENTATION               STARTED
G-2 RUNTIME TESTED                       NO
G-2 MERGED                               NO
G-2 DEPLOYED                             NO
G-2 PRODUCTION-SMOKE-VERIFIED            NO
PILOT-VALIDATED                          NO
```

No runtime files have yet been changed on this implementation branch at the moment of lock claim.

---

# 6. Exact next action

```text
inspect exact Step 1/2/3/4 live DOM + persisted runtime seams
→ confirm every reviewed trigger can obey live-over-cache ownership
→ implement pure G-2 evidence context/evaluator
→ merge contributions into the single existing G-1 render owner
→ add focused G-2 regressions + inherited G-1/C1 regression workflow
→ exact-head CI
→ canonical closeout
→ STOP at IMPLEMENTED / TESTED gate
```

A discovered mismatch that invalidates a frozen trigger, owner, source path or safety assumption is a **REPLAN trigger**. Do not patch around it.

No PR/merge/deploy authority is active.
