# CURRENT.md — OST-UI / Osteoporosis Product Reconstruction

> **STATUS:** R2 COMPLETE / COORDINATOR RECONCILIATION PENDING; R3 NOT STARTED.
> **Updated:** 2026-10-01 Asia/Nicosia.
> **Workstream:** OST-UI.
> **Fresh runtime main reviewed for R2:** **63e903e05c1bfe22ca925374b8994355f6c92baf**.
> **Governing bootstrap PR:** **#122** — OPEN / DRAFT / NOT MERGED.
> **Governing contract head:** **9363ad5dbb5f7c059f351421599bd6ede4db6438**.
> **R1 review branch:** **docs/ost-ui-r1-current-product-audit-2026-09-27**.
> **R1 artifact commit:** **7b23f224cef0c9b721d89c84e431a4582efab2af**.
> **R2 branch:** **docs/ost-ui-r2-longitudinal-trajectory-review-2026-10-01**, based on coordinator head **f04af35b5424328f55a11932224becb43521d831**.
> **R2 artifact commit:** **688b851358dae42b3431aa1fd50c86a077788322**.
> **Root operational writer:** unchanged — PR-1 Heidi-first transcript capture lifecycle.
> **OST-UI runtime writer:** none.
> **Release affecting:** no.

---

## 0. Current R2 checkpoint / exact next action

R2 independently fresh-verified remote `main` at `63e903e05c1bfe22ca925374b8994355f6c92baf` and read all six root canonicals in `AGENTS.md` order. The root phase remains Module-01 closure; the root active slice/writer remains bounded PR-1 implementation. R2 made no overlapping mutation. PR #121 is open at `aa32f7fbd49c02653c11eb35edaf2e24939eb443`; independently reviewed LifeCourse P1 target `b4f917b161f33b0bfb3507328df07cec2d7bd2b6` remains the PASS semantic input. Current-main merged S1 fracture/fragility behavior was inspected directly. Full source identity and the Canonical Bootstrap Manifest are in the R2 artifact.

Completed review artifact:

`programme/OST-UI/R2-LONGITUDINAL-CLINICAL-TRAJECTORY-REVIEW.md` at commit `688b851358dae42b3431aa1fd50c86a077788322`.

Disposition:

**R2 COMPLETE / R3 ELIGIBLE FOR COORDINATOR RECONCILIATION**

R2 found a usable protected patient/encounter/lab substrate, reliable captured actual-administration distinctions, S1-corrected fracture semantics, and helpful read-only history-sensitive projections. It also found that a 10–15-year complete story cannot be guaranteed from the present treatment-epoch, obligation, decision/preference/action, goals, outcome-linkage, investigation-provenance and historical-evidence semantics. The artifact classifies reuse, consolidation, extension, genuine gaps, clinical dependencies and unknowns without selecting an implementation architecture.

**Exact next action:** programme coordinator verifies this R2 branch, final head and artifact, reconciles its disposition into the OST-UI control plane, and only then decides whether to authorize R3. R3, R4 and synthesis remain not started. R2 stops here. No runtime/schema/database, root canonical, PR #121, clinical-rule, PR-1/H-12, merge or deploy action is authorized by this checkpoint.

Sections 1–8A below retain the prior R1/coordinator history as context. This §0 and §9 own the present OST-UI NOW.

---

## 1. R1 Canonical Bootstrap Manifest (historical)

Fresh remote athpapachr-cmd/osteoporosis/main was verified at:

**2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b**

The six active root canonicals were consumed in the required AGENTS.md order:

1. AGENTS.md
2. TODO.md
3. CLINICAL_EXCELLENCE_PLAN.md
4. SLICE_PLAN_CURRENT.md
5. CURRENT_OPERATIONAL.md
6. osteoporosis-change-log.md

### Root phase / writer lock

The root product remains in Module-01 closure / dynamic guided consultation + transcript-assisted capture → Practice Review → measurement/improvement-loop architecture.

The active root slice remains:

**PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16 — ACTIVE / DESIGN VERIFIED / IMPLEMENTATION AUTHORIZED**

CURRENT_OPERATIONAL.md remains the sole repo-wide NOW/writer lock for that PR-1 lifecycle.

OST-UI did not take or alter the root writer lock.

### Permanent invariants preserved during R1

- one root operational writer for overlapping mutation scope;
- OST-UI is a parallel documentation/review workstream only;
- clinical guidance != transcript capture != audit != Practice Review;
- patient truth is not invented from missing data;
- scheduled/planned treatment != actual administration;
- clinical semantics are not changed for UI convenience;
- draft PR material is not silently promoted to current-main authority;
- raw transcript/privacy ownership remains outside OST-UI.

---

## 2. Exact R1 authority and source identity

### Governing OST-UI contract

Draft PR #122 was fresh-verified at exact head:

**9363ad5dbb5f7c059f351421599bd6ede4db6438**

Consumed fully:

- programme/OST-UI/PROJECT-INDEX.md
- programme/OST-UI/CURRENT.md

PR #122 remains open, draft and unmerged.

### Supporting product intent

Draft PR #121 was fresh-verified at exact supporting head:

**80851762f39105628ab9009406cc7828b3e9be19**

programme/PRODUCT-CONSTITUTION-V0.2.md was used only as supporting Product Constitution intent, not current-main runtime authority.

### R1 output identity

Artifact:

**programme/OST-UI/R1-CURRENT-PRODUCT-CONSTITUTION-AUDIT.md**

Artifact commit:

**7b23f224cef0c9b721d89c84e431a4582efab2af**

Review branch:

**docs/ost-ui-r1-current-product-audit-2026-09-27**

---

## 3. R1 current-state disposition

R1 established the current Osteoporosis product as a **hybrid**:

- genuine protected longitudinal patient/encounter/lab substrate;
- real history-sensitive G1/G2/G3/G4 decision-support behavior;
- read-only patient longitudinal summary;
- current Visit Plan / “Γιατί τώρα” / evidence provenance;
- actual-administration and unresolved-task projection;
- but still presented primarily through the visible Baseline Audit / six-step encounter shell.

R1 therefore found:

**current longitudinal substrate: materially present**

and simultaneously:

**longitudinal Product Constitution: only partially expressed as the primary clinician-facing product model**

This is a current-state diagnosis only. R1 does not decide redesign scope.

### Material OBSERVED findings checkpointed

1. Global Clinical Excellence Home correctly exists and Osteoporosis is Module 01.
2. Protected server persistence exists for patients, encounters and laboratory snapshots.
3. Completed/amended history materially changes current guidance.
4. Scheduled-only treatment is not counted as actual administration.
5. Prior tasks, explicit due states and longitudinal conflicts can resurface.
6. G3 exposes course/fracture-risk/DXA/treatment/labs/last-decision/unresolved summary.
7. Current draft is kept distinct from completed historical truth.
8. The visible Module-01 identity remains Baseline Audit / pilot / six-step encounter capture.
9. Local “Case” vocabulary and protected patient/encounter vocabulary coexist.
10. The static Privacy dialog retains older prototype/localStorage wording despite protected clinical-mode sync.
11. Global/module separation is materially improved but not complete: calendar-link.js still dynamically inserts the Physio referral into the Osteoporosis sidebar even though Physio already exists as a global Clinic Utility.
12. The full patient trajectory is partly visible but still substantially reconstructable across summary, encounter history, trends and step-specific state.

### R1 boundary on deeper questions

R1 did **not** classify the following as defects:

- encounter payloads vs first-class durable events/state;
- treatment-epoch sufficiency;
- obligation identity/disposition sufficiency;
- shared patient/Core ownership;
- overlapping longitudinal DXA/FRAX representations;
- historical evidence-at-decision persistence.

Those are routed to R2/R4 according to the existing programme contract.

---

## 4. Preservation / reuse checkpoint

R1 identified the following existing mechanisms for explicit later R4 KEEP/ADAPT investigation:

- protected patient/encounter/lab persistence;
- completed/amended finalization semantics;
- fail-closed historical loading;
- G1 longitudinal projection;
- actual-vs-scheduled administration semantics;
- EncounterContext → VisitPlan → “Γιατί τώρα”;
- G2 evidence/rule/provenance layer;
- G3 longitudinal patient summary and salience;
- G4 sticky/collapsible workspace mechanics;
- longitudinal lab and DXA/FRAX trend mechanisms;
- Global Cockpit shell;
- selected Physio interaction primitives such as progressive disclosure, direct manipulation, live downstream output and evidence-on-demand.

No final R4 KEEP/REPLACE classification was made.

---

## 5. Phase-1 review state

| Review | Status | Output |
|---|---|---|
| R1 — Current Product vs Constitution | **COMPLETE / COORDINATOR ACCEPTED** | R1-CURRENT-PRODUCT-CONSTITUTION-AUDIT.md |
| R2 — Longitudinal Clinical Trajectory | **COMPLETE / COORDINATOR RECONCILIATION PENDING** | R2-LONGITUDINAL-CLINICAL-TRAJECTORY-REVIEW.md |
| R3 — Point-of-Care Interaction | **NOT STARTED** | R3-POINT-OF-CARE-INTERACTION-REVIEW.md |
| R4 — Shared Core / Module Architecture / Reuse | **NOT STARTED** | R4-SHARED-CORE-MODULE-ARCHITECTURE-REUSE.md |
| Synthesis — Product Reconstruction Decision | **BLOCKED ON R1-R4** | OST-PRODUCT-RECONSTRUCTION-DECISION.md |

No additional review lane is authorized.

---

## 6. Dependency state

### Fracture / fragility

Outside OST-UI clinical authority.

R1 inspected how fracture fields/events affect UI/projection only. R2 consumed reviewed LifeCourse P1 current-state ownership and fresh-inspected the merged S1 executable semantics on `main`. R3 must use the reconciled clinical owner and may not invent new fracture/fragility meaning.

### Treatment / evidence

Existing reviewed G2 / clinical contracts remain authoritative. OST-UI does not redefine treatment or guideline semantics.

### Shared Core / patient state

R4 may assess ownership movement/generalization. R1 made no Core ownership decision.

### PR-1 / transcript / future Live Copilot

PR-1 remains the active root lifecycle and read-only architecture context for OST-UI.

No PR-1 change was made.

Live Copilot remains a future compatibility constraint only.

### PR #121 Product Constitution / parallel programme

PR #121 remains unmerged supporting context. No OST-LIFECOURSE or OST-CLINICAL conclusion was promoted to current-main clinical truth by R1.

---

## 7. Historical R1 programme coordinator reconciliation

At the 2026-09-27 R1 checkpoint, programme reconciliation verified:

- runtime `main` remained `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`;
- governing PR #122 remained exactly `9363ad5dbb5f7c059f351421599bd6ede4db6438`;
- R1 branch head was exactly `4d4991ee617e72c3a943601864ba3c3c1d8062a6`;
- the R1 branch was exactly two commits ahead of the governing head;
- the delta was limited to this `CURRENT.md` plus the R1 artifact;
- no runtime, schema, root-canonical or clinical-semantic mutation occurred.

Coordinator disposition:

```text
R1: ACCEPTED / COMPLETE
REPLAN REQUIRED: NO
R2: AUTHORIZED / NOT STARTED
R3/R4/SYNTHESIS: NOT AUTOMATICALLY AUTHORIZED
ROOT PR-1 WRITER: UNCHANGED
```

Accepted R1 diagnosis:

- the current product contains a real longitudinal substrate and history-sensitive decision-support layer;
- the clinician-facing Module-01 shell remains encounter/audit-first and therefore only partially expresses the longitudinal Product Constitution;
- existing persistence, projection, guidance, provenance, summary and workspace mechanisms are preservation/reuse candidates rather than presumptive rewrite targets;
- deeper questions about durable treatment epochs, obligations, goals, shared patient/Core ownership, overlapping longitudinal representations and historical evidence context remain questions for R2/R4, not R1-proven defects.

Dependency state at that reconciliation:

- OST-LIFECOURSE P1 corrected ownership map has now passed fresh independent delta+cumulative review at exact corrected target `b4f917b161f33b0bfb3507328df07cec2d7bd2b6`; DXA/task residuals are closed and the reviewed ownership map is consumable by R2;
- OST-CLINICAL S1 had completed its independent review/release path and was merged on then-current `main`; the later R2 inspection used the fresh `main` identified in §0;
- R2 was authorized as a bounded trajectory-capability review, conditional on fresh owner verification and no promotion of pending/unmerged conclusions into current-main clinical truth.

## 8. Historical R2 authorization, now fulfilled

At that checkpoint, R2 was authorized as the only next substantive OST-UI action. The requirements below were fulfilled in the R2 artifact named in §0.

The authorized review had to:

- consume R1 as established current-product evidence;
- inspect the current longitudinal runtime/persistence mechanisms directly;
- test representative longitudinal journeys across time;
- distinguish EVENT / STATE / DERIVED STATE / DECISION / PLAN / FUTURE OBLIGATION / OUTCOME / UNCERTAINTY;
- determine what is durable truth versus reconstructed/projection/transient state;
- mark unresolved fracture/fragility or other clinical semantics as dependencies rather than resolving them inside OST-UI;
- consume the independently reviewed LifeCourse P1 ownership boundaries as reviewed current-state semantic input;
- still fresh-inspect current main for executable truth and do not convert P1 into P2 architecture;
- keep unresolved future design questions (Q1–Q12) parked.

The authorization ended at the artifact plus this local checkpoint; R2 now returns to the programme coordinator.

## 8A. LifeCourse dependency closure checkpoint

Programme coordinator reconciliation confirms:

```text
OST-LIFECOURSE P1 CORRECTED TARGET
b4f917b161f33b0bfb3507328df07cec2d7bd2b6

INDEPENDENT DELTA+CUMULATIVE REVIEW
PASS

DXA OWNERSHIP RESIDUAL
CLOSED

TASK CONTINUITY RESIDUAL
CLOSED

Q1–Q12
PARKED

R2 CONSUMABLE OWNERSHIP INPUT
READY
```

R2 used the reviewed **current-state ownership distinctions** and fresh-verified runtime source.

This does not authorize P2 target architecture, migration, new stores, new obligation engine, or UI implementation.

## 9. Current stop / hold

R2 is complete on its separate branch and stops after this local checkpoint. The programme coordinator's review of the exact branch/head/artifact is next; R3 is eligible for coordinator reconciliation, not automatically started. R4 and synthesis remain downstream. No screen redesign, prototype, implementation, schema/database change, root-canonical or PR #121 mutation, clinical-rule change, new store, PR-1/H-12 mutation, merge or deploy belongs to this R2 lane.
