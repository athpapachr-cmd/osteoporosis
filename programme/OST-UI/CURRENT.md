# CURRENT.md — OST-UI / Osteoporosis Product Reconstruction

> **STATUS:** CONTROL PLANE BOOTSTRAPPED / DRAFT PR PENDING / R1 NOT STARTED.
> **Updated:** 2026-09-27 Asia/Nicosia.
> **Workstream:** `OST-UI`.
> **Bootstrap main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Branch:** `docs/ost-ui-product-reconstruction-bootstrap-2026-09-27`.
> **Root operational writer:** unchanged — PR-1 Heidi-first transcript capture lifecycle.
> **OST-UI runtime writer:** none.
> **Release affecting:** no.

---

## 1. Canonical Bootstrap Manifest

Fresh remote `athpapachr-cmd/osteoporosis/main` was verified at:

```text
2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b
```

The six active root canonicals were read in the required order:

1. `AGENTS.md`
2. `TODO.md`
3. `CLINICAL_EXCELLENCE_PLAN.md`
4. `SLICE_PLAN_CURRENT.md`
5. `CURRENT_OPERATIONAL.md`
6. `osteoporosis-change-log.md`

### Current major phase

The root product remains in Module-01 closure / dynamic guided consultation + transcript-assisted capture → Practice Review → measurement/improvement-loop architecture.

### Active root slice

`SLICE_PLAN_CURRENT.md` identifies:

```text
PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16
ACTIVE / DESIGN VERIFIED / IMPLEMENTATION AUTHORIZED
```

### Active writer / lock

`CURRENT_OPERATIONAL.md` remains the sole repo-wide operational NOW/writer lock and owns the bounded PR-1 transcript implementation lifecycle.

OST-UI does **not** take or alter that lock.

### Relevant permanent invariants

- one root operational writer for overlapping mutation scope;
- parallel workstreams may maintain local `CURRENT.md` state without replacing root NOW;
- clinical guidance != transcript capture != audit != Practice Review;
- patient truth must not be silently invented from missing data;
- scheduled/planned treatment != actual treatment;
- clinical semantics are not changed for UI convenience;
- raw identifiable transcript/privacy boundaries remain owned outside OST-UI;
- current reviewed evidence/clinical contracts remain authoritative in their own scope;
- historical chat or draft PR content is not canonical truth.

### Exact current OST-UI authority

Allowed now:

```text
read current product/runtime/source
+ maintain programme/OST-UI/PROJECT-INDEX.md
+ maintain programme/OST-UI/CURRENT.md
+ define R1-R4 + one synthesis
+ open documentation-only draft PR
```

Not authorized now:

```text
UI redesign
prototype code
runtime/schema/database mutation
root canonical mutation
clinical-semantic mutation
merge/deploy
extra review lanes
root writer transfer
```

---

## 2. Current-state inspection completed for bootstrap

The bootstrap inspected enough current source to define bounded review inputs. These are **seed observations for review scoping**, not final R1 findings.

### OBSERVED seed — global shell exists

Current `static/cockpit/` provides a reusable Clinical Excellence Home with:

- Today / Clinical Calendar;
- Clinical Modules;
- Learning & Improvement;
- global Clinic Utilities;
- separate Reception link.

The Osteoporosis entry is presented as Module 01 rather than the whole Cockpit.

### OBSERVED seed — current Osteoporosis surface retains baseline-form heritage

Current `static/baseline-audit/index.html` still presents:

- `Osteoporosis Cockpit`;
- `Baseline Audit v1 · prospective encounter capture`;
- pilot/baseline messaging;
- New Case / Cases / disabled KPI Overview / disabled Library sidebar entries;
- six Step tabs from encounter summary through documentation/Heidi.

This is an observed presentation fact only. It does not by itself establish that a rebuild is required.

### OBSERVED seed — protected durable patient/encounter/lab storage exists

Current server persistence in `clinical_data.py` stores:

```text
clinical_patients
clinical_encounters
clinical_lab_snapshots
```

Encounter clinical content is persisted primarily in `payload_json`; labs also have dedicated dated snapshots. Completed/amended finalization semantics are protected server-side.

### OBSERVED seed — longitudinal state is already partially derived from history

Current browser mechanisms include:

- `progressive-guidance-core.js`;
- `progressive-guidance-ui.js`;
- `osteoporosis-longitudinal-summary-core.js`;
- `osteoporosis-evidence-guidance-core.js`.

They read completed/amended historical encounters, derive treatment/administration context, detect selected conflicts, produce longitudinal summaries and influence current guidance/Visit Plan behavior.

### REVIEW QUESTION — durable semantic state vs derived projection

The current backend persistence is encounter/payload centric while important longitudinal current-state interpretation is derived by browser/runtime projection.

R2/R4 must determine, from complete evidence, whether this is sufficient for the target product model or whether some clinically meaningful state/event/obligation concepts require deeper durable representation.

This is deliberately **not pre-classified as a defect**.

### OBSERVED seed — current guidance/evidence machinery is reusable evidence

G-1/G-2/G-3/G-4 already provide reviewed mechanics for:

- encounter-context-sensitive flow;
- `why now`;
- evidence-backed guidance;
- longitudinal summary;
- salience of newly surfaced guidance;
- collapsible/sticky workspace behavior.

R1/R4 must classify these mechanisms rather than assuming replacement.

### OBSERVED seed — Physio provides interaction/reuse evidence

The current Knee-OA Physio product carries an explicit UX contract and live product implementation with patterns including:

- progressive disclosure;
- direct manipulation;
- live downstream output without a Generate step;
- compact evidence-on-demand;
- upstream changes updating downstream prose;
- projection separated from underlying evidence/safety ownership.

OST-UI may reuse these interaction patterns as hypotheses/evidence only. It must not copy the Physio clinical model into Osteoporosis.

---

## 3. Parallel-workstream coexistence

Draft PR #121 is currently open and unmerged.

It contains proposed `programme/OST-LIFECOURSE/`, `programme/OST-CLINICAL/`, Product Constitution and workstream-registry artifacts.

For OST-UI:

```text
PR #121 draft content = coexistence context
PR #121 draft content != current-main authority
```

OST-UI therefore does not modify those files and does not create a second global workstream registry.

If PR #121 merges later, OST-UI navigation can be reconciled in a bounded documentation step without changing R1-R4 decision questions.

---

## 4. Phase-1 review state

| Review | Status | Output |
|---|---|---|
| R1 — Current Product vs Constitution | NOT STARTED | `R1-CURRENT-PRODUCT-CONSTITUTION-AUDIT.md` |
| R2 — Longitudinal Clinical Trajectory | NOT STARTED | `R2-LONGITUDINAL-CLINICAL-TRAJECTORY-REVIEW.md` |
| R3 — Point-of-Care Interaction | NOT STARTED | `R3-POINT-OF-CARE-INTERACTION-REVIEW.md` |
| R4 — Shared Core / Module Architecture / Reuse | NOT STARTED | `R4-SHARED-CORE-MODULE-ARCHITECTURE-REUSE.md` |
| Synthesis — Product Reconstruction Decision | BLOCKED ON R1-R4 | `OST-PRODUCT-RECONSTRUCTION-DECISION.md` |

No additional review lane is authorized.

---

## 5. Dependency state

### Fracture / fragility

External to OST-UI clinical authority. At execution time, R2/R3 must fresh-resolve the current authoritative owner/artifact. Missing semantics become a bounded dependency/referral, not a UI assumption.

### Treatment / evidence

Existing reviewed G-2 / clinical contracts remain authoritative. OST-UI reviews their product projection only.

### Shared Core / patient state

R4 may recommend ownership movement/generalization but cannot perform it.

### PR-1 / transcript / future Live Copilot

Current PR-1 is an active root lifecycle and read-only architecture input for OST-UI. Live Copilot remains a future compatibility constraint, not an implementation target.

---

## 6. Next action

Immediate next durable transition:

```text
open one docs-only DRAFT PR
→ checkpoint PR identity in this CURRENT.md
→ verify OST-UI diff is limited to programme/OST-UI/*
→ STOP bootstrap
```

After bootstrap stop, the next substantive phase action is:

```text
R1 — CURRENT PRODUCT VS PRODUCT CONSTITUTION AUDIT
```

R2/R3/R4/synthesis do not start automatically.

---

## 7. Explicit stop / hold

Do not:

- implement redesign;
- create prototype code;
- change root canonicals;
- change runtime/schema/database;
- change fracture/fragility or treatment semantics;
- create extra reviews;
- merge this bootstrap PR;
- deploy;
- transfer the root writer lock.

