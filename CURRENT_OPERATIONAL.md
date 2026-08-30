# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 CLOSURE PROGRAM — ACTIVE DESIGN / EXECUTION REBASE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Current major phase:** close Osteoporosis Module 01 against explicit exit evidence, then generalize later.
> **ACTIVE CANONICAL WRITER/LOCK:** `design/module01-closure-program-2026-08-30`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Runtime mutation:** NOT AUTHORIZED in this closure-planning slice.
> **Merge/deploy/preview:** NOT AUTHORIZED / NOT DONE.

---

# 1. Product-owner decision

The product owner has explicitly ended the current physiotherapy expansion detour and changed priority to:

```text
PRESERVE completed/tested CU-1 work
→ stop further physiotherapy disease rollout
→ finish Osteoporosis Module 01
→ only then return to later disease work using condition-centered vertical slices
```

This is a priority/slicing decision, not a declaration that Module 01 is already complete.

---

# 2. Physiotherapy state — PARKED / PRESERVED

The production CU-1 v1 baseline already merged/deployed historically remains untouched on `main`.

The later product-reviewed rich-referral enhancement work is preserved separately at:

```text
branch: feat/cu1-rich-referral-global-evidence-2026-08-29
head:   bdd23b83a8252405f5aa5a0c0b67f303ccfcef5f
state:  IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED
PR:     NONE OPEN
MERGED: NO
DEPLOYED: NO
```

Do not delete, rewrite, expand, merge or deploy that parked branch during Module 01 closure work unless the product owner separately authorizes it.

The older design branch:

```text
design/cu1-history-evidence-timeline-2026-08-28
head: cc479f4a1d818481a886916e3f0f05dc56c623b3
```

is no longer the active writer. It is historical/preserved only.

---

# 3. Module 01 closure definition

`MODULE 01 CLOSED` may be claimed only after closure-critical exit evidence is satisfied or an explicit methodological revision is canonically approved.

Closure-critical work is intentionally smaller than the entire long-range TODO.

## 3.1 Engineering / system gates

Required before final closure:

1. safe transcript-assisted candidate extraction with raw transcript ephemeral by default;
2. clinician review/edit/accept/reject before extracted data become authoritative;
3. structured evidence-traceable Quick Practice Review capability;
4. clinician disposition on important observations;
5. denominator-aware Signal promotion and root-cause-appropriate intervention linkage sufficient to support at least one closed improvement loop;
6. adaptive osteoporosis clinical workflow / Close behavior informed by real pilot/review evidence, not one-off preference;
7. privacy/safety controls required by the closure features, without claiming whole-service GDPR compliance beyond evidence.

Deep Review / Red Team completeness, full Patient Voice, benchmarking, full Clinical Excellence Home, Calendar/Secretary integration, RF utility work and comprehensive cross-domain polish are NOT closure blockers unless later evidence makes one of them necessary for safety/data integrity.

## 3.2 Real-practice gates

Required before final closure:

```text
5 consecutive eligible real osteoporosis pilot encounters
→ one deliberate post-pilot refinement
→ freeze Baseline Form v1 + KPI applicability/calculation contract
→ 30 consecutive unique scored baseline encounters
→ baseline lock OR explicit approved methodology revision
→ activate clinician-facing improvement intervention after baseline policy permits it
→ observe at least one repeated actionable pattern/Signal
→ apply the appropriate intervention
→ re-measure in later encounters
→ demonstrate whether the targeted change persisted
```

Engineering completion alone is therefore insufficient to close Module 01.

---

# 4. Current closure sequence

The current approved order is:

```text
C0  canonical closure-program rebase                           ← ACTIVE NOW
C1  five-case pilot readiness / run 5 real cases
C2  one deliberate post-pilot refinement + freeze form/KPI
C3  PR-1 transcript extraction implementation
C4  PR-2 clinician review/acceptance workflow
C5  Quick Practice Review shadow-mode minimum viable engine
C6  30-case scored baseline + baseline lock
C7  activate reviewed Signals/intervention loop
C8  adaptive workflow/Close refinement from observed evidence
C9  re-measure persistence + Module 01 closure review
```

Implementation slices C3–C8 each require their own bounded writer/runtime authorization. Do not convert this closure plan into one giant implementation branch.

---

# 5. Baseline integrity remains binding

The approved sequence remains:

```text
5-case usability/capture pilot
→ one deliberate refinement
→ freeze Baseline Form + KPI contract
→ 30-case scored baseline
→ baseline lock
→ systematic coaching/intervention
```

During the scored 30-case baseline, routine KPI coaching/Practice Review intervention remains hidden by default; safety-critical exceptions remain allowed. If this methodology changes, record the revision explicitly before collecting/labeling the cohort.

---

# 6. Status matrix

```text
PHYSIO CU-1 PARKED/PRESERVED             YES
PHYSIO NEW ROUTE ROLLOUT                  HOLD
MODULE 01 CLOSURE PROGRAM DESIGNED        IN PROGRESS
MODULE 01 ENGINEERING EXIT GATES          NOT YET COMPLETE
5-CASE REAL PILOT                         NOT YET PROVEN
BASELINE FORM/KPI FROZEN                  NO
30-CASE SCORED BASELINE LOCKED            NO
CLOSED IMPROVEMENT LOOP DEMONSTRATED      NO
MODULE 01 CLOSED                          NO
MERGED/DEPLOYED BY THIS SLICE             NO
```

---

# 7. Exact next authorized action

```text
1. freeze the Module 01 closure-critical vs deferred boundary in SLICE_PLAN_CURRENT.md;
2. reconcile TODO priority/order to this closure program;
3. perform a read-only pilot-readiness check against current production/runtime evidence;
4. STOP before runtime mutation;
5. present the exact 5-case pilot protocol / blockers to the product owner.
```

---

# 8. Explicit HOLD

Do not, during this planning slice:

- resume physiotherapy route expansion;
- merge/deploy the parked CU-1 enhancement branch;
- open CU-2 or RF runtime work;
- write PR-1/PR-2/Practice Review runtime code;
- redesign the Baseline Form before pilot evidence;
- expose routine coaching during scored baseline;
- declare Module 01 closed from engineering completion alone;
- claim privacy/GDPR compliance beyond what is actually verified.
