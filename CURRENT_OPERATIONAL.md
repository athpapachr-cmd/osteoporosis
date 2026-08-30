# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 CLOSURE PROGRAM — C0 COMPLETE / C1 PILOT READINESS BLOCKED.
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

# 4. Closure sequence

```text
C0  closure-program canonical rebase                         COMPLETE
C1  five-case pilot readiness / run 5 real cases            BLOCKED BEFORE PILOT
C2  one deliberate post-pilot refinement + freeze form/KPI  NOT STARTED
C3  PR-1 transcript extraction implementation               NOT STARTED
C4  PR-2 clinician review/acceptance workflow               NOT STARTED
C5  Quick Practice Review shadow-mode minimum viable engine NOT STARTED
C6  30-case scored baseline + baseline lock                 NOT STARTED
C7  reviewed Signals/intervention loop                      NOT STARTED
C8  adaptive workflow/Close refinement                      NOT STARTED
C9  re-measure persistence + Module 01 closure review       NOT STARTED
```

Implementation slices after the pilot each require their own bounded writer/runtime authorization. Do not convert this closure plan into one giant implementation branch.

---

# 5. C1 pilot-readiness audit — BLOCKER FOUND

Read-only inspection of the actual `main` browser/runtime seam found a finalization ownership defect that must be resolved before real pilot collection.

Current event flow:

```text
pilot-completion.js
Finish click capture listener
→ preventDefault()
→ stopImmediatePropagation()
→ clicks Save
→ local pilot_completion.status = complete

patient-registry.js
Save listener
→ syncActiveEncounter("draft")

patient-registry.js
Finish listener
→ syncActiveEncounter("completed")
```

Because the pilot-completion listener is a capture listener on the same Finish control and calls `stopImmediatePropagation()`, the later patient-registry Finish listener is not a reliable completion path. The Save click deliberately triggered by pilot-completion schedules a server sync as `draft`.

Therefore the current code can create the contradictory state:

```text
local pilot_completion = complete
server clinical_encounter.status = draft
```

This is a persistence/finalization integrity blocker for the five real pilot encounters.

Existing `test_encounter_finalization.py` proves only the server-side transition function once a requested status reaches the server; it does not exercise this integrated browser event path.

The prior 3/3 live finalization smoke remains valid evidence for completed→completed/amended server semantics, but it is not evidence that this exact Step-6 Finish wiring cannot suppress the `completed` request.

---

# 6. Exact bounded correction required before C1

The next runtime slice must be limited to:

```text
ONE authoritative Finish action
→ persist the current complete case state locally
→ synchronize the same final payload to protected server storage with requested status=completed
→ verify resulting server encounter status=completed
→ preserve later completed/amended integrity rules
```

Acceptance must include an integrated regression proving at minimum:

```text
A. draft encounter + Step-6 Finish
   → local pilot_completion=complete
   → server encounter status=completed
   → final payload present server-side

B. reload/reopen
   → same completed encounter loads successfully

C. no-op Save after completion
   → remains completed

D. material edit + Save
   → becomes amended
```

Do not redesign clinical fields, pilot methodology, KPI rules or unrelated persistence while fixing this seam.

---

# 7. Baseline integrity remains binding

After the blocker is fixed and re-verified, the approved sequence remains:

```text
5-case usability/capture pilot
→ one deliberate refinement
→ freeze Baseline Form + KPI contract
→ 30-case scored baseline
→ baseline lock
→ systematic coaching/intervention
```

During the scored 30-case baseline, routine KPI coaching/Practice Review intervention remains hidden by default; safety-critical exceptions remain allowed. If this methodology changes, record the revision explicitly before collecting/labeling the cohort.

Pilot eligibility remains the existing schema rule: consecutive adult encounters in which osteoporosis, osteopenia, fragility-fracture risk or osteoporosis treatment is materially assessed or managed; purely administrative/non-osteoporosis contacts are excluded. Pilot cases test usability and are not part of the later locked 30-patient baseline.

---

# 8. Status matrix

```text
PHYSIO CU-1 PARKED/PRESERVED             YES
PHYSIO NEW ROUTE ROLLOUT                  HOLD
MODULE 01 CLOSURE PROGRAM DESIGNED        YES
C0 CLOSURE-PLAN REBASE                    COMPLETE
C1 PILOT READINESS                        BLOCKED — FINALIZATION SEAM
5-CASE REAL PILOT                         NOT STARTED
BASELINE FORM/KPI FROZEN                  NO
30-CASE SCORED BASELINE LOCKED            NO
CLOSED IMPROVEMENT LOOP DEMONSTRATED      NO
MODULE 01 CLOSED                          NO
MERGED/DEPLOYED BY THIS SLICE             NO
```

---

# 9. Exact next authorized action

```text
STOP this design/readiness slice.
Present the bounded C1 finalization blocker to the product owner.
If runtime correction is authorized:
→ fresh main verification/bootstrap
→ create a dedicated C1 pilot-finalization-integrity implementation slice/branch
→ claim runtime writer
→ fix only the Finish/local/server completion seam
→ add integrated regression
→ test
→ merge/deploy only with the authorization defined for that slice
→ re-run pilot readiness
→ if PASS, start the 5 real pilot encounters.
```

---

# 10. Explicit HOLD

Do not, during this planning slice:

- start real pilot cases before the C1 blocker is resolved and verified;
- resume physiotherapy route expansion;
- merge/deploy the parked CU-1 enhancement branch;
- open CU-2 or RF runtime work;
- write PR-1/PR-2/Practice Review runtime code;
- redesign the Baseline Form before pilot evidence;
- expose routine coaching during scored baseline;
- declare Module 01 closed from engineering completion alone;
- claim privacy/GDPR compliance beyond what is actually verified.
