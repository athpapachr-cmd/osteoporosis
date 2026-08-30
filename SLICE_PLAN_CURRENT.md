# SLICE_PLAN_CURRENT.md — G-1 Progressive Guidance Runtime Foundation v1

> **STATUS:** IMPLEMENTED / TESTED / RELEASE GATE — G1-R1 + G1-R2 CLOSED IN CODE.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G1-PROGRESSIVE-GUIDANCE-RUNTIME-v1.
> **Fresh verified remote main for R1/R2 correction:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent accepted design:** `design/module01-progressive-guidance-foundations-2026-08-30` @ `298d8d525f1bac97ffb6904fe09800519bd1a584`.
> **Implementation branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **R1/R2 runtime+test head:** `3294deebb97cf3f0a0d8fa2848ac4af7a04b01de`.
> **Runtime writer:** NONE — bounded R1/R2 correction complete.
> **Merge/deploy/production smoke:** NOT AUTHORIZED / NOT DONE.

---

# 1. Objective

G-1 implements the smallest useful runtime foundation for progressive visit guidance without attempting a complete osteoporosis visit taxonomy.

Current flow:

```text
coarse clinician visit intent
+ short first-page visit context
+ protected longitudinal context when available
→ deterministic card relevance/order
→ visible WHY NOW reason(s)
```

The runtime is guidance/navigation infrastructure, not a treatment-recommendation engine.

---

# 2. Existing clinician inputs retained

G-1 reuses existing persisted/runtime fields:

```text
encounter_archetype
quick_notes
patient_relationship_status
fracture_history.events[]
step4.treatment_episodes[]
step4.administrations[]
step4.tasks[]
step4.close
```

Rules:

- `encounter_archetype` remains coarse clinician-declared intent;
- `quick_notes` is fallback visit context, especially for `other` or categories that are still too broad;
- G-1 does not parse free text into new authoritative structured facts;
- later transcript extraction may supply richer candidate context only in PR-1/PR-2.

---

# 3. Longitudinal projection

G-1 reads protected historical encounters and derives only the context needed for current guidance.

Projection includes:

```text
prior encounter count/latest historical encounter
reliable unique actual administration events
administration count by agent when representable
last actual administration by agent
explicit stored next-due context
latest nonempty treatment snapshot / active episode when unambiguous
unresolved prior planned tasks
latest prior unresolved-critical close state
material projection conflicts
```

Hard behavior:

- read-only/ephemeral derived state;
- no new patient-level DB table;
- completed/amended encounters are historical sources;
- current/draft encounter is not historical authority;
- scheduled/planned administration does not count as actual;
- no inferred missing doses from expected cadence;
- no medication-specific due threshold inferred from actual dates alone;
- exact duplicate `agent + actual_date` representation is not double-counted;
- material conflicts remain explicit rather than silently resolved.

---

# 4. Minimal Visit Plan

Reason classes:

```text
VISIT_TYPE_CORE
NEW_EVENT
UNRESOLVED_PRIOR
TREATMENT_CONTEXT
EXPLICIT_DUE_STATE
CONTEXTUAL
```

Priority:

```text
NEW_EVENT
→ UNRESOLVED_PRIOR
→ EXPLICIT_DUE_STATE
→ TREATMENT_CONTEXT
→ VISIT_TYPE_CORE
→ CONTEXTUAL
```

Representative mechanics:

- first assessment produces broad core guidance;
- routine treatment continuation is smaller;
- new fracture overrides routine treatment flow;
- explicit fracture-on-treatment adds treatment administration/transition context;
- unresolved prior tasks surface follow-up context;
- explicitly stored due/overdue state surfaces administration/follow-up context;
- longitudinal conflicts surface treatment/admin review context.

No treatment selection/recommendation is generated.

---

# 5. Progressive-taxonomy invariant

No large new visit enum set was added.

Current dropdown stays primary. `other + quick_notes` is a valid first-use path.

Potential future categories such as:

```text
results/work-up review with management decision
repeat prescription / routine maintenance
```

remain evidence-from-use candidates until post-pilot refinement or a separately approved earlier correction.

---

# 6. G1-R1 correction — longitudinal history availability integrity

Release review identified that a failed protected history request could be represented as an empty history.

Closed invariant:

```text
HISTORY UNAVAILABLE != NO HISTORY
AUTH/NETWORK/SERVER FAILURE != ZERO PRIOR ENCOUNTERS
```

Runtime now carries explicit UI history state:

```text
not_loaded
loading
loaded
unavailable
```

Behavior now proven:

- successful loaded empty history may legitimately display zero previous completed/amended visits;
- failed history load displays unavailable/incomplete longitudinal context and does not claim zero visits;
- current local visit guidance remains usable during loading/unavailable state;
- starting a load for another patient clears prior in-memory historical rows immediately;
- stale completion from a no-longer-active patient request cannot overwrite the current patient's history state.

---

# 7. G1-R2 correction — live UI state owns the current snapshot

Release review identified stale persisted fallback when a clinician cleared live values before Save.

Closed invariant:

```text
IF A LIVE CONTROL EXISTS
→ its present value, including blank/empty, owns today's in-memory guidance snapshot

persisted cache
→ fallback only when the corresponding live control/root is absent
```

Covered current fields:

- encounter archetype;
- encounter date;
- quick notes;
- interval fracture status;
- rendered fracture-event collection.

An explicitly empty live fracture-event container now projects `events=[]` rather than resurrecting old cached events.

---

# 8. UI ownership

G-1 adds a lightweight `Σημερινή ροή` summary and reuses existing cards.

It displays:

- current coarse visit intent;
- short visit context when supplied;
- explicit longitudinal-context availability;
- prioritized relevant domains;
- human-readable `Γιατί τώρα` reasons;
- explicit conflict warning when longitudinal treatment data disagree.

No red/green KPI/performance styling is introduced.

Existing `adaptive-applicability` remains the coarse owner. G-1 does not rewrite that state.

Visual ownership remains:

```text
coarse adaptive classes persist unchanged
+
.guidance-surfaced temporarily overrides collapsed presentation
```

---

# 9. Exact acceptance evidence

R1/R2 runtime+test head:

```text
3294deebb97cf3f0a0d8fa2848ac4af7a04b01de
```

GitHub Actions:

```text
workflow: G1 progressive guidance foundation
run:      33329341340
job:      g1-guidance
result:   SUCCESS
```

Passed at the exact correction head:

- JavaScript syntax checks;
- progressive guidance pure-core regressions;
- guidance wiring/ownership regression;
- **new progressive guidance UI-state regressions**;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

Focused R1/R2 regression coverage proves:

```text
R1-A failed protected history fetch → unavailable
R1-B unavailable history summary does not claim zero prior visits
R1-C new-patient history load clears previous patient's rows before await completion
R1-D successful empty history → loaded + zero is legitimate
R2-A live blank quick_notes overrides persisted nonblank value
R2-B live blank encounter archetype/date/fracture status overrides persisted value
R2-C live empty fracture-event container overrides persisted fracture events with []
R2-D persisted fallback remains only when live control/root is absent
```

The full workflow simultaneously re-proved the inherited C1 authoritative Finish browser/server lifecycle behavior.

---

# 10. Correction scope verification

Exact compare from release-block head `56267d08dc5d68b8c5e4208f2ae3761fa15156b5` to R1/R2 test head `3294deebb97cf3f0a0d8fa2848ac4af7a04b01de` contains only:

```text
static/baseline-audit/progressive-guidance-ui.js
test_progressive_guidance_ui_state.js
.github/workflows/g1-progressive-guidance-tests.yml
CURRENT_OPERATIONAL.md
```

No G-1 core clinical rule, database/schema, KPI, PR-1/PR-2, physiotherapy, RF or medication-specific milestone content changed.

---

# 11. Completion matrix

```text
coarse dropdown reused                     YES / TESTED
quick-notes context reused                 YES / TESTED
protected historical projection            YES / TESTED
history availability state                 YES / TESTED
unavailable != zero history                YES / TESTED
cross-patient transient history cleared    YES / TESTED
live-empty > persisted-cache snapshot      YES / TESTED
actual administration dedup                YES / TESTED
scheduled-only does not count              YES / TESTED
unresolved-prior resurfacing               YES / TESTED
explicit due-state plumbing                YES / TESTED
new-fracture override                      YES / TESTED
WHY NOW                                    YES / TESTED
applicability ownership preserved          YES / TESTED
C1 regressions preserved                   YES / TESTED
full taxonomy                              NO / deliberately deferred
medication-specific milestones             NO
merged                                     NO
deployed                                   NO
production smoke                           NO
real-clinic pilot                          NO
```

G-1 including R1/R2 is code-level IMPLEMENTED / TESTED. It is not production-validated.

---

# 12. Exact next action

STOP at release gate.

No more runtime mutation is authorized by the R1/R2 correction request.

A separate release decision is required before:

```text
fresh current-main verification
→ exact compare/review
→ PR
→ merge
→ normal Render auto-deploy
→ production synthetic smoke of C1 Finish + G-1 guidance/history/context/WHY NOW
→ release evidence closeout
```

PR-1 Heidi extraction and PR-2 inline provisional population remain later bounded slices before the five-case system-assisted real pilot unless separately replanned.