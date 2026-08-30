# SLICE_PLAN_CURRENT.md — G-1 Progressive Guidance Runtime Foundation v1

> **STATUS:** IMPLEMENTED / TESTED / RELEASE GATE.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G1-PROGRESSIVE-GUIDANCE-RUNTIME-v1.
> **Verified remote main at bootstrap:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent accepted design:** `design/module01-progressive-guidance-foundations-2026-08-30` @ `298d8d525f1bac97ffb6904fe09800519bd1a584`.
> **Implementation branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Runtime/test head:** `d66ac728e379a90542f95a8ecdde6d945420f6ae`.
> **Runtime writer:** NONE — bounded implementation complete.
> **Merge/deploy/production smoke:** NOT AUTHORIZED / NOT DONE.

---

# 1. Objective achieved

G-1 implements the smallest useful runtime foundation for progressive visit guidance without attempting a complete osteoporosis visit taxonomy.

Implemented flow:

```text
coarse clinician visit intent
+ short first-page visit context
+ minimal read-only longitudinal context
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

Rules implemented:

- `encounter_archetype` remains coarse clinician-declared intent;
- `quick_notes` is fallback visit context, especially for `other` or categories that are still too broad;
- G-1 does not parse free text into new authoritative structured facts;
- later transcript extraction may supply richer candidate context only in PR-1/PR-2.

---

# 3. Longitudinal projection implemented

G-1 reads protected historical encounters and derives only the context needed for current guidance.

Implemented projection includes:

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

Hard behavior implemented:

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

# 4. Minimal Visit Plan model implemented

Initial reason classes:

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

Representative implemented mechanics:

- first assessment produces broad core guidance;
- routine treatment continuation is smaller;
- new fracture overrides routine treatment flow;
- explicit fracture-on-treatment adds treatment administration/transition context;
- unresolved prior tasks surface follow-up context;
- explicitly stored due/overdue state surfaces administration/follow-up context;
- longitudinal conflicts surface treatment/admin review context.

No treatment selection/recommendation is generated.

---

# 5. Progressive-taxonomy invariant preserved

No large new visit enum set was added.

Current dropdown stays primary. `other + quick_notes` is a valid first-use path.

Potential future categories such as:

```text
results/work-up review with management decision
repeat prescription / routine maintenance
```

remain evidence-from-use candidates until post-pilot refinement or a separately approved earlier correction.

---

# 6. UI implementation

G-1 adds a lightweight `Σημερινή ροή` summary and reuses the existing cards.

It displays:

- current coarse visit intent;
- short visit context when supplied;
- basic protected longitudinal context availability;
- prioritized relevant domains;
- human-readable `Γιατί τώρα` reasons;
- explicit conflict warning when longitudinal treatment data disagree.

No red/green KPI/performance styling was introduced.

The existing `adaptive-applicability` state remains the coarse source of card applicability. G-1 does not rewrite it.

After implementation review, visual ownership was corrected to:

```text
coarse adaptive classes persist unchanged
+
.guidance-surfaced temporarily overrides collapsed presentation
```

so disappearance of a higher-priority trigger restores the existing coarse state automatically.

---

# 7. Acceptance evidence

Runtime/test head:

```text
d66ac728e379a90542f95a8ecdde6d945420f6ae
```

GitHub Actions:

```text
workflow: G1 progressive guidance foundation
run:      33327717796
result:   SUCCESS
```

Passed:

- JavaScript syntax checks;
- progressive guidance core regressions;
- progressive guidance wiring/ownership regression;
- authoritative Finish browser regression;
- server finalization lifecycle regression.

Focused G-1 regression coverage:

1. first assessment → broad core flow;
2. `other` + quick notes preserved/displayed without forced classification;
3. routine administration-history deduplication;
4. scheduled-not-actual event not counted;
5. conflicting explicit next-due facts remain visible conflicts;
6. unresolved prior task and later explicit completion behavior;
7. no due state invented from administration date alone;
8. new fracture/fracture-on-treatment override;
9. draft/current encounter excluded from historical projection;
10. deterministic same context → same Visit Plan;
11. bootstrap preserves authoritative Finish ownership;
12. guidance visual override does not mutate coarse applicability ownership.

---

# 8. Scope verification

Changed for G-1 runtime:

```text
static/baseline-audit/progressive-guidance-core.js   NEW
static/baseline-audit/progressive-guidance-ui.js     NEW
static/baseline-audit/progressive-guidance.css       NEW
static/baseline-audit/adaptive-applicability.css
static/baseline-audit/app.js
test_progressive_guidance_node.js                    NEW
test_progressive_guidance_wiring.js                  NEW
.github/workflows/g1-progressive-guidance-tests.yml  NEW
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
```

Not implemented/changed:

- complete visit taxonomy;
- medication-specific milestone content;
- PR-1 provider/API transcript extraction;
- PR-2 candidate population/acceptance;
- Practice Review runtime;
- clinical database schema/migration;
- KPI semantics;
- physiotherapy/RF code.

---

# 9. Completion matrix

```text
coarse dropdown reused                     YES / TESTED
quick-notes context reused                 YES / TESTED
protected historical projection            YES / TESTED
actual administration dedup                YES / TESTED
scheduled-only does not count              YES / TESTED
unresolved-prior resurfacing               YES / TESTED
explicit due-state plumbing                YES / TESTED
new-fracture override                      YES / TESTED
WHY NOW                                    YES / TESTED at core/wiring level
applicability ownership preserved          YES / TESTED
C1 regressions preserved                   YES / TESTED
full taxonomy                              NO / deliberately deferred
medication-specific milestones             NO
merged                                     NO
deployed                                   NO
production smoke                           NO
real-clinic pilot                          NO
```

G-1 is code-level IMPLEMENTED / TESTED. It is not production-validated.

---

# 10. Exact next action

STOP at release gate.

A separate release decision is required before PR/merge/deploy.

If release is authorized:

```text
fresh six-canonical bootstrap
→ fresh current-main verification
→ inspect complete compare of implementation branch against main
→ verify only accepted Module-01 ancestry is included and parked work is excluded
→ PR/review/merge
→ normal Render auto-deploy
→ production synthetic smoke of C1 Finish + G-1 guidance loading/context/WHY NOW
→ record exact release evidence
```

PR-1 Heidi extraction and PR-2 inline provisional population remain later bounded slices before the five-case system-assisted real pilot unless separately replanned.