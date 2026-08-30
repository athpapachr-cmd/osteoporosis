# SLICE_PLAN_CURRENT.md — G-1 Progressive Guidance Runtime Foundation v1

> **STATUS:** IMPLEMENTATION ACTIVE.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G1-PROGRESSIVE-GUIDANCE-RUNTIME-v1.
> **Verified remote main at bootstrap:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **Parent accepted design:** `design/module01-progressive-guidance-foundations-2026-08-30` @ `298d8d525f1bac97ffb6904fe09800519bd1a584`.
> **Implementation branch:** `feat/module01-g1-progressive-guidance-foundation-2026-08-30`.
> **Runtime writer:** same branch, bounded to G-1 guidance mechanics/tests/bootstrap wiring.
> **Merge/deploy/production smoke:** NOT AUTHORIZED / NOT DONE.

---

# 1. Objective

Implement the smallest useful runtime foundation for progressive visit guidance without attempting a complete osteoporosis visit taxonomy.

Target behavior:

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

Use existing persisted fields:

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
- G-1 must not silently parse free text into new authoritative structured facts;
- later transcript extraction may supply richer candidate context only in PR-1/PR-2.

---

# 3. Minimal longitudinal projection

G-1 may read protected historical encounters and derive only the context needed for current guidance.

Initial projection fields:

```text
prior_encounter_count
latest_completed_or_amended_encounter_date
latest_coarse_visit_intent
latest_visit_context_summary_if_non_sensitive_runtime_field
reliable_actual_administration_events[]
latest_actual_administration_by_agent
unresolved_prior_tasks[]
prior_unresolved_critical_item_optional
material_projection_conflicts[]
```

Hard rules:

- read-only/ephemeral derived state;
- no new patient-level DB table;
- completed/amended encounters are historical sources;
- blank later snapshot must not erase prior known state;
- scheduled/planned administration does not count as actual;
- no inferred missing doses from expected cadence;
- exact duplicate actual administration representations may deduplicate only when identity is reliable;
- conflicts remain explicit rather than silently resolved.

---

# 4. Minimal current Encounter Context

Build a deterministic context from current case + projection:

```text
coarse_visit_intent
visit_context_text
new_fracture_or_fracture_on_treatment
has_prior_unresolved_item
active_or_recent_treatment_agents
latest_actual_administration_by_agent
due_or_overdue_context_only_when_explicit_in_existing_data
prior_encounter_count
projection_conflicts_present
```

Do not invent medication-specific timing thresholds in G-1.

---

# 5. Minimal card guidance model

G-1 may reuse existing cards/domains rather than create a new full card library.

Initial reason classes:

```text
VISIT_TYPE_CORE
NEW_EVENT
UNRESOLVED_PRIOR
TREATMENT_CONTEXT
EXPLICIT_DUE_STATE
CONTEXTUAL
```

Initial priority:

```text
NEW_EVENT
→ UNRESOLVED_PRIOR
→ EXPLICIT_DUE_STATE / TREATMENT_CONTEXT
→ VISIT_TYPE_CORE
→ CONTEXTUAL
```

Every surfaced non-obvious card must expose one or more human-readable `why now` reasons.

No clinical treatment choice is generated.

---

# 6. Progressive-taxonomy invariant

Do not add a large new enum set in G-1.

Current dropdown stays primary. `other + quick_notes` is supported as a valid first-use path.

Potential future categories such as:

```text
results/work-up review with management decision
repeat prescription / routine maintenance
```

remain evidence-from-use candidates until post-pilot refinement or a separately approved earlier correction.

---

# 7. UI boundary

G-1 should minimally:

- show the current coarse visit intent/context near the guidance summary;
- show which cards/domains are prioritized for today;
- show `why now` text;
- keep irrelevant existing cards collapsed rather than delete their stored schema;
- allow existing clinician override/`Χρήση σήμερα` mechanics to remain available;
- avoid KPI/red-green/performance-coaching styling;
- avoid introducing a second competing source of truth.

---

# 8. Acceptance fixtures

Focused synthetic tests must include at least:

1. first assessment → broad core domains surfaced;
2. `other` + quick notes context preserved and displayed without forced classification;
3. routine treatment continuation with explicit actual administration history → treatment-related cards prioritized, full diagnostic work-up not forced;
4. routine continuation + new fracture → fracture/event domains outrank routine flow;
5. unresolved prior task → corresponding follow-up context resurfaces;
6. scheduled administration without actual administration → not counted as administered;
7. duplicate historical actual administration representation → no double count when reliably identical;
8. conflicting historical administration facts → conflict visible, not silently resolved;
9. no medication-specific due threshold invented from dates alone;
10. same structured context → deterministic same Visit Plan ordering/reasons.

---

# 9. Replan triggers

STOP and replan if implementation demonstrates that:

- protected historical encounter payloads cannot support safe minimal projection;
- required prior-task/treatment identity cannot be represented without unsafe guessing;
- existing adaptive card ownership conflicts materially with the new guidance layer;
- G-1 would require medication-specific clinical rules to be useful at all;
- implementation requires a new persistent patient-level source of truth rather than an ephemeral projection;
- free-text context would need automatic clinical inference to drive safe behavior.

---

# 10. Out of scope

No:

- complete visit taxonomy;
- PR-1 provider/API transcript extraction;
- PR-2 inline candidate acceptance;
- Practice Review;
- evidence-backed medication milestone content;
- automatic taxonomy mutation;
- real-patient fixtures;
- physiotherapy/RF mutation;
- merge/deploy.

---

# 11. Exact implementation sequence

```text
1. inspect current protected encounter fetch/browser seams
2. implement read-only longitudinal projection helper
3. implement minimal encounter-context + visit-plan resolver
4. integrate with existing adaptive card UI without replacing storage schema
5. surface WHY NOW + coarse visit context
6. add focused synthetic regression suite
7. run syntax/tests
8. exact implementation review
9. update CURRENT_OPERATIONAL with evidence
10. STOP at TESTED / RELEASE GATE
```
