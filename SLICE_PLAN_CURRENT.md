# SLICE_PLAN_CURRENT.md — G-1 Progressive Guidance Runtime Foundation v1

> **STATUS:** IMPLEMENTED / TESTED / MERGED / DEPLOYED — FINAL PRODUCT-OWNER WHY-NOW RE-SMOKE PENDING.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G1-PROGRESSIVE-GUIDANCE-RUNTIME-v1.
> **Current production correction main SHA:** `d9423f4dcf6bebd056e83407132c6ce3e25d2280`.
> **Original release PR:** `#64`.
> **WHY-NOW correction PR:** `#66`.
> **Runtime writer:** NONE.

---

# 1. Objective

G-1 provides the smallest useful runtime foundation for progressive osteoporosis visit guidance without pretending to be a complete treatment-recommendation engine or exhaustive visit taxonomy.

Current runtime flow:

```text
coarse clinician-declared visit intent
+ short first-page visit context
+ protected longitudinal context when available
→ deterministic card relevance/order
→ visible WHY NOW reason(s)
```

Clinical workflow presentation remains distinct from storage/audit schema.

---

# 2. Existing clinician inputs retained

G-1 reuses existing fields including:

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

- `encounter_archetype` is coarse clinician-declared intent;
- `quick_notes` is contextual text, especially for `other` or broad categories;
- G-1 does not parse free text into authoritative structured facts;
- transcript-derived candidates remain future PR-1/PR-2 work.

---

# 3. Longitudinal projection invariants

G-1 reads completed/amended protected historical encounters and derives read-only ephemeral context for current guidance.

Preserved rules:

```text
HISTORY UNAVAILABLE != NO HISTORY
scheduled/planned administration != actual administration
administration count != elapsed exposure
missing expected doses are not reconstructed
material conflicts remain explicit
current/draft encounter is not historical authority
```

History availability states remain:

```text
not_loaded
loading
loaded
unavailable
```

A failed history request must never render as false zero prior encounters.

---

# 4. Minimal Visit Plan / priority

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

Representative mechanics already tested:

- first assessment broad core flow;
- smaller routine treatment-continuation flow;
- new fracture overrides routine flow;
- fracture on treatment adds treatment/transition context;
- unresolved prior tasks resurface;
- explicit stored due/overdue information can surface administration/follow-up context;
- longitudinal conflicts remain reviewable;
- no treatment selection/recommendation is generated.

---

# 5. Progressive-taxonomy invariant

No large new visit enum set was introduced.

The existing `Τύπος σημερινής επίσκεψης` / `encounter_archetype` dropdown remains primary. `other + quick_notes` remains a valid fallback.

Potential future categories such as results/work-up review with management decision remain evidence-from-use candidates and are not silently introduced by G-1.

---

# 6. G1-R1 / G1-R2 closed integrity corrections

## G1-R1 — history availability

```text
AUTH/NETWORK/SERVER FAILURE != ZERO PRIOR ENCOUNTERS
```

Successful empty history may show zero. Failed/unavailable history must explicitly say it is unavailable/incomplete.

## G1-R2 — live UI owns the current snapshot

```text
IF A LIVE CONTROL EXISTS
→ its current value, including blank/empty, owns today's in-memory guidance snapshot

persisted cache
→ fallback only when corresponding live control/root is absent
```

This includes encounter archetype/date/quick notes/interval fracture status and the rendered fracture-event collection.

---

# 7. WHY-NOW UX contract and production correction

Normative UX invariant:

> A dynamically surfaced item must make its `WHY NOW?` reason discoverable to the clinician at the point of use.

Original G-1 already generated deterministic `item.why_now` and rendered explicit `Γιατί τώρα:` inside destination cards.

Production smoke found a presentation gap:

```text
`Σημερινή ροή` summary
→ reason text present
→ but explicit `Γιατί τώρα:` label absent
→ clinician could not identify/find WHY NOW reliably
```

Bounded correction PR #66 changed only the summary presentation:

```text
Γιατί τώρα: <existing deterministic item.why_now>
```

No guidance reason, priority, clinical rule, recommendation, taxonomy, persistence or schema changed.

A focused regression now locks both:

```text
summary explicit `Γιατί τώρα:`
+
destination-card explicit `Γιατί τώρα:` retained
```

---

# 8. Acceptance / release evidence

Original G-1+C1 release was squash-merged through PR #64 as:

```text
a6ba9ef1719a18a48a1756bf08bbd157d448a63e
```

The production-smoke WHY-NOW correction final PR head was:

```text
e2960454cfa1acf6fa4e2c0735a2e7ba3c267f48
```

Exact-head G-1 workflow runs:

```text
33333512964  SUCCESS
33333526378  SUCCESS
```

PR #66 was squash-merged as:

```text
d9423f4dcf6bebd056e83407132c6ce3e25d2280
```

Render auto-deploy:

```text
dep-daa93ljncjis739ssef0
→ LIVE
→ exact commit d9423f4dcf6bebd056e83407132c6ce3e25d2280
```

No manual duplicate deploy was triggered.

---

# 9. Completion matrix

```text
coarse dropdown reused                         YES / TESTED / PRODUCTION-SEEN
quick-notes context reused                     YES / TESTED
protected historical projection                YES / TESTED
history unavailable != zero                    YES / TESTED
live-empty > persisted-cache snapshot          YES / TESTED
actual administration dedup                    YES / TESTED
scheduled-only does not count                  YES / TESTED
unresolved-prior resurfacing                   YES / TESTED
explicit due-state plumbing                    YES / TESTED
new-fracture override                          YES / TESTED
WHY-NOW core generation                        YES / TESTED
WHY-NOW explicit summary presentation          YES / TESTED / DEPLOYED
applicability ownership preserved              YES / TESTED
C1 regressions preserved                       YES / TESTED
merged                                         YES
deployed                                       YES
final corrected WHY-NOW production re-smoke    PENDING
production-smoke-verified                      NO
real-clinic pilot                              NO
```

---

# 10. Exact next action

STOP runtime mutation.

Product owner performs only the bounded correction re-smoke:

```text
select/use existing `Τύπος σημερινής επίσκεψης`
→ inspect top `Σημερινή ροή`
→ confirm each surfaced reason is visibly prefixed `Γιατί τώρα: ...`
```

If PASS, record `PRODUCTION-SMOKE-VERIFIED`, append final correction/smoke evidence to `osteoporosis-change-log.md`, and close the G-1 production-readiness gate.

If FAIL, reopen only the exact observed presentation seam.

PR-1 Heidi extraction, PR-2 provisional population, new medication-specific milestone content, parked physiotherapy/RF work and real pilot collection remain outside this slice until separately authorized.
