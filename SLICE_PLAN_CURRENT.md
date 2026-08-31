# SLICE_PLAN_CURRENT.md — G-1 Progressive Guidance Runtime Foundation v1

> **STATUS:** IMPLEMENTED / TESTED / MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED — SLICE PRODUCTION-READINESS GATE CLOSED.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Parent product:** Personal Clinical Excellence System.
> **Module:** 01 — Osteoporosis.
> **Slice ID:** M01-G1-PROGRESSIVE-GUIDANCE-RUNTIME-v1.
> **Original release PR:** `#64`.
> **WHY-NOW correction PR:** `#66`.
> **Correction runtime merge SHA:** `d9423f4dcf6bebd056e83407132c6ce3e25d2280`.
> **Runtime writer:** NONE.

---

# 1. Objective

G-1 provides the minimum reusable runtime foundation for progressive osteoporosis visit guidance without pretending to be a complete treatment-recommendation engine or exhaustive visit taxonomy.

Runtime flow:

```text
coarse clinician-declared visit intent
+ short first-page visit context
+ protected longitudinal context when available
→ deterministic card relevance/order
→ visible WHY NOW reason(s)
```

Clinical workflow presentation remains distinct from storage/audit schema.

---

# 2. Preserved clinician inputs / invariants

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

- `encounter_archetype` is coarse clinician-declared visit intent;
- `quick_notes` is contextual fallback text;
- G-1 does not parse free text into authoritative structured facts;
- transcript-derived candidates remain future PR-1/PR-2 work;
- no large visit taxonomy was introduced;
- `other + quick_notes` remains a valid fallback.

Longitudinal integrity remains:

```text
HISTORY UNAVAILABLE != NO HISTORY
scheduled/planned administration != actual administration
administration count != elapsed exposure
missing expected doses are not reconstructed
material conflicts remain explicit
current/draft encounter is not historical authority
```

---

# 3. Minimal Visit Plan / priority

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

Mechanics already tested include:

- first-assessment core flow;
- smaller routine treatment-continuation flow;
- new-fracture override;
- fracture-on-treatment treatment/transition context;
- unresolved-prior resurfacing;
- explicit stored due/overdue plumbing;
- explicit longitudinal conflicts;
- no automatic treatment selection/recommendation.

---

# 4. G1-R1 / G1-R2 closed integrity corrections

## G1-R1 — history availability

```text
AUTH/NETWORK/SERVER FAILURE != ZERO PRIOR ENCOUNTERS
```

History states remain `not_loaded / loading / loaded / unavailable`.

## G1-R2 — live UI owns today's snapshot

```text
IF A LIVE CONTROL EXISTS
→ its current value, including blank/empty, owns today's in-memory guidance snapshot

persisted cache
→ fallback only when corresponding live control/root is absent
```

This includes encounter archetype/date/quick notes/interval fracture status and rendered fracture events.

---

# 5. WHY-NOW UX contract / production correction

Normative invariant:

> A dynamically surfaced item must make its `WHY NOW?` reason discoverable to the clinician at the point of use.

Original G-1 already generated deterministic `item.why_now` and rendered `Γιατί τώρα:` inside destination cards.

Initial production smoke found a presentation defect in the top `Σημερινή ροή` summary: reason text existed but the explicit `Γιατί τώρα:` label was missing, so the clinician could not reliably identify the WHY-NOW explanation.

PR #66 changed only summary presentation:

```text
Γιατί τώρα: <existing deterministic item.why_now>
```

No guidance reason, priority, clinical rule, treatment recommendation, taxonomy, persistence or schema changed.

Focused regression locks both summary and destination-card `Γιατί τώρα:` presentation.

---

# 6. Exact release evidence

Original G-1+C1 release PR #64 was squash-merged as:

```text
a6ba9ef1719a18a48a1756bf08bbd157d448a63e
```

WHY-NOW correction final PR head:

```text
e2960454cfa1acf6fa4e2c0735a2e7ba3c267f48
```

Exact-head G-1 workflow runs:

```text
33333512964  SUCCESS
33333526378  SUCCESS
```

PR #66 squash merge:

```text
d9423f4dcf6bebd056e83407132c6ce3e25d2280
```

Render auto-deploy:

```text
dep-daa93ljncjis739ssef0
→ LIVE
→ exact commit d9423f4dcf6bebd056e83407132c6ce3e25d2280
```

---

# 7. Product-owner production re-smoke — PASS

On 2026-08-31 Asia/Nicosia, the product owner directly confirmed that in production:

```text
existing `Τύπος σημερινής επίσκεψης` is used
→ `Σημερινή ροή` displays explicit `Γιατί τώρα:`
→ guidance changes dynamically with the current visit context
→ the surfaced information is experienced as informative / guiding
```

Therefore:

```text
WHY-NOW discoverability                    PASS
G-1 dynamic production interaction         PASS
PRODUCTION-SMOKE-VERIFIED                  YES
```

The positive usefulness observation is a **product-owner production observation**, not a real-clinic pilot result and not `PILOT-VALIDATED` evidence.

---

# 8. Completion matrix

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
WHY-NOW explicit summary presentation          YES / TESTED / DEPLOYED / PRODUCTION-SEEN
applicability ownership preserved              YES / TESTED
C1 regressions preserved                       YES / TESTED
merged                                         YES
deployed                                       YES
production-smoke-verified                      YES
pilot-validated                                NO
real-clinic pilot                              NO
```

---

# 9. Slice stop rule

G-1 production-readiness is closed. STOP runtime mutation for this slice.

Do not reopen G-1 taxonomy, clinical rules or presentation merely for speculative completeness. Further refinement requires a separately authorized slice driven by evidence-backed guidance content or real-use evidence.

PR-1 Heidi extraction, PR-2 provisional population, medication-specific milestone content, parked physiotherapy/RF work and real pilot collection remain outside this closed slice until separately authorized.
