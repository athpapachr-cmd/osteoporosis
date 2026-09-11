# SLICE_PLAN_CURRENT.md — Physio Referral Knee-OA Dynamic Referral / Template v1

> **STATUS:** DESIGN ACTIVE / STEP 3
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CU1-PRODUCT-KNEE-OA-TEMPLATE-V1-2026-09-11`.
> **Fresh `main`:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-2 parent:** `ab4b349223cd4c461837ab3125967a06d169a7e1`.
> **Branch:** `design/physio-referral-knee-oa-template-v1-2026-09-11`.
> **Vertical slice:** Knee Osteoarthritis only.
> **Runtime implementation authority:** NONE.
> **Merge/deploy authority:** NONE.

---

# 1. Problem

The current CU-1 formatter is deterministic but generic: it joins findings, functions, goals, rehabilitation directions and adjuncts into broad sentences. That is appropriate for the existing utility but not sufficient for the frozen product UX, where the referral must update live and read like one coherent clinical referral rather than a serialized checklist.

Step 3 must create a deterministic product-layer composition contract without rewriting or duplicating the existing CU-1 clinical taxonomy.

---

# 2. Inputs and authority

Authoritative inputs remain:

```text
ReferralDraftV1 structured selections
+ frozen Step-2 evidence contract
+ bounded Knee-OA product phenotype overlay for concepts not safely representable in current CU-1
```

The product overlay may add only presentation-state concepts required by the frozen Knee-OA UX. It must never masquerade as objective CU-1 findings.

Initial product-local phenotype candidates:

```text
stiffness_symptom
weakness_symptom_or_context
```

Reason:

```text
stiffness symptom != active/passive ROM restriction
weakness symptom/context != objectively measured weakness
```

Existing CU-1 findings such as `pain`, `quadriceps_weakness`, `objective_weakness`, ROM restriction and balance deficit remain authoritative when explicitly selected.

---

# 3. Routine referral structure

The standard copied referral should normally contain no more than these semantic blocks:

```text
A. indication / laterality
B. clinical picture + functional impact when selected
C. active rehabilitation plan
D. contextual emphasis when selected and not redundant
E. adjunct/support sentence only when explicitly selected
F. explicit restriction/clinician note only when present
```

Evidence citations, evidence-state labels, recommendation-strength language and `i` explanations remain clinician-facing UI information and do not enter the copied referral by default.

---

# 4. Laterality grammar

The product formatter should use natural Greek rather than generic parenthetical labels.

Candidate canonical phrases:

```text
right      → δεξιού γόνατος
left       → αριστερού γόνατος
bilateral  → και των δύο γονάτων
```

For the Knee-OA prototype, copy-ready state should require `right`, `left` or `bilateral`; `not_stated`, `midline` and `not_applicable` are not acceptable final Knee-OA laterality states.

This is a product copy-readiness rule and does not change the global CU-1 Laterality enum.

---

# 5. Selection vs suggestion boundary

Hard rule:

```text
suggested intervention
!= selected intervention
!= referral text
```

A Step-2 evidence-backed suggestion may influence the UI, but it contributes referral wording only after the clinician selects/adds it.

Removing a default intervention removes its corresponding plan phrase from the live referral.

The formatter must not silently restore an evidence-supported intervention merely because it is normally recommended.

---

# 6. Clinical-picture composition

Clinical-picture text may draw from:

```text
explicit CU-1 findings
+ bounded product phenotype tags
```

Ordering should favour concise symptom/impairment language and suppress duplicates.

Examples of semantic precedence:

```text
quadriceps_weakness selected
→ prefer precise "αδυναμία τετρακεφάλου"
→ suppress generic product weakness phrase

objective_weakness selected without more specific weakness
→ use objective weakness phrase
→ suppress generic product weakness phrase

stiffness_symptom selected
+ active/passive ROM restriction selected
→ stiffness may remain as symptom
→ ROM restriction remains a separate assessed finding
→ never collapse one into the other
```

The output must never convert an unselected/missing finding into reassurance or a negative statement.

---

# 7. Functional-impact composition

Selected functional impairments describe what is difficult; they do not autonomously prescribe a corresponding rehabilitation component.

Examples:

```text
stairs selected
→ referral may state difficulty on stairs
→ functional-task retraining enters the PLAN only if selected

walking limitation selected
→ referral may state reduced walking tolerance
→ gait/endurance work enters the PLAN only if selected
```

This preserves clinician autonomy and the Step-2 suggestion boundary.

---

# 8. Plan composition and ordering

Plan phrases are emitted only from selected `rehab_directions`/supported product items.

Preferred deterministic ordering:

```text
1. therapeutic exercise / active rehabilitation
2. progressive strengthening
3. education + self-management
4. mobility / graded activity / endurance
5. neuromuscular / balance
6. gait / functional-task retraining
7. home programme / delivery support
8. walking-aid/support context
9. adjuncts
```

Generic active-rehabilitation wording should not duplicate therapeutic-exercise wording. The implicit physiotherapy-assessment concept may shape the sentence but need not appear as another visible comma-separated item.

No exact sets/repetitions/frequency are invented.

---

# 9. Context-aware phrase refinement

The formatter may refine a selected intervention using an already-selected relevant finding/function, but the refinement must not add a new clinical fact or a new treatment selection.

Examples:

```text
progressive_strengthening + quadriceps_weakness
→ may mention emphasis on quadriceps strengthening

mobility_exercise_when_restricted + ROM restriction
→ may mention mobility directed at the documented ROM restriction

functional_task_retraining + stairs
→ may mention functional retraining for stairs

functional_task_retraining + sit_to_stand
→ may mention sit-to-stand retraining

neuromuscular/balance component + balance_deficit
→ may mention balance/neuromuscular control
```

If the intervention is not selected, the formatter does not create it from the finding/function.

---

# 10. Goals

The routine product path should not require a separate generic Goals screen.

Existing CU-1 goals remain valid structured data for compatibility/power-user use, but generic goals that merely restate an already rendered plan should be suppressed from the standard concise referral.

Goals that add non-redundant information may render, especially:

```text
return to work/sport
patient-priority activity
mobility independence / walking-aid objective
```

This is de-duplication, not deletion of clinician-selected structured state.

---

# 11. Adjuncts and evidence boundary

Adjuncts render only when explicitly selected.

For mixed-guideline items such as manual therapy or acupuncture:

```text
UI → shows evidence-state cue + source detail
referral → neutral adjunct wording only
```

The copied referral does not automatically say `guidelines differ` or attach citations.

Every selected adjunct must remain linguistically subordinate to the active rehabilitation plan; it must not become the main treatment sentence.

---

# 12. Weight-management and walking-aid seams

Step 2 found:

```text
weight management → no dedicated current CU-1 selectable ID
walking aid       → canonical ID exists but not exposed in current Knee UI scope
```

Step 3 therefore does not invent hidden selections.

- weight-management content is not auto-inserted into the referral;
- walking-aid wording may be specified for future bounded UI exposure, but it does not become visible/selectable merely because the template supports it.

---

# 13. Copy-readiness / safety boundary

Live preview may exist before full readiness, but the primary Copy action must respect inherited CU-1 validation/safety authority.

Copy-ready requires at minimum:

```text
valid Knee-OA primary route
clinician-established diagnosis semantics satisfied by the product flow
right/left/bilateral laterality
no CU-1 formatter-blocking validation error
no unresolved blocking/urgent safety state
```

The product template must not bypass the existing CU-1 validation engine.

---

# 14. Manual edit boundary

Default output mode:

```text
auto_live
```

An explicit secondary `Επεξεργασία` action may create an ephemeral manual text buffer from the current derived referral.

Hard rules:

```text
manual text edit does not mutate structured selections
structured selections do not get reverse-inferred from edited prose
manual buffer is not patient persistence
```

The implementation must not silently overwrite a dirty manual buffer after later structured changes. Exact edit-mode interaction may be finalized during Step 5 prototype UX, but semantic ownership is frozen here.

---

# 15. Determinism and de-duplication

For identical structured/product state, output must be identical.

The template contract must define:

```text
phrase-group ordering
canonical phrase ownership
specific-over-generic precedence
duplicate suppression
punctuation/conjunction rules
empty-group omission
adjunct subordination
```

No LLM generation is required for routine referral text.

---

# 16. Acceptance evidence

Step 3 is design-complete only when machine fixtures prove at least:

```text
1. default right Knee-OA referral
2. pain + stiffness + weakness + stairs
3. ROM restriction + selected mobility work
4. balance deficit + selected neuromuscular/balance work
5. walking limitation + selected gait/endurance work
6. selected functional-task retraining for stairs/sit-to-stand
7. selected mixed-guideline adjunct without evidence text leaking into referral
8. omission of a default core intervention removes its wording
9. suggestion without selection does not render
10. bilateral laterality grammar
11. generic weakness suppressed by specific quadriceps/objective weakness
12. no machine-ID leak / no duplicate phrase / no invented exact dosage
```

---

# 17. Out of scope

```text
runtime implementation
visual prototype code
second diagnosis
billing/auth/persistence
AI-generated referral prose
autonomous evidence updates
PR/merge/deploy/production smoke
```

---

# 18. REPLAN triggers

Replan rather than patch around the contract if:

- the existing `ReferralDraftV1` cannot carry required selected state without semantic falsehood;
- product-local phenotype tags begin duplicating substantial clinical taxonomy rather than filling bounded UI gaps;
- safe live generation requires hidden treatment selection;
- de-duplication would discard clinically material clinician-selected information;
- the current CU-1 safety/validation engine cannot protect Copy readiness for the product flow.

---

# 19. Exact next gate

```text
create human Step-3 design
→ create machine template contract + deterministic fixtures
→ validate against CU-1 IDs + frozen Step-2 evidence contract
→ exact active-writer design review
→ freeze/release writer if clean
```
