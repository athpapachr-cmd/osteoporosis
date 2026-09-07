# UX_CONTRACT_CURRENT.md — Physio Referral Product UX v1

> **STATUS:** PRODUCT-OWNER APPROVED / FROZEN FOR KNEE-OA PROTOTYPE.
> **Date:** 2026-09-07 Asia/Nicosia.
> **Scope:** Physio Referral productization, Knee Osteoarthritis vertical slice only.
> **Implementation:** NOT IMPLEMENTED by this document.
> **Design intent:** minimal, modern, mobile-first, direct-manipulation experience with clinical depth hidden beneath a calm surface.

---

# 1. North-star interaction principle

```text
COMPLEXITY LIVES UNDERNEATH.
CONFIDENCE LIVES ON THE SURFACE.
```

The interface should feel closer to a first-party mobile product than to a conventional medical web form.

The user should not experience:

```text
form completion
→ validation page
→ generate
→ output
```

The intended mental model is:

```text
choose
→ refine
→ see the referral adapt immediately
→ understand evidence only when needed
→ copy
```

Hard UX rule:

> The interface should explain itself through layout, state and direct manipulation. If routine use requires repeated labels such as “Why?”, “Advanced settings”, “Generate” or instructional paragraphs, the interaction design should be reconsidered.

---

# 2. Default surface — what is visible first

For the prototype, the product opens directly into the clinical task.

```text
Παραπομπή φυσιοθεραπείας
Οστεοαρθρίτιδα γόνατος
```

No dashboard, onboarding carousel or welcome screen is required for the single-diagnosis prototype.

## 2.1 Laterality

Use a compact segmented control:

```text
Δεξί | Αριστερό | Αμφω
```

Large tap targets; no dropdown for the normal path.

## 2.2 Clinical picture

Primary selectable rows, not checkboxes:

```text
Πόνος
Δυσκαμψία
Αδυναμία
Περιορισμός λειτουργικότητας
```

A selected row changes state directly.

When `Περιορισμός λειτουργικότητας` is selected, reveal only the relevant secondary choices, for example:

```text
Βάδιση
Σκάλες
Έγερση
Άσκηση
```

This is progressive disclosure: the user sees only the complexity created by their own choices.

---

# 3. Smart evidence-based starting plan

The user does not start from a blank rehabilitation plan.

Choosing Knee OA loads a reviewed default plan. The exact clinical contents are owned by the later Knee-OA evidence knowledge module; the UX contract defines only the behavior.

Each default plan row contains:

```text
intervention label
minimal evidence state cue
optional chevron or direct row affordance
```

There is no permanent explanatory paragraph beside every intervention.

The default should make the common evidence-aligned outcome require fewer actions than a poorly supported outcome.

The clinician can remove or change any non-safety-locked choice.

---

# 4. Evidence-state visual language

Color is a first-order cue but must not be the **only** cue.

The interface therefore combines color with one subtle non-color signal such as weight, underline/accent rule, dot or compact symbol.

## 4.1 `recommended_or_supported`

Desired surface character:

```text
stronger text weight
+ green accent / fine underline or equivalent
+ small non-color cue
```

Do not display a large permanent badge saying `Recommended` for every supported item.

## 4.2 `conditional_or_context_dependent`

Desired surface character:

```text
neutral / blue-grey accent
+ restrained secondary cue
```

This means the option may be reasonable depending on phenotype, goals, response or context.

## 4.3 `limited_or_insufficient_evidence`

Desired surface character:

```text
amber accent
+ compact uncertainty cue
```

Meaning:

> Evidence is insufficient/uncertain for routine recommendation.

This must **not** visually imply proven ineffectiveness.

## 4.4 `recommendation_against_routine_use`

Desired surface character:

```text
muted red or clearly distinct caution accent
+ compact caution cue
```

Meaning:

> A reviewed guideline/evidence source recommends against routine use for this indication.

This is distinct from limited evidence.

## 4.5 `not_yet_assessed`

Desired surface character:

```text
grey / unclassified
+ neutral information cue
```

Meaning:

> The product has not yet completed an evidence assessment for this intervention/indication.

Never convert `not assessed` into `not recommended`.

---

# 5. Evidence bubbles — contextual, modern, non-intrusive

The default surface should not show warning boxes.

When a clinically relevant evidence state needs attention, use a small contextual bubble anchored close to the selected item.

Examples of semantic content:

```text
Περιορισμένη τεκμηρίωση
```

or

```text
Δεν συνιστάται για συνήθη χρήση
```

The bubble should:

- be visually compact;
- avoid modal interruption;
- not block continuation;
- disappear/collapse naturally when context changes;
- provide a small `i` affordance when deeper explanation exists.

Avoid:

```text
WARNING!
Are you sure?
Keep anyway?
Guideline conflict!
```

The product informs rather than scolds.

---

# 6. Evidence detail — `i` is the main disclosure control

A small information control may be used, but sparingly.

The first tap should open a compact sheet/drawer, not navigate away.

Example structure:

```text
Θεραπευτική άσκηση

[Evidence state in plain language]

1–3 concise lines:
what the intervention is expected to improve / why it is relevant

Source name · publication/guideline year
Evidence reviewed · date

Τεκμηρίωση ›
```

`Τεκμηρίωση` may open a deeper layer containing full source details, recommendation wording summary, strength/certainty where available and relevant notes.

Information hierarchy:

```text
LEVEL 1 — color + subtle cue
LEVEL 2 — short contextual bubble when needed
LEVEL 3 — i → concise rationale + source/year + reviewed date
LEVEL 4 — full evidence details only on explicit request
```

No PubMed-like citation wall on the routine surface.

---

# 7. Suggestions — evidence-backed and easy to accept

The product may surface a suggestion when:

- a strongly supported core intervention was removed/omitted;
- the selected phenotype/function makes an additional intervention relevant;
- a selected intervention creates an evidence-sensitive alternative worth surfacing.

Suggestion presentation should be a compact contextual bubble/row, for example semantically:

```text
Πρόταση
Θεραπευτική άσκηση
Ισχυρή σύσταση · [source/year]
+
```

The exact wording is subject to UI polish; the semantic requirements are frozen:

```text
what is suggested
+ evidence-strength/status cue
+ short source/year cue
+ one-tap add
+ optional i for rationale
```

The suggestion must never cite a source/year that has not been reviewed and linked in the evidence module.

A suggestion does not automatically become selected clinical truth.

---

# 8. Power-user layer — full capability with near-zero default footprint

Power-user options are required.

They must not occupy routine screen space before needed.

Default collapsed representation:

```text
Περισσότερα  ›
```

This should be a single light row, not a large card.

When expanded, it may reveal context-sensitive advanced options such as:

```text
adjuncts
manual therapy/supports
balance/gait detail
return-to-sport/work detail
special restrictions
clinician-entered nuance
additional measurements/findings
```

Only options relevant to Knee OA and the current phenotype should appear.

If advanced options are active after collapse:

```text
Περισσότερα · 2 ενεργά  ›
```

The user therefore retains awareness without keeping the advanced panel open.

---

# 9. Dynamic referral — no Generate button

The referral is a live projection of current selections.

Every meaningful choice updates the referral immediately.

Examples:

```text
Αδυναμία
→ strength emphasis appears

Σκάλες
→ functional stair-training emphasis appears

remove intervention
→ associated referral wording disappears or adjusts
```

The user must not have to press:

```text
Generate
Validate
Refresh referral
```

to see the current output.

The referral remains editable before copying, but editing text must not silently rewrite structured selections unless a later explicit bidirectional-edit design is approved.

---

# 10. Referral template model

The generated referral is built from:

```text
stable clinical core
+ diagnosis/laterality
+ selected phenotype/findings
+ functional limitation modifiers
+ evidence-backed plan components
+ clinician-selected optional components
+ safety/restriction context when applicable
```

The template should preserve physiotherapist autonomy.

Prefer:

> progressive strengthening / loading adapted to clinical response

rather than excessively prescriptive dose-level instructions unless a later explicit clinical use case requires them.

The referral is not a hidden attempt to prescribe an entire physiotherapy session.

---

# 11. Referral preview behavior

## Mobile

A compact bottom surface remains available:

```text
Παραπομπή
Έτοιμη   ↑
```

Expanding it reveals the live referral.

Primary action:

```text
Αντιγραφή
```

Secondary actions live under one compact overflow control, for example:

```text
•••
```

Candidate secondary actions:

```text
Επεξεργασία
Εκτύπωση
PDF
Νέα παραπομπή
```

## Desktop

Use the larger screen without inflating complexity:

```text
left/main ~55%   clinical flow
right ~45%       live referral
```

Evidence detail may open as a compact side sheet/drawer.

---

# 12. Output-length behavior

Do not keep a permanent `Short / Detailed` mode switch on the default surface.

The default output should simply be good and appropriately concise.

A secondary action such as:

```text
Συντομότερο κείμενο
```

may be available in edit/overflow context if needed.

The user should not be asked to make formatting decisions before seeing a useful referral.

---

# 13. Final quality/safety state

Do not show a numeric quality score.

Normal state:

```text
Έτοιμη
```

If a material evidence/safety point requires review:

```text
Έτοιμη · 1 σημείο για έλεγχο
```

Tapping it reveals only the relevant point(s).

Potential checks include:

```text
strongly supported option omitted
selected option has limited evidence
selected option has recommendation against routine use
required structural/safety context unresolved
```

Do not render a dashboard of green ticks for routine normal state.

---

# 14. Visual design language

Design targets:

- generous white space;
- near-black primary text;
- calm grey secondary text;
- one primary interaction accent plus evidence-state semantic accents;
- system/native-feeling typography;
- large touch targets;
- very light dividers;
- restrained corner radii;
- minimal shadows;
- no decorative gradients;
- no card-inside-card proliferation;
- no small checkbox grids on the routine path;
- no icon where plain text or direct manipulation is clearer;
- bottom sheets / side sheets for deeper information;
- motion only when it explains state transition, never for decoration.

The design should feel deliberate and quiet rather than “medical dashboard”.

---

# 15. Mobile-first acceptance path

A typical routine referral should be achievable in approximately the conceptual flow:

```text
Knee OA
→ Right
→ Pain
→ Weakness
→ Stairs
→ Copy
```

The prototype should test whether the common case can be completed in roughly 5–7 meaningful taps without typing.

Typing should be optional for normal routine use.

---

# 16. Accessibility invariants

- color is never the only evidence-state signal;
- touch targets remain comfortably tappable;
- focus states and keyboard interaction remain supported on desktop;
- evidence cues retain readable text equivalents for assistive technology;
- muted warnings must remain sufficiently distinguishable without using aggressive visual alarms;
- motion respects reduced-motion preferences if implemented.

---

# 17. Language principles

The routine interface uses concise natural Greek.

Avoid interface language that sounds like generic SaaS/web tooling:

```text
Why?
More options
Advanced settings
Generate
Keep anyway
Are you sure?
```

Preferred interaction semantics are conveyed by structure or concise local labels such as:

```text
Περισσότερα
Πρόταση
Τεκμηρίωση
Αφαίρεση
Διατήρηση
Αντιγραφή
```

Even these labels should appear only where they materially help the action.

---

# 18. Prototype information architecture

The Knee-OA prototype includes only:

```text
1. Knee OA identity + laterality
2. clinical phenotype
3. functional limitations
4. evidence-aware default plan
5. context-driven suggestions
6. compact power-user expansion
7. evidence-state visual cues
8. evidence bubbles
9. i evidence sheet
10. live dynamic referral
11. editable output boundary
12. quality/safety review state
13. copy + secondary output actions
14. evidence reviewed/version metadata
15. ephemeral patient-draft behavior by default
```

---

# 19. Explicit Step-1 non-goals

This UX contract does **not** yet define or authorize:

- exact Knee-OA evidence claims;
- exact source set;
- exact evidence-strength mapping;
- runtime code changes;
- redesign of other diagnoses;
- billing/subscription implementation;
- account/entitlement implementation;
- patient persistence;
- AI literature updating;
- autonomous diagnosis;
- autonomous treatment decision;
- second disease vertical slice.

Those remain later steps.

---

# 20. UX acceptance criteria for implementation/prototype

The later prototype should fail review if any of these occur:

```text
routine path looks like a long medical form
common referral requires repeated scrolling through irrelevant choices
Generate is required before output updates
advanced options dominate the first screen
bibliography occupies routine surface
limited evidence and evidence-against are visually/semantically collapsed
color is the only evidence-state signal
user cannot override a non-safety evidence suggestion
suggestions lack traceable evidence provenance
guideline publication year is confused with product review date
mobile routine flow requires significant typing
normal state is cluttered by validation ticks/warnings
```

Positive target:

> A clinician should be able to create a good routine Knee-OA referral quickly without studying the interface, while deeper clinical/evidence information remains one deliberate tap away when wanted.

---

# 21. Freeze / replan triggers

This contract is frozen for the Knee-OA prototype.

A material change requires explicit replan if evidence/data design later proves that:

- the five-state evidence model is insufficient;
- required safety logic cannot remain non-intrusive;
- a clinically essential input cannot fit the progressive-disclosure flow;
- direct live generation creates unsafe ambiguity;
- the existing CU-1 structured state cannot support the required UX without a substantive contract change;
- accessibility requires a different evidence cue architecture.

Cosmetic tuning within these semantics does not reopen the contract.
