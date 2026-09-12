# UX_CONTRACT_CURRENT.md — Physio Referral Product UX v1.1

> **STATUS:** PRODUCT-OWNER APPROVED STEP-1 UX / STEP-2 EVIDENCE-STATE REPLAN INCORPORATED.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Scope:** Knee Osteoarthritis vertical slice only.
> **Implementation:** NOT IMPLEMENTED by this document.
> **Design intent:** minimal, modern, mobile-first, direct-manipulation experience with clinical/evidence complexity hidden beneath a calm surface.

---

# 1. North star

```text
COMPLEXITY LIVES UNDERNEATH.
CONFIDENCE LIVES ON THE SURFACE.
```

The user should experience:

```text
choose
→ refine
→ see the referral adapt immediately
→ inspect evidence only when wanted
→ copy
```

Not:

```text
complete long form
→ validate
→ generate
→ inspect warnings
→ output
```

Routine use should not depend on labels such as `Why?`, `Advanced settings`, `Generate`, `Keep anyway` or explanatory paragraphs.

---

# 2. Default task surface

The prototype opens directly into:

```text
Παραπομπή φυσιοθεραπείας
Οστεοαρθρίτιδα γόνατος
```

No dashboard/onboarding is required for the single-diagnosis prototype.

## Laterality

Compact segmented control:

```text
Δεξί | Αριστερό | Αμφω
```

## Clinical picture

Large selectable rows, not checkbox grids:

```text
Πόνος
Δυσκαμψία
Αδυναμία
Περιορισμός λειτουργικότητας
```

Selecting functional limitation progressively reveals only relevant subchoices such as:

```text
Βάδιση
Σκάλες
Έγερση
Άσκηση
```

The interface expands only in response to the user's choices.

---

# 3. Smart starting plan

Choosing Knee OA loads a reviewed starting plan rather than a blank rehabilitation form.

The exact evidence-authoritative contents are owned by:

```text
KNEE_OA_EVIDENCE_DESIGN_V1.md
contracts/knee_oa_evidence_contract_v1.yaml
```

Current reviewed starting plan:

```text
implicit: individualized physiotherapy assessment / active rehabilitation
visible selected: therapeutic exercise
visible selected: progressive strengthening
visible selected: education & self-management
```

Context-dependent rehabilitation appears only when the phenotype/function makes it relevant.

The clinician may remove/change any non-safety-locked choice.

---

# 4. Evidence visual language — six states

Step 2 proved that the original five-state model could not honestly represent material guideline disagreement. The current UX therefore uses six states.

Color is an important cue but never the only cue. Every state also has a subtle non-colour signal such as weight, underline/accent rule, dot, compact symbol or accessible text equivalent.

## 4.1 `recommended_or_supported`

Surface character:

```text
stronger text weight
+ green accent / fine underline or equivalent
+ subtle second cue
```

No large permanent `Recommended` badge is required.

## 4.2 `conditional_or_context_dependent`

```text
neutral / blue-grey accent
+ restrained second cue
```

Meaning: reasonable only when the patient's phenotype, impairment, function, preferences or delivery context supports it.

## 4.3 `limited_or_insufficient_evidence`

```text
amber accent
+ compact uncertainty cue
```

Meaning: evidence is insufficient/uncertain for routine recommendation.

This must never imply proven ineffectiveness.

## 4.4 `guideline_conflict_or_mixed`

Greek semantic:

```text
Οι οδηγίες διαφέρουν
```

Use when reviewed credible frameworks materially differ in recommendation direction or practical use.

Visual character should be distinct from green, amber and danger/red. A restrained indigo/violet or split-state accent with a non-colour cue is preferred; exact styling belongs to prototype design.

This is not a vote or arithmetic consensus score.

## 4.5 `recommendation_against_routine_use`

```text
muted red / caution accent
+ compact caution cue
```

Meaning: reviewed guidance supports avoiding routine use in the relevant indication/context.

The evidence detail must still distinguish `recommended against` from `proven harmful`.

## 4.6 `not_yet_assessed`

```text
grey / unclassified
+ neutral information cue
```

Meaning: the product has not completed an evidence assessment. Never convert `not assessed` to `not recommended`.

---

# 5. Contextual evidence bubbles

Do not use large warning boxes for routine evidence states.

When attention is useful, show a compact anchored bubble such as:

```text
Περιορισμένη τεκμηρίωση
Οι οδηγίες διαφέρουν
Δεν συνιστάται για συνήθη χρήση
```

The bubble:

- stays close to the selected item;
- does not block continuation;
- collapses when context changes;
- may expose a small `i` control;
- never scolds the clinician.

Avoid:

```text
WARNING!
Are you sure?
Guideline conflict!
Keep anyway?
```

---

# 6. Evidence detail — small `i`, deep evidence underneath

The `i` affordance is used sparingly.

First disclosure opens a compact bottom/side sheet containing:

```text
intervention
plain-language evidence state
1–3 lines: what it is expected to achieve / why it matters
source name + guideline/version year
Reviewed · date
Τεκμηρίωση ›
```

For mixed guidance the sheet must show the source-specific positions instead of a synthesized fake consensus.

Deeper `Τεκμηρίωση` may show full source details, native strength/certainty and concise recommendation summaries.

Information hierarchy:

```text
LEVEL 1 — color + subtle cue
LEVEL 2 — contextual bubble only when needed
LEVEL 3 — i → rationale + source/year + review date
LEVEL 4 — deeper evidence only on explicit request
```

No citation wall on the normal surface.

---

# 7. Evidence-backed suggestions

Suggestions may appear when:

- a reviewed core component is omitted;
- an explicit phenotype/function makes a contextual component relevant.

A compact suggestion should communicate:

```text
what is suggested
+ evidence-state cue
+ one source/year cue where appropriate
+ one-tap add
+ optional i
```

No suggestion becomes selected clinical truth automatically.

Current Step-2 policy allows omission suggestions for:

```text
therapeutic exercise
progressive strengthening
education & self-management
```

Context-driven examples:

```text
ROM restriction       → mobility
quadriceps weakness   → strengthening emphasis
balance deficit       → neuromuscular / balance work
walking limitation    → graded activity/endurance/gait
stairs / sit-to-stand → task-specific retraining
```

The first prototype does not automatically promote adjuncts such as manual therapy, soft-tissue techniques, acupuncture, taping, brace, walking aid or weight management.

---

# 8. Power-user layer

Power-user capability is required but should occupy almost no default screen space.

Collapsed:

```text
Περισσότερα ›
```

Expanded content is context-sensitive and may include:

```text
adjuncts
manual / soft-tissue options
supports / brace / taping
gait/balance detail
return-to-work/activity detail
special restrictions
clinician-entered nuance
additional findings/measurements
```

If active selections remain after collapse:

```text
Περισσότερα · 2 ενεργά ›
```

No giant “advanced” card.

---

# 9. Live referral — no Generate button

Every meaningful selection updates the referral immediately.

Examples:

```text
Αδυναμία
→ strength emphasis changes

Σκάλες
→ task-specific functional wording appears

remove intervention
→ associated referral wording disappears/adjusts
```

Routine flow must not require:

```text
Generate
Validate
Refresh referral
```

The final text may be edited before copying, but free-text editing must not silently rewrite structured selections unless a later bidirectional-edit design is explicitly approved.

---

# 10. Referral template principle

Generated referral =

```text
stable clinical core
+ diagnosis/laterality
+ selected phenotype/findings
+ functional modifiers
+ reviewed plan components
+ clinician-selected optional components
+ safety/restriction context when relevant
```

Preserve physiotherapist autonomy.

Prefer:

> progressive strengthening/loading adapted to clinical response

rather than prescribing detailed sets/reps or pretending the referral tool replaces physiotherapy assessment.

---

# 11. Preview / actions

## Mobile

Persistent compact bottom surface:

```text
Παραπομπή
Έτοιμη ↑
```

Expanded sheet shows the live referral.

Primary action:

```text
Αντιγραφή
```

Secondary actions under one compact overflow control:

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

Use space without adding complexity:

```text
left/main ~55%   clinical flow
right ~45%       live referral
```

Evidence opens in a compact side sheet/drawer.

---

# 12. Output length

Do not keep a permanent Short/Detailed switch on the routine surface.

Default output should simply be useful and concise.

A secondary action such as `Συντομότερο κείμενο` may exist later in edit/overflow context.

---

# 13. Final state

No numeric quality score.

Normal:

```text
Έτοιμη
```

If material review remains:

```text
Έτοιμη · 1 σημείο για έλεγχο
```

Tapping reveals only the relevant issue, such as:

```text
core option omitted
evidence limited
reviewed guidelines differ
option advised against for routine use
required safety context unresolved
```

No dashboard of green ticks.

---

# 14. Visual design language

- generous white space;
- near-black primary text;
- calm grey secondary text;
- one main interaction accent plus restrained semantic evidence accents;
- system/native-feeling typography;
- large touch targets;
- light dividers;
- restrained radii;
- minimal shadows;
- no decorative gradients;
- no card-inside-card proliferation;
- no routine checkbox matrix;
- no unnecessary icons;
- sheets/drawers for depth;
- motion only when it explains state transition;
- respect reduced-motion preferences.

The result should feel quiet and direct, not like a medical dashboard.

---

# 15. Mobile-first acceptance path

Typical routine conceptual flow:

```text
Knee OA
→ Right
→ Pain
→ Weakness
→ Stairs
→ Copy
```

Target: roughly 5–7 meaningful taps with little or no typing.

---

# 16. Accessibility invariants

- color is never the sole evidence signal;
- comfortable touch targets;
- keyboard/focus support on desktop;
- readable text equivalents for evidence cues;
- muted cautions remain distinguishable;
- no evidence meaning depends on animation.

---

# 17. Language principles

Routine UI uses concise natural Greek.

Avoid generic SaaS/tool language where structure can communicate the action:

```text
Why?
More options
Advanced settings
Generate
Keep anyway
Are you sure?
```

Useful local labels may include:

```text
Περισσότερα
Πρόταση
Τεκμηρίωση
Αφαίρεση
Διατήρηση
Αντιγραφή
```

Use them only when they materially help.

---

# 18. Prototype information architecture

The Knee-OA prototype includes only:

```text
1. Knee OA + laterality
2. clinical phenotype
3. functional limitations
4. evidence-aware starting plan
5. context-driven suggestions
6. compact power-user expansion
7. six evidence visual states
8. contextual bubbles
9. i evidence sheet
10. live referral
11. editable-output boundary
12. final review state
13. copy + secondary actions
14. source year + Reviewed metadata
15. ephemeral patient draft by default
```

---

# 19. Explicit non-goals

This contract does not authorize:

- runtime code changes;
- second diagnosis;
- billing/subscription/account work;
- patient persistence;
- autonomous evidence updating;
- autonomous diagnosis;
- autonomous treatment selection;
- exhaustive modality catalog.

---

# 20. Prototype fail conditions

Fail review if:

```text
routine path becomes a long form
irrelevant choices dominate the screen
Generate is required
advanced options dominate default view
bibliography occupies routine surface
insufficient / against / conflict states are collapsed together
color is the only evidence signal
clinician cannot override a non-safety suggestion
suggestion lacks traceable provenance
source publication year is confused with product review date
mobile routine path requires substantial typing
normal state is cluttered by warnings/ticks
```

Positive target:

> A clinician can create a good routine Knee-OA referral without studying the interface, while evidence depth remains one deliberate tap away.

---

# 21. Replan triggers

Material replan is required if later design/testing proves that:

- six evidence states remain insufficient;
- safety logic cannot remain non-intrusive;
- a clinically essential input cannot fit progressive disclosure;
- live generation creates unsafe ambiguity;
- the existing CU-1 state cannot support the product without substantive contract change;
- accessibility requires a different cue architecture.

Cosmetic tuning within these semantics does not reopen the contract.
