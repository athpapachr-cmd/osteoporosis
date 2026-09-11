# KNEE_OA_TEMPLATE_DESIGN_V1.md — Physio Referral Step 3

> **STATUS:** REVIEW-HARDENED DESIGN CANDIDATE — 2026-09-11.
> **Scope:** Knee Osteoarthritis only.
> **Parent:** frozen Step-1 UX + frozen Step-2 evidence design.
> **Existing CU-1 runtime/formatter:** read-only in this slice.
> **Machine companion:** `contracts/knee_oa_template_contract_v1.yaml`.
> **Fixtures:** `contracts/knee_oa_template_fixtures_v1.yaml`.
> **Runtime implementation:** NOT AUTHORIZED.

---

# 1. Product objective

The routine interaction should feel as if the referral already exists and becomes more specific while the clinician taps the few clinically relevant choices.

The output model is therefore:

```text
structured state
→ deterministic semantic projection
→ deterministic Greek phrase composition
→ live referral
```

There is no routine Generate step and no LLM prose generation requirement.

---

# 2. Authority boundary

The template reads:

```text
ReferralDraftV1
+ KneeOAProductPhenotypeV1
+ frozen Step-2 evidence item identities/state
```

It does not write back inferred findings/goals/interventions.

Hard boundary:

```text
OUTPUT PROJECTION != CLINICAL STATE MUTATION
```

A phrase may become more specific because two already-selected facts coexist; the formatter may not manufacture either fact.

---

# 3. Bounded product phenotype overlay

The frozen Step-1 UX includes simple routine concepts that current CU-1 cannot represent faithfully as the same thing.

The product overlay is intentionally tiny:

```text
KneeOAProductPhenotypeV1
  stiffness_symptom: boolean
  weakness_symptom_or_context: boolean
```

These values are clinician-selected presentation tags and remain ephemeral with the referral draft.

They are necessary because:

```text
stiffness symptom != active/passive ROM restriction
weakness symptom/context != objective weakness
```

Precedence:

```text
quadriceps_weakness
> objective_weakness
> weakness_symptom_or_context
```

A more specific selected CU-1 finding suppresses the generic weakness phrase in prose only; it does not delete the product tag.

---

# 4. Diagnosis and laterality

Routine opening:

```text
Παραπομπή για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας {laterality_phrase}.
```

Natural laterality:

```text
right      → δεξιού γόνατος
left       → αριστερού γόνατος
bilateral  → και των δύο γονάτων
```

The product does not use `Οστεοαρθρίτιδα γόνατος (δεξιά)` in the final referral.

The single-diagnosis prototype must not diagnose Knee OA merely because the clinician opened the Knee-OA flow. Copy readiness therefore requires the existing CU-1 formal-diagnosis semantics to be satisfied, including explicit clinician assertion (`formal_assertion_state=yes`) for this route, in addition to laterality `right | left | bilateral`.

Preview may exist before copy readiness, but final Copy must remain blocked until those semantics and inherited validation/safety requirements are satisfied.

---

# 5. Clinical-picture sentence

If at least one clinical-picture item exists:

```text
Η κλινική εικόνα περιλαμβάνει {clinical_list}.
```

Routine phrase ownership:

```text
pain                              → πόνο
stiffness_symptom                 → δυσκαμψία
weakness_symptom_or_context       → μυϊκή αδυναμία
quadriceps_weakness               → αδυναμία τετρακεφάλου
objective_weakness                → αντικειμενικά διαπιστωμένη μυϊκή αδυναμία
effusion                          → ενδαρθρική συλλογή
balance_deficit                   → έλλειμμα ισορροπίας
active_rom_restricted             → περιορισμό ενεργητικού εύρους κίνησης
passive_rom_restricted            → περιορισμό παθητικού εύρους κίνησης
active + passive ROM restriction  → περιορισμό ενεργητικού και παθητικού εύρους κίνησης
```

The combined ROM phrase replaces the two separate ROM phrases when both are selected.

Functional-limitation findings such as `walking_limitation`, `stairs_limitation` and `sit_to_stand_limitation` are projected into the functional block rather than duplicated in the clinical-picture block.

---

# 6. Functional-impact sentence

If at least one functional item exists:

```text
Λειτουργικά, υπάρχει δυσχέρεια {functional_list}.
```

Canonical phrases:

```text
walking_tolerance / walking_limitation → στη βάδιση
standing_tolerance                    → στην παρατεταμένη ορθοστασία
stairs / stairs_limitation            → στις σκάλες
sit_to_stand / sit_to_stand_limitation→ στην έγερση από καθιστή θέση
squat                                 → στο βαθύ κάθισμα
kneeling                              → στο γονάτισμα
running                               → στο τρέξιμο
sport_gym / sport_or_exercise_limitation → στην άθληση ή την άσκηση
manual_work                           → στη χειρωνακτική εργασία
community_mobility                    → στην κοινοτική κινητικότητα
patient_priority_activity             → στη δραστηριότητα προτεραιότητας του ασθενούς
```

Output-only aliasing de-duplicates equivalent finding/function concepts; structured state remains unchanged.

---

# 7. Stable active-plan sentence

The active plan always preserves the frozen Step-2 implicit request for individualized active rehabilitation.

If at least one selected plan phrase exists:

```text
Παρακαλώ για ενεργητικό, εξατομικευμένο πρόγραμμα με {plan_list}, προσαρμοσμένο στην κλινική ανταπόκριση και στους λειτουργικούς στόχους.
```

If all optional/visible plan items have been removed:

```text
Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένο ενεργητικό πρόγραμμα, προσαρμοσμένο στην κλινική ανταπόκριση και στους λειτουργικούς στόχους.
```

This fallback is the explicit Step-2 implicit core; it is not a hidden re-selection of a removed visible intervention.

---

# 8. Plan phrase ordering

Deterministic order:

```text
therapeutic_exercise
progressive_strengthening
education_and_self_management
mobility_exercise_when_restricted
graded_activity_exposure
progressive_endurance_or_capacity_work
neuromuscular_proprioceptive_training
balance_stepping_recovery_training
gait_walking_practice
functional_task_retraining
home_exercise_programme
walking_aid_assessment_and_training
```

The implicit `physiotherapy_assessment_and_individualized_active_rehabilitation` shapes the sentence and is not repeated as another list item.

---

# 9. Context-aware plan phrases

Selected plan items may be refined using already-selected context.

## Progressive strengthening

Default:

```text
προοδευτική ενδυνάμωση
```

If `quadriceps_weakness` is selected:

```text
προοδευτική ενδυνάμωση με έμφαση στον τετρακέφαλο
```

Generic product weakness does not justify the quadriceps-specific phrase.

## Education / self-management

```text
education_and_self_management
→ εκπαίδευση για αυτοδιαχείριση
```

The phrase is intentionally concise so it joins naturally with other rehabilitation components.

## Mobility

Default:

```text
ασκήσεις κινητικότητας
```

If active/passive ROM restriction exists:

```text
ασκήσεις κινητικότητας για το περιορισμένο εύρος κίνησης
```

## Neuromuscular / balance

```text
neuromuscular_proprioceptive_training → νευρομυϊκή και ιδιοδεκτική επανεκπαίδευση
balance_stepping_recovery_training    → εκπαίδευση ισορροπίας και αντιδράσεων ανάκτησης
```

## Gait

```text
gait_walking_practice → επανεκπαίδευση βάδισης
```

## Functional task retraining

If selected and one or more mapped functional limitations are selected:

```text
λειτουργική επανεκπαίδευση για {task_list}
```

Task wording is grammatically ready for the preposition `για`:

```text
stairs       → τις σκάλες
sit_to_stand → την έγερση από καθιστή θέση
squat        → το βαθύ κάθισμα
kneeling     → το γονάτισμα
running      → το τρέξιμο
manual_work  → τις απαιτήσεις χειρωνακτικής εργασίας
sport_gym    → την άθληση ή την άσκηση
patient_priority_activity → τη δραστηριότητα προτεραιότητας του ασθενούς
```

If no mapped task is selected, use the generic phrase:

```text
λειτουργική επανεκπαίδευση
```

A functional limitation never creates `functional_task_retraining` unless that intervention is selected.

---

# 10. Default core output

Frozen Step-2 visible default:

```text
therapeutic_exercise
progressive_strengthening
education_and_self_management
```

Final exact default plan phrase:

```text
θεραπευτική άσκηση, προοδευτική ενδυνάμωση και εκπαίδευση για αυτοδιαχείριση
```

Removing any one default removes that item's phrase.

---

# 11. Goals and redundancy

The product does not require a routine Goals screen.

Generic CU-1 goals that only restate the rendered plan are omitted from standard concise prose, for example:

```text
improve_strength
improve_self_management
restore_safe_functional_rom
```

Potentially non-redundant goals may render in a separate final emphasis clause if selected, especially:

```text
graded_return_to_work
graded_return_to_sport
patient-priority activity context
optimize_safe_walking_aid_use
maintain_or_regain_adl_independence
```

No selected goal is deleted from structured state.

---

# 12. Adjunct sentence

Adjuncts are never merged into the core active-plan list.

If one or more adjuncts are explicitly selected:

```text
Συμπληρωματικές επιλογές: {adjunct_list}, ως προσθήκη και όχι υποκατάστατο του ενεργητικού προγράμματος.
```

Supported product phrases:

```text
manual_therapy              → manual therapy
soft_tissue_techniques      → τεχνικές μαλακών μορίων
acupuncture                 → βελονισμός
taping                      → taping
orthosis_or_brace_context   → νάρθηκας ή ορθωτικό βοήθημα
```

Step-2 evidence state remains UI-only by default. Selecting acupuncture/manual therapy does not inject `Οι οδηγίες διαφέρουν` or bibliography into the referral.

`dry_needling` is not selectable on the Knee-OA product surface under frozen Step 2.

---

# 13. Evidence separation

The clinician-facing product may show:

```text
state colour/cue
bubble
source/year
i sheet
full evidence detail
```

The copied referral contains none of those by default.

This prevents the clinical handoff from becoming a literature summary while preserving evidence transparency for the clinician making the selection.

---

# 14. Restrictions and clinician notes

Existing CU-1 explicit restrictions retain authority and render before optional clinician free text.

Restrictions are not inferred from Knee OA or from an evidence state.

Clinician free text remains literal clinician-owned text after existing normalization/safety boundaries; the template does not parse it back into structure.

---

# 15. Walking aid / weight management

`walking_aid_assessment_and_training` already has a CU-1 ID. The template defines its eventual phrase but does not expose the option; UI exposure remains a later bounded Step-5/presentation decision.

Weight management remains Step-2 advisory-only because no dedicated current CU-1 selectable product ID exists. It is not auto-inserted into referral prose.

---

# 16. Live output / Copy readiness

Preview is live for every structured/product-state change.

Copy readiness is separate from preview existence.

Copy requires:

```text
primary route = knee_osteoarthritis
formal diagnosis semantics satisfied by explicit clinician assertion
laterality = right | left | bilateral
no inherited CU-1 formatter-blocking validation error
no unresolved inherited blocking/urgent safety state
```

The template does not create a second diagnosis, validation or safety engine.

---

# 17. Manual edit mode

Default:

```text
auto_live
```

`Επεξεργασία` creates an ephemeral text snapshot.

Semantic rules:

```text
edited text != structured clinical state
edited text does not write back to selections
structured selections are never inferred from prose
```

A dirty manual buffer must not be silently overwritten by later live projection. The Step-5 prototype may choose the exact interaction for leaving manual edit mode, but it must preserve this ownership boundary.

---

# 18. Exact deterministic examples

## Default right Knee OA

```text
Παραπομπή για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας δεξιού γόνατος. Παρακαλώ για ενεργητικό, εξατομικευμένο πρόγραμμα με θεραπευτική άσκηση, προοδευτική ενδυνάμωση και εκπαίδευση για αυτοδιαχείριση, προσαρμοσμένο στην κλινική ανταπόκριση και στους λειτουργικούς στόχους.
```

## Pain + stiffness + generic weakness + stairs

```text
Παραπομπή για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας δεξιού γόνατος. Η κλινική εικόνα περιλαμβάνει πόνο, δυσκαμψία και μυϊκή αδυναμία. Λειτουργικά, υπάρχει δυσχέρεια στις σκάλες. Παρακαλώ για ενεργητικό, εξατομικευμένο πρόγραμμα με θεραπευτική άσκηση, προοδευτική ενδυνάμωση και εκπαίδευση για αυτοδιαχείριση, προσαρμοσμένο στην κλινική ανταπόκριση και στους λειτουργικούς στόχους.
```

Machine fixtures own final exact wording for all reviewed scenarios.

---

# 19. Hard invariants

```text
DIAGNOSIS MUST BE CLINICIAN-ASSERTED BEFORE COPY
SUGGESTION != SELECTION != OUTPUT
STIFFNESS != ROM RESTRICTION
GENERIC WEAKNESS != OBJECTIVE WEAKNESS
FUNCTIONAL LIMITATION != AUTOMATIC REHAB SELECTION
EVIDENCE CUE != REFERRAL PROSE
ADJUNCT != CORE ACTIVE REHABILITATION
OUTPUT PROJECTION != STATE MUTATION
MANUAL EDIT != STRUCTURED WRITEBACK
MISSING != NEGATIVE
NO machine id leak
NO exact exercise dose invention
NO LLM required for routine text
```

---

# 20. Step-3 non-goals

```text
runtime/UI implementation
billing/account/persistence
patient identifiers
second diagnosis
AI prose generation
autonomous evidence update
full evidence-interaction implementation
```

---

# 21. Exact review gate

Step 3 freezes only after:

```text
machine template contract
+ deterministic fixtures
+ validator against existing CU-1 and frozen Step-2 IDs
+ exact design review
+ clean machine gate
```
