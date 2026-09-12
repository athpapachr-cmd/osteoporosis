# KNEE_OA_TEMPLATE_DESIGN_V1.md — Physio Referral Step 3

> **STATUS:** REVIEW-HARDENED DESIGN CANDIDATE — 2026-09-11.
> **Scope:** Knee Osteoarthritis only.
> **Parent:** frozen Step-1 UX + frozen Step-2 evidence design.
> **Existing CU-1 runtime/formatter:** read-only in this slice.
> **Machine companion:** `contracts/knee_oa_template_contract_v1.yaml`.
> **Primary fixtures:** `contracts/knee_oa_template_fixtures_v1.yaml`.
> **Edge fixtures:** `contracts/knee_oa_template_edge_fixtures_v1.yaml`.
> **Runtime implementation:** NOT AUTHORIZED.

---

# 1. Objective

The Knee-OA referral should already exist as a live deterministic projection and become more specific as the clinician selects the few relevant findings, functions and rehabilitation components.

```text
structured clinician-selected state
+ bounded Knee-OA product phenotype state
→ deterministic semantic projection
→ deterministic Greek composition
→ live referral preview
```

No routine `Generate` action and no LLM prose generation are required.

---

# 2. Authority boundary

The template reads but does not rewrite:

```text
ReferralDraftV1
+ KneeOAProductPhenotypeV1
+ frozen Step-2 evidence identities/states
```

Hard rule:

```text
OUTPUT PROJECTION != CLINICAL STATE MUTATION
```

A finding/function may refine the wording of an intervention only when both the context and the intervention are already selected.

```text
SUGGESTION != SELECTION != OUTPUT
FUNCTIONAL LIMITATION != AUTOMATIC REHAB SELECTION
```

---

# 3. Diagnosis authority and laterality

The fixed Knee-OA product screen must not autonomously diagnose Knee OA merely because it is open.

Copy readiness requires the existing CU-1 formal-diagnosis contract to be satisfied:

```text
route = knee_osteoarthritis
wording semantics = formal diagnosis
formal_assertion_state = yes
laterality = right | left | bilateral
```

Natural Greek laterality:

```text
right      → δεξιού γόνατος
left       → αριστερού γόνατος
bilateral  → και των δύο γονάτων
```

Opening phrase:

```text
Παραπομπή για εξατομικευμένη φυσιοθεραπευτική αποκατάσταση λόγω οστεοαρθρίτιδας {laterality_phrase}.
```

Preview may exist before copy readiness. Copy remains blocked if the clinician diagnosis assertion, laterality, product-scope validity or inherited CU-1 validation/safety gate is unresolved.

---

# 4. Bounded product phenotype overlay

The frozen minimal UX contains two concepts that must not be falsified into more specific CU-1 findings:

```text
stiffness_symptom: boolean
weakness_symptom_or_context: boolean
```

These are ephemeral clinician-selected presentation tags.

```text
stiffness symptom != active/passive ROM restriction
generic weakness != objective weakness
```

Weakness wording precedence:

```text
quadriceps_weakness
> objective_weakness
> weakness_symptom_or_context
```

The higher-specificity phrase suppresses the lower-specificity phrase in output only. Structured state is not deleted.

---

# 5. Product-supported power-user scope

The modern `Περισσότερα` layer must not silently expose the whole global Knee catalog. Step 3 defines a bounded Knee-OA subset in the machine contract across:

```text
findings
functional impairments
rehab directions
adjuncts
goals
```

Hard rule:

```text
EVERY SELECTED PRODUCT ITEM MUST:
render
OR fail closed
OR be covered by an explicit semantic de-duplication rule that preserves its meaning
```

Examples of explicit semantic de-duplication include:

```text
specific pain phrase supersedes generic pain phrase
confirmed effusion supersedes generic swelling phrase
specific/objective weakness supersedes generic weakness phrase
active + passive ROM restriction merges into one combined phrase
generic goal already represented by the active plan is suppressed in prose only
```

None of these de-duplication rules deletes clinician-selected structured state.

An item outside the bounded product scope causes product projection to fail closed rather than disappearing from the referral. The machine contract is normative for the exact supported IDs.

---

# 6. Clinical-picture composition

When at least one supported clinical-picture item is selected:

```text
Η κλινική εικόνα περιλαμβάνει {clinical_list}.
```

Examples of exact phrase ownership include:

```text
pain                        → πόνο
joint_line_pain             → πόνο στη μεσάρθρια γραμμή
anterior_peripatellar_pain  → πρόσθιο ή περιεπιγονατιδικό πόνο
stiffness_symptom           → δυσκαμψία
quadriceps_weakness         → αδυναμία τετρακεφάλου
objective_weakness          → αντικειμενικά διαπιστωμένη μυϊκή αδυναμία
generic weakness            → μυϊκή αδυναμία
effusion                    → ενδαρθρική συλλογή
swelling                    → οίδημα
tenderness                  → ευαισθησία στην ψηλάφηση
extension_lag               → υστέρηση έκτασης
balance_deficit             → έλλειμμα ισορροπίας
subjective_giving_way       → υποκειμενικό αίσθημα υποχώρησης του γόνατος
recurrent_instability       → υποτροπιάζοντα επεισόδια αστάθειας
```

Missing information never generates a negative/reassuring statement.

---

# 7. Functional-impact composition

Function is rendered separately from symptoms/findings:

```text
Λειτουργικά, υπάρχει δυσχέρεια {functional_list}.
```

The supported function map includes routine and power-user items such as:

```text
βάδιση
παρατεταμένη ορθοστασία
σκάλες
έγερση από καθιστή θέση
βαθύ κάθισμα
γονάτισμα
τρέξιμο
άλματα / προσγειώσεις
στροφές / αλλαγές κατεύθυνσης
άθληση / άσκηση
χειρωνακτική εργασία
κοινοτική κινητικότητα
patient-priority activity
```

Equivalent structured finding/function aliases are merged only in output, for example:

```text
walking_limitation + walking_tolerance
→ one walking phrase
```

---

# 8. Stable active-plan composition

The frozen Step-2 implicit core remains:

```text
individualized physiotherapy assessment / active rehabilitation
```

Visible default selected items remain:

```text
therapeutic_exercise
progressive_strengthening
education_and_self_management
```

Default plan phrase:

```text
θεραπευτική άσκηση, προοδευτική ενδυνάμωση και εκπαίδευση για αυτοδιαχείριση
```

Plan sentence:

```text
Παρακαλώ για ενεργητικό, εξατομικευμένο πρόγραμμα με {plan_list}, προσαρμοσμένο στην κλινική ανταπόκριση και στους λειτουργικούς στόχους.
```

If every visible plan component is removed, the implicit core renders as the fallback:

```text
Παρακαλώ για φυσιοθεραπευτική αξιολόγηση και εξατομικευμένο ενεργητικό πρόγραμμα, προσαρμοσμένο στην κλινική ανταπόκριση και στους λειτουργικούς στόχους.
```

This is not silent re-selection of a removed visible item.

---

# 9. Context-aware plan refinement

A selected plan component may become more specific from already-selected context.

Examples:

```text
progressive_strengthening + quadriceps_weakness
→ προοδευτική ενδυνάμωση με έμφαση στον τετρακέφαλο

mobility_exercise_when_restricted + ROM restriction
→ ασκήσεις κινητικότητας για το περιορισμένο εύρος κίνησης

functional_task_retraining + stairs
→ λειτουργική επανεκπαίδευση για τις σκάλες
```

A finding/function does not create the corresponding intervention by itself.

The exact deterministic ordering and phrase ownership live in the machine contract.

---

# 10. Goals without another routine screen

The routine iPhone-like path does not require a separate generic Goals screen.

Generic goals that simply repeat the already-rendered active plan are explicitly classified as **output-redundant**: they remain in structured state but do not create repetitive prose.

Non-redundant selected goals may render, including:

```text
graded return to work
graded return to sport
safe walking-aid use
ADL independence
```

One goal uses `Επιπλέον στόχος`; multiple goals use `Επιπλέον στόχοι`.

---

# 11. Adjuncts remain subordinate

Adjuncts render only after explicit clinician selection and in a separate sentence:

```text
Συμπληρωματικές επιλογές: {adjunct_list}, ως προσθήκη και όχι υποκατάστατο του ενεργητικού προγράμματος.
```

The referral does not include Step-2 evidence badges, `Οι οδηγίες διαφέρουν`, guideline names or citations by default. Those remain clinician-facing UI evidence.

Dry needling remains excluded from the Knee-OA product surface under frozen Step 2.

---

# 12. Restrictions and clinician note are preserved

Explicit CU-1 restrictions remain clinician-owned state and use existing CU-1 Greek restriction labels.

If present:

```text
Να τηρηθούν οι καταγεγραμμένοι περιορισμοί: {restriction_list}.
```

Optional clinician free text is appended only after existing normalization:

```text
Κλινική σημείωση: {clinician_note}.
```

Neither is reverse-parsed into structured state. Edge fixtures explicitly prove that these selections are not silently lost.

---

# 13. Safety integration finding — true locking

The frozen Knee profile contains:

```text
true_locking_or_major_mechanical_rom_block
```

but the current CU-1 rule catalog does not autonomously turn that finding itself into a safety trigger. Safety flags are explicit clinician-confirmed concerns.

Therefore the first Knee-OA product surface does **not** expose true locking as a selectable power-user finding.

```text
true locking exists in CU-1 taxonomy
+ no current finding→safety trigger
→ product exposure deferred
```

Future exposure requires a bounded safety/reassessment mapping rather than a second hidden product safety engine.

---

# 14. Walking aid and weight-management seams

Walking aid already has a canonical CU-1 rehab ID. The template contains its phrase, but current Knee UI exposure remains separately gated.

Weight management remains advisory-only in Step 3 because frozen Step 2 found no dedicated current CU-1 selectable machine ID. No weight-management sentence is inferred or auto-inserted.

---

# 15. Copy readiness vs live preview

Live preview may update after every valid product selection.

Copy requires:

```text
explicit clinician OA assertion
right/left/bilateral laterality
all selected items within product-supported scope
no inherited CU-1 formatter-blocking validation error
no unresolved inherited blocking/urgent safety state
```

Negative Copy-readiness fixtures explicitly prove:

```text
formal_assertion_state != yes → formal_diagnosis_assertion_required
laterality not copy-ready      → product_laterality_required
```

The product does not create a second diagnosis, validation or safety engine.

---

# 16. Manual edit boundary

Default mode:

```text
auto_live
```

An explicit `Επεξεργασία` action may create an ephemeral text snapshot.

```text
manual text != structured state
manual edits do not reverse-write selections
structured selections are not inferred from prose
```

A dirty manual buffer must not be silently overwritten by subsequent live projection. Exact edit-mode interaction remains Step-5 prototype UX work.

---

# 17. Deterministic test boundary

Primary fixtures cover routine behavior including:

```text
default referral
pain/stiffness/generic weakness
specific quadriceps weakness
ROM combination
balance/neuromuscular context
walking de-duplication
mixed-guideline adjunct without evidence leakage
core omission
suggestion without selection
bilateral grammar
objective-vs-generic weakness precedence
implicit-core fallback
```

Edge fixtures additionally prove:

```text
power-user finding/function preservation
plural non-redundant goals
explicit restriction + clinician note preservation
unsupported finding fails closed
true locking fails closed pending safety mapping
missing clinician diagnosis assertion blocks Copy
non-copy-ready laterality blocks Copy
```

---

# 18. Hard invariants

```text
DIAGNOSIS MUST BE CLINICIAN-ASSERTED BEFORE COPY
EVERY SELECTED PRODUCT ITEM MUST RENDER, BLOCK, OR BE EXPLICITLY SEMANTICALLY DE-DUPLICATED
TRUE LOCKING NOT PRODUCT-EXPOSED WITHOUT SAFETY MAPPING
SUGGESTION != SELECTION != OUTPUT
STIFFNESS != ROM RESTRICTION
GENERIC WEAKNESS != OBJECTIVE WEAKNESS
FUNCTIONAL LIMITATION != AUTOMATIC REHAB SELECTION
EVIDENCE CUE != REFERRAL PROSE
ADJUNCT != CORE ACTIVE REHABILITATION
OUTPUT PROJECTION != STATE MUTATION
MANUAL EDIT != STRUCTURED WRITEBACK
MISSING != NEGATIVE
NO machine-ID leak
NO exact exercise dose invention
NO LLM required for routine text
```

---

# 19. Step-3 non-goals

```text
runtime implementation
visual prototype implementation
second diagnosis
billing/account/persistence
patient identifiers
AI prose generation
autonomous evidence updating
```

---

# 20. Exact freeze gate

Step 3 freezes only after:

```text
human + machine contract agreement
+ primary and edge deterministic fixtures
+ validator against existing CU-1 + frozen Step-2
+ exact full-diff design review
+ clean exact-head machine gate
```
