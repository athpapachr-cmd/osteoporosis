# PHYSIO_REFERRAL_PRODUCT_CONTEXT_CURRENT.md

> **ROLE:** durable product-context authority for the physiotherapy-referral product track.
> **ROOT CANONICALS STILL GOVERN:** always bootstrap `AGENTS.md`, `TODO.md`, `CLINICAL_EXCELLENCE_PLAN.md`, `SLICE_PLAN_CURRENT.md`, `CURRENT_OPERATIONAL.md`, `osteoporosis-change-log.md` first.
> **CURRENT ACTIVE DIAGNOSIS VERTICAL:** Knee Osteoarthritis only.
> **CURRENT PRODUCT STATE:** synthetic prototype, technically gated, not merged/deployed/real-patient validated.
> **LATEST TESTED SUBSTANTIVE V3 HEAD:** `9deafa2db43d3498c5becf20f77a804b03849d53`.
> **LATEST V3 BRANCH:** `feat/physio-knee-oa-more-redesign-v3-2026-09-12`.

---

# 1. Product thesis

Build a clinician-facing paid referral product that turns a few meaningful clinical selections into a concise, clinically credible, evidence-aware physiotherapy referral without forcing the clinician through a traditional medical form.

Initial commercial hypothesis is approximately `€9.99/month` or similar low-friction subscription pricing. This remains a hypothesis, not validated willingness-to-pay.

The first vertical is intentionally **Knee Osteoarthritis**. Do not add a second diagnosis merely to make the product look larger. Knee-OA must first prove the product interaction model, receiver usefulness and commercial value.

Core product promise:

```text
fast clinician input
+
clinically meaningful handoff
+
evidence complexity underneath
+
calm/simple surface above
```

---

# 2. Product philosophy / interaction spirit

The intended design philosophy is closer to Steve Jobs-era product reduction than to a comprehensive clinical form:

> **Do not remove capability merely to look simple. Remove unnecessary decisions from the user until the moment they matter.**

Operational translation:

```text
power underneath
!= all controls visible at once

clinically possible
!= worth showing now

more data
!= better handoff

feature completeness
!= product quality
```

Preferred experience:

- modern, calm, native/iPhone-like visual language;
- direct manipulation rather than long form completion;
- typical routine referral in roughly 5–7 meaningful taps;
- live referral updates continuously;
- no routine `Generate` button;
- primary action is `Αντιγραφή`;
- `✎ Επεξεργασία` is directly visible, not hidden behind overflow;
- complexity appears through progressive disclosure;
- one interaction accent, restrained evidence/safety semantics;
- generous whitespace and visual hierarchy rather than card proliferation;
- mobile-first behavior must remain coherent, with desktop benefiting from the same hierarchy.

A recurring anti-pattern to reject:

```text
"this option might occasionally be useful"
→ therefore keep it permanently visible
```

That reasoning is how a simple product becomes an aircraft cockpit made of checkboxes.

---

# 3. General clinical-utility gate

Permanent rule for this product track:

```text
clinically interesting
!= workflow-useful
!= receiver-useful
!= worth adding
```

Before adding any field, qualifier, intervention, alert, evidence cue, output sentence, country rule or Product Owner/reviewer suggestion, ask:

1. What downstream management/safety/handoff/workflow decision does it change?
2. Who is the consumer of the information?
3. Is the referring clinician expected to know/measure it accurately at referral time?
4. Will the receiving physiotherapist simply repeat it anyway?
5. Does the incremental value justify another tap, visual element and maintenance burden?
6. Is there evidence of utility, or only plausible clinical logic?
7. If uncertain, can it be tested before becoming a permanent field?

Hard governance invariant:

```text
PRODUCT OWNER REQUEST
!= CLINICAL EVIDENCE
!= RECEIVER VALUE
!= IMPLEMENTATION AUTHORITY
```

The assistant/reviewer is expected to challenge, verify or defer Product Owner ideas when appropriate rather than follow them blindly.

---

# 4. Core semantic invariants

Never collapse these distinctions:

```text
symptom != objective finding != diagnosis
suggestion != clinician selection != referral output
evidence state != selection state
availability/provenance != evidence direction
safety state != evidence state
manual text != structured state
favorite != clinical selection
contextual shortcut != clinical inference
```

Examples:

- pes-anserine pain/tenderness does **not** auto-diagnose pes-anserine bursitis;
- generic weakness does **not** become objective weakness;
- quadriceps weakness in output requires explicit examination semantics;
- morning stiffness >30 minutes is a review clue, not an alternate diagnosis or treatment selector;
- FFD/passive extension deficit is distinct from subjective stiffness and active extension lag;
- a suggestion may become visible but never becomes selected until the clinician explicitly adds it.

Missing information must not be converted into a negative finding.

---

# 5. Current Knee-OA clinical capture

## Routine surface

- diagnosis: explicit `Οστεοαρθρίτιδα γόνατος` selection acts as clinician assertion;
- laterality: `Δεξί | Αριστερό | Άμφω`;
- core clinical picture:
  - `Πόνος`
  - `Δυσκαμψία`
  - `Αδυναμία`
  - `Λειτουργικότητα`

Diagnosis and laterality are required before export. Missing state is specific and local, not a generic red warning wall.

## Progressive symptom qualifiers

### Pain

Optional locations:

- Έσω μεσάρθρια
- Έξω μεσάρθρια
- Πρόσθιος / περιεπιγονατιδικός
- Χήνειος πόδας
- Οπίσθιος
- Διάχυτος

Focal locations may coexist; diffuse is exclusive.

### Stiffness

- Πρωινή
- Μετά από ακινησία
- if morning: `≤30′` / `>30′`

`>30′` is non-blocking review information only.

### Weakness

- generic weakness remains patient-reported/contextual unless refined;
- `Αντικειμενική στην εξέταση` is explicit examination meaning;
- `Τετρακέφαλος στην εξέταση` is explicit quadriceps examination meaning;
- visible atrophy can be recorded with bounded location;
- no MRC/dynamometry workflow was added.

## Advanced examination

Only if actually examined/measured:

- extension lag;
- effusion;
- passive extension deficit;
- focal tenderness (medial/lateral joint line or pes-anserine region);
- other existing supported findings remain reachable through the advanced category.

### Passive extension deficit / FFD

It remains advanced and optional. If a degree is entered it must represent a positive deficit; `0° FFD` is rejected. Referral wording uses `παθητικό έλλειμμα έκτασης`, not permanence language and not unnecessary English terminology.

Do **not** promote routine FFD measurement unless receiver testing proves it changes the handoff enough to justify routine capture.

## Functional baseline / main goal

A dedicated structured main-activity/baseline field was deliberately **not added**. Rehabilitation literature supports goals/baseline during physiotherapy assessment, but evidence is insufficient that the referring doctor should be forced to capture another structured field.

Current physician-side functional categories/free text remain until receiving-physiotherapist validation demonstrates incremental referral value.

---

# 6. Referral semantics

The referral is deterministic from clinician-selected state plus the bounded phenotype overlay. No LLM generates the routine clinical text.

Principles:

- low-information referral → proportionally short output;
- richer patient-specific information → richer output;
- treatment wording is framed as physiotherapy assessment plus **indicative priorities**, not physician-prescribed dose/technique/progression;
- no invented repetitions/sets/frequency;
- selected structured clinical information is not silently discarded;
- evidence metadata never gets copied into the referral itself.

Manual editing:

```text
structured referral
→ clinician chooses ✎ Επεξεργασία
→ clinician-owned manual buffer

later structured change
→ manual text is preserved
→ export becomes stale/blocked
→ clinician explicitly chooses own text or regenerated text
```

No reverse parsing and no silent merge of manual prose back into structured state.

---

# 7. Evidence model

Evidence design is deliberately separate from selection state.

Six evidence states:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

Greek mixed label:

`Οι οδηγίες διαφέρουν`

Hard evidence distinctions:

```text
insufficient != ineffective
conflict != consensus
source year != reviewed_on
broad recommendation != narrow strong recommendation
missing context != negative context
suggestion != selection
```

Current source set includes VA/DoD OA 2026, Singapore ACE 2026, EULAR recommendations, NICE NG226, AAOS OAK3 and ACR/AF.

Routine supported/conditional detail uses progressive disclosure. **Mixed guidance is an exception:** opposing material source positions remain immediately visible. Simplicity may not be achieved by hiding inconvenient guideline disagreement.

A distinct source-to-claim audit maps product claim → exact source → scope → strength → locator → reviewed date. It is an evidence-integrity audit, not a fifth product reviewer.

---

# 8. Suggestions behavior

Suggestions are deterministic and bounded.

- first/current primary suggestion is directly actionable;
- if more candidates exist, routine surface shows a compact bordered summary:
  - `Άλλες {n} προτάσεις ›`
  - short titles only;
- opening it exposes the full suggestion sheet;
- summary does not select anything;
- each suggestion keeps explicit Add, evidence access and dismissal;
- stale suggestion candidates fail closed;
- core defaults remain reviewed defaults, not magical automatic treatment decisions.

Do not turn the main screen into a recommendation feed.

---

# 9. `Περισσότερα` visual/IA contract — v3

The Product Owner identified v2 `Περισσότερα` as overpopulated even though Favorites were useful. The accepted solution is not deletion of capability but **scan-first progressive disclosure**.

Current v3 structure:

```text
★ Συχνά
Σχετικά τώρα
Όλα
```

## `★ Συχνά`

- personal shortcuts only;
- bounded to six in the synthetic prototype;
- favorite/pin never selects a clinical item;
- normal use does not show stars on every row;
- `Προσαρμογή Συχνών` deliberately reveals pin controls;
- no permanent Hide;
- synthetic favorites are ephemeral and cleared on reset/page lifecycle;
- future persistence, if justified, belongs to clinician/account preference state, never patient/referral state.

## `Σχετικά τώρα`

- maximum small set (currently at most three);
- deterministic from already-declared structured state;
- only surfaces existing fields that may now be relevant;
- never invents finding/diagnosis/treatment;
- never selects anything merely by appearing;
- examples include offering an examination refinement after generic weakness, or focal tenderness after focal pain, only as a shortcut.

This is progressive disclosure, not an AI guess layer.

## `Όλα`

Six scan-friendly category rows:

1. `Εξέταση`
2. `Λειτουργία & στόχοι`
3. `Αποκατάσταση`
4. `Συμπληρωματικά`
5. `Περιορισμοί & σημείωση`
6. `Κλινικός έλεγχος`

Each row:

- has restrained iconography;
- shows selected summary/count without opening when relevant;
- opens one category into the existing sheet/modal host;
- keeps full underlying capability reachable.

Visual grammar:

- whitespace over dense borders;
- section typography creates hierarchy;
- large quiet rows instead of chip walls;
- muted secondary summaries;
- no badge confetti;
- clear target sizes;
- full detail only after deliberate navigation.

Search is intentionally absent. If a Knee-OA referral needs a search field, the information architecture has already failed.

---

# 10. Safety/privacy/product boundaries

Prototype is synthetic and loopback-only.

Current invariants:

- no real patient data;
- no patient draft persistence;
- no analytics;
- no production credentials;
- no public preview;
- no autonomous evidence updating;
- no production FastAPI registration;
- no `localStorage`/`sessionStorage` for clinical draft or synthetic Favorites;
- inherited unresolved/urgent safety can block export;
- safety-critical behavior must not depend on the clinician browsing `Περισσότερα`.

Real-patient production integration requires a separate privacy/auth/hosting design and authorization.

---

# 11. Jurisdiction strategy

Architecture:

```text
international clinical-evidence core
+
optional jurisdiction/local-system overlay
```

Current context seam:

```text
id: CY_GESY
label: Κύπρος · ΓεΣΥ
```

At present this is **context only**. It does not change evidence states, defaults, suggestions or referral prose.

Cyprus/HIO/GeSY guidance must be audited recommendation-by-recommendation before it can affect product behavior. Local reimbursement/resource/feasibility policy must not be represented as stronger clinical-efficacy evidence.

The initial target audience is GeSY clinicians, so relevant GeSY guidance should ultimately be visible when authoritative and useful.

The architecture must not trap the product in Cyprus. Future `GR` or `UK_ENGLAND` profiles remain dormant until actual workflow/market need is proven. No country selector or location inference is needed simply because the architecture supports profiles.

Country/local profile should eventually come from explicit configuration/account preference.

---

# 12. Independent-review architecture

Four **separate** independent review tracks are required for major candidate review:

1. Clinical / Evidence
2. Physiotherapy / receiving-professional utility
3. UX / Product
4. Commercial / Product-Market

Rules:

- same pinned candidate for all four;
- each gets its own prompt/packet;
- reviewers do not see one another's conclusions before completing their own review;
- no implementation during review;
- every reviewer must identify what should be removed/simplified, not only what should be added;
- useful forcing question: `If forced to simplify by 20%, what would you remove first and why?`
- after all four: cross-review synthesis, not a fifth review;
- findings are dispositioned by Product Owner before implementation.

The first four Knee-OA reviews all returned **Conditional Pass** with no reported blockers at the synthetic stage. Their findings led to the post-review clinical/semantic correction cycle, v2 usability changes and v3 `Περισσότερα` redesign.

Technical PASS never substitutes for clinical/physio/UX/commercial validation.

---

# 13. Important lineage

Key design/implementation lineage:

```text
Evidence design frozen/closed
  ab4b349223cd4c461837ab3125967a06d169a7e1

Dynamic referral design frozen/closed
  8489e2ee32f7aeae6f678c7db2838930c0759eb4

Interaction/traceability frozen/closed
  4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6

Prototype v1 closeout
  c3f79a192a3fcac9fb11a5245df4e93312c4522a

Step-6A qualifier refinement closeout
  6539351c592c1dc3e49931057b63925dea3cb94d

Post-review semantic/UX amendment closeout
  0f38f4d411146667d854c32c9f5f639f344d7c7f

Usability v2 closeout
  2c19eb7283c96af7868690a892d7e61155674d6b

More redesign v3 tested substantive
  9deafa2db43d3498c5becf20f77a804b03849d53
```

Historical rich-referral branch remains parked and must not be conflated with the active product vertical:

`feat/cu1-rich-referral-global-evidence-2026-08-29`

---

# 14. Latest v3 technical evidence

Successful run:

```text
workflow                                  Physio Knee OA prototype gate
run                                       34677022119
head                                      9deafa2db43d3498c5becf20f77a804b03849d53
scope + syntax                            PASS
real CU-1 / HTTP                          15 / 15 PASS
frozen Step-3 exact-output fixtures       15 PASS
post-review clinical/output               11 / 11 PASS
inherited Chromium                        12 / 12 PASS
post-review qualifier Chromium             9 / 9 PASS
usability-v2 Chromium                      5 / 5 PASS
More-v3 Chromium                           5 / 5 PASS
Greek source-summary coverage             54 positions
packaged dependency closure               PASS
```

Artifact:

```text
id      10292822525
digest  sha256:1cc6156d5de37e2ef1fcc15de68e6df7811535e10dfec86c20ed1723ba81fb0e
```

Screenshots include desktop/mobile, evidence conflict, suggestions, v2 Favorites and v3 overview/Favorites/mobile states.

Known still-unproven items:

```text
actual iPhone Safari / VoiceOver
measured full accessibility/contrast acceptance
receiving-physiotherapist real-user utility
Cyprus/GeSy recommendation-by-recommendation fidelity
real-patient privacy/auth/hosting
paid conversion / retention
Greece/England market need
```

---

# 15. What not to do next by default

Do not infer authorization for:

```text
second diagnosis
new functional-baseline field
routine FFD measurement
permanent Hide
cross-device favorite persistence
search/tabs added to More
unaudited GeSY clinical semantics
UK/GR localized content
analytics
billing/auth
patient persistence
production integration
PR / merge / deploy
```

Do not redesign after every isolated comment unless there is a real usability/safety/data-integrity reason. Avoid refinement for its own sake.

---

# 16. New-conversation bootstrap for this product

A fresh ChatGPT/Codex conversation that will work on this physiotherapy product must:

1. fresh-fetch `main` and read all six root canonicals in required order;
2. read this file completely;
3. read `UX_CONTRACT_CURRENT.md`;
4. read the current `SLICE_PLAN_CURRENT.md` and `CURRENT_OPERATIONAL.md` again if the product track is active;
5. inspect the exact current branch/head before mutation;
6. respect the one-writer lock;
7. distinguish `DESIGNED / IMPLEMENTED / TESTED / MERGED / DEPLOYED / PILOT-VALIDATED`;
8. never reconstruct product truth from chat memory alone.

Useful product documents for deeper work:

- `KNEE_OA_EVIDENCE_DESIGN_V1.md`
- `KNEE_OA_STEP6A_QUALIFIER_REFINEMENT_RESULT.md`
- `KNEE_OA_POST_REVIEW_AMENDMENT_RESULT_V1.md`
- `KNEE_OA_USABILITY_REFINE_V2_RESULT.md`
- `KNEE_OA_MORE_REDESIGN_V3_RESULT.md`
- evidence/template/interaction YAML contracts
- independent-review prompts/packets and cross-review synthesis artifacts

If this context file conflicts with a newer root `CURRENT_OPERATIONAL.md` or `SLICE_PLAN_CURRENT.md`, the newer canonical operational state wins and this file must be reconciled before further mutation.

---

# 17. Exact next product action at this handoff

**Product Owner visually/use-tests the exact tested More-v3 synthetic artifact and gives concrete keep/change/remove feedback.**

After Product Owner acceptance, the next decision is deliberate: either close/freeze Knee-OA for real receiver/user validation, or authorize a bounded remaining correction. Do not automatically add a second diagnosis or release to production.
