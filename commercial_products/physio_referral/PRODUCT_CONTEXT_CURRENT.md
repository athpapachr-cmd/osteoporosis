# PHYSIO_REFERRAL_PRODUCT_CONTEXT_CURRENT.md

> **ROLE:** durable product-context authority for the physiotherapy-referral product track.
> **ROOT CANONICALS STILL GOVERN:** always bootstrap `AGENTS.md`, `TODO.md`, `CLINICAL_EXCELLENCE_PLAN.md`, `SLICE_PLAN_CURRENT.md`, `CURRENT_OPERATIONAL.md`, `osteoporosis-change-log.md` first.
> **CURRENT ACTIVE DIAGNOSIS VERTICAL:** Knee Osteoarthritis only.
> **CURRENT PRODUCT STATE:** Knee-OA V5.1 plus bounded post-use receiver refinements released / live / authenticated-smoke-verified; `CY_GESY` active.
> **CURRENT MAIN RELEASE COMMIT:** `b962485c741f558121e8daabfcf1d20c84f31f63`.
> **LATEST AUTHENTICATED PRODUCTION SMOKE:** `35019200920` — SUCCESS.
> **PRECEDING POST-USE SMOKE:** `34957778592` — SUCCESS on the PR #107 release state.
> **CURRENT WRITER:** none.

---

# 1. Product thesis

Build a clinician-facing paid referral product that turns a few meaningful clinical selections into a concise, clinically credible, evidence-aware physiotherapy referral without forcing the clinician through a traditional medical form.

Initial commercial hypothesis remains approximately `€9.99/month` or similar low-friction pricing. This remains a hypothesis, not validated willingness-to-pay.

The first vertical remains **Knee Osteoarthritis**.

Core product promise:

```text
fast clinician input
+
clinically meaningful handoff
+
international evidence complexity underneath
+
reviewed jurisdiction context where useful
+
calm/simple surface above
```

A second diagnosis is not automatically authorized. It requires a fresh bounded product decision.

---

# 2. Product philosophy / interaction spirit

The intended design philosophy is reduction, not superficial minimalism:

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
- few meaningful routine taps;
- live deterministic referral updates;
- no routine `Generate` button;
- primary action `Αντιγραφή`;
- `✎ Επεξεργασία` directly reachable;
- progressive disclosure for detail/evidence;
- one restrained interaction grammar rather than badge/checkbox proliferation;
- mobile-first behavior with coherent desktop hierarchy.

Anti-pattern:

```text
"this option might occasionally be useful"
→ therefore keep it permanently visible
```

That logic is explicitly rejected.

---

# 3. Permanent clinical/product utility gate

Permanent rule:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

Before adding any field, qualifier, intervention, alert, evidence cue, output sentence, jurisdiction rule or reviewer suggestion, ask:

1. What downstream management/safety/handoff/workflow decision does it change?
2. Who consumes the information?
3. Is the referring clinician expected to know/measure it accurately at referral time?
4. Will the receiving professional simply repeat it anyway?
5. Does the value justify another tap, visual element and maintenance burden?
6. Is there evidence of utility or only plausible clinical logic?
7. Can it remain progressive/on-demand instead of routine?

Hard governance invariant:

```text
PRODUCT OWNER REQUEST
!= CLINICAL EVIDENCE
!= RECEIVER VALUE
!= IMPLEMENTATION AUTHORITY

EXTERNAL FEEDBACK
!= AUTOMATIC IMPLEMENTATION AUTHORITY
```

The assistant/reviewer is expected to challenge, verify or defer ideas when appropriate rather than implement them mechanically.

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
international evidence != jurisdiction/local policy
clinical guidance != reimbursement/admin rule
planned != active
```

Examples:

- pes-anserine pain/tenderness does not auto-diagnose pes-anserine bursitis;
- generic weakness does not become objective weakness;
- quadriceps weakness in output requires explicit examination semantics;
- quadriceps atrophy remains an objective examination finding, not a weakness subtype;
- morning stiffness >30 minutes is a review clue, not an alternate diagnosis or treatment selector;
- FFD/passive extension deficit is distinct from subjective stiffness and active extension lag;
- a suggestion may become visible but never becomes selected until the clinician explicitly adds it;
- a Cyprus/GeSY local position never silently changes the international evidence state.

Missing information must not be converted into a negative finding.

---

# 5. Current Knee-OA clinical capture and released interaction

## Routine surface

- explicit `Οστεοαρθρίτιδα γόνατος` clinician assertion;
- laterality: `Δεξί | Αριστερό | Άμφω`;
- core clinical picture: `Πόνος`, `Δυσκαμψία`, `Αδυναμία`, `Λειτουργικότητα`.

Diagnosis/laterality requirements and export readiness remain deterministic and fail closed.

## Released V5 / V5.1 interaction

```text
inactive Pain / Stiffness / Weakness first tap
→ select generic symptom
→ no forced popup

second tap while selected
→ optional focused refinement
```

`Λειτουργικότητα` remains a first-tap chooser. V5.1 adds directional knee weakness, progressive ROM detail, crepitus, expanded tenderness localization and objective stability findings while preserving symptom != objective finding != diagnosis.

The later post-use clinical-review layer treats recent trauma, rapid worsening/deformity and hot/swollen joint as non-blocking review observations. They do not themselves infer fracture, septic arthritis, SIFK/SONK, automatic imaging or a second diagnosis. Separate explicit unresolved safety concerns remain blocking.

## Receiver-side refinement

`Περισσότερα → Περιορισμοί & σημείωση` contains optional `Χρονιότητα συμπτωμάτων` as a numeric duration `1..99` plus `εβδομάδες / μήνες / έτη`. It is context-only and may render, for example, `Συμπτωματολογία διάρκειας 8 μηνών.`

When `functional_task_retraining` is selected, the plan uses the compact receiver wording `λειτουργική επανεκπαίδευση με έμφαση στις καταγεγραμμένες λειτουργικές δυσχέρειες` instead of repeating the already-recorded task list.

No structured previous-physiotherapy/response field exists. Permanent boundary:

```text
ADMINISTRATIVE PHYSIO ACTIVITY != KNOWN TREATMENT PROGRAM != KNOWN RESPONSE
```

No extra ADL / patient-centred boilerplate was added.

---

# 6. Referral semantics

The referral is deterministic from clinician-selected state plus bounded product context. No LLM generates routine clinical text.

Principles:

- low-information referral → proportionally short output;
- richer explicit clinical state → richer output;
- treatment wording is framed as physiotherapy assessment plus indicative priorities, not physician-prescribed dose/technique/progression;
- no invented repetitions/sets/frequency;
- selected structured information is not silently discarded;
- evidence/jurisdiction metadata never gets copied into referral prose merely because it exists.

Current released copy refinements:

- pain qualifier ownership removes redundant location prose such as pes-anserine + joint-line tails;
- clinical picture / functional impact and physiotherapy assessment / priorities use a paragraph boundary;
- `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` is replaced by connected human prose;
- low-information output remains compact and does not gain an artificial second paragraph.

Manual editing contract remains:

```text
structured referral
→ clinician chooses ✎ Επεξεργασία
→ clinician-owned manual buffer

later structured change
→ manual text preserved
→ export becomes stale/blocked
→ clinician explicitly chooses own text or regenerated text
```

No reverse parsing and no silent merge of manual prose into structured state.

---

# 7. Evidence model

Six international evidence states remain:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

Hard evidence distinctions:

```text
insufficient != ineffective
conflict != consensus
source year != reviewed_on
broad recommendation != narrow strong recommendation
missing context != negative context
suggestion != selection
```

Current international source set remains represented through the frozen evidence contract. Mixed guidance remains visible as disagreement rather than being averaged into false consensus.

The released V5/V5.1 and post-use refinements do not reclassify any evidence state or source position.

---

# 8. Jurisdiction strategy — released and preserved through V5.1/post-use refinements

Architecture:

```text
international clinical-evidence core
+
JurisdictionOverlayV1
+
explicit production profile CY_GESY
```

Production profile:

```text
id: CY_GESY
label: Κύπρος · ΓεΣΥ
selection_source: explicit_account_configuration
```

The overlay is released and active in production.

Released/current behavior:

- international evidence state remains unchanged;
- local agreement remains silent on the routine surface;
- relevant local difference may appear through restrained progressive disclosure;
- local-only interventions do not automatically become product controls;
- GeSY administrative/reimbursement/system-lifecycle rows remain separate from clinical evidence;
- planned GeSY IT integration is not treated as active by inference;
- no patient-location/IP/browser inference chooses jurisdiction;
- no Cyprus/GeSY wording is injected into routine referral prose merely because the overlay is active.

Production activation remains:

`PHYSIO_REFERRAL_JURISDICTION_PROFILE=CY_GESY`

Current protected Cockpit/jurisdiction semantics were preserved through PR #106, PR #107 and PR #109; authenticated live smokes `34957778592` and `35019200920` both passed with explicit `CY_GESY`.

Future `GR` or `UK_ENGLAND` profiles remain dormant until separately justified and authorized.

---

# 9. Suggestions / More / progressive disclosure

Suggestions remain deterministic, bounded and non-selecting until the clinician explicitly acts.

The `Περισσότερα` philosophy remains scan-first progressive disclosure rather than a permanent chip/checkbox wall.

Favorites/pins, contextual shortcuts and advanced categories are navigation conveniences, not clinical inference.

Quadriceps atrophy now illustrates this rule directly: capability is preserved under `Περισσότερα → Εξέταση` without duplicating it inside the compact weakness refinement.

Search remains intentionally absent unless future evidence proves the information architecture genuinely needs it.

---

# 10. Safety/privacy/product boundaries

Current invariants:

- no patient-draft persistence;
- no analytics in this product slice;
- no autonomous evidence updating;
- production routes remain behind the existing Clinical Excellence authentication boundary;
- no clinical draft state in `localStorage`/`sessionStorage`;
- inherited unresolved/urgent safety can block export;
- safety-critical behavior cannot depend on browsing `Περισσότερα` or evidence detail;
- authenticated release smokes use only generated UUID + non-identifiable state and never print protected keys.

The released V5/V5.1/post-use workflow preserves these boundaries. Any future persistence, analytics, billing or entitlement work requires separate authority.

---

# 11. External feedback / receiver-validation policy

Formal physiotherapist/receiver evaluation remains optional external evidence and is **not a blocking gate** for the current released Knee-OA workflow.

The Product Owner may ask clinician colleagues later, at a time of their choosing.

```text
physiotherapist / receiver feedback
= optional later external evidence
!= current release gate
!= mandatory next action
!= automatic implementation authority
```

This is a sequencing decision, not a claim that receiver utility has been scientifically proven.

Do not describe the product as receiver-validated unless actual external evidence is later collected and recorded.

---

# 12. Review architecture

For major product changes, useful review lenses remain:

1. Clinical / Evidence
2. Receiving-professional / workflow utility
3. UX / Product
4. Commercial / Product-Market

These are review lenses, not mandatory bureaucratic gates for every small bounded refinement.

Independent V5 review concluded:

```text
ACCEPT WITH REQUIRED CHANGES
PATCH V5 THEN MERGE
```

The required historical change was fresh-main integration, not clinical redesign. That blocker was resolved and V5 subsequently released; V5.1 and the bounded post-use refinements followed under their own tested release cycles.

Technical PASS never proves product value, but lack of optional receiver feedback does not prohibit a Product Owner-authorized bounded release.

---

# 13. Current released state

Current release chain:

```text
V5 baseline release runtime  8cfb22fd2478e7832b9b8642f7ae5241d7e1a267
V5.1 merge                   8064999ea70e0a90f6073fc6d66a0c8caaba1538  (PR #106)
post-use clinical review     a19f4b9d52c3076f715fd864a5f7664d2b140c81  (PR #107)
receiver/chronicity          b962485c741f558121e8daabfcf1d20c84f31f63  (PR #109)
```

Authenticated production evidence:

```text
34957778592 — SUCCESS — PR #107 post-use live state
35019200920 — SUCCESS — current receiver/chronicity live state
```

The current release preserves `CY_GESY`, shared protected Cockpit semantics, evidence/safety/default-selection boundaries and no-persistence behavior.

Current next bounded action is not another automatic feature. Use real-use evidence to decide what, if anything, deserves the next slice.

---

# 14. Validation truth still unproven

The current bounded live behavior is proven by authenticated production smoke. Do not overclaim beyond that:

```text
current protected Cockpit bootstrap / project path   proven
current chronicity + receiver compression behavior   proven live
CY_GESY current production profile                    proven live
real clinical pilot                                   not proven
paid conversion / willingness-to-pay                  not proven
retention                                             not proven
formal external receiver validation                   not proven and currently deferred
actual iPhone Safari / VoiceOver                       not proven unless separately recorded
second-diagnosis usefulness                            not proven
Greece/England market need                             not proven
```

---

# 15. What not to do next by default

Do not infer authorization for:

```text
second diagnosis
new mandatory functional-baseline field
routine FFD measurement
permanent Hide
cross-device favorite persistence
search/tabs added to More
new local intervention selectors solely because Cyprus mentions them
UK/GR localized content
analytics
billing/auth expansion
patient persistence
```

Do not redesign after every isolated comment unless there is a real usability, safety, evidence-integrity or workflow reason.

---

# 16. New-conversation bootstrap for this product

A fresh ChatGPT/Codex conversation that will work on this product must:

1. fresh-fetch `main` and read all six root canonicals in required order;
2. read this file completely;
3. read `CURRENT.md`, `PRODUCT_PLAN.md` and `UX_CONTRACT_CURRENT.md`;
4. inspect the exact current branch/head before mutation;
5. respect the one-writer lock;
6. distinguish `DESIGNED / IMPLEMENTED / TESTED / PR-REVIEWED / MERGED / DEPLOYED / PILOT-VALIDATED / COMMERCIALLY-VALIDATED`;
7. never reconstruct product truth from chat memory alone.

If this context file conflicts with newer root canonicals, the newer root operational state wins and this file must be reconciled before further product mutation.

---

# 17. Exact handoff state

```text
Knee-OA production foundation          released
CY_GESY overlay                         released / active / smoke-verified
V5                                      released / smoke-verified
V5.1                                    released
PR #107 post-use refinement             released / live smoke 34957778592 PASS
PR #109 receiver/chronicity refinement  released / live smoke 35019200920 PASS
formal receiver review                  deferred by Product Owner; not a gate
real clinical pilot                     not started
commercial validation                   not started
second diagnosis                        not authorized
active writer                           none
```

**Exact next action:** do not invent another implementation slice. Use Product Owner real-use evidence to decide whether another bounded Knee-OA refinement, pilot/commercial-validation step or separately authorized next diagnosis is justified.
