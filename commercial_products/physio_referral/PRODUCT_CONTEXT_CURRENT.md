# PHYSIO_REFERRAL_PRODUCT_CONTEXT_CURRENT.md

> **ROLE:** durable product-context authority for the physiotherapy-referral product track.
> **ROOT CANONICALS STILL GOVERN:** always bootstrap `AGENTS.md`, `TODO.md`, `CLINICAL_EXCELLENCE_PLAN.md`, `SLICE_PLAN_CURRENT.md`, `CURRENT_OPERATIONAL.md`, `osteoporosis-change-log.md` first.
> **CURRENT ACTIVE DIAGNOSIS VERTICAL:** Knee Osteoarthritis only.
> **CURRENT PRODUCT STATE:** Knee-OA released; `CY_GESY` jurisdiction overlay released and production-smoke-verified; V5 fresh-main integration implemented/tested; Product Owner release HOLD.
> **RELEASED JURISDICTION RUNTIME:** `e52a4851b504476c1e361575d08664c05467ff53`.
> **AUTHENTICATED CY_GESY PRODUCTION SMOKE EVIDENCE:** `34703453615` — SUCCESS.
> **CURRENT INTEGRATED V5 CANDIDATE:** `9a7f360745710deacb0ff82f03249723bdfe87d6`; gate `34736389860` — SUCCESS.
> **ORIGINAL REVIEWED V5:** `a47357c602120d3678e8f2f23b99775e616c79e1`; gate `34693751545` — SUCCESS.
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

# 5. Current Knee-OA clinical capture and V5 candidate interaction

## Routine surface

- explicit `Οστεοαρθρίτιδα γόνατος` clinician assertion;
- laterality: `Δεξί | Αριστερό | Άμφω`;
- core clinical picture:
  - `Πόνος`
  - `Δυσκαμψία`
  - `Αδυναμία`
  - `Λειτουργικότητα`

Diagnosis/laterality requirements and export readiness remain deterministic and fail closed.

## V5 first-tap / second-tap interaction

In the fresh-main integrated V5 candidate:

```text
inactive Pain / Stiffness / Weakness first tap
→ select generic symptom
→ no forced popup

second tap while selected
→ optional focused refinement
```

`Λειτουργικότητα` remains a first-tap chooser because an unqualified generic function state is not sufficiently meaningful.

## Progressive qualifiers

Pain detail may include medial/lateral joint line, anterior/peripatellar, pes-anserine, posterior or diffuse distribution. Focal locations may coexist; diffuse remains exclusive where defined by the contract.

Stiffness may distinguish morning stiffness, post-immobility stiffness and the reviewed `≤30′` / `>30′` morning-duration context.

Weakness semantics preserve patient/context weakness versus explicit examination findings.

The V5 weakness second-tap contains only:

```text
Μυϊκή αδυναμία στην εξέταση
Αδυναμία τετρακεφάλου στην εξέταση
```

The Product Owner accepted removal of duplicated `Ατροφία τετρακεφάλου` from this sheet. Atrophy remains available under:

```text
Περισσότερα → Εξέταση → Ατροφία τετρακεφάλου
```

The weakness badge/count represents weakness refinement only and does not count separately selected atrophy.

## Advanced examination

Advanced findings remain optional and only if actually examined/measured, including extension lag, effusion, passive extension deficit, focal tenderness and quadriceps atrophy.

Do not promote routine FFD measurement merely because it is clinically possible.

## Functional capture

A dedicated mandatory physician-side structured baseline/main-activity field is still not required. Current functional categories/free text are sufficient unless later workflow evidence justifies another mandatory decision.

External physiotherapist feedback is not required to allow current product progression.

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

V5 candidate copy refinements:

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

V5 does not reclassify any evidence state or source position.

---

# 8. Jurisdiction strategy — released and preserved in V5 integration

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

Final V5 integration gate `34736389860` explicitly tested V5 and the current jurisdiction behavior together on exact head `9a7f3607...`.

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

The V5 integrated candidate preserves these boundaries. Any future persistence, analytics, billing or entitlement work requires separate authority.

---

# 11. External feedback / receiver-validation policy

Formal physiotherapist/receiver evaluation is **not needed now** and is **not a blocking gate** for V5.

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

The required change was fresh-main integration, not clinical redesign. That blocker is resolved at the candidate/test level on `9a7f3607...`.

Technical PASS never proves product value, but lack of optional receiver feedback does not prohibit a Product Owner-authorized bounded release.

---

# 13. V5 candidate — current state

Historical reviewed/tested candidate:

```text
head    a47357c602120d3678e8f2f23b99775e616c79e1
gate    34693751545 — SUCCESS
```

Fresh-main integrated candidate:

```text
branch  feat/physio-knee-oa-v5-integration-2026-09-13
base    8aeb91ae37b83caaa188054128db98e04b638fd8
head    9a7f360745710deacb0ff82f03249723bdfe87d6
gate    34736389860 — SUCCESS
artifact 10311108002
```

The integration preserved `CY_GESY`, shared protected Cockpit semantics, evidence/safety/default-selection boundaries and no-persistence behavior.

The candidate is implemented/tested but **not merged/deployed**.

Current next bounded action:

```text
OPEN / REVIEW INTEGRATION PR UNDER RELEASE HOLD
```

If explicit release authority is later given:

```text
merge
→ Render deploy
→ authenticated V5 production smoke
→ Product Owner real-device/use acceptance
→ canonical release closeout
```

---

# 14. Validation truth still unproven

Do not overclaim:

```text
V5 live production behavior                  not proven until deploy/smoke
real clinical pilot                          not proven
paid conversion / willingness-to-pay         not proven
retention                                    not proven
formal external receiver validation          not proven and currently deferred
actual iPhone Safari / VoiceOver              not proven unless separately recorded
second-diagnosis usefulness                   not proven
Greece/England market need                    not proven
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
V5 fresh-main candidate                 integrated / exact-head tested / HOLD
V5 merged                               no
V5 deployed                             no
formal receiver review                  deferred by Product Owner; not a gate
real clinical pilot                     not started
commercial validation                   not started
second diagnosis                        not authorized
active writer                           none
```

**Exact next action: open/review the fresh-main V5 integration PR while preserving Product Owner release HOLD.**