# PHYSIO_REFERRAL_PRODUCT_CONTEXT_CURRENT.md

> **ROLE:** durable product-context authority for the physiotherapy-referral product track.
> **ROOT CANONICALS STILL GOVERN:** always bootstrap `AGENTS.md`, `TODO.md`, `CLINICAL_EXCELLENCE_PLAN.md`, `SLICE_PLAN_CURRENT.md`, `CURRENT_OPERATIONAL.md`, `osteoporosis-change-log.md` first.
> **CURRENT ACTIVE DIAGNOSIS VERTICAL:** Knee Osteoarthritis only.
> **CURRENT PRODUCT STATE:** Knee-OA V5 merged and deployed; `CY_GESY` overlay active; authenticated post-V5 live smoke queued / not yet executed.
> **V5 RELEASE RUNTIME:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **RENDER DEPLOY:** `dep-daj2398u01pc738ojvkg` — LIVE at exact V5 runtime SHA.
> **PR:** `#101` — squash-merged.
> **PR-HEAD V5 GATE:** `34736919952` — SUCCESS.
> **AUTHENTICATED V5 LIVE SMOKE:** `34737351354` — QUEUED / NOT YET EXECUTED at this reconciliation.
> **CURRENT WRITER:** none.

---

# 1. Product thesis

Build a clinician-facing paid referral product that turns a few meaningful clinical selections into a concise, clinically credible, evidence-aware physiotherapy referral without forcing the clinician through a traditional medical form.

Initial commercial hypothesis remains approximately `€9.99/month` or similar low-friction pricing. This remains a hypothesis, not validated willingness-to-pay.

The first and only active diagnosis vertical remains **Knee Osteoarthritis**.

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

A second diagnosis is not automatically authorized.

---

# 2. Product philosophy

The design principle is reduction, not superficial minimalism:

> **Do not remove capability merely to look simple. Remove unnecessary decisions from the user until the moment they matter.**

Operationally:

```text
power underneath != all controls visible at once
clinically possible != worth showing now
more data != better handoff
feature completeness != product quality
```

Preferred experience remains modern, calm, mobile-first, direct and deterministic, with progressive disclosure for detail/evidence and no routine Generate button.

Permanent utility rule:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

And:

```text
PRODUCT OWNER REQUEST
!= CLINICAL EVIDENCE
!= RECEIVER VALUE
!= IMPLEMENTATION AUTHORITY

EXTERNAL FEEDBACK
!= AUTOMATIC IMPLEMENTATION AUTHORITY
```

---

# 3. Core semantic invariants

Never collapse:

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

Missing information must not be converted into a negative finding.

---

# 4. Current Knee-OA V5 interaction

Routine surface:

- explicit `Οστεοαρθρίτιδα γόνατος` clinician assertion;
- laterality `Δεξί | Αριστερό | Άμφω`;
- `Πόνος`;
- `Δυσκαμψία`;
- `Αδυναμία`;
- `Λειτουργικότητα`.

V5 interaction:

```text
inactive Pain / Stiffness / Weakness first tap
→ select generic symptom
→ no forced popup

second tap while selected
→ optional focused refinement
```

`Λειτουργικότητα` remains a first-tap chooser.

Weakness second-tap contains only:

```text
Μυϊκή αδυναμία στην εξέταση
Αδυναμία τετρακεφάλου στην εξέταση
```

`Ατροφία τετρακεφάλου` is deliberately not duplicated there. It remains an objective examination finding under:

```text
Περισσότερα → Εξέταση → Ατροφία τετρακεφάλου
```

Weakness count represents weakness refinement only and does not count separately selected atrophy.

Pain, stiffness and function qualifiers remain optional and bounded by their existing semantic contracts. Bare ambiguous `Περιαρθρικά` remains absent from the visible routine/advanced UI.

---

# 5. Referral semantics

Referral output remains deterministic from clinician-selected state plus bounded product context. No LLM generates routine referral prose.

Principles:

- low-information referral → proportionally short output;
- richer explicit clinical state → richer output;
- no invented dose/frequency/protocol;
- selected structured information is not silently discarded;
- evidence/jurisdiction metadata does not leak into routine referral prose.

V5 copy refinements now deployed:

- pain qualifier ownership removes redundant location tails;
- richer referrals separate clinical picture/function from physiotherapy assessment/priorities with a paragraph boundary;
- `Επιπλέον στόχος:` / `Επιπλέον στόχοι:` becomes connected human prose;
- low-information output remains compact.

Manual editing remains clinician-owned and fail closed on stale reconciliation after later structured changes.

---

# 6. Evidence model

Six international evidence states remain unchanged:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

V5 does not reclassify any evidence state or source position.

Hard distinctions remain:

```text
insufficient != ineffective
conflict != consensus
source year != reviewed_on
broad recommendation != narrow strong recommendation
local policy != stronger international evidence
```

---

# 7. Jurisdiction strategy

Production architecture remains:

```text
international clinical-evidence core
+
JurisdictionOverlayV1
+
explicit production profile CY_GESY
+
V5 presentation/prose layer
```

Production profile:

```text
id: CY_GESY
label: Κύπρος · ΓεΣΥ
selection_source: explicit_account_configuration
```

The overlay remains active and separate:

- international evidence state unchanged;
- local agreement silent on routine surface;
- relevant local difference progressively disclosed;
- local-only interventions do not automatically become product controls;
- GeSY administrative/reimbursement/system-lifecycle rows remain separate from clinical evidence;
- planned GeSY IT integration remains planned unless separately verified active;
- no patient-location/IP/browser inference chooses jurisdiction;
- no Cyprus/GeSY wording is injected into routine referral prose by activation alone.

Production activation remains `PHYSIO_REFERRAL_JURISDICTION_PROFILE=CY_GESY`.

Future `GR` / `UK_ENGLAND` content remains dormant until separately justified and authorized.

---

# 8. Safety / privacy boundaries

Current invariants remain:

- no patient/referral draft persistence in this product;
- no clinical draft state in `localStorage` / `sessionStorage`;
- no product analytics added by V5;
- no autonomous evidence updating;
- production routes remain behind existing Clinical Excellence authentication;
- unresolved/urgent safety can block export;
- safety behavior cannot depend on browsing progressive detail;
- authenticated smoke uses only generated UUID + non-identifiable state and never prints protected keys.

---

# 9. Release evidence

The historical V5 candidate was tested before the released `CY_GESY` integration. Independent review therefore required a fresh-main integration patch.

The final reviewed PR head was:

`4e4bd2ae40c606562a982b3e38f9f859b49986eb`

Successful exact-head gates:

```text
V5 integration                  34736919952
CY_GESY jurisdiction            34736920005
clinical-sheet v4               34736920059
prototype                       34736919984
protected Cockpit integration   34736920081
evidence design                 34736919959
CU-1 focused                    34736919945
```

All were `SUCCESS`.

Final artifact `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

Clinical Learning red checks were adjacent-owner scope-only after substantive tests/frozen-owner guards passed.

PR #101 was squash-merged to:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render auto-deploy `dep-daj2398u01pc738ojvkg` reached `live` at that exact commit. Startup logs confirm application startup complete and configured clinical authentication.

---

# 10. Authenticated post-V5 verification

Temporary non-merged ops branch:

`ops/physio-knee-oa-v5-live-smoke-2026-09-13`

Workflow commit:

`b46c948c38181336d23f90899cdd331dffe09f4c`

Authenticated V5 run:

`34737351354`

It is designed to verify protected bootstrap/project operation, active `CY_GESY`, unchanged international evidence states/default selections/referral semantics, V5 live static-asset routing for quadriceps atrophy and no-browser-storage markers.

At this reconciliation the run is queued awaiting a GitHub runner. A rerun of the previously successful authenticated jurisdiction smoke is also queued. Thus no application smoke failure has been observed, but V5 must not yet be called `PRODUCTION-SMOKE-VERIFIED`.

---

# 11. External feedback policy

Formal physiotherapist/receiver evaluation is optional later external evidence, not a blocking release gate.

Do not describe receiver utility as proven unless actual external evidence is collected.

---

# 12. Validation truth

```text
Knee-OA V5 merged                         yes
V5 deployed                               yes
exact V5 runtime on Render                verified
CY_GESY active                            yes
post-V5 authenticated production smoke    pending execution
real clinical pilot                       not proven
paid conversion / willingness-to-pay      not proven
formal receiver validation                not proven / deferred
actual iPhone Safari / VoiceOver           not proven unless separately recorded
second diagnosis                          not authorized
Greece/England market need                not proven
```

---

# 13. What not to do next by default

Do not infer authority for:

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

---

# 14. New-conversation bootstrap

A fresh conversation working on this product must:

1. fresh-fetch `main` and read the six root canonicals;
2. read this file completely;
3. read `CURRENT.md`, `PRODUCT_PLAN.md`, `UX_CONTRACT_CURRENT.md` and current release records;
4. inspect exact current branch/head before mutation;
5. respect the one-writer lock;
6. distinguish `DESIGNED / IMPLEMENTED / TESTED / MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED / PILOT-VALIDATED / COMMERCIALLY-VALIDATED`;
7. never reconstruct product truth from chat memory alone.

If this file conflicts with newer root canonicals, the newer root operational state wins and this file must be reconciled.

---

# 15. Exact handoff state

```text
Knee-OA V5                           merged / deployed
CY_GESY overlay                      active
Render runtime                       8cfb22fd... / LIVE
authenticated V5 smoke               queued / unverified
formal receiver review               deferred; not a gate
real clinical pilot                  not started
commercial validation                not started
second diagnosis                     not authorized
active writer                        none
```

**Exact next action: complete authenticated V5 production smoke, then finalize the canonical smoke closeout.**
