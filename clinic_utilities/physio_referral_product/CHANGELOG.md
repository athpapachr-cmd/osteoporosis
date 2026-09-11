# CHANGELOG.md — Physio Referral productization

> **ROLE:** append-only supporting product-design history.
> **Parent repo authority:** root six canonicals remain authoritative for repo-wide operations.

---

## 2026-09-07 — Productization track activated

The existing CU-1 Physiotherapy Referral utility was selected as the fastest initial path toward modest recurring revenue, with an initial target of approximately €9.99/month and no immediate objective beyond roughly €1,000 monthly recurring revenue.

The intended longer-term architecture is a unified Clinical Cockpit with separately activatable modules; Physio Referral is the first focused module and should remain independently usable while being architecturally compatible with later cockpit entitlements.

The product must not be positioned as a simple text generator. Subscription value is expected to come from evidence-aware guidance, reviewed updates, clinical structure, flexibility and speed.

---

## 2026-09-07 — Single-diagnosis vertical slice selected

The product owner rejected parallel development of five conditions for the first productization experiment.

First complete vertical slice:

```text
Knee Osteoarthritis only
```

The single slice must prove the reusable architecture before a second diagnosis is added.

---

## 2026-09-07 — UX contract v1 frozen for prototype

The product owner approved a minimal, modern, mobile-first interaction model inspired by direct-manipulation first-party mobile software rather than conventional medical form UX.

Frozen principles include:

- smart evidence-aware starting plan instead of blank form;
- selectable rows/direct manipulation instead of routine checkbox grids;
- progressive disclosure by clinical context;
- compact `Περισσότερα` power-user layer with active-count memory when collapsed;
- live referral projection with no routine Generate button;
- color as an important evidence-state cue, never the only cue;
- green/strong emphasis for supported recommendations;
- neutral/blue-grey for context-dependent options;
- amber for limited/insufficient evidence;
- distinct caution state for recommendation against routine use;
- grey for not-yet-assessed evidence state;
- compact contextual evidence bubbles rather than warning boxes;
- small `i` control for concise rationale/source details;
- full bibliography only on deeper explicit request;
- evidence-backed suggestions with source/year and one-tap add;
- guideline publication year kept separate from product evidence-review date;
- normal final state shown simply as `Έτοιμη` rather than a quality score or dashboard of ticks;
- typical mobile routine path targeted at approximately 5–7 meaningful taps with little or no typing.

The UX contract was design-frozen for the Knee-OA prototype and later replanned only where Step-2 evidence review proved the five-state evidence model insufficient.

---

## 2026-09-11 — Step-2 Knee-OA evidence design frozen

The first evidence knowledge module was completed for Knee Osteoarthritis only on branch:

```text
design/physio-referral-knee-oa-evidence-v1-2026-09-11
```

Fresh base:

```text
d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37
```

Reviewed substantive head:

```text
6b82691c8431b699d20752c83b443793989f6402
```

The evidence review used current/relevant positions from VA/DoD 2026, Singapore ACE 2026, EULAR 2023 update, NICE NG226 2022, AAOS OAK3 2021 and ACR/AF 2019.

A material UX-design finding emerged: five evidence states could not honestly represent real guideline disagreement. Acupuncture was the clearest example, with reviewed major frameworks ranging from recommendation against to conditional/limited support and insufficient evidence.

The evidence-state model was therefore expanded to:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

The compact Greek surface semantic for real disagreement is:

```text
Οι οδηγίες διαφέρουν
```

A second integrity layer was added to prevent broad evidence strength from being silently transferred to narrower product items. Source positions now distinguish:

```text
direct_item_recommendation
named_component_of_broader_recommendation
broader_recommendation_only
contextual_clinical_mapping
```

The reviewed visible Knee-OA smart default is deliberately small:

```text
therapeutic exercise
progressive strengthening
education & self-management
```

with individualized physiotherapy/active rehabilitation implicit in the referral itself. Graded activity and more specific rehabilitation components are context-driven rather than universal defaults.

Manual therapy, soft-tissue techniques and acupuncture retain explicit mixed-guideline semantics. Dry needling remains excluded from the Knee-OA selectable surface; the evidence detail preserves the distinction between NICE recommendation against and AAOS/VA-DoD uncertainty. Weight management is strongly supported when overweight/obesity applies but remains advisory-only because the current CU-1 machine catalog has no dedicated selectable ID. The existing walking-aid ID is retained but is not yet exposed in the current Knee UI relevance scope.

Machine contract:

```text
clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml
```

Human design:

```text
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_V1.md
```

Exact design review:

```text
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md
DESIGN PASS / MATERIAL OPEN FINDING NONE
```

Machine gates:

```text
run 34559461372 — SUCCESS on substantive head 6b82691...
run 34559680326 — SUCCESS on review-artifact head f20f01f...
```

No production CU-1 runtime/API/formatter/static UI/database mutation occurred. Step 2 closes as a design/evidence contract only.

Exact next product-design step:

```text
STEP 3 — dynamic Knee-OA referral/template contract
```

---

## 2026-09-11 — Step-3 dynamic Knee-OA referral/template design frozen

Step 3 defined the deterministic live referral composition layer without modifying the deployed CU-1 runtime.

Branch:

```text
design/physio-referral-knee-oa-template-v1-2026-09-11
```

Frozen Step-2 parent:

```text
ab4b349223cd4c461837ab3125967a06d169a7e1
```

Reviewed substantive head:

```text
cd4a42b4582921df7eb64d6ff3fb7c718a141c3a
```

Review-artifact head:

```text
11f04d9ea316ba7fc30a03f4aaa13b60ea5c1e65
```

The product now has a machine-defined live composition model:

```text
clinician-selected CU-1 state
+ bounded ephemeral Knee-OA phenotype overlay
→ deterministic semantic projection
→ deterministic Greek referral text
```

No LLM is required for routine referral generation.

Material review findings corrected before closure include:

- explicit clinician OA assertion is required before Copy; opening the fixed Knee-OA screen does not itself establish the diagnosis;
- laterality must be `right | left | bilateral` for Copy;
- `stiffness symptom != ROM restriction` and `generic weakness != objective weakness` are preserved through a tiny product-local phenotype overlay;
- exact Greek laterality and task grammar were polished and frozen in fixtures;
- a bounded Knee-OA power-user subset was defined so unsupported selections fail closed instead of silently disappearing;
- selected items must render, block, or be covered by explicit semantic de-duplication that preserves structured state;
- restrictions and clinician notes have deterministic output ownership;
- evidence labels/citations remain UI-only and do not leak into copied referral prose;
- mixed-guideline adjuncts remain subordinate to active rehabilitation;
- `true_locking_or_major_mechanical_rom_block` is deliberately not exposed in the first product surface because the existing CU-1 rule catalog does not derive a safety trigger from that finding itself; future exposure requires a bounded safety/reassessment mapping rather than a second hidden product safety engine;
- manual edited prose remains an ephemeral output buffer and does not reverse-write structured clinical state.

Exact design review:

```text
clinic_utilities/physio_referral_product/KNEE_OA_TEMPLATE_DESIGN_REVIEW_V1.md
DESIGN PASS / MATERIAL OPEN FINDING NONE
```

Machine evidence:

```text
run 34561575795 — SUCCESS on substantive head cd4a42b...
run 34561638107 — SUCCESS on review-artifact head 11f04d9...
```

The validated fixture set covers routine exact-output cases, power-user preservation, semantic de-duplication, unsupported-selection fail-closed behavior and negative Copy-readiness cases.

No production CU-1 runtime/API/formatter/static UI/database mutation occurred.

Exact next product-design step:

```text
STEP 4 — Evidence Interaction / Traceability Layer
```

---

## 2026-09-11 — Step-4 evidence interaction / traceability design frozen

Branch: `design/physio-referral-knee-oa-interaction-v1-2026-09-11`.
Parent: `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
Reviewed substantive head: `e1039809818ddf4061e6d0350905578ff2ca16aa`.

Step 4 froze six distinct colour/non-colour evidence cues, independent neutral selection state, one untimed expanded contextual bubble, one information-sheet host, full mixed-source first disclosure, faithful source scope/version/review-date presentation, explicit source-level locator precision, positive-only source-backed suggestions with stale-candidate rejection and scoped dismissal, retained advanced selections and all-export safety/revision precedence.

The clinical evidence corpus and deterministic template stayed unchanged. Their exact Git blob identities, together with inherited UX identity, are pinned in the new contract and checked by the focused validator.

GitHub Actions `Physio Knee OA interaction design gate`, run `34565131646`, job `103155584052`, completed successfully on the substantive head. It reported **46 synthetic scenario/mutation checks**, three parent-blob checks and six preserved evidence states. This is design-model evidence, not a browser, clinical-efficacy or independent-review PASS.

`KNEE_OA_EVIDENCE_INTERACTION_REVIEW_V1.md` records the active-writer review and freezes that exact content. This record/root closeout supersedes the creation-state candidate headers without modifying the reviewed substantive artifacts.

No product runtime, API, existing formatter, production UI, database, clinical taxonomy or source recommendation was changed. No PR, merge, deployment or production smoke occurred. The Step-4 writer was released.

Remaining acceptance is explicit: actual visual/mobile prototype, Safari/VoiceOver and measured contrast/reflow, final reviewed Greek source summaries, source-to-claim independent verification, product-owner usability and independent clinical/physiotherapy/UX/commercial review. Current clinical locators point to sources, not verified individual recommendation numbers/pages.

Next: **Step 5 bounded functional Knee-OA prototype implementation gate**, requiring its own authorization and writer scope. No second diagnosis or commercial release is inferred from this design closeout.

---

## 2026-09-11 — Step-5 functional synthetic prototype implemented and technically tested

After explicit product-owner progression to Step 5, a new isolated branch was created from the frozen Step-4 parent:

```text
branch feat/physio-referral-knee-oa-prototype-v1-2026-09-11
parent 4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6
tested substantive head 6595bf4cc41388dbd796f9ba5b53ae7c49bafdee
```

The prototype is runnable locally on `127.0.0.1` and has not been registered with or deployed into production. The actual CU-1 engine owns validation and safety; frozen Step-3/4 functions are read-only prototype dependencies. The new Greek UI includes live text, source-aware suggestions, distinct evidence/selection state, a single evidence sheet, compact advanced choices, guarded manual-text reconciliation, synthetic copy/print and in-memory-only draft state.

Initial CI run `34568902516` passed the backend suite but failed one of twelve real browser tests because Tab could leave the modal. The application was corrected without weakening the test. Corrected exact-head run `34569247051`, job `103167619691`, passed all 15 backend/HTTP methods, 12 real Chromium tests and packaged dependency-closure smoke. Backend coverage includes 15 exact frozen Greek-output fixtures and 54 Greek source-summary display positions; those counts are not additional independent clinical tests.

The tested downloadable artifact `10187149429` contains the runnable source ZIP and actual desktop/mobile/evidence screenshots. The ZIP includes setup instructions and a build source/file-hash manifest. Seventy-seven packaged file hashes were verified after download; no font binary is supplied. No production code, clinical source recommendation, patient storage, production configuration or secret was changed. No PR, merge, deploy or production smoke occurred.

`KNEE_OA_PROTOTYPE_REVIEW_V1.md` records the author's technical review, including the initial failure and final passing identity. It is not independent clinical, accessibility or commercial review. Greek summaries still require clinical acceptance; source links remain source-level. Actual Safari/VoiceOver, live BFCache, complete accessibility audit, product-owner usability, willingness-to-pay and independent multi-axis review remain open acceptance boundaries.

The bounded Step-5 writer is released. **Next: Step 6 product-owner trial using synthetic cases**, followed by the separate independent review before expansion or release. No second diagnosis, real-patient use, public hosting or subscription implementation is authorized by this milestone.
