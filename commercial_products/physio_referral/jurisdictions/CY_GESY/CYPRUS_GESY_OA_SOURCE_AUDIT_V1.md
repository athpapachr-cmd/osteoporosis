# CYPRUS_GESY_OA_SOURCE_AUDIT_V1

> **STATUS:** PRIMARY-SOURCE AUDIT V1 — DESIGN EVIDENCE ONLY / NOT RUNTIME AUTHORITY
> **Jurisdiction profile:** `CY_GESY`
> **Diagnosis vertical:** Knee Osteoarthritis only
> **Reviewed on:** 2026-09-12
> **International comparison authority:** `clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml`
> **Runtime mutation authorized:** NO

## 1. Audit purpose

Audit the official Cyprus/HIO/GeSY osteoarthritis position without converting local policy into international evidence and without assuming that publication, reimbursement, or planned information-system integration are the same thing.

Permanent interpretation:

```text
INTERNATIONAL EVIDENCE CORE
!= CYPRUS CLINICAL ADAPTATION
!= GESY ACCESS / REIMBURSEMENT POLICY
!= GESY INFORMATION-SYSTEM ENFORCEMENT
```

A local recommendation may inform an optional jurisdiction overlay. It never silently overwrites the international evidence state.

## 2. Source register

### CY_OA_GUIDELINE_PUBLIC_ARTIFACT

- owner: Health Insurance Organisation (HIO/OAY), Cyprus
- title: adapted NICE NG226 — Osteoarthritis in over 16s: diagnosis and management
- public artifact metadata: cover `ΠΡΟΣΧΕΔΙΟ`; body `Προσχέδιο, Δεκέμβριος 2025`; cover date placeholder `XX Μήνας 2026`
- locator: `https://www.gesy.org.cy/el-gr/annualreport/greek-translated-oa-19-12-2025-hio-circ-0.pdf`
- status interpretation: official HIO-hosted recommendation text currently linked from GeSY; document-version metadata remains stale/draft-labelled
- reviewed_on: 2026-09-12

### CY_OA_ADAPTATION_SUMMARY

- owner: HIO/OAY
- title: adaptation-process summary and characteristic changes to NICE NG226
- source role: explicit comparison of selected NICE wording, adapted wording, and committee rationale
- locator: `https://www.gesy.org.cy/el-gr/annualreport/oa-%CF%83%CF%8D%CE%BD%CF%84%CE%BF%CE%BC%CE%BF-%CE%AD%CE%BD%CF%84%CF%85%CF%80%CE%BF-%CF%80%CE%B1%CF%81%CE%BF%CF%87%CE%AE%CF%82-%CF%80%CE%BB%CE%B7%CF%81%CE%BF%CF%86%CE%BF%CF%81%CE%B9%CF%8E%CE%BD-%CE%B4%CE%B7%CE%BC%CF%8C%CF%83%CE%B9%CE%B1-%CE%B4%CE%B9%CE%B1%CE%B2%CE%BF%CF%8D%CE%BB%CE%B5%CF%85%CF%83%CE%B7-19.12.2025-gr-0.pdf`
- reviewed_on: 2026-09-12

### CY_OA_IMPLEMENTATION_ANNOUNCEMENT_2026

- owner: HIO/OAY
- announcement number: `052026INP08064`
- provider-announcement index date: 2026-05-25
- underlying PDF filename: `ann-osteoarthirtis-guideline-20260521.pdf`
- source role: authoritative lifecycle/implementation announcement
- locator: `https://www.gesy.org.cy/sites/Sites?d=Desktop&locale=el_GR&lookuphost=%2Fel-gr%2F&lookuppage=announcementdef%2Fann-osteoarthirtis-guideline-20260521.pdf`
- key status: adaptation process stated complete; providers informed of implementation; future GeSY information-system integration explicitly described as a future action
- reviewed_on: 2026-09-12

### CY_OA_CURRENT_GUIDELINE_INDEX

- owner: HIO/OAY
- current adult-guideline index: `https://www.gesy.org.cy/el-gr/guide-adults-2025`
- OA consultation/detail page: current HIO OA page under `page-osteoarthritis`
- status caveat: the adult-guideline index lists OA as an adult guideline, while the OA detail page and linked PDF still retain consultation/draft wording from late 2025
- reviewed_on: 2026-09-12

### NICE_NG226

- owner: NICE
- guideline: `NG226`
- published: 2022-10-19
- recommendations locator: `https://www.nice.org.uk/guidance/ng226/chapter/recommendations`
- role: source guideline used for Cyprus adaptation and comparison anchor
- reviewed_on: 2026-09-12

### GESY_ALLIED_ACCESS

- owner: HIO/OAY
- page: access to nurses, midwives and allied health professionals
- locator: `https://www.gesy.org.cy/el-gr/hionurses-midwives-alliedhealthprofessionalsaccess`
- rule class: administrative/access
- reviewed_on: 2026-09-12

### GESY_PHYSIO_SERVICE_RULES

- owner: HIO/OAY
- document: allied-health rules, physiotherapy section
- locator: `https://www.gesy.org.cy/el-gr/annualreport/rules-allied-sep-2025.pdf`
- rule class: administrative/reimbursement/documentation
- reviewed_on: 2026-09-12

## 3. Source-status finding

The official lifecycle evidence is internally imperfect but reconcilable without inventing facts:

```text
HIO May-2026 announcement:
  adaptation complete
  guideline implementation announced
  guideline + quality indicators published on GeSY site

same announcement:
  GeSY IT integration = future/planned action

current linked recommendation PDF:
  still labelled draft / December 2025
```

Disposition:

- `guideline_publication_status = active_announced_by_HIO`
- `public_text_version_finality = ambiguous_stale_draft_metadata`
- `gesy_information_system_integration = planned_not_verified_active`
- `reimbursement_enforcement_from_guideline = not_inferred`

No recommendation below is promoted to an automated live rule from this status alone.

## 4. Strength convention in the Cyprus source

The Cyprus artifact explicitly says recommendation strength is signalled by wording:

- mandatory language: legal/extreme-consequence `must / must not` equivalent;
- strong directive language: offer / do not offer / advise / ask;
- weak language: consider.

The audit therefore records `strong` or `weak` only where that source convention maps directly to the recommendation wording. It does not invent GRADE certainty.

## 5. Recommendation-by-recommendation audit — product-relevant non-pharmacological care

> `verbatim_anchor` is intentionally a short identifying fragment. Full official wording remains at the exact recommendation locator rather than being duplicated here.

### CY-OA-1.3.1 — tailored therapeutic exercise

- verbatim_anchor: `Προσφέρετε θεραπευτική άσκηση`
- normalized local position: offer tailored therapeutic exercise to all people with OA; examples include local muscle strengthening and aerobic exercise
- scope: all OA; directly relevant to Knee-OA referral
- direction: for
- strength: strong by source wording convention
- relationship_to_NICE: agreement; substance retained from NICE 1.3.1
- Cyprus_changes_NICE: no material change identified
- local_change_rationale: not applicable
- rationale_class: not_applicable
- product relationship: agrees with current `therapeutic_exercise = recommended_or_supported`
- guideline operational status: active announced by HIO
- GeSY IT integration: planned/not verified active
- locator: CY_OA_GUIDELINE_PUBLIC_ARTIFACT, rec 1.3.1, p5
- reviewed_on: 2026-09-12

### CY-OA-1.3.2 — supervised therapeutic exercise

- normalized local position: consider supervised therapeutic-exercise sessions
- scope: all OA; delivery-mode decision
- direction: conditional_for
- strength: weak by source wording convention
- relationship_to_NICE: agreement with NICE 1.3.2
- Cyprus_changes_NICE: no
- rationale_class: not_applicable
- product relationship: compatible with current individualized rehabilitation; does not justify mandatory supervised delivery
- status/locator: active announced; CY_OA_GUIDELINE_PUBLIC_ARTIFACT rec 1.3.2, p5
- reviewed_on: 2026-09-12

### CY-OA-1.3.3 — exercise expectations and adherence

- normalized local position: advise that initial pain/discomfort can occur and that regular long-term adherence improves pain/function/quality of life
- direction: for / counselling
- strength: strong by wording convention
- relationship_to_NICE: agreement with NICE 1.3.3
- Cyprus_changes_NICE: no
- product relationship: supports current education/self-management core; no new referral field required
- status/locator: active announced; rec 1.3.3, p5
- reviewed_on: 2026-09-12

### CY-OA-1.3.4 — exercise + education/behaviour-change package

- normalized local position: consider combining therapeutic exercise with education or behaviour-change approaches in a structured package
- direction: conditional_for
- strength: weak
- relationship_to_NICE: agreement with NICE 1.3.4
- Cyprus_changes_NICE: no
- product relationship: agrees with current `education_and_self_management = recommended_or_supported`; does not require a new visible control
- status/locator: active announced; rec 1.3.4, p5
- reviewed_on: 2026-09-12

### CY-OA-1.3.5 — weight management

- normalized local position: for people with overweight/obesity, advise/support weight loss; any loss may help and 10% is likely more beneficial than 5%
- direction: for when applicable
- strength: strong counselling language
- relationship_to_NICE: agreement with NICE 1.3.5
- Cyprus_changes_NICE: no
- product relationship: agrees with current international state; current product correctly requires explicit overweight/obesity context and does not infer it
- routine referral implication: none by default
- status/locator: active announced; rec 1.3.5, p6
- reviewed_on: 2026-09-12

### CY-OA-1.3.6 / 1.3.7 — manual therapy

- normalized local position: only consider manual therapy for hip/knee OA and only alongside therapeutic exercise; explain insufficient evidence for manual therapy alone
- direction: conditional_for_as_adjunct
- strength: weak for use; explanatory qualifier for evidence limitations
- relationship_to_NICE: agreement with NICE 1.3.6–1.3.7
- Cyprus_changes_NICE: no
- product relationship: Cyprus aligns with the NICE side of an **internationally mixed** product state; it must not overwrite `guideline_conflict_or_mixed`
- routine referral implication: no local default or auto-suggestion
- status/locator: active announced; recs 1.3.6–1.3.7, p6
- reviewed_on: 2026-09-12

### CY-OA-1.3.8 — acupuncture / dry needling

- verbatim_anchor: `Μην προσφέρετε βελονισμό`
- normalized local position: do not offer acupuncture or dry needling to manage OA
- direction: against
- strength: strong against by wording convention
- relationship_to_NICE: agreement with NICE 1.3.8
- Cyprus_changes_NICE: no
- product relationship:
  - acupuncture: local position aligns with NICE but international product state remains `guideline_conflict_or_mixed`
  - dry needling: consistent with current product exclusion / against-routine-use state
- routine referral implication: do not auto-add or make local default
- status/locator: active announced; rec 1.3.8, p6
- reviewed_on: 2026-09-12

### CY-OA-1.3.9 / 1.3.10 — electrotherapy

- verbatim_anchor: `μόνο για βραχυπρόθεσμη μείωση του πόνου`
- normalized local position: consider specified electrotherapy modalities only for short-term pain relief, together with therapeutic exercise and weight management; explain no demonstrated functional improvement and no stand-alone role
- included modalities: TENS, interferential therapy, laser, pulsed short-wave therapy, NMES and ultrasound; ultrasound is described as less effective than other listed modalities
- direction: conditional_for_short_term_pain_adjunct
- strength: weak for use by wording convention
- certainty explicitly stated locally: moderate-quality evidence for short-term pain relief; no evidence of functional improvement
- relationship_to_NICE: **genuine difference**; NICE 1.3.9 says do not offer listed electrotherapy because evidence of benefit is insufficient
- Cyprus_changes_NICE: yes
- stated local rationale: committee reviewed newer systematic reviews/guidelines; judged no meaningful functional benefit but possible short-term pain reduction; clarified electrotherapy should not be used alone and should accompany exercise
- rationale_class: clinical_evidence_based
- practice-impact note: adaptation states this reflects current Cyprus practice and is not expected to materially change practice
- product relationship: no electrotherapy item currently exists in the reviewed Knee-OA international product contract; therefore no core evidence state should be mutated
- routine referral implication: none now; future local-only detail if such an item is ever introduced
- status/locator: active announced; recs 1.3.9–1.3.10, p7; characteristic-change summary row 1
- reviewed_on: 2026-09-12

### CY-OA-1.3.11 — radiofrequency nerve ablation

- verbatim_anchor: `νευρική κατάλυση με ραδιοσυχνότητες`
- normalized local position: consider radiofrequency nerve ablation for severe localized pain after non-surgical options have failed and before surgery; explain that effect is temporary and operator expertise matters
- direction: conditional_for_selected_late_pathway
- strength: weak by wording convention
- relationship_to_NICE: local addition; no corresponding recommendation in NICE NG226 non-pharmacological section
- Cyprus_changes_NICE: adds a local recommendation rather than reversing a NICE recommendation
- stated local rationale: included because it is used in Cyprus as a temporary option for localized severe pain before surgery, usually with temporary relief
- rationale_class: clinical_practice_context; no explicit reimbursement/cost motive and no separate evidence-grade rationale stated in the characteristic-change summary
- product relationship: outside current physiotherapy-referral intervention contract and routine receiver need
- routine referral implication: should not be shown routinely
- status/locator: active announced; rec 1.3.11, p7–8; characteristic-change summary row 2
- reviewed_on: 2026-09-12

### CY-OA-1.3.12 — walking aids

- normalized local position: consider walking aids for lower-limb OA
- direction: conditional_for
- strength: weak
- relationship_to_NICE: agreement with NICE 1.3.10; renumbered due local additions
- Cyprus_changes_NICE: no material content change
- product relationship: agrees with current `walking_aid_assessment_and_training = conditional_or_context_dependent`; current Knee UI intentionally does not expose it
- routine referral implication: no change without receiver/workflow evidence
- status/locator: active announced; rec 1.3.12, p8
- reviewed_on: 2026-09-12

### CY-OA-1.3.13 — braces/supports/tape/insoles

- normalized local position: do not routinely offer; consider only when instability/abnormal loading is present, exercise is ineffective/unsuitable without the device, and movement/function is likely to improve
- direction: against_routine_use_with_selected_exception
- strength: strong against routine use
- relationship_to_NICE: agreement with NICE 1.3.11; renumbered
- Cyprus_changes_NICE: no
- product relationship: agrees with current conditional/context-dependent orthosis/brace state
- routine referral implication: no new default or auto-suggestion
- status/locator: active announced; rec 1.3.13, p8
- reviewed_on: 2026-09-12

### CY-OA-1.3.14 — podiatry for painful calluses

- normalized local position: consider podiatry referral when painful calluses impair gait or mobility
- direction: conditional_for
- strength: weak
- relationship_to_NICE: local addition
- Cyprus_changes_NICE: adds recommendation
- stated local rationale: treating painful calluses may improve adherence to therapeutic exercise
- rationale_class: clinical_workflow
- practice-impact note: local summary anticipates potentially increased specialist-podiatry use
- product relationship: outside current Knee-OA physiotherapy-referral handoff value proposition
- routine referral implication: do not show routinely
- status/locator: active announced; rec 1.3.14, p8; characteristic-change summary row 3
- reviewed_on: 2026-09-12

## 6. Recommendation audit — local clinical differences outside the current physiotherapy-referral core

These recommendations matter to OA clinical truth but do **not** justify routine physiotherapy-referral controls.

### CY-OA-1.4.7 — glucosamine + chondroitin

- normalized local position: consider glucosamine plus chondroitin alongside therapeutic exercise and weight management
- direction: conditional_for
- strength: weak
- relationship_to_NICE: material change from NICE NG226, which recommends against offering glucosamine
- stated local rationale: Cyprus use in current practice plus newer systematic reviews/meta-analyses judged supportive of therapeutic benefit
- rationale_class: clinical_evidence_based_plus_practice_context
- product relationship: outside current referral-intervention contract
- routine referral implication: do not show
- status/locator: active announced; rec 1.4.7, p10; rationale pp27–29
- reviewed_on: 2026-09-12

### CY-OA-1.4.9 — intra-articular hyaluronan

- verbatim_anchor: `Εξετάστε ... ενδοαρθρικών ενέσεων υαλορουνικού οξέος`
- normalized local position: consider intra-articular hyaluronan after other treatments are ineffective/unsuitable for selected mild-to-moderate knee OA
- direction: conditional_for_selected_cases
- strength: weak
- relationship_to_NICE: **genuine difference**; NICE 1.4.9 says do not offer intra-articular hyaluronan for OA
- Cyprus_changes_NICE: yes
- stated local rationale: more recent umbrella review plus expert view supporting a role in selected knee-OA pain when alternatives fail or are unsuitable
- rationale_class: clinical_evidence_based
- practice-impact note: source says recommendation reflects current Cyprus practice
- product relationship: outside current physiotherapy-referral intervention contract
- routine referral implication: do not show
- status/locator: active announced; rec 1.4.9, p10; rationale p29; characteristic-change summary row 4
- reviewed_on: 2026-09-12

### CY-OA-1.4.10 — PRP after unsuccessful hyaluronan

- verbatim_anchor: `PRP`
- normalized local position: after lack of benefit from hyaluronan, consider intra-articular PRP for short-term pain relief in knee OA
- direction: conditional_for_selected_cases
- strength: weak
- relationship_to_NICE:
  - no direct PRP recommendation in NG226;
  - separate NICE interventional/HealthTech guidance states efficacy evidence is limited quality and requires special governance/consent/audit arrangements
- Cyprus_changes_NICE: local addition relative to NG226; not a direct reversal of an NG226 recommendation
- stated local rationale in currently linked HIO full artifact: **not separately identified in the rationale section reviewed**
- rationale_class: unclear
- reimbursement/cost motive: not stated; do not infer
- product relationship: outside current physiotherapy-referral contract
- routine referral implication: do not show
- status/locator: active announced; rec 1.4.10, p10
- reviewed_on: 2026-09-12

### CY-OA-1.4.11 — HA/PRP during active inflammation

- normalized local position: do not offer HA or PRP for knee OA during active inflammation
- direction: against_in_context
- strength: strong against
- relationship_to_NICE: local addition; no direct NG226 counterpart
- stated local rationale: not separately identified in reviewed public rationale text
- rationale_class: unclear
- product relationship: outside current physio referral
- status/locator: active announced; rec 1.4.11, p10
- reviewed_on: 2026-09-12

### CY-OA-1.4.13 — intra-articular corticosteroid

- normalized local position: consider when other pharmacological treatments are ineffective/unsuitable or to support exercise; explain short-term relief only
- direction: conditional_for
- strength: weak
- relationship_to_NICE: agreement with NICE 1.4.10, renumbered after local additions
- Cyprus_changes_NICE: no material change identified
- product relationship: outside routine physiotherapy-referral intervention selection
- status/locator: active announced; rec 1.4.13, p11
- reviewed_on: 2026-09-12

### CY-OA-1.5.4 — imaging in non-surgical management

- verbatim_anchor: `Μη χρησιμοποιείτε συστηματικά απεικονιστικές εξετάσεις`
- normalized local position: do not routinely use imaging for follow-up or to guide non-surgical management; local Appendix II adds more detailed imaging guidance
- direction: against_routine_use
- strength: strong against routine use
- relationship_to_NICE: core recommendation agrees with NICE 1.5.4; Cyprus adds Appendix-II guidance
- Cyprus_changes_NICE: adds local implementation detail, not a reversal
- stated local rationale: expert reasoning plus additional EULAR-based guidance for when imaging is appropriate
- rationale_class: clinical_evidence_and_expert_guidance
- product relationship: no routine imaging demand should be added to physio referral
- status/locator: active announced; rec 1.5.4, p12; rationale pp31–32; characteristic-change summary row 5
- reviewed_on: 2026-09-12

## 7. Separate GeSY administrative/access/reimbursement audit

These entries are **not clinical evidence recommendations** and must never change an international evidence state.

### GESY-ADMIN-PHYSIO-ACCESS

- source: GESY_ALLIED_ACCESS
- normalized rule: access to physiotherapy within GeSY requires referral by a participating personal or specialist physician and an eligible diagnosis; direct physiotherapy access is not covered
- rule_class: administrative_access
- clinical evidence relationship: not_comparable
- operational status: current published access rule
- product implication: confirms that a GeSY-oriented referral workflow is operationally relevant; does not determine treatment content
- reviewed_on: 2026-09-12

### GESY-ADMIN-PHYSIO-SESSION-LIMIT

- source: GESY_ALLIED_ACCESS
- normalized rule: GeSY covers a defined maximum number of sessions per diagnosis/beneficiary
- rule_class: reimbursement_access
- exact numeric knee-OA session entitlement: not established by this general access page; do not infer
- clinical evidence relationship: not_comparable
- product implication: not routine clinical evidence UI
- reviewed_on: 2026-09-12

### GESY-ADMIN-PHYS02

- source: GESY_PHYSIO_SERVICE_RULES
- normalized rule: one-to-one therapy session with minimum 30-minute duration; reimbursement documentation includes service, assessment findings, treatment plan and progress in clinical notes
- rule_class: reimbursement_documentation
- clinical evidence relationship: not_comparable
- operational status: current published service rule
- product implication: receiver/service workflow context only; must not be rendered as evidence strength
- reviewed_on: 2026-09-12

### GESY-ADMIN-PROVIDER-UNITS

- source: GESY_PHYSIO_SERVICE_RULES
- normalized rule: provider reimbursement is capped by monthly units; exceptional additional patient sessions require justification/approval under published processes
- rule_class: reimbursement_resource
- clinical evidence relationship: not_comparable
- product implication: **do not show routinely** in a clinical referral product; this is provider/payment mechanics, not patient-specific evidence
- reviewed_on: 2026-09-12

## 8. Source-audit conclusions

1. Cyprus is not merely mirroring NICE. Verified local clinical differences/additions exist.
2. The most directly physiotherapy-adjacent genuine NICE divergence is **electrotherapy**.
3. Cyprus agrees with NICE on exercise, manual-therapy conditions, walking aids, device restrictions and acupuncture/dry needling; however agreement with NICE does not erase conflict in the broader international evidence core.
4. Local additions such as radiofrequency ablation and podiatry are clinically interesting but not automatically receiver-useful for a routine physio referral.
5. Hyaluronan, PRP and glucosamine/chondroitin are important OA-management differences but outside the current referral product's treatment-content boundary.
6. GeSY physiotherapy access/reimbursement rules are operationally real but are not clinical evidence.
7. HIO has announced guideline implementation, while information-system integration remains planned/not proven active.
8. The public OA text still carries draft metadata; this version-status inconsistency must remain explicit until HIO publishes a clearly versioned final artifact.
9. No verified source supports silently replacing the current international evidence state with a Cyprus position.
