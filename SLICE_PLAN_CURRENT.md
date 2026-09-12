# SLICE_PLAN_CURRENT.md — CYPRUS / GESY OA JURISDICTION OVERLAY V1 runtime

> **STATUS:** IMPLEMENTATION ACTIVE — REVIEWED DESIGN ACCEPTED BY PRODUCT OWNER.
> **Branch:** `feat/physio-cy-gesy-overlay-v1-runtime-2026-09-12`.
> **Bootstrap main:** `2eb9c9c21c17537d8827c5ecf9aedb3a802f4193`.
> **Writer:** `feat/physio-cy-gesy-overlay-v1-runtime-2026-09-12`.
> **Diagnosis vertical:** Knee Osteoarthritis only.
> **Accepted design authority:** PR `#92` + `CYPRUS_GESY_OA_OVERLAY_DESIGN_REVIEW_V1.md`.

## 1. Objective

Implement the reviewed first real jurisdiction capability without contaminating the international Knee-OA evidence core or expanding the routine referral UI.

Architecture:

```text
InternationalEvidenceItem
  = global reviewed evidence truth

JurisdictionOverlayV1
  = separate local clinical / admin / reimbursement / lifecycle truth

ResolvedEvidenceView
  = international view unchanged
    + optional jurisdiction context for display
```

## 2. Runtime ownership

### New generic runtime owner

`clinic_utilities/physio_referral_product/jurisdiction_overlay.py`

Responsibilities:

- validate profile and local-position machine data;
- resolve only reviewed active profile configured explicitly for the deployment/account;
- map local positions to existing product items when a reviewed mapping exists;
- return display-only local context;
- keep clinical guidance separate from admin/reimbursement/system-lifecycle rules;
- fail closed for malformed, inactive or unknown profile data.

Forbidden responsibilities:

- changing international evidence state;
- source voting;
- selecting treatment;
- changing referral prose;
- inferring jurisdiction from patient location;
- persisting patient data.

### Cyprus profile data

`clinic_utilities/physio_referral_product/jurisdictions/CY_GESY/knee_oa_overlay_v1.yaml`

Contains reviewed local positions only, derived from the accepted primary-source audit and difference matrix. The machine file must preserve provenance, local/core relationship, policy class, operational status and display policy.

## 3. Activation contract

Profile activation must come from explicit deployment/account configuration.

For this single-clinic production deployment the server-side configuration key is:

`PHYSIO_REFERRAL_JURISDICTION_PROFILE`

Allowed first-runtime values:

- unset / empty -> no jurisdiction overlay;
- `CY_GESY` -> reviewed Cyprus/GeSY profile.

Any other value fails closed to no overlay and must not create a patient-facing or clinician-facing error during routine referral generation.

Tests must never rely on workstation/IP/geolocation.

## 4. First-runtime data scope

Machine representation may include all audited local positions needed to prove class separation, but visible item mapping is initially bounded to existing product items.

Existing mapped clinical items with useful on-demand context:

- therapeutic_exercise — local agreement, silent routine;
- progressive_strengthening — local agreement, silent routine;
- education_and_self_management — local agreement, silent routine;
- manual_therapy — local position within international conflict;
- soft_tissue_techniques — local position within international conflict;
- acupuncture — local position within international conflict / local difference;
- dry_needling — local agreement, current item excluded from routine Knee-OA surface;
- walking_aid_assessment_and_training — local agreement, current Knee-OA UI not exposed;
- orthosis_or_brace_context — local agreement/contextual;
- weight_management — local agreement, product advisory only.

Local-only clinical additions/differences such as electrotherapy, RF ablation, podiatry, glucosamine/chondroitin, hyaluronan and PRP remain machine-representable but create no new routine product item or selector.

## 5. Evidence payload contract

Existing international evidence payload remains authoritative and byte/semantic-compatible for its current fields.

A mapped item may receive an additional field:

```json
"jurisdiction": {
  "profile_id": "CY_GESY",
  "label": "Κύπρος · ΓεΣΥ",
  "relationship_to_core": "local_position_within_international_conflict",
  "policy_class": "clinical_guidance",
  "local_direction": "against",
  "normalized_local_position": "...",
  "display_policy": {...},
  "operational_status": {...},
  "source_provenance": {...}
}
```

Invariants:

- `evidence_state` is unchanged whether jurisdiction is active or not;
- source-specific international positions are unchanged;
- no local row is inserted into the international source list;
- admin/reimbursement entries never appear as clinical evidence positions.

Bootstrap may expose non-patient profile metadata needed by UI:

```text
jurisdiction_profile: null | {profile_id,label,selection_source}
```

## 6. UX behavior

Routine screen:

- no new country selector;
- no badge beside every item;
- local agreement silent;
- routine referral text unchanged.

Evidence detail:

- if mapped local context is `local_difference` or `local_position_within_international_conflict`, show one restrained `Κύπρος · διαφέρει` / local-position section;
- keep existing international state and source rows intact;
- clearly label local row `Κύπρος · ΓεΣΥ`;
- local agreement may remain hidden or appear only in deep detail;
- source/status/rationale remain progressive disclosure.

Operational GeSY information:

- no routine display in this slice;
- machine data must remain separately queryable/testable for future workflow seams.

## 7. Test contract

Required focused tests:

1. schema/profile validation accepts exact reviewed `CY_GESY` data;
2. malformed policy class/direction/status/review state fails closed;
3. unknown/inactive profile produces no overlay;
4. mapped local clinical position attaches without changing `evidence_state`;
5. local-only position does not invent an international product item;
6. admin/reimbursement positions never attach to clinical evidence view;
7. acupuncture international state remains `guideline_conflict_or_mixed` while Cyprus local direction remains separately `against`;
8. manual therapy international mixed state remains mixed with local conditional adjunct context;
9. routine referral text is identical with overlay on/off for same clinical state;
10. no storage/persistence introduced;
11. browser evidence sheet shows separate Cyprus detail only when relevant;
12. existing v4 Knee-OA, CU-1 integration and safety tests remain PASS.

## 8. Scope exclusions

- no v5 post-use merge or edits;
- no second diagnosis;
- no Greece/England content;
- no electrotherapy selector;
- no billing/session calculator;
- no provider-unit UI;
- no planned-IT enforcement;
- no automated local recommendation activation;
- no evidence-contract reclassification.

## 9. REPLAN triggers

Stop and replan if implementation would require:

- modifying international evidence semantics;
- adding a routine clinical control solely because a local source mentions it;
- treating GeSY reimbursement/admin status as efficacy evidence;
- patient/location-derived jurisdiction selection;
- changing referral prose to mention GeSY without receiver/workflow evidence;
- overlapping mutation with the held v5 branch that cannot be isolated cleanly.

## 10. Exit gate

Implementation-complete means exact-head focused + inherited tests PASS and branch diff remains bounded.

Release-complete later requires reviewed PR, merge, Render auto-deploy verification and authenticated production smoke with `CY_GESY` explicitly configured.
