# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** PHYSIO REFERRAL STEP 4 — EVIDENCE INTERACTION / TRACEABILITY DESIGN FROZEN / CLOSED.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Frozen Step-3 parent:** `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
> **Design branch:** `design/physio-referral-knee-oa-interaction-v1-2026-09-11`.
> **Closed slice:** `CU1-PRODUCT-KNEE-OA-INTERACTION-V1-2026-09-11`.
> **Reviewed substantive head:** `e1039809818ddf4061e6d0350905578ff2ca16aa`.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE; Step 4 closed.
> **ACTIVE RUNTIME WRITER:** NONE.
> **PR/merge/deploy/production smoke authority:** NONE.
> **Patient-data / production config / secrets authority:** NONE.

## 1. Preserved state and ancestry

Main and the Step-3 branch were freshly verified before work. Main has no active writer and records Clinical Learning L-1D closed. Step 4 is a stacked design branch from the exact frozen Step-3 parent, not a merge into main. Existing CU-1, Learning Hub, RF, API, production UI and database owners are unchanged.

Steps 1–3 remain frozen as previously recorded. Step 4 did not independently re-review the clinical evidence or repeat inherited tests for reassurance. Its focused gate pins the exact Step-2 evidence, Step-3 template and UX blobs and validates the new interaction design against them.

## 2. Frozen Step-4 artifacts

All product paths below are under `clinic_utilities/physio_referral_product/`:

- `KNEE_OA_EVIDENCE_INTERACTION_DESIGN_V1.md`
- `contracts/knee_oa_evidence_interaction_v1.yaml`
- `contracts/knee_oa_evidence_interaction_fixtures_v1.yaml`
- `validate_knee_oa_evidence_interaction_v1.py`
- `KNEE_OA_EVIDENCE_INTERACTION_REVIEW_V1.md`

Workflow: `.github/workflows/physio-knee-oa-interaction-design.yml`.

This closeout freezes the substantive content at `e1039809818ddf4061e6d0350905578ff2ca16aa`. Candidate headers inside that immutable reviewed content record its pre-review creation state; this canonical and the review record own the subsequent freeze disposition. No clinical or interaction rule is changed by the closeout.

## 3. Evidence actually obtained

```text
workflow: Physio Knee OA interaction design gate
run: 34565131646
job: 103155584052
head: e1039809818ddf4061e6d0350905578ff2ca16aa
job result: SUCCESS
validator: 46 synthetic scenario/mutation checks PASS
parent blob identity checks: 3 PASS
six evidence states: preserved
review: ACTIVE-WRITER DESIGN PASS, not independent review
```

The exact comparison to Step-3 parent was ahead 2 / behind 0, with seven changed files confined to Step-4 design/contracts/checks/workflow and root operational/slice documents. Closure descendants update documentation/navigation/history only.

## 4. Key interaction decisions

Selection, evidence, source availability and safety are separate axes. Six evidence states retain distinct visible non-colour cues. There is one evidence control per item, at most one expanded non-modal bubble without a timer, and one modal information-sheet host with keyboard/focus rules. All source positions appear on first disclosure for mixed guidance.

Suggestions remain explicit clinician actions, scope-labelled and source-linked. Duplicate reasons merge. Dismissal is local to the current draft/item/reasons/package. Stale or no-longer-eligible suggestions cannot be added. Collapsing advanced choices does not clear them. Source failures do not erase opposing guidance or change the last reviewed clinical verdict.

Step-3 validation/safety remains authoritative over every export action. Evidence disclosure/dismissal cannot clear a safety block, mutate structured clinical facts, change the referral or overwrite manual edits. Missing/stale gate or projection state is not labelled ready.

## 5. Not proven and required later

- No browser UI or functional prototype has been implemented.
- No measured contrast, reflow, touch-target, Safari/VoiceOver or usability PASS exists.
- Clinical positions are inherited from Step 2, not clinically re-certified in this session.
- Current clinical locators are source-level, not per-recommendation/page verification.
- Final reviewed Greek source-summary copy, actual design tokens, focus behavior and adapter wiring require prototype acceptance.
- Independent clinical/physiotherapy/UX/commercial review and willingness-to-pay validation remain pending.

## 6. Exact next boundary

```text
STEP 5 — bounded functional Knee-OA prototype implementation gate
```

Start from fresh main plus this exact stacked design ancestry; read the six canonicals. Establish the narrow implementation scope/entrypoint and claim a new writer before writing runtime. Build one synthetic, non-production Knee-OA flow reusing frozen Step-2/3 identities and Step-4 presentation semantics. Do not generalize to a second diagnosis.

This closeout is not implementation authority. No PR, merge, deployment, production smoke, billing/auth/entitlements, patient persistence or autonomous evidence update is authorized. A final branch-head design gate may verify this documentation-only closeout without changing the reviewed substantive identity.
