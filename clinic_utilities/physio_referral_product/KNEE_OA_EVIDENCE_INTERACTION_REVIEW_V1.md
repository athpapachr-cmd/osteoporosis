# Knee OA Step-4 exact design review and freeze record

> **Date:** 2026-09-11 Asia/Nicosia.
> **Reviewer:** the active Step-4 design writer in this session.
> **Independent review:** NOT PERFORMED.
> **Disposition:** DESIGN PASS for the bounded interaction contract; freeze the reviewed substantive content.
> **Reviewed head:** `e1039809818ddf4061e6d0350905578ff2ca16aa`.
> **Parent:** `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
> **Verified main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.

## 1. What this review establishes

The human interaction specification, machine contract, synthetic fixtures and design validator were checked together against the unchanged Step-2 evidence and Step-3 template. The review addresses Step-4 design consistency, state ownership and traceability, not independent clinical validation or usability.

The freeze record supersedes the pre-review `design_candidate` creation labels at the reviewed head without editing that substantive content. Lifecycle authority remains the root CURRENT/SLICE canonicals.

## 2. Concrete improvements incorporated

- Separate neutral selection indication from evidence support, availability and safety.
- Six distinct visible non-colour cues; evidence meaning does not rely on six colours or a screen-reader-only label.
- One evidence control per item and one information-sheet host; no nested interactive row buttons or popup stacks.
- At most one expanded evidence bubble, without automatic timeout or focus capture; other notes remain discoverable.
- Mixed-guidance first disclosure retains every reviewed source position, including neutral and opposing positions.
- Source-specific strength, direction and claim scope remain independent. Broad exercise recommendations are not laundered into stand-alone strong recommendations for narrower tasks.
- Source/version/publication year stays separate from the clinical review date. Opening the UI and checking a link cannot refresh it.
- Source-level URL precision is explicit; no exact recommendation number/page or independent verification is invented.
- Availability is a separate overlay. Inactive material dependencies suppress new endorsement/promotion but do not erase existing clinician selections, rewrite the clinical verdict or remove an opposing source.
- Positive-only typed trigger facts, explicit aliases, one candidate per item, stable ordering and scoped dismissal.
- Explicit add revalidates draft identity/revision, package, item and reason signature. Old suggestions cannot enter a changed or new referral.
- Step-3 safety/validation and revision checks govern all export actions. An evidence dismissal cannot clear clinical safety or overwrite a manual buffer.

## 3. Machine evidence

```text
workflow: Physio Knee OA interaction design gate
run: 34565131646
job: 103155584052
exact checked commit: e1039809818ddf4061e6d0350905578ff2ca16aa
result: SUCCESS
```

Observed job output:

```text
Step-4 design PASS: 46 synthetic scenario/mutation checks; 3 pinned parent blobs; 6 evidence states.
NOT TESTED: browser rendering, pixel contrast, VoiceOver/Safari, usability, clinical source re-review, independent review.
```

Check composition: 13 suggestion cases, 7 evidence cases, 8 readiness cases, 6 synthetic evidence-state projections, 1 scope/date preservation case, 1 explicit-add case, 4 stale-candidate rejection cases, 1 dismissal/new-reason case, 1 UI-only transition/count model case and 4 checker mutation oracles.

Parent Git blob pins verified by the job:

```text
Step-2 evidence: 8f4c657904ee028bba39d7a0557461a6acafadaf
Step-3 template: e6c6a285ed4d2d64df1dca7b29630ef5053831e8
inherited UX: f65343c0bc9652c715075853d3fc42c1910ff212
```

No inherited broad runtime regression was rerun merely for reassurance. GitHub Actions performed the executable checks; local repository execution was not claimed.

## 4. Scope/ancestry evidence

Exact comparison parent → reviewed head: ahead 2, behind 0. Seven changed files only:

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_INTERACTION_DESIGN_V1.md
clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_interaction_v1.yaml
clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_interaction_fixtures_v1.yaml
clinic_utilities/physio_referral_product/validate_knee_oa_evidence_interaction_v1.py
.github/workflows/physio-knee-oa-interaction-design.yml
```

No production runtime, API, existing formatter, static application UI, clinical taxonomy, evidence positions or database owner changed. Subsequent closeout updates are supporting documentation/navigation/history only.

## 5. Limits, not hidden PASS claims

The executable model is a design oracle, not production integration. Its pure transitions do not establish that a future DOM implementation preserves focus, selection or patient privacy. Those properties need actual browser acceptance in Steps 5/6.

Visual cues, actual contrast, iPhone Safari/VoiceOver behavior, sheet interaction, final Greek source-summary copy and measured routine tap count remain untested. The source corpus is inherited, not independently re-reviewed here. Source URLs do not yet supply precise per-recommendation locators. These limitations are explicit in the design and future acceptance matrix.

Clinical safety thresholds are not redefined by this review. The inherited CU-1 safety result must be provided by its owner; the Step-4 model cannot manufacture clearance from an empty gate.

No material contradiction blocking the bounded Step-4 design was identified by this active writer. This is not `INDEPENDENT PASS`, `CLINICALLY VALIDATED`, `ACCESSIBILITY COMPLIANT`, `IMPLEMENTED`, `COMMERCIAL READY` or `DEPLOYED`.

## 6. Next gate

Release the Step-4 writer and stop. Next is a separately authorized Step-5 functional Knee-OA prototype with a narrow entrypoint, approved implementation boundaries and a real browser acceptance plan. Independent clinical, physiotherapy, UX and commercial review remains after the functional slice and before expansion/release.
