# Clinical Documents CURRENT — Sick Leave template themes

> **STATUS:** RELEASE COMPLETE / PR #117 MERGED / RENDER AUTO-DEPLOY LIVE / USER-FLOW SMOKE PENDING.
> **Workstream:** Clinical Documents — Sick Leave visual templates.
> **Branch:** `feat/sick-leave-template-themes-2026-09-26`.
> **Base main:** `c0b89f9c49239142e94e0630580771180d6fcadb`.
> **Exact tested head:** `256cc6c1d5c20ffad6c8575cde240ce3060ca465`.
> **Root writer lock:** unchanged; PR-1 transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this slice is limited to sick-leave presentation/rendering and focused tests.

## Product-owner request

Add user-selectable professional appearances for the sick-leave certificate so a future commercial user can choose another presentation if the default appearance is not preferred.

The Product Owner also requested selectable document color themes.

## Frozen design

```text
one sick-leave clinical/document data model
+
one renderer contract
+
5 layout templates
+
6 controlled color palettes
=
multiple visual appearances with identical document semantics
```

Layouts:

1. `classic` — current medical-card appearance;
2. `modern` — spacious contemporary clinic document;
3. `minimal` — restrained letterhead-style document;
4. `compact` — denser professional certificate;
5. `formal` — stronger institutional/executive presentation.

Color themes:

- `navy`
- `teal`
- `graphite`
- `burgundy`
- `forest`
- `monochrome`

## Safety / data contract

- template and color affect presentation only;
- patient identity, diagnosis, dates, clinician identity, relation metadata and signature semantics do not change;
- imported previous-document metadata remains compatible;
- no patient persistence/autosave/localStorage/sessionStorage is introduced;
- no free-form color picker in this slice;
- server validates template/color IDs and fails closed on unsupported values;
- every template remains A4, one page for the existing bounded content contract;
- generated PDF metadata schema/version remain unchanged.

## UX contract

The form will show visual template cards and color swatches before Preview / Download.

Selection is session-only in the browser. It is not persisted as a patient or browser preference in this slice.

Preview and Download must use the same selected layout and color.

## Verification

Exact tested head:

`256cc6c1d5c20ffad6c8575cde240ce3060ca465`

Workflow evidence:

- Clinical Documents P1: `36267760233` — SUCCESS;
  - Python syntax PASS;
  - JavaScript syntax PASS;
  - **32 sick-leave deterministic tests PASS**;
  - Clinic Utilities navigation regression PASS.
- Clinical Documents P2: `36267760213` — SUCCESS;
  - Clinical Documents syntax PASS;
  - OpenAI Responses SDK contract PASS;
  - Medical Report V1/V1.1 deterministic regressions PASS;
  - Sick Leave regressions PASS;
  - Clinic Utilities navigation regression PASS.

Proven behavior:

- all 5 layouts render one-page parseable A4 PDFs with the required Greek clinical content;
- all 6 controlled color themes render;
- unsupported template/theme IDs fail closed;
- PDF clinical metadata round-trip is unchanged across layouts;
- signature rendering remains supported on non-default appearance;
- preview/download API use the same selected appearance;
- UI exposes exactly 5 template choices and 6 color themes;
- no localStorage/sessionStorage/patient persistence was introduced.

## Current release state

```text
IMPLEMENTED: YES
TESTED: YES
PR: NO
MERGED: NO
DEPLOYED: NO
PRODUCTION SMOKE: NO
```

## Release PR

```text
PR: #117
URL: https://github.com/athpapachr-cmd/osteoporosis/pull/117
base: main
base_sha: c0b89f9c49239142e94e0630580771180d6fcadb
head: feat/sick-leave-template-themes-2026-09-26
head_sha at PR creation: a6cfc04771311ba1781deeebd97150fb40ce020f
draft: NO
merged: NO
deploy: NO
```

The PR includes the required Canonical Impact Declaration with:

```text
release_affecting: yes
checkpoint_stage: release_hold
workstream_current: update
workstream_current_path: clinic_utilities/clinical_documents/CURRENT.md
changelog: defer_until_completion
```

## Release completion

```text
PR: #117
merge method: squash
merge commit: ad68935750e7ab3101b253fad4d3b94725204461
merged: YES
Render service: osteoporosis / srv-d5qfk31r0fns73di596g
deploy: dep-das28crtqb8s739ddpkg
deploy trigger: new_commit
deploy status: live
manual redeploy: NO
```

PR checks on the final pre-merge head succeeded:

- Canonical impact guard: SUCCESS;
- Clinical Documents P1: SUCCESS;
- Clinical Documents P2: SUCCESS;
- CU-1 focused tests: SUCCESS.

Render deployed exactly merge commit `ad68935750e7ab3101b253fad4d3b94725204461`.

A direct authenticated/browser user-flow smoke was not executed by the assistant because the production Sick Leave route is protected and the available external HTTP fetchers could not resolve/access the Render endpoint from this environment. This is an evidence limitation only; Render reports the exact deployment live.

## Current release state

```text
IMPLEMENTED: YES
TESTED: YES
PR: #117 MERGED
DEPLOYED: YES / LIVE
MANUAL REDEPLOY: NO
PRODUCTION USER-FLOW SMOKE: PENDING
```

## Exact next action

HOLD. No further code or deployment mutation is required. The next real-user use of the Sick Leave form can serve as production smoke; if any visual issue is observed, treat it as a bounded presentation follow-up.
