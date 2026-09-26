# Clinical Documents CURRENT — Sick Leave template themes

> **STATUS:** ACTIVE / IMPLEMENTATION AUTHORIZED.
> **Workstream:** Clinical Documents — Sick Leave visual templates.
> **Branch:** `feat/sick-leave-template-themes-2026-09-26`.
> **Base main:** `c0b89f9c49239142e94e0630580771180d6fcadb`.
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

Required focused evidence:

- all 5 layouts render parseable A4 PDFs with required Greek clinical text;
- all 6 palettes are accepted;
- invalid layout/theme fails closed;
- clinical metadata round-trip is invariant across layouts/colors;
- signature rendering remains supported;
- preview/download API forwards appearance selection;
- UI exposes 5 template choices and 6 controlled themes;
- no browser storage is introduced;
- existing Clinical Documents P1/P2 and navigation regressions remain green.

## Exact next action

Implement the bounded renderer/API/UI/test slice on this branch, run the existing Clinical Documents P1/P2 gates, then checkpoint the exact tested head before release review.
