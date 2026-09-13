# CURRENT_OPERATIONAL.md — Clinical Documents Medical Report V1.1

> **STATUS:** IMPLEMENTED / EXACT-HEAD TESTED / RELEASE HOLD.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Fresh bootstrap main:** `53ea38318d99e81218e5402e3c206ba30c51a8a9`.
> **Branch:** `fix/clinical-documents-v1-1-source-resolution-r2-2026-09-13`.
> **Slice:** `CU-CLINICAL-DOCUMENTS-V1-1-SOURCE-RESOLUTION-2026-09-13`.
> **Exact reviewed/tested runtime head:** `4c30867e8aea00842772b6df7d624a79bbf3f19e`.
> **Clinical Documents V1.1 gate:** `34740684612` — SUCCESS.
> **Writer:** none — implementation checkpoint closed in release HOLD.
> **Production state:** unchanged by this branch; Medical Report V1 remains the deployed production runtime until a separately authorized release.

## 1. Product Owner authority exercised

The Product Owner asked to implement the Medical Report defects discovered during real production use and explicitly asked for canonicals to be maintained so work can continue safely in a fresh conversation.

The triggering real-use findings were:

- selected uploaded files could not be removed;
- prescription/referral/Heidi source categories were missing;
- image-only clinical PDFs were detected but could not be read;
- a readable external sick-leave certificate was not reliably converted into structured leave dates;
- AI surfaced source/date conflicts but there was no clinician resolution/conversation loop;
- pending investigations could be described too much like missing results;
- occupation absence generated noise even where irrelevant;
- long AI calls had poor progress feedback.

The real patient documents used to discover these issues were inspected only in the private conversation context and were **not** copied into this public repository, tests or fixtures.

## 2. Implemented V1.1 behavior

### Source intake / browser

- each selected file can be removed before generation;
- source classifications now include `prescription`, `imaging_referral`, `specialist_referral`, `lab_or_service_referral`, `heidi_transcript`;
- changing/removing source inputs invalidates stale draft state through the inherited source-change behavior;
- long AI operations show elapsed time and explicitly warn against double submission.

### Clinical source semantics

Hard distinctions are now included in the AI contract:

```text
referral != completed examination / consultation
prescription != medication actually taken or administered
requested/pending != completed/result available
source fact != clinician resolution
```

### Image-only PDFs

When deterministic PDF text extraction finds no usable text layer, V1.1 may render up to 12 pages locally and use the already-gated AI provider for visible-document extraction.

The resulting source remains explicitly marked:

```text
status = visual_extracted
extraction_method = visual_ai
review_required = true
```

It is never represented as deterministic text extraction. If visual reading cannot run or fails, the original `no_extractable_text` state remains visible and the main analysis may continue from other usable sources.

### Sick-leave sources

For a source classified `sick_leave_certificate`, the runtime now attempts structured leave extraction before/alongside general synthesis:

1. compatible Clinical Documents embedded metadata when available;
2. explicit text dates such as `Αναρρωτική άδεια από ... έως ...`;
3. after visual extraction if the original PDF had no text layer.

Structured leave dates become provenance-bearing `work_absence` evidence and a typed work-absence interval. They remain clinician-review-required.

### Clinician resolution + AI discussion

A protected, session-only `/api/refine` path and browser discussion layer now allow the clinician to explain conflicts or ask follow-up questions after the initial draft.

Examples supported by the design:

- another clinician's retrospective note contains a wrong date;
- primary emergency documentation is accepted by the clinician as authoritative;
- an MRI was requested and is still pending rather than missing;
- a source statement requires contextual clarification.

The original Evidence Ledger and source summaries are immutable during refinement. Clinician corrections are represented separately as `ClinicianResolutionV1`. The AI may revise downstream timeline/report prose but is rejected if it attempts to rewrite the original evidence ledger.

Proposed refinement is not applied automatically; the browser exposes explicit Apply / Discard controls. Applying a refinement invalidates prior research output so stale literature is not silently carried into the final report.

### Occupation warning

Missing occupation is suppressed as a generic warning unless the stated report purpose makes occupation/work-capacity/return-to-work reasoning relevant.

## 3. Privacy / persistence boundary

Unchanged:

- no Medical Report patient/case database;
- no localStorage/sessionStorage/indexedDB PHI state;
- no autosave;
- source files remain request scoped;
- AI refinement conversation remains browser-session/request scoped;
- no OpenAI Files/Vector Store persistence;
- protected clinical authentication and existing AI/PHI provider gates remain mandatory;
- real patient source files are forbidden in public repository tests/fixtures.

## 4. Acceptance evidence

Exact tested runtime head:

`4c30867e8aea00842772b6df7d624a79bbf3f19e`

Clinical Documents workflow:

`34740684612` — **SUCCESS**.

That exact run passed:

- Python syntax;
- base + V1.1 JavaScript syntax;
- OpenAI Responses SDK contract guard;
- inherited Medical Report V1 deterministic tests;
- V1.1 source-type and semantic tests;
- deterministic sick-leave extraction;
- image-only PDF visual-source semantics;
- synthetic visual PDF rendering to bounded JPEG data URLs;
- fake-provider visual fallback API path;
- immutable Evidence Ledger refinement guard;
- clinician-resolution/refinement API path;
- context-sensitive occupation warning;
- V1.1 browser extension loading/order/static privacy checks;
- existing Sick Leave V1 regression;
- existing Clinic Utilities navigation regression.

Fresh exact comparison to bootstrap main remained ahead-only with no Physio/RF/Clinical Learning/Osteoporosis clinical-rule owner mutation.

## 5. Release boundary

This V1.1 candidate is **not merged or deployed** by this implementation checkpoint.

```text
IMPLEMENTED             YES
EXACT-HEAD TESTED       YES
PRODUCTION V1.1         NO
V1.1 PRODUCTION SMOKE   NO
```

The next legitimate action is a fresh-main release review/PR followed by merge/Render auto-deploy only after explicit Product Owner release authority. Production must not be called V1.1 until that separate release path completes.