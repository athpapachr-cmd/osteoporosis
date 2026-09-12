# Knee-OA Step-5 prototype: author review and technical evidence

> **Date:** 2026-09-11 Asia/Nicosia.
> **Reviewer:** implementing assistant / active Step-5 writer.
> **Independent review:** NOT PERFORMED.
> **Disposition:** bounded synthetic prototype implementation and focused technical gate PASS; proceed to product-owner acceptance, not clinical release.
> **Tested substantive head:** `6595bf4cc41388dbd796f9ba5b53ae7c49bafdee`.
> **Parent:** `4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6`.
> **Production main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`, unchanged.

## Delivered implementation

The runnable entrypoint is `prototype/server.py`, with Greek HTML/CSS/JavaScript and source-specific display summaries. It uses the actual existing CU-1 engine for validation, rather than a browser-provided success flag. The inherited clinical corpus, taxonomy and Step-3/4 semantics remain read-only. A minimal ZIP includes its required CU-1 source/contracts/profiles and a reproducible build manifest, not the production application or any patient database.

The UI implements the selected narrow workflow. Neutral selection checks do not confer evidence endorsement. Optional choices retain state when advanced groups collapse. Contextual suggestions are source/scope-labelled and add only through explicit action. Mixed guidance retains positive, opposing and neutral positions. Live text is deterministic; source labels stay out of the referral itself. Manual edits remain distinct from structured truth. Export is revision-bound and synthetic-stamped.

## Evidence and actual defect correction

Initial real CI run:

```text
run 34568902516
head 2e9798b668fa65870e7831a4670c04acf67946b6
backend tests 15/15 PASS
browser tests 11/12 PASS
failure: test_10_forced_colours_and_modal_keyboard
```

The failing assertion showed focus escaping the modal during repeated Tab navigation. The application now explicitly cycles Tab/Shift-Tab inside the current sheet without trapping Escape or browser shortcuts. The test was not weakened or removed.

Corrected exact-head CI:

```text
workflow Physio Knee OA prototype gate
run 34569247051
job 103167619691
head 6595bf4cc41388dbd796f9ba5b53ae7c49bafdee
result SUCCESS
backend / HTTP tests 15/15 PASS
actual Chromium tests 12/12 PASS
package dependency-closure smoke PASS
scope and syntax PASS
```

Observed logs also reported 15 exact frozen Greek-output fixtures through the real adapter and Greek display-summary coverage for 54 source positions. These counts are coverage within the backend suite, not an extra 69 independent clinical tests.

The browser suite exercised normal task and clipboard output, isolated info interaction and focus restoration, source-backed explicit add/dismiss, five-source acupuncture disclosure, retained advanced selection/count, manual buffer/reconciliation, actual safety block, deliberate network failure, reflow at four viewport widths, 200% root text sizing, visible button hitboxes, six SVG cue identities in forced colours, modal keyboard containment, absence of browser storage/unrequested external requests, reset and simulated page-transition cleanup.

No test substitutes a mock engine for CU-1. The deliberate network abort is an adverse-path test. A local mocked-fetch harness was used only for preparatory visual inspection when sandbox networking prevented repository/server navigation; its results are not counted as clinical integration evidence. Final supplied screenshots originate from the real CI browser run.

## Scope and package inspection

GitHub comparison from frozen parent to substantive head: ahead 17, behind 0; thirteen changed files only, consisting of ten new prototype files, one focused workflow, and root CURRENT/SLICE. No existing production runtime/API/formatter/UI/database or frozen source contract changed.

The successful substantive CI artifact is `10187149429`. Its downloadable inner package contains `BUILD_PROVENANCE.json`, source commit and SHA-256 file identities. The downloaded package was inspected and all 77 manifest entries matched. No font binaries are supplied. Substantive ZIP SHA-256: `9e92586d3f597dbec1c75052943fbb508caa9f45938e7d54401c4ade3132724f`.

Documentation-only closeout descendants may produce a new ZIP hash because the included supporting records change. That does not replace the tested substantive head. A final exact-head workflow verifies the closeout package separately.

## Known limits and required later work

1. **Non-production only.** Loopback access, no production registration, no public preview and no real patient use. Synthetic declarations and banners are not an automatic PHI detector. Export intentionally writes a synthetic sample to the user's clipboard or print output.
2. **Clinical review remains separate.** Evidence positions and review date are inherited, not re-certified here. Greek display summaries need clinician acceptance. Links remain source-level; no verified recommendation/page number is invented. Manual prose is not automatically evidence-validated.
3. **Accessibility is not fully proven.** Automated Chromium viewport/text/keyboard checks do not establish actual Safari/VoiceOver operation, complete contrast compliance, real BFCache behavior or an iPhone-like experience. The BFCache test dispatches synthetic events. Real print-dialog/PDF interoperability also remains user acceptance.
4. **Prototype dependency only.** Reuse of frozen design-checker functions is explicit. A later production version needs reviewed runtime extraction/integration, broader security/privacy and clinical-safety acceptance. The loopback server must not simply be exposed to the internet.
5. **Product validation remains open.** No owner usability trial, independent reviewer, external clinician willingness-to-pay or commercial pilot has occurred. Tap-count/time goals remain hypotheses. No claim of clinical certification, commercial readiness or complete guideline coverage is made.

No unresolved failure remains in the executed bounded technical gate. That statement is not an assertion that every possible clinical, security or UX defect has been excluded.

## Next boundary

Release the Step-5 writer. Product owner tries the exact local package with invented cases, including normal use, a source disagreement, extra choices and a manual-text change. Capture practical friction and what should be removed. Then arrange the separate clinical / physiotherapy / UX / commercial independent review before expansion or release.
