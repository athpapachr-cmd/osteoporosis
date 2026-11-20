# Osteoporosis — Adaptive Treatment Strategy (synthetic J/JD preview)

**Bounded synthetic implementation.** No production patient source, API, FRAX/NOGG engine, clinical record write, treatment selection, prescription, or administered-dose mutation.

## What the preview does

- J: separates *no prior treatment recorded* from explicitly confirmed absence of previous treatment; explores only relevant treatment approaches without choosing medication.
- JD: recognizes prior denosumab exposure; does **not** infer continuation or discontinuation. Shows only actual documented administration dates, not scheduled events as exposure.
- Keeps **approach considered**, **clinician recommendation**, **patient preference** and **final decision** independent. Displays a *synthetic-only* handoff that never becomes a real Plan/task/administration.
- Reuses immutable synthetic patient snapshots in the adapter/core. No state is stored outside page memory.
- Illustrates S3–S8 as additional regression fixtures. Developer fixture picker is in `dev.html` only; not on the clinical surface.

## Open preview

From this directory, open `offline-preview.html` directly in a browser (single self-contained file, no server or network needed), or serve the directory locally and open `index.html`. Add `?case=JD` to the URL for the prior-denosumab variant. Further cases are linked from `dev.html` when served locally.

## Test

```bash
node --test test_treatment_strategy_core.cjs
python test_treatment_strategy_browser.py
```

The browser test uses Python Playwright and `/usr/bin/chromium` (local test environment); run only where those are installed. Tests inspect the same pure core/fixtures/browser scripts used in the preview without calling external systems. Tested J and JD flows, actual-vs-planned administration, unknown/conflicting status, preference/recommendation/decision separation, no real writes, and mobile widths.

## Source ownership and next gate

This isolated preview intentionally does **not** bind to the unmerged FRAX adaptive files on another worktree or the production patient registry. The next phase would require Product Owner smoke acceptance and a separate reviewed read-only FRAX contract binding. Real persistence requires a separately approved semantic extension of the existing Step 4 decision owner. No merge, deploy or real patient use is implied by this preview.

### Design decisions included

1. `JD` question is **not** “how do we continue Prolia?” until current status is established; it asks how prior Prolia influences today's decision.
2. No initial catalogue of drugs or three generic equal-weight cards. The clinician sees one clinical question first, with treatment approaches revealed only after relevant history has been clarified.