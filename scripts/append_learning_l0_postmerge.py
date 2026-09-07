from pathlib import Path

p = Path('osteoporosis-change-log.md')
s = p.read_text(encoding='utf-8')
marker = '## 2026-09-07 — Clinical Learning Hub design merged to main'
if marker not in s:
    s = s.rstrip() + '''

---

## 2026-09-07 — Clinical Learning Hub design merged to main

PR #78 (`Clinical Learning Hub: activate L-0 canonical design`) was squash-merged to `main` as `6fa2099647513e4a6b0e71ec30eb5275164626ed`. Render auto-deploy `dep-dafdb695efls73anlt6g` reached `LIVE`; the PR was documentation/design-only and did not change runtime behavior.

The repository now carries `CLINICAL_LEARNING_HUB_DESIGN_V1.md` as product-owner-approved canonical design direction and activates slice `CORE-LEARNING-HUB-L0-2026-09-07` for contract/design freeze. L-1 runtime implementation remains unauthorized until L-0 field-level contracts, provenance/revision semantics, PHI boundaries, due-state behavior and owner seams pass exact design review.
'''
p.write_text(s, encoding='utf-8')
