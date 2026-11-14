# Compact weekly calendar release

- Product Owner requested compact readable appointment rows and one-click details.
- PR #290 final head `39a992448ec28d433f9ceddb6d14d43cea7fcfde` passed all final checks; substantive behavior `3704cef` received independent R1 closure PASS0:0:0 after correction of duplicate-ID, pending-dialog and uncertain-write findings.
- Squash merge `7d773b037bb64f3e6f914c60636cdf3442061229`; native Render deploy `dep-dbrvbhe417fc73eui53g` LIVE 2026-11-14 05:48:29Z.
- Rows show time/name and category indicator; details retain existing chart and classification actions. Full protected weekly agenda qualifies duplicate IDs before local category filtering; duplicate IDs suppress ambiguous actions.
- Permanent calendar CI runs synthetic Chromium tests. Synthetic busy week of 75 appointments showed 15 rows per workday within a 1440×900 viewport. Small screens, details, keyboard, filtering, network failure and authentication were tested.
- Live no-PHI/no-write smoke matched the three served assets to source and exercised 75 synthetic appointments, detail, Escape and filter at 1440 and 390 widths, with all clinical requests intercepted. Render ERROR and HTTP500/502/503 queries from live through 05:49:18Z were empty.
- Real authenticated source interactions and clinician usability are not established by synthetic smoke. Duplicate qualification covers the returned week; a timed-out classification write may still complete later and the UI states that uncertainty.
- Final documentation publication uses `[skip render]`.
