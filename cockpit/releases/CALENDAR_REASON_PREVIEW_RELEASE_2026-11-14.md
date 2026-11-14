# Compact calendar reason preview release

- Product Owner requested a small reason line under each appointment name when present; the full original remains accessible in details.
- PR #292 final head `b4b41880d06e16e99811247e423c29a4308ba933` passed CI. Independent R1 review of candidate `420bba4` PASS0:0:0; subsequent CSS-only readability adjustment passed focused browser tests.
- Merge `2446be68bfa3e6982814fde571790dae4e02dc55` went LIVE on Render `dep-dbrvs3qjnfac73bnl500` at 2026-11-14 06:23:48Z.
- Three deployed assets matched source. Synthetic no-PHI browser smoke on the deployed page passed at 1440/390 with 75 appointments, optional/long reason, full details, Escape and filter. All clinical calls were intercepted.
- Render ERROR and HTTP500/502/503 queries from live through 06:24:19Z were empty. Real authenticated data and clinician acceptance remain unverified; no clinical/provider writes.
- Documentation-only closeout uses `[skip render]`.
