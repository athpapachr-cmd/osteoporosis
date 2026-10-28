# Complete clinical summary — released 2026-10-28

The clinician's final chart showed only the pain, despite a draft containing fracture history, findings and a conditional alendronate recommendation. Source inspection proved that the overview selected only reason, impression and decisions; the saved reader also omitted history/recommendations/follow-up.

One actual provider call through the existing production extraction route with the clinician's fictional text returned all four requested facts (run40740933424). A separate read of the already signed synthetic visit confirmed fracture before two years, pain duration three months, T-score -2.8 and alendronate after dental assessment were already stored (run40741316431). No production clinical Save was performed.

## Released behavior

A common read-only renderer shows populated clinical sections, medications, next contact and existing coding/completeness without truncation in the latest chart card, saved visit reader and automatically prepared draft. Recommendations remain recommendations; no treatment start is invented. Draft obligations remain proposed, persisted obligations keep their actual lifecycle. Pasted new visits show chart-style result cards first; Edit opens the established editor. Explicit Save, backend identity/write checks, amendments and audio recovery remain.

Independent R1 found and closed one further omission of stored coding. Final runtime/test head adc5f728 passed COMPLETE_FOR_DECLARED_SCOPE, P0:P1:P2=0:0:0. Local 62 affected tests passed; focused CP2 plus three no/incomplete/complete-coding browser cases passed4/4. The four-fact oracle checks before Save, actual local DB Save, desktop/mobile reload and historical reader. All23 reported PR #252 checks passed.

## Production evidence

- PR252: https://github.com/athpapachr-cmd/osteoporosis/pull/252
- Merge77b1656cfe15145e14d3abb64b0cc051251cb781.
- Render dep-dbh3lpss728c73bvk4d0 LIVE18:13:42.732339Z, native auto-deploy, no manual duplicate.
- HTML SHA256 ecf7710f443c4705c3ad91b8efca88cefe5e8bb2faf8480c79c537974b14a459 matched merge.
- patient-chart.js SHA256 511aea1b78d6ad12cf87324a7e2d7f11cc1459fff5df1b39c390cdcd56abc9cd matched merge.
- Anonymous chart HTTP401; observed release-window app ERROR/HTTP5xx empty.
- Production Chromium run40743751686 PASS1280/390: existing signed synthetic chart shows all four facts, cropped screenshot inspected, zero clinical writes and no provider replay.
- Initial UI-proof harness attempts needed normal sign-in ordering and distinction between read-only RF search POSTs and clinical mutation endpoints; neither failure identified a runtime defect. Final observed non-login POSTs were only /clinical/clinic-utilities/rf/api/outcomes/search.

Verification-only branch codex/summary-provider-proof is never merged. It uses the existing protected source and fictional data; no API keys or actual patient data were exposed in output.

## Handoff

The existing saved visit needs only page refresh. Its source data was not rewritten. The corrected final card displays fracture history, three-month pain duration, T-score and the dental prerequisite for alendronate. No new recording or repeated clinical entry is required.
