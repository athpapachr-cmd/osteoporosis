# Web Visit Capture — standard OpenAI retention release receipt

2026-10-27, Asia/Nicosia. This records technical deployment and configuration, separate from the clinician's first real use.

## Decision and reviewed source

- The Product Owner replaced the earlier requirement for verified Zero Data Retention with standard OpenAI API retention. The clinician notice discloses the OpenAI transcription request and the Responses request containing transcript plus selected-chart name/date of birth when present. Responses uses `store:false`, but standard abuse-monitoring logs may contain content normally up to 30 days, with documented exceptions. This is not a deletion guarantee or a legal-basis certification. [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data).
- [PR #235](https://github.com/athpapachr-cmd/osteoporosis/pull/235) final head `884483078fdd59f8878b73337ce6c75ea98bb2e2` passed 14/14 exact-head GitHub checks. Corrected independent pre-code and post-code R2 both passed P0:P1:P2=0:0:0. The protected server checks explicit standard-retention acknowledgement before reading audio; browser withdrawal discards recording. Existing chart authority, editable source review, Preview and one explicit Save remain.
- PR #235 squash merged to `bb33258ea31a961560c3c34c2d93d7d50fbbcbe3`. Render event `evt-dbgekt3rjlhs73ag7mmg` ignored the commit because its squash body contained `[skip render]`. Exactly one justified manual code deploy `dep-dbgelj0n74is739bjjog` of that merge reached LIVE at 2026-10-27T18:19:55.690046Z.

## Configuration and no-PHI smoke

- In the Product Owner-confirmed Render workspace `tea-d5qfh5vpm1nc738pvbv0`, service `srv-d5qfk31r0fns73di596g`, the three dedicated settings were merged: `VISIT_AUDIO_OPENAI_PHI_APPROVED=1`, `VISIT_AUDIO_RETENTION_MODE=standard`, and `VISIT_AUDIO_OPENAI_PROJECT_ID` set to the project ID supplied by the Product Owner. Existing masked `OPENAI_API_KEY` and other settings were not changed. No `VISIT_AUDIO_ZDR_VERIFIED` or `VISIT_AUDIO_ZDR_PROJECT_ID` bypass was set. The Dashboard showed the three new variable names.
- The configuration update triggered one same-code deploy `dep-dbgen2gn74is739bon1g`, LIVE at 2026-10-27T18:23:03.682205Z with exact code commit `bb33258ea31a961560c3c34c2d93d7d50fbbcbe3`.
- After each LIVE transition, served `static/cockpit/patient-chart.js` and `static/cockpit/index.html` SHA-256 matched the reviewed source bytes. Anonymous capability, draft, context, preview and save routes returned HTTP 401. Render application error and request HTTP 5xx logs were empty in both release windows (18:17:45–18:21:00Z and 18:20:50–18:24:00Z). The smoke used no patient audio or clinical write.

## Practical limit and handoff

The dedicated standard-retention configuration is LIVE. The existing API key remains masked, so its association with the Product Owner-supplied OpenAI project was not independently inspected. Both provider calls send the explicit `OpenAI-Project` header and will fail closed if that key is not authorized for the project; no organization header was added because no organization ID was verified. An authenticated capability response and actual provider routing were not exercised. The clinician's first informed-patient Start/Stop, draft inspection/correction, identity confirmation, Preview and explicit Save remain the acceptance step. No agent request sent real patient content to OpenAI, and no clinical record was written.

This receipt is to be merged as docs only through [PR #236](https://github.com/athpapachr-cmd/osteoporosis/pull/236) after exact-head CI, using `[skip render]` so documentation does not create another app deploy.
