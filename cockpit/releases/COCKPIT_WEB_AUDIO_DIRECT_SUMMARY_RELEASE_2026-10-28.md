# Direct audio summary release — 2026-10-28

PR #250: https://github.com/athpapachr-cmd/osteoporosis/pull/250

The clinician's fictional recording reached transcript-only identity refusal and exposed a checkbox with no candidate, locked paste, and no automatic summary. This correction shows the server-rendered clinical summary immediately after successful processing. Transcript and editor remain optional; the identity checkbox is removed. The explicit Save button names the displayed patient destination and obtains a fresh preview of the exact current candidate before writing. Existing backend and local-audio identity requirements remain.

The prompt now explicitly treats unnamed age/sex/history as clinical facts rather than identifying evidence, including the clinician's generic 68-year-old example against a synthetic chart label. A presenting complaint can supply the reason without a heading. Genuine conflict/uncertain results remain blocked; no model verdict is coerced.

Manual paste unlocks after processing. Its separate context is adopted only on success. Failure preserves both transcript-only signed recovery and a previous completed audio candidate/summary/Save. Independent R2 found and closed both no-loss edge cases. Final code/test head 8e9623aa passed independent R2 COMPLETE, P0:P1:P2=0:0:0; 47 local tests including Chromium passed. PR final head 8b997ae9 passed all 15 checks.

Merge: 796b11cdb02884deea4a35463c69eff94458cfc2.
Native Render deploy: dep-dbh29tfb9u1s73e71900, LIVE at 2026-10-28T16:40:12.574051Z.

No-PHI smoke:
- index.html SHA-256 bc9235b4e841ad5c3245b9a0cc141fd85c2523e0593bb231dcc7262bc2b6491e matched merge.
- patient-chart.js SHA-256 a24303d78c85d6b2a31be7434e29c62e3e74159ea813372608e9b02244a13b0d matched merge.
- Anonymous capability/draft/reprocess: 401 each.
- 16:38:11Z–16:40:47Z app ERROR and HTTP 5xx: none.
- No agent audio upload, OpenAI call or clinical write. Actual model quality on the clinician's fictional recording remains unproven by the synthetic provider fixtures.

Handoff: save any currently retained audio locally before refreshing the old page. On the new page use Start/Stop; the summary appears automatically. Edit only when desired; explicit Save remains the persistence action. No further confirmation checkbox is required.
