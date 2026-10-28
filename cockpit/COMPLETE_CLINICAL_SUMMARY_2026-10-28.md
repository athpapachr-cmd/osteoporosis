# Complete clinical summary projection

User acceptance: the final chart-style result must show prior wrist fracture two years ago, thoracic pain duration three months, T-score -2.8, and recommendation to start alendronate after dental assessment.

Source-proven omission: patient-chart.js overview renders only reason/impression/decisions; history/findings/recommendations are excluded. Saved visit reader also omits history/recommendations/follow_up. The API returns the full signed candidate unchanged.

R1: extend read-only projections of existing validated candidate fields; no inference, model, backend/schema/write change. Render every populated clinical section without truncation in latest card and visit reader, reuse exact text and existing medication/next-contact formatting. Keep obligations with their true lifecycle in the adjacent card. Existing older visits remain separate.

For new manual/pasted drafts use the same chart-style read projection as automatic audio summary, with an optional Edit button opening the unchanged full editor. Do not auto-save; existing explicit Save remains. Amendments retain current UI. Unresolved identity stays blocked. Audio no-loss/candidate authority/late-result guards remain. Display draft obligations as proposed, not completed or already persisted.

Evidence: exact fictional four-fact fixture -> extract candidate -> automatic draft card -> explicit Save -> actual DB chart -> reload/independent session, desktop/mobile. Verify all four facts in visible latest card and historical reader, recommendation stays recommendation, no invented treatment result or extra Save. Also affected audio/manual/amendment tests and R1 fidelity review. Add read-only live provider qualification using only fictional text on a never-merge verification branch if available; report separately from deterministic projection proof.
