# Web visit audio recovery and contextual review — production release receipt

2026-10-28, Asia/Nicosia. This records a production release verified with synthetic tests and a no-PHI smoke. It does not establish real-patient clinical quality and contains no recording, provider request or clinical write.

## Product correction

The Product Owner identified three defects in fictional use: the review became a very long form with unnecessary confirmations; clinically important symptoms, treatment, plan and obligations could be omitted; and a successful transcript could be lost when structured processing failed. The Product Owner also requested bounded suggestions for the next visit based on the combined context rather than only verbatim spoken instructions.

[PR #247](https://github.com/athpapachr-cmd/osteoporosis/pull/247) preserves the exact unverified transcript after successful transcription even when structuring fails, retains the original audio Blob only in browser memory until success or discard, and allows one protected transcript-only retry without retranscription. A local audio download remains available after failure. The transcript and processed clinical text are compact controls, both closed by default and rendered only after the clinician selects one; a transcript-only result cannot Preview or Save.

The same release adds display-only contextual clinical considerations over all validated visit facts. They are visibly labelled as potentially wrong AI reasoning, remain outside the candidate, Preview and Save, and have no adoption control. Existing typed next-visit suggestions remain separate and enter the editable plan only through the clinician's explicit bulk action. The final patient/content confirmation, Preview and explicit Save remain the sole clinical authority path.

Models and privacy settings are unchanged: `gpt-transcribe` performs transcription and `gpt-6-luna` performs structured analysis and the optional whole-visit consideration pass under the approved standard-retention disclosure, explicit project routing, `store:false`, no-tools rule and temporary-content lifecycle.

## Review, correction and migration

The no-loss backend passed corrected pre-code and exact-code R2 with P0:P1:P2 = 0:0:0. The collapsed presentation passed exact-code R1 with 0:0:0. PR #247 final release head passed all 15 reported checks and squash-merged as `ef4d633ab9796b5efe29932491133d02eab2361b`.

Native Render deploy `dep-dbh09nuq1p3s73bg835g` built successfully but failed before cutover with PostgreSQL `42501`: application startup attempted to create `clinical_visit_audio_recovery` while `cockpit_app` correctly lacked CREATE on `public`. The preceding deploy remained LIVE, so there was no production outage or partial activation. Schema rights were not widened.

[PR #248](https://github.com/athpapachr-cmd/osteoporosis/pull/248) removes application-startup DDL and moves the opaque recovery table to a one-time administrator migration. Exact reviewed SQL SHA-256 `5b29d4d1d13ead6c7d0dd376a0a622fad5da1f6c668adeba8292a294d8787863` was installed over verified TLS after preflight proved the table absent. It created only `id`, `state`, `expires_at`, a primary key and state/expiry indexes. The table was empty. `cockpit_app` has SELECT/INSERT/DELETE and only `UPDATE(state)`; id/expiry UPDATE, TRUNCATE and schema CREATE are false. No patient table or row was read. PR #248 final head `63f2b7b1` passed all 10 checks and squash-merged as `3500cb49f27a7c5a89641a6bb9ce4bb9488bcae8`.

## Deploy and no-PHI smoke

Render automatically deployed exact corrective merge `3500cb49f27a7c5a89641a6bb9ce4bb9488bcae8` once as `dep-dbh0muivcj2c73e00gj0`, LIVE at 2026-10-28T14:51:34.150398Z. No duplicate manual deploy, restart, environment-variable change, secret rotation, scaling or network-rule change occurred.

Served assets matched merged source SHA-256:

- `static/cockpit/index.html`: `a8078b89b1a9c340d783abb8004b089c9941f95be37c903762cbb2852110d031`
- `static/cockpit/patient-chart.js`: `d7ec9f8dabb0c433aa9737fc4a2232cfcf6ff027cfc33aeb17499556093a5e53`
- `static/cockpit/doctor-shell.css`: `990224485d8120c70609566c70b56151d22acf947a7db9501277d127a2ade686`

Anonymous capability, audio-draft and transcript-reprocess routes each returned HTTP 401. From merge/deploy creation at 14:49:28Z through the post-LIVE smoke at 14:53:04Z, Render reported no app ERROR and no HTTP 5xx. No audio was recorded or uploaded, no OpenAI request was sent, and no clinical Save was performed.

## Handoff

The technical release is complete. The next Product Owner step is one fictional recording without Save. Confirm that both content views start closed; open the transcript only when needed; verify the structured sections preserve symptoms, therapy, plan, obligations and follow-up; and assess whether the separate contextual considerations and next-visit suggestions are useful. If structuring fails after transcription, the transcript, retry action and local audio download must remain available. Real-patient use and clinical quality acceptance remain clinician decisions under the existing consent and standard-retention notice.
