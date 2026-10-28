# Web visit audio semantic completeness — production release receipt

2026-10-28, Asia/Nicosia. This records a production correction verified with synthetic tests and a no-PHI release smoke. It does not establish real-patient clinical quality and contains no clinical write.

## Product correction

The Product Owner's fictional comparison showed that the previous compact draft lost clinically material clauses from compound speech: symptoms, treatment, decisions, explicit obligations and follow-up could be missing or routed to uncertainties. It also could not suggest bounded next-visit checks derived from what was said.

[PR #245](https://github.com/athpapachr-cmd/osteoporosis/pull/245) replaces whole-unit single-category classification with exact-source, non-overlapping multi-fact extraction. The server deterministically maps validated excerpts into the existing structured sections and fails closed on ambiguous, overlapping or incompletely accounted source text. It does not infer a medication object, diagnosis, code, date, responsible person or completed status.

The browser now shows compact audio-only groups for history and symptoms, findings, treatment and recommendations, decisions and plan, next steps and obligations, and follow-up. A separate `Προτάσεις AI για την επόμενη επίσκεψη` block can contain at most six server-rendered, non-prescriptive checks selected from bounded typed intents. Suggestions are excluded from the candidate, Preview and Save unless the clinician uses one bulk `Προσθήκη στο πλάνο` action. The single final patient/content confirmation, Preview and explicit Save remain.

Models and privacy settings are unchanged: `gpt-transcribe` performs transcription and `gpt-6-luna` performs the structured analysis under the existing standard-retention disclosure/gate, explicit project routing, `store:false`, no-tools rule and temporary-content lifecycle.

## Review and CI

Independent corrected pre-code and post-code R2 closures each passed with no open findings. The corrected contract rejects provider-authored suggestion prose and accepts only typed intent plus compatible cited source ID; the server owns the fixed Greek wording.

PR #245 final head passed all 16 exact-head checks. Both Visit Capture Chromium jobs completed their full suites, including exact-source multi-clause mapping, compact pending/follow-up display, one final confirmation, suggestion separation and clinician bulk adoption into the Save payload. Focused non-browser tests, syntax and diff hygiene also passed.

## Deploy and no-PHI smoke

PR #245 squash-merged as `d4e66a0147771834e27b1f6d1639ecaae1909f8e`. The existing Render service created one automatic deploy, `dep-dbgt23ijnfac739m9ipg`, which reached LIVE at 10:42:36.804727 UTC on that exact merge. No manual duplicate deploy, restart, environment-variable change, secret change or scaling action occurred.

Served assets matched merged source SHA-256:

- `static/cockpit/index.html`: `b9c642ccd815af348755f23523972922a7c02272adaa4b928fcf64cbae07864d`
- `static/cockpit/patient-chart.js`: `106df241ab059aa12e6e6c4d97ad41726ed5c992a14883d62b56b283ed77fbef`
- `static/cockpit/doctor-shell.css`: `bcc12a991055fc00447680625c0487f53ccc6f4cbffe9f28a37ba62a2ace259a`

Anonymous capability, context, audio-draft, preview and save routes each returned HTTP 401. From deploy creation at 10:40:14Z through the post-LIVE smoke, Render reported no app ERROR and no HTTP 5xx. No audio, provider request or clinical write was made.

## Handoff

The correction is LIVE. The next Product Owner step is one short fictional recording without Save. Check that the draft preserves the spoken symptoms, treatment, plan, obligations and follow-up, and that the separate next-visit suggestion block is useful. Delete any unwanted suggestion lines; use the one bulk action only if they should enter the editable plan. There should be no per-item confirmation sequence and only one final patient/content confirmation.
