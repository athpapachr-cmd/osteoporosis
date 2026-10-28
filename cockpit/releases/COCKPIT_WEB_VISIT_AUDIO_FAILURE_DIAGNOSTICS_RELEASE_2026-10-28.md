# Web visit audio failure diagnostics — production release receipt

2026-10-28, Asia/Nicosia. This records a diagnostic correction, not a successful clinical recording or a determination of the original failure.

## Incident and reviewed change

On the synthetic chart, protected context renewal returned 200 and `/clinical/visit-capture/audio/draft` returned 502 at 07:20:04 UTC. The original route discarded its temporary audio and returned one generic message for every internal processing failure, so the exact cause cannot be recovered from that request. Render had no service restart in the incident window. Local synthetic transcription and classification requests returned 200 without the production project header; this does not verify the Render key/project association.

[PR #238](https://github.com/athpapachr-cmd/osteoporosis/pull/238) adds one private warning with a fixed pipeline stage, fixed failure kind and provider HTTP status when applicable. It does not log audio, transcript, clinical text, patient/context identifiers, credentials, provider body or exception text. The public 502, fail-closed validation, temporary-file cleanup, clinician review and explicit Save remain unchanged. Eight focused local synthetic tests passed. Independent Security Review and the Product Owner-selected Bugbot found no issue. Final PR head `d08934ef299506bbc760788c51b5e3209232237a` passed all 10 GitHub checks. One intermediate docs-head Visit Capture run had 45 tests pass and one inherited appointment browser timeout; one failed-job rerun passed, and the final head Visit Capture run passed directly.

## Deploy and no-PHI smoke

PR #238 squash merged as `803f82ebb1a1395a34e19764dc688199c455b482`. Native Render deploy `dep-dbgqpcp5efls73au76s0` reached LIVE at 08:07:34.338319 UTC on that exact commit. No manual duplicate deployment or environment-variable change occurred.

Served `static/cockpit/patient-chart.js` and `static/cockpit/index.html` SHA-256 values matched the merged source. Anonymous capability, audio draft, context renewal, preview and save routes all returned 401. The 08:05–08:10 UTC application-log window contained no app ERROR, HTTP 5xx or audio-diagnostic failure line. No agent recording, provider request or clinical write was made after deployment.

## Handoff

The original 502 cause remains **UNKNOWN** because no stage metadata existed for that attempt and the audio was not retained. A clinician-owned short recording with fictional content in the synthetic chart can now exercise the live route. If it fails, the private Render log can distinguish transcription, classification, source-unit or candidate failure without exposing clinical content. Do not Save a synthetic draft. Real-patient readiness and clinical quality are separate from this diagnostic release.
