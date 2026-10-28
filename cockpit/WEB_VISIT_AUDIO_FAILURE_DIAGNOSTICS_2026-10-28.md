# Web visit audio failure diagnostics — 2026-10-28

## Observed failure

The clinician's synthetic chart showed the generic recording-processing error. The protected context renewal returned 200 and `/clinical/visit-capture/audio/draft` returned 502 at 2026-10-28 07:20:04 UTC. The current endpoint collapses provider HTTP rejection, network failure, invalid model output and candidate validation into the same 502 without a stage marker. Render reported no service restart in the incident window. The recording is transient and unavailable for replay. Local synthetic requests without the production project header returned 200 for both provider endpoints; they do not prove the Render key/project association.

## Bounded correction

Keep the clinician-facing 502 text and all clinical authority unchanged. On an audio draft failure, emit one private server log record containing only a fixed pipeline stage (`transcription`, `source_units`, `classification`, `candidate` or `pipeline`), a fixed failure kind (`provider_http`, `timeout`, `network`, `invalid_output`, `disconnected` or `unexpected`) and, for a provider HTTP rejection, its three-digit HTTP status. Never log exception text, provider response body, request headers, audio, transcript, source units, identity, candidate, patient/context ID or credentials. Keep the existing fail-closed response and temporary-file cleanup.

## Evidence and release

Synthetic tests must prove that provider HTTP rejection yields the bounded private fields and generic public 502; unexpected exceptions with patient-like text must not appear in the log or response. Existing cleanup and valid-draft tests remain applicable. This is an R2 privacy-boundary correction. Independent review, exact-head CI and a guarded release are required before asking the clinician for a new synthetic recording. A future 502 can then identify the failing stage without creating a clinical write or retaining audio.
