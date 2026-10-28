# Web Visit Audio transcript recovery and exact failure diagnostics — 2026-10-28

## Product Owner checkpoint

**Problem now:** a recording can be transcribed successfully and then fail during structured processing. The released route returns one generic 502 and the browser no longer has either the transcript or the recorded Blob. A whole visit can therefore be lost.

**What the clinician sees now:** only `Η ηχογράφηση δεν επεξεργάστηκε. Δοκίμασε ξανά.`

**What changes in this step:** the active review has two explicit views: `Μη επαληθευμένη μεταγραφή` and `Επεξεργασμένο κλινικό κείμενο`. Once transcription succeeds, the transcript is returned to the same protected browser request even if structuring fails. The clinician can retry structuring from that transcript without recording again. Until a structured result succeeds or the clinician discards it, the original audio Blob remains only in browser memory and can be downloaded locally after failure.

**Concrete example:** transcription succeeds, but `gpt-6-luna` returns a structurally invalid classification. The page shows the complete unverified transcript, says that structured processing failed, exposes `Επανάληψη επεξεργασίας`, and keeps `Αποθήκευση ήχου στο Mac`. No chart Save is available until a valid structured candidate exists.

**Outside this step:** no server-side audio/transcript persistence, browser storage, automatic chart write, automatic adoption of transcript prose, new clinical authority, real-patient validation, merge or deployment. The contextual-consideration and typed-suggestion authority boundaries remain unchanged.

**Why now:** this is a critical safety/data-integrity correction and precedes release of PR #247 or any real visit use.

The Product Owner explicitly confirmed the two-result direction and required investigation of the failed processing. The selected provider mode remains standard OpenAI retention with `store:false` for Responses.

## Production evidence and bounded conclusion

Two fictional recording attempts produced these private Render records:

```text
2026-10-28T12:33:24.322470288Z stage=classification kind=invalid_output provider_http_status=None
2026-10-28T12:36:15.068259471Z stage=classification kind=invalid_output provider_http_status=None
```

Observed conclusion: transcription completed and the failure occurred during structured classification. The provider did not reject either request with an HTTP error. The released diagnostic format did not include the deterministic validator subtype, so the exact historical subtype cannot be recovered. It must remain `UNKNOWN within classification invalid_output`; guessing a schema/identity/accounting subtype is forbidden.

## Reuse decision

**EXTEND** the existing protected audio route, provider clients, patient-context binding, exact-source classifier, candidate builder, standard-retention gate, Preview and explicit Save. Do not create a second recorder, clinical writer, provider owner or persistence store.

A new protected transcript-reprocess route is an adjacent transport entry to the same classification/candidate function. It exists only so a transcript already returned to the clinician can be resubmitted without retranscription.

## Server contract

### Shared classification function

Extract the current `source_units → batch classification → candidate → optional considerations` stages into one shared async function. It accepts the transcript, selected identity, patient/context/date binding, deadlines and existing summarizer provider. It rechecks the protected context before classification, before candidate authority and before response. It returns the existing successful candidate payload unchanged.

### Audio draft response

`POST /clinical/visit-capture/audio/draft` retains its current auth, media limits, format checks, standard-retention acknowledgement and temporary server-audio cleanup.

After transcription succeeds, the route owns an ephemeral `transcript` string. Classification/candidate failure no longer discards it. The route returns HTTP 200 with exactly one state:

```text
processing_status: complete | transcript_only
context_id
patient_id
transcript
processing_failure: null | {
  stage: source_units | classification | candidate | pipeline
  code: fixed allowlisted code
  provider_http_status: null | 100..599
}
```

`complete` additionally returns the existing `candidate`, `review_units`, `suggested_next_checks`, contextual status/items and rendered detail. `transcript_only` returns none of those authority-bearing fields. Transcription failure remains a sanitized non-200 response because no transcript exists to recover.

No response includes prompts, provider bodies, exception text, credentials or patient identifiers beyond the already required protected `patient_id/context_id` binding.

### Transcript reprocessing

Add `POST /clinical/visit-capture/audio/reprocess` under the same protected dependency. JSON is exact-key and bounded:

```text
context_id: existing protected context
encounter_date: ISO date
transcript: 10..120000 characters
```

The request requires `X-Visit-Retention-Acknowledged: standard`, binds the same patient context and sends the transcript only through the existing `gpt-6-luna` classification/contextual path. It never calls transcription. It returns the same `complete | transcript_only` envelope, allowing repeat classification failure without losing the transcript.

### Safe diagnostic code

Add a fixed mapping from stage/exception type to an allowlisted code. It may include existing deterministic codes such as `classification_keys`, `classification_identity`, `classification_values`, `classification_excerpt`, `classification_overlap`, `classification_accounting`, `classification_suggestions`, plus fixed provider/timeout/network/identity/source-unit/candidate codes. JSON parse messages or arbitrary exception strings never enter logs or responses.

Every final processing failure emits only:

```text
visit_audio_processing_failure stage=<fixed> kind=<fixed>
provider_http_status=<integer-or-None> validation_code=<fixed-or-None>
```

No audio, transcript, source excerpt, candidate, patient/context identity, provider body or exception text is logged.

## Browser lifecycle and UX

- After `MediaRecorder.stop`, retain the Blob in one in-memory variable before upload.
- A successful `complete` result clears/revokes the Blob; the transcript remains only for the active two-view review.
- A `transcript_only` result keeps both transcript and Blob in memory.
- A pre-transcription failure keeps the Blob and offers `Επανάληψη αποστολής ήχου`.
- A post-transcription failure offers `Επανάληψη επεξεργασίας`, using the transcript endpoint without retranscription.
- On failure only, offer `Αποθήκευση ήχου στο Mac` using a temporary object URL. Revoke it after the click completes or when the draft is discarded.
- `Απόρριψη προσωρινών δεδομένων` clears Blob, object URL, transcript and partial state.
- Closing/reloading the page still destroys in-memory content. The existing unload guard must include retained Blob/transcript and the failure UI must state this plainly.
- Never use localStorage, sessionStorage, IndexedDB, Cache API, server files beyond the request, database or logs for audio/transcript.

The transcript view is labelled `Μη επαληθευμένη μεταγραφή` and states that ASR may contain errors and nothing in it is saved automatically. The structured view is separate and remains editable. No additional item-by-item confirmations are introduced. Existing informed-recording consent and one final patient/content confirmation remain the only confirmations.

The transcript never enters Preview or Save. Only a successfully validated structured candidate can enable the existing Preview/explicit Save path.

## Retention and authority

Returning the transcript to the authenticated browser does not change provider retention: the audio/transcript already crossed the approved standard-retention OpenAI boundary. Each reprocess call is a new provider submission under the same disclosure. The page must not imply deletion from OpenAI when local transient data are discarded.

This explicit recovery view supersedes the earlier ordinary-UX preference to hide raw transcripts only for the active failed/successful recording review. Ephemeral-by-default and non-authoritative status remain unchanged.

## Required evidence

1. Classification validator failure after successful transcription returns `transcript_only`, exact transcript and a fixed code, with no candidate/Preview/Save authority.
2. Classification provider HTTP, timeout, network and malformed JSON paths return transcript recovery and content-free logs.
3. Transcription failure returns no invented transcript; browser retains the original Blob and can retry/download it.
4. Reprocess calls classification but never transcription and can transition `transcript_only → complete` for the same context/date/transcript.
5. Patient/context/date change fails closed and does not expose the transcript to a different chart.
6. Browser shows two distinct views, retains Blob on failure, downloads locally, clears all transient state on discard and does not use browser persistence.
7. Save remains disabled for transcript-only state. The transcript and contextual AI prose are absent from Preview/Save payloads.
8. Existing exact-source candidate, unresolved identity fail-closed behavior, one bounded structural retry, typed suggestion adoption, contextual display-only layer, standard-retention disclosure and one final confirmation do not regress.

## Review and release

R2 because this changes identifiable-transcript exposure/lifecycle and clinical failure behavior. One independent pre-code review is required on this exact design. After implementation and affected tests, one independent exact-code R2 is required. PR exact-head CI, merge, Render deploy and no-PHI production smoke remain separate gates. No agent real recording, provider replay or clinical write.
