# Priority 4 — local visit audio capture and source-faithful Visit Capture

> **DATE:** 2026-10-26 Asia/Nicosia.
> **STATUS:** DESIGN FROZEN FOR INDEPENDENT PRE-CODE R2; no runtime authority before PASS or corrected closure.
> **PRODUCT-OWNER DIRECTION:** recording, transcription and first-pass structuring stay on the clinician's Mac; no visit audio or raw transcript is sent to Render, OpenAI or another cloud AI service. The clinician must not have to read a twenty-minute transcript line by line.
> **OWNER:** the existing protected Visit Capture flow in `clinical_data.py` and `static/cockpit/visit-capture/`, extended by a loopback-only macOS companion.

## 1. Outcome and non-goals

After the clinician has selected and confirmed the destination patient, Visit Capture offers:

```text
patient informed + clinician presses Start
→ browser MediaRecorder captures the visit
→ clinician presses Stop
→ audio crosses only to a loopback-only local companion
→ whisper.cpp transcribes locally
→ a local LLM classifies source units by identifier
→ deterministic code copies the exact source units into a candidate
→ clinician reviews a compact structured view and explicit risk flags
→ protected server preview
→ explicit clinician Save
```

The slice also provides a deterministic, concise **Copy for GeSY** view derived from the reviewed candidate.

This slice does not:

- record in the background or start recording automatically;
- send audio or raw transcript to Render, OpenAI, Ollama Cloud or another remote service;
- retain audio, WAV, raw transcript, model prompt or model response after the local request finishes;
- infer that the recorded speaker is the selected patient;
- write an encounter, pending item or GeSY text without an explicit clinician action;
- generate a free-form clinical summary whose wording or numbers can become record truth;
- diagnose, recommend treatment, compare against a prior visit, code ICD-10, or complete missing clinical detail;
- fall back silently to a cloud transcription or summarization path.

## 2. Evidence behind the architecture

### 2.1 Local transcription on the target Mac

On the actual Apple M4 / 16 GB Mac, a synthetic 287.76-second mixed Greek/English clinical recording was tested with `whisper.cpp` at commit `231638a0717b7822784297e5980965ffcec05810` and `ggml-large-v3-turbo`:

- cold wall time: 28.22 seconds;
- warm wall time: 19.16 seconds;
- warm projected time for 20 minutes: approximately 80 seconds;
- measured peak memory: approximately 2.08 GB.

The same target Mac reported FileVault enabled during the design check. Runtime setup still verifies this condition instead of assuming it remains true.

MLX Whisper took 24.35 seconds warm for the same recording, projecting approximately 102 seconds for 20 minutes. The benchmark supports `whisper.cpp` with Metal as the default, but is not a guarantee for every real recording.

The synthetic sample showed good recognition of mixed terms such as `painful arc`, `Hawkins test`, `supraspinatus`, `partial thickness`, `rotator cuff`, `DEXA`, `T-score`, `femoral neck`, `calcium`, `vitamin D`, `creatinine` and `dental status`. A bounded initial prompt improved `alendronate 70 mg`. Ordinary Greek, names and some spoken dose wording still require review.

### 2.2 Free-form local summarization is unsafe for record truth

A local 4-bit 4B model was asked to form a structured visit summary from the same synthetic source. It produced mostly plausible sections but changed the source `T-score −2.8` into `−2.16` in its generated GeSY prose. A later exact-quote/classification experiment also produced an off-by-one sentence mapping.

Therefore JSON syntax, a low temperature and a good Greek model are not sufficient truth controls. The runtime must constrain the model to sentence identifiers and bounded labels, then validate and copy source text deterministically.

### 2.3 Browser-to-loopback connectivity

A production-page `fetch()` from the HTTPS Cockpit origin to `127.0.0.1` was blocked by the tested browser's loopback/private-network permission boundary, even with an explicit loopback target address space and CORS/PNA response headers. A direct production-to-loopback request would therefore depend on a browser permission and is not the default design.

The local companion instead owns a top-level launcher page on `http://127.0.0.1`. That local page opens the production Visit Capture page and uses a strict `postMessage` channel. The local page, not the production page, sends the audio to the same-origin loopback API. This avoids a public-origin-to-loopback network request.

## 3. Local components and fixed boundaries

### 3.1 macOS companion

The repository gains a bounded `local_visit_audio/` component. Its first reviewed deployment uses companion origin `http://127.0.0.1:8765`, dedicated Ollama endpoint `http://127.0.0.1:11435`, and production origin `https://ortho-reception-backend.onrender.com`. A port conflict or production-origin change fails setup and requires reviewed configuration rather than choosing an arbitrary origin/port. The component contains:

- a loopback HTTP service bound only to `127.0.0.1`, never `0.0.0.0`;
- a local launcher page;
- an install/check command for exact local dependencies;
- a run command and optional user LaunchAgent definition;
- deterministic transcript segmentation, validation and candidate construction;
- synthetic unit/integration fixtures with no real patient data.

Required local engines:

- `whisper.cpp` with Metal pinned to commit `231638a0717b7822784297e5980965ffcec05810` and `ggml-large-v3-turbo.bin` SHA-256 `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69` (1,624,555,275 bytes);
- a dedicated loopback Ollama process with `OLLAMA_NO_CLOUD=1`, request-body debug logging disabled, an explicit local model directory and no remote model identifier in the runtime allowlist;
- Ollama client/server version `0.40.11` for the first reviewed install;
- local alias `cockpit-krikri:priority4-v1`, created only from official ILSP `Llama-Krikri-8B-Instruct-GGUF` Q4_K_M blob SHA-256 `e3f93427fe0fcc016e31ed4bfbc1d10916fb97f6b2b3298445196490d315376f` (5.04 GB).

The default model is the pinned Krikri alias selected by the bounded comparison in section 11. There is no unreviewed model substitution. If the expected model/digest is unavailable, processing fails closed with setup guidance.

Whisper and the LLM run sequentially. The audio model is released before local LLM inference. Startup must prove that Ollama reports cloud disabled and that the pinned local model is present before recording is enabled. The Ollama request uses temperature `0`, a reviewed JSON schema, an explicit 16,384-token context setting, and `keep_alive: 0`.

Ollama's chat path may truncate over-context input and does not expose a supported preflight token-count endpoint. The companion therefore never submits a whole long transcript in one request. It deterministically partitions complete numbered source units into independently classified batches whose complete fixed instructions plus source text are at most `8,192` UTF-8 bytes, with `num_predict <= 1,024`. The pinned model/template must demonstrate no more than 1,024 additional template tokens in the installer self-test. Even at one token per input byte, this stays well below the 16,384-token context. The companion records and checks `prompt_eval_count`/`eval_count` after every batch. A unit too large for one batch, a response at the context boundary, or any batch failure rejects the result; no unit is split or silently truncated.

### 3.2 Production Cockpit

The existing Visit Capture page remains the patient-selection and protected-write owner. It gains:

- local-companion connection state;
- a required “Ο ασθενής ενημερώθηκε” control;
- Start, elapsed-time and Stop controls;
- progress states for transfer, transcription and local structuring;
- a compact structured review editor;
- a focused review panel for risk-bearing source units;
- an explicit assertion that the recording belongs to the confirmed destination patient;
- `Copy for GeSY` after server preview;
- the existing explicit Save as the only encounter/pending write.

The page never calls a Render audio/transcription endpoint. It does not send a raw transcript to any Render endpoint. Only the clinician-reviewed `VisitCaptureCandidateV1` is sent to the existing protected `/clinical/visit-capture/preview` and `/clinical/visit-capture/save` routes.

## 4. Local launcher handshake

The launcher generates a cryptographically random per-launch channel nonce and opens the exact allowlisted production Visit Capture URL in a new top-level window. The nonce is carried in the URL fragment, which is not sent to Render in the HTTP request. The production page reads it once and immediately removes the fragment with `history.replaceState` before beginning the handshake.

Both windows enforce all of the following before accepting a message:

- exact expected `event.origin`;
- exact `event.source` window identity;
- exact channel nonce;
- exact protocol version and message type;
- unique request identifier;
- monotonic state transition for one active recording/job.

After the origin/source/nonce bootstrap succeeds, the launcher transfers one `MessageChannel` port to the production child. The child clears `window.opener`; all later traffic uses only that dedicated port plus the channel nonce and request IDs. The audio `ArrayBuffer` is transferred, not cloned.

The production window sends the final `ArrayBuffer`, MIME type, measured duration and request ID to the verified local opener. The local opener performs a same-origin request to its loopback service and returns only the validated candidate and review flags. It never returns the raw transcript to the production window.

If `window.opener` is absent, the origin does not match, the nonce is wrong, or the companion is unavailable, local recording is disabled. There is no cloud fallback and no manual credential/token bridge.

## 5. Loopback service security

The local service:

- binds exclusively to `127.0.0.1` on the configured port;
- accepts only exact loopback `Host` values and rejects DNS-rebinding hosts;
- has no permissive CORS response;
- requires an HttpOnly, SameSite=Strict session cookie plus a per-page CSRF token;
- requires same-origin `Origin` and expected `Sec-Fetch-Site` for state-changing requests;
- accepts only the expected audio content types and one active job;
- enforces maximum duration, encoded byte size, decoded duration and WAV size;
- does not expose transcript, prompt, model output or audio through logs or errors;
- returns bounded error codes with a random request ID;
- rejects symlinks and writes only inside a newly created mode-0700 temporary directory;
- creates content files mode 0600 and deletes the complete temporary directory in `finally`, including on timeout, cancellation or model failure.

The companion process has no database credential, Clinical Key, Notion token or Render token.

## 6. Recording and retention contract

- Recording is available only in record mode after a protected patient context is confirmed.
- Start remains disabled until the clinician checks that the patient has been informed.
- The informed-patient checkbox is an immediate recording gate, not a claim of legal advice and not a durable consent registry in this slice.
- Start requires a direct user gesture and a granted microphone permission.
- A persistent red recording indicator and elapsed time are visible.
- Stop is explicit. Navigation, logout, patient change, lost opener, page hide/unload, media error and maximum duration stop all tracks and discard an incomplete recording.
- First implementation maximum duration: 30 minutes.
- First implementation maximum encoded upload: 64 MiB; the actual media duration is independently verified after decoding.
- No browser storage, service worker cache, IndexedDB, localStorage or sessionStorage holds audio, transcript, candidate or clinical flags.
- The browser releases the Blob/ArrayBuffer after local acknowledgement or failure.
- The local service retains content only for the duration of the request, unlinks every content path in `finally`, and verifies that those paths no longer exist before responding. This is bounded logical deletion; the design does not claim forensic secure erasure of SSD blocks. If FileVault/storage-at-rest protection is unavailable, setup warns and requires an explicit clinician decision before activation.
- Normal logs contain timing, model identifier/digest, byte counts and sanitized outcome codes only.

## 7. Source-faithful extraction contract

### 7.1 Deterministic source units

After Whisper, deterministic code normalizes only transport-level whitespace and produces numbered source units. It retains the original normalized text for each identifier. It must not normalize numbers, medication names, laterality, negation or clinical wording.

### 7.2 Model output

The local model receives numbered units and returns schema-constrained identifier lists only:

```json
{
  "reason_ids": [0],
  "history_ids": [1],
  "finding_ids": [2, 3],
  "impression_ids": [],
  "treatment_ids": [4],
  "medication_ids": [5],
  "referral_or_test_ids": [9],
  "pending_ids": [9],
  "follow_up_ids": [10],
  "safety_net_ids": [11],
  "uncertainty_ids": [],
  "administrative_ids": [],
  "unclassified_ids": []
}
```

These exact lists correspond to:

```text
reason | history | finding | impression | treatment | medication |
referral_or_test | pending | follow_up | safety_net | uncertainty |
administrative | unclassified
```

A source unit may appear in more than one list. The model has no field in which it can supply source wording, a diagnosis, a dose, a date or a free-form rationale.

The model is not asked to quote, summarize, translate, calculate, normalize or produce GeSY prose. The companion rejects:

- duplicate identifiers inside one list or out-of-range identifiers;
- unknown or missing schema keys;
- malformed schema output;
- a reason that is absent or cannot be selected without an explicit clinician correction.

After a valid response, deterministic code computes the union of all returned identifiers. Any source identifier omitted by the model is placed into `unclassified_ids` and the focused review panel; it is never silently dropped. Model output is not allowed to remove an identifier from the source inventory.

### 7.3 Deterministic draft construction

For every accepted identifier, the companion looks up and copies the exact stored source unit. It returns a strict `LocalVisitAudioResultV1` containing category-owned exact-text draft fields, separate pending proposals, review flags and no transcript. The production page applies clinician edits/selections and only then constructs `VisitCaptureCandidateV1` with:

- new source role `local_visit_audio` and an ephemeral session reference;
- `comparison_basis.type = unavailable` and no `what_changed` items;
- reason, findings, impression, decisions, pending, follow-up, safety-net and uncertainties made only from exact stored source text;
- treatments, medication statements and referrals represented as exact-text decisions with a category-owned fixed decision type, not as invented structured dose/action fields;
- coding empty and incomplete;
- only clinician-selected pending proposals; descriptions are copied exactly, with no inferred due date, external actor or trigger; default status `open` and responsible role `unspecified` unless the clinician edits them;
- no automatically populated structured medication object in this slice;
- `source_identity_state = uncertain` until the clinician explicitly confirms the recording belongs to the selected patient.

The server schema gains only the `local_visit_audio` source role and a deterministic `gesy_copy` preview field. Database schema is unchanged.

## 8. Compact clinician review

The clinician does not review the raw transcript by default. The production page shows:

- reason for visit;
- findings;
- clinical impression;
- treatment/medication/referral decisions;
- pending obligations/tests;
- follow-up;
- safety-net;
- uncertainties/unclassified content.

Every item remains editable and removable before server preview, with a one-control category correction instead of raw JSON editing. A proposed pending item has a separate unchecked “Δημιουργία εκκρεμότητας” control; the model's category alone can never create a durable pending record. The raw transcript is not available to the production page. A local-only expandable transcript may be shown in the launcher solely during the active result screen if a troubleshooting need is proven; it is excluded from first implementation.

The focused “Έλεγξε αυτά” panel includes any selected source unit containing:

- a digit, decimal, percentage, dose/frequency/time expression or measurement;
- a medication statement;
- laterality or anatomical side;
- negation or exclusion wording;
- uncertainty/conflict wording;
- an `unclassified` category;
- a Whisper low-confidence or no-speech anomaly when available.

The panel is an attention aid, not a claim that a flagged fact is wrong. The clinician must explicitly confirm patient identity and the reviewed candidate before Save. Save remains disabled after patient/context change, local-session change, validation failure or auth loss.

## 9. Deterministic Copy for GeSY

`gesy_copy` is rendered on the protected server from the same reviewed candidate, using fixed headings and exact candidate text. It includes only non-empty reason, findings/impression, decisions, pending/follow-up and safety-net sections. It performs no rewriting and introduces no new clinical language.

Copying is a clipboard action, not a database write and not proof that GeSY was updated. The UI states this explicitly.

## 10. Failure behavior

| Failure | Required behavior |
|---|---|
| microphone denied/unavailable | no recording; actionable local message |
| companion missing/handshake invalid | recording disabled; local setup guidance; no cloud fallback |
| audio too long/large/undecodable | delete temporary content; sanitized failure; no candidate |
| Whisper timeout/failure | delete temporary content; no partial transcript/candidate |
| context/token limit | reject before LLM inference; no truncation |
| Ollama/model missing or digest mismatch | fail closed; no alternate model |
| model output malformed | reject entire candidate; no partial salvage |
| valid model output omits source IDs | deterministically place every omitted ID in `unclassified`; never drop content |
| no safe reason for visit | request one short clinician correction before preview |
| patient/context/auth changes | invalidate recording/candidate and require new confirmation |
| Render preview rejects candidate | show sanitized validation error; Save disabled |
| local path cleanup cannot be verified | return privacy failure, expose only the local request ID and remediation, and retain no candidate in the production page; do not claim that the residual local file was erased |

## 11. Model selection result

The candidates were evaluated locally with Ollama `0.40.11` on the target M4 / 16 GB Mac against the same 15-unit synthetic/de-identified mixed Greek/English case. It included negation, laterality, dose, decimal result, completed versus planned action, follow-up, safety-net, multi-category units and a prompt-like sentence inside quoted patient speech.

### 11.1 Official ILSP Krikri Q4_K_M — selected with guardrails

The official 5.04 GB GGUF produced schema-valid per-category identifier lists in 15.37 seconds cold, including 4.09 seconds model load (`prompt_eval_count=905`, `eval_count=128`). It correctly identified the reason, explicit negative history, main examination/MRI findings, treatment, medications, future tests, follow-up and safety-net.

It was not perfect: one completed injection unit was omitted from all category lists, one completed DEXA result was also labelled as a referral/test, and a future pre-treatment check was not separately labelled pending. These are merge-blocking observations for an unguarded design, but they are contained by the frozen contract:

- every omitted ID is deterministically added to `unclassified` and shown in “Έλεγξε αυτά”;
- numbers/laterality/medication/negation are always flagged;
- classification changes only a proposed section, never source wording;
- no proposed pending record is durable unless the clinician separately selects it;
- the clinician can change a category with one control before protected preview/Save.

Krikri is therefore selected as a bounded classifier, not as a trusted summarizer or clinical reasoner.

### 11.2 Official Ollama `gemma4:12b` — rejected for this runtime

The current official default downloaded as the 7.7 GB MLX `nvfp4` 12.4B runner (`c7597fc90b86`). On the target Homebrew Ollama `0.40.11`, the structured-output request failed with HTTP 501 before inference. Ollama logged `Structured output is unavailable` because `libollama_xgrammar.dylib` was absent from the MLX runner path.

The design does not weaken the contract to prompt-only JSON. A separate 8.0 GB llama.cpp Gemma tag may be evaluated in a later reviewed model-change slice, but is not part of Priority 4 v1 and cannot be substituted at runtime.

### 11.3 Gate disposition

The selected model satisfies the v1 gate only together with deterministic complete-ID accounting, exact-source copying, risk flags, editable categories and explicit clinician write controls. Any future Krikri/Gemma/Ollama version or digest change reopens this gate and pre-code/post-code evidence as applicable.

## 12. Test and review gates

Before implementation:

- finalize the model benchmark and exact pinned identifiers/digests;
- verify the popup/opener handshake in the supported Dia/Chromium runtime;
- independent exact-head pre-code R2 review of privacy, identity, source fidelity, retention, local-network security and clinical write boundaries;
- explicit PASS or corrected closure before runtime code.

Implementation evidence must include:

- unit tests for source segmentation, full ID coverage, exact-copy construction, risk flags and candidate mapping;
- adversarial tests for number changes, hallucinated IDs, duplicates, omissions, negation, laterality, medication wording, mixed Greek/English and malicious prompt-like transcript text;
- loopback Host/Origin/CSRF/content-type/size/symlink/concurrency tests;
- timeout/cancellation/deletion tests proving content cleanup;
- browser tests for informed-patient gate, Start/Stop, visible recording state, popup handshake, patient switch/auth loss invalidation, compact review, GeSY copy and explicit Save;
- a synthetic local end-to-end audio test with network observation proving no audio/transcript request reaches Render or another remote host;
- existing Visit Capture focused tests and applicable full regression suites;
- independent exact-head post-code R2 before merge/release.

Release requires guarded PR/CI, reviewed merge, Render static/backend verification, no-write production smoke and a local synthetic companion smoke. Agents must not use real patient audio. The first real, informed-patient recording is a Product Owner acceptance step after technical release.

## 13. Replan triggers

Reopen design review if any of the following is required:

- direct HTTPS production-origin access to loopback instead of the launcher channel;
- persistence of audio or transcript beyond the active request;
- cloud transcription, cloud summarization or remote telemetry containing visit content;
- a model-generated free-form summary becoming candidate or GeSY truth;
- automatic identity acceptance, automatic Save, automatic GeSY submission or background recording;
- a different model/digest, context/truncation behavior, or material schema change;
- inability to prove deletion or to keep all source units accounted for.
