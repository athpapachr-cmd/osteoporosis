# Priority 4 — local visit audio capture and source-faithful Visit Capture

> **DATE:** 2026-10-26 Asia/Nicosia.
> **STATUS:** FIRST PRE-CODE R2 `0:5:0`; first closure closed P1-01/P1-03/P1-04/P1-05 and left only P1-02 context renewal open (`0:1:0`). Atomic exact-prior-context renewal/CAS correction applied for focused independent closure. No runtime authority before closure PASS.
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

- Homebrew FFmpeg `9.0.2_1`, official Apple-silicon bottle SHA-256 `d521452157b1117f909f441ce90165c360a7dcd9785c87d14cabc28934572a90`, with resolved `ffmpeg` executable SHA-256 `7f6028378534aa10a2dfa07d807ddcc0d18c869bf5c2f64c89efcc18ee88c4ee`; the installer also freezes and verifies the Homebrew receipt/SBOM and resolved dynamic-library manifest before activation;
- `whisper.cpp` with Metal pinned to commit `231638a0717b7822784297e5980965ffcec05810` and `ggml-large-v3-turbo.bin` SHA-256 `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69` (1,624,555,275 bytes);
- a dedicated loopback Ollama process with `OLLAMA_NO_CLOUD=1`, request-body debug logging disabled, an explicit local model directory and no remote model identifier in the runtime allowlist;
- Ollama client/server version `0.40.11` for the first reviewed install;
- local alias `cockpit-krikri:priority4-v1`, created only from official ILSP `Llama-Krikri-8B-Instruct-GGUF` Q4_K_M blob SHA-256 `e3f93427fe0fcc016e31ed4bfbc1d10916fb97f6b2b3298445196490d315376f` (5.04 GB).

The default model is the pinned Krikri alias selected by the bounded comparison in section 11. There is no unreviewed model substitution. If the expected model/digest is unavailable, processing fails closed with setup guidance.

FFmpeg is the only audio decoder in v1. The companion passes only two already-created mode-0600 paths beneath the request's mode-0700 temporary directory; no user-controlled URL, protocol or output name reaches the command. The exact argument vector, except for those two validated paths, is:

```text
/opt/homebrew/Cellar/ffmpeg/9.0.2_1/bin/ffmpeg
-nostdin -hide_banner -nostats -loglevel error -xerror
-protocol_whitelist file -max_alloc 67108864 -threads 1
-i <input-path> -map 0:a:0 -map_metadata -1 -map_chapters -1
-vn -sn -dn -t 1800.25 -ac 1 -ar 16000 -c:a pcm_s16le
-f wav -y <output-path>
```

The process receives a minimal environment and operating-system limits of 120 CPU seconds, 120 wall-clock seconds, 2 GiB address space and 64 MiB output file. Stdout is discarded; at most 8 KiB of stderr is retained only in memory for a sanitized error code and is never logged. Standard-library WAV parsing independently requires mono, 16 kHz, 16-bit PCM, a complete header/frame count, duration no greater than exactly 1,800 seconds and total size no greater than 64 MiB. Decoding to 1,800.25 seconds ensures an input just over the ceiling is observed and rejected instead of accepted after clipping. The version-controlled decoder manifest in the implementation must contain the exact Homebrew receipt/SBOM identity and SHA-256 of the resolved executable and every non-system dynamic library; setup compares the full manifest and refuses activation on any missing, extra or changed entry. Decoder identity mismatch, malformed/bomb-like input, limit/timeout, unexpected extra stream behavior or truncated WAV fails the request and runs the same cleanup path. No `ffprobe` or alternate decoder fallback is permitted.

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

Protocol `local_visit_audio_v1` has exactly one bootstrap and one job. The launcher generates a 32-byte cryptographically random base64url nonce and a separate 16-byte base64url `local_session_id`, stores them only in page memory, and opens only the allowlisted production Visit Capture URL with fragment `#local_visit_audio_v1=<nonce>`. The fragment is not sent to Render. The child reads it once, rejects malformed/duplicate parameters and immediately removes the full fragment with `history.replaceState` before sending any message.

The only window messages are:

1. child → exact opener, exact launcher origin: `{"protocol":"local_visit_audio_v1","type":"bootstrap_ready","nonce":"<nonce>"}`;
2. opener → exact child, exact production origin: `{"protocol":"local_visit_audio_v1","type":"bootstrap_port","nonce":"<nonce>","local_session_id":"<id>"}` with exactly one newly created `MessagePort` transferred.

Each side requires the exact expected `event.origin`, exact `event.source` window identity, exact key set and bounded string lengths. The opener accepts `bootstrap_ready` once, atomically consumes the nonce and removes its window-message listener before transferring the port. The child accepts `bootstrap_port` once, verifies the nonce, transferred-port count and session ID, removes its listener and clears `window.opener`. A second bootstrap, extra/unknown field, wrong origin/source/nonce or missing/multiple port closes any supplied port and permanently disables local audio in that child.

`companion_ready` has the exact envelope `protocol`, `type`, `nonce`, `local_session_id` and an empty payload. Every later port message additionally requires `request_id` plus its type-specific payload; unknown or missing fields reject the message. `request_id` is a fresh 16-byte base64url value generated by the child for the single job. The allowed monotonic state machine and directions are:

```text
opener → child  companion_ready
child  → opener job_start       (one transferred ArrayBuffer + metadata)
opener → child  job_progress     (bounded enum only, zero or more)
child  → opener job_cancel       (optional, terminal request by child)
opener → child  job_result | job_error (exactly one terminal response)
child  → opener job_ack          (exactly one terminal acknowledgement)
either side closes the port after job_ack or terminal timeout
```

The nonce is exactly 43 base64url characters; session/request IDs are exactly 22 base64url characters. `job_start` payload has only `mime_type` literal `audio/webm;codecs=opus`, integer `duration_ms` in `1..1800000`, integer `byte_length` in `1..67108864`, and exactly one transferred `ArrayBuffer` whose actual length equals `byte_length`. Unsupported browser recording formats disable v1 rather than selecting another decoder path. `job_progress` has only enum `transferring|decoding|transcribing|classifying|validating|cleaning`. `job_cancel` and `job_ack` have empty payloads. `job_error` has only `code` from `filevault_unavailable|invalid_request|audio_limit|decode_failed|transcription_failed|classification_failed|validation_failed|timeout|cancelled|cleanup_failed|internal_failure` and the matching random local `request_id`; it contains no free text from tools. `job_result` has exactly one `LocalVisitAudioResultV1` from section 7.3. The complete serialized non-audio result is limited to 1 MiB, 512 items and the per-field limits in that schema.

Every inbound port message must match the stored nonce, local session and active request. A stale/duplicate/unknown request, second `job_start`, extra transferred object, wrong direction or non-monotonic transition is terminal. Bootstrap timeout is 10 seconds; upload acknowledgement is 15 seconds; local processing timeout is 10 minutes after Stop; terminal acknowledgement is 5 seconds. Timeout, child close/navigation, port close/message error or `job_cancel` aborts local subprocesses, discards the browser buffer, runs verified cleanup, returns no candidate and closes the port. A late result is ignored and destroyed locally. There is no reconnect/retry on the same nonce or request; the clinician must start a fresh launcher session.

The local opener performs the audio POST to its same-origin loopback service and returns only the validated structured item inventory and review flags—never Whisper JSON, timestamps, prompt, model response or an unstructured transcript. The audio `ArrayBuffer` is transferred, never cloned. If any bootstrap or lifecycle check fails, recording is disabled. There is no cloud fallback or manual credential/token bridge.

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
- The local service retains content only for the duration of the request, unlinks every content path in `finally`, and verifies that those paths no longer exist before responding. This is bounded logical deletion; the design does not claim forensic secure erasure of SSD blocks.
- Priority 4 v1 fails closed unless setup, every service start and every new recording job positively verify FileVault/storage-at-rest protection. Recording controls remain disabled on an unknown, encrypting/decrypting, disabled or unverifiable state. There is no clinician override in v1; any override requires a separate explicit Product Owner/privacy design decision and new review.
- Normal logs contain timing, model identifier/digest, byte counts and sanitized outcome codes only.

The 30-minute recording ceiling can outlive the current server context, which expires 30 minutes after patient confirmation. The original context is therefore never reused for final preview. Immediately before the first protected preview—and again after any later context expiry—the production page calls a new protected `POST /clinical/visit-capture/context/renew` CAS endpoint with the exact closed body `{"prior_context_id":"...","patient_id":"...","client_session_id":"..."}`. All three strings retain the existing context field limits; unknown fields are forbidden.

In one database transaction, renewal locks and loads the exact prior context row by ID even if its lease has expired, verifies its stored client session and patient equal the request, locks the current patient row, and compares the prior context's stored `patient_updated_at` to the current patient `updated_at`. Only after every check passes does it delete all older contexts for that client session and issue one fresh 30-minute context with the same patient and current matching version. Missing prior row, session/patient mismatch or patient-version mismatch returns 409 without deleting a context or issuing a replacement. The existing ordinary context-create endpoint is not a renewal path. No database schema change is required.

The browser requests renewal only while the authenticated session, selected patient, original context ID, local session ID, local request ID and clinician's explicit source-identity confirmation are unchanged. It replaces the original context ID only with the successful CAS response. Auth loss, 401/409, patient/version mutation, local-session/request change or renewal failure destroys the candidate and requires a new patient confirmation; it never changes the candidate's destination in response to renewal. Preview/Save use only the newest returned context ID, so a stale old context cannot authorize a write.

## 7. Source-faithful extraction contract

### 7.1 Deterministic source units

After Whisper, deterministic code normalizes only transport-level whitespace and produces numbered source units. It retains the original normalized text for each identifier. It must not normalize numbers, medication names, laterality, negation or clinical wording.

### 7.2 Model output

The local model receives numbered units and returns schema-constrained identifier lists only:

```json
{
  "reason_ids": [0],
  "patient_history_ids": [1],
  "objective_result_ids": [2, 3],
  "clinician_interpretation_ids": [],
  "option_discussed_ids": [],
  "clinician_recommendation_ids": [4],
  "final_decision_ids": [5],
  "completed_action_ids": [],
  "patient_accepted_ids": [],
  "patient_declined_ids": [],
  "patient_undecided_ids": [],
  "follow_up_ids": [9],
  "pending_obligation_ids": [9],
  "safety_net_ids": [11],
  "uncertainty_ids": [],
  "administrative_ids": [],
  "unclassified_ids": []
}
```

These exact lists correspond to:

```text
reason | patient_history | objective_result | clinician_interpretation |
option_discussed | clinician_recommendation | final_decision |
completed_action | patient_accepted | patient_declined | patient_undecided |
follow_up | pending_obligation | safety_net | uncertainty |
administrative | unclassified
```

A source unit may appear in more than one list. The model has no field in which it can supply source wording, a diagnosis, a dose, a date or a free-form rationale.

The model is not asked to identify treatment type, medication structure, decision type, pending item type, quote, summarize, translate, calculate, normalize or produce GeSY prose. Transcript content is untrusted data even when it contains instructions addressed to the model; the fixed prompt explicitly delimits it as non-instruction source material. The companion rejects:

- duplicate identifiers inside one list or out-of-range identifiers;
- unknown or missing schema keys;
- malformed schema output;
- a reason that is absent or cannot be selected without an explicit clinician correction.

After a valid response, deterministic code computes the union of all returned identifiers. Any source identifier omitted by the model is placed into `unclassified_ids` and the focused review panel; it is never silently dropped. Model output is not allowed to remove an identifier from the source inventory.

### 7.3 Deterministic draft construction

For every identifier, the companion looks up and copies the exact stored source unit. The exact closed `LocalVisitAudioResultV1` object is:

```text
schema_version       literal "local_visit_audio_result_v1"
protocol             literal "local_visit_audio_v1"
local_session_id     exact 16-byte base64url ID from bootstrap
request_id           exact 16-byte base64url active job ID
source_ref           literal prefix "local-audio:" + request_id; <= 64 chars
captured_at          RFC3339 UTC string
duration_ms          integer 1..1800000
items                array 1..512, total serialized result <= 1 MiB

item.id               unique contiguous integer 0..items.length-1
item.text             exact normalized source unit; 1..4000 Unicode chars and <= 16384 UTF-8 bytes
item.proposed_dispositions
                        unique non-empty array from the exact section-7.2 enum
item.risk_flags       unique array from
                        digit_or_measurement | medication | laterality |
                        negation | uncertainty_or_conflict | unclassified |
                        semantic_status_conflict
item.whisper_flags    unique array from low_confidence | no_speech_anomaly
```

Every object uses `additionalProperties: false`; all fields are required, arrays have fixed maxima and all enum/format/length/total limits are validated before the result leaves loopback. Every source ID appears exactly once in `items`. The item inventory is complete but is not Whisper JSON or a second unstructured transcript: it contains no token probabilities, timestamps, prompt, model response or alternate generated wording. Validation failure returns no partial result.

The production page shows the model labels only as proposals. It uses this complete disposition table; no unrelated-field fallback is permitted:

| Reviewed disposition | Candidate destination |
|---|---|
| `reason` | exact text in required `reason_for_visit`; multiple reviewed units are joined only with a deterministic newline |
| `patient_history` | additive `history[]` exact-text facts |
| `objective_result` | existing `findings[]` exact-text findings |
| `clinician_interpretation` | existing `clinical_impression[]` exact-text facts |
| `option_discussed` | additive `options_discussed[]` exact-text facts; never a decision |
| `clinician_recommendation` | additive `recommendations[]` exact-text items; clinician must select a clinical domain; never a final decision |
| `final_decision` | existing `decisions[]` only after the clinician explicitly confirms final-decision status and selects `DecisionType` |
| `completed_action` | additive `completed_actions[]` exact-text facts; never a pending item or future decision |
| `patient_accepted`, `patient_declined`, `patient_undecided` | additive `patient_preferences[]` exact text plus the corresponding fixed status; never silently converted to a final decision |
| `follow_up` | additive `follow_up[]` exact-text facts; `next_contact` remains empty unless the clinician explicitly supplies its structured purpose/date/time |
| `pending_obligation` | local pending proposal only; excluded from `VisitCaptureCandidateV1.pending` until the separate checkbox and required item type are supplied |
| `safety_net` | existing `safety_net[]` exact-text facts |
| `uncertainty` or `unclassified` | existing `uncertainties[]` exact-text facts, visibly labelled for review |
| `administrative` | additive `administrative_notes[]` exact-text facts; never squeezed into a clinical field |

The server candidate makes additive, default-empty, maximum-60 arrays `history`, `options_discussed`, `completed_actions`, `follow_up` and `administrative_notes`, each containing the existing closed `VisitCaptureFact` shape. `recommendations` contains a closed `VisitCaptureRecommendation` shape: the existing fact fields plus required clinician-selected `type`. `patient_preferences` contains a closed `VisitCapturePreference` shape: the existing fact fields plus required fixed `status=accepted|declined|undecided`. `DecisionType` gains `investigation` so a clinician-confirmed order/plan for a test is not mislabelled referral/therapy. Recommendation and final-decision types are selected by the clinician from `therapy|referral|medication|investigation|advice|sick_leave|admin`; the local model never selects them. Database tables do not change because the encounter payload is already stored as JSON.

An item may have multiple proposed dispositions, but mutually inconsistent semantic-status proposals—option/recommendation/final decision/completed action, or accepted/declined/undecided—receive `semantic_status_conflict` and cannot enter preview until the clinician resolves them. Only an explicitly clinician-reviewed `final_decision` may enter `decisions`. A completed investigation result is `objective_result`; a proposed investigation is `clinician_recommendation`; a clinician-confirmed ordered investigation is `final_decision` with clinician-selected `investigation`. Negative medication history remains `patient_history`; a medication recommendation remains `clinician_recommendation`; neither becomes a medication/final decision by lexical inference.

For pending work, the separate checkbox starts unchecked. When checked, the clinician must choose exactly one `item_type` from `test_or_investigation|referral|follow_up|administrative|other`; `description` and optional `source_excerpt` are copied exactly, with no inferred due date, actor or trigger. Status is `open` and responsible role `unspecified` unless the clinician edits them. No pending candidate is produced from `follow_up`, `recommendation` or `final_decision` alone.

The production page then constructs `VisitCaptureCandidateV1` with new source role `local_visit_audio`, the ephemeral `source_ref`, `comparison_basis.type = unavailable`, no `what_changed`, empty/incomplete coding, no automatically structured medication object and `source_identity_state = uncertain` until explicit identity confirmation. Unedited facts retain provenance to the local source reference. If the clinician changes wording, the UI adds the existing `clinician_edit` source binding and uses that provenance with `verification=clinician_confirmed`; it never attributes edited wording to local audio. The schema gains the additive semantic fields above plus deterministic `gesy_copy`; database schema is unchanged.

## 8. Compact clinician review

The clinician does not review a raw transcript view by default. The production page shows:

- reason for visit;
- patient history and objective findings;
- clinical impression, options discussed and recommendations as distinct sections;
- completed actions, final decisions and patient accepted/declined/undecided state as distinct sections;
- pending obligations/tests;
- follow-up;
- safety-net;
- administrative notes and uncertainties/unclassified content.

Every item remains editable and removable before server preview, with a one-control disposition correction instead of raw JSON editing. Decision and recommendation domains have no model-supplied default; final-decision status and domain require clinician action. Mutually inconsistent semantic statuses block preview until resolved. A proposed pending item has a separate unchecked “Δημιουργία εκκρεμότητας” control plus required item type; the model's category alone can never create a durable pending record. The page receives the complete bounded exact-source item inventory required for source accounting, but not Whisper JSON/timestamps or a separate unstructured transcript view. A local-only expandable transcript may be added later solely during the active result screen if a troubleshooting need is proven; it is excluded from first implementation.

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

`gesy_copy` is rendered on the protected server from the same reviewed candidate, using fixed headings and exact candidate text. It includes only non-empty reason, history, objective findings, impression, options/recommendations, completed actions/final decisions/preferences, pending/follow-up, safety-net, administrative and uncertainty sections. Distinct semantic sections remain distinct. It performs no rewriting and introduces no new clinical language.

Copying is a clipboard action, not a database write and not proof that GeSY was updated. The UI states this explicitly.

## 10. Failure behavior

| Failure | Required behavior |
|---|---|
| FileVault/storage encryption disabled, changing or unverifiable | recording disabled; no v1 override |
| microphone denied/unavailable | no recording; actionable local message |
| companion missing, handshake invalid/replayed/stale or port timeout | recording disabled or active job cancelled; close port; local setup guidance; no cloud fallback |
| decoder identity mismatch or audio too long/large/malformed/bomb-like/undecodable | stop decoder, delete temporary content; sanitized failure; no candidate |
| Whisper timeout/failure | delete temporary content; no partial transcript/candidate |
| context/token limit | reject before LLM inference; no truncation |
| Ollama/model missing or digest mismatch | fail closed; no alternate model |
| model output malformed | reject entire candidate; no partial salvage |
| valid model output omits source IDs | deterministically place every omitted ID in `unclassified`; never drop content |
| no safe reason for visit | request one short clinician correction before preview |
| patient/context/auth/local-session/request changes | invalidate recording/candidate and require new patient confirmation |
| fresh protected-context renewal fails | destroy candidate; no preview/Save; require new patient confirmation |
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

The first independent pre-code R2 returned `BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:5:0`. The first independent correction closure closed FileVault, protocol/result, semantic mapping and decoder P1s, but correctly kept context renewal open (`0:1:0`) because ordinary context creation could adopt a patient version changed during recording. This revision changes only that seam to the atomic exact-prior-context/patient-version CAS in section 6. It requires one focused P1-02 delta + affected closure before code.

Before implementation:

- finalize the model benchmark and exact pinned identifiers/digests;
- verify the popup/opener handshake in the supported Dia/Chromium runtime;
- independent exact-head pre-code R2 closure of privacy, identity, source fidelity, retention, local-network security, decoder and clinical write boundaries;
- explicit closure PASS before runtime code.

Implementation evidence must include:

- setup tests proving FileVault-off/unknown/changing and every decoder/tool/model/manifest mismatch disable recording;
- unit tests for source segmentation, full ID coverage, exact-copy construction, result-schema limits, risk flags and the complete disposition-to-candidate mapping;
- semantic cases for negative history, completed action, option, recommendation, final decision, accepted/declined/undecided, investigation and administrative content, proving each unit has an explicit non-fallback disposition;
- adversarial tests for number changes, hallucinated IDs, duplicates, omissions, negation, laterality, medication wording, mixed Greek/English and malicious prompt-like transcript text;
- loopback Host/Origin/CSRF/content-type/size/symlink/concurrency tests;
- protocol tests for repeated bootstrap, replay, stale result, wrong origin/source/nonce/session/request/direction, extra keys/transfers, oversized result, non-monotonic state, timeout/cancel/navigation and terminal port closure;
- decoder tests for exact identity/receipt/manifest, fixed arguments, malformed/bomb-like/over-duration input, CPU/memory/file/timeout bounds, no network and content-free logs;
- timeout/cancellation/deletion tests proving content cleanup and no candidate after any cleanup-verification failure;
- browser tests for informed-patient gate, Start/Stop, visible recording state, popup handshake, patient switch/auth loss invalidation, compact review, GeSY copy and explicit Save;
- maximum-duration synthetic recording plus processing/review followed by successful exact-prior-context CAS renewal; expired-but-retained prior context success; and negative missing prior row, wrong client session/patient, changed `patient_updated_at`, concurrent mutation, 401, local-session/request/identity-change cases proving no replacement context/candidate/preview/Save;
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
- a decoder/tool identity outside the frozen executable/receipt/manifest or a requirement to operate without positively verified FileVault;
- inability to prove deletion or to keep all source units accounted for.
