# SLICE_PLAN_CURRENT.md — Priority 4 local visit audio

> **STATUS (2026-10-27): READY_FOR_PRODUCT_OWNER_SMOKE.** PR #227 feature merge `1c46da5`, PR #228 target-Mac correction `a467ae5` and PR #230 first-use entry correction `20b6e2a` are LIVE; the latest native Render deploy is `dep-dbg1oqnf3r2c73b4d6ug`. Production bytes, exact local-launcher routing, ordinary redirect, malformed-fragment rejection and error/HTTP-5xx logs passed no-write smoke. FileVault and the exact pinned FFmpeg/Whisper/Krikri/Ollama local runtime remain installed; the final real loopback synthetic 4:48 execution returned 200 with strict schema, 55 exact-source units, risk flags and 11 explicit unclassified units, then cleaned temporary content. The missing Start/Stop entry blocker is closed and the technical writer is released. No real patient audio/data or clinical write occurred; first informed-patient use and clinician accuracy/timing acceptance remain.
> **Previous active plan:** Priority 3 Office Cases is archived at `archive/SLICE_PLAN_PRIORITY3_OFFICE_CASES_2026-10-26.md`; its reviewed production state is unchanged.

## 1. Objective and acceptance

Extend the existing protected Visit Capture flow with clinician-started local recording. After the patient is informed and the clinician presses Start/Stop, `whisper.cpp` transcribes on the Mac and a pinned local model classifies numbered source units. Deterministic code copies exact source text into a reviewable candidate. The clinician confirms patient identity, edits/removes items, separately opts into any durable pending obligation, previews on the protected server and explicitly saves.

The screen also offers deterministic **Copy for GeSY** text from the reviewed candidate. Copy is not a write to GeSY.

Acceptance requires a compact review rather than line-by-line transcript reading, no free-form model prose in record truth, no background recording, no automatic Save and no cloud fallback.

## 2. Owning design and seams

`cockpit/LOCAL_VISIT_AUDIO_PRIORITY4_2026-10-26.md` is the exact design. The existing owners remain:

- confirmed Cockpit patient context for destination identity;
- `VisitCaptureCandidateV1`, protected preview and explicit Save for clinical writes;
- one encounter as the source of Snapshot / Visit Brief / Encounter Detail;
- pending items as separate durable records only after explicit clinician selection.

The new bounded seams are a loopback-only `local_visit_audio/` macOS companion, a strict launcher/popup `postMessage` channel, local `whisper.cpp` + Ollama orchestration, deterministic exact-source candidate construction, and a narrow extension of the Visit Capture UI/schema for local source role and GeSY copy.

## 3. Safety contracts

- The production page never uploads visit audio or raw transcript to Render. The local service has no database, Clinical Key, Notion or Render credential.
- Recording requires confirmed patient context, an informed-patient checkbox and direct clinician Start/Stop actions. Patient/context/auth/navigation changes stop and invalidate the flow.
- FileVault must be positively verified at setup, service start and each job; v1 has no override. A new protected context-renew CAS endpoint locks the exact prior context (even expired) and patient row, compares the stored/current patient version and issues a fresh context only for the unchanged authenticated client session and patient; browser patient/local-session/request/identity must also remain unchanged.
- The loopback service binds only to `127.0.0.1`, enforces Host/Origin/CSRF/content/size/concurrency guards, uses private temporary files and deletes audio/WAV/transcript/prompt/model output on every outcome.
- The one-shot popup/MessageChannel protocol and `LocalVisitAudioResultV1` use exact schemas, bounded fields, request/session binding, terminal timeouts and replay rejection.
- The local model returns identifiers/canonical semantic dispositions only. It cannot supply clinical wording. Missing IDs become explicit `unclassified` review items; they are never dropped. Options, recommendations, final decisions, completed actions and accepted/declined/undecided remain distinct; only clinician-confirmed final decisions enter decisions.
- Model categories do not create pending records automatically. Each proposed obligation has a separate clinician control.
- Source identity stays uncertain until the clinician explicitly confirms that the recording belongs to the selected patient.
- Model/digest mismatch, context overflow, malformed output, deletion failure or missing handshake fails closed. No cloud fallback.

## 4. Evidence and review gates

Observed on the target M4/16 GB Mac: `whisper.cpp` large-v3-turbo processed a 287.76-second synthetic mixed Greek/English recording in 19.16 seconds warm, projecting about 80 seconds for 20 minutes. A free-form local summary changed a synthetic T-score, proving that generated prose cannot be record truth. Direct HTTPS production-page fetch to loopback was browser-blocked; an actual Dia/Chromium popup probe preserved `window.opener` and the nonce-bearing URL fragment, supporting the local-launcher channel.

The official Krikri Q4_K_M and official Gemma 4 12B were compared on the same synthetic identifier-only task. Krikri produced schema-valid IDs in 15.37 seconds but required the frozen omitted-ID/category/pending guardrails. The current official Gemma default selected its MLX runner and failed schema-constrained inference with HTTP 501 because structured-output support was unavailable in that runner. Priority 4 v1 therefore pins the official Krikri Q4_K_M blob and forbids substitution. The corrected design also pins Homebrew FFmpeg `9.0.2_1`, its official bottle and resolved executable identities, fixed offline decoding arguments and resource limits.

The initial independent exact-head R2 found five P1 design gaps: encryption override, 30-minute context expiry, underspecified cross-origin/result protocol, incomplete semantic mapping and unpinned decoder. First closure closed four and left only context expiry open because ordinary context creation could adopt a changed patient version. The smallest correction is now an atomic exact-prior-context renewal/CAS endpoint. One focused independent P1-02 closure must return PASS before implementation. Implementation then requires synthetic/adversarial unit, API, browser and local end-to-end evidence, followed by independent exact-code post-code R2, applicable CI and guarded release.

## 5. Replan triggers and next action

Any requirement for cloud processing, transcript persistence, free-form generated record text, direct production-to-loopback networking, automatic identity acceptance, automatic pending/write behavior, a different model/digest or inability to prove complete source accounting reopens design R2.

Next action: clinician starts the installed loopback companion, performs the first informed-patient Start/Stop from local Visit Capture, reviews every risk/unclassified item, confirms source identity, previews and explicitly saves only if correct. Record actual elapsed time and any terminology/omission correction. No additional technical writer is active.
