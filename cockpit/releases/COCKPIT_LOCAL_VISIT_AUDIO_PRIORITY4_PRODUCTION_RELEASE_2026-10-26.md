# Cockpit Priority 4 local visit audio — production and Mac release

**Date:** 2026-10-26 Asia/Nicosia; first-use entry correction verified 2026-10-27. **State:** READY_FOR_PRODUCT_OWNER_SMOKE. Web/API and the corrected local-launcher entry are LIVE; the pinned local companion is installed and synthetic end-to-end verified. The first informed-patient recording remains unverified and belongs to the clinician.

## Released behavior and safety boundary

Visit Capture now offers clinician-started local recording only after confirmed patient context and an informed-patient checkbox. Audio, decoded WAV, Whisper transcript, numbered source units and Krikri classification remain on the FileVault-enabled Mac. The local model can assign identifiers and semantic dispositions only; deterministic code copies the exact transcribed source units into a compact editable review. Missing classifications are explicit `unclassified` items rather than omissions.

Patient/source identity remains unconfirmed until the clinician checks it. The browser renews the exact prior protected patient context atomically before preview. The existing protected preview and explicit Save remain the only clinical write. Proposed obligations require a separate opt-in. GeSY output is copy-only. There is no cloud fallback, background recording, persistent audio/transcript or automatic clinical write.

## Source, review and CI

| Gate | Verified result |
|---|---|
| Main feature PR | [#227](https://github.com/athpapachr-cmd/osteoporosis/pull/227), squash `1c46da558e88f14f8974de862ff5eebf6a5766f3` |
| Frozen reviewed implementation | `b6fd2e3f2d8547bc57a33c917ec7c342bddc614a` |
| Independent pre/post-code R2 | PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0 |
| Main feature exact-head CI | 15/15 PASS |
| Target-Mac correction PR | [#228](https://github.com/athpapachr-cmd/osteoporosis/pull/228), squash `a467ae510ef0fc792845b65ddcf895b30d4684ce` |
| Correction exact-head CI | 11/11 PASS |
| Local affected tests | 55 Python + three Node protocol/browser smokes PASS |
| Shared Cockpit regression | 37/37 PASS with actual Chromium |
| Release gates | Fresh Ops RELEASE PASS for both releases; correction successor admission PASS; `natural_language_semantic_change=false` |

The first stable local installation exposed two fail-closed target-Mac portability defects before patient use. Ollama 0.40.11 reports a model manifest as bare 64-character hex while the service expected the equivalent `sha256:` form. The correction canonicalizes only these two exact SHA-256 representations, rejects malformed/uppercase/other algorithms and retains exact manifest equality. macOS also exposes `RLIMIT_AS` but rejects setting it in the decoder child. The correction skips only that unimplemented macOS call; CPU and output-file limits, 64 MiB upload bound, 30-minute duration bound, pinned decoder identity and all cleanup/timeouts remain.

## Render and production no-write smoke

Native Render auto-deploy `dep-dbfscc142hec739rnrs0` made the feature merge LIVE. Corrective native auto-deploy `dep-dbfsm3e0tbcc73du91qg` made exact final main `a467ae5` LIVE at 2026-10-26T21:52:17.957Z. No manual duplicate deploy, secret, environment, schema, database or network change was made.

| Check | Observed result |
|---|---|
| Served Visit Capture bytes | `index.html`, `app.js`, `styles.css` SHA-256 values exactly match final merged source |
| Anonymous protected POST | context, context-renew, preview and save each returned HTTP 401 |
| Error-level app logs | Zero from the corrective release start through smoke |
| HTTP 500/502/503 logs | Zero in the same window |
| Squash fidelity | Tested correction tree and final merge tree both `8f1b20e77aac61ea994ff1a9ffa0371725066cc4` |

## Installed Mac runtime and synthetic qualification

FileVault is ON. Homebrew CMake 4.4.4 was added because the reviewed installer requires `/opt/homebrew/bin/cmake`. The installer verified Homebrew FFmpeg 9.0.2_1 and its bottle/SBOM/dynamic-library closure, built pinned `whisper.cpp` commit `231638a0717b7822784297e5980965ffcec05810` with Metal, verified the pinned `large-v3-turbo` model, verified the official Krikri Q4_K_M source digest, and created the dedicated Ollama 0.40.11 model/manifest under `~/Library/Application Support/OrthoCockpitLocalAudio`. The general Ollama port/model store is not used.

The final merged-tree qualification used the real loopback HTTP service with an existing 287.76-second synthetic mixed Greek/English recording. Result: HTTP 200, strict `local_visit_audio_result_v1` / `local_visit_audio_v1`, 55 exact-source units, 51 risk-flagged review items and 11 explicit `unclassified` units. The category inventory was returned without printing transcript text. Temporary request audio, decoded WAV, transcript, prompt and model output were removed when the request completed. No patient data, clinical key, database credential, Render secret, cloud AI request or clinical write was used.

The installed assets leave approximately 13 GiB free on the Mac at this checkpoint. The source GGUF and dedicated Ollama copy are both retained because startup re-verifies their independent identities.

## 2026-10-27 first-use entry correction

The clinician's first launcher attempt proved that the loopback companion opened the correct nonce-bearing URL, but `visit-capture/entry.js` redirected that URL into the ordinary integrated Cockpit shell and discarded the local-audio fragment. The visible result was the Dia-summary composer without Start/Stop. This was an entry-routing defect, not a launcher, microphone or clinician-operation failure.

[PR #230](https://github.com/athpapachr-cmd/osteoporosis/pull/230) preserves the dedicated Visit Capture page only for an exact `#local_visit_audio_v1=` fragment with a 43-character base64url nonce. Ordinary patient entry and malformed fragments retain the safe integrated-shell redirect. Runtime and regression commit `dc2acb236ea49c44d58c8fd5154b797bc3e994c4` received independent affected R1 PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0; final exact PR head `cbd0eea1fc5fd33162f6a9d89d1e28d2768c6342` passed 14/14 checks and squash-merged as `20b6e2a0120a3e18ecb89b33e29d65a433d3180c`.

Native Render auto-deploy `dep-dbg1oqnf3r2c73b4d6ug` reached LIVE at 2026-10-27T03:39:08.372Z. Served `entry.js` SHA-256 `c3a0c21c9aa63b0497561e784c550982caac3a591878f9f81dd3a2e377e48269` exactly matched the merge. Hidden real-browser production smoke confirmed the exact local fragment remained on `/static/cockpit/visit-capture/index.html`, ordinary synthetic patient entry redirected to `/static/cockpit/index.html?capture=1&entry_patient=SYN-000`, and a malformed local fragment redirected to `/static/cockpit/index.html?capture=1`. Error-level and HTTP 500/502/503 logs were empty from deploy start through smoke. No manual duplicate deploy, patient audio/data or clinical write occurred.

## Clinician handoff

Start the companion from the final repository checkout with:

```bash
python3 -m local_visit_audio.service
```

Open `http://127.0.0.1:8765`, enter Visit Capture from the local launcher, select the patient, confirm that the patient has been informed, then Start/Stop. For the first real visit, inspect the compact prepared fields and every risk/unclassified item, confirm source identity, preview and only then explicitly Save. Closing/navigation/cancel before completion terminates local processing and cleans temporary content.

Technical release is complete. Clinical accuracy and real-use timing remain to be accepted on the first informed-patient recording; no agent performed that step.
