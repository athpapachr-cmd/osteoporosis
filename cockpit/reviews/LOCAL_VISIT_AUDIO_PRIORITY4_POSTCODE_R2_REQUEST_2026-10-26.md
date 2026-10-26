# Priority 4 local visit audio — independent exact-code POST-CODE R2 request

**Date:** 2026-10-26 Asia/Nicosia.  
**Mode:** one fresh independent **READ-ONLY** implementation-fidelity, privacy and clinical-write review; stop after the finite verdict. No edits, dependency/model installation, real patient audio/data, database access, merge, deployment or external write.

## Exact target and authority

- Repository/branch: `athpapachr-cmd/osteoporosis` / `codex/priority4-local-visit-audio`
- Fresh remote base: `main=889cc14b0efb607b553db4f2735a3c3bac9dae34`
- Exact implementation+evidence target: `dc5894ec80d29d4a3a6c2f06dba598be931af032`
- Runtime/test commit: `a66f3b442b68fd98c42e8d796bbcbcc467ea55b6`
- Frozen design: `9027d07152729d83972ca725feb533c0e987205d`
- Design blob: `644c36a8b557be46e4165a6cce9c63fa4ed54290`
- Pre-code result: `PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0`, receipt `cockpit/reviews/LOCAL_VISIT_AUDIO_PRIORITY4_PRECODE_R2_FINAL_PASS_RECEIPT_2026-10-26.md`

The reviewer must inspect the committed range `889cc14..dc5894e` and prove any later branch-head drift is documentation-only before relying on this target.

## Exact affected blobs

- `clinical_data.py` — `b931b90ad04d14309cde063e4aad5a79958dc76b`
- `static/cockpit/visit-capture/index.html` — `34dc4906fa28126f8d7e1aa6cdb887b7aa58edbd`
- `static/cockpit/visit-capture/app.js` — `0904847de4744e350e75267dde11f6f4563ab082`
- `static/cockpit/visit-capture/styles.css` — `c86b04a4b737ffebb9d31736628cee1efa7bafc1`
- `local_visit_audio/install.py` — `4f76de75d4b3ef05d09737d87b375a9f26363ebb`
- `local_visit_audio/launcher.js` — `c2b60cffd35032eec11b5ac124650a9a475a8431`
- `local_visit_audio/pipeline.py` — `927aac080c0d4f330226c6d3d9e6b26a6fb1fbfe`
- `local_visit_audio/service.py` — `7b49bbc9dd4a46314173d1d0b6e76be5a28065ac`
- `local_visit_audio/decoder_manifest_v1.json` — `1855dde55e7e76ef77e4ae8f6ae5f212f0e59f98`
- `test_visit_capture.py` — `9c87fd118859d1cf15dfb65648eebed6917c0494`
- `test_visit_capture_ui.py` — `13e12d9def9ad2614eaeecfa8656543cd0a07f20`
- `test_local_visit_audio.py` — `e0b63cb052fb3b3630847043a855d99f24fb693b`
- `test_visit_capture_smoke_ui.js` — `c714a7309b66361b403ee7c7457f20be4c12f1e5`
- `test_visit_capture_local_audio_ui.js` — `899e148eb987ea0ba25dd58ce654bf05bc768e49`
- `.github/workflows/visit-capture-tests.yml` — `5f1d77c1bd1cd44616a5ff85ecacd4b2073204a4`

## Product invariant

Audio, raw transcript, prompt and model response stay on the FileVault-enabled clinician Mac. The local model returns identifiers/categories only; deterministic code copies exact source units. No result reaches protected preview until the clinician reviews categories/text, separately opts into pending work and confirms patient identity. Existing protected preview and explicit Save remain the only clinical-write path. No cloud fallback.

## Finite review questions

1. **Local-only boundary and retention:** Is there any reachable audio/transcript/prompt/model-output path to Render/cloud, persistent browser storage, content logs or a successful response before complete temporary cleanup?
2. **Lifecycle/protocol security:** Do informed-patient Start/Stop, exact origin/source/nonce/session/request state, transferred buffer, Host/Origin/Sec-Fetch/cookie/CSRF checks, one job, cancellation, ten-minute total deadline, stale/replay rejection and terminal closure implement the frozen contract without a confused-deputy or late-result path?
3. **Supply chain and offline runtime:** Are FileVault, exact FFmpeg executable/receipt/SBOM/dylib closure, whisper commit/model, Ollama version/cloud-off environment, official Krikri source hash and installed manifest identity all checked fail-closed with no substitution?
4. **Source fidelity and hostile input:** Are source units complete and bounded; model output ID-only and schema-closed; omitted IDs surfaced; duplicates/out-of-range/context overflow/malformed or prompt-like input rejected; exact text retained; risk/conflict review unavoidable where required?
5. **Destination identity and CAS:** Does context renewal lock the exact prior context and patient, compare patient version before replacement, preserve client/patient/local-session/request identity in the browser, and fail closed without deleting/replacing state on mismatch?
6. **Clinical semantics and writes:** Are option/recommendation/final decision/completed action/preferences distinct; clinician-only type/status selection preserved; pending starts unchecked and requires its own type; deterministic GeSY copy is copy-only; no automatic preview/Save or alternate clinical owner exists?
7. **Failure/cleanup behavior:** Do decoder/Whisper/Ollama/validation/auth/navigation/cancel/timeout/cleanup failures yield only bounded codes, no candidate/write and verified cleanup? Inspect cancellation while each subprocess/network stage is active.
8. **Evidence coverage:** Do synthetic tests cover the affected server/API, protocol/service, complete-source inventory, cancellation/cleanup, stale identity/CAS, browser-reviewed flow, existing no-patient UX and preserved shared Visit Capture/CP regressions? Identify only a material missing oracle.

## Author evidence (reuse; do not rerun broadly for reassurance)

- Syntax: Python changed modules/tests PASS; Visit Capture, launcher and local-flow JavaScript `node --check` PASS.
- Focused affected set: `47 passed` for `test_visit_capture.py test_visit_capture_ui.py test_local_visit_audio.py test_cockpit_home.py`.
- Browser simulations: existing no-patient smoke PASS; new local-audio reviewed-flow smoke PASS.
- Shared Visit Capture/CP regression: `37 passed` for CP1, CP2, patient entry, appointment chart and visit amendments.
- Local prerequisite probe before freeze: FileVault and exact decoder manifest verification PASS; actual synthetic Whisper JSON output matched the parser. No stable Krikri install or real patient material was used.
- Broad unscoped `pytest -q` did not run tests because collection found missing local `psycopg`; this is a local environment dependency gap, not presented as product evidence. The applicable workflow installs `requirements.txt` and now includes all new Priority-4 tests.

## Required handback

Return:

```text
REVIEW = PRIORITY 4 LOCAL VISIT AUDIO POST-CODE R2
REMOTE MAIN = <fresh SHA>
TARGET = <branch/head/runtime+test commit and drift result>
VERDICT = PASS | BLOCK | UNKNOWN
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL
P0:P1:P2 = x:y:z
F1 = ...
F2 = ...
F3 = ...
F4 = ...
F5 = ...
F6 = ...
F7 = ...
F8 = ...
MATERIAL FINDINGS = ...
IMPLEMENTATION FIDELITY = ...
REMAINING RELEASE/FIRST-USE GATES = ...
STOP = ...
```

A PASS is implementation-fidelity disposition only. It does not authorize model installation, real patient use, merge, deploy or clinical writes.
