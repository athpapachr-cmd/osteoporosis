# Priority 4 local visit audio — independent PRE-CODE R2 request

**Date:** 2026-10-26 Asia/Nicosia.
**Mode:** one fresh independent **READ-ONLY** architecture, privacy and clinical-write review; stop after the finite verdict. No implementation, dependency installation, model download, real patient audio/data, database access, merge, deployment or external write.

## Frozen target

- Design commit: `33e696b3934ce410b27ae4cab930273edee02bbf`
- Design: `cockpit/LOCAL_VISIT_AUDIO_PRIORITY4_2026-10-26.md`
- Design blob: `975e5f814ea9e298b6287baa1cd9d8f53675e18e`
- Active plan: `SLICE_PLAN_CURRENT.md`
- Base before this Priority 4 design chain: remote `main=889cc14b0efb607b553db4f2735a3c3bac9dae34`

The reviewer must inspect the exact committed objects above. Author benchmarks and prose are evidence inputs, not an independent PASS.

## Product-owner direction and invariant

Visit audio, raw transcript, prompt and model response stay on the clinician's FileVault-enabled Mac. The local model may return only source-unit identifiers and categories. Deterministic code copies exact source text; the clinician reviews/edits, separately selects durable pending work, confirms destination identity, uses protected preview and explicitly saves. No free-form model wording or rewritten number may become record truth. No cloud fallback.

## Required review questions

1. **Local data and retention boundary:** Does the contract actually keep audio/transcript/prompt/model response off Render/OpenAI/cloud, prevent browser persistence and patient-content logs, clean all temporary paths on every outcome, and avoid overstating logical deletion as forensic erasure?
2. **Consent, lifecycle and destination identity:** Are informed-patient gating, explicit Start/Stop, visible recording state, maximum duration, navigation/auth/patient-change cancellation, source-identity uncertainty and explicit clinician confirmation sufficient and fail-closed?
3. **Popup and loopback security:** Is the local-launcher → production-popup nonce/origin/source/MessageChannel design sound? Are exact Host/Origin/Sec-Fetch/CSRF/session checks, loopback-only binding, request IDs, content limits, symlink prevention and one-job concurrency adequate against cross-site requests, DNS rebinding, confused deputy and replay?
4. **Source fidelity and hostile input:** Does identifier-only structured output plus exact-source lookup prevent generated clinical wording? Are prompt-injection-like transcript content, duplicate/out-of-range/missing IDs, omitted IDs, multi-category IDs, context batching, response limits and silent truncation handled deterministically and visibly?
5. **Visit Capture and pending/GeSY semantics:** Does the proposed `local_visit_audio` candidate preserve current preview/Save authority and comparison semantics? Can no pending record be created without its own clinician opt-in? Is deterministic `gesy_copy` clearly a clipboard view and never an inferred write?
6. **Supply chain and 16 GB feasibility:** Are pinned whisper.cpp/model/Ollama/Krikri identities, digest checks, no-substitution behavior, sequential execution, context limits and installer self-tests sufficiently exact? Identify any dependency/model-download boundary that still permits an unverified executable or blob.
7. **Failures, cleanup and observability:** Does every microphone, opener, transfer, decode, Whisper, Ollama, validation, timeout, cancellation, cleanup and auth failure leave no candidate/write and provide only bounded non-clinical diagnostics? Flag any path that could return success after incomplete cleanup or partial source accounting.
8. **Smallest safe implementation/test boundary:** Is the declared implementation seam narrow enough? State the minimum synthetic/adversarial unit, browser and local integration evidence required before post-code R2, without widening into unrelated Cockpit work.

## Required handback

Return:

- `PRE-CODE R2 PASS | BLOCK | UNKNOWN`;
- `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`;
- exact `P0:P1:P2` counts;
- exact reviewed commit and design blob;
- one disposition for each R1–R8 above, with precise file/section evidence;
- for every material finding, the reachable failure and smallest design correction;
- the exact narrow implementation and test boundary if PASS;
- **STOP READ-ONLY**.

A PASS authorizes only bounded implementation under the recorded Product Owner direction. It does not authorize real patient use, merge, deployment, database changes or a clinical write.
