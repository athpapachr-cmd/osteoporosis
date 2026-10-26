# Priority 4 local visit audio — PRE-CODE R2 P1 correction closure request

**Mode:** one independent **READ-ONLY** delta + affected-cumulative closure of the original five P1 findings, then STOP. No new architecture, implementation, install/download, real patient audio/data, database access, merge, deployment or external write.

## Exact targets

- Original reviewed design: commit `33e696b3934ce410b27ae4cab930273edee02bbf`; blob `975e5f814ea9e298b6287baa1cd9d8f53675e18e`.
- Original request: commit `f20e66b7822d44f7742296aeb518ea1148654379`.
- Original independent verdict: `BLOCK / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:5:0`, recorded in `cockpit/reviews/LOCAL_VISIT_AUDIO_PRIORITY4_PRECODE_R2_BLOCK_RECEIPT_2026-10-26.md`.
- Corrected design commit: `8c6712b56ef55731da0f4130dd5b27e053bc1894`.
- Corrected design blob: `6597c1e4b43cc36cae05e55c53b992557615c16d`.
- Base remains remote `main=889cc14b0efb607b553db4f2735a3c3bac9dae34`.

Inspect the exact corrected commit/blob and the relevant existing schema/context code. Author prose is not an independent closure.

## Required closures

1. **P1-01 FileVault:** confirm v1 has no override and recording fails closed on disabled, changing, unknown or unverifiable storage encryption at setup, process start and each job.
2. **P1-02 context expiry:** confirm final preview uses a fresh protected context issued for the unchanged authenticated client session and selected patient; the old context is invalidated; auth/patient/local-session/request/identity changes destroy the candidate.
3. **P1-03 exact protocol/result:** confirm the one-shot window bootstrap and port state machine have exact direction, origin/source/nonce/session/request/key/transfer/size/time/state/terminal rules, and that `LocalVisitAudioResultV1` has a closed bounded schema with complete source accounting.
4. **P1-04 clinical distinctions/mapping:** confirm history, objective result, interpretation, option, recommendation, final decision, completed action, accepted/declined/undecided, follow-up, pending, uncertainty and administration remain distinct; only clinician-confirmed final decisions enter decisions; investigation and pending item types have exact non-fallback handling.
5. **P1-05 decoder:** confirm one exact FFmpeg bottle/executable/receipt/library-manifest contract, fixed offline argument vector, independent WAV validation, resource/time limits, content-free logs and fail-closed cleanup.
6. **Affected cumulative safety:** confirm the five corrections do not break the original R1–R8 boundaries: no cloud/raw-audio upload, no model prose as truth, no automatic pending/write, explicit identity/preview/Save, complete cleanup/failure behavior and proportionate implementation tests.

## Required handback

Return `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, exact `P0:P1:P2`, each P1-01–P1-05 `CLOSED | OPEN | UNKNOWN`, affected-cumulative disposition, exact corrected commit/blob, any new directly reachable material issue plus smallest correction, and the narrow authorized implementation/test boundary if PASS. **STOP READ-ONLY.**

A PASS authorizes only bounded implementation. It does not authorize real patient use, merge, deployment, database operations or a clinical write.
