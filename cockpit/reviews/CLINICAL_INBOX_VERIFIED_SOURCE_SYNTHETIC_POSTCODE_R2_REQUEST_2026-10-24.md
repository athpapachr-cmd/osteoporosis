# Clinical Inbox — fake verified ADT/referral source-link A — independent POST-CODE R2

**Date:** 2026-10-24 Asia/Nicosia. **MODE:** ONE **fresh independent READ-ONLY affected POST-CODE R2** for the user-authorized **synthetic-only #161** identity-flow implementation. Inspect exactly the changed synthetic code, UI and tests; STOP at PASS/BLOCK/UNKNOWN. No fixes, patient data, provider connection, credentials, merge, deploy or external messages.

**Repo:** `athpapachr-cmd/osteoporosis`
**PR:** [#212](https://github.com/athpapachr-cmd/osteoporosis/pull/212) — DRAFT, unmerged, release HOLD.
**Implementation branch:** `fix/cockpit-synthetic-inbox-verified-source-20261024`
**Verified branch base/main at activation:** `2b05fecfd23e940e463c98d408cf05a8acc8adc7` (the independent #209 top-right peeks correction is already merged/LIVE).
**Frozen CODE+TEST substantive HEAD:** `3a31c0c8364720931f64820db39bfe67b022716e`. Compare any later PR HEAD to this HEAD; later documentation-only commits do not reopen source review.
**Pre-code design:** [PR #210](https://github.com/athpapachr-cmd/osteoporosis/pull/210), independently **PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0**, contract blob `4f1f1111814d841989954ec0d6095635d8f2ccd7`; separate docs-only DRAFT (not production main). Reuse all unchanged PR #208 ClinicalAttention Q1–Q6, per-source revisions/CAS/401/mixed-file R2 evidence; do not restart those reviews.

## Exact implementation+test blobs

| File | Frozen blob |
|---|---|
| `clinical_attention_phase_a.py` | `8bb49f142535b77cca6cf7736f9cd00949a315b7` |
| `clinical_synthetic_161.py` | `12f910014c67a747e35b03a2f051e502764c39b2` |
| `static/cockpit/clinical-inbox.js` | `9b546f41e464a93f9177410fc626aa02003e8b09` |
| `static/cockpit/index.html` | `b48721be88f8ef794a3404382d15561cefdc702f` |
| `test_cockpit_attention_phase_a.py` | `e7bafdd974303946ab3ee5a6f9388a31bc6d2f62` |
| `test_cockpit_synthetic_161.py` | `b45d5f8a0b4f77c8cc75baa60565c26c42fb11bf` |

Author evidence: On the exact frozen substantive code/test HEAD, GitHub Actions [Visit Brief/#161 workflow 40118327298](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40118327298) **SUCCESS**, observed job logs **51 existing Python tests PASSED + 16 synthetic/backend/actual Chromium tests PASSED**, with V3 JS, V4 real-Chromium popovers, syntax and diff hygiene PASS. Relevant [Home 40118327337](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40118327337), [Canonical 40118327274](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40118327274), Visit Capture and Surgery CI SUCCESS. Reuse this author evidence without treating it as **independent post-code R2 PASS** or as merge/deploy authority. Do not run duplicate broad tests merely to review the same bytes.

## Five finite affected questions

1. **True synthetic qualified source evidence.** Does the in-memory fiction distinguish internal `clinical_patient_id` from fake issuer-qualified ADT, synthetic prior confirmed referral and clinician-verified original-order evidence? Is the prior referral attached **only after protected synthetic patients are actually prepared**? Are ADT issuer, exact referral item/subject, evidence validity and one target rechecked server-side at mutation/read instead of trusting a browser checkbox, name, DOB, sender, bare referral or surgical registry? Does revoked/conflicting evidence remove Brief projection and prevent stale clinical action?
2. **Minimal UX without mandatory DOB.** Is the `verified_dob` requirement absent from the synthetic Inbox API and modal, a unique qualified ADT candidate linked with one explicit confirmation, prior verified referral reused without typing any personal identifier, and weak/ambiguous source retained unlinked without mandatory DOB or fabricated independent evidence? Does the UI's eligible-patient lookup guard 401, late sessions, and old item selections?
3. **Correction, relink and concurrency.** Does a synthetic source revision update the same attention item with CAS/409 and idempotency, invalidate manual source evidence bound to the earlier revision, remove the previous patient's Brief and old review/disposition/unsent states, and require fresh evidence+review before changing any patient-specific decision? Do direct cross-patient attempts, stale version, changed replay payload, mixed/unknown subject and revoked identity remain fail-closed?
4. **Clinical meaning and no side effects.** Is `SYN-ATT-GESY` only `gesy_notice: results_available`, with **no lab value, reviewed/normal result, accepted clinical fact, CareTask completion, outbound communication, booking or SMS**? Do lab/MRI remain source candidates and mixed PDF remain Inbox-only? Is all content synthetic, restricted to protected `/clinical/synthetic-161/`, with no new DDL, real patient identifier persistence, mailbox connector or provider authorization?
5. **Preserved UX and bounded release gate.** Are #209 top-right icon peeks retained in normal/synthetic Cockpit, including the correct synthetic Inbox focus, and are session/scope reset, current vs stale 401 and separate clinician review axes preserved? Do the changed tests and exact-head CI prove only synthetic behavior, leaving persistent real-identity B and live-source C separately unauthorized?

**Decision format:** `PASS|BLOCK|UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE|PARTIAL`, `P0:P1:P2`, precise per-question disposition, exact HEAD/blobs and targeted evidence. For BLOCK give one source-reachable issue and smallest correction. No speculative second design, no duplicate review chain. **STOP READ-ONLY, with no merge/deploy authority even if PASS.**
