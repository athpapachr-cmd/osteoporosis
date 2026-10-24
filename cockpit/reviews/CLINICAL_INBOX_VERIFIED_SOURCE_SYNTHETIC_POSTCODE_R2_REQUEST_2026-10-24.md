# Clinical Inbox — synthetic verified-source identity — ONE affected POST-CODE R2 with P1-01 correction

**Date:** 2026-10-24 Asia/Nicosia. **MODE:** ONE **fresh independent READ-ONLY affected POST-CODE R2** for the user-authorized **synthetic-only #161** identity-flow implementation. Inspect exactly the changed synthetic code, UI and tests; STOP at PASS/BLOCK/UNKNOWN. No fixes, patient data, provider connection, credentials, merge, deploy or external messages.

**Repo:** `athpapachr-cmd/osteoporosis`
**PR:** [#212](https://github.com/athpapachr-cmd/osteoporosis/pull/212) — DRAFT, unmerged, release HOLD.
**Implementation branch:** `fix/cockpit-synthetic-inbox-verified-source-20261024`
**Verified branch base/main at activation:** `2b05fecfd23e940e463c98d408cf05a8acc8adc7` (the independent #209 top-right peeks correction is already merged/LIVE).
**Original implementation/test HEAD:** `3a31c0c8364720931f64820db39bfe67b022716e`. **Corrected frozen CODE+TEST HEAD:** `a4726ebbd31d862a5ccb57b2c37c974cef33cc5f`. Compare any later PR HEAD to this corrected freeze; documentation-only successor commits need no repeat implementation review. The original R2 was still pending; this is its ONE corrected affected review request, not an additional review chain.
**Pre-code design:** [PR #210](https://github.com/athpapachr-cmd/osteoporosis/pull/210), independently **PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0**, contract blob `4f1f1111814d841989954ec0d6095635d8f2ccd7`; separate docs-only DRAFT (not production main). Reuse all unchanged PR #208 ClinicalAttention Q1–Q6, per-source revisions/CAS/401/mixed-file R2 evidence; do not restart those reviews.

## Exact implementation+test blobs

| File | Frozen blob |
|---|---|
| `clinical_attention_phase_a.py` | `4d69a1773928a7de0b5ab5d9ce15f1a0aef79cff` |
| `clinical_synthetic_161.py` | `12f910014c67a747e35b03a2f051e502764c39b2` |
| `static/cockpit/clinical-inbox.js` | `9b546f41e464a93f9177410fc626aa02003e8b09` |
| `static/cockpit/index.html` | `b48721be88f8ef794a3404382d15561cefdc702f` |
| `test_cockpit_attention_phase_a.py` | `6b3a9a4bb635c61c54d18087041324b1b4eeea5f` |
| `test_cockpit_synthetic_161.py` | `b45d5f8a0b4f77c8cc75baa60565c26c42fb11bf` |

**Corrected author evidence:** Exact corrected code/test SHA `a4726ebbd31d862a5ccb57b2c37c974cef33cc5f`: [Visit Brief/#161 workflow 40123877867](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40123877867) **SUCCESS**, observed job logs **51 Python tests + 19 synthetic/backend/actual Chromium tests PASSED**; V3 JavaScript, V4 real Chromium popovers, syntax and diff hygiene PASS. [Home 40123877830](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40123877830), [Canonical 40123877828](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40123877828), Surgery and Clinical Learning L0/L1 gates SUCCESS. Prior #208/#209 R2/R1 evidence reused. **Author CI is not independent R2 PASS or release authority.**

## P1-01 — smallest affected author correction and closure oracle

**Previously reachable issue:** The original `SyntheticAttentionStore.revise` retained a verified ADT/referral patient association when the corrected source still had `source_subject_scope=single_subject`. The previous source evidence could remain authoritative in a patient Brief without being independently requalified for the new revision.

**Corrected behavior:** On every source revision, including unchanged `single_subject`, the synthetic owner atomically clears current patient ID, identity-link kind/reference, linked revision, actor/time, **issuer-qualified source ID/reference**, review actor/state, candidate and disposition; the stable logical item/receipt lineage and sanitized audit remain. `_link_current` now checks `linked_revision` against the active revision. Prior idempotency responses after a newer item version return 409; for the unchanged version after evidence revocation, the replay response is rebuilt from the current safe projection.

**Regression oracles:** Two new parameterized API cases for `SYN-ATT-MRI` (qualified fictional ADT) and `SYN-ATT-GESY` (previously confirmed fictional referral), both corrected `single_subject → single_subject`, plus an audit-specific test: old Brief excluded, old strong candidate not reoffered, old approval and old replay rejected, repeated `/prepare` cannot restore a stale link, no task closure/communication/PHI, and source history retained without raw ID. Existing immediate-version idempotency, mixed-subject isolation, 401/sessions, #209 popovers and prior reviews are reused.

**One independent affected cumulative R2 requested:** confirm P1-01 is CLOSED and check only the affected five questions below. **STOP** at PASS/BLOCK/UNKNOWN; no other design or review cycle.


## Five finite affected questions

1. **True synthetic qualified source evidence.** Does the in-memory fiction distinguish internal `clinical_patient_id` from fake issuer-qualified ADT, synthetic prior confirmed referral and clinician-verified original-order evidence? Is the prior referral attached **only after protected synthetic patients are actually prepared**? Are ADT issuer, exact referral item/subject, evidence validity and one target rechecked server-side at mutation/read instead of trusting a browser checkbox, name, DOB, sender, bare referral or surgical registry? Does revoked/conflicting evidence remove Brief projection and prevent stale clinical action?
2. **Minimal UX without mandatory DOB.** Is the `verified_dob` requirement absent from the synthetic Inbox API and modal, a unique qualified ADT candidate linked with one explicit confirmation, prior verified referral reused without typing any personal identifier, and weak/ambiguous source retained unlinked without mandatory DOB or fabricated independent evidence? Does the UI's eligible-patient lookup guard 401, late sessions, and old item selections?
3. **Correction, relink and concurrency — P1-01 closure.** Does a corrected source always revoke existing fake ADT, referral and clinician-confirmed evidence—even when `single_subject` is unchanged—and remove the old patient's Brief/record projection, strong candidate and stale review/disposition? Does the revised document remain one versioned attention item, with sanitized provenance and new verification required? Are stale CAS actions and old idempotency replays 409, while exact immediate retries work and revoked mappings never leak an old cached patient projection? Do cross-patient, mixed/unknown subject, conflicting issuer and stale form cases still fail closed?
4. **Clinical meaning and no side effects.** Is `SYN-ATT-GESY` only `gesy_notice: results_available`, with **no lab value, reviewed/normal result, accepted clinical fact, CareTask completion, outbound communication, booking or SMS**? Do lab/MRI remain source candidates and mixed PDF remain Inbox-only? Is all content synthetic, restricted to protected `/clinical/synthetic-161/`, with no new DDL, real patient identifier persistence, mailbox connector or provider authorization?
5. **Preserved UX and bounded release gate.** Are #209 top-right icon peeks retained in normal/synthetic Cockpit, including the correct synthetic Inbox focus, and are session/scope reset, current vs stale 401 and separate clinician review axes preserved? Do the changed tests and exact-head CI prove only synthetic behavior, leaving persistent real-identity B and live-source C separately unauthorized?

**Decision format:** `PASS|BLOCK|UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE|PARTIAL`, `P0:P1:P2`, precise per-question disposition, exact HEAD/blobs and targeted evidence. For BLOCK give one source-reachable issue and smallest correction. No speculative second design, no duplicate review chain. **STOP READ-ONLY, with no merge/deploy authority even if PASS.**
