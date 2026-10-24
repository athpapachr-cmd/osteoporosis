# Clinical Inbox Phase A — focused POST-CODE R2 P1-01/P1-02 closure request

**DATE:** 2026-10-24 (Asia/Nicosia).
**MODE:** ONE fresh independent **READ-ONLY**, bounded **delta + affected-cumulative** POST-CODE R2 correction-closure. **STOP after verdict.** Do not repeat the already independently CLOSED PRE-CODE R2 Q1–Q6/P1/P2, nor unaffected first POST-CODE Q1/Q5.

**Repository:** `athpapachr-cmd/osteoporosis`
**PR:** [#208](https://github.com/athpapachr-cmd/osteoporosis/pull/208) — DRAFT, unmerged, release HOLD.
**Unchanged remote production main (author verified):** `32a87ec8757b21b64dd8837e362810b5a2b9b510`.
**Original independent POST-CODE R2 source:** `ad34412bce4498e31c7daeda12088ec93b0d6938`; review request checkpoint `d9a108fe4c5ad2ac73094e3b03c7a813150718b9`.
**Actual prior independent handback in the clinician conversation:** `BLOCK | PARTIAL | P0:P1:P2 = 0:2:0`. Do not infer an independent closure PASS from author CI.
**Frozen AUTHOR-CORRECTED code+test commit:** `36387ec4696de07b847f52fae891a79d3db279a6`. Any later CURRENT/review/PR metadata-only commit is NOT a new corrected code target.
**Prior frozen design:** `7a79ec44ae010cfc68f87c39b6f2bc301fa3cc2c`, contract blob `79696ff1972c12d2156c9c7bc0741db798cf11ad`. Prior PRE-CODE independent final PASS 0:0:0 reused.

## Exact changed source/test paths and blobs

| Path | Corrected blob |
|---|---|
| `clinical_attention_phase_a.py` | `c8dafde7d2dfbbc34d7a7491c31e47706be47817` |
| `static/cockpit/clinical-inbox.js` | `7e3d103b12c737801df45f3605ed4e82f7518c09` |
| `test_cockpit_attention_phase_a.py` | `7d47f3bc6cdf31ab0093405068a74bcb461f61c6` |
| `test_cockpit_synthetic_161.py` | `bbdd73ad77944dede424dcc436d2cfcd228273d9` |

GitHub compare of original request checkpoint `d9a108fe4c5ad2ac73094e3b03c7a813150718b9` to corrected source commit lists **exactly these four files**; no backend authority, schema, provider adapter, task writer, or production route changes. No real patient data may be exercised.

## Prior material findings and scoped correction

**P1-01 — cross-patient re-link could inherit old review/disposition.** Before the fix, a synthetic source reviewed and marked `no_communication` for patient A could be re-linked through the protected API to patient B while retaining that old assessment. Corrected `SyntheticAttentionStore.act(confirm_link)` first requires independently verified exact synthetic patient ID and DOB, then for an already confirmed **different patient** atomically under the existing store lock and expected-version/CAS clears old `review_actor`, `reviewed_at`, `candidate_summary`, sets `review_state=needs_re_review` and `disposition=undecided`, and only then updates the patient link and increments audit/version. Old patient Brief loses the item; the new matching synthetic patient Brief sees one identical item requiring fresh review. Old disposition cannot immediately recur. New regression checks stale version, wrong DOB, idempotent replay, old/new Brief, no premature `no_communication`, fresh review and unchanged Pending.

**P1-02 — current eligible-patient HTTP 401 silently swallowed.** Before the fix, the `/eligible-patients` promise used a blank `.catch` and left a previously authorized synthetic projection/form on screen. Corrected `clinical-inbox.js` validates synthetic response shape, guards `generation + current item + live DOM form` before acting, removes the patient-link form on request error, dispatches existing `clinical:unauthenticated` and `clinical:request-login` for a genuine current 401, and cannot dispatch those events for a superseded 401. The real-Chromium regression at 680px suspends the old eligible-patient request, performs **a real successful login**, releases old HTTP 401 and checks no logout; then injects a current eligible-patient 401 and asserts Inbox+rows+form are purged and existing key panel is opened. No fake identity switching as substitute for login.

## Author evidence — NOT independent reviewer evidence

- [Visit Brief workflow 40061086373](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40061086373) **SUCCESS** at corrected `36387ec...`: 51 existing clinical/temporal tests and **14** protected synthetic/backend/real Chromium tests PASS, plus controller/Chromium shell and diff hygiene.
- [Home 40061086424](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40061086424), [Visit Capture 40061086416](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40061086416), [Surgery 40061086393](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40061086393), [Pending Demo 40061086395](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40061086395), [Canonical Impact 40061086376](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40061086376) — SUCCESS on same corrected source SHA.
- Separate [Physio Knee OA Cockpit 40061086375](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40061086375) now SUCCESS (prior original-source run had an isolated failure). Do not expand this focused review to unrelated Physio architecture.

## Four finite independent closure questions

1. **P1-01 reachable security invariant:** Does actual `confirm_link` with patient A previously reviewed/disposed reject stale expected_version and wrong DOB, preserve stable `item_id`, invalidate old review/disposition/unsent derived candidate, remove A's Brief exposure and show only one safe, needs-re-review item for B? Can `no_communication` be reached before B is reviewed? Does actor/idempotency replay remain exact, with no Existing Pending/task mutation?
2. **P1-02 actual browser/session invariant:** Does a CURRENT `/eligible-patients` 401 fail closed and purge all protected Inbox DOM/form/async state through existing auth owners? Does the OLD 401 delivered after **successful re-login** remain suppressed, without logging out the new session? On invalid payload/non-401 failure, can any old patient-link form remain actionable?
3. **Affected cumulative boundaries:** Recheck only interactions of these corrections with the previously confirmed protected #161 synthetic router, mixed/unknown Inbox-only prohibition, original versioned revision/CAS and shared Visit Brief. Normal Cockpit still has no connected email inbox or fabricated badge. No live PHI, provider, schema/grant, retention, task/booking/communication authorizations were introduced. Reuse unaffected original post-code Q1/Q5 and PRE-CODE PASS.
4. **Provenance:** Verify exact four-file code/test delta `d9a108... → 36387ec...`, source blobs, real 51+14 CI on corrected SHA, unchanged production main, and any docs-only later HEAD. No broader review or unrelated CI demands.

**Required handback:** `PASS | BLOCK | UNKNOWN`; `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`; exact P0:P1:P2 counts; explicit P1-01/P1-02 CLOSED/OPEN; exact code HEAD/blobs/CI evidence; any new reachable material finding and **smallest** correction; **STOP READ-ONLY**.

**Authority:** This focused review has **no** merge, deployment, provider activation, real PHI, clinical persistence/write, migration/grant, outreach, or new review-chain authority. One later separate clinician decision is required for any next implementation/release.
