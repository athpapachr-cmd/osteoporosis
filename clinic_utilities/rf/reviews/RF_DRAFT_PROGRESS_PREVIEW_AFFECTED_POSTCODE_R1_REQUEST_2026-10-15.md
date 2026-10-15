# RF live and saved-draft progress preview — one independent affected POST-CODE R1
**Date:** 2026-10-15. **Mode:** fresh READ-ONLY, one bounded affected implementation-fidelity check; no code/merge/deploy.
**Repo:** `athpapachr-cmd/osteoporosis`; **PR:** #158 (DRAFT/UNMERGED).
**Frozen source+test HEAD:** `f82d6ff380469859bd455a2bf764e076dc49863e`; base/main `686a8f68e3330e467a6422056296faa52a8cacfd`. Later changes only this request and checkpoints; verify blob equality.

### Declared affected scope
`static/clinic-utilities/rf/{app.js,index.html,styles.css}`, `test_rf_tracking_browser.py`, `test_rf_v2_ui.js`; reuse released RF #157/API/saved draft authority, no new backend. Inspect actual diff, handlers, browser fixtures, and CI logs.

1. Live creator renders **existing vs missing** A1/A2 fields as an informational, non-authoritative progress view; updates on input/change, pathway, doctor and imaging changes, retains existing PDF validation.
2. Readonly protected draft preview requires selected draft ID and expected revision; no silent mutation of unsaved form, active draft, issuer, attachment choices, clinical status, or issuance; no client storage/URL/external request. Stale asynchronous responses and 401/403 clear protected preview/list.
3. Missing binary imaging in saved drafts is correctly marked as requiring reattachment; no imagined persisted attachment/old PDF. Optional medications/physio/Horos not falsely mandatory; A2 history does not imply proven procedure without real validation; clinical validation remains server authoritative.
4. Form and draft one-tap UX, 390/680/1280 real Chromium synthetic, no horizontal overflow, browser keyboard/readability. Authenticated privacy only; no PII in URL/logs.
5. No touched API/persistence, current RF tracking/calendar, exact-ID deletion, A1/A2 issuance, rejected correction, shared Clinical Key or other Cockpit authority; reuse prior PASS without restarting #157 R2.

**Real exact-head tests:** [Native + Chromium 38789411596](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38789411596) SUCCESS (54 native RF tests, real Chromium three widths, UI). [Hotfix 38789403573](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38789403573) SUCCESS; [CU-1 38789411531](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38789411531) SUCCESS; [Canonical 38789411594](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38789411594) SUCCESS. Also check protected preview invalidation on `clinical:authenticated` and stale draft-list reads. Scope-only red jobs unrelated to RF are not assumed green.

Return `IMPLEMENTATION-FIDELITY PASS/BLOCK/UNKNOWN`, `COVERAGE COMPLETE_FOR_DECLARED_SCOPE/PARTIAL`, P0:P1:P2, exact source/test identities and path/line/CI evidence. If BLOCK, one smallest correction. If PASS, reuse unaffected RF #157 PASS and return PO release handback. Stop. NO author self-certification, no review loop without material finding, no DB/PHI/provider mutation.
