# Cockpit Real Attention Summary — P1-01/P1-02 independent PRE-CODE R2 closure

**Date:** 2026-10-14. **MODE:** ONE fresh, separate independent **READ-ONLY** delta + affected-cumulative **PRE-CODE R2** closure; then STOP. No implementation, merge, deploy, protected/live patient data, database or external-provider operations.

Repository: `athpapachr-cmd/osteoporosis`. DRAFT PR [#152](https://github.com/athpapachr-cmd/osteoporosis/pull/152), `feat/cockpit-attention-source-summary-20261014`. Original independently reviewed design HEAD: `47a25a444d7eec34804df30517e2e2df5c4b4eb8`. Original design blob: `3c6ef3131aeba2a847e4658eaf802ff48e9bed8d`. **Fresh-resolve the current PR HEAD and corrected design blob**; the correction is docs-only. Owning corrected contract: [Real Attention Summary](../COCKPIT_ATTENTION_REAL_SUMMARY_DESIGN_2026-10-14.md).

## Original independent R2 result to close

```
PRE-CODE R2 = BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE
P0:P1:P2 = 0:2:0
P1-01 RF count/label = OPEN
P1-02 surgery stale protected request = OPEN
G3 Medical Reports = PASS (reuse)
G5 scope/owner = PASS (reuse)
```

- **P1-01:** Existing `clinic_utilities/rf/persistence.py::list_rf_application_progress` computes `counts.pending` as `stage == "submitted"` (current awaiting decision). It is not the sum of all previously submitted attempts. `denominator` is issued RF attempts; reported colleague decisions may lack a recorded submission. Earlier design's example incorrectly displayed `Υποβλήθηκαν`.
- **P1-02:** Existing `static/cockpit/app.js::loadSurgeryQueue()` has no async generation guard: older successful `GET /clinical/surgeries` could complete after a newer authorization failure and republish protected singleton name/procedure into `window.CockpitSurgerySummary` / `cockpit:surgery-counts`.

Product Owner expressly authorized one limited correction after this independent BLOCK. This request is **not** a new broad review. Reuse previously closed Home V4 affected R1, RF outcomes pre-code/post-code R2, #151 global Clinical Key/navigation and original G3/G5. Do not restart them.

## Exactly four finite closure questions

**C1 — P1-01 RF source semantics (must CLOSE).** Compare corrected design/examples to actual existing `clinic_utilities/rf/{persistence.py,api.py}`. Does the design label `counts.pending` **Αναμένουν απόφαση** and `denominator` **Εκδοθείσες αιτήσεις**, never `Υποβλήθηκαν`? Are `issued_unknown`/colleague reported decisions treated as unknown submission status, approved+scheduled/performed non-additive, personal responsibility only RF-verified `mine_actionable`? Are nonnegative integer counters, required strict boolean `truncated`, missing/malformed/401/403/503/error and stale RF responses explicitly fail closed, with no false zero or exhaustive total when truncated? No individual RF patient details in Home.

**C2 — P1-02 surgery protected request ordering (must CLOSE).** Examine corrected design and existing `static/cockpit/app.js` source. Does the implementation contract require a strictly monotonic request generation for **the same existing surgery loader**, guarding both success and failure for initial/Clinical Key re-auth/post-write reads and clearing previous personal Home data on auth refresh and latest source failure? Must explicitly prevent A late success after newer 401/403/503 B from restoring any name/procedure/count, and A late failure after newer B success from clearing B. Require non-array/invalid response fail-closed, zero additional reader, no protected identifiers in URL/log/storage, no clinician linkage and single-pending-row-per-date disclosure only.

**C3 — smallest affected cumulative tests/invariants.** Verify the design names finite synthetic regressions for 0/1/2+ same-day surgery, invalid/overdue/undated dates (no implied performed event), Cyprus date-only comparison, protected race/error/reauth, RF stage/truncation/own-followup/absent data, mobile 390/680/1280px and one-tap access. Preserve existing Home V4 partial-calendar label, three recent encounters, disabled Inbox, shared Clinical Key, Visit Capture protected patient selection/Save, medical-report truthful no-ledger state and no external writes. Reuse all previous settled tests rather than rerunning them now.

**C4 — evidence identity / proportionality.** Verify PR #152 diff from original `47a25a444d7eec34804df30517e2e2df5c4b4eb8` to corrected HEAD is **documentation only**, no runtime, test, persistence or provider edits; branch remains DRAFT/unmerged. The correction does not introduce a new store, backend endpoint, surgery reader, RF event or broader reporting ledger. Classify only material residual risk; no new general design round.

## Output and authority

Return **`PRE-CODE R2 PASS | BLOCK | UNKNOWN`**, **`COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`**, **`P0:P1:P2`**, dispositions C1–C4, original P1-01/P1-02 CLOSED/OPEN, exact fresh PR SHA + corrected design blob and smallest remaining blocker, then **STOP**.

If PASS: only the already Product Owner-approved bounded code implementation becomes eligible **in a separate author session**. The correction author cannot independently certify this closure. **No merge, deploy, identified patient use, or outside/provider authorization** follows from PASS.
