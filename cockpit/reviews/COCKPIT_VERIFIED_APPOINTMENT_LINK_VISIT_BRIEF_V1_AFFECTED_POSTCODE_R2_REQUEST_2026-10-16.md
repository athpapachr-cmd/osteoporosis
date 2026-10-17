# Cockpit Verified Appointment ↔ Patient & Visit Brief v1 — independent affected POST-CODE R2

**DATE:** 2026-10-16, Asia/Nicosia. **MODE:** fresh independent READ-ONLY, ONE bounded affected-cumulative implementation-fidelity review. No code/test changes, PR mutation, merge, deploy, production/provider processing or real clinical data.

**Repository:** athpapachr-cmd/osteoporosis
**Implementation DRAFT PR:** [#161](https://github.com/athpapachr-cmd/osteoporosis/pull/161), branch `feat/cockpit-verified-appointment-visit-brief-v1-20261016`; its base is the **design-only branch**, not `main`.
**Exact frozen implementation+synthetic-test HEAD:** `f3b7fb654155907df663b10a00435a71f9d031b2`.
**Activation main:** `4cf219e8f3f254caeb88134577af397afc229410`.
**Corrected pre-code design-bearing HEAD:** `199e37c28024391387546fb662350dbabc2b5c1b`.
**Frozen corrected design blob:** `2015dd1db0e2fd7c12a26664be456da10c77f6d7`.
**Design branch handoff checkpoint:** `f4a585aaa91cd2107af21fa9ee1188d161f33353`.
**Earlier independent two-P1 pre-code DELTA+CUMULATIVE closure:** **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0, P1-01 & P1-02 CLOSED**, as delivered in the Product Owner conversation. This review does **not** retest the settled design or unrelated Visit Brief+Clinical Inbox / Unified Visit V3 / accepted Home #160. The underlying corrected design still expressly withholds automatic one-click patient-history release when Reception's attendee-continuity semantics remain unqualified.

## Fixed code/test blob oracle (reverify against the frozen implementation HEAD)

| Owner | Exact blob SHA |
| --- | --- |
| `clinical_visit_brief.py` | `b1489752e02866de2bf35bf7bcf5104fc7310456` |
| `clinical_calendar.py` | `be991cc4f47d8fc689250ad67643bf499f114f1b` |
| `static/cockpit/index.html` | `651e32d963eee2df39a4d259868248646460802c` |
| `static/cockpit/clinical-workspace.js` | `0ba7a2c33455b6b056a3981c1a377a96159436f0` |
| `static/cockpit/visit-brief.js` | `4b2582cc3311a3576167ccc9ca2db7667984d1d8` |
| `static/cockpit/clinical-workspace.css` | `f02aa880b4e141a6d4a495d4646a445526360d9f` |
| `test_clinical_visit_brief.py` | `095cc673c242fcad05c9499f5a186975d4e304ab` |
| `test_cockpit_visit_brief_ui.js` | `cf4f4f1f3de4d79ee674f6a1bc817682419ea516` |
| `test_cockpit_doctor_browser.py` | `e2f6de5fa7fa12ce58244efcce200190b29507db` |
| `.github/workflows/cockpit-visit-brief-tests.yml` | `5b1faa13d32563e2ac48427aeff8751f072d6f0a` |

If subsequent review-request / CURRENT commits occur, they are **documentation checkpoints only** and must not replace the code+test HEAD. Inspect source and actual GitHub Actions on the exact frozen code HEAD; never infer run PASS from a link alone.

## Finite gates

**R1 — source authority, persistence and attendee/person safety:** Current Reception `ActualBooking` does not provide qualified immutable attendee identity. The existing Calendar router owns protected Reception source reads. Evaluate unique exact source+UID, 5-minute freshness, Cyprus coverage, unqualified/changed/cancelled/colliding UID fail-closed, observation drift 409, and explicit independent protected registry selection. Versioned booking-specific clinical sidecar and audit must avoid personal-actor attribution under the shared Clinical Key; corrections/revocations use CAS/409, never provider writes or fuzzy name/phone matching. **All later read-intent requests after a manual link remain booking-only**, including exact same UID/name; only the direct successful independent per-opening confirmation response may carry a bounded patient brief. No new reusable session/GET permission or patient prefetch. Provider qualification for *automatic one-click* is NOT established, and must remain a separate release hold, not a fabricated PASS.

**R2 — clinical evidence and temporal truth:** Check `clinical_data.py` unchanged signed Visit Capture Save and `EncounterORM` statuses/ timestamps. The bounded projection must select only a supported signed/completed prior encounter provably authored and not later amended after the *selected booking start*, including same-day timestamp checks, late backfill, selected Cyprus date and DST; signed Visit Capture `signed_at` is **naive UTC** and must not be interpreted as Cyprus wall time. No draft/generic visit type/unsourced legacy as reviewed summary, no fabricated medication intervention. Preserve source roles, verification/uncertainty, clinician-recorded plan. `next_contact.when` and `ClinicalPendingORM.review_date` must equal the selected booking's Cyprus-local date to be displayed as selected-date checks; undated/nonmatching plans historical only. Pending statuses last-recorded **at read**, never historical as-of guarantee or implied current closure.

**R3 — actual UI, races and accepted Home:** Reuse floating `#visitBriefOverlay` and existing Home #160 Previous/Now/Next position and central full-registry search; no layout swap or parallel chart. Independently verify synthetic Chromium at 1280, 680, 390: booking-only click, protected patient search, independent confirmation checkbox, bounded brief, reopening identical UID returns booking-only, central search not obstructed by compact recent rail, keyboard modal focus, direct protected chart, no Dia draft/Save regression. Actual Node controller tests verify stale UID A vs B and late successful confirmation cannot cross sessions/openings; 401/403/503 and close/login invalidate PHI. No browser local/session PHI, URL tokens or raw encounter prefetch.

**R4 — finite synthetic evidence and untouched owners:** Inspect exact-head CI for focused Visit Brief, Cockpit Home/Chromium, Calendar snapshot, Visit Capture and Canonical Impact. New test coverage must be synthetic SQLite/provider/DOM/real Chromium only; no real provider/PHI. Reuse previous passed RF #159, Medical Reports #153 HOLD, Home #160 and Clinical Inbox; no reopening those audits.

**R5 — release boundary:** New clinical link writes OFF by default unless clinic enables `CLINICAL_VISIT_BRIEF_LINKS_ENABLED`. Creation of new SQL tables is not equivalent to a live retention/privacy/role-policy authorization. No new Reception/Cal/Secretary/RF/Gmail provider writes, no automatic AI/CareTasks, no changes to Visit Capture Save. Even an R2 PASS closes only **implementation fidelity**. Automatic repeat one-click access needs separate upstream attendee continuity qualification; production merge/deploy/live identifiable data require explicit Product Owner decision and clinic retention/privacy gate.

## Exact-head GitHub Action runs (verify final conclusions)

- [Focused Visit Brief #38963593262](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38963593262)
- [Cockpit Home / Chromium #38963593625](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38963593625)
- [Clinical Calendar #38963593226](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38963593226)
- [Visit Capture #38963593274](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38963593274)
- [Canonical Impact #38963593357](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38963593357)

Do not infer these ran successfully until the status/conclusion and tested SHA are reverified.

## Required disposition and STOP

Return **POST-CODE R2 PASS / BLOCK / UNKNOWN**, **COMPLETE_FOR_DECLARED_SCOPE / PARTIAL**, P0:P1:P2 and R1–R5 PASS/BLOCK/UNKNOWN with narrow source/line and exact Actions evidence. Distinguish code-correct fail-closed manual fallback from **unqualified automatic one-click release**. For BLOCK prescribe only minimal affected correction; do not start another review of prior accepted Home, Visit Capture, Inbox, RF or Medical Reports.

**STOP — fresh independent READ-ONLY. No implementation, merge, deploy or live-PHI access.**
