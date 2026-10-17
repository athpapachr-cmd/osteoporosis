# Cockpit Visit Brief staff security — limited affected correction R2

Date: 2026-10-17 Asia/Nicosia.
MODE: FRESH INDEPENDENT READ-ONLY POST-CODE R2; one bounded affected correction only.
No code/merge/deploy/configuration/provider/PHI/production DB operations.
Repository: `athpapachr-cmd/osteoporosis`.
Child DRAFT/HOLD PR #163, stacked on DRAFT/HOLD #161.

## Freeze

**Exact corrected CODE+TEST HEAD: `148e96adbf05aead719ad6d6b9c8cefe5f51317e`.**
Any later review-request-only commit is documentation, NOT a substitute code head.
Parent #161 frozen HEAD `3999ea46402090c3d9912ce99995a02dc63c81d0`.
Original #163 tested HEAD `a918d82507f749fb138ac39151a79972ccc8d2b2`.
Reuse original unaffected PASS and the 2026-10-17 initial independent R2 findings; do NOT do a broad re-review.

### Pin blobs
- `clinical_auth.py` `b6f49f2b0e4c1a970f3af809b4e90c32083f169b`
- `clinical_visit_brief.py` `2ab8040052776c7560bd376915d23dfdc7b25057`
- `static/cockpit/clinical-workspace.js` `fee615ac1ecc7965d6227e53f8f659fe8ed94627`
- `static/cockpit/visit-brief.js` `2475bbce293cce9568098d590055c2df39896771`
- `test_clinical_visit_brief.py` `c3ecd843c91fd4ae3b4baeffab7074193fbc5b78`
- `test_cockpit_visit_brief_ui.js` `483bb42bdb6b25fdbd9526947a17cc44a7d45494`
- `test_cockpit_doctor_browser.py` `a54a95689beabe135c31e80ab420f11bae6d0d04`

## Closure targets

**P1-01 — One personal clinician staff key opens all authorized clinician screens, without residual broad cookie after rotation/expiry.**
Inspect `clinical_auth.py` and `test_clinical_visit_brief.py`: a successful clinician staff-key login provides the staff session ONLY; middleware authorizes the existing Clinical Data endpoints for that *live clinician session*. Secretary staff cookie never authorizes Clinical Data. Rotation/expiry/logout must return 401 to both `/clinical/status`, `/clinical/patients`, and Visit Brief. Staff clinician does not require a second credential for Home, Visit Capture, Clinical Data, RF/Reports where the existing Clinical Key cookie was previously accepted. Existing legacy Clinical Key continues its previous views but must never create a staff identity or grant Visit Brief. Do not weaken this distinction just to simplify login.

**P1-02 — Attribution of allowed/denied medical brief and link actions.**
Inspect `clinical_visit_brief.py`: booking-only read, registry search, independently confirmed one-time medical read, link creation/reconfirmation/correction/revocation, and source/revision/correction denials require attributable staff audit. Rejection should not be silently rolled back together with the clinical transaction. Missing/invalid staff identity cannot claim a real principal. Audit history remains clinician-only. Review fail-closed behavior for races and no link/pending deletion.

**P1-03 — Full protected workspace purge on 401 or staff principal switch.**
Inspect `clinical-workspace.js` and `visit-brief.js`: registry results (including names/DOB/IDs), selected patient, Dia preview, hidden modal, outstanding requests, and transient state are cleared; late fetch responses cannot re-populate protected DOM. A valid secretary session hitting a clinician-only legacy endpoint is NOT treated as global staff logout.

**P2-01 — Role-based controls.**
Clinician can access protected chart/revoke controls after verified read. Secretary cannot see clinician-only controls (including booking-only revoked-link button), and server independently denies chart/save/revoke/correct/broad data.

## Synthetic author evidence (not independent verdict)

Exact corrected code/test HEAD `148e96adbf05aead719ad6d6b9c8cefe5f51317e`:
- Visit Brief Python/Node/real Chromium: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39009873079 — SUCCESS
- Cockpit Home Python/Node/real Chromium: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39009873085 — SUCCESS
- Calendar snapshot: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39009873094 — SUCCESS
- Canonical Impact: https://github.com/athpapachr-cmd/osteoporosis/actions/runs/39009873092 — SUCCESS

Check 1280/680/390 browser widths, real clinician controls, protected DOM purge, secretary limitations, old Clinical Key backward compatibility, and user requirement: ONE clinician staff key opens ALL previously Clinical Key-authorized screens plus staff Visit Brief.

## Out of scope / hard release hold

No rewrite of unrelated Home #160, Visit Capture Save, RF, Medical Reports, existing clinical records or Reception source. No alteration to additive reviewed-only SQL migration. The four-visit/elapsed-time automatic purge remains withdrawn.
A5 live readiness stays UNKNOWN until the separately required private DB/TLS/backup/least-privilege/MFA/rate-limit/credential issuance/legal link+audit retention and reviewed migration gates are actually qualified.
Feature switches remain OFF by default. **Neither author CI PASS nor this independent R2 authorizes merging, deploying, migrating production DB or processing real PHI.** Both #163 and #161 remain DRAFT/HOLD.

## Required output

Return PASS/BLOCK/UNKNOWN; COMPLETE_FOR_DECLARED_SCOPE or PARTIAL; P0:P1:P2; P1-01/P1-02/P1-03/P2-01 and A1–A6 affected results, with exact GitHub line evidence and smallest necessary follow-up, then STOP READ-ONLY.
