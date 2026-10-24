# Cockpit Home — Top-right recent-item dropdown peeks · exact-head POST-CODE R1

**Date:** 2026-10-24 Asia/Nicosia. **MODE:** fresh independent **READ-ONLY R1 post-code implementation fidelity**, ONE bounded review then STOP. No code changes, PR merge, deploy, provider calls, real patient data or production write. This review examines only the Product Owner's UX correction and directly affected read/privacy/navigation seams, not new Clinical Attention/identity R2 design.

**Repository:** `athpapachr-cmd/osteoporosis`; **Draft PR:** [#209](https://github.com/athpapachr-cmd/osteoporosis/pull/209).
**Frozen tested code+test HEAD:** `064fb2b31cbfe6292be2b809ee7940e94e7b03f8`.
**Base verified main:** `bccaf10aead6394f545ec44f735aa12abcb37fed` (#208 synthetic Inbox LIVE and released).
**Branch:** `fix/cockpit-topbar-peeks-20261024`.
**Changed runtime:** `static/cockpit/doctor-shell.js`, `static/cockpit/obligations-view.js`, `static/cockpit/clinical-inbox.js`, `static/cockpit/index.html`.
**Changed tests:** `test_cockpit_doctor_shell.js`, `test_cockpit_doctor_browser.py`, `test_cockpit_synthetic_161.py`, `test_cockpit_cp2.py`. **No** backend, schema, authentication owner, provider/token, clinical identity, CareTask writer or Reception change.

## Product Owner demand / exact scope

The clinician reports ☑ and ✉ top-right buttons now behave as direct full-screen shortcuts, **instead of the requested dropdown with the newest updates/obligations**. Restore the established top-right popover interaction (Calendar and Actions unchanged). The new normal ☑ popover shows **up to 4 most recently updated active Pending obligations from the existing protected `/clinical/clinical-updates?scope=real&active_only=true` first-100-row projection**. Labels say the view is not exhaustive. Clicking a row explicitly opens the same pending task in the **standalone Obligations screen dialog**, not an automatic patient chart. A separate “Όλες οι υποχρεώσεις” opens that same existing full screen. A blocked unsaved clinical draft cannot be discarded by navigation, but the read-only peek may open.

The ✉ icon opens the same dropdown: on the strictly synthetic #161 route, show only 3-4 **synthetic** attention items and explicit full Inbox; in normal Cockpit, preserve the true “no connected real source” state and never invent a badge/count or show unrelated clinical Pending as inbox messages. No live email source, real clinical Inbox, PHI processing or new patient link.

## Finite questions / acceptance evidence

1. **Peek vs navigation:** Clicking ☑/✉ opens the popover first; `aria-expanded`, close/X/Esc/outside click, 1280/680/390 layouts behave. Only explicit recent-item/full-view control navigates; date / actions / chart continue working.
2. **Truthful bounded Pending read:** Only existing protected source, real-vs-#161 account routing and active statuses; max 4 newest by existing `updated_at`, no global counts from first-100 subset, no mixing with surgeries/RF/Gmail, no PHI cache or browser storage. A missing/401/unavailable source fails closed instead of "empty". Preview clicked item resolves against **fresh protected data** and opens the correct existing on-page modal with its current permissions; a missing/status-changed row cannot force stale chart/writer.
3. **Async/auth safety:** Late async response after popover close, another panel, logout, successful same-tab principal change or stale request cannot repopulate patient details. Existing `CockpitObligations` cancel/draft/CAS/auth behavior unchanged; normal synthetic Inbox uses protected #161 route exclusively; no fabricated provider use.
4. **Scope/reuse:** Existing Calendar/Actions, task edit due/category/cancel/undo, patient chart and shared Visit Brief/ClinicalAttention R2-closed logic retained; no backend/schema/task writer/identity authority added. Reject any accidentally changed release function.
5. **Fidelity evidence:** Independent read of the exact changed source/tests, GitHub file compare `main...code`, and already observed CI: [Home 40065474851](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40065474851) **11+ Doctor Shell + actual Chromium PASS**; [Visit Brief/#161 40065474901](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40065474901) **51+14 PASS**; [Visit Capture/CP2 40065474843](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40065474843) **33+37 PASS**, and [Canonical Impact 40065474928](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40065474928) SUCCESS. Earlier corrected failed runs were superseded by tested head; reuse latest evidence. If later PR HEAD docs-only, compare unchanged runtime/test bytes.

**Required verdict:** `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, `P0:P1:P2`, exact inspected HEAD and affected source/blob evidence, five finite dispositions, smallest material fix if BLOCK, and STOP. One R1 review; bounded correction+one affected closure only if material finding. **No merge or deploy authorization** follows automatically; Product Owner chooses separately. 
