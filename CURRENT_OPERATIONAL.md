# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 NATIVE CLINIC UTILITY — MERGED / DEPLOYED — AUTHENTICATED PRODUCTION SMOKE PENDING
> **Updated:** 2026-09-06 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `51714f9c74e96ec4fdf62493a0772ea07fcc8c1a`.
> **Release PR:** #75 — merged.
> **Render deploy:** `dep-dae8doeq1p3s73csqvbg` — LIVE.
> **Current frozen slice:** `CU-RF-V2-NATIVE-2026-09-05`.
> **Exact accepted runtime before squash merge:** `aa2f92cce5d4cd2cfd02cafc59413be7bdc0d5fb`.
> **Exact final branch gate:** `91136c2fabf68edb74ce3a6b586baa02e508d9dc`, workflow run `33992246398` — SUCCESS.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE on `main`; docs reconciliation is staged on `docs/clinical-learning-hub-rf-post-merge-2026-09-06` and intentionally held unmerged until RF smoke result is known.
> **Production config/secret authority:** NONE.
> **Production-smoke state:** PENDING clinician-run authenticated verification.

---

# 1. Production release now live

Native RF v2 was squash-merged through PR #75 as:

```text
51714f9c74e96ec4fdf62493a0772ea07fcc8c1a
```

Render auto-deploy followed the normal `main` commit path:

```text
dep-dae8doeq1p3s73csqvbg
status: LIVE
source: 51714f9c74e96ec4fdf62493a0772ea07fcc8c1a
```

The active browser seam is native:

```text
authenticated Clinical Excellence browser
→ /clinical/clinic-utilities/rf
→ native RF router
→ existing clinical auth boundary
→ RF-specific protected persistence
→ official RF v2 PDF stamping + imaging append
```

Legacy `rf_gateway.py` remains unmounted rollback/reference code only.

---

# 2. Proven before release

The accepted candidate proved Category-A A.1/A.2 workflow, exact official PDF identity and A.1/A.2 generation, required imaging append, exact 3+3 medication fail-closed validation, deterministic medication/physio parsing, separate application-request versus actual-procedure history, `clinician_manual` provenance, data minimization and inherited CU-1/G4/G3/G2/G1/C1 regressions.

Hard invariant:

```text
RF APPLICATION REQUEST
!=
ACTUAL RF PROCEDURE
```

---

# 3. What remains unproven

```text
AUTHENTICATED NATIVE RF PRODUCTION UI SMOKE     NO
A.1 END-TO-END PRODUCTION CREATE/PDF            NO
A.2 HISTORY/CONTINUATION PRODUCTION PATH         NO
OFFICIAL GENERATED PDF VISUAL CHECK IN PROD     NO
PRODUCTION CONFIG SUFFICIENCY                    NOT YET VERIFIED BY SMOKE
PILOT VALIDATION                                 NO
```

Do not infer these states from automated tests or LIVE deploy status.

---

# 4. Browser-session recovery

Clearing browser data removes `clinical_session`.

```text
open Clinical Excellence root
→ Patient Registry shows `Clinical access key`
→ enter existing CLINICAL_DATA_KEY locally
→ POST /clinical/login
→ secure HttpOnly SameSite clinical_session restored
→ open /clinical/clinic-utilities/rf
```

Never place the clinical key in URL, repository, chat or screenshot.

---

# 5. Exact next action — authenticated RF production smoke

Minimum first smoke:

```text
restore clinical_session
→ open native RF route
→ confirm Clinical Excellence RF UI, not old external form
→ confirm NEW / CONTINUATION and Category-A indication set
→ confirm no auth/upstream-gateway error
```

Then test A.1/A.2 end-to-end with synthetic/non-identifiable smoke data unless a real administrative workflow is intentionally being performed. Missing doctor/product configuration is a bounded config blocker; do not invent/mutate values without authority.

---

# 6. Approved future Clinical Learning Hub

Detailed future architecture is recorded in `CLINICAL_LEARNING_HUB_DESIGN_V1.md` and the phase plan. It combines Foundation Map, weekly Clinical Challenges, Daily Heidi-backed Real-Case Review, longitudinal Signals and targeted/spaced learning. Planning only; no learning runtime writer is active.

Visible daily AI coaching is an intervention and remains shadow/hidden during the scored 30-case baseline by default unless methodology is explicitly replanned.

---

# 7. Exact sequence

```text
RF production smoke
→ record PASS / bounded blocker
→ merge one docs-only canonical reconciliation after smoke
→ then consider bounded L-0 Clinical Learning Hub design slice
```
