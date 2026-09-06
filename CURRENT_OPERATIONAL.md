# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 NATIVE CLINIC UTILITY — MERGED / DEPLOYED — AUTH/UI SMOKE PARTIAL PASS / CONFIG + UX BLOCKERS
> **Updated:** 2026-09-06 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `51714f9c74e96ec4fdf62493a0772ea07fcc8c1a`.
> **Release PR:** #75 — merged.
> **Render deploy:** `dep-dae8doeq1p3s73csqvbg` — LIVE.
> **Current frozen slice:** `CU-RF-V2-NATIVE-2026-09-05`.
> **Exact accepted runtime before squash merge:** `aa2f92cce5d4cd2cfd02cafc59413be7bdc0d5fb`.
> **Exact final branch gate:** `91136c2fabf68edb74ce3a6b586baa02e508d9dc`, workflow run `33992246398` — SUCCESS.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE on `main`; docs reconciliation is staged on `docs/clinical-learning-hub-rf-post-merge-2026-09-06` and intentionally held unmerged until RF smoke closeout is known.
> **Production config/secret authority:** NONE.
> **Production-smoke state:** PARTIAL PASS — authenticated native UI confirmed; end-to-end A.1/A.2 PDF paths not yet proven.

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

# 2. Production smoke evidence observed 2026-09-06

Clinician-run smoke after browser-session restoration confirmed:

```text
AUTHENTICATED SESSION RESTORED                         PASS
NATIVE CLINICAL EXCELLENCE RF UI VISIBLE              PASS
OLD EXTERNAL / GATEWAY UI NOT SHOWN                   PASS
NEW / CONTINUATION WORKFLOW PRESENT                    PASS
CATEGORY-A INDICATION WORKFLOW PRESENT                 PASS
SERVER-CONFIG WARNING VISIBLE                          PASS / EXPECTED FAIL-CLOSED
PRODUCT CONFIGURATION SUFFICIENT                       NO
A.1 END-TO-END CREATE/PDF                              NOT TESTED
A.2 HISTORY/CONTINUATION                               NOT TESTED
```

The observed server-config warning is not an auth failure. At least the product catalog remains missing/incomplete in production. Do not infer doctor-profile configuration state unless contract output or a later smoke proves it separately.

A second finding is a real UI/ergonomics defect, not a browser artifact: after selecting a standard anatomical indication plus laterality, `exact_location` remains manual. The current production JavaScript reads `exactLocation` but contains no deterministic autofill from indication/laterality.

Product-owner requirement from smoke:

```text
STANDARD INDICATION + LATERALITY
→ prefill exact_location deterministically
→ keep exact_location clinician-editable
→ later manual edit must not be silently overwritten
```

Minimum explicit example:

```text
KNEE_OA_KL34 + right  → Δεξί γόνατο
KNEE_OA_KL34 + left   → Αριστερό γόνατο
KNEE_OA_KL34 + bilateral → Αμφότερα γόνατα
```

For sites where laterality does not fully specify the anatomical target (for example Morton neuroma), any autofill must remain a partial suggestion and the more specific site must remain clinician-entered/reviewed.

---

# 3. Proven before release

The accepted candidate proved Category-A A.1/A.2 workflow, exact official PDF identity and A.1/A.2 generation, required imaging append, exact 3+3 medication fail-closed validation, deterministic medication/physio parsing, separate application-request versus actual-procedure history, `clinician_manual` provenance, data minimization and inherited CU-1/G4/G3/G2/G1/C1 regressions.

Hard invariant:

```text
RF APPLICATION REQUEST
!=
ACTUAL RF PROCEDURE
```

---

# 4. What remains unproven / blocked

```text
A.1 END-TO-END PRODUCTION CREATE/PDF            NO
A.2 HISTORY/CONTINUATION PRODUCTION PATH         NO
OFFICIAL GENERATED PDF VISUAL CHECK IN PROD     NO
PRODUCT CATALOG CONFIG SUFFICIENCY               BLOCKED / MISSING OR INCOMPLETE
DOCTOR PROFILE CONFIG SUFFICIENCY                NOT YET SEPARATELY VERIFIED
EXACT-LOCATION AUTOFILL UX                       MISSING IN RELEASED UI
PILOT VALIDATION                                 NO
```

Do not infer these states from automated tests or LIVE deploy status.

---

# 5. Configuration source discipline

Native RF intentionally fails closed unless exact server-side configuration is present:

```text
RF_DOCTOR_PROFILE_JSON
RF_PRODUCT_CATALOG_JSON
```

The prior Reception implementation contains explicit Medikey/DIROS/Thermedico code/description/quantity values and can be used as a candidate migration source, but they must be reviewed as still-current before production config is changed. Do not invent or silently infer configuration values.

No production config mutation is authorized by this operational record.

---

# 6. Exact next sequence

```text
1. confirm current Medikey / DIROS / Thermedico code-description-quantity values
2. separately authorize and set RF_PRODUCT_CATALOG_JSON (and doctor profile only if still missing)
3. bounded exact-location autofill hotfix from production main
4. regression test: autofill + clinician override + no overwrite
5. merge/deploy only under explicit release authority
6. repeat authenticated A.1/A.2 production smoke
7. visually inspect generated official PDF
8. record PASS / remaining bounded blocker
9. merge docs-only canonical reconciliation
```

The configuration fix and UI hotfix are distinct changes and should remain distinguishable in evidence.

---

# 7. Approved future Clinical Learning Hub

Detailed future architecture is recorded in `CLINICAL_LEARNING_HUB_DESIGN_V1.md` and the phase plan. It combines Foundation Map, weekly Clinical Challenges, Daily Heidi-backed Real-Case Review, longitudinal Signals and targeted/spaced learning. Planning only; no learning runtime writer is active.

Visible daily AI coaching is an intervention and remains shadow/hidden during the scored 30-case baseline by default unless methodology is explicitly replanned.
