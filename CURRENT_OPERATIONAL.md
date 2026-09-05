# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 NATIVE CLINIC UTILITY — APPROVED/FROZEN / IMPLEMENTATION ACTIVE
> **Updated:** 2026-09-05 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Active branch:** `feat/clinic-utilities-rf-v2-native-2026-09-05`.
> **Implementation base:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Current frozen slice:** `CU-RF-V2-NATIVE-2026-09-05` in `SLICE_PLAN_CURRENT.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** ChatGPT — native RF v2 slice only.
> **ACTIVE RUNTIME WRITER/LOCK:** ChatGPT — native RF v2 slice only.
> **Implementation authority:** YES — product-owner approved continuation and native ownership architecture.
> **Merge authority:** NONE.
> **Production config/secret authority:** NONE.
> **Deploy authority:** NONE.
> **Production-smoke authority:** NONE.

---

# 1. Production baseline / prior RF state

Production runtime remains:

```text
main / deployed code source:
8aa8b38e3fa9a8f8ba0618868b452b1835be0d47

release origin:
PR #73 — G4 hotfix: authenticated RF gateway
```

Prior Render auto-deploy from that merge was established as live. The server-side `RF_GATEWAY_ACCESS_KEY` was later configured by the product owner. Production logs and product-owner browser evidence subsequently proved the first RF leg:

```text
GET /clinical/clinic-utilities/rf
→ Osteoporosis gateway 200
→ upstream RF /rf 200
→ form rendered
```

The old workflow did NOT complete full history/create/PDF production smoke before a new official RF form was introduced.

Therefore:

```text
OLD GATEWAY AUTH + FORM LEG            PASS
OLD GATEWAY FULL END-TO-END SMOKE      NOT COMPLETED
OLD RF FORM FIDELITY                   OBSOLETE AFTER OFFICIAL FORM CHANGE
G4 PILOT VALIDATION                    NO
```

The old gateway remains the currently deployed implementation until a later native RF release is separately merged/deployed.

---

# 2. New product-owner decision

The product owner explicitly approved migrating RF ownership from `ortho-reception-backend-v2` into the Clinical Excellence runtime.

Target:

```text
/clinical/clinic-utilities/rf
→ native Clinical Excellence RF utility
→ existing clinical auth boundary
→ RF-specific persistence on the existing protected database engine
→ official PDF stamping/assembly
```

The native utility is a reusable Clinic Utility. It is not osteoporosis-specific encounter content.

No Ortho-Reception runtime/config/secret mutation is authorized or needed for this implementation.

---

# 3. Active implementation scope

Frozen scope is defined in `SLICE_PLAN_CURRENT.md` and includes:

- Category A only;
- A.1 new treatment and A.2 continuation;
- approved user-specific indication subset + dynamic Other presets/custom entry;
- required imaging attachment;
- structured 4a/4b, VAS/date capture;
- deterministic automatic 3 NSAID + 3 other analgesic selection from pasted medication history;
- optional adverse effects;
- conditional SI/hip intervention evidence;
- pasted physiotherapy dates → first/last/count;
- A.2 identity/site-aware procedure history and legacy manual backfill;
- separate application-request vs actual-procedure persistence;
- official PDF page selection/stamping and imaging append;
- Clinical Excellence visual language;
- same protected RF browser URL.

Explicit exclusions:

```text
B / Γ implementation
Ortho-Reception mutation
bulk migration of old external RFA database
osteoporosis encounter schema changes
G1/G2/G3/C1 clinical semantics
new cloud service/infrastructure
LLM medication selection
production config
merge
deploy
production smoke
```

---

# 4. Source/evidence anchors

Official form supplied by product owner:

```text
Radiotherapy Eligibility Form.pdf
12 pages
A4 595 x 842 pt
non-fillable / no AcroForm
```

Local analysis copy in the current working environment:

```text
/mnt/data/rf-v2/Radiotherapy Eligibility Form.pdf
```

Repository target for the exact authoritative binary template:

```text
clinic_utilities/rf/templates/rf_official_form_v2.pdf
```

The current GitHub text-write connector cannot upload binary PDF content. This is a known implementation dependency, not authorization to substitute/recreate the official document. If no binary-capable path becomes available, the product owner will perform one mechanical upload of that exact file before final PDF tests/review.

---

# 5. Existing capabilities that must survive

```text
ClinicalCookieMiddleware / CLINICAL_DATA_KEY protection
G4 same-origin Clinic Utilities navigation
CU-1 physiotherapy utility
G3 patient summary / G2 evidence guidance / G1 visit plan
C1 authoritative Finish/finalization
identifier privacy: no identity/GeSY query-string URLs
```

Native cutover must mount exactly one RF route owner. The external gateway must not remain mounted in parallel on the same prefix.

---

# 6. Known canonical drift to close before PR

Production `main` canonicals are stale relative to events after PR #73:

- `CURRENT_OPERATIONAL.md` and old `SLICE_PLAN_CURRENT.md` still described pre-merge gateway state;
- `TODO.md` still lists the old gateway release/config/smoke path as future work;
- open docs-only PR #74 (`f79bcba...`) records only the early post-merge state and is now stale after key configuration, successful auth/form smoke, official-form change and native-ownership replan.

This branch supersedes those active instructions for the RF path. Before this candidate is presented for merge, the stale TODO/changelog state and PR #74 disposition must be reconciled so no future session is directed back toward completing the obsolete old-form gateway workflow.

PR #74 is not merged by this implementation authority.

---

# 7. Current lifecycle matrix

```text
RF v2 DESIGN                         APPROVED / FROZEN
RF v2 IMPLEMENTATION                 ACTIVE
RF v2 TESTED                         NO
OFFICIAL TEMPLATE IN BRANCH          NO
PDF VISUAL VERIFICATION              NO
INDEPENDENT EXACT-HEAD REVIEW        NO
PR                                   NO
MERGED                               NO
DEPLOYED                             NO
PRODUCTION-SMOKE-VERIFIED            NO
PILOT-VALIDATED                      NO
```

---

# 8. Exact next action

```text
implement native RF source + UI + RF persistence
→ calibrate official PDF coordinates locally
→ obtain exact official binary template at frozen repo path
→ focused automated tests + render verification
→ reconcile TODO/changelog + stale PR #74
→ independent exact-head review
→ HOLD for separate product-owner merge decision
```

Forbidden under current authority:

```text
NO merge
NO production config/secret changes
NO Render deploy
NO production smoke
NO Ortho-Reception runtime/config mutation
NO identifiable patient data in source/tests
```
