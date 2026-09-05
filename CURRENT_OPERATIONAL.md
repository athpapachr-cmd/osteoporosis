# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 NATIVE CLINIC UTILITY — IMPLEMENTATION ACTIVE / FUNCTIONAL TEST GATE PASS / BINARY TEMPLATE PACKAGING BLOCKER
> **Updated:** 2026-09-05 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Active branch:** `feat/clinic-utilities-rf-v2-native-2026-09-05`.
> **Implementation base:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Current frozen slice:** `CU-RF-V2-NATIVE-2026-09-05` in `SLICE_PLAN_CURRENT.md`.
> **Exact automated-test runtime head:** `d459a25c654eb748484c428ce8f3dd2b43a31efe`.
> **Exact workflow:** `RF v2 native clinic utility`, run `33979595202` — SUCCESS.
> **ACTIVE CANONICAL WRITER/LOCK:** ChatGPT — native RF v2 slice only.
> **ACTIVE RUNTIME WRITER/LOCK:** ChatGPT — native RF v2 slice only.
> **Implementation/test authority:** YES.
> **PR authority:** NONE.
> **Merge authority:** NONE.
> **Production config/secret authority:** NONE.
> **Deploy authority:** NONE.
> **Production-smoke authority:** NONE.

---

# 1. Production baseline / prior RF state

Production remains at:

```text
main / deployed code source:
8aa8b38e3fa9a8f8ba0618868b452b1835be0d47

release origin:
PR #73 — G4 hotfix: authenticated RF gateway
```

The old gateway auth/form leg was later proven in production after the product owner configured `RF_GATEWAY_ACCESS_KEY`:

```text
GET /clinical/clinic-utilities/rf
→ Osteoporosis gateway 200
→ upstream RF /rf 200
→ form rendered
```

Full old-form history/create/PDF smoke was not completed before the authoritative RF form changed. The old gateway remains the currently deployed implementation until a later native RF release is separately authorized, merged and deployed.

```text
OLD GATEWAY AUTH + FORM LEG            PASS
OLD GATEWAY FULL END-TO-END SMOKE      NOT COMPLETED
OLD RF FORM FIDELITY                   OBSOLETE AFTER OFFICIAL FORM CHANGE
G4 PILOT VALIDATION                    NO
```

---

# 2. Approved ownership replan

The product owner approved migrating RF ownership from `ortho-reception-backend-v2` into the Clinical Excellence runtime.

Target architecture:

```text
/clinical/clinic-utilities/rf
→ native Clinical Excellence RF utility
→ existing ClinicalCookieMiddleware / CLINICAL_DATA_KEY boundary
→ RF-specific persistence on the existing protected database engine
→ official PDF stamping/assembly
```

RF remains a reusable Clinic Utility and is not written into osteoporosis encounter payloads.

No Ortho-Reception runtime/config/secret mutation is authorized or required for this implementation.

---

# 3. Implemented native surface at tested runtime head

At exact runtime head `d459a25c654eb748484c428ce8f3dd2b43a31efe`:

- native Category-A-only RF router is mounted from `clinic_utilities.rf.api`; the legacy gateway is retained as rollback-only code but is not mounted;
- A.1 and A.2 request models/validation are active;
- user-specific indication subset + dynamic Other presets/custom entry are present;
- imaging PDF is required and bounded;
- 4a/4b, VAS/date capture, SI/hip intervention requirements and physiotherapy-date parsing are implemented;
- pasted medication history is deterministically classified/deduplicated and automatically selects up to 3 NSAIDs + 3 other analgesics;
- duplicate drug-family entries prefer the more complete/longer documented trial;
- blank browser numeric fields remain `null` rather than becoming false numeric zero; age zero is rejected server-side;
- RF application requests and actual procedure history remain distinct persistence objects;
- A.2 can use existing procedure history or one-time `legacy_manual` procedure backfill;
- the obsolete 10-week repeat rule is not enforced;
- official PDF generation code selects only common+A.1 or common+A.2 pages and appends the imaging report;
- `PyMuPDF` and `pypdf` are declared runtime dependencies;
- native RF UI uses same-origin protected Clinical Excellence routes.

---

# 4. Exact automated test evidence

Workflow:

```text
RF v2 native clinic utility
run: 33979595202
head: d459a25c654eb748484c428ce8f3dd2b43a31efe
result: SUCCESS
```

The first run (`33979443996`) correctly failed 2/14 focused tests and exposed two medication-parser defects: accent-normalized duration output and duplicate same-drug-family selection not preferring the longer documented trial. Both defects were corrected before the successful exact runtime head.

Successful run evidence includes:

```text
Native RF focused Python tests          14/14 PASS
Native RF UI integrity regression       PASS
Native route/dependency assertions      PASS
Adjacent CU-1 regressions               45/45 PASS
Legacy RF gateway rollback regressions  11/11 PASS
G4 workspace regression                 PASS
G3 summary/salience/wiring/cache         PASS
G2 contract/core/live-state/wiring       PASS
G1 core/wiring/UI/WHY-NOW                PASS
C1 authoritative Finish browser/API      PASS
```

Focused RF coverage includes:

- clinical authentication required;
- Category A / A.1 / A.2 only and no doctor secret/profile values exposed in contract responses;
- automatic 3+3 medication selection;
- Greek duration parsing;
- same-active-drug deduplication with longer documented trial preference;
- physiotherapy date sort/dedup/count + ambiguous short-date rejection;
- blank/zero age protection;
- A.1 three-calendar-month requirement;
- mandatory SI intervention documentation;
- legacy procedure dedup/filtering;
- A.2 transition path accepts a recent actual procedure without resurrecting the old 10-week rule;
- synthetic six-page official-form geometry exercises A.1 page assembly (4 form pages + imaging) and A.2 page assembly (3 form pages + imaging);
- native `main.py` route ownership and legacy gateway non-mounting.

Synthetic tests contain no identifiable patient data.

---

# 5. Official PDF evidence / remaining release blocker

Authoritative form supplied by product owner:

```text
Radiotherapy Eligibility Form.pdf
12 pages in the supplied source document
A4 595 x 842 pt pages for the RF form section analyzed
non-fillable / no AcroForm
```

The exact user-supplied PDF was analyzed locally and used for coordinate calibration and visual synthetic A.1/A.2 stamping. Local visual render verification passed for the mapped A.1/A.2 fields.

The runtime expects the authoritative binary at:

```text
clinic_utilities/rf/templates/rf_official_form_v2.pdf
```

That exact binary is **not yet packaged in the branch**. The successful workflow explicitly reported:

```text
Official RF v2 binary template packaged: NO — release packaging blocker remains
```

Therefore the successful automated gate is a functional/code test result, not a complete release-candidate test. No production release should occur until the exact authoritative binary is present and the gate is rerun against that packaged candidate.

---

# 6. Canonical drift still to reconcile before any release PR

Production `main` canonicals and open docs-only PR #74 are stale relative to:

- PR #73 merge/deploy;
- later key configuration and successful auth/form smoke;
- authoritative form change;
- native RF ownership replan;
- current native implementation/test evidence.

Before any release PR, reconcile `TODO.md`, append durable history to `osteoporosis-change-log.md`, and disposition stale PR #74 without rewriting history.

---

# 7. Current lifecycle matrix

```text
RF v2 DESIGN                               APPROVED / FROZEN
RF v2 IMPLEMENTATION                       ACTIVE — binary asset still missing
RF v2 FUNCTIONAL AUTOMATED TEST GATE       PASS @ d459a25c... / run 33979595202
RF v2 RELEASE-CANDIDATE TESTED             NO — exact official binary not in branch
OFFICIAL TEMPLATE IN BRANCH                NO
LOCAL OFFICIAL-PDF VISUAL VERIFICATION      PASS
INDEPENDENT EXACT-HEAD REVIEW              NO
PR                                         NO
MERGED                                     NO
DEPLOYED                                   NO
PRODUCTION-SMOKE-VERIFIED                  NO
PILOT-VALIDATED                            NO
```

`FUNCTIONAL TEST PASS != RELEASE-CANDIDATE TESTED != MERGED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED`.

---

# 8. Exact next action

```text
package exact authoritative RF v2 PDF binary
→ rerun focused + inherited exact-head gate against packaged candidate
→ reconcile TODO/changelog + stale PR #74
→ independent exact-head source/security/scope review
→ HOLD for separate product-owner PR/merge decision
```

Forbidden under current authority:

```text
NO PR
NO merge
NO production config/secret changes
NO Render deploy
NO production smoke
NO Ortho-Reception runtime/config mutation
NO identifiable patient data in source/tests
```
