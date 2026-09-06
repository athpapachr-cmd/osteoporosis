# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 PRODUCTION — PARTIAL SMOKE PASS / IMAGING-ATTACHMENT SEMANTIC GUARD TESTED / EXACT-HEAD REVIEW PASS / RELEASE HOLD
> **Updated:** 2026-09-06 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Production deploy:** `dep-daei1sh42hec73ccthr0` — LIVE.
> **Hotfix branch:** `fix/rf-imaging-attachment-semantic-guard-2026-09-06`.
> **Implementation base / merge base:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Exact tested clean hotfix head:** `814a62d3b31ae76d19c6f5da3f824e9137011e96`.
> **Exact workflow:** `RF v2 hotfix regression gate`, run `34031607422` — SUCCESS.
> **Current frozen slice:** `CU-RF-IMAGING-SEMANTIC-GUARD-2026-09-06`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — bounded implementation/test/review phase closed.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after this closeout commit.
> **Implementation/test authority:** CONSUMED.
> **PR authority:** NONE unless separately granted.
> **Merge authority:** NONE unless separately granted.
> **Production config authority:** NONE.
> **Deploy authority:** NONE unless separately granted.
> **Production-smoke authority:** separate after any separately authorized release.

---

# 1. Production truth

Native RF v2 was released through PR #75 and correction PR #76.

Current production identity:

```text
main: e8bf4bac16eff5e0c2101ec891483b81b14765e1
PR #76: merged by squash
Render: dep-daei1sh42hec73ccthr0
status: LIVE
```

Server-side production configuration is present for:

```text
RF_PRODUCT_CATALOG_JSON
RF_DOCTOR_PROFILE_JSON
```

The imaging semantic guard described below is **not yet in production**.

---

# 2. Production smoke evidence so far

Product-owner smoke has established:

```text
clinical authentication/session                    PASS
native RF v2 UI visible                             PASS
Category A A.1/A.2 UI                               PASS
product catalog                                     PASS after server config
doctor profile                                      PASS after server config
single-side rule / derived target                   MERGED + DEPLOYED; re-smoke pending
medication capacity 0..3                            MERGED + DEPLOYED; re-smoke pending
Narox/Melox/Panadol/Parcoten parser corrections     MERGED + DEPLOYED; re-smoke pending
A.1 official form generation                        PASS on pre-#76 production smoke
uploaded PDF concatenation                          PASS on pre-#76 production smoke
attachment semantic suitability                     DEFECT FOUND / HOTFIX TESTED
full A.1 end-to-end production smoke                NOT YET PASS
full A.2 end-to-end production smoke                NOT YET PASS
PILOT-VALIDATED                                     NO
```

The smoke attachment was deliberately unrelated to imaging. The generated A.1 package nevertheless checked the official item-3 declaration and appended it, proving that production currently validates PDF presence/structure rather than attachment semantic suitability.

---

# 3. Tested imaging semantic-guard behavior

The bounded hotfix implements deterministic in-memory classification:

```text
IMAGING_SUPPORTED
→ create allowed without extra confirmation

CLEARLY_NON_IMAGING
→ create rejected fail-closed
→ clinician confirmation cannot override

AMBIGUOUS_OR_UNREADABLE
→ explicit clinician confirmation required
→ intended for scanned/image-only or otherwise unclassifiable PDFs
```

Server authority:

- existing extension/content-type/20 MB guards remain;
- PDF must actually parse and contain pages; `%PDF` magic bytes alone are insufficient;
- extracted text is bounded and ephemeral;
- `POST /clinical/clinic-utilities/rf/api/validate-imaging` returns only status / confirmation requirement / bounded message;
- `/api/create` independently repeats semantic assessment;
- the browser cannot override a clearly non-imaging classification;
- only bounded provenance (`auto_supported` or `clinician_confirmed`) may enter the RF application payload;
- extracted attachment text is never returned or persisted.

This classifies document **type suitability only**. It does not claim that imaging findings clinically prove the selected RF diagnosis.

The same branch also corrects stale UI copy so medication capacity is displayed as `0..3` / `έως 3`, matching the already-released backend contract.

---

# 4. Exact automated evidence

Clean exact head:

```text
814a62d3b31ae76d19c6f5da3f824e9137011e96
```

Full gate:

```text
workflow: RF v2 hotfix regression gate
run: 34031607422
result: SUCCESS
```

Passed evidence includes:

```text
Python / JavaScript syntax                          PASS
Official 12-page RF template identity/geometry      PASS
Real packaged-template A.1 generation               PASS
Real packaged-template A.2 generation               PASS
Existing RF v2 focused regressions                  PASS
Synthetic radiology attachment → supported          PASS
Synthetic laboratory attachment → rejected          PASS
Textless valid PDF → confirmation required          PASS
Ambiguous without confirmation → rejected           PASS
Ambiguous with confirmation → accepted               PASS
Clearly non-imaging even if confirmed → rejected    PASS
Fake %PDF bytes → rejected                          PASS
Preview endpoint does not return extracted text     PASS
UI semantic-validation wiring                       PASS
Adjacent CU-1 regressions                           PASS
Legacy RF gateway rollback regressions              PASS
G4/G3/G2/G1/C1 regression ancestry                  PASS
Full branch-vs-production diff hygiene              PASS
```

All committed test documents/data are synthetic.

A private, non-persisted verification using the deliberately unrelated smoke PDF also produced zero imaging signals and multiple independent laboratory signals, so that exact smoke document would be rejected by the hotfix. No content or identifier from that document was committed.

---

# 5. Exact-head review

Compare against current production main:

```text
base / merge base: e8bf4bac16eff5e0c2101ec891483b81b14765e1
head:              814a62d3b31ae76d19c6f5da3f824e9137011e96
behind_by:         0
```

Expected changed files only:

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
clinic_utilities/rf/api.py
static/clinic-utilities/rf/app.js
static/clinic-utilities/rf/index.html
test_rf_v2_native.py
test_rf_v2_unilateral_ui.js
```

Temporary patch workflow is absent from the final diff.

Review result:

```text
scope drift                         NONE
PHI in committed tests/source       NONE
extracted attachment text returned  NO
extracted attachment text persisted NO
OCR / LLM dependency                NO
browser-only enforcement            NO — server independently enforces
release-blocking finding            NONE
```

Known deliberate limitation: a readable document with a strong imaging token is treated as document-type supported even if other text is present. This is a conservative deterministic MVP guard, not clinical report interpretation; deeper content validation would require a separate REPLAN.

---

# 6. Lifecycle matrix

```text
RF #76 CORRECTION IMPLEMENTED/TESTED/MERGED/DEPLOYED  YES
RF #76 PRODUCTION RE-SMOKE                            PENDING
IMAGING SEMANTIC-GUARD DESIGN                         FROZEN
IMAGING SEMANTIC-GUARD IMPLEMENTED                    YES
IMAGING SEMANTIC-GUARD TESTED                         YES @ 814a62d3... / 34031607422
IMAGING SEMANTIC-GUARD EXACT-HEAD REVIEW              PASS
IMAGING SEMANTIC-GUARD PR                             NO
IMAGING SEMANTIC-GUARD MERGED                         NO
IMAGING SEMANTIC-GUARD DEPLOYED                       NO
FULL RF A.1 PRODUCTION-SMOKE-VERIFIED                 NO
FULL RF A.2 PRODUCTION-SMOKE-VERIFIED                 NO
PILOT-VALIDATED                                       NO
```

---

# 7. Release hold / exact next action

Implementation/test/review authority is consumed.

Next possible sequence requires separate product-owner authority:

```text
open bounded imaging semantic-guard PR
→ verify PR-head checks
→ separate merge decision
→ normal Render auto-deploy
→ product-owner re-smoke:
   1. #76 single-side/derived-location behavior
   2. #76 medication parser/capacity behavior
   3. obvious laboratory PDF rejected
   4. real imaging PDF accepted or scanned report explicitly confirmed
   5. inspect final A.1 PDF
   6. exercise A.2 path
```

Until then production remains `e8bf4bac...` / `dep-daei1sh42hec73ccthr0`.

```text
NO PR
NO merge
NO deploy
NO production config mutation
NO claim of full RF production-smoke verification
```
