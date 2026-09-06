# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 PRODUCTION — PARTIAL SMOKE PASS / IMAGING-ATTACHMENT SEMANTIC GUARD HOTFIX ACTIVE
> **Updated:** 2026-09-06 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Production deploy:** `dep-daei1sh42hec73ccthr0` — LIVE.
> **Active branch:** `fix/rf-imaging-attachment-semantic-guard-2026-09-06`.
> **Implementation base / merge base:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Current frozen slice:** `CU-RF-IMAGING-SEMANTIC-GUARD-2026-09-06`.
> **ACTIVE RUNTIME WRITER/LOCK:** ChatGPT — bounded RF imaging-attachment validation only.
> **ACTIVE CANONICAL WRITER/LOCK:** ChatGPT — bounded operational/slice reconciliation for this hotfix.
> **Implementation/test authority:** GRANTED by product owner in current smoke session.
> **PR authority:** NONE unless separately granted.
> **Merge authority:** NONE unless separately granted.
> **Production config authority:** NONE for this hotfix.
> **Deploy authority:** NONE unless separately granted.
> **Production-smoke authority:** current product-owner smoke session only after any separately authorized release.

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

No config mutation is part of the current imaging hotfix.

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
attachment semantic suitability                     DEFECT FOUND
full A.1 end-to-end production smoke                NOT YET PASS
full A.2 end-to-end production smoke                NOT YET PASS
PILOT-VALIDATED                                     NO
```

The test attachment used during smoke was deliberately unrelated to imaging. It was a laboratory report. The generated A.1 package nevertheless checked the official declaration that an imaging report was attached and appended that laboratory PDF.

Therefore current production only proves:

```text
valid PDF exists
```

not:

```text
valid imaging-report evidence exists
```

---

# 3. Current defect / invariant

Official A.1/A.2 item 3 must not be marked solely because an arbitrary PDF exists.

Required invariant:

```text
PDF STRUCTURE VALID
+
ATTACHMENT SEMANTICALLY SUITABLE OR EXPLICITLY CLINICIAN-CONFIRMED
→ official imaging-attached declaration may be checked
```

Hard safety/privacy rules:

- attachment text is processed ephemerally in memory;
- extracted attachment text is not persisted in PostgreSQL, logs, source or public fixtures;
- clearly non-imaging documents such as laboratory reports are rejected;
- readable documents with strong imaging/radiology evidence may pass automatically;
- scanned/image-only or otherwise ambiguous PDFs require explicit clinician confirmation;
- confirmation never converts a clearly classified laboratory/non-imaging document into acceptable imaging evidence;
- generated application persistence may retain only bounded review provenance, not extracted document text.

---

# 4. Current bounded implementation scope

Allowed mutation:

```text
clinic_utilities/rf/api.py
static/clinic-utilities/rf/index.html
static/clinic-utilities/rf/app.js
static/clinic-utilities/rf/styles.css
focused synthetic RF imaging-validation tests
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
```

Expected behavior:

```text
upload PDF
→ structural validation
→ ephemeral text extraction
→ classify:
   imaging_supported
   clearly_non_imaging
   ambiguous_or_unreadable

imaging_supported
→ create allowed without extra confirmation

clearly_non_imaging
→ reject fail-closed

ambiguous_or_unreadable
→ require explicit clinician checkbox
→ create allowed only when confirmed
```

The create endpoint independently repeats the semantic assessment. Browser validation alone is not authoritative.

---

# 5. Explicitly out of scope

```text
NO OCR service
NO LLM document classifier
NO persistence of raw/extracted attachment text
NO patient-record mutation
NO RF procedure-history semantic changes
NO product/doctor config changes
NO Ortho-Reception mutation
NO PR / merge / deploy without separate authority
```

---

# 6. Exact next action

```text
implement semantic attachment assessment
→ add synthetic radiology / laboratory / scanned-like regressions
→ run full RF gate + inherited regressions
→ exact-head review
→ HOLD for product-owner PR/release decision
```

Lifecycle distinction remains:

```text
IMPLEMENTED != TESTED != PR != MERGED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED != PILOT-VALIDATED
```
