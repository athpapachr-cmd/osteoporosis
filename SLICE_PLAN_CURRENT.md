# SLICE_PLAN_CURRENT.md — RF imaging-attachment semantic guard

> **STATUS:** APPROVED / FROZEN — IMPLEMENTED / TESTED / EXACT-HEAD REVIEW PASS — RELEASE HOLD
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Scope:** reusable Clinical Excellence Clinic Utilities RF attachment validation; not osteoporosis encounter semantics.
> **Slice ID:** `CU-RF-IMAGING-SEMANTIC-GUARD-2026-09-06`.
> **Production base:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Production deploy:** `dep-daei1sh42hec73ccthr0` — LIVE.
> **Branch:** `fix/rf-imaging-attachment-semantic-guard-2026-09-06`.
> **Exact tested clean head:** `814a62d3b31ae76d19c6f5da3f824e9137011e96`.
> **Test evidence:** `RF v2 hotfix regression gate`, run `34031607422` — SUCCESS.
> **Implementation/test authority:** CONSUMED.
> **PR / merge / deploy / production-config authority:** NONE unless separately granted.

---

# 1. Trigger

Production smoke demonstrated that RF v2 validated only that an attachment was a PDF. A deliberately unrelated laboratory report was accepted, appended to an A.1 package and caused the official item-3 imaging declaration to be checked.

Required correction:

```text
PDF PRESENT
!=
IMAGING-REPORT EVIDENCE PRESENT
```

---

# 2. Frozen semantic contract

Exactly three outcomes:

```text
IMAGING_SUPPORTED
CLEARLY_NON_IMAGING
AMBIGUOUS_OR_UNREADABLE
```

## IMAGING_SUPPORTED

Readable extracted text contains strong imaging/radiology vocabulary.

```text
automatic document-type acceptance
no extra confirmation required
```

This does **not** mean the system clinically interprets the report or proves that its findings support the chosen RF diagnosis.

## CLEARLY_NON_IMAGING

Readable text contains multiple strong laboratory/non-imaging markers and no strong imaging signal.

```text
fail closed
HTTP 422 on create
clinician confirmation cannot override
```

## AMBIGUOUS_OR_UNREADABLE

PDF is structurally valid but text is absent/too limited or cannot be safely classified, including scanned/image-only reports.

```text
explicit clinician confirmation required
```

No OCR or LLM document classifier is introduced in this slice.

---

# 3. Server-authoritative implementation

The server now:

1. retains existing extension/content-type/20 MB checks;
2. opens the PDF with PyMuPDF and requires a real parseable document with at least one page;
3. extracts a bounded amount of text in memory only;
4. normalizes text deterministically;
5. classifies using conservative imaging/laboratory phrase sets;
6. rejects `CLEARLY_NON_IMAGING`;
7. requires `imaging_review_confirmed=true` for `AMBIGUOUS_OR_UNREADABLE`;
8. permits `IMAGING_SUPPORTED` without confirmation;
9. persists only bounded review provenance, not extracted text.

Protected preview endpoint:

```text
POST /clinical/clinic-utilities/rf/api/validate-imaging
multipart: imaging_report
```

Response is bounded to:

```text
status
requires_confirmation
message
```

Extracted document text is never returned.

The create endpoint independently repeats the assessment. Browser state is not authoritative.

---

# 4. Deterministic classifier boundary

Strong imaging vocabulary includes normalized Greek/English variants for radiology/radiograph/X-ray, MRI/magnetic resonance, CT/computed tomography, ultrasound and DXA/densitometry.

Strong laboratory vocabulary includes normalized variants for biochemistry/haematology, serum/plasma, reference ranges, laboratory/validator, common laboratory units and representative analytes.

A clearly non-imaging rejection requires multiple laboratory markers and no strong imaging signal. A single incidental laboratory word is insufficient.

Known limitation deliberately accepted for this MVP: a strong imaging token is sufficient for document-type support even if other content exists. This is a guard against obvious attachment-type mistakes, not semantic interpretation of the medical findings.

---

# 5. Browser UX

After file selection the browser calls the protected preview endpoint and shows one of:

```text
✓ imaging document type supported
✕ clearly non-imaging / laboratory document
? ambiguous/unreadable — explicit clinician confirmation required
```

Changing the file resets prior confirmation.

Only the ambiguous state exposes:

```text
Επιβεβαιώνω ότι το επιλεγμένο PDF είναι η απεικονιστική έκθεση που απαιτεί το σημείο 3.
```

The same implementation also corrects stale medication UI copy to reflect the already-authoritative capacity:

```text
0..3 NSAIDs
0..3 other analgesics
```

not a minimum 3+3 requirement.

---

# 6. Privacy / data minimization

```text
raw uploaded PDF               ephemeral for response assembly
extracted attachment text      ephemeral only
extracted text in logs          FORBIDDEN
extracted text in database      FORBIDDEN
extracted text in API response  FORBIDDEN
public test documents           synthetic only
```

`imaging_review_confirmed` is not persisted as raw browser state. Application persistence retains only bounded provenance:

```text
auto_supported
clinician_confirmed
```

---

# 7. Acceptance evidence — PASS

Exact clean head:

```text
814a62d3b31ae76d19c6f5da3f824e9137011e96
```

Workflow:

```text
RF v2 hotfix regression gate
run 34031607422
SUCCESS
```

Proven scenarios:

```text
synthetic radiology-text PDF                 IMAGING_SUPPORTED / PASS
synthetic multi-marker laboratory PDF        CLEARLY_NON_IMAGING / PASS
textless valid PDF                           AMBIGUOUS_OR_UNREADABLE / PASS
ambiguous without confirmation               BLOCKED / PASS
ambiguous with confirmation                  ALLOWED / PASS
clearly non-imaging even when confirmed      BLOCKED / PASS
fake %PDF magic bytes                        BLOCKED / PASS
preview extracted-text leakage               NONE / PASS
official-template A.1/A.2 assembly           PASS
RF focused regressions                       PASS
CU-1 regressions                             PASS
legacy RF rollback regressions               PASS
G4/G3/G2/G1/C1 ancestry                      PASS
branch-vs-main diff hygiene                  PASS
```

All repository fixtures are synthetic.

---

# 8. Exact-head review — PASS

Compared with production main:

```text
base / merge base: e8bf4bac16eff5e0c2101ec891483b81b14765e1
head:              814a62d3b31ae76d19c6f5da3f824e9137011e96
behind_by:         0
```

Expected implementation/canonical files only. Temporary patch workflow was removed before the tested clean head.

Review found:

```text
scope drift                         NONE
committed PHI                       NONE
raw/extracted attachment persistence NONE
API extracted-text disclosure       NONE
browser-only trust                   NONE
external OCR/LLM dependency          NONE
release-blocking finding             NONE
```

---

# 9. Out of scope

```text
OCR
LLM/vision document classification
clinical interpretation of imaging findings
validation that imaging proves the chosen diagnosis
attachment retention/archive
new patient-record writes
RF procedure-history redesign
product/doctor config changes
Ortho-Reception changes
```

---

# 10. Lifecycle / release hold

```text
DESIGN                 FROZEN
IMPLEMENTATION         COMPLETE
TESTED                 YES
EXACT-HEAD REVIEW      PASS
PR                     NO
MERGED                 NO
DEPLOYED               NO
PRODUCTION SMOKE       NO for this hotfix
```

Current production remains:

```text
e8bf4bac16eff5e0c2101ec891483b81b14765e1
dep-daei1sh42hec73ccthr0 — LIVE
```

Next possible sequence requires separate product-owner authority:

```text
open bounded PR
→ PR-head verification
→ separate merge decision
→ normal Render auto-deploy
→ production re-smoke with obvious lab PDF + real/scanned imaging PDF
```

Opening a PR, merge, deploy or production config mutation is not authorized by this slice closeout.
