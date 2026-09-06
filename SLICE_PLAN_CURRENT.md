# SLICE_PLAN_CURRENT.md — RF imaging-attachment semantic guard

> **STATUS:** APPROVED / FROZEN — IMPLEMENTATION ACTIVE
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Scope:** reusable Clinical Excellence Clinic Utilities RF attachment validation; not osteoporosis encounter semantics.
> **Slice ID:** `CU-RF-IMAGING-SEMANTIC-GUARD-2026-09-06`.
> **Production base:** `e8bf4bac16eff5e0c2101ec891483b81b14765e1`.
> **Production deploy:** `dep-daei1sh42hec73ccthr0` — LIVE.
> **Branch:** `fix/rf-imaging-attachment-semantic-guard-2026-09-06`.
> **Product-owner approval:** explicit agreement during production smoke to proceed with the bounded attachment-semantic hotfix.
> **Implementation/test authority:** YES — bounded to this frozen slice.
> **PR / merge / deploy / production-config authority:** NONE unless separately granted.

---

# 1. Trigger

Production smoke demonstrated that the current RF v2 upload path validates only that the attachment is a structurally valid PDF. A deliberately unrelated laboratory report was accepted, appended to the generated A.1 package, and the official item-3 declaration was checked.

This means the current behavior proves:

```text
PDF PRESENT
```

but not:

```text
IMAGING REPORT PRESENT
```

The official RF form states that the clinician declares an imaging report confirming the diagnosis is attached. The utility therefore needs a bounded semantic guard before it may automatically stamp that declaration.

---

# 2. Desired clinician outcome

The clinician uploads the required report once. The system should prevent obvious mistakes without pretending it can perfectly diagnose arbitrary documents.

Target flow:

```text
select attachment
→ validate PDF structure/size
→ extract available text ephemerally
→ classify document confidence
→ show immediate status
→ create endpoint independently re-validates
→ stamp item 3 only when allowed
```

---

# 3. Frozen classification contract

Exactly three semantic outcomes:

```text
IMAGING_SUPPORTED
CLEARLY_NON_IMAGING
AMBIGUOUS_OR_UNREADABLE
```

## 3.1 IMAGING_SUPPORTED

Readable extracted text contains strong imaging/radiology evidence such as an imaging modality/report vocabulary. Examples include bounded Greek/English terms for radiograph/X-ray, MRI, CT, ultrasound, DXA or radiology.

Behavior:

```text
automatic acceptance
no extra clinician checkbox required
```

This is not a diagnostic interpretation of the report and does not validate whether its clinical conclusion actually proves the selected RF indication.

## 3.2 CLEARLY_NON_IMAGING

Readable extracted text strongly indicates a different document class and contains no strong imaging signal. First bounded rejection class is laboratory/biochemistry/haematology evidence, using multiple independent markers rather than one incidental word.

Behavior:

```text
fail closed
HTTP 422 on create
browser shows rejection reason
clinician confirmation cannot override this class
```

## 3.3 AMBIGUOUS_OR_UNREADABLE

PDF is structurally valid but text is absent/too limited or lacks enough evidence to classify safely. This includes scanned/image-only reports without text extraction.

Behavior:

```text
explicit clinician confirmation required
confirmation wording: clinician confirms this PDF is the imaging report required for item 3
create endpoint requires confirmation flag
```

No OCR/LLM classifier is introduced in this slice.

---

# 4. Server authority

Browser feedback is ergonomic only. The create endpoint is authoritative.

Server must:

1. enforce existing PDF extension/content-type/size/header rules;
2. open the PDF with the existing PDF library;
3. extract bounded text in memory;
4. classify with deterministic rules;
5. reject `CLEARLY_NON_IMAGING`;
6. reject `AMBIGUOUS_OR_UNREADABLE` unless explicit confirmation is true;
7. allow `IMAGING_SUPPORTED` without confirmation;
8. persist only bounded review provenance such as `auto_supported` or `clinician_confirmed`, never extracted attachment text.

Malformed PDFs that merely begin with `%PDF` must fail rather than reaching PDF assembly as if valid.

---

# 5. Privacy / data minimization

```text
raw uploaded PDF               ephemeral for response assembly
extracted attachment text      ephemeral only
extracted text in logs          FORBIDDEN
extracted text in database      FORBIDDEN
extracted text in repo/tests    synthetic only
```

No user-uploaded clinical document is committed to the public repository.

---

# 6. Browser UX

The imaging card should show one of:

```text
✓ Αναγνωρίστηκε απεικονιστική έκθεση
✕ Το PDF φαίνεται να είναι εργαστηριακή/μη απεικονιστική εξέταση
? Δεν μπορεί να ταξινομηθεί με ασφάλεια — απαιτείται επιβεβαίωση ιατρού
```

For the ambiguous state only, show an explicit checkbox:

```text
Επιβεβαιώνω ότι το επιλεγμένο PDF είναι η απεικονιστική έκθεση που απαιτεί το σημείο 3.
```

Changing the selected file resets prior confirmation and status.

The UI must not claim that the report clinically confirms the diagnosis; it only guards the document type required for attachment.

---

# 7. Deterministic classifier boundary

Use conservative token/phrase sets. Strong imaging terms may include normalized Greek/English variants for:

```text
radiology / radiological / ακτινολογ
radiograph / x-ray / ακτινογραφ
MRI / magnetic resonance / μαγνητικ
CT / computed tomography / αξονικ
ultrasound / υπερηχο
DXA / densitometry / οστική πυκνότητα when used as imaging report vocabulary
```

Strong laboratory markers may include normalized Greek/English variants for:

```text
biochemistry
haematology / hematology
serum / plasma
reference range / τιμές αναφοράς
laboratory / εργαστήριο
validator
mg/dL / mmol/L
calcium / magnesium / phosphate and similar analyte table context
```

Rejection requires multiple laboratory markers and no strong imaging signal. A single incidental laboratory word must not reject an otherwise clear imaging report.

---

# 8. API seam

Add a protected preview endpoint, e.g.:

```text
POST /clinical/clinic-utilities/rf/api/validate-imaging
multipart: imaging_report
```

Response contains only bounded classification metadata:

```json
{
  "status": "imaging_supported | clearly_non_imaging | ambiguous_or_unreadable",
  "requires_confirmation": false,
  "message": "bounded clinician-facing status"
}
```

Do not return extracted document text.

`RFApplicationDraft` gains a boolean confirmation field used only for ambiguous attachments.

---

# 9. Acceptance evidence

Focused synthetic tests must prove:

1. structurally valid radiology-text PDF → `IMAGING_SUPPORTED`;
2. synthetic lab report with several laboratory markers and no imaging terms → `CLEARLY_NON_IMAGING`;
3. scanned-like/textless valid PDF → `AMBIGUOUS_OR_UNREADABLE`;
4. ambiguous attachment without confirmation → create blocked;
5. ambiguous attachment with explicit confirmation → create allowed;
6. clearly non-imaging attachment with confirmation → still blocked;
7. malformed `%PDF` bytes → blocked;
8. extracted text is not returned by preview endpoint and not persisted in application payload;
9. existing official-template A.1/A.2 assembly remains intact;
10. inherited RF/CU-1/G4/G3/G2/G1/C1 regressions remain green.

All fixtures must be synthetic and contain no identifiable patient data.

---

# 10. Out of scope

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

# 11. Release / rollback

Implementation occurs only on:

```text
fix/rf-imaging-attachment-semantic-guard-2026-09-06
```

No merge/deploy authority is implied by implementation approval.

Production remains the known live SHA:

```text
e8bf4bac16eff5e0c2101ec891483b81b14765e1
```

until a separately authorized PR/merge/deploy occurs.

---

# 12. REPLAN triggers

Stop and replan if implementation shows that:

- reliable semantic guard requires OCR/LLM or external document storage;
- browser confirmation cannot be enforced independently server-side;
- PDF text extraction creates a persistence/logging leak;
- the official form requires semantic validation beyond document-type suitability;
- the bounded change would require altering RF procedure-history or osteoporosis encounter ownership.
