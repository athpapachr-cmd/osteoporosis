# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-28 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified maintenance base main:** `d1716f8ea889a9369367c3bb18e469e9bbfef9f0`.
> **Prior CU-1 runtime implementation:** PR #56 squash-merged as `c1da07f581cf8ccf1159d18bb63c23b674cbe9bd`.
> **Prior CU-1 technical smoke:** authenticated product-owner browser smoke passed load/validate/generate/copy/print/no-persistence checks.
> **New product-quality defect:** generated referral prose is machine-like, contains English/machine-derived wording, and Short vs Detailed are insufficiently differentiated.
> **Current major phase:** bounded CU-1 formatter-quality maintenance.
> **CU-1 status:** REOPENED FOR FORMATTER QUALITY CORRECTION — prior technical smoke remains valid but clinician-facing prose acceptance failed.
> **ACTIVE CANONICAL WRITER/LOCK:** `fix/cu1-greek-human-referral-formatting-2026-08-28`.
> **ACTIVE RUNTIME WRITER/LOCK:** `fix/cu1-greek-human-referral-formatting-2026-08-28`.
> **CU-2:** NOT AUTHORIZED.
> **PR-1 Transcript Intake:** intentionally paused at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

---

# 1. Defect statement

The deployed v1 formatter passed technical end-to-end smoke but failed clinician-facing quality acceptance.

Observed product defects reported by the product owner:

```text
1. referral prose does not read like text a clinician would naturally write
2. generated content contains English / machine-derived wording instead of Greek clinician-facing prose
3. Short and Detailed outputs do not differ meaningfully enough
```

Root cause identified in runtime:

```text
selected machine IDs
→ _humanize_id()
→ underscore replacement / English token exposure
→ section-style serialization rather than natural referral composition
```

This is a clinically meaningful usability defect because the utility's purpose is a ready-to-copy referral, not a structured debug rendering.

---

# 2. Authorized maintenance boundary

Authorized changes:

```text
CU1 formatter language/prose contract amendment
Greek clinician-facing phrase catalog
ShortReferralFormatter prose composition
DetailedReferralFormatter prose composition
formatter-specific tests and synthetic output fixtures
UI labels only if required to prevent machine-English exposure
canonical/changelog reconciliation after verified fix
```

Explicitly out of scope:

```text
clinical taxonomy changes
route ownership/precedence changes
new diagnoses/findings/goals/adjuncts
safety-rule changes
route validation changes
persistence/patient-registry linkage
CU-2 work
PR-1 work
```

Existing validation, gateway, safety and no-persistence invariants remain frozen unless the formatter fix reveals a direct contradiction.

---

# 3. Formatter quality acceptance gate

Before MERGE-READY, executable evidence must demonstrate:

```text
A. no generated referral contains raw machine IDs or underscore-humanized English phrases
B. all generated clinician-facing referral text is Greek except unavoidable standard abbreviations/proper names
C. Short output is compact natural prose, normally 2–4 sentences
D. Detailed output has materially greater clinical/contextual information and a distinct medical-referral structure
E. Short and Detailed preserve identical clinical truth and safety restrictions
F. explicit restrictions remain visible in both modes when clinically material
G. not_assessed/unselected values never become reassuring negatives
H. existing CU-1 gateway/safety/no-persistence tests remain green
I. representative clinician-style synthetic cases pass exact assertions
```

At least these representative cases must be tested:

```text
knee OA
cervical nonspecific pain
lumbar nonspecific pain
shared fracture with restriction
shared muscle injury
postoperative route
```

---

# 4. Exact next action

```text
1. freeze formatter language/prose amendment
2. implement Greek label/phrase authority and natural prose composition
3. update focused tests
4. run exact-head CI
5. independent exact-head review
6. STOP at MERGE-READY or BLOCK
```

No merge/deploy is implied until the acceptance gate is clean.
