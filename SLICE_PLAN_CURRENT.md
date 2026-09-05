# SLICE_PLAN_CURRENT.md — Clinic Utilities RF v2 Native Ownership

> **STATUS:** APPROVED / FROZEN — IMPLEMENTATION AUTHORIZED / ACTIVE
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Scope:** reusable Clinical Excellence Clinic Utilities, not osteoporosis clinical encounter semantics.
> **Slice ID:** `CU-RF-V2-NATIVE-2026-09-05`.
> **Production base:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Branch:** `feat/clinic-utilities-rf-v2-native-2026-09-05`.
> **Product-owner architecture approval:** explicit agreement to migrate RF ownership into the Clinical Excellence runtime.
> **Implementation authority:** YES — bounded to this frozen slice.
> **Merge / deploy / production config / production smoke authority:** NONE unless separately granted.

---

# 1. Trigger / new evidence / REPLAN reason

The prior G-4 RF hotfix solved only the cross-service authentication transport problem. It was merged and deployed from PR #73, and production evidence later proved the authenticated RF form path could return `200 OK` through the gateway.

Before full create/PDF smoke completed, the external authoritative RF form changed materially. The new official form is a 12-page non-fillable PDF with Category A/B/Γ and separate A.1/A.2 workflows. The existing external RF implementation still models the old form and therefore no longer represents the requested clinical-administrative workflow.

Product-owner use is narrower than the complete official form:

```text
Category A only
→ A.1 new treatment
OR
→ A.2 continuation after a previous actual RF treatment
```

B and Γ are not part of this clinician's ordinary RF workflow.

A second product problem also became explicit: the external RF page has a separate visual language and feels like a different application rather than a native Clinical Excellence Clinic Utility.

The product owner approved the architecture change:

```text
RF business/UI/PDF/history ownership
FROM ortho-reception-backend-v2
TO   osteoporosis / Clinical Excellence Clinic Utilities
```

This is a REPLAN because it changes the runtime owner, persistence boundary and release dependency. It is not an authorization to mutate Ortho-Reception.

---

# 2. Desired clinician outcome

From the Clinical Excellence Cockpit the clinician can complete the current official Category-A RF pre-approval form quickly, with minimal duplicate typing:

```text
same protected Clinical Excellence session
→ native RF utility
→ choose NEW or CONTINUATION
→ enter/paste only variable patient/treatment evidence
→ deterministic automation for medication and physiotherapy history
→ generate the correct official A.1 or A.2 PDF pages
→ append the required imaging report
→ retain RF history needed for later A.2 use
```

The utility should reduce clicks and transcription work without inventing clinical facts, treatments, doses, durations, dates or treatment outcomes.

---

# 3. Chosen architecture

Final target:

```text
authenticated Clinical Excellence browser
→ /clinical/clinic-utilities/rf
→ native RF router/service
→ separate RF persistence tables on the existing protected database engine
→ official RF PDF stamping + imaging-report append
```

The active route no longer requires:

```text
RF_GATEWAY_ACCESS_KEY
X-RF-Key
external RF cookie/session
HTML transport rewriting
fixed upstream RF service
cross-service error translation
```

The old gateway implementation may remain in source during the candidate as rollback/reference, but `main.py` must mount exactly one RF owner. Native and gateway writers must never be active simultaneously on the same browser route.

Rollback before/after release remains the previous known-good production SHA `8aa8b38e...` until a later release proves a newer known-good identity.

---

# 4. Alternatives considered

## A. Keep RF in Ortho-Reception and rebuild it there

Rejected for this slice because RF is not appointment/voice/reception semantics, would remain coupled to an unrelated runtime/release train, would still require cross-service authentication and would make visual integration harder.

## B. Native Clinical Excellence Clinic Utility — CHOSEN

Advantages:

- same authentication/session boundary as the Cockpit;
- same Clinic Utilities package and visual language as physiotherapy;
- one runtime owner for UI/API/PDF/history;
- no RF credential in the active path;
- no dependency on the active Ortho-Reception implementation writer;
- easier future addition of other clinic utilities without misusing osteoporosis encounter storage.

## C. Create a third standalone RF service

Rejected as unnecessary infrastructure. It would reproduce the cross-service auth/deploy problem without a demonstrated product need.

---

# 5. Ownership and data boundaries

Canonical owner after this slice:

```text
Clinical Excellence Clinic Utilities RF module
```

It owns:

- RF web UI;
- Category-A workflow contract;
- deterministic validation;
- deterministic medication/physiotherapy parsing;
- RF-specific protected persistence;
- official PDF page selection/stamping;
- generated PDF response.

It does NOT own or modify:

- osteoporosis Clinical Guidance rules;
- osteoporosis encounter payload semantics;
- G1/G2/G3 clinical guidance state;
- C1 finalization semantics;
- Ortho-Reception appointment/voice semantics;
- Ortho-Reception runtime/config/secrets;
- B/Γ workflow implementation in this first release.

Hard data separation:

```text
RF application / RF procedure history
!=
osteoporosis clinical encounter data
```

RF records may share the protected SQLAlchemy engine but use RF-specific tables/models.

---

# 6. Official-form scope

The authoritative input is the newly supplied official non-fillable RF eligibility PDF.

First release supports only:

```text
Page 1 common clinician/patient/product/category data
A.1 pages 2-4
A.2 pages 5-6
```

Generated packages:

```text
A.1 → official pages 1-4 + uploaded imaging-report PDF
A.2 → official pages 1,5,6 + uploaded imaging-report PDF
```

Omitting unused B/Γ pages is an approved workflow/output decision for this clinician-facing utility; the underlying official pages themselves are not recreated or rewritten.

PDF form has no AcroForm fields, so implementation uses calibrated non-fillable stamping. Exact coordinates must be derived and visually verified against the supplied PDF before the candidate can be considered tested.

---

# 7. Fixed clinician and product data

Doctor details are invariant for this user and must not require repeat entry.

Because the repository is public, personal clinician identifiers/contact details are not committed. Runtime obtains them from one protected server-side configuration object, e.g. `RF_DOCTOR_PROFILE_JSON`. PDF generation fails closed if required doctor fields are absent.

Products remain exactly the three established choices:

```text
Medikey
DIROS
Thermedico
```

Product code / required description-supplier text / quantity must be supplied from one authoritative protected server-side product catalog (e.g. `RF_PRODUCT_CATALOG_JSON`) or an equally bounded server-side config. The UI selects the provider/product once; known product metadata is auto-filled. Unknown product metadata must not be invented.

Production config values require separate config authority and are outside implementation authority.

---

# 8. A.1 new-treatment workflow

## 8.1 Allowed indications

Routine UI presents only the clinician's actual use:

```text
KNEE_OA_KL34
  Γόνατο → OA Kellgren-Lawrence 3ου/4ου βαθμού

SI_DEGENERATIVE
  Ιερολαγόνια → εκφυλιστική παθολογία

HIP_OA_KL34
  Ισχίο → OA Kellgren-Lawrence 3ου/4ου βαθμού

MORTON_NEUROMA
  Νευρίνωμα Morton

SHOULDER_OA_KL34
  Ώμος → OA Kellgren-Lawrence 3ου/4ου βαθμού

SHOULDER_IRREPARABLE_CUFF
  Ώμος → μη χειρουργικά αποκαταστάσιμη εκτεταμένη ρήξη στροφικού πετάλου

OTHER_LATERAL_EPICONDYLITIS
  Άλλο → αγκώνας / έξω επικονδυλίτιδα

OTHER_DEQUERVAIN
  Άλλο → καρπός / De Quervain

OTHER_CUSTOM
  Άλλο → clinician-entered περιοχή + διάγνωση
```

Chronic-postoperative options and the A.1 item-2 operation/date fields are hidden and unsupported in this first workflow because the product owner does not use them.

## 8.2 Imaging item 3

The official imaging-attached declaration is derived from a real required uploaded PDF. The generated official form is checked only after valid PDF upload succeeds.

No synthetic `yes` without an attachment.

## 8.3 Item 4a — RF rationale

Clinician-controlled multi-select presets:

```text
failed pharmacologic treatment
failed conservative treatment
patient does not want surgery
major comorbidity / high surgical risk
other free text
```

The UI may offer a fast combined preset for the common `failed pharmacologic + conservative treatment` case, but it must remain an explicit clinician action rather than a silently asserted default.

Selected reasons are deterministically composed into the official free-text field.

## 8.4 Item 4b — exact site

Structured laterality + anatomical site generates an editable exact phrase, e.g. `Αριστερός ώμος`, `Δεξιός αγκώνας`, `Αριστερός καρπός`.

The final text field remains editable because the official form requests exact application location.

## 8.5 Item 5 — pain

Required structured fields:

```text
pain_onset_date
pain_onset_vas 0..10
last_assessment_date
last_assessment_vas 0..10
```

The official >=3-month persistence requirement is validated deterministically from entered dates. No date or VAS value is inferred.

## 8.6 Item 6 — medication automation

Primary time-saving contract:

```text
paste complete medication history
→ deterministic entry parsing
→ medication classification
→ canonical ingredient/brand deduplication
→ automatically select up to 3 NSAID trials
→ automatically select up to 3 other analgesic trials
→ extract dose/duration only when explicitly present
→ clinician intervenes only for missing/ambiguous values or corrections
```

The first release uses one deterministic server-side classifier/catalog. No LLM is required.

Selection ranking favors entries with explicit usable evidence such as dose/duration while preserving source truth. It must not invent medication, dose, frequency or duration.

Corticosteroid/local-anesthetic injection is not one of the 3+3 medication rows; it belongs to item 8 where applicable.

## 8.7 Item 7 — adverse effects

Optional, collapsed by default. Clinician may add treatment + adverse effect only when relevant. Never auto-write `none` or fabricate an adverse effect.

## 8.8 Item 8 — interventions

Routine SI path:

```text
laterality / exact application site
injection date
VAS before
VAS after
```

Official intervention is corticosteroid/local anaesthetic injection.

Hip remains selectable because it is an official A.1 indication, but the clinician usually refers these cases to a pain clinic. If HIP is selected for an RF application, the official diagnostic-block data become required. The utility must block final PDF creation rather than silently leave the official requirement incomplete.

Facet is not offered in this clinician's first-release indication list.

## 8.9 Item 9 — physiotherapy

Primary input is pasted session dates.

Deterministic parser:

```text
extract supported dates
→ normalize
→ reject/flag malformed ambiguity
→ deduplicate
→ sort
→ derive first date
→ derive last date
→ count sessions
```

Derived values remain reviewable/editable before PDF generation.

## 8.10 Item 10

Free clinician text only.

---

# 9. A.2 continuation workflow

A.2 uses the same supported indication/site model as A.1.

The official prior-treatment fields are:

```text
previous_actual_application_date
previous_vas_before
previous_vas_after
last_followup_date
last_followup_vas
```

The legacy hard-coded `10 week remission` rule from the old RF implementation is explicitly NOT carried forward because the new supplied A.2 form does not contain that requirement.

Item 3 imaging is again derived from a valid required upload.

A.2 item 4 is free text.

---

# 10. RF history / transition-period data model

Identity is the primary patient lookup key, but never the unique key of a procedure episode.

Required model distinction:

```text
RF APPLICATION REQUEST
!=
ACTUAL RF PROCEDURE
```

Creating a PDF/application does not prove a procedure happened.

Use separate RF persistence domains such as:

```text
clinic_rf_applications
clinic_rf_procedure_history
```

`clinic_rf_applications` records generated requests/application evidence.

`clinic_rf_procedure_history` records actual prior treatment evidence needed by A.2, including at least:

```text
patient identity key
site / laterality / exact location
indication
actual procedure date
VAS before
VAS after
last follow-up date
last follow-up VAS
provenance
created/updated timestamps
```

Transition behavior:

```text
identity lookup
→ matching procedure histories found
   → show/select relevant episode, preferring recent compatible site/side

→ no matching history
   → `Καταχώρηση προηγούμενης εφαρμογής`
   → clinician enters the five A.2 historical values once
   → persist as `legacy_manual` / retrospective provenance
   → reuse on later applications
```

No bulk import from Ortho-Reception is required for this first release. The old external RF database remains untouched. A later one-time import is separate work only if it becomes useful.

---

# 11. Privacy/security invariants

```text
existing clinical_session / X-Clinical-Key protection retained
no identity/GeSY in browser URL/query string
no RF payload in public logs/source fixtures
no patient data in repo
no personal doctor profile in repo
no RF provider catalog values invented
no external RF credential required by active path after cutover
no Ortho-Reception secret/config/runtime mutation
uploaded imaging kept only as long as needed to assemble the response unless a separately approved retention design exists
```

Generated PDF/application identifiers must be opaque/bounded and protected by the existing clinical auth boundary.

---

# 12. UI / Clinical Excellence design contract

RF is visually native to Clinic Utilities and reuses the established physiotherapy/Cockpit language:

```text
Inter/system sans
#f4f7fb workspace background
white clinical cards
#213b58 / #233a55 primary/nav accents
compact form controls
responsive desktop/mobile layout
clear progress/workflow sections
sticky review/output panel where useful
```

The RF page must not retain the old beige/serif editorial design.

The UI should optimize default common paths for speed:

```text
NEW | CONTINUATION
patient
product
indication/site
rationale
pain
paste medication → auto 3+3
conditional intervention
paste physio dates → auto summary
imaging
notes
review → PDF
```

---

# 13. Implementation seams

Expected bounded code surface:

```text
clinic_utilities/rf/
  __init__.py
  api.py
  models.py
  catalog.py
  medications.py
  physio_dates.py
  persistence.py
  pdf.py
  validation.py as needed
  templates/rf_official_form_v2.pdf  [binary authoritative template]

static/clinic-utilities/rf/
  index.html
  styles.css
  app.js

main.py
requirements.txt
focused RF tests
canonical closeout files
```

The exact module split may remain smaller if fewer files preserve clear ownership; do not add abstraction for its own sake.

Current `clinic_utilities/rf_gateway.py` is not the semantic owner after cutover. It may remain unmounted for rollback/reference in this slice. Do not keep both routers mounted on the same prefix.

---

# 14. Pre-implementation regression-threat / capability-preservation gate

Existing working capabilities to preserve:

| Capability | Current path | Post-slice path | Preservation invariant |
| --- | --- | --- | --- |
| Protected RF navigation | Cockpit → same-origin gateway | Cockpit → native same-origin RF | same URL and clinical auth |
| G4 Clinic Utilities nav | G4 JS | unchanged | physio + RF links remain functional |
| Physiotherapy utility | native CU-1 router | unchanged | no route/style/runtime regression |
| G1/G2/G3/C1 clinical workflow | existing app | unchanged | no clinical semantic/persistence change |
| Identifier URL privacy | POST local history body | POST/JSON native history body | no identity/GeSY query URL |
| RF access control | clinical auth + gateway + RF key | clinical auth only | never weaken protected clinical boundary |
| Official imaging append | external RF PDF assembly | native PDF assembly | uploaded PDF validated and appended |

Invalid cutover:

```text
old gateway unmounted
+
native root/history/create/pdf path not complete
=
RELEASE BLOCK
```

No new infrastructure/runtime service is required.

---

# 15. Smallest sufficient evidence plan

Design broad; test narrow.

Required focused evidence before PR review:

1. native root route protected and renders Clinical Excellence RF UI;
2. B/Γ absent from clinician workflow; only approved A indications exposed;
3. A.1 validation including imaging, >=3-month pain evidence and conditional SI/hip requirements;
4. medication parser selects max 3 NSAIDs + max 3 other analgesics, deduplicates active ingredient and never invents missing dose/duration;
5. physiotherapy date parser derives first/last/count and flags ambiguity;
6. A.2 lookup supports multiple episodes and legacy manual backfill;
7. application-request row never auto-creates an actual-procedure row;
8. identity/GeSY never required in browser URL;
9. correct A.1 page package and A.2 page package generated from supplied official template;
10. visual render verification of stamped fields/checkmarks with no clipping/overlap;
11. imaging PDF appended after selected official pages;
12. existing G4 physio navigation and relevant inherited G3/G2/G1/C1 smoke/regression seam remains intact;
13. `py_compile` / syntax and `git diff --check` or equivalent exact-head checks.

Do not create a broad generic validation program.

---

# 16. Binary-template constraint

The official PDF supplied by the product owner is the authoritative template. The current GitHub connector can mutate UTF-8 source but cannot directly publish binary PDF bytes.

Implementation may proceed around this, but the candidate cannot be declared fully tested/merge-ready until the exact official PDF is present at the frozen repository path and coordinate/render checks run against it.

If no connected binary-upload mechanism is available, the product owner will be asked for exactly one mechanical GitHub upload of the supplied PDF to the specified path. No secret or patient data is involved.

---

# 17. Definition of Done

```text
DESIGN/FROZEN                     YES
NATIVE RF IMPLEMENTED             YES
OFFICIAL TEMPLATE PRESENT         YES
FOCUSED TESTS                     PASS
PDF VISUAL VERIFICATION           PASS
EXACT-HEAD REVIEW                 PASS
OLD GATEWAY NOT ACTIVE            YES
CANONICAL CONTRADICTIONS CLOSED   YES
PR                                OPEN / REVIEWABLE
```

Release states remain separate:

```text
MERGED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED != PILOT-VALIDATED
```

Production config, merge, deploy and smoke require separate product-owner authority.

---

# 18. REPLAN triggers

STOP and REPLAN if implementation proves any of the following necessary:

- B or Γ must be supported for the accepted outcome;
- a separate runtime/service is required;
- Ortho-Reception must be mutated for correctness;
- a second source of RF procedure truth is unavoidable;
- the new official PDF cannot be safely stamped/assembled as designed;
- medication automation requires probabilistic/LLM inference to meet the accepted outcome;
- patient identifiers would need URL transport;
- RF history must be mixed into osteoporosis encounter payloads;
- a generated application must be treated as evidence of actual procedure;
- scope materially expands beyond one coherent native RF utility PR.

---

# 19. Current authorization / exact next action

```text
PRODUCT-OWNER DESIGN APPROVAL      YES
IMPLEMENTATION AUTHORITY           YES / ACTIVE
CANONICAL WRITER                   ChatGPT — this RF v2 branch only
RUNTIME WRITER                     ChatGPT — this RF v2 branch only
MERGE AUTHORITY                    NONE
DEPLOY AUTHORITY                   NONE
PRODUCTION CONFIG AUTHORITY        NONE
PRODUCTION SMOKE AUTHORITY         NONE
```

Next sequence:

```text
complete canonical replan on this branch
→ implement native RF utility
→ add exact official binary template
→ focused tests + PDF render verification
→ independent exact-head review
→ HOLD for separate merge decision
```
