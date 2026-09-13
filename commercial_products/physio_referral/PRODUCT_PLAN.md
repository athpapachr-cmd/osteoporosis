# PRODUCT_PLAN.md — Physio Referral productization

> **STATUS:** KNEE-OA V5 MERGED / DEPLOYED; CY_GESY ACTIVE; AUTHENTICATED V5 SMOKE PENDING EXECUTION.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Parent:** existing CU-1 Physiotherapy Referral foundation.
> **Current diagnosis vertical:** Knee Osteoarthritis only.
> **V5 release runtime:** `8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`.
> **Render deploy:** `dep-daj2398u01pc738ojvkg` — LIVE.
> **Authenticated V5 smoke:** `34737351354` — QUEUED / NOT YET EXECUTED at this reconciliation.
> **Current writer:** none.

---

# 1. Product objective

Build a small clinician-facing paid product that creates a concise, clinically credible and evidence-aware physiotherapy referral quickly, while preserving clinician autonomy and keeping complexity under the surface.

Initial commercial hypothesis remains approximately:

```text
€9.99 / month
```

This remains a commercial hypothesis, not validated willingness-to-pay.

The first goal remains product usefulness, not feature count.

---

# 2. Product positioning

The product is **not** merely a text generator.

Target value proposition:

> Create a clinically useful physiotherapy referral quickly while the interface quietly keeps the selected plan aligned with reviewed evidence, shows uncertainty or disagreement honestly, incorporates reviewed local-system context when useful, and preserves clinician authority.

Value stack:

```text
speed
+ clinical structure
+ deterministic referral output
+ international evidence transparency
+ jurisdiction/local-system context
+ flexible clinician override
+ low cognitive load
```

---

# 3. Current production architecture

The deployed Knee-OA V5 product uses:

```text
CU-1 clinical/runtime foundation
+
Knee-OA deterministic projection
+
international evidence core
+
JurisdictionOverlayV1
+
CY_GESY explicit production profile
+
V5 workflow / presentation layer
+
progressive evidence disclosure
```

Hard rules remain:

```text
local clinical guidance != international evidence state
administrative/reimbursement rule != clinical-efficacy evidence
local difference != silent overwrite
planned != active
suggestion != clinician selection
manual text != structured state
```

The Cyprus/GeSY overlay does not silently change defaults, selections, referral prose or safety behavior. V5 does not change those semantics.

---

# 4. Single-diagnosis rule

The active product vertical remains:

```text
Knee Osteoarthritis only
```

A second diagnosis is **not automatically authorized** by the V5 release. Do not add another diagnosis merely to make the product look larger. Any expansion requires a fresh bounded Product Owner decision based on workflow/product value.

---

# 5. Frozen product principles

## Minimal surface

Routine use should feel direct and mobile-first. Clinical and evidence complexity belongs underneath progressive disclosure.

V5 applies that principle directly:

```text
first tap Pain / Stiffness / Weakness
→ generic symptom selection
→ no forced detail sheet

second tap
→ optional focused detail
```

`Λειτουργικότητα` remains a first-tap chooser because an unqualified generic Function state is not sufficiently useful.

Quadriceps atrophy is preserved as capability but removed from duplicate weakness access:

```text
Ατροφία τετρακεφάλου
→ Περισσότερα → Εξέταση
!= weakness second-tap subtype
```

## Smart reviewed defaults

The Knee-OA starting plan remains:

```text
individualized physiotherapy assessment / active rehabilitation
+ therapeutic exercise
+ progressive strengthening
+ education & self-management
```

Other components remain contextual or clinician-selected.

## Clinician authority

Suggestion != clinician selection.

The product informs and proposes. It does not silently choose treatment.

## Evidence-state integrity

International evidence states remain:

```text
recommended_or_supported
conditional_or_context_dependent
limited_or_insufficient_evidence
guideline_conflict_or_mixed
recommendation_against_routine_use
not_yet_assessed
```

Hard distinctions remain:

```text
INSUFFICIENT != AGAINST
CONFLICT != CONSENSUS
SOURCE YEAR != REVIEW DATE
BROAD RECOMMENDATION != ITEM-SPECIFIC STRONG RECOMMENDATION
LOCAL POLICY != STRONGER INTERNATIONAL EVIDENCE
```

## Deterministic live referral

Routine output remains deterministic:

```text
structured clinician-selected state
+ bounded phenotype/product context
→ deterministic semantic projection
→ deterministic Greek referral
```

No routine LLM generation is required.

## Update governance

No autonomous literature-to-live updates:

```text
surveillance
→ candidate change
→ source review
→ impact classification
→ clinician/Product Owner approval
→ versioned contract
→ tests/review
→ separate runtime release
```

---

# 6. Privacy/product boundary

Current product behavior remains data-minimizing:

- no patient/referral draft persistence in this product;
- no clinical draft state in browser storage;
- no analytics added by V5;
- any future persistence, analytics, billing or entitlement work requires separate authority.

---

# 7. V5 release evidence

Historical reviewed V5:

```text
head  a47357c602120d3678e8f2f23b99775e616c79e1
gate  34693751545 — SUCCESS
```

Independent review required fresh-main integration because later `CY_GESY` and shared Cockpit work had landed.

Final reviewed PR head:

```text
4e4bd2ae40c606562a982b3e38f9f859b49986eb
```

Exact PR-head gates:

```text
V5 integration                  34736919952  SUCCESS
CY_GESY jurisdiction            34736920005  SUCCESS
clinical-sheet v4               34736920059  SUCCESS
prototype                       34736919984  SUCCESS
protected Cockpit integration   34736920081  SUCCESS
evidence design                 34736919959  SUCCESS
CU-1 focused                    34736919945  SUCCESS
```

Final artifact `10311660626`, digest `sha256:003311ab6d45297211d9c5e24bfb26768d26d6c435c9b9069e7fff5acf140277`.

Clinical Learning red checks were adjacent-owner scope-only after substantive/frozen-owner tests passed.

PR #101 was squash-merged to release runtime:

`8cfb22fd2478e7832b9b8642f7ae5241d7e1a267`

Render auto-deploy `dep-daj2398u01pc738ojvkg` reached `live` at that exact commit.

---

# 8. V5 production behavior

The deployed code provides:

- first-tap generic Pain/Stiffness/Weakness without forced popup;
- second-tap optional refinement;
- Function first-tap chooser;
- only two explicit weakness-examination options in weakness second-tap;
- quadriceps atrophy only under `Περισσότερα → Εξέταση` for this access-path question;
- no phantom weakness count from separately selected atrophy;
- pain-location duplication correction;
- paragraph separation in richer referrals;
- connected functional-goal prose;
- compact low-information output.

No evidence reclassification, safety change, second diagnosis or new local treatment authority is introduced.

---

# 9. Production verification state

Current production facts:

```text
Knee-OA V5 merged                              yes
V5 exact runtime deployed                      yes
CY_GESY production profile                     active
Render service startup                         verified
prior CY_GESY authenticated smoke               pass
post-V5 authenticated smoke                     pending execution
real clinical pilot                             no
paid conversion / retention validation          no
second diagnosis                                no
```

Authenticated V5 smoke run `34737351354` is queued awaiting a GitHub Actions runner. Its temporary ops workflow uses the protected repository secret without printing it and sends only generated/non-identifiable state.

A rerun of the previously successful authenticated jurisdiction smoke is also queued, so the current block is runner scheduling rather than an observed application failure. A queued run is not a PASS.

---

# 10. External feedback policy

Formal physiotherapist/receiver evaluation remains optional later external evidence and is not a blocking prerequisite for this V5 release.

```text
receiver / physiotherapist feedback
= optional later external evidence
!= mandatory immediate release gate
!= automatic implementation authority
```

Absence of that feedback is not proof of receiver utility.

Permanent utility rule:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

---

# 11. Commercial validation sequence

The sequence now becomes:

```text
Knee-OA V5 + CY_GESY deployed
→ authenticated V5 live smoke closeout
→ Product Owner real-device/use acceptance
→ explicit decision whether to run a small clinical/workflow pilot
→ explicit commercial validation
→ first paying clinician
→ 5
→ 10
→ €250 MRR
→ €500 MRR
→ ~€1,000 MRR
```

External colleague/receiver feedback may be inserted later wherever useful. It is not a fixed gate.

---

# 12. Current next action

```text
NEXT BOUNDED ACTION
= complete authenticated V5 production smoke and canonical final closeout
```

Do **not** start a second diagnosis, analytics, billing, patient persistence, Greece/England profile work or a new recommendation surface by default.

No active implementation writer exists.
