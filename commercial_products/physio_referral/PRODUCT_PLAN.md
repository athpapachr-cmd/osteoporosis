# PRODUCT_PLAN.md — Physio Referral productization

> **STATUS:** KNEE-OA V1 RELEASED; CY_GESY OVERLAY RELEASED; V5 FRESH-MAIN INTEGRATED/TESTED; RELEASE HOLD.
> **Updated:** 2026-09-13 Asia/Nicosia.
> **Parent:** existing CU-1 Physiotherapy Referral foundation.
> **Current diagnosis vertical:** Knee Osteoarthritis only.
> **Current integrated V5 candidate:** `9a7f360745710deacb0ff82f03249723bdfe87d6`.
> **Current integrated V5 gate:** `34736389860` — SUCCESS.
> **Current writer:** none.

---

# 1. Product objective

Build a small clinician-facing paid product that creates a concise, clinically credible and evidence-aware physiotherapy referral quickly, while preserving clinician autonomy and keeping complexity under the surface.

Initial commercial hypothesis remains approximately:

```text
€9.99 / month
```

This is a commercial hypothesis, not validated willingness-to-pay.

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

# 3. Current released architecture

The released Knee-OA product uses:

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
progressive evidence disclosure
```

The jurisdiction layer remains deliberately separate from international evidence.

Hard rules:

```text
local clinical guidance != international evidence state
administrative/reimbursement rule != clinical-efficacy evidence
local difference != silent overwrite
planned != active
suggestion != clinician selection
manual text != structured state
```

The released Cyprus/GeSY overlay does not silently change defaults, selections, referral prose or safety behavior.

V5 remains a candidate until merge/deploy. Its fresh-main integrated test evidence does not itself alter production.

---

# 4. Single-diagnosis rule

The active product vertical remains:

```text
Knee Osteoarthritis only
```

A second diagnosis is **not automatically authorized** by completion of Knee-OA, the jurisdiction overlay, or V5.

Do not add another diagnosis merely to make the product look larger. A second diagnosis requires a fresh bounded Product Owner decision based on workflow/product value.

---

# 5. Frozen product principles

## Minimal surface

Routine use should feel direct and mobile-first. Clinical and evidence complexity belongs underneath progressive disclosure.

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

Preferred architecture remains data-minimizing:

```text
account/preferences may be persisted later if justified
patient-identifiable referral draft need not be persisted
```

Current product release and V5 candidate retain no patient-draft persistence.

Any future persistence, analytics, billing or entitlement work requires separate authority.

---

# 7. Released lifecycle state

Current verified production state:

```text
Knee-OA product released                         yes
CY_GESY overlay released                         yes
explicit production jurisdiction configuration   yes
authenticated CY_GESY production smoke            pass
real clinical pilot                               no
paid conversion / retention validation            no
second diagnosis                                  no
```

Authenticated CY_GESY production smoke evidence remains run `34703453615` — SUCCESS.

---

# 8. V5 fresh-main integrated candidate

The historical reviewed V5 candidate was:

```text
head    a47357c602120d3678e8f2f23b99775e616c79e1
gate    34693751545 — SUCCESS
```

Independent review accepted the product/UX/copy direction but required integration onto current production ancestry before release because `CY_GESY` and shared integrations landed later.

That integration has now been completed from fresh main `8aeb91ae37b83caaa188054128db98e04b638fd8`.

Current candidate:

```text
branch  feat/physio-knee-oa-v5-integration-2026-09-13
head    9a7f360745710deacb0ff82f03249723bdfe87d6
gate    34736389860 — SUCCESS
artifact 10311108002
```

V5 behavior:

- first tap on `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects generic symptom without forced popup;
- second tap opens optional detail;
- `Λειτουργικότητα` retains first-tap chooser;
- weakness second-tap exposes only explicit weakness examination concepts;
- Product Owner-approved simplification removes duplicated `Ατροφία τετρακεφάλου` from weakness second-tap;
- quadriceps atrophy remains available through `Περισσότερα → Εξέταση` as an objective finding;
- weakness badge/count does not count separately selected atrophy;
- bare `Περιαρθρικά` remains absent from visible routine/advanced UI;
- overlapping pain wording is reconciled;
- clinical picture and physiotherapy plan become separate paragraphs;
- trailing `Επιπλέον στόχος:` becomes natural connected prose;
- low-information output remains compact.

The integrated gate proves V5 together with the current jurisdiction overlay and protected Cockpit integration on one exact SHA.

V5 is **not merged/deployed**. Release remains a separate Product Owner decision.

---

# 9. External feedback policy

External feedback remains useful, but its role is explicit.

Formal physiotherapist/receiver evaluation is **not a blocking prerequisite** for V5 or current bounded progress.

The Product Owner may ask clinician colleagues later.

```text
receiver / physiotherapist feedback
= optional later external evidence
!= V5 release gate
!= mandatory immediate acceptance step
!= automatic implementation authority
```

Absence of that feedback is not proof of receiver utility. Equally, the product must not be frozen waiting for feedback deliberately deferred by the Product Owner.

Permanent utility rule:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

---

# 10. Commercial validation sequence

The sequence remains intentionally lightweight:

```text
released Knee-OA + CY_GESY foundation
→ fresh-main V5 integration/test
→ bounded V5 PR review
→ explicit Product Owner release decision
→ if accepted: V5 merge/deploy + authenticated smoke
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

The first paying clinician remains more meaningful than speculative projections.

---

# 11. Review governance

Major product changes may still benefit from separate review axes:

1. Clinical / Evidence
2. Receiving-professional / workflow utility
3. UX / Product
4. Commercial / Product-Market

These are review lenses, not mandatory sequential bureaucratic gates for every bounded refinement.

Technical PASS never proves product value. Conversely, absence of formal receiver review does not prohibit a Product Owner-approved bounded release when clinical/evidence/safety contracts remain unchanged and the change is appropriately tested.

Every substantial review should still ask:

> What should be removed or simplified?

V5 applied that principle by removing duplicate access to quadriceps atrophy from the weakness second-tap while preserving the examination capability under `Περισσότερα → Εξέταση`.

---

# 12. Current next action

```text
NEXT BOUNDED ACTION
= open/review fresh-main V5 integration PR under RELEASE HOLD
```

After exact-head PR verification, an explicit Product Owner release decision is required before merge/deploy.

If release is later accepted:

```text
merge
→ Render deploy
→ authenticated V5 production smoke
→ Product Owner real-device acceptance
→ canonical release closeout
```

Do **not** start a second diagnosis, analytics, billing, patient persistence, Greece/England profile work or a new recommendation surface by default.

No active implementation writer exists.