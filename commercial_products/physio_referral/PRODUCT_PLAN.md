# PRODUCT_PLAN.md — Physio Referral productization

> **STATUS:** KNEE-OA V5.1 + POST-USE RECEIVER REFINEMENTS RELEASED / LIVE / AUTHENTICATED-SMOKE-VERIFIED; `CY_GESY` ACTIVE.
> **Updated:** 2026-09-15 Asia/Nicosia.
> **Parent:** existing CU-1 Physiotherapy Referral foundation.
> **Current diagnosis vertical:** Knee Osteoarthritis only.
> **Current main release commit:** `b962485c741f558121e8daabfcf1d20c84f31f63`.
> **Current authenticated live smoke:** `35019200920` — SUCCESS.
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

V5 and V5.1 are released. The current post-use receiver refinements are also live and authenticated-smoke-verified; the latest release commit is `b962485c741f558121e8daabfcf1d20c84f31f63`.

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

# 8. Current released V5.1 and post-use state

The V5 baseline was released and smoke-verified first. V5.1 then added examination/detail discoverability and production load-order hardening. Two bounded post-use refinements followed without changing the frozen international evidence/default-plan/safety contracts.

Current release chain:

```text
PR #106 → V5.1 merge 8064999ea70e0a90f6073fc6d66a0c8caaba1538
PR #107 → Cyprus / clinical-review refinement a19f4b9d52c3076f715fd864a5f7664d2b140c81
          authenticated live smoke 34957778592 — SUCCESS
PR #109 → receiver prose / chronicity refinement b962485c741f558121e8daabfcf1d20c84f31f63
          authenticated live smoke 35019200920 — SUCCESS
```

The current receiver refinement intentionally keeps prior physiotherapy response unstructured and preserves:

```text
ADMINISTRATIVE PHYSIO ACTIVITY != KNOWN TREATMENT PROGRAM != KNOWN RESPONSE
```

No second diagnosis, evidence-state reclassification, default-plan change, imaging inference, browser persistence or adjacent-owner mutation is implied by these releases.

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

The product is now beyond the V5/V5.1 release gate. The lightweight sequence is:

```text
released Knee-OA V5.1 + bounded post-use refinements
→ Product Owner real-device / real-use evidence
→ explicit decision whether a small clinical/workflow pilot adds value
→ explicit commercial validation
→ first paying clinician
→ 5
→ 10
→ €250 MRR
→ €500 MRR
→ ~€1,000 MRR
```

External colleague/receiver feedback may be inserted wherever useful. It remains evidence, not automatic implementation authority.

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
= use Product Owner real-use evidence to decide whether another small Knee-OA refinement, a pilot/commercial-validation step, or a separately authorized next diagnosis is justified
```

No active implementation writer exists. Do not start a second diagnosis, analytics, billing, patient persistence, Greece/England profile work or a new recommendation surface by default.
