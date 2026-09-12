# PRODUCT_PLAN.md — Physio Referral productization

> **STATUS:** KNEE-OA V1 RELEASED; CY_GESY OVERLAY RELEASED; V5 TESTED CANDIDATE AWAITS PRODUCT OWNER DISPOSITION.
> **Updated:** 2026-09-12 Asia/Nicosia.
> **Parent:** existing CU-1 Physiotherapy Referral foundation.
> **Current diagnosis vertical:** Knee Osteoarthritis only.
> **Current canonical main:** `273e22a0ea2bc9b2e5a996fac22a0a88eb54d30e`.
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

The released Knee-OA product now uses:

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

The jurisdiction layer is deliberately separate from international evidence.

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

---

# 4. Single-diagnosis rule

The active product vertical remains:

```text
Knee Osteoarthritis only
```

A second diagnosis is **not automatically authorized** by completion of Knee-OA or the jurisdiction overlay.

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

Current product release retains no patient-draft persistence.

Any future persistence, analytics, billing or entitlement work requires separate authority.

---

# 7. Released lifecycle state

Current verified state:

```text
Knee-OA product released                         yes
CY_GESY overlay released                         yes
explicit production jurisdiction configuration   yes
authenticated production smoke                    pass
real clinical pilot                               no
paid conversion / retention validation            no
second diagnosis                                  no
```

The final canonical production smoke is run `34703453615` — SUCCESS.

---

# 8. V5 tested candidate

A bounded post-use UI/prose refinement exists at:

```text
branch  fix/physio-knee-oa-v5-optional-refinement-prose-2026-09-12
head    a47357c602120d3678e8f2f23b99775e616c79e1
gate    34693751545 — SUCCESS
```

V5 addresses observed workflow friction without changing evidence semantics:

- first tap on `Πόνος`, `Δυσκαμψία`, `Αδυναμία` selects the generic symptom without a forced popup;
- second tap opens optional detail;
- `Λειτουργικότητα` retains its chooser;
- weakness choices become clearer and less duplicative;
- bare `Περιαρθρικά` leaves the routine surface;
- overlapping pain wording is reconciled;
- clinical picture and physiotherapy plan become separate paragraphs;
- the trailing `Επιπλέον στόχος:` label becomes natural prose.

V5 is implemented/tested but not merged/deployed. Product Owner disposition is the next bounded product decision.

---

# 9. External feedback policy

External feedback remains useful, but its role is now explicit.

On 2026-09-12 the Product Owner decided that formal physiotherapist/receiver evaluation is **not a blocking prerequisite** for current progress.

The Product Owner intends to ask clinician colleagues for feedback later, but not now.

Therefore:

```text
receiver / physiotherapist feedback
= optional later external evidence
!= V5 release gate
!= mandatory immediate acceptance step
!= automatic implementation authority
```

Absence of that feedback must not be misrepresented as proof of receiver utility. Equally, the product must not be frozen waiting for feedback the Product Owner has deliberately deferred.

Any later colleague feedback should be triaged through the permanent utility rule:

```text
CLINICALLY INTERESTING
!= WORKFLOW-USEFUL
!= RECEIVER-USEFUL
!= WORTH ADDING
```

---

# 10. Commercial validation sequence

The current sequence is intentionally lightweight:

```text
released Knee-OA + CY_GESY foundation
→ Product Owner disposition of V5
→ if accepted: V5 release + authenticated smoke
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

External colleague/receiver feedback may be inserted later wherever the Product Owner finds it useful. It is not a fixed gate in this sequence.

The first paying clinician remains more meaningful than speculative projections.

---

# 11. Review governance

Major product changes may still benefit from separate review axes:

1. Clinical / Evidence
2. Receiving-professional / workflow utility
3. UX / Product
4. Commercial / Product-Market

These are review lenses, not mandatory sequential bureaucratic gates for every small bounded refinement.

Technical PASS never proves product value. Conversely, absence of formal receiver review does not prohibit a Product Owner-approved bounded release when clinical/evidence/safety contracts remain unchanged and the change is appropriately tested.

Every substantial review should still ask:

> What should be removed or simplified?

The main product risk remains burying a simple workflow under too many correct ideas.

---

# 12. Current next action

```text
NEXT BOUNDED PRODUCT DECISION
= Product Owner disposition of exact tested V5 candidate
```

If V5 is accepted, use the normal lifecycle:

```text
exact candidate verification
→ PR / merge
→ Render deploy
→ authenticated production smoke
→ Product Owner real-device acceptance
→ canonical closeout
```

Do **not** start a second diagnosis, analytics, billing, patient persistence, Greece/England profile work or a new recommendation surface by default.

No active implementation writer exists until the Product Owner explicitly authorizes the next slice.
