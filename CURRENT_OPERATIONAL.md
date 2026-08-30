# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CU-1 CLINICAL-CONTEXT COMPOSITION — IMPLEMENTED / TESTED / PRODUCT-OWNER REVIEWED.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29` — HOLD; no further CU-1 runtime mutation without a new explicit product defect/authorization.
> **Composition implementation commit:** `38f7977811d50636a1585225c74306bef496601c`.
> **Accepted product-review correction head:** `e0f690818c63c146a08a5e508a8123b9059b6b33`.
> **CI evidence:** workflow `CU-1 focused tests`, run `33311066018` / run #398 — compile PASS, browser JavaScript syntax PASS, Python focused suite PASS.
> **Merge/deploy authorization:** NO.
> **Preview deployment:** NOT REQUESTED / NOT AUTHORIZED.
> **Further route-by-route rollout:** HOLD pending an explicit delivery-model decision.
> **CU-2 / PR-1:** HOLD.

---

# 1. Current state

The initial rich-referral clinical-context serialization defect has been corrected on the active feature branch and the resulting frozen-shoulder Detailed referral wording has now been product-owner reviewed.

Canonical product rule:

```text
SELECTION INPUT != REFERRAL TEXT
```

Implemented seam:

```text
validated normalized ReferralDraftV1
→ shared deterministic CU1ClinicalContextComposer
→ reviewed selected-fact fusion/subsumption
→ composed clinical-context sentences
→ existing CU1RichReferralRenderer
→ Short / Detailed referral
```

No patient-specific fact is introduced from diagnosis or route identity alone. Unselected findings/functions remain unrendered. Unsupported combinations are not guessed.

---

# 2. Implemented architecture

Implemented and tested:

- shared `clinic_utilities/physio_clinical_composition.py`;
- versioned normative `clinic_utilities/contracts/cu1_clinical_composition_el_v1.yaml`;
- manifest precedence for clinical-context composition;
- route-data grammar hook for a reviewed problem phrase without route-specific Python branching;
- deterministic priority-ordered finding/function fusion rules;
- explicit `consume` and `suppress_if_matched` semantics;
- residual selected facts preserved through existing clinician-facing Greek labels;
- integration at the formatter → rich-renderer seam;
- exact frozen-shoulder Short/Detailed golden outputs;
- partial-selection/no-invention regression tests;
- existing context-gated `formatter_blocked=true` / `text=null` behavior preserved.

The composer source contains no `adhesive_capsulitis_frozen_shoulder` / frozen-shoulder treatment branch.

---

# 3. Product-owner reviewed frozen-shoulder wording

For the Detailed referral, product-owner review on 2026-08-30 locked two bounded corrections:

1. remove the sentence asserting that exercise/technique selection and dosage are individualized by the physiotherapist from the rendered Detailed referral because it is unnecessary referral prose;
2. replace the generic reassessment wording with an escalation-oriented handoff rule:

```text
Παρότι έχει προγραμματιστεί ιατρική επανεκτίμηση, συνιστάται επικοινωνία με τον θεράποντα ιατρό για νωρίτερη επανεκτίμηση σε περίπτωση επιδείνωσης, εμφάνισης νέων κλινικών ή τραυματικών στοιχείων ή άλλης ουσιώδους μεταβολής της κλινικής εικόνας.
```

The second rule communicates only when earlier medical reassessment should be activated; it does not impose a universal routine follow-up interval on the physiotherapist.

The route-level `reassessment_el` authority and rendered Detailed section now carry the same reviewed wording.

---

# 4. Test evidence

The first product-review correction head exposed one obsolete frozen-shoulder test oracle that still required the removed physiotherapist-dosage sentence. That oracle was corrected to assert the sentence is absent and the new earlier-reassessment communication rule is present.

Accepted exact-head evidence:

```text
head: e0f690818c63c146a08a5e508a8123b9059b6b33
workflow: CU-1 focused tests
run id: 33311066018
run number: 398
compile: PASS
browser JavaScript syntax: PASS
Python focused suite: PASS
```

---

# 5. Status matrix

```text
DESIGNED                   YES
IMPLEMENTED                YES
TESTED                     YES
PRODUCT-OWNER REVIEWED     YES
MERGED                     NO
DEPLOYED                   NO
PREVIEWED                  NO
PRODUCTION-SMOKE-VERIFIED  NO
```

---

# 6. Explicit HOLD

Do not:

- merge to `main` without explicit authorization;
- deploy;
- create preview service;
- resume new disease-route rollout without an explicit delivery-model decision;
- change persistence;
- reopen CU-2 or PR-1 without explicit authorization;
- alter underlying clinical-record semantics;
- add inferred diagnosis/severity/phase/function.

---

# 7. Exact next action

```text
CU-1 reviewed wording slice is closed on the feature branch
→ keep merge/deploy separate decisions
→ decide next product slice explicitly
→ for future disease work, evaluate disease-centered delivery in which one reviewed condition model can project into clinical assessment and physiotherapy referral without duplicating clinical truth
```
