# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CU-1 CLINICAL-CONTEXT COMPOSITION — IMPLEMENTED / TESTED / PRODUCT REVIEW GATE.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified remote `main`:** `08ecd3ab33e98d567c47042a8a1de482df6952b9`.
> **ACTIVE CANONICAL WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29`.
> **ACTIVE RUNTIME WRITER/LOCK:** `feat/cu1-rich-referral-global-evidence-2026-08-29` — HOLD except for product-owner-requested correction of this slice.
> **Composition implementation commit:** `38f7977811d50636a1585225c74306bef496601c`.
> **Accepted exact-head test commit:** `9b46623b2c991df631698bf018749550dd843f87`.
> **CI evidence:** workflow `CU-1 focused tests`, run `33306399908` / run #394 — compile PASS, browser JavaScript syntax PASS, Python suite 127/127 PASS.
> **Merge/deploy authorization:** NO.
> **Preview deployment:** NOT REQUESTED / NOT AUTHORIZED.
> **Further route-by-route rollout:** HOLD pending product-owner review of the composed referral output.
> **CU-2 / PR-1:** HOLD.

---

# 1. Current state

The initial rich-referral clinical-context serialization defect has been corrected on the active feature branch.

Canonical product rule now has a concrete runtime implementation:

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

# 3. Frozen-shoulder accepted test case

For:

```text
formal diagnosis
primary frozen shoulder
right
pain
active ROM restricted
passive ROM restricted
painful active ROM
painful passive ROM
overhead activity difficulty
lifting/carrying difficulty
driving difficulty
high irritability
```

the runtime output is locked by exact deterministic tests to begin:

```text
Ασθενής με συμφυτική θυλακίτιδα / παγωμένο ώμο δεξιά,
με επώδυνο και περιορισμένο ενεργητικό και παθητικό εύρος κίνησης.
Λειτουργικά αναφέρεται δυσκολία σε δραστηριότητες πάνω από το ύψος του ώμου,
στην άρση ή μεταφορά φορτίου και στην οδήγηση.
Κλινική ερεθιστικότητα: υψηλή.
```

The generic `pain` selection is subsumed only when the complete reviewed painful-ROM fusion matches. Partial selection remains literal/safe and does not invent passive ROM, painful ROM, lifting/carrying or driving.

---

# 4. Test evidence

First implementation CI at `38f7977811d50636a1585225c74306bef496601c` correctly exposed one obsolete grammatical oracle: a pre-composition test still required nominative `παγωμένος ώμος` although the natural sentence requires accusative `παγωμένο ώμο`.

The obsolete assertion was corrected without changing the desired runtime output.

Accepted exact-head evidence:

```text
head: 9b46623b2c991df631698bf018749550dd843f87
workflow: CU-1 focused tests
run id: 33306399908
run number: 394
compile: PASS
browser JavaScript syntax: PASS
Python focused suite: 127/127 PASS
```

---

# 5. Status matrix

```text
DESIGNED                   YES
IMPLEMENTED                YES
TESTED                     YES
PRODUCT-OWNER REVIEWED     PENDING
MERGED                     NO
DEPLOYED                   NO
PREVIEWED                  NO
PRODUCTION-SMOKE-VERIFIED  NO
```

---

# 6. Explicit HOLD

Do not:

- merge to `main`;
- deploy;
- create preview service;
- resume new disease-route rollout;
- change persistence;
- reopen CU-2 or PR-1;
- alter underlying clinical-record semantics;
- add inferred diagnosis/severity/phase/function.

---

# 7. Exact next action

```text
show actual Short + Detailed outputs to product owner
→ product-owner clinical wording review
→ if accepted: keep slice closed and decide separately whether/when to merge
→ if specific defect identified: bounded correction on same writer branch + exact-head CI
```
