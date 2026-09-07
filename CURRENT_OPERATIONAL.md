# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-0 — CONTRACT CANDIDATE COMPLETE / MACHINE GATE PASS / ACTIVE-WRITER REVIEW PASS / INDEPENDENT REVIEW PENDING
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified `main` / merge base:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`.
> **Active branch:** `design/clinical-learning-l0-contract-freeze-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L0-2026-09-07`.
> **Detailed design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **Substantive exact tested contract head:** `76d5fba68a3c4f289fe0d8438fbb1be3f5689a09`.
> **Machine evidence:** `Clinical Learning L0 contract gate`, run `34144552849` — SUCCESS.
> **Exact design review:** `CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` — PASS by active writer, explicitly NOT independent.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE DESIGN WRITER/LOCK:** NONE after canonical closeout; branch is review-only pending independent review.
> **Learning runtime implementation authority:** NONE.
> **L-0 merge authority:** HOLD pending independent exact-head review.
> **Patient-data mutation authority:** NONE.
> **Production config/secret authority:** NONE.
> **RF runtime authority:** NONE — RF is production-smoke-verified and closed for now.

---

# 1. Production truth / closed RF context

RF v2 remains production-smoke-verified through PRs #75, #76 and #77. Manual/external signing is accepted; no signature image is stored in the public repository. RF is not an active workstream.

---

# 2. L-0 contract candidate complete

Field-level contracts now exist for:

```text
ClinicalLearningChallengeV1
LearningFactV1 / mandatory Fact Ledger
ProgressiveDisclosureV1
ClinicianReasoningResponseV1
LearningObservationV1
LearningReferenceV1
LearningActionV1
FoundationAssessmentEvidenceV1
FoundationAssessmentAttemptV1
FoundationDomainStateV1
DailyCaseSelfReviewV1
ReviewEvidenceDescriptorV1
DailyCaseReviewV1
LearningDueStateV1
Osteoporosis Foundation Map v1
L-1 persistence / API / UI ownership
```

Machine contract set:

```text
schemas/clinical_learning_core_v1.yaml
schemas/osteoporosis_foundation_map_v1.yaml
schemas/clinical_learning_l1_boundary_v1.yaml
schemas/clinical_learning_contract_manifest_v1.yaml
schemas/clinical_learning_design_fixtures_v1.yaml
```

Validation:

```text
test_clinical_learning_l0_contract.py
test_clinical_learning_l0_boundary.py
.github/workflows/clinical-learning-l0-contract-tests.yml
```

---

# 3. Frozen candidate decisions

## Learning truth boundary

```text
LEARNING RECORD != PATIENT RECORD
EVERY LearningFactV1.authoritative_for_patient = false
```

Real de-identified case provenance does not confer patient-record authority.

## Challenge revision / duplicate boundary

```text
challenge_id = stable UUID
(challenge_id, revision) = immutable accepted content revision
exact duplicate = idempotent no-op
same id/revision + changed content = conflict
new accepted content = latest revision + 1
accepted revision never updated in place
```

## External JSON authority

External Challenge JSON cannot self-certify:

```text
clinician review state
review timestamp
Signal linkage
reference verification
```

Preview normalizes those states; final clinician confirmation is server-authoritative.

## Delete/privacy recovery

```text
confirmed delete
→ purge Challenge content/revisions + targeted due items
→ retain non-content tombstone only:
   challenge_id / deleted_at / max_deleted_revision
```

## Foundation state

```text
FORMAL_SOLID
INTUITIVE_UNSTRUCTURED
FRAGMENTED
UNKNOWN_UNTESTED
```

State is not monotonic and not self-rating. L-1 state mutation requires an explicit clinician-reviewed Foundation assessment. Challenge/Daily Case/Practice Review evidence integration is deferred.

## Topics / ontology

```text
foundation_node_ids = controlled structural classification
topics = normalized descriptive tags in L-1
```

No premature second ontology.

## Due state

Due items use repeatable `occurrence` semantics. Completed occurrences remain historical; a later repetition creates the next occurrence. No adaptive spacing algorithm is invented in L-0.

## Daily Case Review

Daily Case Review is defined for L-2 compatibility but not implemented in L-1.

```text
persist review only when eligible
no eligible case → status/due state only
raw Heidi transcript not stored in learning record
self-review before visible AI critique
```

During the scored baseline, a due review may have `delivery_mode = shadow_hidden`.

---

# 4. Privacy boundary

L-1 deterministic PHI validation is path-scoped to clinical/learning text and repeats before persistence.

Bibliographic locators:

```text
PMID
DOI
URL
```

are excluded from patient-number/phone heuristics so valid evidence metadata is not rejected as PHI.

Known limitation is explicit:

```text
DETERMINISTIC PHI GUARD != PERFECT DE-IDENTIFICATION
```

Free-text person-name detection is not guaranteed. Clinician de-identification attestation remains mandatory for Challenge import, and real/mixed source cases require explicit de-identification state.

---

# 5. Automated evidence

Substantive exact contract head:

```text
76d5fba68a3c4f289fe0d8438fbb1be3f5689a09
```

Workflow:

```text
Clinical Learning L0 contract gate
run 34144552849
SUCCESS
```

Passed:

```text
YAML/object-reference integrity                    PASS
Foundation graph integrity                         PASS
Fact Ledger fixtures/invariants                    PASS
revision / duplicate semantics                     PASS
external-import server authority                   PASS
content-purge/tombstone semantics                  PASS
path-scoped privacy contract                       PASS
bibliographic locator exclusion                    PASS
Foundation transition guards                       PASS
L-1 explicit Foundation-assessment source          PASS
due occurrence/repeat semantics                    PASS
Daily Case Review eligible-only persistence        PASS
raw transcript non-persistence                     PASS
baseline shadow semantics                          PASS
L-1 owner/exclusion boundaries                     PASS
design-only scope                                  PASS
diff hygiene                                       PASS
```

---

# 6. Exact design review

`CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` records the active-writer exact-head review.

Review result:

```text
FIELD-LEVEL COMPLETENESS        PASS
OWNER/BOUNDARY REVIEW           PASS
PRIVACY/DATA-SEPARATION REVIEW  PASS WITH EXPLICIT LIMITATION
MACHINE GATE                    PASS
MATERIAL OPEN FINDING           NONE
INDEPENDENT REVIEW              PENDING
```

The review explicitly does **not** claim independence.

---

# 7. Lifecycle

```text
PRODUCT DIRECTION                   APPROVED
L-0 FIELD-LEVEL CONTRACT            CANDIDATE COMPLETE
L-0 MACHINE GATE                    PASS
ACTIVE-WRITER EXACT DESIGN REVIEW   PASS
INDEPENDENT EXACT-HEAD REVIEW       PENDING
L-0 CONTRACT FROZEN / COMPLETE      NO
L-0 MERGED TO MAIN                  NO
L-1 RUNTIME IMPLEMENTATION          NOT AUTHORIZED
```

---

# 8. Exact next action / HOLD

The branch is now review-only.

```text
independent exact-head design review
→ if CLOSURE PASS:
   apply only material review corrections if required
   rerun exact-head contract gate
   mark L-0 CONTRACT FROZEN / COMPLETE
   open/merge bounded design PR through normal discipline
→ HOLD for separate product-owner L-1 implementation authority
```

If independent review finds a material ownership, privacy, revision, due-state, Foundation or baseline-methodology issue, that is a contract-correction/REPLAN trigger; do not code around it.

Forbidden now:

```text
NO L-1 runtime/database implementation
NO learning API runtime routes
NO Learning Hub production UI
NO external learning credential
NO Daily Case Review runtime
NO raw transcript persistence
NO patient-record mutation
NO Signal promotion
NO background cron
NO production config mutation
```
