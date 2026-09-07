# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-0 — CONTRACT FROZEN / COMPLETE / PR #80 OPEN — RELEASE-DESIGN HOLD
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified `main` / merge base:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`.
> **Active branch:** `design/clinical-learning-l0-contract-freeze-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L0-2026-09-07`.
> **Detailed design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **Corrected substantive contract head:** `afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96`.
> **Substantive machine evidence:** `Clinical Learning L0 contract gate`, run `34147429373` — SUCCESS.
> **Active-writer review:** `CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` — PASS, not independent.
> **Independent review:** `CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md` — CLOSURE PASS / material open finding NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE DESIGN/CANONICAL WRITER:** NONE — L-0 contract freeze is closed; PR #80 is release-design HOLD.
> **Learning runtime implementation authority:** NONE.
> **L-0 merge authority:** NONE — PR #80 is open and must not be merged without a separate explicit product-owner decision.
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

# 5. Independent closure evidence

Corrected substantive contract head:

```text
afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96
```

Machine gate:

```text
Clinical Learning L0 contract gate
run 34147429373
SUCCESS
```

Independent review:

```text
CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md
CLOSURE PASS
MATERIAL OPEN FINDING NONE
```

The independent review did not adopt the active-writer review as its conclusion. It identified material bounded contract defects, corrected them within L-0 design authority, and reran the complete machine gate before issuing CLOSURE PASS.

Key corrected ownership/integrity boundaries include reusable reference-verification overlay ownership, due-item source provenance for Challenge deletion, recursive unknown-field/PHI validation with sanitized rejection, fail-closed internal references, Daily Case immutable revision semantics, shared Signal-engine authority, append-only Foundation assessment evidence, and deterministic deferred due-state reactivation.

---

# 6. Final clean freeze gate — PASS

The clean post-closeout/pre-PR branch head `f947b12ce77db2ad1e5ff9117d7a4f794e224b60` passed the complete `Clinical Learning L0 contract gate`, run `34148333413` — SUCCESS. Temporary closeout helpers were absent from that clean head. This closes the L-0 contract/design freeze evidence gate.

Any future material contract change requires reopening review; status-only PR metadata does not authorize semantic mutation.

# 7. Lifecycle

```text
PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             FROZEN / COMPLETE
SUBSTANTIVE CONTRACT GATE            PASS — run 34147429373
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
CANONICAL CLOSEOUT                   COMPLETE
CLEAN FREEZE GATE                    PASS — f947b12ce77db2ad1e5ff9117d7a4f794e224b60 / run 34148333413
PR #80                               OPEN / MERGEABLE / RELEASE-DESIGN HOLD
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME IMPLEMENTATION           NOT AUTHORIZED
```

---

# 8. Exact next action / HOLD

```text
PR #80 OPEN
→ HOLD for separate product-owner merge decision
→ if merged later: docs/design-only Render auto-deploy may follow main
→ L-1 remains separately unauthorized until explicit product-owner implementation authority
```

Current hold:

```text
NO merge of PR #80
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
