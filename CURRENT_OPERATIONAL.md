# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-0 — FIELD-LEVEL CONTRACT FREEZE ACTIVE
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified `main` at entry:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`.
> **Active branch:** `design/clinical-learning-l0-contract-freeze-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L0-2026-09-07`.
> **Detailed design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE CANONICAL/DESIGN WRITER:** ChatGPT — bounded L-0 contract freeze only.
> **Learning design authority:** GRANTED / ACTIVE.
> **Learning runtime implementation authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Production config/secret authority:** NONE.
> **RF runtime authority:** NONE — RF is production-smoke-verified and closed for now.

---

# 1. Production truth / closed RF context

RF v2 remains production-smoke-verified through PRs #75, #76 and #77. Manual/external signing is accepted for now; no signature image belongs in the public repository. RF is not an active workstream.

---

# 2. L-0 objective

Freeze an implementable, reusable Core contract for:

```text
ClinicalLearningChallengeV1
LearningFactV1 / mandatory Fact Ledger
LearningObservationV1
LearningReferenceV1
LearningActionV1
FoundationAssessmentAttemptV1
FoundationDomainStateV1
DailyCaseReviewV1
LearningDueStateV1
Osteoporosis Foundation Map v1
L-1 persistence / API / UI ownership
```

L-0 must eliminate semantic invention before any runtime work begins.

---

# 3. Key contract decisions already made on this branch

## 3.1 Learning fact authority

Hard rule:

```text
EVERY LEARNING FACT authoritative_for_patient = false
```

Even `real_deidentified_case_fact` describes provenance/scope only. A learning artifact never acquires patient-record authority and cannot write back to patient truth.

## 3.2 Challenge revision and duplicate semantics

```text
challenge_id = stable UUID across revisions
(challenge_id, revision) = unique immutable content revision
exact same normalized payload = idempotent no-op
same id/revision + different content = conflict
new content revision = latest + 1 only
accepted revision never updated in place
```

Clinician edits after persistence create the next immutable revision.

## 3.3 Delete semantics

L-1 challenge deletion is an explicit confirmed hard delete of challenge revisions plus linked due items. This provides a privacy-safe recovery path for an accidentally sensitive import. Future audit logging may retain non-content metadata only if separately designed.

## 3.4 Reference verification

Imported references default to `unverified`. Record acceptance does not imply bibliographic or evidentiary verification. Verification progresses separately through locator/content checks.

## 3.5 Foundation state

Foundation state is not monotonic and is not a self-rating. State changes require a clinician-reviewed `FoundationAssessmentAttemptV1`. Challenges/case reviews may provide candidate evidence but cannot directly mutate the state.

`FORMAL_SOLID` requires reviewed evidence of formal/mechanistic explanation plus reviewed evidence of transfer, boundary/exception recognition or evidence-directness calibration. Retention is tracked separately.

## 3.6 Daily Case Review evidence boundary

Daily Case Review does not create a second transcript owner. It can later consume either:

```text
newly re-provided Heidi transcript through protected ephemeral PR-1 boundary
OR
approved PR-1/PR-3 structured review evidence
OR
accepted encounter data + clinician context when sufficient
```

`ReviewEvidenceDescriptorV1` stores source metadata only; it never stores raw transcript text. Protected source references are excluded from default exports.

---

# 4. New design artifacts on active branch

```text
schemas/clinical_learning_core_v1.yaml
schemas/osteoporosis_foundation_map_v1.yaml
schemas/clinical_learning_l1_boundary_v1.yaml
```

Planned before L-0 closeout:

```text
schemas/clinical_learning_contract_manifest_v1.yaml
synthetic design fixtures
contract validation tests
L-0 design completeness review
canonical slice/status reconciliation
```

---

# 5. Privacy / baseline invariants

```text
LEARNING RECORD != PATIENT RECORD
raw Heidi transcript ephemeral by default
no patient identifiers in public repo or Challenge import payloads
no patient write from learning objects
missing assessment != failure
no fabricated Daily Case Review when no eligible case exists
visible systematic Daily Case Review coaching = intervention
30-case baseline default = shadow/feedback-hidden review
no composite Clinical Knowledge / Excellence score
```

The deterministic PHI guard is explicitly not claimed to provide perfect free-text de-identification; clinician de-identification attestation remains required for imported de-identified real/mixed cases.

---

# 6. L-1 owner freeze candidate

Future L-1 expected seams, not yet authorized for implementation:

```text
clinical_learning/             reusable Core runtime owner
schemas/clinical_learning_*    machine contracts
static/clinical-learning/      clinician-facing Learning Hub UI
/clinical/learning             existing protected clinical auth boundary
```

L-1 may own Challenge + Foundation persistence and due items. It may not own transcript extraction, Practice Review AI, Signal promotion, patient writes, RF or production configuration.

---

# 7. Exact next action

```text
finish contract manifest + synthetic fixtures
→ validate YAML/reference/revision/Foundation invariants
→ exact design completeness/privacy/owner review
→ reconcile SLICE_PLAN_CURRENT / TODO / changelog
→ if no material unresolved design defect: mark L-0 CONTRACT FROZEN / COMPLETE
→ HOLD for separate L-1 runtime implementation authority
```

Forbidden in current authority:

```text
NO learning database/runtime implementation
NO learning API runtime routes
NO Learning Hub production UI
NO external learning credential
NO raw transcript persistence
NO patient-record mutation
NO background cron
NO production config mutation
```
