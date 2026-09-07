# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-0 exact contract freeze

> **STATUS:** CONTRACT FROZEN / COMPLETE / INDEPENDENT CLOSURE PASS / PR #80 OPEN — RELEASE-DESIGN HOLD
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L0-2026-09-07`.
> **Fresh `main` / merge base:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`.
> **Design branch:** `design/clinical-learning-l0-contract-freeze-2026-09-07`.
> **Detailed product design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **Field-level contract manifest:** `schemas/clinical_learning_contract_manifest_v1.yaml`.
> **Substantive corrected contract head:** `afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96`.
> **Substantive machine evidence:** `Clinical Learning L0 contract gate`, run `34147429373` — SUCCESS.
> **Active-writer design review:** `CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` — PASS, explicitly NOT independent.
> **Independent review:** `CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md` — CLOSURE PASS / material open finding NONE on the corrected substantive contract head.
> **Runtime implementation authority:** NONE.
> **Design PR:** #80 OPEN; merge authority NONE.
> **Patient-data mutation authority:** NONE.
> **Production config/secret authority:** NONE.

---

# 1. L-0 purpose

Freeze enough field-level semantics, provenance, ownership and safety boundaries that L-1 can later be implemented without inventing architecture during coding.

The Clinical Learning Hub keeps three instruments separate:

```text
FOUNDATION MAP
→ structural/theoretical understanding

CLINICAL CHALLENGE
→ controlled novel-case reasoning

DAILY REAL-CASE REVIEW
→ reasoning/execution in an actual encounter
```

They may later feed the shared Signal engine, but they do not measure the same construct and must not collapse into a composite score.

---

# 2. Frozen Core learning objects

Machine owner:

```text
schemas/clinical_learning_core_v1.yaml
```

Frozen objects:

```text
LearningFactV1
ProgressiveDisclosureV1
ClinicianReasoningResponseV1
LearningReferenceV1
LearningObservationV1
LearningActionV1
ClinicalLearningChallengeV1
FoundationAssessmentEvidenceV1
FoundationAssessmentAttemptV1
FoundationDomainStateV1
DailyCaseSelfReviewV1
ReviewEvidenceDescriptorV1
DailyCaseReviewV1
LearningDueStateV1
```

Hard invariant:

```text
LEARNING RECORD != PATIENT RECORD
```

Every `LearningFactV1` has:

```text
authoritative_for_patient = false
```

This remains true even for `real_deidentified_case_fact`. Learning provenance never acquires patient-record authority.

---

# 3. Challenge / Fact Ledger contract

`ClinicalLearningChallengeV1` is a structured learning artifact, not a chat transcript.

Required semantic domains include:

```text
stable challenge identity
immutable revision
module/date/title/mode
topics + Foundation linkage
initial case
mandatory Fact Ledger
progressive disclosures
clinician reasoning responses
observations
references
learning actions
review/privacy state
```

Fact scopes are explicitly distinct:

```text
real_deidentified_case_fact
synthetic_case_fact
clinician_hypothesis
ai_inference
counterfactual_teaching_point
```

A progressive disclosure may add synthetic teaching facts but those facts can never later be represented as real patient facts merely because they were used in the same challenge.

---

# 4. Challenge identity / revision contract

Stable identity:

```text
challenge_id
```

Immutable version:

```text
(challenge_id, revision)
```

Frozen behavior:

```text
revision 1
→ supersedes_revision = null

revision N > 1
→ supersedes_revision = N - 1

same id + revision + same normalized server hash
→ idempotent no-op

same id + revision + different content
→ conflict

new accepted content
→ latest revision + 1

accepted revision
→ never updated in place
```

The server derives the canonical SHA-256 content hash; imported clients do not own it.

---

# 5. External Challenge import authority

External assistant JSON is candidate content, not authority.

Inbound fields that are explicitly untrusted:

```text
record_review_state
reviewed_at
linked_signal_ids
references[].verification_state
```

Preview normalization:

```text
record_review_state = imported_pending_review
reviewed_at = null
linked_signal_ids = []
reference verification = unverified
```

Final persistence requires explicit clinician confirmation. At final save the server sets the clinician-reviewed state/timestamp. L-1 does not accept authoritative Signal IDs from imported JSON.

Imported observations begin `pending` unless created/dispositioned by the clinician during review. Material observations must be accepted, modified or dismissed before the record is finalized as clinician-reviewed.

---

# 6. Reference verification contract

Challenge acceptance and bibliographic verification are separate operations.

Reference states:

```text
unverified
verified_locator
verified_content
invalid_or_unresolved
```

Imported references default to `unverified` regardless of an external assistant's supplied claim.

```text
verified_locator
→ locator resolves to expected source

verified_content
→ material source content has actually been reviewed for the relevant claim
```

No imported citation becomes evidence-authoritative merely because it appears plausible.

---

# 7. Challenge deletion / privacy-recovery contract

Deletion is explicit and confirmed.

Frozen behavior:

```text
DELETE challenge
→ purge all Challenge revision content/hashes
→ remove due items targeting that Challenge
→ retain only non-content tombstone:
   challenge_id
   deleted_at
   max_deleted_revision
```

The tombstone must not retain:

```text
title
topics
facts
reasoning
observations
references
patient/encounter references
```

A tombstoned `challenge_id` cannot be silently reused in L-1.

---

# 8. Privacy / PHI contract

The L-1 privacy guard runs:

```text
before preview acceptance
AND
again before persistence
```

Global structured direct-identifier field names are rejected.

Deterministic free-text scanning is restricted to clinical/learning content paths such as:

```text
title
initial_case
fact statements
progressive-disclosure narrative
clinician reasoning
final decision
observations
learning-action text
next challenge topic
```

Bibliographic numeric locator fields are excluded from patient-phone/identity-number heuristics:

```text
references[].pmid
references[].doi
references[].url
```

This avoids false positives on legitimate bibliographic identifiers.

Known limitation is explicit:

```text
DETERMINISTIC GUARD != PERFECT DE-IDENTIFICATION ENGINE
```

Free-text person-name detection is not guaranteed. Clinician attestation that direct identifiers have been removed is required for Challenge import. `deidentified_real_case` and `mixed` modes with a real-case component require explicit de-identification confirmation.

---

# 9. Osteoporosis Foundation Map v1

Machine owner:

```text
schemas/osteoporosis_foundation_map_v1.yaml
```

Initial controlled nodes:

```text
ost.foundation.bone_remodeling
ost.foundation.dxa_lsc
ost.foundation.vfa
ost.foundation.fracture_risk_frax
ost.foundation.secondary_osteoporosis
ost.foundation.giop
ost.foundation.antiresorptive_pharmacology
ost.foundation.anabolic_pharmacology
ost.foundation.sequencing
ost.foundation.denosumab_rebound
ost.foundation.btms
ost.foundation.treatment_safety_transitions
ost.foundation.evidence_appraisal
ost.foundation.communication_sdm_continuity
```

The graph expresses learning dependencies/relations, not clinical prerequisites.

`foundation_node_ids` are the controlled structural classification. Challenge `topics` in L-1 are normalized descriptive tags, not a second mandatory ontology.

---

# 10. Foundation assessment / state contract

Allowed states:

```text
FORMAL_SOLID
INTUITIVE_UNSTRUCTURED
FRAGMENTED
UNKNOWN_UNTESTED
```

State is not monotonic and `UNKNOWN_UNTESTED` is not failure.

State changes only through a clinician-reviewed `FoundationAssessmentAttemptV1`.

L-1 authoritative Foundation evidence is **explicit Foundation assessment only**. Challenge, Daily Case Review and Practice Review evidence integration is deferred until those owners exist and cross-artifact privacy/deletion semantics are separately activated.

Self-rating alone cannot change state.

`FORMAL_SOLID` requires reviewed evidence of:

```text
formal/unaided or mechanistic explanation
+
novel transfer OR boundary/exception recognition OR evidence-directness calibration
```

Retention is modeled separately from Foundation state.

---

# 11. Due-state / spaced-repetition contract

`LearningDueStateV1` is scheduling state, not competence evidence.

L-1 materializes due items for:

```text
challenge_repetition
foundation_reassessment
learning_action
progress_review
```

Daily Case Review due-state is L-2 scope.

Due items use repeatable `occurrence` semantics:

```text
occurrence starts at 1
uncompleted occurrence may be rescheduled
completed occurrence remains historical
new schedule after completion creates occurrence + 1
```

No adaptive spacing algorithm is invented in L-0. L-1 may use explicit reviewed due dates; adaptive scheduling requires later design authority.

---

# 12. Daily Case Review target contract

`DailyCaseReviewV1` is defined in L-0 for compatibility with later L-2, but L-1 does not implement its storage/API/UI.

Persisted Daily Case Review requires:

```text
eligibility_state = eligible
```

These are status/due states, not persisted Review records:

```text
no_eligible_case
insufficient_review_evidence
source_or_privacy_blocked
```

Hard rules:

```text
no fabricated case
self-review before visible AI critique
raw Heidi transcript not persisted in learning object
evidence summary normalized/paraphrased, not verbatim transcript
opaque encounter/source references excluded from default export
learning facts remain non-authoritative for patient truth
```

---

# 13. Transcript / Practice Review / Signal ownership

Clinical Learning must not create parallel semantic owners.

```text
Transcript extraction / candidate provenance
→ PR-1 / PR-2 owner

Practice Review interpretation / observations
→ PR-3 owner

Signal promotion / recurrence reliability
→ shared Signal engine

Patient clinical truth
→ clinical encounter/longitudinal owner
```

Future Daily Case Review may consume a newly re-provided Heidi transcript through the protected ephemeral transcript boundary or approved structured review evidence. L-1 does neither.

---

# 14. L-1 persistence boundary

Candidate protected tables:

```text
clinical_learning_challenge_revisions
clinical_learning_challenge_tombstones
clinical_learning_foundation_attempts
clinical_learning_foundation_state
clinical_learning_due_items
```

They are distinct from patient encounter, laboratory and RF persistence.

Not created in L-1:

```text
Daily Case Review tables
review-evidence tables
Signal candidate/promotion tables
transcript storage
```

---

# 15. L-1 API / UI boundary

Future protected base:

```text
/clinical/learning
```

L-1 API scope is bounded to:

```text
Challenge preview/import/history/revision/delete
Foundation registry/state + explicit assessment preview/persist
learning due-state retrieval
```

Explicitly forbidden in L-1:

```text
external bearer ingestion credential
patient write endpoint
transcript storage endpoint
Signal promotion endpoint
Daily Case Review endpoint
```

Future UI owner:

```text
static/clinical-learning/
```

First views:

```text
Challenge Import
Challenge History
Foundation Map
Due / Learning Actions
```

No composite Clinical Knowledge/Excellence score.

---

# 16. Baseline methodology invariant

Visible systematic Daily Case Review coaching is an intervention.

During the 30-case scored system-assisted baseline, default behavior remains:

```text
review may run in shadow
routine AI critique/coaching hidden
safety-critical feedback allowed
```

A real due state can remain `due` while delivery mode is `shadow_hidden`.

If visible daily coaching is desired during the scored baseline, methodology must be explicitly replanned and the cohort relabelled before continuing.

---

# 17. Machine contract set

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

# 18. Acceptance evidence / independent closure review

Corrected substantive contract head:

```text
afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96
```

Workflow:

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

The original pre-independent-review candidate did not receive an automatic pass. Independent review found material but bounded L-0 contract defects; they were corrected within the existing design authority and the full gate was rerun successfully before the CLOSURE PASS.

The corrected contract/gate now covers:

```text
YAML/object/reference integrity                         PASS
Foundation graph integrity                              PASS
Fact Ledger / supersession / disclosure references      PASS
immutable revision/idempotency/conflict semantics       PASS
external-import server authority                        PASS
reusable reference-verification overlay                 PASS
content purge + non-content tombstone                    PASS
due-item source provenance / deletion integrity          PASS
unknown-field rejection + PHI/privacy boundary           PASS
sanitized errors/logging                                 PASS
bibliographic PMID/DOI/URL numeric exclusions            PASS
Foundation transition/evidence authority                 PASS
Daily Case Review eligible-only + revision integrity     PASS
Signal authority separation                              PASS
deferred/completed due occurrence semantics              PASS
baseline shadow methodology                              PASS
L-1 owner/exclusion boundary                             PASS
design-only scope + diff hygiene                         PASS
```

---

# 19. Findings corrected across L-0 reviews

Active-writer review corrected bounded issues including ineligible Daily Case Review persistence, bibliographic-number privacy false positives, Challenge tombstone semantics, external-import authority, topic/ontology ownership, due occurrence semantics and premature Challenge-to-Foundation coupling.

Independent review then found and closed additional material contract defects:

```text
reference verification vs immutable revision authority
→ reusable server/clinician-owned verification overlay

Challenge deletion vs nested learning-action due rows
→ source-artifact provenance + source/target cleanup

incomplete PHI scan surface / unknown imported keys
→ recursive unknown-field rejection + all persistable untrusted strings scanned

privacy rejection could echo rejected content
→ sanitized field-path/code errors and non-content routine logs

internal cross-object reference corruption paths
→ fail-closed uniqueness/resolution/integrity constraints

DailyCaseReview revision predecessor ambiguity
→ explicit immutable predecessor/revision semantics

embedded linked_signal_ids could duplicate Signal authority
→ future shared Signal engine remains dynamic authority

source_artifact_deleted conflicted with append-only Foundation attempts
→ deletion state resolved externally without rewriting reviewed attempts

deferred due state lacked deterministic reactivation
→ future defer remains deferred; today due; past overdue; completed occurrence terminal
```

Independent disposition: **CLOSURE PASS / material open finding NONE**.

# 20. Out of scope / deferred

```text
L-1 ORM/runtime/API/UI implementation
exact runtime PII-regex implementation
adaptive spaced-repetition algorithm
external scoped learning credential
Challenge-derived Foundation evidence integration
Signal recurrence/promotion engine
Daily Case Review runtime
Practice Review AI implementation
provider/privacy gate for identifiable Heidi transcript use
```

---

# 21. Lifecycle / exact next gate

```text
PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             FROZEN / COMPLETE
SUBSTANTIVE MACHINE GATE             PASS — run 34147429373
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
CLEAN FREEZE GATE                    PASS — f947b12ce77db2ad1e5ff9117d7a4f794e224b60 / run 34148333413
DESIGN PR #80                        OPEN / RELEASE-DESIGN HOLD
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME AUTHORITY                NONE
```

Exact next action:

```text
PR #80 OPEN
→ RELEASE / DESIGN HOLD
→ separate product-owner merge decision required
→ after any later merge, separate product-owner L-1 runtime implementation decision required
```

Any material contract change after the independently reviewed substantive head invalidates the CLOSURE PASS and requires another substantive review. Status/documentation-only closeout changes do not authorize L-1 implementation.
