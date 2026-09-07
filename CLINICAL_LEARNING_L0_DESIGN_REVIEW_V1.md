# Clinical Learning Hub L-0 — Exact-Head Design Completeness Review v1

> **REVIEW TYPE:** exact-head design completeness / owner / privacy review by the active ChatGPT design writer
> **INDEPENDENCE:** NOT INDEPENDENT — a separate independent review remains a distinct gate
> **SLICE:** `CORE-LEARNING-HUB-L0-2026-09-07`
> **BASE:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`
> **EXACT REVIEWED HEAD:** `76d5fba68a3c4f289fe0d8438fbb1be3f5689a09`
> **AUTOMATED GATE:** `Clinical Learning L0 contract gate`, run `34144552849` — SUCCESS

---

# 1. Review question

Does the L-0 candidate define a sufficiently bounded, internally coherent and privacy-aware contract to make a later L-1 Challenge + Foundation MVP implementable **without inventing semantic ownership during coding**?

Review result:

```text
MATERIAL DESIGN DEFECT FOUND AFTER FINAL CORRECTIONS: NO
L-0 MACHINE CONTRACT GATE: PASS
ACTIVE-WRITER EXACT DESIGN REVIEW: PASS
INDEPENDENT REVIEW: PENDING
L-1 RUNTIME AUTHORITY: NONE
```

---

# 2. Reviewed contract set

```text
CLINICAL_LEARNING_HUB_DESIGN_V1.md
schemas/clinical_learning_core_v1.yaml
schemas/osteoporosis_foundation_map_v1.yaml
schemas/clinical_learning_l1_boundary_v1.yaml
schemas/clinical_learning_contract_manifest_v1.yaml
schemas/clinical_learning_design_fixtures_v1.yaml
test_clinical_learning_l0_contract.py
test_clinical_learning_l0_boundary.py
.github/workflows/clinical-learning-l0-contract-tests.yml
```

The branch also updates `CURRENT_OPERATIONAL.md` only to claim the bounded design writer and record the current L-0 decisions.

---

# 3. Object completeness

## PASS — Challenge artifact

`ClinicalLearningChallengeV1` now has stable identity and immutable revisions, explicit challenge mode, Foundation linkage, initial case, mandatory Fact Ledger, progressive disclosures, clinician reasoning responses, observations, references, actions, review state and privacy envelope.

Key distinction:

```text
challenge_id = stable learning identity
revision = immutable accepted content version
```

External import cannot self-certify clinician review, Signal linkage or reference verification. Those are server/clinician-controlled states in L-1.

## PASS — Fact Ledger

`LearningFactV1` preserves:

```text
real_deidentified_case_fact
synthetic_case_fact
clinician_hypothesis
ai_inference
counterfactual_teaching_point
```

Every learning fact has:

```text
authoritative_for_patient = false
```

This remains true even for a fact derived from a real de-identified case. Learning provenance never becomes patient-record authority.

## PASS — Learning observations / references / actions

Observations carry category, importance, fact/reference linkage, gap class and clinician disposition. Pending/dismissed observations have no Signal authority.

Imported references default to `unverified`; accepting a Challenge does not imply bibliographic or evidence verification.

Learning actions remain explicit objects rather than being inferred from a generic score.

---

# 4. Revision / duplicate / delete semantics

## PASS — Duplicate and revision contract

Frozen behavior:

```text
same challenge_id + revision + same normalized hash
→ idempotent no-op

same challenge_id + revision + different content
→ conflict

new accepted content
→ latest revision + 1

accepted revision
→ never updated in place
```

## PASS — Delete / privacy recovery

Challenge deletion is not a misleading row-level soft delete that retains sensitive content.

Frozen behavior:

```text
explicit confirmed DELETE
→ purge all Challenge content revisions
→ purge linked Challenge due items
→ retain only non-content tombstone:
   challenge_id
   deleted_at
   max_deleted_revision
```

The tombstone prevents silent identity reuse without retaining title, facts, reasoning, observations, references or patient/encounter references.

L-1 Foundation attempts use explicit Foundation assessment evidence only, so Challenge content deletion cannot leave copied Challenge content inside L-1 Foundation evidence. Cross-artifact evidence linkage remains a later separately reviewed concern.

---

# 5. Privacy review

## PASS with explicit known limitation

The L-1 privacy boundary has two layers:

1. hard rejection of direct-identifier structured fields;
2. deterministic scanning of clinical free-text paths.

The numeric identity/phone heuristic is deliberately **not** applied to bibliographic locator fields:

```text
references[].pmid
references[].doi
references[].url
```

This avoids treating valid PMID/DOI/URL values as patient identifiers.

The contract explicitly does **not** claim perfect free-text de-identification or reliable person-name detection. Clinician attestation that direct identifiers have been removed is required before Challenge persistence, and the server must re-run validation at persistence rather than trusting preview/browser state.

Residual risk is accepted at design level because L-1 is a protected personal learning store, not a public export, and because the system explicitly forbids claiming the deterministic guard is a complete de-identification engine. A future stronger NLP/PII detector would be a separate security/privacy enhancement, not an implicit L-1 dependency.

---

# 6. Foundation Map / mastery review

## PASS

The Module-01 Foundation Map contains 14 initial nodes and explicit learning dependency/related-node edges.

The controlled structural classification is `foundation_node_ids`. Challenge `topics` remain normalized descriptive tags in L-1; this avoids prematurely creating a second competing ontology.

Foundation state:

```text
FORMAL_SOLID
INTUITIVE_UNSTRUCTURED
FRAGMENTED
UNKNOWN_UNTESTED
```

State is not monotonic and cannot be changed by self-rating alone, one AI judgment, a Challenge result or a Daily Case Review directly.

State mutation requires `FoundationAssessmentAttemptV1` with clinician-reviewed evidence.

`FORMAL_SOLID` requires evidence of:

```text
formal/mechanistic explanation
+
novel transfer OR boundary/exception recognition OR evidence-directness calibration
```

Retention is separately represented; no numerical mastery score or adaptive scheduler is invented in L-0.

---

# 7. Due-state / spaced-repetition review

## PASS

Due state is a materialized schedule, not evidence of competence.

L-1 due items use repeatable `occurrence` semantics:

```text
completed occurrence
→ remains historical

new repetition after completion
→ occurrence + 1

uncompleted occurrence
→ may be rescheduled from a later reviewed source
```

No fixed adaptive spacing algorithm is frozen. L-1 may materialize explicit due dates from reviewed Challenge/assessment/action data; an adaptive scheduler remains later work.

Daily Case Review is excluded from L-1 due persistence and belongs to L-2.

---

# 8. Daily Case Review / transcript ownership review

## PASS

`DailyCaseReviewV1` is defined so L-2 has a compatible target contract, but L-1 does not create its tables, routes or UI.

Hard rules:

```text
Daily Case Review persisted only when eligibility_state = eligible
no eligible case → no fabricated review record
raw Heidi transcript → not persisted in learning object
evidence_summary → normalized/paraphrased, not verbatim raw transcript
self-review → before visible AI critique
```

Learning does not create a parallel transcript owner. Future Daily Case Review consumes either the PR-1 protected ephemeral transcript boundary or approved structured review evidence from the transcript/Practice Review owners.

During the 30-case scored baseline, the due state may remain real while `delivery_mode = shadow_hidden`; hiding coaching does not falsify whether review would otherwise be due.

---

# 9. L-1 owner / persistence / API review

## PASS

Future L-1 owner is bounded to a reusable `clinical_learning/` package plus protected learning tables and `static/clinical-learning/` UI.

L-1 owns:

```text
Challenge validation/import/revisions/deletion
Foundation explicit assessments/state
learning due materialization
protected learning UI/API
```

L-1 explicitly does NOT own:

```text
patient writes
transcript extraction/storage
Practice Review AI owner
Signal promotion
Daily Case Review runtime
external learning bearer credential
RF / Clinic Utilities
production config
```

Proposed API remains under the existing protected clinical auth boundary. Direct external ingestion is deliberately deferred.

---

# 10. Automated evidence

Exact head:

```text
76d5fba68a3c4f289fe0d8438fbb1be3f5689a09
```

Workflow:

```text
Clinical Learning L0 contract gate
run 34144552849
SUCCESS
```

The gate proves:

```text
YAML parse / contract identity                         PASS
object-reference integrity                            PASS
Foundation graph reference integrity                  PASS
valid/invalid Fact Ledger fixtures                     PASS
real-case de-identification attestation                PASS
revision/idempotency/conflict obligations              PASS
content-purge/tombstone obligation                     PASS
path-scoped PHI guard contract                         PASS
bibliographic numeric-locator exclusion                PASS
Foundation self-rating prohibition                     PASS
FORMAL_SOLID evidence-category requirement             PASS
Daily Case Review eligible-only persistence            PASS
raw-transcript non-persistence obligation              PASS
baseline shadow delivery semantics                     PASS
external import server-authority boundary              PASS
topic tags vs Foundation controlled structure          PASS
due occurrence/repeat semantics                        PASS
L-1 explicit Foundation assessment source only         PASS
L-1 transcript/Signal/Daily Case exclusions            PASS
design-only branch scope                               PASS
diff hygiene                                           PASS
```

---

# 11. Findings corrected during review

The exact-head review found and corrected before this review artifact:

1. **Ineligible Daily Case Review ambiguity** — corrected so `no_eligible_case`, insufficient evidence and privacy/source block are due/status states, not persisted Review records.
2. **PHI heuristic overreach** — corrected to path-scoped clinical text scanning so numeric PMID/DOI/URL metadata is not mistaken for patient ID/phone data.
3. **Challenge deletion referential/privacy ambiguity** — corrected to content purge + non-content tombstone.
4. **External import authority** — corrected so an external assistant cannot self-certify clinician review, Signal IDs or reference verification.
5. **Premature second topic ontology** — corrected; L-1 topics are normalized descriptive tags while Foundation nodes remain the controlled structure.
6. **Due-item repeat ambiguity** — corrected with explicit occurrence semantics.
7. **Premature Challenge→Foundation coupling** — excluded from L-1; Foundation state is changed only through explicit clinician-reviewed Foundation assessment.

No additional material finding remains in this active-writer review.

---

# 12. Residual / deferred items

These are deliberate later-stage items, not L-0 defects:

```text
runtime ORM/API implementation
exact deterministic PII regex/normalizer implementation
adaptive spaced-repetition algorithm
external scoped learning credential
Challenge-derived Foundation evidence integration
Signal promotion/reliability engine
Daily Case Review runtime
Practice Review AI implementation
provider/privacy gate for identifiable real Heidi transcript processing
```

---

# 13. Review disposition

```text
FIELD-LEVEL CONTRACT COMPLETENESS     PASS
OWNER/BOUNDARY REVIEW                 PASS
PRIVACY/DATA-SEPARATION REVIEW        PASS WITH EXPLICIT LIMITATION
MACHINE CONTRACT GATE                 PASS
ACTIVE-WRITER EXACT-HEAD REVIEW       PASS
INDEPENDENT REVIEW                    PENDING
```

Therefore this review does **not** independently close L-0. The correct next state is:

```text
L-0 CONTRACT CANDIDATE COMPLETE
→ independent exact-head design review
→ if CLOSURE PASS: mark L-0 CONTRACT FROZEN / COMPLETE
→ HOLD for separate L-1 runtime implementation authority
```
