# Clinical Learning Hub L-0 — Independent Exact-Head Design Review v1

> **REVIEW TYPE:** independent design / ownership / privacy / revision / referential-integrity review
> **SLICE:** `CORE-LEARNING-HUB-L0-2026-09-07`
> **CANONICAL BASE:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`
> **ORIGINAL REVIEW INPUT HEAD:** `986067d88700686a1eb049608d5c6f0bf05fa719`
> **SUBSTANTIVE CORRECTED CONTRACT HEAD REVIEWED FOR CLOSURE:** `afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96`
> **MACHINE GATE ON CORRECTED CONTRACT HEAD:** `Clinical Learning L0 contract gate`, run `34147429373` — SUCCESS
> **INDEPENDENCE:** this review did not adopt `CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` as its conclusion; that file was treated only as an artifact under review.
> **RUNTIME AUTHORITY:** NONE

---

# 1. Review question

Does the L-0 Clinical Learning contract define enough field-level semantics, provenance, privacy behavior, revision behavior, deletion behavior, Foundation transition logic, due-state behavior and owner seams that a later L-1 Challenge + Foundation MVP can be implemented without inventing material semantic architecture during coding?

Final disposition after independent findings and bounded L-0 corrections:

```text
INDEPENDENT REVIEW                    CLOSURE PASS
MATERIAL OPEN FINDING                 NONE
CORRECTED SUBSTANTIVE CONTRACT HEAD   afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96
MACHINE GATE                          PASS — run 34147429373
L-0 CONTRACT                          ELIGIBLE TO FREEZE / COMPLETE
L-1 IMPLEMENTATION                    NOT AUTHORIZED
```

The original head `986067d8...` did **not** receive a closure pass. Material findings were found and corrected inside the granted L-0 design authority before this disposition.

---

# 2. Independent review scope

The review independently re-examined:

```text
object ownership
field-level completeness
Fact Ledger provenance
patient/learning truth separation
PHI/privacy semantics
unknown-field behavior
revision/duplicate semantics
internal reference integrity
reference verification authority
delete/tombstone/referential behavior
Foundation state transitions and evidence ownership
due/repetition/defer semantics
Daily Case Review eligibility and revision semantics
transcript / Practice Review owner boundaries
Signal owner boundary
baseline methodology
L-1 persistence/API/UI seams
contradiction/drift against canonicals and PR-1 architecture
```

Canonical truth remained:

```text
LEARNING RECORD != PATIENT RECORD
```

No learning fact acquires patient-record authority. Raw Heidi transcript remains ephemeral by default. Daily Case Review does not create a second transcript owner. Practice Review remains PR-3-owned. Dynamic Signal authority remains owned by the future shared Signal engine.

---

# 3. Material findings on the original candidate and bounded corrections

## Finding 1 — reference verification conflicted with immutable revisions

Original problem:

```text
references[].verification_state
was stored inside immutable Challenge revision content
+
reference verification was declared independently advanceable after Challenge acceptance
```

That produced an impossible ownership choice: mutate an accepted immutable revision or create a semantically artificial content revision for bibliography metadata.

Correction:

```text
immutable LearningReferenceV1 state = persistence-time snapshot
current verification = server/clinician-owned external overlay
```

The overlay is reusable across immutable learning artifact types:

```text
(artifact_type, artifact_id, artifact_revision, reference_id)
```

L-1 activates `artifact_type = challenge`; later Daily Case Review must reuse this owner rather than create a second verification store.

Result: **CLOSED**.

---

## Finding 2 — Challenge deletion could orphan learning-action due items

Original problem:

```text
challenge_repetition due item target = challenge_id
learning_action due item target = nested action_id
```

Deleting due items only by `challenge_id` could therefore purge Challenge content while leaving an action schedule whose owning action no longer existed.

Correction:

Every L-1 materialized due item now carries source provenance:

```text
source_artifact_type
source_artifact_id
source_revision
```

Challenge deletion removes due rows whose **target or source artifact** is the deleted Challenge before content purge/tombstone completion.

Result: **CLOSED**.

---

## Finding 3 — path whitelist could leave persistable PHI-bearing text unscanned

Original problem:

The deterministic Challenge PHI contract listed selected narrative paths but omitted other persistable untrusted text, including fields such as fact source, disclosure label and bibliographic descriptive text. The contract also did not fail closed on unknown imported keys.

Correction:

```text
unknown imported fields -> reject recursively
all persistable untrusted strings -> deterministic direct-identifier scan
except exact bibliographic locator exclusions from numeric phone/identity heuristics:
  references[].pmid
  references[].doi
  references[].url
```

Reference titles/framework labels/verification notes remain scanned. Foundation assessment notes and reference-verification notes are also within the protected learning-text privacy boundary.

The known limitation remains explicit: deterministic scanning is not claimed to be perfect free-text de-identification or reliable person-name recognition. Clinician de-identification attestation remains required for Challenge import.

Result: **CLOSED**.

---

## Finding 4 — privacy rejection itself could leak rejected content

Original problem:

The contract rejected PHI-like content but did not explicitly forbid error responses or routine logs from echoing the rejected value/payload.

Correction:

```text
validation/privacy errors -> bounded codes + field paths only
NO rejected values
NO imported payload fragments
NO candidate clinical values
NO raw exception bodies containing learning content

routine logs -> NO imported Challenge/Foundation payloads or rejected PHI-like values
```

This aligns the Learning boundary with the sanitized PHI-bearing request principle already used by the PR-1 transcript design.

Result: **CLOSED**.

---

## Finding 5 — internal learning references allowed structurally corrupt artifacts

Original problem included missing explicit closure rules for:

```text
LearningFact.supersedes_fact_id
ProgressiveDisclosure.released_after_response_id
Fact/reference/action ID uniqueness
Daily Case Review fact/reference links
Foundation node references
```

Correction freezes fail-closed internal integrity, including:

```text
fact IDs unique
supersedes_fact_id resolves in same Fact Ledger and is not self
response/disclosure/observation/reference/action IDs unique in parent artifact
progressive disclosure sequence unique + contiguous from 1
released_after_response_id resolves in same Challenge
every observation/action fact/reference link resolves in same artifact
Foundation node IDs resolve through module registry
Foundation materialized evidence attempts resolve to same module/node
```

Result: **CLOSED**.

---

## Finding 6 — Daily Case Review revision semantics were under-specified

Original problem:

`DailyCaseReviewV1` exposed `revision` and `supersedes_revision` but did not freeze valid predecessor states.

Correction:

```text
revision 1 -> supersedes_revision = null
revision N > 1 -> supersedes exactly N-1
accepted Daily Case Review revision -> immutable when L-2 is separately authorized
```

L-1 still does not implement Daily Case Review persistence/API/UI.

Result: **CLOSED**.

---

## Finding 7 — dynamic Signal links risked duplicate semantic authority

Original problem:

Learning artifacts contain `linked_signal_ids`, while Signal promotion/reliability is a separate future shared owner. Treating the embedded field as current authority would either duplicate Signal truth or require in-place mutation of immutable learning revisions.

Correction:

```text
future shared Signal engine = authoritative dynamic relationship owner
L-1 imported linked_signal_ids = normalized empty
embedded linked_signal_ids = non-authoritative snapshot/projection only
later Signal promotion/backlink must not mutate accepted learning revision in place
```

Result: **CLOSED**.

---

## Finding 8 — Foundation source deletion marker conflicted with append-only reviewed attempts

Original problem:

`FoundationAssessmentEvidenceV1.source_artifact_deleted` implied that a reviewed append-only assessment might later be mutated when the source artifact was deleted, while the deletion contract explicitly forbids rewriting historical reviewed assessment content.

Correction:

The mutable deletion-status field was removed from reviewed evidence content. Current source deletion status is resolved from the external source owner/tombstone when that future cross-artifact integration is authorized.

```text
source deletion != mutation of reviewed Foundation attempt
```

L-1 remains simpler: only explicit Foundation-assessment evidence is authoritative for Foundation state.

Result: **CLOSED**.

---

## Finding 9 — deferred due-state expiry was ambiguous

Original problem:

`deferred` and `deferred_until` existed without deterministic reactivation semantics.

Correction:

```text
completed_at -> terminal completed occurrence

deferred -> requires future deferred_until + null completed_at

deferred_until = today -> due
deferred_until < today -> overdue
```

Completed occurrences remain historical; a new repetition after completion creates `occurrence + 1`.

Result: **CLOSED**.

---

# 4. Owner / architecture review after corrections

Final owner map is coherent:

```text
Clinical Learning Core
  -> learning schema/import/revisions/privacy/persistence/due/Foundation validation/reference-verification overlay

Module 01 — Osteoporosis
  -> Foundation registry/content/module-specific assessment content

PR-1 / PR-2
  -> transcript extraction / candidate provenance / later clinician-approved patient writes

PR-3
  -> Practice Review interpretation / observations

Shared Signal engine
  -> authoritative dynamic Signal promotion/recurrence/backlinks

Clinical encounter / longitudinal owner
  -> patient clinical truth
```

No second transcript owner, second reference-verification owner, second Signal authority or learning-to-patient mutation path remains in L-0.

---

# 5. Foundation / mastery review

Foundation state remains:

```text
FORMAL_SOLID
INTUITIVE_UNSTRUCTURED
FRAGMENTED
UNKNOWN_UNTESTED
```

Review confirms:

```text
state is not monotonic
UNKNOWN_UNTESTED is not failure
self-rating alone cannot mutate state
one AI judgment cannot mutate state
Challenge / Daily Case / Practice Review cannot directly mutate state
L-1 authoritative state change requires clinician-reviewed FoundationAssessmentAttemptV1
FORMAL_SOLID requires formal/mechanistic evidence + transfer/boundary/evidence-directness evidence
retention is separate from conceptual state
```

No misleading mastery score or automatic competence inference is introduced.

---

# 6. Daily Case Review / PR-1 / Practice Review / baseline review

No contradiction remains with the canonical PR architecture:

```text
no eligible case -> status/due outcome, not fake DailyCaseReviewV1
persisted DailyCaseReviewV1 -> eligibility_state = eligible
self-review -> before visible AI critique
raw Heidi transcript -> not persisted by learning
review evidence summary -> normalized/paraphrased, not verbatim transcript
protected encounter/source refs -> excluded from default export
```

During the 30-case scored baseline:

```text
review may run in shadow
routine coaching remains hidden
safety-critical feedback may surface
```

Visible routine coaching still requires explicit methodology REPLAN and cohort relabelling.

---

# 7. L-1 implementation seam review

L-1 remains bounded to:

```text
Challenge preview/import/history/revision/delete
reusable reference-verification overlay for Challenge references
Foundation registry/state + explicit assessment preview/persist
learning due-state retrieval/materialization
protected Learning UI/API
```

L-1 remains forbidden from implementing:

```text
patient write endpoint
raw transcript persistence
transcript extraction owner
Practice Review AI owner
Signal promotion endpoint
Daily Case Review runtime/API/UI
external bearer learning credential
RF runtime
background cron
composite Clinical Excellence score
```

No runtime/database/API/UI code was added by L-0; only machine-readable contracts, tests, workflow evidence and design/canonical documentation are in scope.

---

# 8. Machine evidence

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

The gate covers YAML/object integrity, Foundation graph integrity, revision behavior, external-import authority, privacy boundary, bibliographic-locator exclusions, reusable reference-verification ownership, deletion/tombstone/due referential semantics, Foundation transition boundaries, Daily Case source/reference/revision constraints, baseline semantics, L-1 owner exclusions, design-only scope and diff hygiene.

A final exact-head gate must still be rerun after this review artifact and closeout status documents are committed. Those closeout changes are evidence/status-only; any material contract change after `afaf9d5d...` would invalidate this CLOSURE PASS and require another substantive review.

---

# 9. Final independent disposition

```text
ORIGINAL HEAD 986067d8...             MATERIAL FINDINGS FOUND — NOT PASS
BOUNDED L-0 CORRECTIONS               APPLIED
CORRECTED CONTRACT HEAD afaf9d5d...   INDEPENDENT CLOSURE PASS
MATERIAL OPEN FINDING                 NONE
REPLAN TRIGGER REMAINING              NONE
L-0                                   ELIGIBLE FOR CONTRACT FROZEN / COMPLETE
L-1                                   NOT AUTHORIZED
```

Correct next action:

```text
update canonical closeout status
→ rerun gate on final branch exact head
→ open/update bounded design PR to main
→ RELEASE / DESIGN HOLD unless explicit merge authority exists
→ HOLD for separate product-owner L-1 implementation decision
```
