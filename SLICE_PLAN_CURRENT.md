# SLICE_PLAN_CURRENT.md — Clinical Learning Hub L-1 Challenge + Foundation MVP

> **STATUS:** PRODUCT-OWNER AUTHORIZED / IMPLEMENTATION ACTIVE
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Slice ID:** `CORE-LEARNING-HUB-L1-CHALLENGE-FOUNDATION-MVP-2026-09-07`.
> **Fresh implementation base:** `5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6`.
> **Implementation branch:** `feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07`.
> **Frozen L-0 contract owner:** `schemas/clinical_learning_contract_manifest_v1.yaml`.
> **L-1 boundary owner:** `schemas/clinical_learning_l1_boundary_v1.yaml`.
> **Foundation registry:** `schemas/osteoporosis_foundation_map_v1.yaml`.
> **Product design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **Merge/deploy authority:** NONE in this slice activation.
> **Patient-data mutation authority:** NONE.
> **Production config/secret authority:** NONE.

---

# 1. Objective

Implement the first protected Clinical Learning runtime without reopening L-0 semantics:

```text
Challenge JSON import / preview / review / save
+ immutable Challenge history / revision / delete
+ reusable reference-verification overlay
+ Osteoporosis Foundation Map + explicit assessments
+ deterministic learning due state
+ clinician-facing MVP UI
```

This slice is learning infrastructure. It is not patient documentation and it does not implement Daily Case Review, transcript intake, Practice Review or Signal promotion.

---

# 2. Non-negotiable inherited invariants

```text
LEARNING RECORD != PATIENT RECORD
EVERY LearningFactV1.authoritative_for_patient = false
RAW TRANSCRIPT != L-1 STORAGE
IMPORTED JSON != CLINICIAN-REVIEW AUTHORITY
REFERENCE VERIFICATION != IMMUTABLE CHALLENGE PAYLOAD
FOUNDATION STATE != SELF-RATING
DUE STATE != COMPETENCE EVIDENCE
```

Additional boundaries:

- no direct patient identifier may be persisted in L-1 learning content;
- every external/imported unknown field is rejected recursively;
- privacy/schema failures expose only bounded codes/paths, never rejected values/payload fragments;
- accepted Challenge revisions are immutable;
- Challenge deletion purges content and leaves only the frozen non-content tombstone;
- Foundation state changes only from a valid explicit clinician-reviewed `FoundationAssessmentAttemptV1`;
- no composite Clinical Knowledge / Clinical Excellence score;
- no autonomous Signal authority.

---

# 3. Frozen implementation owners

Allowed runtime owners:

```text
clinical_learning/
  __init__.py
  models.py
  contracts.py
  privacy.py
  persistence.py
  service.py
  api.py

static/clinical-learning/
  index.html
  styles.css
  app.js

main.py
L-1-focused tests/workflow
six canonicals when lifecycle state changes
```

Normative frozen contracts are read-only during implementation:

```text
schemas/clinical_learning_core_v1.yaml
schemas/clinical_learning_l1_boundary_v1.yaml
schemas/clinical_learning_contract_manifest_v1.yaml
schemas/clinical_learning_design_fixtures_v1.yaml
schemas/osteoporosis_foundation_map_v1.yaml
```

If runtime implementation requires a material change to those contracts, STOP and REPLAN; do not patch around them.

---

# 4. Explicit overlap exclusion — Physiotherapy / Clinic Utilities

A separate conversation has a paused physiotherapy productization branch. L-1 must not modify or merge any physiotherapy/CU-1 owner, including:

```text
clinic_utilities/physio_referral_*
clinic_utilities/contracts/cu1_*
static/clinic-utilities/physio-referral/*
physiotherapy design/productization branches
```

L-1 also must not mutate RF owners.

To minimize future branch conflict, this slice does not add Learning navigation into the existing physiotherapy/Clinic Utilities surface or baseline-audit navigation. The protected Learning Hub is directly reachable at `/clinical/learning`; navigation integration can be reconciled later after the paused physiotherapy workstream is resumed/merged.

---

# 5. Protected persistence

Use the existing protected SQLAlchemy engine with a dedicated `ClinicalLearningBase` and the frozen tables:

```text
clinical_learning_challenge_revisions
clinical_learning_reference_verification
clinical_learning_challenge_tombstones
clinical_learning_foundation_attempts
clinical_learning_foundation_state
clinical_learning_due_items
```

No patient encounter/lab/RF table is read or written by this L-1 package.

No L-1 tables for:

```text
Daily Case Review
review evidence
Signal candidates/promotions
raw transcripts
```

---

# 6. Authentication / route ownership

Protected base:

```text
/clinical/learning
```

Authentication reuses the existing Clinical Excellence browser session and `X-Clinical-Key` contract. L-1 owns a local learning auth dependency; it must not import auth logic from the physiotherapy utility.

`ClinicalCookieMiddleware` continues to translate a valid `clinical_session` cookie into the protected request header.

`main.py` changes are bounded to:

```text
mount the Clinical Learning router
+
apply no-store/no-cache headers to /static/clinical-learning/*
```

No other application composition change is authorized.

---

# 7. Challenge import / preview runtime profile

Frozen endpoint:

```text
POST /clinical/learning/api/challenges/preview
```

MVP request envelope:

```text
{
  "challenge": <ClinicalLearningChallengeV1>
}
```

Preview is no-write and must:

1. reject malformed JSON/schema and recursive unknown fields;
2. run the PHI guard;
3. validate all internal Fact/Disclosure/Observation/Reference/Action/Foundation references;
4. normalize server-authoritative imported fields:
   - `record_review_state = imported_pending_review`
   - `reviewed_at = null`
   - `linked_signal_ids = []`
   - imported reference verification = `unverified` with null note;
   - imported observations default to `pending`;
5. normalize topics by trim + casefold filter key while preserving the first display label;
6. calculate duplicate/revision state against persistence;
7. return only sanitized errors/warnings.

`normalized_summary` may contain the validated normalized structured Challenge needed by the clinician-facing preview. It is learning content only after the PHI guard passes; rejected payload fragments are never echoed.

Duplicate states used by the implementation may include:

```text
new_identity
exact_idempotent_duplicate
same_revision_conflict
next_revision_candidate
stale_revision_rejected
tombstoned_identity_rejected
```

They are implementations of the frozen revision rules, not new lifecycle semantics.

---

# 8. Challenge save / edit profile

Create:

```text
POST /clinical/learning/api/challenges
{
  "challenge": <reviewed candidate>,
  "confirm_save": true
}
```

Edit:

```text
PUT /clinical/learning/api/challenges/{challenge_id}
{
  "challenge": <edited candidate>,
  "confirm_save": true
}
```

Hard rules:

- server revalidates schema, privacy and internal references immediately before persistence;
- `confirm_save=true` is mandatory;
- server owns `record_review_state=clinician_reviewed` and `reviewed_at`;
- server owns canonical SHA-256 content hash;
- server owns next revision / `supersedes_revision` on clinician edit;
- accepted revisions are never updated in place;
- exact same accepted normalized payload is idempotent;
- same id+revision with changed content conflicts;
- tombstoned identities cannot be reused.

L-1 MVP uses the conservative review rule:

```text
EVERY imported observation is treated as material for finalization
→ it must be accepted / modified / dismissed before save
```

This avoids inventing an unreviewed importance threshold. A modified observation requires `clinician_modified_statement`.

Canonical content hashing excludes server timestamps/server hash/server-authoritative review metadata and external mutable overlays. Reference-verification overlay changes never change Challenge payload/hash.

---

# 9. Challenge history / delete / export

Frozen read routes:

```text
GET /clinical/learning/api/challenges
GET /clinical/learning/api/challenges/{challenge_id}
```

History supports the frozen filters:

```text
topic
foundation_node
date
challenge_mode
review_state
```

Detail returns immutable revision history plus current external reference-verification projection and due state.

Delete:

```text
DELETE /clinical/learning/api/challenges/{challenge_id}
```

requires explicit confirmation and one transaction:

```text
purge reference overlay rows
→ purge due rows where target OR source belongs to challenge
→ purge all Challenge revisions/hashes
→ create non-content tombstone
```

MVP JSON/Markdown export is browser-side from a successfully fetched Challenge revision. It creates learning exports only; protected/internal references are excluded.

---

# 10. Reference-verification overlay

Frozen endpoint:

```text
POST /clinical/learning/api/challenges/{challenge_id}/revisions/{revision}/references/{reference_id}/verification
```

Server verifies that artifact/revision/reference exists. Imported state never advances the overlay.

Allowed states:

```text
unverified
verified_locator
verified_content
invalid_or_unresolved
```

`verified_locator` and `verified_content` require at least one reference locator (`pmid`, `doi` or `url`) in the immutable source reference. The clinician owns the verification action; L-1 does not independently claim source-content verification.

---

# 11. PHI/privacy runtime profile

The guard runs before preview acceptance and again before persistence.

Direct structured identifier keys fail closed globally:

```text
patient_name
full_name
identity_number
gesy_number
phone
email
address
date_of_birth
dob
```

All persistable untrusted strings are scanned unless the exact path is one of:

```text
references[].pmid
references[].doi
references[].url
```

Minimum deterministic signals:

```text
email address
phone-like sequence
explicit identity/GeSY-number phrase
explicit full-DOB phrase
explicit postal-address phrase
```

The implementation must not claim perfect name/de-identification detection. Clinician de-identification attestation remains mandatory, and real/mixed cases require the frozen source-case de-identification state.

Routine logs contain no Challenge/Foundation payloads, rejected values or verification notes.

---

# 12. Foundation Map / assessment profile

Frozen routes:

```text
GET  /clinical/learning/api/foundation
POST /clinical/learning/api/foundation/{foundation_node_id}/assessments/preview
POST /clinical/learning/api/foundation/{foundation_node_id}/assessments
```

The registry comes only from `schemas/osteoporosis_foundation_map_v1.yaml`.

No persisted row means:

```text
state = UNKNOWN_UNTESTED
retention_state = not_scheduled
```

Assessment request envelope:

```text
{
  "attempt": <FoundationAssessmentAttemptV1>,
  "next_review_due": <optional explicit YYYY-MM-DD>,
  "confirm_save": <required only for persistence>
}
```

The server validates:

- node/module consistency;
- unique evidence IDs;
- all L-1 evidence has `source_artifact_type=foundation_assessment`;
- self-rating alone cannot change Foundation state;
- non-unknown state requires reviewed non-self-rating evidence;
- `FORMAL_SOLID` requires demonstrated formal/mechanistic evidence plus demonstrated transfer/boundary/evidence-directness evidence.

The server does not invent a state; `proposed_state` and `clinician_final_state` remain explicit clinician-reviewed assessment content.

Retention remains separate. For the MVP, only scheduling state is derived from explicit `next_review_due`:

```text
null   → not_scheduled
future → scheduled
today/past → due
```

L-1 does not infer `retained` or `needs_refresh` from Challenge performance.

---

# 13. Due-state profile

GET:

```text
/clinical/learning/api/due
```

L-1 materializes only contract-authorized due items that have a concrete reviewed source:

```text
challenge_repetition
learning_action from Challenge revision
foundation_reassessment from explicit Foundation assessment schedule
```

`progress_review` remains representable by the schema but is not fabricated without a reviewed progress-schedule owner.

Occurrence semantics:

- start at 1;
- uncompleted occurrence may be rescheduled from a later reviewed revision;
- completed occurrence is terminal/historical;
- a later schedule after completion creates occurrence + 1;
- deferred state requires future `deferred_until` and reactivates deterministically;
- source-artifact provenance is mandatory.

L-1 exposes due state; it does not invent an adaptive spacing algorithm.

---

# 14. Clinician-facing MVP UI

`static/clinical-learning/` provides four views in one protected workspace:

1. **Challenge Import** — paste JSON → server preview → clinician review/disposition/edit → confirm save.
2. **Challenge History** — filters, revision history, delete, JSON/Markdown export, reference verification.
3. **Foundation Map** — 14 Module-01 nodes, current state, retention state, evidence count, last assessment, next due; explicit assessment preview/save.
4. **Due / Learning Actions** — due Challenge repetitions, Foundation reassessments and learning actions.

UI requirements:

- visible challenge mode/topics/Foundation nodes;
- Fact Ledger scope badges;
- progressive-disclosure provenance;
- observation category/importance/disposition;
- reference verification state;
- privacy/de-identification attestation;
- server normalization warnings;
- no composite score.

No raw transcript upload UI. No Daily Case Review UI. No external connector UI.

---

# 15. Acceptance evidence

Before implementation can be called TESTED, exact-head evidence must cover at minimum:

```text
valid + invalid Challenge schema
recursive unknown-field rejection
all persistable untrusted-string privacy scan coverage
sanitized privacy/schema errors and non-content logging
external-import server-authority normalization
idempotent duplicate / conflict / append-only revision
reference overlay without immutable-payload mutation
transactional tombstone/content purge
nested learning-action due cleanup on Challenge delete
synthetic PHI guard + bibliographic locator exclusions
topic normalization without second taxonomy
Fact Ledger / disclosure / observation / reference integrity
Foundation transition guards
Foundation state→attempt reference integrity
L-1 explicit Foundation assessment only
Foundation explicit due scheduling without adaptive cadence
due occurrence/reschedule/repeat/defer/source provenance
protected session/header auth
no patient/transcript/Signal/DailyCase write path
main router/no-store ownership
browser UI smoke for Import/History/Foundation/Due
frozen L-0 contract regression
full branch-vs-main diff hygiene
NO physiotherapy/CU-1/RF file mutation
```

Public fixtures are synthetic only.

---

# 16. REPLAN triggers

STOP implementation and return to design if any of these becomes necessary:

- change to a frozen L-0 object or ownership contract;
- a second transcript ingestion owner;
- any patient-record read/write required by L-1;
- raw transcript persistence;
- identifiable patient content in learning records/tests/repo;
- autonomous Signal creation/promotion;
- Daily Case Review runtime;
- broad external bearer credentials;
- adaptive repetition algorithm not frozen in L-0;
- Challenge-derived Foundation state mutation;
- overlap with the paused physiotherapy/CU-1 productization scope;
- schema/database ownership that cannot be implemented with the existing protected engine as frozen.

---

# 17. Lifecycle / next gate

```text
L-0 CONTRACT                         FROZEN / COMPLETE / MERGED
L-1 PRODUCT-OWNER AUTHORITY          GRANTED
L-1 SLICE DESIGN                     FROZEN BY THIS FILE
L-1 IMPLEMENTATION                   ACTIVE
L-1 TESTED                           NO
L-1 EXACT-HEAD REVIEW                NO
L-1 PR                               NONE
L-1 MERGED                           NO
L-1 DEPLOYED                         NO
L-1 PRODUCTION-SMOKE-VERIFIED        NO
```

Exact next action:

```text
claim CURRENT_OPERATIONAL writer lock
→ implement only frozen owners/seams
→ run complete L-1 + inherited L-0 regression gate
→ exact-head scope/security/privacy review
→ canonical closeout to IMPLEMENTED / TESTED or REPLAN
→ HOLD for separate release/merge authority
```
