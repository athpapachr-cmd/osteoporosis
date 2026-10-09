# Visit Capture — Dia / Heidi / GESY → Cockpit bounded delta v1

> **DATE:** 2026-10-08 Asia/Nicosia.
> **STATUS:** PRODUCT OWNER CONFIRMED / DELTA DESIGN FROZEN FOR INDEPENDENT R2 / NO RUNTIME AUTHORITY.
> **PARENT DESIGN:** `cockpit/VISIT_BRIEF_CLINICAL_INBOX_DESIGN_2026-10-07.md`.
> **PARENT R2:** PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0. That review remains closed for Q1–Q6.
> **SCOPE:** one new material behavior only: clinician-reviewed Visit Capture into the existing protected clinical encounter owner.
> **OUT OF SCOPE:** live Gmail, Zadarma, Heidi backend/API integration, GESY automation/API integration, authoritative lab-result acceptance, booking, deployment, real patient-data activation.

## 1. Product Owner checkpoint

The Product Owner clarified the real daily workflow:

```text
Cockpit runs inside Dia browser.

Open tabs:
1. Heidi summary — today's visit
2. GESY — today's visit
3. GESY — immediately previous visit
4. Cockpit — confirmed patient context

Dia uses the selected tabs as context
→ Dia inserts a proposed structured capture into Cockpit
→ Cockpit renders the proposed encounter
→ clinician reviews
→ ONE explicit Save
→ protected longitudinal clinical state
```

The clinician does **not** dictate to Dia and should not have to complete a multi-screen form. The desired workflow is near-one-step after the sources are already open.

Concrete example: after an orthopaedic follow-up, Heidi contains the richer consultation summary, today's GESY visit contains visit/coding/admin context, and the prior GESY visit may provide a comparison baseline. Dia prepares one candidate. Cockpit shows Snapshot / Visit Brief / Encounter Detail. The clinician checks the content and presses Save once.

Why this delta comes before code: the already-reviewed Visit Brief/Clinical Inbox design was read-oriented in first-code A. This delta introduces a protected clinical write, so identity, signoff, versioning and source semantics must be frozen first.

## 2. Mechanism decision — EXTEND, do not replace

Existing protected Clinical Excellence ownership remains authoritative:

- `clinical_patients.patient_id` = internal patient identity;
- `clinical_encounters` = existing protected encounter owner;
- encounter payload = existing structured clinical payload seam;
- existing server finalization semantics already distinguish `draft | completed | amended`;
- G3/validated longitudinal projections remain the read/summary mechanism.

Visit Capture therefore **EXTENDS** the existing encounter owner. It must not create a second patient database, a Dia-owned record, a GESY-owned record, or a new longitudinal truth store.

Current limitation to preserve explicitly: the existing encounter PUT marks later changed completed content as `amended`, but it overwrites the current payload and does not by itself prove immutable revision history. Visit Capture must not silently claim immutable amendments until a bounded revision mechanism is added or an equivalent existing mechanism is proven.

## 3. Destination identity is Cockpit-owned

The destination patient is selected only by the already-confirmed Cockpit patient context.

Hard rules:

- Dia cannot choose `patient_id`.
- GESY beneficiary number, name, phone, page title and browser-tab text are not the destination key.
- Candidate content must not be allowed to retarget the write.
- The server-side Save is scoped to the protected internal `patient_id`.
- Patient/context switch, logout, navigation or a stale capture context invalidates the unsaved candidate.
- The Visit Capture surface must carry a server-issued/current capture-context token or equivalent version guard bound to the patient context.
- Save with a stale/mismatched context fails closed.
- Unsaved candidate content is discarded on patient/context switch.

Source identity observations may be included only as **untrusted evidence**. If selected source tabs appear to concern different patients, or source attribution is uncertain, the candidate is `identity_conflict` / `identity_uncertain` and cannot silently save as normal clinical truth.

## 4. Source roles

### 4.1 Heidi today

Primary rich source for what occurred in today's consultation.

It may support:
- reason for visit;
- symptoms/course;
- findings stated in the summary;
- clinical impression;
- decisions;
- medications/instructions actually recorded;
- pending actions;
- next review;
- safety-net only when source-stated.

Heidi content remains source material. Dia extraction is a candidate until clinician review + Save.

### 4.2 GESY today

Current-visit supporting source for:
- visit date/type/specialty where present;
- coding/billing entries;
- administrative/referral context;
- other explicitly stated current-visit facts.

GESY coding is a separate coding layer, not the complete clinical source of truth. Missing coding means `coding_complete: false` or equivalent, not an automatic clinical conflict.

### 4.3 GESY previous

A bounded comparison source only.

It may support `what_changed` only where the current and previous facts are genuinely comparable. No invented `improved / worsened / unchanged / resolved` state is allowed.

Comparison basis is explicit:
- previous structured encounter;
- previous GESY visit;
- unavailable.

If no safe basis exists, omit the change claim or show comparison unavailable.

## 5. Candidate insertion surface

First-code UI should expose one Dia-friendly Visit Capture surface inside Cockpit, reachable from confirmed patient/appointment context.

Preferred interaction:

```text
Dia inserts ONE structured candidate payload
→ Cockpit validates locally/server-side
→ Cockpit renders human-readable preview
→ clinician reviews
→ Save
```

The clinician should not have to review raw JSON field by field. A machine-readable candidate may be inserted into one controlled input, while Cockpit renders the three human projections before Save.

Insertion/preview is **not** a clinical write.

A malformed payload, unsupported version, identity conflict, stale context or validation failure must remain unsaved and clearly visible.

## 6. One encounter, three projections

One underlying structured encounter produces:

### Level 1 — Snapshot
Rapid glance, approximately 4–6 lines:
- today's core problem/reason;
- key change where safely supported;
- decision;
- open/pending item;
- next contact/review.

### Level 2 — Visit Brief
Normal clinician working view:
- why the patient came;
- important course/change;
- key findings;
- clinical impression;
- decisions;
- pending items;
- next review;
- safety-net only when actually stated.

### Level 3 — Encounter Detail
Richer source-backed view:
- detailed course/findings;
- source bindings/provenance;
- coding layer;
- medication details actually supplied;
- uncertainty/conflict;
- administrative/legal items when clinically relevant;
- full pending/next-review context.

The three levels are renderers/projections, not three separately authored or separately authoritative summaries.

## 7. Minimal encounter payload contract

Exact runtime schema names may reuse current payload conventions, but the semantic minimum is:

```text
encounter metadata
- encounter_date
- visit_type/specialty where available
- reason_for_visit

source_bindings[]
- source_ref
- role: heidi_today | gesy_today | gesy_previous | clinician_edit
- captured_at / source date where available
- no raw page body required

comparison_basis
- type
- source_ref if applicable

what_changed[]
findings[]
clinical_impression[]
coding
decisions[]
medications[]
next_contact
safety_net[]
uncertainties[]
```

Each material extracted fact can carry:
- provenance/source reference(s);
- verification state, e.g. `source_stated | clinician_confirmed`;
- certainty state, e.g. `certain | uncertain | conflicting`.

Do not collapse provenance/verification with certainty.

The clinician's Save is encounter-level signoff. It does not automatically rewrite every source-stated fact as `clinician_confirmed`.

## 8. Partial medication data

Structured medication fields must allow missing values.

Example source:

```text
Narox approximately 10 days
```

Valid capture:

```text
name = Narox
action = prescribed/continued as source-stated
duration.value = 10
duration.unit = days
duration.approximate = true
dose = null
frequency = null
route = null
```

Missing source detail remains missing. Dia/Cockpit must not complete dose, frequency, route or exact duration by inference.

## 9. Pending items are first-class

Pending work must not be buried only in prose.

Minimum semantics:
- stable `pending_id`;
- type/description;
- trigger/condition where applicable;
- status;
- responsible owner/role;
- external dependency separately where applicable;
- review/due context when known;
- provenance;
- created_from encounter;
- resolution reference when later closed.

Representative states may include:
`not_yet_indicated | open | awaiting_patient | awaiting_result | resolved | cancelled`.

Do not force the exact enum if an existing CareTask/pending mechanism can safely be EXTENDED.

Owner and dependency are different. Example: "medical report if requested by lawyer" may have clinician as responsible owner and lawyer/request as the dependency.

## 10. Later events do not rewrite the visit

Events after the encounter are separate longitudinal events/attention/task/communication objects under their rightful existing owners.

Examples:
- MRI arrived;
- lab result arrived;
- clinician reviewed;
- patient informed;
- follow-up occurred.

The historical encounter remains unchanged. Visit Brief may project these later events together with the encounter.

Do not implement a mutable `events_since[]` inside the old encounter if every later event would require rewriting the visit.

## 11. Signoff, amendment and version safety

Candidate state:
```text
Dia/browser insertion → transient candidate
```

Authoritative transition:
```text
clinician reviews → explicit Save → completed/signed encounter state
```

After Save:
- silent overwrite is forbidden;
- a later correction must be attributable as an amendment/revision;
- stale expected version must fail closed;
- the original signed state must remain reconstructable before production use claims amendment history.

The existing `draft/completed/amended` semantics are a REUSE seam, not proof of immutable revision history. First implementation may either:
1. add the minimum append-only encounter revision mechanism; or
2. keep post-signoff editing disabled/fail-closed until that bounded mechanism exists.

It must not pretend overwritten payload is immutable history.

## 12. Privacy / retention boundary

No patient clinical content belongs in GitHub.

For Visit Capture:
- browser/Dia reads selected sources;
- Cockpit receives the proposed structured candidate;
- raw Heidi/GESY page bodies are not persisted merely because they were used;
- persist only the protected encounter content and minimum useful source/provenance references;
- no source page content in public logs, analytics or error messages;
- candidate content remains transient until Save;
- failed/abandoned candidates receive bounded transient handling and are not indefinite hidden records.

Real identifiable use of Dia/Heidi/GESY remains behind a processor/privacy/live-activation qualification. That gate does not prevent synthetic/manual first-code work.

## 13. Relationship to ClinicalAttentionItem

Do not collapse:
```text
ClinicalEncounter
!= PendingItem/CareTask
!= LongitudinalEvent
!= ClinicalAttentionItem
!= CommunicationAction
!= accepted authoritative lab result
```

Visit Capture creates/updates the clinician-owned encounter and deliberate pending work. It does not turn today's Heidi/GESY tabs into Clinical Inbox items, accept later results, send messages or close future tasks automatically.

Clinical Inbox events can later project into Visit Brief without rewriting this encounter.

## 14. Narrow first-code boundary

If the independent delta R2 passes, the preferred first-code boundary is:

1. confirmed patient → **Καταγραφή επίσκεψης** surface;
2. one synthetic/manual `VisitCaptureCandidateV1` insertion surface;
3. strict validation + patient/capture-context stale guard;
4. human preview of Snapshot / Visit Brief / Encounter Detail from one candidate;
5. one explicit Save into the existing protected encounter owner;
6. first-class pending items only through an existing suitable owner or the minimum bounded extension;
7. no raw source persistence;
8. no live Dia/Heidi/GESY provider access in tests;
9. no Gmail;
10. no Zadarma;
11. no authoritative lab-result acceptance.

Implementation must be synthetic/manual first. Browser automation may emulate a Dia insertion without using real Dia or real patient data.

## 15. Finite synthetic acceptance oracles

1. **Straightforward capture:** confirmed synthetic patient; Heidi-like current source + current GESY-like metadata + prior baseline; one candidate; Save once; all 3 projections come from one encounter.
2. **No safe comparison:** encounter saves; change section is unavailable/omitted; no invented trend.
3. **Cross-patient source conflict:** candidate is visibly conflicted and cannot silently save as normal truth.
4. **Stale patient context:** candidate prepared under A; context changes; old candidate/context Save fails closed.
5. **Partial medication:** name + approximate duration only; absent dose/frequency stay null.
6. **Later event:** a later synthetic MRI/lab/communication event does not mutate the signed encounter.
7. **Post-signoff correction:** no silent overwrite; amendment is attributable, or editing is explicitly fail-closed until revision support exists.
8. **Three projections:** Snapshot/Brief/Detail are regenerated from the same saved structured encounter, not separately stored prose authorities.

## 16. Explicit live-activation gates

Remain OPEN and separate from synthetic first-code authorization:
- actual Dia browser processor/privacy suitability for identifiable clinical data;
- actual Heidi terms/processor/privacy posture for this workflow;
- actual GESY access/processor/session policy for browser-assisted use;
- source-tab scoping/identity reliability in real Dia behavior;
- transient browser/cache/log handling qualification;
- field-level production retention/access/deletion policy where not already closed;
- any live external provider integration.

These gates must not be converted into a claim that synthetic/manual first code is live-ready.

## 17. Review boundary and stop rule

This delta requires ONE fresh independent pre-code R2 because it adds a clinical write boundary after the parent R2 PASS.

The reviewer must assess only:
- existing-owner reuse;
- patient/write safety;
- encounter contract;
- longitudinal separation;
- Dia/Heidi/GESY browser boundary;
- narrow synthetic first-code eligibility.

Do not reopen parent Q1–Q6 unless this delta directly contradicts one of them.

PASS / P0:P1:P2 = 0:0:0 closes this delta for declared scope and stops review. A concrete BLOCK receives one bounded correction and one affected closure review only.
