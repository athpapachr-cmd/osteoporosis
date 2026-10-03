# Cockpit CURRENT — global Home and bounded clinical projections

> **NOW (2026-10-02):** Home v1, Pending Surgery Queue and the bounded Clinical Calendar reason bridge are released. **D1 Today Context Strip V1 is deployed but has a Product Owner source/window defect:** it reads the Osteoporosis Clinical Calendar/day window rather than the Digital Secretary global appointment schedule, so it can show no Previous/Current/Next even while the Reception dashboard has real appointments. The next bounded Cockpit action is a D1.1 source/window correction checkpoint. **D2 Relevant Communication Context is deferred as a supporting input to Visit Brief, not the next standalone Home priority.**
> **Product boundary:** `cockpit/PRODUCT_CONSTITUTION.md`; cross-programme architecture: `CLINICAL_EXCELLENCE_PLAN.md` §§33–34.
> **Root lock:** `CURRENT_OPERATIONAL.md` remains the sole repo-wide NOW. Its PR-1 release lifecycle is in HOLD with no active runtime writer at the verified PR-1 branch checkpoint; this Cockpit file does not claim that lock.
> **Visit Intelligence dependency:** `programme/OST-VISIT-INTELLIGENCE/CURRENT.md`; P0-V0 contract/projector branch `b22fd5c610d20baf0f8ef3384ca16472cd2d1f92` is not on `main`. The external cumulative PASS handback has not yet been reconciled into that workstream CURRENT or released.

## Current bounded action

The active Cockpit step is **D1.1 — Global Appointment Context source/window correction** in **R2 PRE-CODE BLOCK / BOUNDED DESIGN CORRECTION**. Product Owner authorized “D1.1 → Visit Brief” on 2026-10-02. No D1.1 runtime implementation is authorized until one independent pre-code review passes. Visit Brief remains the immediately following product slice. Exact review target: branch `design/cockpit-d1-1-global-context-2026-10-02` head `9b11f722c485247754fa7219e9b0f0d9167f8734`, design blob `2b438c949e30d00a1d11d9d1431d766954320f15`. Runtime implementation remains **NO**. Product Owner's 2026-10-03 handback is BLOCK, complete declared coverage, P0:P1:P2=0:2:0: D1.1-PRE-01 requires snapshot-only `other` admission while preserving legacy discard; D1.1-PRE-02 requires phone/link minimization on every transition to effective `other`, including clearing a manual override. Next: bounded design correction and one independent delta + affected cumulative closure review; no author-certified PASS.


### Product Owner correction — 2026-10-02

Production use exposed that the deployed D1 satisfies its tested local semantics but is connected to the wrong product projection for the **global** Cockpit.

Observed behavior:
- Reception / Digital Secretary already shows real clinic appointments from its existing schedule view;
- Cockpit D1 reads the Osteoporosis Clinical Calendar endpoint and today's local-day window;
- therefore a non-osteoporosis appointment — and an immediate next appointment on the following day — may be visible in Reception while Cockpit shows no Previous / Current / Next.

**D1.1 required correction:**
- Previous / Current / Next on global Cockpit must come from the existing Digital Secretary / appointment schedule truth, not from the Osteoporosis-only Clinical Calendar projection;
- reuse the Digital Secretary's existing protected schedule mechanism (currently surfaced by its dashboard via `GET /dashboard/schedule`) rather than creating another Cal.com fetcher/calendar;
- the projection must be global across appointment types/clinics;
- `Next` must resolve the immediate future appointment across the relevant schedule horizon and must not stop at the local-day boundary merely because the Home is labelled “Today”;
- the weekly Osteoporosis Clinical Calendar remains a separate Module-01 view and must not be broadened merely to make global Cockpit work;
- Digital Secretary / booking source remains owner; Cockpit remains read-only;
- cross-service authentication/data minimization must be designed explicitly rather than reusing a browser dashboard cookie implicitly.

The existing D1 Aclasta-overlap/start-order behavior remains a useful scheduling rule where the required appointment classification is available; D1.1 must preserve it or fail closed if the source cannot support that classification safely.


**Visit Brief interaction decision — Product Owner 2026-10-02:** keep Home compact. Previous / Current / Next should show only concise appointment context by default. Clicking/tapping the patient opens a floating/overlay Visit Brief rather than expanding all history inline. The overlay should prioritize: main problem/reason → relevant history → prior-visit decision/plan → pending/awaited items → what is expected/checking today → relevant communication only when it materially changes preparation. A full-record navigation action may remain available from the overlay. This is a product checkpoint only; no Visit Brief implementation authority is granted yet.

**Product priority correction:** Relevant Communication Context is not the primary next clinician-facing layer. The next meaningful Home layer after correct appointment context is a compact **Visit Brief / Patient Context** answering, in this order:

1. who is this patient / what is the main problem or reason for today's visit;
2. what matters from the relevant clinical history;
3. what was decided/said at the previous visit;
4. what remains pending or awaited;
5. what we are expecting/checking today;
6. only then, any communication that materially changes the visit preparation.

Communication remains bidirectional and Digital Secretary-owned, but it should normally appear as supporting evidence/context inside the Visit Brief rather than as the dominant card.

Where no strongly linked protected clinical patient record exists, Cockpit must not infer one from phone/name alone. It should show only the bounded appointment/operational context and fail closed for patient-specific clinical history until an authorized strong or clinician-confirmed link exists.

D1 is a small, directly visible, read-only Dashboard projection of **Previous / Current / Next** appointment context from the existing normalized Clinical Calendar. It lets the clinician see who was just seen, who is being seen now and who is next without opening the full calendar. The existing weekly Osteoporosis Clinical Calendar remains available through a link labelled **«Άνοιγμα εβδομαδιαίου ημερολογίου»** and is not replaced or removed.

**R1 classification:** bounded UI/read-only projection using unchanged protected Calendar data, with no new clinical fact authority, patient identity authority, write path or external side effect. Expected implementation scope is `static/cockpit/index.html`, `static/cockpit/app.js`, `static/cockpit/styles.css`, `test_cockpit_home.py` and this workstream checkpoint; root `CURRENT_OPERATIONAL.md` records the bounded writer lock. The Clinical Calendar and its source keep ownership. D1 authorizes no appointment write, booking/cancellation/rescheduling, second calendar, Setmore reminder, Digital Secretary workflow change, messaging transport, Visit Intelligence clinical-state write or patient-matching redesign.

**Accepted D1 semantics:** clinician attention follows appointment `start_at` order. `Τώρα` is the single active appointment; when an earlier still-active row is an Aclasta infusion slot and a later non-Aclasta appointment has started, the later appointment is Current. Other simultaneous active overlaps fail closed visibly. `Προηγούμενο` is the immediately preceding appointment in start order relative to Current, or the latest-started completed appointment when there is no Current. `Επόμενο` is the earliest appointment whose `start_at` is later than now. This is schedule context, not live patient-presence tracking. The Home may show `patient_display_name` only from the authenticated protected Calendar response for these slots; it must not render `phone_e164` or `linked_patient_id`.


### D1 implementation/tested checkpoint — 2026-10-01

```text
BASE MAIN:             87aedad3ad512e4b17a1eb737f0ff8302857aff2
BRANCH:                feat/cockpit-today-context-strip-v1-2026-10-01
SUBSTANTIVE HEAD:      b4ce49b6cdb1083d56453abb1e4cbc10c834b294
R0 DOCS-CORRECTION:    a965cec7c307c354d7a2ef98d7e2ff2937fe46ee
PR:                    #130 / MERGED
IMPLEMENTED:           YES
FOCUSED TESTED:        YES
INDEPENDENT R1 REVIEW: CLOSURE PASS / REVIEW CHAIN STOPPED
MERGED:                YES / db42903e08b11dfe52d427a410f8b15547fb5507
DEPLOYED:              YES / LIVE / dep-davbs48473hc73eust00 / source 794d96dc4e0676691a4d6333b8f47bdd335cdb6c
```

Implemented behavior:

- Cockpit Home renders **Προηγούμενο / Τώρα / Επόμενο** directly from the existing authenticated Clinical Calendar read endpoint.
- Previous / Current clinician attention follows deterministic `start_at` order. A later-started non-Aclasta visit may lawfully become Current while an earlier Aclasta infusion slot remains active; other simultaneous active overlaps fail closed visibly. Next remains the earliest future `start_at`.
- the context refreshes once per minute while the Dashboard remains open;
- `patient_display_name` is used only as protected appointment display context; D1 does not consume `phone_e164` or `linked_patient_id`;
- the existing weekly Osteoporosis calendar remains unchanged and reachable through **«Άνοιγμα εβδομαδιαίου ημερολογίου»**;
- daily-feed freshness remains stated in clinician-facing language without exposing technical sync metadata;
- no Calendar write, booking lifecycle, Setmore, Secretary, Zadarma, Visit Intelligence, OST-UI or PR-1 behavior changed.

Focused evidence on the substantive head:

- Cockpit Home tests — run `36897402962` — **SUCCESS**;
- Cockpit surgery queue — run `36897403159` — **SUCCESS**;
- Canonical impact guard — run `36897403120` — **SUCCESS**;
- Physio Knee OA Cockpit integration — run `36897403296` — **SUCCESS**;
- Physio Knee OA V5 integration — run `36897403218` — **SUCCESS**;
- Physio jurisdiction overlay — run `36897403208` — **SUCCESS**;
- Clinical Learning L1 / L1B / L1C inherited gates — **SUCCESS**.

Clinical Learning L0 run `36897403005` validated its own contracts but failed only its **design-only scope** assertion because D1 intentionally changes Cockpit runtime files. That scope gate is not applicable evidence for this R1 Cockpit implementation and does not indicate a Clinical Learning contract regression.

**Independent R1 result:** the first exact-head review returned **BLOCK** because it interpreted `Previous` as “the completed row with greatest `end_at`” and used an overlapping historical example (08:00–11:00 and 09:00–10:00).

**Final Product Owner clarification of D1 scheduling semantics:** normal appointments are sequential, but **Aclasta is a legitimate exception**. Aclasta may occupy a one-hour treatment slot while the patient is in a clinic room and the clinician may start another appointment during the last part of that infusion slot. Therefore clinician attention follows **appointment start order**, not “greatest end time”.

Accepted examples:

- 08:00–09:00 A, 09:00–10:00 B, now 11:30 → **Previous = B**;
- Aclasta 09:00–10:00 + Review 09:40–10:20, now 09:50 → **Previous = Aclasta / Current = Review**;
- Aclasta 09:00–10:00 + Review 09:40–09:55, now 10:05 → **Previous = Review**, even though Aclasta ended later.

**Bounded implementation rule:** D1 uses the ordered `start_at` sequence for Previous/Current attention. If multiple appointments are active and all earlier active rows are `aclasta` while the latest-started active row is non-Aclasta, the latest-started row is Current and the immediately preceding scheduled row is Previous. Other concurrent-current overlaps remain ambiguous and fail closed visibly. Next remains the earliest future `start_at`. Weekly Calendar preservation, identity/privacy boundaries and ownership are unchanged.

**Final bounded correction tested candidate:** substantive runtime/test head `b4ce49b6cdb1083d56453abb1e4cbc10c834b294`.

Focused evidence on that head:

- Cockpit Home — run `36904690143` — **SUCCESS**; this executes the production `appointmentContext` logic and covers sequential Previous, lawful Aclasta overlap, post-overlap Previous by later start, non-Aclasta overlap fail-closed, syntax, workspace navigation and diff hygiene;
- Cockpit Surgery Queue — run `36904690332` — **SUCCESS**;
- Canonical Impact — run `36904690084` — **SUCCESS**;
- Clinical Learning L1/L1B/L1C inherited gates — **SUCCESS**.

The persistent Clinical Learning L0 red check remains the same non-applicable design-only changed-file scope assertion; its contract validation step passes and D1 does not change Learning contracts.

**Final independent closure disposition:** all runtime/product D1 behavior above was accepted. The only remaining BLOCK was two stale canonical statements that still described the superseded `end_at` semantics. Those two statements were corrected in the R0 canonical-only delta at `a965cec7c307c354d7a2ef98d7e2ff2937fe46ee`; no runtime or test semantics changed. Under `PROCEDURES.md` P4/P5/P7 this requires no new product/implementation review, so the D1 review chain is stopped.

**Exact next action:** the Product Owner has now authorized the lawful PR #130 merge/release step provided the current exact-head state and applicable CI remain clean. PR #130 was squash-merged as `db42903e08b11dfe52d427a410f8b15547fb5507`. Render deploy `dep-davbs48473hc73eust00` reached **LIVE** from source `794d96dc4e0676691a4d6333b8f47bdd335cdb6c`. D1 is released and its bounded writer is released. No production smoke beyond successful deploy health is claimed. **Exact next action:** present the D2 Relevant Communication Context plain-language Product Owner checkpoint with concrete UI examples; do not implement D2 until the Product Owner confirms or corrects it. Do not start D2 implementation before D1 is durably released.

**D2 — Relevant Communication Context** is the next bounded concept after D1, not part of this governance implementation. Its bidirectional read boundary and phone-correlation limit live in `cockpit/PRODUCT_CONSTITUTION.md`. D2 needs its own bounded design/authority before any integration or UI work.

**Visit Brief / What Changed** remain product direction, downstream of durable P0-V0 closure/release and the OST-UI coordinator synthesis/Product Owner decision where the Module-01 interaction is affected. The current-problem Osteoporosis workspace also depends on that synthesis. A Next Visit continuity write path remains a separate bounded clinical-authority slice. None of these dependent slices starts from this checkpoint, and OST-UI's R4 completion does not authorize a prototype.

The release and follow-up records below are as-of checkpoints. Their former “next” text is historical and cannot override this current action or the owning workstream CURRENT.

---

# Cockpit Home v1 release checkpoint

> **STATUS:** COCKPIT HOME V1 RELEASE COMPLETE / PR #118 MERGED / RENDER LIVE / ROOT SMOKE VERIFIED.
> **Workstream:** Clinical Excellence Cockpit Home v1.
> **Branch:** feat/cockpit-home-v1-2026-09-27.
> **Base main:** 88ad125f0a25a471b0151eeb26e68b8b8a93c84f.
> **Exact tested head:** 7744e2142b54ba0b7c8b92c87e4e511cffffc26b.
> **Root writer lock:** unchanged; PR-1 Heidi-first transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this slice is limited to global Cockpit navigation/home presentation and Osteoporosis sidebar cleanup.

## Product-owner decisions

- / must open the global Cockpit Home, not the Osteoporosis module.
- Osteoporosis is Module 01 / proving ground, not the whole Cockpit.
- Osteoporosis sidebar must contain only Module-01 navigation.
- Global Clinic Utilities belong on Cockpit Home.
- The duplicate physiotherapy referral entry must collapse to one global tool entry.
- The top-level Heidi AI navigation item must be removed from Osteoporosis; Heidi remains an encounter-capture capability inside the Module-01 workflow and future reusable Core.
- The separate Reception/calls dashboard remains owned by ortho-reception-backend-v2; Cockpit Home links to it rather than duplicating its implementation.
- Calendar/Cal.com reason ingestion is a separate follow-up integration slice because its source-of-truth boundary spans repositories.

## Home v1 information architecture

Cockpit Home
- Today
  - Clinical Calendar
- Clinical Modules
  - Osteoporosis · Module 01 · Active
- Learning & Improvement
  - Clinical Learning Hub
- Clinic Utilities
  - Παραπεμπτικό Φυσιοθεραπείας
  - Αναρρωτική άδεια
  - Ιατρικές εκθέσεις
  - Ραδιοκύματα
- Reception
  - Call / Reception dashboard (external owner)

Module 02+ appear as future placeholders only; no fake runtime is implied.

## Safety / scope

- no patient/clinical data model change;
- no Cal.com/Setmore/Digital Secretary mutation in this slice;
- no PR-1 transcript runtime mutation;
- no duplicate copy of the Reception dashboard;
- no new browser patient persistence;
- root Clinical Auth boundary remains unchanged.

## Verification

Cockpit Home gate:

```text
run: 36273987676
head: 7744e2142b54ba0b7c8b92c87e4e511cffffc26b
result: SUCCESS
```

Verified:

- Python syntax PASS;
- Cockpit browser JavaScript syntax PASS;
- 4 deterministic Cockpit Home tests PASS;
- Osteoporosis G4 workspace/navigation regression PASS;
- diff hygiene PASS.

Behavior proven:

- / enters /static/cockpit/;
- global Cockpit Home contains Modules, Calendar, Learning, Clinic Utilities and Reception;
- Clinic Utilities are absent from the Osteoporosis sidebar;
- top-level Heidi AI is absent from the Osteoporosis sidebar;
- one global physiotherapy referral entry exists;
- Home calendar summary uses aggregate counts only and does not render patient identity fields.

## Release PR

PR: #118
URL: https://github.com/athpapachr-cmd/osteoporosis/pull/118
base: main
base_sha: 88ad125f0a25a471b0151eeb26e68b8b8a93c84f
head: feat/cockpit-home-v1-2026-09-27
head_sha at PR creation: f0f252dfaef9c08146c636e5d10fbf7ede97d553
draft: NO
merged: NO
deploy: NO

## Release completion

```text
PR: #118
merge method: squash
merge commit: b7b8779d943eb1d8db1f8966a81796bc69647c3b
merged: YES
Render service: osteoporosis / srv-d5qfk31r0fns73di596g
deploy: dep-das41frncjis73e6q47g
deploy trigger: new_commit
deploy status: live
manual redeploy: NO
```

Final PR exact head `ba2de644f4fd16e70d12f04831b28f5fd42730ec` passed all 16 checks, including Cockpit Home, Canonical Impact, G3/G2/G1, Clinical Learning L1/L1B/L1C, Clinical Documents, CU-1 and Physio inherited browser gates.

Production smoke from Render logs:

```text
GET /                 -> 307 Temporary Redirect
GET /static/cockpit/  -> 200 OK
Application startup   -> complete
Clinical storage      -> PostgreSQL online
```

Released navigation ownership:

- global Clinic Utilities live only on Cockpit Home;
- Osteoporosis remains Module 01 and no longer owns the general utilities group;
- top-level Heidi AI navigation is absent from the Osteoporosis sidebar;
- Heidi capture/exposure content remains inside the Module-01 encounter workflow;
- exactly one global physiotherapy referral entry is exposed on Home;
- Reception remains a linked separately-owned system.

## Current release state

```text
IMPLEMENTED: YES
TESTED: YES
PR #118: MERGED
DEPLOYED: YES / LIVE
ROOT -> COCKPIT HOME: VERIFIED
MODULE-01 SIDEBAR CLEANUP: VERIFIED
CAL.COM REASON BRIDGE: CLOSED / DAILY 04:00 UTC
```

## Historical Home v1 follow-up

HOLD the completed Home v1 release. The Cal.com → Clinical Calendar bridge is now CLOSED on a daily 04:00 UTC cadence; do not reopen that integration without new production failure evidence. The active Cockpit follow-up is the separately bounded Pending Surgery Queue v1.


---

# Clinical Calendar reason bridge — bounded follow-up slice

> **STATUS:** CLOSED / DAILY RUNTIME + REPO CADENCE ALIGNED.
> **Workstream:** Cal.com visit-reason normalization into the Clinical Calendar.
> **Branch:** `feat/calendar-reason-snapshot-v1-2026-09-27`.
> **Base main:** `a0cf912b0fcb5caf54f4a48f8f8a698908fbb0a4`.
> **Product-owner resume:** 2026-09-27 — continue the exact follow-up slice named above after recovery from the interrupted PR1 Βελτιώσεις Eval conversation.
> **Root writer lock:** unchanged; PR-1 Heidi-first transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this branch is limited to the Clinical Calendar integration contract and its tests/docs. It does not mutate PR-1 transcript runtime.

## Accepted boundary

- Cal.com remains the appointment-source truth for the feed.
- Reception/Digital Secretary remains the external integration owner; no Cal API credential is added to this repository.
- Clinical Calendar remains the normalized clinical consumer.
- This slice is read-only with respect to bookings: it must not create, cancel, reschedule or authorize appointments.
- Reception reason-triage / availability / booking semantics are explicitly out of scope.
- The existing Clinical Calendar classifier is reused; no second semantic reason owner is introduced.
- Snapshot reconciliation must remove stale source rows inside the declared source/time window so cancellations/reschedules do not remain as phantom appointments.
- Raw provider payloads are not persisted; only the existing normalized appointment fields are stored.

## Current implementation order

1. **DONE / MERGED / LIVE:** consumer snapshot endpoint and reconciliation contract via PR #119.
2. **DONE:** initial producer PR #152 merged after exact-head compatibility PASS.
3. **DONE / SOURCE-CORRECTED:** provider evidence showed no dedicated visit-reason `bookingFieldsResponses` field on current Limassol/Evrychou Cal.com event types. The Digital Secretary's own booking schema/create path and provider logs establish `metadata.notes` as the Secretary-created booking reason source.
4. **DONE / MERGED / LIVE:** producer correction PR #153 binds the clinical feed strictly to `metadata.notes`; missing notes stay blank and generic reason-like booking-field fallbacks remain forbidden.
5. **DONE / CONFIGURED:** consumer shared ingest authentication and producer snapshot URL/shared ingest key are live. No Cal.com custom-field configuration was added.
6. **DONE / PRODUCTION EVIDENCE:** lawful daily runs on 28–30/9 reached the bridge but exposed a 20-second free-service cold-start timeout, not an auth/provider-field defect.
7. **DONE / CORRECTED / LIVE:** Backend PR #154 raised only the Clinical Calendar delivery timeout to 90 seconds and aligned `render.yaml` to the final daily `0 4 * * *` cadence. The existing Render cron is daily and no second cron exists.
8. **CLOSED:** the next normal daily run is post-close monitoring only and may reopen the slice only on new failure evidence.

## Recovery checkpoint

```text
RECOVERY CHECKPOINT ID: COCKPIT-CALENDAR-REASON-BRIDGE-20260927-A
STATUS: CLOSED / DAILY 04:00 UTC
IMPLEMENTATION COMMITS:
- 20e5687fc4283318966fca7e680b3e62eaede43f — snapshot reconciliation endpoint + shared import path
- 83da6339c8e4c4841e910a8e6135a91bf00380e5 — canonical source / exact window validation
- 426a4037687d83179d8df5639957f621a579cfa2 — focused snapshot tests
- 3bfba4c9d13b0a4a955ed3d470e80115c75c7d84 — current snapshot/reason integration contract
- c07daaf75359b071ec2f5b51d5864127965d122a — path-scoped Clinical Calendar CI gate
- c28ba8c5901e43801ab5998781b45521dd3495e5 — DST/cardinality acceptance correction
- cb0752d770d2afc2b6828c6fa4e7c48f25c12754 — deterministic compatibility boundary tests
- bbd5c7bca0fd109797fac5a03c15a51f0f53af21 — schema contract alignment / final reviewed consumer head
VERIFICATION:
- Clinical Calendar snapshot workflow run 36308943111 / job 108590995955: PASS
- Python syntax: PASS
- deterministic Clinical Calendar tests: 12 passed in 1.29s
- diff hygiene: PASS
- Canonical impact guard run 36308943000: PASS
PR #119: MERGED — Clinical Calendar: add source snapshot reconciliation
PR URL: https://github.com/athpapachr-cmd/osteoporosis/pull/119
EVIDENCE HEAD: bbd5c7bca0fd109797fac5a03c15a51f0f53af21
MERGE COMMIT: 08251120a1743a1ac629e38626d04747a8fda1e9
PRODUCER INITIAL MERGE: PR #152 / 4923cc83a1a57a844be5c713b83af125f7c86f61
PROVIDER-EVIDENCE CORRECTION: PR #153 / d99738940ed439fedbcadde100c3993177aa556d / workflow 36310831720 / 16 tests PASS
PROVIDER REASON SOURCE: metadata.notes for Secretary-created bookings; no current custom reason-field slug observed on event types 341357/341358
CONSUMER AUTH DEPLOY: dep-daseid97lnhs738rkd1g / LIVE
PRODUCER ACTIVATION DEPLOY: dep-dasejdgjo6nc73bf2keg / LIVE
POST-ACTIVATION MANUAL SYNC: 0
LAST SAFE RESUME POINT: consumer and corrected producer are live and mutually configured; first real snapshot delivery has not yet been observed and must not be inferred from configuration alone
EXACT NEXT ACTION: verify the first existing scheduled /admin/trigger-sync produces clinical_calendar configured=true / attempted=true / sent=true; then reconcile the separate cron cadence drift
FORBIDDEN ON RESUME: PR-1 mutation, Reception reason/availability/booking behavior change, second sync cron, guessed Cal.com reason field, phone/SIP or booking/business mutation
```


---

# Pending Surgery Queue — bounded Cockpit follow-up slice

> **STATUS:** V1 MERGED / LIVE.
> **Workstream:** global Clinical Excellence Cockpit / pending surgery coordination.
> **Branch:** `feat/cockpit-surgery-queue-v1-2026-09-30`.
> **Base main:** `2ae9f01ded14bd2106ee09f47acb3fea4d72bc4b`.
> **Product-owner direction:** 2026-09-30 — add a pending surgery list to the dashboard with patient identity/contact data, procedure/laterality, manual ordering/sorting and surgery date.
> **Root writer lock:** unchanged; this workstream uses `cockpit/CURRENT.md` under AGENTS §4.2.

## V1 product contract

- Global Cockpit feature; not Osteoporosis-specific and not Reception booking logic.
- Reuse the existing protected Clinical Auth browser session and the existing PostgreSQL engine.
- Patient identity stays server-side; no localStorage/sessionStorage persistence.
- Pending list fields:
  - ονοματεπώνυμο;
  - ΑΔΤ / stable patient identity;
  - ημερομηνία γέννησης;
  - τύπος επέμβασης;
  - πλευρά;
  - τηλέφωνο;
  - ημερομηνία χειρουργείου.
- Manual persisted queue order supports move up/down.
- Table supports non-persistent sorting by displayed columns without silently changing the manual queue order.
- Marking a case completed removes it from the default pending list but preserves the record.
- Surgery Queue stores the requested identity/contact snapshot in its own protected server-side table. It does **not** assume generic `clinical_patients.patient_id` equals ΑΔΤ and does not silently mutate the longitudinal patient registry. Explicit future patient-linking is a separate product slice.
- Synthetic test data only; no identifiable patient data in repository or CI.

## Implementation checkpoint

- protected `clinical_surgery_queue.py` API with PostgreSQL persistence;
- protected queue-owned identity/contact fields; no implicit `clinical_patients` mutation or ΑΔΤ→generic patient-id assumption;
- create/update/list/move/complete lifecycle;
- Cockpit table with sortable columns, inline surgery date, ↑/↓ manual order, edit and complete controls;
- no browser storage of patient identity;
- focused synthetic API/UI tests + dedicated CI workflow;
- no Reception/Cal.com/booking behavior change.

## Release evidence and source-driven correction

- PR #124: `Cockpit: add protected pending surgery queue`.
- exact implementation head before this checkpoint: `a4feb0aeec452677b2b07cc283bd5c16aa2e0ec4`.
- focused workflow `36714622069`: PASS.
- Python syntax: PASS.
- Cockpit JavaScript syntax: PASS.
- surgery queue API/UI tests: **10 passed in 0.64s**.
- diff hygiene: PASS.
- Canonical impact guard `36714622173`: PASS.
- inherited Cockpit Home, Clinical Documents P1/P2, Learning L1, CU-1, Physio integration and G3 regression gates: PASS.
- **Source-driven correction after that head:** existing `clinical_patients.patient_id` is generic, not canonically ΑΔΤ. The queue contract was corrected to keep identity fields in the protected surgery row and avoid hidden mutation/duplication of longitudinal patient identity.
- corrected final head: `14707fc79e4941f65b1e180a4ff2510b420420f2`.
- corrected focused surgery workflow `36715409386`: PASS / **10 tests in 0.81s** / Python + JavaScript syntax + diff hygiene PASS.
- corrected Canonical Impact `36715409552`: PASS.
- corrected Cockpit Home, Clinical Documents P1/P2, Learning L1, CU-1, G3 and Physio Cockpit browser/integration gates: PASS.
- PR #124 merge: `3ab6a0fe0dc83047ab37925a8dddde2c032ba8fe`.
- initial merge deploy `dep-daug4j49v7es73bjk910` was superseded by the required post-merge checkpoint deploy;
- final Render deploy `dep-daug4tnf3r2c73fr2h30`: **LIVE** / source `205ca3369abf5b0dcec5a8db30b945e1748bce98`.

## Release state

- PR #124: MERGED.
- merge commit: `3ab6a0fe0dc83047ab37925a8dddde2c032ba8fe`.
- final deployed source/checkpoint: `205ca3369abf5b0dcec5a8db30b945e1748bce98`.
- Render deploy: `dep-daug4tnf3r2c73fr2h30` / **LIVE**.
- focused surgery queue evidence: 10 tests PASS.
- inherited Cockpit/Clinical Documents/Learning/CU-1/G3/Physio browser integration gates: PASS.
- patient identity remains protected in the surgery queue's server-side table; no browser storage and no implicit mutation of `clinical_patients`.

## Historical Pending Surgery Queue follow-up

Use the live Pending Surgery Queue in the Cockpit. Further additions (for example surgeon, hospital, insurer/GESY authorization, pre-op checklist, priority reason, or archive/history view) are separate product increments, not blockers for v1.


## Pending Surgery Queue v1.1 — delete action

> **STATUS:** MERGED / LIVE.

- PR #126 merged as `3ed0c9a52083eed6151c3e880eac2aad71d0cf4c`.
- Adds protected pending-row deletion with confirmation and soft-delete audit semantics.
- The visible control is now refined to a compact trash icon in the next combined UX release.

## Clinical Calendar manual classification + Surgery Queue icon — 2026-09-30

> **STATUS:** MERGED / AUTO-DEPLOY PENDING.

- PR #127 merged as `fb5e8596334372d7f5e9bc3ae6ccd8dfb4be43ea`.
- `osteoporosis_unspecified` remains the safe automatic result when the reason establishes osteoporosis but duration does not establish first/review.
- Appointment cards expose explicit choices: Πρώτη επίσκεψη / Επανέλεγχος / Prolia / Aclasta.
- Manual classification is stored in a separate protected consumer-side override table and survives future Cal.com snapshots.
- Selecting «Αυτόματο» clears the override and returns to classifier-derived behavior.
- No manual classification writes back to Cal.com or Reception.
- Surgery Queue delete is displayed as a compact trash icon while preserving confirmation + soft-delete behavior.
- Canonical impact, Clinical Calendar snapshot, Surgery Queue and Cockpit Home gates: PASS.

**As-of 2026-09-30 follow-up:** observe the automatic Render deploy and then use the live controls. Verify deploy state afresh before making a current-state claim.
