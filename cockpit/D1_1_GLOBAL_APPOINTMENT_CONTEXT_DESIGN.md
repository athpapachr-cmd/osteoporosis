# COCKPIT D1.1 — Global Appointment Context correction

> **STATUS:** PRE-CODE R2 DESIGN CANDIDATE.
> **DATE:** 2026-10-02.
> **PRODUCT OWNER:** clinician.
> **BASE:** `df55c9a5b6e06e8ed04ff85eb5cedf4b68e94ed3`.
> **BRANCH:** `design/cockpit-d1-1-global-context-2026-10-02`.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED pending one independent pre-code review.

## 1. Product problem

The released D1 Today Context Strip renders Previous / Current / Next from the protected Clinical Calendar, but the consumer persists and returns only osteoporosis-related rows. The Digital Secretary producer already sends the complete bounded Cal.com snapshot, including non-osteoporosis appointments. Those rows are currently classified `other` and discarded by the Clinical Calendar consumer.

Observed consequence: Reception can show a real clinic appointment while Cockpit shows no Previous / Current / Next.

The global Cockpit must reflect the doctor's global appointment context without turning the weekly Osteoporosis Calendar into a general booking calendar.

## 2. Product Owner intent

Home remains compact.

Default D1.1 surface:
- Previous;
- Current;
- Next;
- concise time/date, patient display name, clinic/reason where available;
- weekly Osteoporosis Calendar remains a separate Module-01 view.

Click/tap patient is reserved for the immediately following **Visit Brief / Patient Context** slice. Visit Brief is not implemented in D1.1.

The next appointment may be on a later day. A Friday-evening Cockpit may therefore show Saturday/Monday as Next rather than “no next appointment today”.

## 3. Existing-mechanism reuse decision

### Producer — REUSE unchanged

Reuse the existing Digital Secretary Clinical Calendar feed:

```text
Cal.com / Digital Secretary schedule truth
→ existing cal_setmore_sync.py bounded future snapshot
→ existing clinical_calendar_feed.py normalized appointment
→ existing X-Clinical-Ingest-Key transport
→ Clinical Excellence protected appointment store
```

No new Cal.com reader.
No new Reception endpoint.
No new Reception credential.
No Reception runtime mutation.
No second booking/calendar owner.

Current producer already carries the fields needed for D1.1 and a bounded ~30-day future horizon.

### Consumer — EXTEND existing normalized store

`clinical_appointments` becomes the protected normalized source store for all valid appointments received in the existing complete snapshot, including `category=other`.

The weekly Osteoporosis endpoint remains a filtered **view** over that store.

This is preferable to a second global appointment table because it keeps one normalized appointment copy per source appointment and one snapshot-reconciliation owner.

## 4. Why R2 remains required

No new schema or external write is introduced, but D1.1 changes protected persistence scope:

- non-osteoporosis appointment identity/context that was previously discarded will now be retained;
- the browser receives a new global appointment-context projection;
- data-minimization/privacy behavior changes.

Therefore D1.1 remains **R2** under `PROCEDURES.md`: one independent pre-code design review and one post-code exact-head fidelity review.

## 5. Persistence and minimization contract

For every valid snapshot row:

1. classify using the existing classifier/manual override;
2. upsert one `ClinicalAppointmentORM` row regardless of whether category is osteoporosis-related or `other`;
3. preserve snapshot reconciliation: same-source rows absent from the complete declared snapshot window are deleted;
4. preserve manual osteoporosis classification overrides;
5. no new table / migration.

For rows whose effective category remains `other`:

- retain:
  - source + source appointment ID;
  - start/end/duration;
  - clinic;
  - patient display name;
  - source-proven human visit reason/comment;
  - status;
  - updated timestamp;
  - category=`other`;
- **do not retain `phone_e164` or `linked_patient_id`** for D1.1. Persist them as empty / null for `other` rows.
- if a later source snapshot or manual override makes the row osteoporosis-relevant, the next upsert may retain the already-authorized osteoporosis fields as today.

D1.1 does not activate phone correlation, communication linkage or clinical patient identity linkage.

## 6. Projection separation

### Existing Module-01 endpoint — unchanged behavior

`GET /clinical/calendar/appointments?start=...&end=...`

- remains protected by `CLINICAL_DATA_KEY`;
- retains the existing `RELEVANT_CATEGORIES` filter;
- therefore the weekly Osteoporosis Calendar remains osteoporosis-only;
- existing manual classification behavior remains.

### New Cockpit context endpoint

Add a protected endpoint:

`GET /clinical/calendar/cockpit-context`

It returns a minimal context object, not a month-long appointment list:

```json
{
  "generated_at": "...",
  "source_updated_at": "...",
  "today_total": 0,
  "previous": null,
  "current": null,
  "current_conflict_count": 0,
  "next": {
    "appointment_id": "...",
    "start_at": "...",
    "end_at": "...",
    "clinic": "evrychou",
    "category": "other",
    "patient_display_name": "Synthetic Patient",
    "reason": "..."
  }
}
```

Allowed browser fields:
- appointment_id;
- start_at / end_at;
- clinic;
- category;
- patient_display_name;
- sanitized human reason;
- status if needed;
- generated/source-updated timestamps;
- today_total;
- conflict count.

Forbidden browser fields:
- phone_e164;
- linked_patient_id;
- raw transport metadata;
- booking/provider secrets;
- source payload.

The endpoint uses the same `CLINICAL_DATA_KEY` protection already used by the Cockpit browser.

## 7. Context semantics

Use Asia/Nicosia local-day semantics.

### Previous

Latest clinician-attention appointment that started today before the current context.

If there is no Current and no overlap conflict:
- Previous = latest-started appointment today whose start is <= now.

No claim is made about previous appointments before the retained source snapshot window.

### Current

One active appointment where:

`start_at <= now < end_at`

If multiple active rows exist:
- preserve the D1 lawful Aclasta exception only when all earlier active rows are explicitly category `aclasta` and the latest-started active row is non-Aclasta;
- otherwise fail closed visibly with `current=null` + `current_conflict_count > 1`.

No category inference is introduced for global `other` rows.

### Next

Earliest stored appointment with `start_at > now`, across the existing bounded future snapshot horizon — **not restricted to today**.

When Next is not today, the UI displays date + time.

## 8. Freshness boundary

D1.1 reuses the already released Digital Secretary → Clinical Calendar snapshot cadence. It does **not** change sync frequency or claim live parity with the Reception dashboard for appointments created after the latest snapshot.

The endpoint exposes bounded `source_updated_at` so the UI can fail visibly if the projection is stale/unavailable. The normal clinician-facing surface should use plain language, not technical sync provenance.

Changing cadence, adding live Reception pull, or adding a manual sync action is a separate bounded operational slice.

## 9. Home UI correction

The top card remains visually compact.

- Previous / Current / Next remain the primary strip.
- Remove the osteoporosis-only numeric counters from the **global** Today card; they falsely imply the card itself is osteoporosis-scoped.
- Keep **«Άνοιγμα εβδομαδιαίου ημερολογίου»** as the explicit Module-01 deeper view.
- Keep the Reception dashboard link separately.
- Patient card becomes click/tap-ready for the future floating Visit Brief but D1.1 does not open clinical history yet.
- If context cannot load: show a visible unavailable state; do not fall back silently to the osteoporosis-only endpoint.

## 10. Legacy-oracle disposition

The following legacy expectations are **superseded only for storage**, not for the weekly osteoporosis projection:

- `test_snapshot_upserts_relevant_rows_filters_unrelated_and_removes_missing`:
  - old storage oracle “unrelated rows are absent from DB” becomes stale for D1.1;
  - preserved oracle: weekly osteoporosis endpoint must still exclude `other`.
- `test_snapshot_reclassification_removes_previously_relevant_row`:
  - old deletion oracle becomes stale;
  - new expectation: row remains in normalized global store as `other`, is excluded from weekly osteoporosis endpoint, and global-sensitive fields are minimized.

Still-current oracles:
- incomplete/ambiguous snapshots fail before destructive reconciliation;
- stale missing rows are removed inside exact source/window;
- manual classification survives future snapshots;
- weekly osteoporosis classification semantics remain;
- D1 Aclasta lawful overlap / other overlap fail-closed semantics remain.

## 11. Implementation seams after pre-code PASS

Expected changed files only:

- `clinical_calendar.py`;
- `test_clinical_calendar_snapshot.py`;
- `static/cockpit/app.js`;
- `static/cockpit/index.html` and/or `styles.css` only as required for date/global labels;
- `test_cockpit_home.py`;
- `test_cockpit_surgery_queue_ui.py` only if a current shared UI assertion requires alignment;
- `cockpit/CURRENT.md`;
- root `CURRENT_OPERATIONAL.md` checkpoint as required.

No Reception repository mutation.
No Render secret/config change.
No database schema migration.
No Visit Brief runtime in this slice.

## 12. Focused acceptance evidence

Pre-code review must validate:
- one normalized storage owner / no duplicate calendar path;
- privacy minimization for `other` rows;
- snapshot cancellation/reschedule reconciliation remains sound;
- weekly osteoporosis view is unchanged;
- global context cannot expose phone/link;
- cross-day Next semantics;
- overlap fail-closed behavior.

Post-code focused tests must prove at minimum:

1. snapshot with osteoporosis + unrelated appointment stores both;
2. weekly `/appointments` returns only osteoporosis row;
3. `/cockpit-context` can return unrelated appointment as Previous/Current/Next;
4. global response contains no phone/link/raw metadata;
5. `other` stored row has phone empty + linked patient null;
6. reclassification relevant → other retains row but removes it from weekly view and strips global-sensitive fields;
7. missing source row in later complete snapshot is removed;
8. incomplete/invalid snapshot does not destructively reconcile;
9. Next may be tomorrow/later day;
10. Aclasta overlap remains lawful; other overlap fails closed;
11. Cockpit no longer calls the osteoporosis-only list endpoint for D1 context;
12. weekly Calendar link remains;
13. no browser localStorage/sessionStorage of appointment identity.

## 13. Rollback

Rollback is code-only:
- restore prior “store relevant only” import behavior;
- restore D1 consumer to prior osteoporosis endpoint if required.

No schema rollback is required.

Rows persisted as `other` during a candidate period are protected server-side data; rollback/release procedure must explicitly decide whether to delete candidate-only `other` rows rather than silently leave them outside the old retention contract.

## 14. REPLAN triggers

REPLAN before implementation if review/source proves any of:

- producer snapshot is not complete enough to support global context;
- retaining `other` in the existing table breaks snapshot reconciliation or manual overrides;
- D1.1 requires Reception runtime mutation;
- D1.1 requires phone/link retention;
- global Visit Brief identity is being smuggled into this slice;
- a second appointment owner/store becomes necessary.

## 15. Explicitly deferred

- floating Visit Brief runtime;
- protected patient-record linkage;
- GESY Browser Bridge;
- D2 communication correlation;
- phone-based communication association;
- live Reception/dashboard pull;
- sync cadence change/manual sync;
- appointment booking/cancel/reschedule/write.
