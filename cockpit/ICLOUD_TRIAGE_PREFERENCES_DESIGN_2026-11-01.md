# iCloud email classification — bounded design candidate

**Status:** NARROW STORAGE EXCEPTION APPROVED / INDEPENDENT PRE-CODE R2 CLOSED PASS0:0:0. Exact reviewed contract `8c34467681b78a430cdae6742efb013f669e1aae`; [closure receipt](reviews/ICLOUD_TRIAGE_PREFERENCES_PRECODE_R2_CLOSURE_PASS_2026-11-01.md). Subsequent disposition text changes no material contract. Not implemented, tested, merged or deployed.

**Source:** fresh GitHub main `b8637bcc8af0deebee3716cf7cd98bc99bf2e912`, verified 2026-11-01. Local branch `docs/icloud-triage-design-20261101`. Product Owner authorized investigation and a concrete plan before implementation, then explicitly approved the recommended one-table operational-preference exception and independent pre-code review on 2026-11-01. This is design/review authority; no execution of DDL/grants, runtime implementation or release is authorized yet. No real mailbox, database, patient or credential access was used in this investigation. Screenshots are evidence of missing controls, not fixture data for this public repository.

## Product Owner step card

- **Today:** messages carry suggested categories but the doctor cannot correct/save/undo them or define a personal rule. The banner falsely promises classification, visit linking and review recording.
- **After this proposed step:** choose a message category, Save once, see its correct group, reload and retain that decision; undo it. Separately choose whether to save an explicit rule for future similar messages. See why a category was chosen.
- **Example, synthetic:** a WeTransfer notification titled “Αξονικί δοκιμή.rar” can be moved from Other to Clinical. A separately approved rule for that sender plus the term can classify future similar notifications. It does not mark the scan received, reviewed or attached to a patient.
- **Outside:** patient matching/linking, clinical result acceptance, Pending/checkpoint changes, visit changes, communication and mailbox flags/folders.
- **Why first:** reliable sorting and correction are a prerequisite for a usable incoming-mail queue. Existing chart navigation does not supply them.
- **Next milestone:** separate implementation decision after the closed independent design gate; implementation, provisioning and release have not begun.

## Source-proven findings and reuse decision

1. `main.py:69–72` mounts `build_icloud_mail_router()` without a database engine and does not mount the old action router. `clinical_icloud_mail.py:555–563` therefore returns no stored overrides. No durable correction can currently reach the viewer.
2. `static/cockpit/icloud-inbox.js` renders suggestions and existing-patient/Pending navigation; it has no classification/rule editor. `static/cockpit/index.html:281` still promises unsupported actions.
3. `clinical_icloud_triage_rules.py:14–21` lacks the Greek CT term. Existing `fold()` already normalizes Greek case/accents. Earlier local synthetic probes of the unchanged classifier returned Other for both “Αξονική” and “Αξονικί” in generated transfer notifications; this proves those examples only, not general classification quality.
4. `clinical_icloud_actions.py:52–84, 399–405` defines a two-table mail decision/event mechanism, with patient/booking/Pending references, review/contact state and startup table creation. It is disconnected; re-enabling it would reintroduce unapproved storage and unrelated clinical semantics. Historical SQLSTATE 42501 is prior evidence, not a new live schema inspection or proof of today's installed tables.
5. Repository ORM declarations expose no general Cockpit settings store. The closest working manual-classification mechanism is `ClinicalAppointmentClassificationORM` (`clinical_calendar.py:46–51`), whose key and values belong to appointments. Its UX concept is reusable; inserting email IDs/rules there would corrupt ownership. `RFDoctorProfileORM` is a typed RF identity profile, not arbitrary preferences (`clinic_utilities/rf/persistence.py:59–74`). Patient/Encounter JSON and Pending/checkpoint history are clinical owners, unsuitable for administrative or unmatched mail.

**Decision:** REUSE the existing Clinical Key, source reader, MIME/UID qualification, paging, viewer, category vocabulary and text normalization; EXTEND their category projection and UI. Do not rebind clinical or calendar storage. The proposed distinct adjacent role is operational email-sorting preferences only. It cannot create a clinical task or clinical fact.

## Storage decision — narrow exception approved, execution not authorized

The earlier blanket no-new-table boundary was escalated as required by the Product Constitution. On 2026-11-01 the Product Owner explicitly approved ONE small operational-preferences table in the existing database and independent review before implementation. The constitution and affected canonical scope now record this narrow exception. Every other no-new-auth/database/clinical-store/provider-write restriction remains binding; this approval does not execute provisioning or grant runtime/release authority.

**Approved design exception:** one small operational-preferences table in the existing database, with one bounded document per configured mailbox scope. No new database/service, second connector, users/roles, clinical ledger or activation of the old two-table router. The design includes reviewed provisioning and minimum application DML privileges for this table only; actual provisioning/grant execution remains a separate authorized release step. Never grant application schema-CREATE privilege. No startup `create_all`, implicit migration or provision-on-read. Existing clinical tables remain untouched.

Candidate table `icloud_triage_preferences`:

- `mailbox_key`: primary key, opaque hash of canonical configured account + alias + folder, server-derived; never accepted as client authority.
- `revision`: monotonically increasing integer, initially zero in a virtual empty projection; first explicit Save inserts revision one.
- `state_json`: versioned validated document containing only manual category overrides, explicit rules and a bounded recent-operation receipt list.
- `updated_at`: UTC timestamp. Existing single-clinician authority; no new principal registry or copied authentication credential.

Single-row optimistic concurrency makes a per-message change plus optional rule creation atomic without a separate event table. Start with finite limits: 5,000 overrides, 100 rules, 64 operation receipts, 1 MiB encoded state. At capacity reject the new operation with a clear message; never silently discard a doctor's choice. No record-wide reset/delete that restarts its revision. Any later retention/cleanup must be explicit and preserve concurrency identity.

Stored message override: source identity hash, one of `clinical`, `administrative`, `other`, `review`, and update time. Source identity includes account/alias/folder/UIDVALIDITY/UID; use structured canonical encoding before hashing. Do not persist message subject/body, attachments, patient references or clinical review/contact state. A hash is an opaque identifier, not a claim of anonymization. Rule criteria may themselves identify a sender and must be treated as private protected data.

**Alternatives considered:**

- Browser-only storage can survive a refresh in that browser/profile, but is lost on clearing it and does not follow the doctor between browsers/devices. It must be explicitly labelled local-only and is not recommended as the durable clinic-wide behavior promised here. It is a different scope requiring an explicit choice, not a hidden fallback.
- Memory or Render ephemeral files do not establish persistence across restarts/deploys. A persistent disk adds another storage service/operational requirement and is not a smaller justified bypass of the storage decision.
- The existing private environment hints support only a limited administrative heuristic and require operator configuration; they cannot implement per-message edits, undo or an in-app rule manager.
- Patient, visit, RF, calendar or checkpoint rows must not be fabricated or overloaded to avoid naming the missing storage capability.

**Scope decision:** the shared server-backed preference option was selected. Do not substitute browser-only or ephemeral storage as an equivalent implementation.

## User-visible contract

1. The selected message exposes category choices **Κλινικό / Διοικητικό / Άλλο / Αχαρακτήριστο**, then one explicit **Αποθήκευση**. Reading/choosing a radio option makes no write. Saved category, origin and a clear success state appear only after confirmed server commit. On a category move, refresh the bounded source page; never leave an item selected under an old filter. Keep All permanently accessible.
2. **Αχαρακτήριστο** is a persistent manual `review` choice. It is different from **Επαναφορά αυτόματης πρότασης**, which removes the manual override and re-evaluates rules/defaults. Explain the distinction in the editor.
3. Show **Αναίρεση τελευταίας αλλαγής** after Save. It is an explicit conditional operation restoring only the affected previous override/rule state; no blind reversal if another tab changed the preferences in the meantime. Reopening the editor always permits another explicit correction.
4. **Κανόνας για επόμενα παρόμοια** is opt-in, off by default and displays the exact proposed criteria before saving. No automatic training on individual corrections, model calls or hidden rule creation. A separate rule manager lists, edits and disables saved rules. A message override always wins.
5. Remove banner claims about patient/visit association and clinical review. Display the actual feature available in this release. Sorting is not clinical identity, receipt, assessment or completion.

## Deterministic rules and precedence

- Effective category: explicit message override > applicable enabled personal rule > built-in header suggestion. Show which layer produced it. If applicable personal rules disagree, use `review` and show the conflict; do not choose an arbitrary winner or silently move it to Administrative/Other.
- Rule fields: stable ID, revision, enabled flag, exact sender address or properly bounded sender domain, optional subject terms, target category, creation time and source-scope activation watermark. Accept no arbitrary executable expressions/regex. Limits and validation apply before persistence. Subject terms use the existing Unicode/case/accent folding and token boundaries; retain an explicit finite variant for “αξονικί”, not broad fuzzy patient/name matching.
- For mixed-content services such as transfer platforms, require a subject criterion as well as sender. Sender-only private rules may be explicitly chosen for genuinely uniform sources, with a clear statement of their reach. Sender names/addresses remain untrusted routing hints, never authenticated clinical provider or patient evidence.
- A future-only rule starts after the server-observed current INBOX UID high-water mark under its exact UIDVALIDITY, obtained read-only. Preview is labelled with bounded coverage and does not claim a full mailbox count. If the source watermark cannot be qualified, rule Save fails visibly; the doctor can still choose to save the individual correction alone as a separate operation. No automatic partial success for a combined Save.
- A UIDVALIDITY change prevents applying old per-message overrides to reused UIDs. Future-only rules whose watermark is no longer comparable become visibly inactive pending explicit reactivation against the new source epoch. Account/alias/folder changes never inherit another scope's choices silently.
- Editing a rule re-displays its scope and future-only activation point. It affects no previously saved manual override. Disabling an applicable rule re-evaluates the currently viewed messages against the remaining rules/defaults; show that preview before confirmation. No destructive rewrite of old overrides.

## Protected writes, replay and failure behavior

- Existing Clinical Key/session and same-origin write checks only; reuse the established auth and origin-validation patterns. No extra login, role or secret. All preferences/responses private/no-store; never log criteria, sender, subject, body, SQL parameters or identifiers.
- Proposed narrow API under the current protected iCloud boundary: GET current preferences/revision; POST explicit validated triage operation. Keep the legacy clinical action router unmounted. Item changes requalify the exact alias-scoped source UID using existing read-only IMAP. Rule edits validate the server-derived mailbox scope and the rule's own identity.
- Every mutation carries expected document revision plus UUID operation ID. Atomic insert/CAS update advances the revision even for an accepted no-op and records a bounded operation ID + canonical payload hash + resulting revision. Exact retained replay returns acknowledgment and the **current** projection without repeating the change; same ID/different payload conflicts. An old evicted operation cannot overwrite newer state: its old expected revision conflicts. Do not automatically retry a conflict as a new user choice.
- Two tabs racing on the same document: one accepted operation, the other clear conflict with current state; preserve the uncommitted editor values for deliberate reconciliation. Undo uses the revision produced by the operation being undone and a typed inverse stored in its bounded receipt, never a full-document rollback over later writes. A new edit or external revision invalidates that one-click undo.
- Validate record/version/size limits before commit; unexpected storage errors roll back the entire combined operation. Do not show saved success from an optimistic browser-only change. No patient record or mailbox mutation is part of this transaction.
- On preference read/store failure or missing provisioned table, retain the independent read-only mailbox view. Saved decisions are **unknown**, not absent: stop category-based hiding, expose All with a clear degraded-state notice, disable unsafe classification/rule controls and do not substitute empty preferences as verified state. Preserve existing source-coverage limits. On 401/403 purge protected message/preferences DOM and cache under the existing auth-loss behavior.

## Category projection coherence — P2-01 bounded correction

The independent original review of `65b665d3eccffc31bd48826a2953fd4d30815930` found P2-01: authentication-only cache reuse can preserve a filtered empty peek/badge after a saved category move. Original receipt: `reviews/ICLOUD_TRIAGE_PREFERENCES_PRECODE_R2_BLOCK_2026-11-01.md`. The following contract closes the specified design gap; it changes no store, actor, clinical or provider authority.

- Every list/peek/detail category projection carries its server-derived mailbox scope, UIDVALIDITY and the exact preference document revision used to classify/filter it. An unavailable preference read is explicitly unqualified, never revision zero. Scope/revision metadata is protected/no-store and is not accepted from a client as write authority.
- Confirmed Save/reset/undo/rule create/edit/disable returns the current committed scope/revision. It invalidates **both** the full inbox cached list/selection and the shell unread peek/badge, resets the full inbox page chain to its first page, aborts pending projection reads and advances their request generations. Unread unknown/pending is not displayed as a verified zero. The next displayed projection must match the scope/epoch and use the confirmed revision or newer; no old-generation callback may render or populate either cache, even if the response succeeds after abort.
- Before reusing either cached projection on reopen, tab/window return, or a previously suspended refresh, perform the existing auth qualification plus a protected read-only qualification of current mailbox scope/epoch/preference revision. A minimal metadata projection of the same preferences GET suffices; this is not an IMAP rescan or a second preference owner. Cache reuse requires equality with the freshly qualified scope/epoch/revision as well as existing source freshness constraints. If different, invalidate and refetch the category projection. This check is mandatory even when no cross-tab notification arrived.
- A same-origin in-memory notification (for example BroadcastChannel) can promptly propagate a confirmed scope/revision change to other open tabs and revoke their category projections; no message headers, criteria, body, credentials or persistent browser storage are part of that notification. It is an optimization, not authority or a replacement for server qualification. Subsequent projection requests retain the highest revision confirmed for that scope and discard older results. Auth loss still purges all protected DOM/data.
- A source epoch/scope mismatch invalidates the old cached projection. Requalification or storage failure uses the existing unknown/degraded All contract; do not present old filtered lists or an old empty peek/badge as current. The inbox's transient-source-error cache fallback is subject to the same revision/generation checks and explicit stale-source label; it cannot resurrect categories whose revision is now unqualified.

## Exact implementation seams after separate authority

- `clinical_icloud_triage_rules.py`: reuse `fold()`; add finite Greek CT coverage and a pure bounded precedence/rule evaluator. No LLM/provider calls.
- New narrowly scoped preferences module: one operational model/validation/transaction/API seam; no import of patient/Pending/Encounter writers. Existing generic shared settings infrastructure, if discovered later, must trigger a reuse reassessment rather than a second store.
- `clinical_icloud_mail.py`: apply one qualified preferences snapshot to candidate headers **before** category filtering, including list and peek, and to the selected item's category. Preserve UID binding, alias filter, bounded scans/cursors and read-only commands. Report preference unavailability explicitly rather than presenting incorrect filtered results.
- `main.py`: bind only the reviewed triage read/write adapter to the existing engine after explicit provisioning; leave the old action router disabled. No schema creation in startup/GET paths.
- `static/cockpit/icloud-inbox.js`, `static/cockpit/doctor-shell.js`, `static/cockpit/index.html` and affected styles: native editor, rule manager, undo, meaningful origin labels, stale-selection protection, truthful banner and shared revision-qualified projection invalidation/cache reuse. Preserve the existing patient/Pending navigation without adding a saved clinical association.

## Finite acceptance — required before claiming ready

Use disposable SQLite/PostgreSQL and actual synthetic Chromium; fake source messages only, never screenshot identities or live patient records.

1. Each category can be saved from every other category, including Clinical <-> Administrative; saved choice survives page reload and a second browser session against the same test server.
2. Explicit `review`, reset-to-automatic and undo produce different correct outcomes; undo never reverses an intervening tab's save.
3. “Αξονική”, “αξονική”, “ΑΞΟΝΙΚΗ” and finite “αξονικί” examples behave as proposed; unrelated administrative, newsletter and ambiguous examples remain recoverable. Prove suggestions remain suggestions, including spoofed clinical-looking senders.
4. Optional rule is absent by default; explicit create/edit/disable works. A mixed WeTransfer sender cannot by itself classify all its mail as clinical. Rule conflict produces visible `review`; message override wins.
5. Future-only UID watermark excludes older messages; account/alias/folder separation and changed UIDVALIDITY prevent cross-source reuse. Partial preview remains explicitly partial.
6. Combined correction + rule is atomic. Real separate-connection SQLite/PostgreSQL races, exact retry, mismatched retry, evicted-receipt stale retry and conditional undo preserve current decisions and revision.
7. Source qualification failure, missing store, unavailable store and commit failure show no false Save and no silent loss of existing manual classifications. All remains readable with truthful degraded category state.
8. Category filtering runs after preferences for list/peek/detail, at exact page bounds. Moving a selected message clears stale candidate/actions and resets the page chain; an old response cannot affect another selected message. One real synthetic browser regression primes BOTH full-inbox and shell unread caches (including an empty clinical peek), saves a category/rule change, then reopens both within TTL and delivers a delayed successful old response. Assert the saved revision governs list/peek/badge, old-generation responses cannot refill caches, and a second tab qualifies the current revision on return even if notifications were missed. Exercise unavailable revision/store qualification as unknown/degraded, never a cached verified empty clinical queue.
9. Unauthenticated/cross-origin writes rejected; auth loss clears protected DOM; GET/source reading performs zero writes. No body/attachments/clinical identity persisted or logged. Rule inputs rendered as text, not HTML.
10. Existing IMAP EXAMINE/BODY.PEEK behavior, original Pending navigation and existing clinical owner regressions remain passing. Instrumented synthetic browser/server proves no IMAP mutation, patient/visit/Pending/checkpoint mutation or outbound patient contact.
11. Desktop/mobile doctor completes category change -> reload -> undo -> opt-in rule -> rule disable through the visible controls. Test absence of the old unsupported clinical-action promises. Technical CI alone is not clinician usability acceptance.

Review tier **R2**: new authoritative preference writes, scope identity and possible hiding of clinical-looking incoming items, despite no clinical-record writes. One pre-code review of the approved product storage contract and one exact-code fidelity review after authorized implementation. The original independent review returned BLOCK0:0:1 for cache coherence; ONE independent delta plus affected cumulative closure of exact `8c34467681b78a430cdae6742efb013f669e1aae` returned PASS0:0:0, all material findings closed and complete declared scope. The design review chain is closed; see the closure receipt. No tests of the proposed behavior have run because it is not implemented; the earlier unchanged-classifier probes are limited diagnostic evidence.

## Rollback and handback

Before any later release: verify provisioning separately, exact reviewed HEAD/CI, native automatic deployment and a no-patient-write smoke; production rule/category writes require an explicitly approved synthetic source fixture or the doctor's deliberate operation. No automatic production trial on real messages.

Rollback disables the triage write adapter and returns to a clearly labelled read-only All view, retaining the preference data for recovery. Do not automatically drop the table, change provider state or remove patient data.

**Handback:** approved narrow one-table operational-preference contract and independent pre-code R2 are CLOSED PASS at the exact reviewed contract above. Docs writer released after receipt and one bounded consistency audit. Await separate implementation authority. No runtime implementation, proposed-behavior test execution, public PR, merge, deploy, provisioning/grant execution or live-data changes have occurred. No further design review is required without a genuinely new material delta.
