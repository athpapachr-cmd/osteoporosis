# Bounded category-save refresh correction

User report: classification works, but every Save rereads the mailbox page. Source proof: `confirmed()` invalidates both consumers; full Inbox handler clears/page-resets/loads; shell eagerly fetches hidden peek. No production reproduction/write is needed. Base main d545b1fc4834d0017a864ab8a4c432eca394b6c5.

This is a bounded correction within already approved category/reset/conditional-undo behavior. Reuse current frontend owners, protected item GET and proven server CAS; no new write/store/auth/backend semantics. R1 deterministic read-only projection, one independent exact-code post-review. Original R2 authoritative-write/identity contract and tests remain. No merge/deploy or patient/provider changes.

## Eligible one-item refresh

- Only the originating tab may supply a volatile hint for a successful category operation with no new rule, or its local conditional undo. Hint contains UID-qualified item ID and expected revision; it is never broadcast/persisted or used as server write authority.
- Require a fresh non-replayed commit at expected+1 whose current projection equals that committed revision. Current selected detail and every retained visible page row must be qualified at exactly that expected revision and same mailbox scope/UIDVALIDITY. No hint, mismatch or uncertain/replayed/newer result uses existing full invalidation.
- Immediately advance inbox generations, abort old requests, invalidate source caches and unread projection. Retain the current source page/cursor and selected body while the one affected category is visibly updating; it is not rendered as verified old category. No source checked-at timestamp is renewed by a preference Save.
- Read only the selected UID through the existing protected detail GET. Publish the new category only if its server projection exactly matches the committed hint and current revision floor. Unchanged page rows can then advance through the proven one-override delta. Reset/undo category is always read from the server; never duplicate the classifier or infer unread status.
- Apply existing view filter without backfilling/rescanning; selected email stays open even if it moves out of the chosen filter, and undo can restore its row. Preserve current page/cursor chain; label counts only visible retained rows. Cached source projections are discarded, not given a new freshness age.
- Another revision/epoch, remote notification, rule change, pending projection, missing qualified page or failed/invalid targeted read falls back to normal full qualification/invalidation. Auth loss always purges; delayed old list/detail/peek callbacks cannot repopulate state.
- Shell revokes old unread cache/badge/pending request immediately, but does not scan hidden peek while full Inbox is open. Explicitly opening peek still qualifies and fetches; Home/background refresh remains bounded. Unknown unread is not presented as confirmed zero.

## Finite evidence/review questions

1. Actual browser Save/reset/undo changes only one protected item GET, zero full list/peek requests or IMAP SEARCH while full Inbox stays open; unchanged page/body/selection retained.
2. Current filters and a later source page/cursor remain correct, including row removal/reappearance and no source freshness renewal.
3. Delayed old list/detail/peek results, focus/reopen, other-tab writes, replay/newer revision and source epoch mismatch never resurrect old categories; broad/unknown deltas use full fallback.
4. Pending Save controls/draft safeguards, failed targeted read and auth purge retain prior safe behavior; no clinical/provider writes.

Use disposable SQLite/PostgreSQL and real synthetic browser fixtures. Reuse backend CAS/idempotency/qualification evidence because those bytes are unchanged. Stop at tested/reviewed candidate, without merge/deploy.
