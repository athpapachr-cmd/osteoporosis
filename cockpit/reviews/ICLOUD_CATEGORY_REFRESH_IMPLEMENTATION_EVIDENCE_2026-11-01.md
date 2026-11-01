# iCloud category-save refresh — candidate evidence

Base main: d545b1fc4834d0017a864ab8a4c432eca394b6c5. Bounded correction to already approved Save/reset/conditional-undo behavior; no backend/schema/auth/provider-write change. Runtime delta: existing triage local notification, full Inbox projection consumer, hidden shell peek scheduler, asset versions.

- Local single-item hint only for non-replayed expected+1/current committed category operations without rules, and their conditional undo. Broadcast metadata is reconstructed from allowlisted fields; remote hints cannot acquire the single-item path.
- Exact qualified selected detail + retained page at the pre-commit revision are required. Old callbacks aborted/generation-invalidated; old source cache discarded. One existing protected item GET supplies authoritative reset/undo/category, which must equal the confirmed revision/epoch/floor. No classifier duplication, optimistic category or new freshness timestamp. Current page/cursors/selected body retained while visibly updating; existing filter applied without refill, undo restores row.
- Rule/remote/replay/unqualified/newer/failed target deltas use existing full fallback. Existing auth purge and late-response guards preserved. Hidden unread cache is revoked immediately and not fetched while full Inbox is active; explicit peek still qualifies/fetches.

## Finite local evidence

- Actual existing fixture API + disposable SQLite/PostgreSQL18.6 + synthetic IMAP + actual Chromium1280/390:12 PASS (original coherence4 + added performance/fallback8; overlapping runs not summed). JUnit `/tmp/cockpit-pr279-release/refresh-complete-browser.xml`.
- Performance oracle: successful category/reset/undo on second page:0 full-list requests,0 peek requests,0 IMAP SEARCH; exactly3 targeted item GETs for that same UID, all FETCH operations bound only to it. Page2, next/newer controls, selected content and body DOM during pending read retained. Filtered row removal/reappearance via undo verified.
- Actual second writer without notification overtakes pending target GET: full fallback reveals its separate change; stale undo409 cannot overwrite it. Lost successful POST response retries same UUID as replay and fully requalifies. Late targeted read after another selection cannot replace new detail. Auth denial during pending targeted read purges rows/detail. Prior delayed-list/peek/focus/unavailable/rule-draft regressions remain in original4 tests.
- Initial fallback-test assertion expected translated HTTP-error copy after an intentionally aborted transport; browser correctly reported native Failed to fetch. Corrected only the test oracle; subsequent4 fallback tests PASS. This was not a runtime/transaction failure.
- Both existing Node UI/shell regressions and all3 JS syntax checks PASS. Existing actual Cockpit Chromium desktop1280/halfwidth680/mobile390/key/popovers suite PASS; artifact output outside repository.
- Backend CAS/idempotency/current-projection/source qualification/SQLite/PostgreSQL tests from PR279 are reused unchanged; no claim of new production write, IMAP mutation or live service access.

One exact-code independent R1 review is required next. No independent PASS, merge/deploy or production performance claim yet.
