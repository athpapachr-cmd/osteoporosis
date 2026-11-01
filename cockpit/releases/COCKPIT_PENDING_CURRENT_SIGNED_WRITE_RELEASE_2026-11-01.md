# Pending current-signed writes — release receipt

The Product Owner explicitly authorized PR #277 merge, one native Render deploy and smoke without patient writes. Technical source release and bounded no-patient-write smoke are complete. This does not claim a real-patient checkpoint experiment or production concurrency stress test.

## Exact identities and preflight

- Source base/main before release: `01f0f4bf07fdcd00cdbb08eafe51bbcb40c2e21d`.
- Approved design: `c975960af6f40a2e143c40fa6aeb9f2dbe3dfd3d`; independent design closure PASS0:0:0.
- Exact implementation/tests independently reviewed R2 PASS0:0:0: `02755139b1f2d307909baf471a58d528a1f06936`. Receipt: `cockpit/reviews/PENDING_CURRENT_SIGNED_WRITE_POSTCODE_R2_PASS_2026-11-01.md`. Later PR commits change documentation only.
- Frozen PR #277 HEAD: `53c6212c10bb6ab0ef4c33d6804ab8a92a9e3bc5`. After marking ready and updating release authority in its body, all13 check runs SUCCESS; none pending. Displayed checks deduplicate repeated canonical-impact runs.
- Targeted final-head CI PostgreSQL17+SQLite:111 PASS/1 not applicable skip,32.88s; existing core56 PASS and capture86 PASS. Local PostgreSQL18.6/SQLite111 PASS plus API21/browser7 PASS were inspected by the independent reviewer; these overlapping executions are not added into a fabricated total.
- PR #277 squash merge: `7956aceb2e0a436b4289bfeb161406748bef99f3`,2026-11-01T07:37:39Z. Its complete tree equals the frozen PR HEAD tree.
- Confirmed existing Render workspace `tea-d5qfh5vpm1nc738pvbv0`, service `srv-d5qfk31r0fns73di596g` (osteoporosis), branch main, autoDeploy on commit. Before release: live `dep-dbje0gbm8hqs73fmjpng`, source `9d3cb34e08bbea15550df71bfdbc63a9aa052858`.
- Native auto-deploy `dep-dbjeohjncjis73dgmvv0`, created2026-11-01T07:37:42Z, LIVE2026-11-01T07:39:41Z (09:39:41 Asia/Nicosia), exact merge source `7956aceb2e0a436b4289bfeb161406748bef99f3`.

Release authority, ready/exact-CI, merge identity, native deploy identity and exact-source LIVE were checkpointed locally before the next deliberate step, outside the frozen PR HEAD. The final evidence-only checkpoint uses `[skip render]` so its publication does not deploy a second version. No manual deploy, settings/secrets/config change, migration, grant or new production table was performed.

## Bounded production smoke

Actual production requests and fresh anonymous Chromium contexts were used; no clinical key/cookie/secret was read, entered or reused. Smoke interval 2026-11-01T07:40:35.109945+00:00 through 2026-11-01T07:40:44.645936+00:00.

- Health:200/status ok.
- Four public served assets:200 and byte-identical to deployed source:
- `static/cockpit/index.html`: SHA-256 `0cd9116450814e583c79488f5a96c8c07b338070fc4da6b1b40a005c6ccdf4bf`.
- `static/cockpit/icloud-inbox.js`: SHA-256 `f4ee907d865d721b7ca70d1dc7bf62c8d218fc9f5f1f681134a965328165a90a`.
- `static/cockpit/doctor-shell.js`: SHA-256 `ff894bbae98ba8b5bb56830c42d16186fed2a652595e577658d8f2ecb007efca`.
- `static/cockpit/patient-chart.js`: SHA-256 `c3e57b96a6939da8321f21b8ab34380aa0c928116225f1c057012688079cda67`.
- Eight protected anonymous GETs returned401: status, registry search, signed-only Pending list, chart, iCloud full list, peek, selected item and attachment.
- Five protected anonymous POST denial probes returned401: checkpoint-plan, checkpoint, metadata, lifecycle restore and visit amendment. Only fake `SYNTHETIC-NO-PHI` identifiers were supplied, without credentials. These requests were rejected by authentication before clinical transactions; no clinical record/event was created or modified.
- Removed mail-action router GET:404.
- Actual anonymous Chromium at1280/390: page200, expected Inbox/navigation functions loaded, zero JavaScript errors, zero browser clinical mutation attempts, zero external browser requests and zero horizontal overflow. Protected patient/mail DOM stayed empty after denied access.
- Existing Home RF summary sent its one read-intent outcomes/search POST per context. That established read operation is distinct from a clinical mutation. Other browser mutation requests remained blocked; none were attempted.
- Render logs2026-11-01T07:37:42Z through07:41:24Z: app ERROR0, HTTP5xx wildcard query0, obsolete `icloud_action_storage_unavailable` warning0. All queries returned hasMore=false. This is a bounded observation window.

Local production smoke output/screenshots: `/tmp/cockpit-p2-evidence.CCXSd7/release/`. These contain no patient/mail data.

## Result and limits

`DEPLOYED_EXACT_SOURCE=YES`; `BOUNDED_NO_PATIENT_WRITE_SMOKE=PASS`.

The separate P2 source correction is now deployed: new progress/restore checks current signed Pending membership after valid replay, ordered Encounter -> Pending locking, SQLite write-intent shared with amendment POST, rollback/409 for specific lock conflicts, and metadata exact replay before mutable category eligibility. Lawful mutable metadata and historical lifecycle rules are retained. No frontend/auth/API/schema/provider owner was introduced.

Authenticated clinical checkpoint/amendment/metadata/lifecycle actions were exercised only on disposable synthetic databases before release, not against production. Real-patient functional acceptance, real mailbox use and production concurrency behavior remain NOT_VERIFIED. No patient write, provider/model call or patient communication was performed by the release coordinator.

Release writer `codex/pending-p2-release-closeout-20261101` released after the documentation checkpoint. No additional tuning/review chain is required absent a new material issue. Next: normal clinician use; report a concrete failure if observed.
