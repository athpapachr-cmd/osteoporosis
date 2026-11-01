# iCloud to existing signed Pending — release receipt

Product Owner explicitly authorized PR #275 merge, deploy and a smoke check without patient writes. This is the technical release of clinician-selected navigation, not automatic result acceptance or full clinical follow-through.

## Source and release identities

- Base/main before release: `6a91fba4572c245f3d489ff08db11f4864827de9`.
- Reviewed substantive code/tests: `59ac02ad78100712081260dec55399a34b247282`.
- Frozen PR #275 HEAD: `e541ba3dda2deac7d74bb768524db8861ad86382`; 21/21 check runs completed SUCCESS immediately before merge. This includes two successful canonical-impact runs; the PR checks display deduplicates their name. The substantive head had 26/26 SUCCESS.
- Independent bounded R1 navigation PASS, new-change P0:P1:P2=0:0:0: https://github.com/athpapachr-cmd/osteoporosis/pull/275#issuecomment-6554573381.
- PR #275 squash merge: `9d3cb34e08bbea15550df71bfdbc63a9aa052858`, at 2026-11-01T06:46:22Z. Its complete tree equals the frozen reviewed PR HEAD tree.
- Confirmed existing Render workspace `tea-d5qfh5vpm1nc738pvbv0`, service `srv-d5qfk31r0fns73di596g` (osteoporosis), main auto-deploy on commit.
- Native auto-deploy `dep-dbje0gbm8hqs73fmjpng`, created 2026-11-01T06:46:25Z, LIVE at 2026-11-01T06:48:29Z (08:48:29 Asia/Nicosia), exact merge source `9d3cb34e08bbea15550df71bfdbc63a9aa052858`.
- No manual duplicate deploy; existing configuration, secrets and database permissions/schema were not changed by the release coordinator. The optional mail-action router is no longer mounted.

## Checkpoint continuity

Release authority/exact-head CI, exact merge, native deploy identity and exact-source LIVE were each checkpointed in the local release branch before the next deliberate release step. These checkpoint commits were kept outside the frozen PR head and rebased onto its identical squash-merge tree. This final docs-only publication carries the completed receipt and current state; `[skip render]` prevents a second deployment for documentation alone. No new runtime or design review is required for this evidence-only publication.

## Bounded production smoke

Actual public production requests and a fresh anonymous Chromium context were used; no clinical credentials were read, entered or reused. Smoke interval: 2026-11-01T06:49:51.384587+00:00 to 2026-11-01T06:49:59.298877+00:00.

- `GET /health`: 200 / status ok.
- Four served assets returned 200 and were byte-identical to the deployed merge source:
- `static/cockpit/index.html`: SHA-256 `0cd9116450814e583c79488f5a96c8c07b338070fc4da6b1b40a005c6ccdf4bf`.
- `static/cockpit/icloud-inbox.js`: SHA-256 `f4ee907d865d721b7ca70d1dc7bf62c8d218fc9f5f1f681134a965328165a90a`.
- `static/cockpit/doctor-shell.js`: SHA-256 `ff894bbae98ba8b5bb56830c42d16186fed2a652595e577658d8f2ecb007efca`.
- `static/cockpit/patient-chart.js`: SHA-256 `c3e57b96a6939da8321f21b8ab34380aa0c928116225f1c057012688079cda67`.
- Anonymous status, registry search, signed-only Pending, chart, iCloud full list, peek, selected-item and attachment reads: 8/8 returned 401. The fake identifier `SYNTHETIC-NO-PHI` was used; no patient record or real mailbox item was read.
- Unmounted `GET /clinical/icloud-actions/items/765-4`: 404.
- Actual anonymous Chromium at 1280 and 390 pixels: page 200, new Inbox/navigation functions loaded, protected mail/patient DOM empty after access denial, JavaScript errors 0, horizontal overflow 0, external requests 0, clinical mutation requests 0.
- The existing Home RF summary issues one read-intent `POST /clinical/clinic-utilities/rf/api/outcomes/search` per browser context. Its handler was source-checked as a read operation; it is not a patient write. The first smoke harness conservatively blocked that request, then was corrected to allow only this existing read-intent POST and rerun successfully. All other mutating requests remained blocked; no patient mutation was attempted or sent.
- Render queries covering 2026-11-01T06:46:25Z through 06:50:10Z: app ERROR 0, HTTP 500/502/503/504 0, `icloud_action_storage_unavailable` 0; all queries returned hasMore=false. These are bounded log-query observations, not a guarantee of future behavior.

## Result, separate P2 and limits

`DEPLOYED_EXACT_SOURCE=YES`; `BOUNDED_NO_PATIENT_WRITE_SMOKE=PASS`.

The independent PASS covers explicit clinician-selected navigation only. The existing checkpoint writer rechecks signed-source status but not current signed Pending membership at action time. This pre-existing P2 remains separate, unresolved and outside this release. No end-to-end race-proof clinical workflow claim is made.

Authenticated real-mail reading, real-patient navigation/checkpoint actions, clinical result acceptance, clinician acceptance and real-use/pilot validation were not exercised by this smoke and remain NOT_VERIFIED. No new auth/table/migration/grant, provider flag mutation, patient communication, Save/amend/checkpoint/lifecycle/metadata write, or model call was performed by the release coordinator.

Release writer `codex/pr275-release-closeout-20261101` is released after the documentation checkpoint. Next: normal clinician use of the existing read-only Inbox/navigation with its original clinical action owner; any correction to the separate P2 requires its own bounded scope.
