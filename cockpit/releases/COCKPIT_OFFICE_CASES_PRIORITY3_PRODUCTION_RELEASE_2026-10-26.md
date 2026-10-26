# Cockpit Priority 3 Office Cases production release

**Date:** 2026-10-26 Asia/Nicosia. **State:** READY_FOR_PRODUCT_OWNER_SMOKE; technical no-write production smoke PASS. Clinician use with real records remains unverified.

## Scope and authority

The Product Owner accepted Priority 2, requested the next roadmap priority and specified one record per individual accident payment. The existing Cockpit autonomous release mission covers PR, merge, native Render deployment and bounded smoke after required gates. This release adds one **Περιστατικά ιατρείου** screen with Χειρουργεία, Τροχαία and RF tabs. Surgery room-entry time is optional and Cyprus-local; accident cases keep separate append-only payment events; eligible exact RF applications can be added to the surgery list after clinician confirmation. RF tracking and scheduling/performance states remain their existing owners.

## Reviewed source, database and CI

| Item | Verified result |
|---|---|
| PR | [#223](https://github.com/athpapachr-cmd/osteoporosis/pull/223), squash merged |
| Reviewed runtime/SQL/test | `54d72e9d815fc0e2d25bd5038f4666e045443575` |
| Independent post-code R2 | PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0 |
| Final PR HEAD | `a52eaef4b76e4026bbe70fd4160e0562a82d48aa` |
| Final CI | 19/19 SUCCESS, including Office Cases Chromium and PostgreSQL 17 migration jobs |
| Merge/main | `bd6133fef96a4e0f4a88e96907ec03ef5a16a263` |
| Release gate | Fresh Ops P0.4 RELEASE sidecars and `orctl gate` PASS before merge and after merge; unrelated Ops `orctl verify` stale Backend-main declaration was qualified to that separate Backend scope. |

The reviewed SQL SHA-256 `3c1f67b55555894eb3dfdd1c8f734f5f62d2dbaccbf3268adb685bb0199e37a8` was installed once in Aiven `osteoporosis` by `avnadmin` with verify-full TLS before code deployment. Metadata checks found four empty new tables, the required foreign keys and unique payment replay constraint, app SELECT/INSERT/limited UPDATE grants, and no app DELETE/schema CREATE. The Product Owner authorized temporary `80.244.29.141/32` access; it was removed after installation, restoring the original three allowlist entries. No clinical row content was read or written.

## Native Render deployment and no-write smoke

| Check | Observed result |
|---|---|
| Service | `osteoporosis`, Render `srv-d5qfk31r0fns73di596g` |
| Native deploy | `dep-dbfecim7bikc73c4dbj0`, triggered by merge; no manual duplicate |
| Exact deployed source | `bd6133fef96a4e0f4a88e96907ec03ef5a16a263` |
| Status | LIVE, finished 2026-10-26 05:36:06 UTC |
| Static assets | `office-cases.js`, `doctor-shell.js`, `doctor-shell.css` and Cockpit `index.html` returned 200 and SHA-256 matched the merged workspace bytes |
| Anonymous protected reads | `/clinical/office-cases/accidents`, `/clinical/office-cases/rf`, `/clinical/surgeries` each returned 401 |
| Anonymous write probe | Invalid `{}` POST to `/clinical/office-cases/accidents` returned 401; no row was created |
| Error-level app logs | Zero in the queried release window from 05:34 UTC |
| Config/secret/provider mutations in this deploy | NONE; the reviewed additive Aiven migration was the separately completed production data change |

This smoke verifies deployment identity, served bytes and authentication boundary. It does not prove a real patient registration, payment entry, room time edit or RF transfer. The agent did not use an authenticated clinical key or create/alter a real surgery, accident, RF or payment record. The clinician's first real case entry and visual acceptance are next. Priority 4 requires a separate Product Owner acceptance decision.
