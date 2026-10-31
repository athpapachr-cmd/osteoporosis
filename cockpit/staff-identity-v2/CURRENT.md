# Cockpit Individual Staff V2 — attributable actor foundation

**2026-10-31 · DRAFT / FEATURE-OFF / NO MERGE OR DEPLOY**

This is a production-bound identity prerequisite for Issue #266, not another clinical task engine.

## Implemented

- Existing /clinical/login, role-wide V1 staff credentials, Visit Brief and broad Clinical Key continue unchanged by default.
- New /clinical/individual-staff-v2/login, /me and /logout routes are mounted but V2 sign-in is disabled unless COCKPIT_INDIVIDUAL_STAFF_V2_ENABLED=1 and valid private configuration exists.
- Private server-only actors JSON contains per-person opaque principal_id, clinician/secretary role, active flag, revision and SHA-256 hash of a 256-bit randomly generated URL-safe API token. This hash scheme is ONLY for random tokens; **never** for human-memorable passwords.
- Independent HMAC session signing, HttpOnly Secure SameSite Strict cookie, 60-minute expiry, immediate invalidation after account disabling, removal, role/credential or revision change.
- Individually scoped dependency guards for clinician and secretary. They never trust browser-supplied role, legacy Clinical Key, or role-wide V1 keys as personal identity.
- Individual login clears broad/v1 cookies. Presence of an expired or invalid V2 cookie blocks previously issued broad-cookie fallback.
- NO new clinical patient/visit write route, no real credentials, no ClinicalPending mutation, no iCloud, no SMS, no database migration and no PHI in this PR.

## Next use of this identity authority

The future real follow-through adapter must separately bind these server-verified principals to durable actor-attributed events, use the existing signed ClinicalPending identity and approved database migration, and expose only minimum-data secretary contact tasks. A person being authenticated is necessary but not sufficient for a clinical action; order/patient identity and explicit clinician decision still gate it.

Release remains separately gated by provider/order qualification, patient contact verification, privacy/retention/backup and independently reviewed staff/clinical source authorization. No rollout of V2 is requested here.

## Offline acceptance

TestClient tests role separation, unrelated shared Clinical Key, existing v1 staff sessions, fallback prevention, token lifecycle, revocation and rotation, tampering, HTTPS-only, malformed configuration and default-off behavior. CI also runs inherited Visit Brief regressions.

## Review separation

Parent PRs #267 and #268 are frozen synthetic checkpoints, each with past findings corrected downstream by #269 (reported independent R5 PASS for the cumulative synthetic scope). This identity V2 PR does not change those snapshots, is independent of them and grants no production write or deploy authority.
