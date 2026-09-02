# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-4 DEPLOYED / PRODUCTION SMOKE PARTIAL; RF AUTH GATEWAY HOTFIX PR #73 OPEN / PR CHECKS PASS / MERGE HOLD.
> **Updated:** 2026-09-02 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 release PR:** `#72` — squash merged.
> **G-4 merge/deployed source:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 Render deploy:** `dep-dac27kojo6nc739biu80` — product-owner supplied `live` evidence at exact merge source.
> **Hotfix branch:** `fix/module01-g4-rf-auth-gateway-2026-09-02`.
> **Hotfix exact tested runtime head:** `29140a6cd4c9f57b454daa6e4a2883ec0345b53f`.
> **Pre-PR canonical-closeout head:** `97f9ffd05637d838ae5c93a85f9be8bd15bc0247`.
> **PR #73 exact checked head:** `1c2533616e1606b6d9fe005afbe3b45be12aac01`.
> **PR #73 CU-1 workflow:** run `33648573608` — SUCCESS.
> **PR #73 G3/G4 inherited workflow:** run `33648573632` — SUCCESS.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR authority:** GRANTED AND CONSUMED.
> **Merge authority:** NONE.
> **Production config/secret authority:** NONE.
> **Deploy authority:** NONE.

---

# 1. Verified G-4 production state

```text
G-4 DESIGN                         COMPLETE
G-4 IMPLEMENTED                    YES
G-4 TESTED                         YES
G-4 RELEASE-READINESS REVIEW       PASS
G-4 PR                             #72
G-4 MERGED                         YES
G-4 DEPLOYED                       YES
G-4 PRODUCTION-SMOKE-VERIFIED      NO — RF authorized usability blocker remains unresolved in production
G-4 PILOT-VALIDATED                NO
```

Product-owner production smoke after G-4 deployment verified collapse/expand, independent current-flow collapse, sticky patient summary, physiotherapy navigation and navigation to the intended RF service. RF authorized form usability failed with the protected service returning `{"detail":"Απαιτείται εξουσιοδοτημένη πρόσβαση."}`.

---

# 2. Bounded RF-auth hotfix final runtime boundary

Implemented architecture:

```text
authenticated Cockpit browser
→ /clinical/clinic-utilities/rf
→ existing ClinicalCookieMiddleware / clinical-key gate
→ Osteoporosis server-side RF gateway
→ X-RF-Key from RF_GATEWAY_ACCESS_KEY
→ fixed existing RF service
```

Final browser-facing routes:

```text
GET  /clinical/clinic-utilities/rf
POST /clinical/clinic-utilities/rf/history
POST /clinical/clinic-utilities/rf/create
GET  /clinical/clinic-utilities/rf/pdf/{application_id}
```

The local history route is POST-only so identity/GeSY identifiers are carried in a bounded form-urlencoded body rather than an Osteoporosis query-string/browser-history URL. The gateway maps only the allowed fields server-to-server to the existing RF history contract.

Preserved boundaries:

```text
NO RF secret in JS / HTML / URL
NO clinical_session forwarding upstream
NO browser Authorization/X-Clinical-Key forwarding upstream
NO user-controlled upstream host/path family
NO RF data persisted into osteoporosis encounter state
NO RF template/PDF/business-rule duplication
NO RF backend auth weakening
NO ortho-reception runtime/config mutation
```

---

# 3. Pre-PR executable evidence

Exact tested runtime head:

```text
29140a6cd4c9f57b454daa6e4a2883ec0345b53f
```

Workflow:

```text
G3 guidance salience longitudinal summary
run 33640110048
SUCCESS
```

This gate passed RF gateway syntax/security/privacy regressions plus inherited G4/G3/G2/G1/C1 regressions. No identifiable patient data was used.

Canonical closeout after that runtime head changed only `CURRENT_OPERATIONAL.md`, `SLICE_PLAN_CURRENT.md`, `TODO.md` and append-only `osteoporosis-change-log.md`. Pre-PR compare showed branch behind `main` by 0 with merge-base exactly `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.

---

# 4. PR #73 exact-head evidence

PR:

```text
#73 — G4 hotfix: authenticated RF gateway
base main @ 338830340f6fed2ae1a3f08f6fdb0b8059932a66
checked head 1c2533616e1606b6d9fe005afbe3b45be12aac01
mergeable YES
```

PR-triggered checks at that exact checked head:

```text
CU-1 focused tests
run 33648573608
SUCCESS

G3 guidance salience longitudinal summary
run 33648573632
SUCCESS
```

The G3/G4 workflow passed:

- JavaScript syntax;
- RF gateway Python syntax;
- G4 RF authenticated-gateway regression;
- POST-only RF history privacy regression;
- G4 workspace ergonomics/RF utility regression;
- G3 salience/longitudinal summary and ownership regressions;
- G3 production visibility/cache;
- frozen G2 contract/core/live-state/wiring regressions;
- G1 core/wiring/UI/WHY-NOW regressions;
- C1 authoritative Finish browser regression;
- server finalization lifecycle regression.

CU-1 focused tests also passed, preserving the adjacent Clinic Utilities package boundary.

---

# 5. Hotfix release state

```text
HOTFIX DESIGN                         COMPLETE
HOTFIX IMPLEMENTED                    YES
HOTFIX TESTED                         YES
HOTFIX EXACT-HEAD REVIEW              PASS
HOTFIX CANONICAL CLOSEOUT             PASS
FINAL POST-RUNTIME DOCS-ONLY DRIFT    PASS
HOTFIX PR                             #73 OPEN
HOTFIX PR CHECKS                      PASS @ 1c2533616e1606b6d9fe005afbe3b45be12aac01
HOTFIX MERGED                         NO
RF_GATEWAY_ACCESS_KEY CONFIGURED      NO / NOT VERIFIED
HOTFIX DEPLOYED                       NO
HOTFIX PRODUCTION-SMOKE-VERIFIED      NO
HOTFIX PILOT-VALIDATED                NO
ACTIVE WRITER                         NONE
```

`IMPLEMENTED != TESTED != PR-CHECKED != MERGED != CONFIGURED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED != PILOT-VALIDATED`.

C2 remains separately implemented/tested and unreleased. PR-1/PR-2 and later rich physiotherapy work are outside this hotfix.

---

# 6. Cross-repository / configuration constraint

The RF backend remains governed independently by `athpapachr-cmd/ortho-reception-ops`. No RF backend runtime, config or secret was mutated.

Future production enablement requires separately authorized server-side configuration on the Osteoporosis Render service:

```text
RF_GATEWAY_ACCESS_KEY
```

No credential value belongs in GitHub or chat.

---

# 7. Exact next action / STOP gate

```text
PR #73 OPEN + exact checked head PASS
→ STOP for separate product-owner merge decision
```

Forbidden without new explicit authority:

```text
NO MERGE
NO production config/secret mutation
NO manual Render deploy
NO mutation of ortho-reception-backend-v2
NO claim of RF production-smoke verification
NO pilot claim
```

After any future authorized merge/config/deploy, production smoke must confirm RF form rendering, POST-only history lookup, create/PDF flow and credential non-exposure while the already-passed G4 ergonomics remain intact.
