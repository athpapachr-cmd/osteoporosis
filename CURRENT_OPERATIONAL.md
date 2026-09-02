# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-4 DEPLOYED / PRODUCTION SMOKE PARTIAL; RF AUTH GATEWAY HOTFIX PR #73 OPEN / MERGE HOLD.
> **Updated:** 2026-09-02 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 release PR:** `#72` — squash merged.
> **G-4 merge/deployed source:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 Render deploy:** `dep-dac27kojo6nc739biu80` — product-owner supplied `live` evidence at exact merge source.
> **Hotfix branch:** `fix/module01-g4-rf-auth-gateway-2026-09-02`.
> **Hotfix exact tested runtime head:** `29140a6cd4c9f57b454daa6e4a2883ec0345b53f`.
> **Hotfix exact workflow:** `G3 guidance salience longitudinal summary` run `33640110048` — SUCCESS.
> **Hotfix PR:** `#73` — open against `main`.
> **Pre-PR canonical-closeout head:** `97f9ffd05637d838ae5c93a85f9be8bd15bc0247`.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR authority:** GRANTED AND CONSUMED by product-owner direction `προχώρα σε auth hotfix PR`.
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

Product-owner production smoke after G-4 deployment verified:

```text
Σύνοψη ασθενούς collapse/expand        PASS
Σημερινή ροή independent collapse      PASS
sticky patient summary                 PASS
physiotherapy utility navigation        PASS
RF navigation reaches intended service  PASS
RF authorized form usability            FAIL
```

Observed RF result:

```text
{"detail":"Απαιτείται εξουσιοδοτημένη πρόσβαση."}
```

The defect is a cross-service authentication integration gap, not evidence of RF form/PDF-engine failure.

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

The browser never receives the RF credential. The RF service remains owner of form content, history/persistence, validation, templates and generated PDF.

Final browser-facing routes:

```text
GET  /clinical/clinic-utilities/rf
POST /clinical/clinic-utilities/rf/history
POST /clinical/clinic-utilities/rf/create
GET  /clinical/clinic-utilities/rf/pdf/{application_id}
```

The local history route is deliberately POST-only. Identity/GeSY identifiers are placed in a bounded form-urlencoded body rather than an Osteoporosis query string/browser-history URL. The gateway translates only the three allowed fields to the existing upstream GET contract server-to-server.

Hard boundaries preserved:

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

# 3. Exact executable evidence

Exact tested runtime head:

```text
29140a6cd4c9f57b454daa6e4a2883ec0345b53f
```

GitHub Actions:

```text
workflow: G3 guidance salience longitudinal summary
run:      33640110048
result:   SUCCESS
```

The exact runtime gate passed RF gateway syntax/security/privacy regressions plus inherited G4/G3/G2/G1/C1 regressions. No real RF patient payload or identifiable patient data was used.

Independent review corrected the local history transport to POST-only before the final tested runtime head. The final runtime also keeps a fixed upstream origin/route family, bounded request bodies/application IDs, sanitized upstream auth/config/network failures and fail-closed upstream-form compatibility adaptation.

---

# 4. Canonical / drift evidence before PR

Canonical closeout completed after the runtime test gate.

Compare:

```text
29140a6cd4c9f57b454daa6e4a2883ec0345b53f
→ 97f9ffd05637d838ae5c93a85f9be8bd15bc0247
```

showed only canonical documentation changes:

```text
CURRENT_OPERATIONAL.md
SLICE_PLAN_CURRENT.md
TODO.md
osteoporosis-change-log.md
```

No runtime/test/workflow drift occurred after the exact tested runtime head.

Pre-PR compare against production `main` showed:

```text
behind by 0
merge base = exact main 338830340f6fed2ae1a3f08f6fdb0b8059932a66
```

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
HOTFIX PR CHECKS                      PENDING / MUST VERIFY EXACT CURRENT PR HEAD
HOTFIX MERGED                         NO
RF_GATEWAY_ACCESS_KEY CONFIGURED      NO / NOT VERIFIED
HOTFIX DEPLOYED                       NO
HOTFIX PRODUCTION-SMOKE-VERIFIED      NO
HOTFIX PILOT-VALIDATED                NO
ACTIVE WRITER                         NONE
```

Hard status distinction:

```text
IMPLEMENTED
!= TESTED
!= PR-CHECKED
!= MERGED
!= CONFIGURED
!= DEPLOYED
!= PRODUCTION-SMOKE-VERIFIED
!= PILOT-VALIDATED
```

C2 remains separately implemented/tested and unreleased. PR-1/PR-2 and later rich physiotherapy referral work are outside this hotfix.

---

# 6. Cross-repository / configuration constraint

The RF backend remains governed independently by `athpapachr-cmd/ortho-reception-ops`. No RF backend runtime, config or secret was mutated by this hotfix.

Production enablement later requires a separately authorized server-side copy of the existing accepted RF credential into the Osteoporosis Render service under:

```text
RF_GATEWAY_ACCESS_KEY
```

No credential value belongs in GitHub or chat.

---

# 7. Exact next action / STOP gate

Current authorized sequence:

```text
PR #73 open
→ verify all PR-triggered checks on the exact current PR head
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
