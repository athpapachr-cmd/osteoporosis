# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-4 DEPLOYED / PRODUCTION SMOKE PARTIAL; RF AUTH GATEWAY HOTFIX IMPLEMENTED / TESTED / REVIEW PASS; RELEASE HOLD.
> **Updated:** 2026-09-02 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 release PR:** `#72` — squash merged.
> **G-4 merge/deployed source:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 Render deploy:** `dep-dac27kojo6nc739biu80` — product-owner supplied `live` evidence at exact merge source.
> **Hotfix branch:** `fix/module01-g4-rf-auth-gateway-2026-09-02`.
> **Hotfix exact tested runtime head:** `29140a6cd4c9f57b454daa6e4a2883ec0345b53f`.
> **Hotfix exact workflow:** `G3 guidance salience longitudinal summary` run `33640110048` — SUCCESS.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after this canonical closeout.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR authority:** NONE unless separately granted.
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

# 2. Bounded RF-auth hotfix final state

Implemented architecture:

```text
authenticated Cockpit browser
→ /clinical/clinic-utilities/rf
→ existing ClinicalCookieMiddleware / clinical-key gate
→ Osteoporosis server-side RF gateway
→ X-RF-Key from RF_GATEWAY_ACCESS_KEY
→ fixed existing RF service
```

The browser never receives the RF credential. The RF service remains owner of the form, history/persistence, validation, templates and generated PDF.

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

The exact-head gate passed:

1. JavaScript syntax;
2. RF gateway Python syntax;
3. G4 RF authenticated-gateway regression;
4. POST-only RF history privacy regression;
5. G4 workspace ergonomics/RF-navigation regression;
6. G3 salience/longitudinal-summary regressions;
7. G3 production visibility/cache;
8. frozen G2 contract/runtime regressions;
9. G1 core/wiring/UI/WHY-NOW regressions;
10. C1 authoritative Finish browser regression;
11. server finalization lifecycle regression.

No real RF patient payload or identifiable patient data was used in this pre-release evidence.

---

# 4. Independent review disposition

Independent source/security/scope review of the final runtime found no remaining release-blocking defect in the bounded hotfix.

One material privacy issue was identified during review **before** the final head: a local browser GET history lookup would have created an additional identifier-bearing URL/access-log surface in the Osteoporosis service. That design was corrected to POST-only history transport and covered by the final successful exact-head workflow.

Final reviewed properties:

```text
fixed upstream origin
fixed route family
server-only RF credential
POST-only local identifier history lookup
bounded request bodies / application id
sanitized upstream auth/config/network failures
fail-closed upstream-form compatibility adaptation
no RF business/template duplication
no osteoporosis clinical-state mutation
no G1/G2/G3/C1 semantic leakage
```

---

# 5. Hotfix state matrix

```text
HOTFIX DESIGN                         COMPLETE
HOTFIX IMPLEMENTED                    YES
HOTFIX TESTED                         YES
HOTFIX EXACT-HEAD REVIEW              PASS
HOTFIX CANONICAL CLOSEOUT             PASS
FINAL POST-RUNTIME DOCS-ONLY DRIFT    PASS subject to immediate branch compare verification
HOTFIX PR                             NONE
HOTFIX MERGED                         NO
RF_GATEWAY_ACCESS_KEY CONFIGURED      NO / NOT VERIFIED
HOTFIX DEPLOYED                       NO
HOTFIX PRODUCTION-SMOKE-VERIFIED      NO
ACTIVE WRITER                         NONE
```

Hard status distinction:

```text
IMPLEMENTED
!= TESTED
!= MERGED
!= CONFIGURED
!= DEPLOYED
!= PRODUCTION-SMOKE-VERIFIED
!= PILOT-VALIDATED
```

C2 remains separately implemented/tested and unreleased. PR-1/PR-2 and later rich physiotherapy referral work are outside this hotfix.

---

# 6. Cross-repository constraint

The RF backend remains governed independently by `athpapachr-cmd/ortho-reception-ops`. No RF backend runtime, config or secret was mutated by this hotfix.

The current solution requires only a future server-side copy of the existing accepted RF credential into the Osteoporosis service under:

```text
RF_GATEWAY_ACCESS_KEY
```

That production configuration mutation is **not authorized by this closeout** and no credential value belongs in GitHub or chat.

---

# 7. Exact next action / STOP gate

The implementation writer is released. The current authorized work ends after the requested final docs-only drift verification.

Next possible release sequence requires separate explicit product-owner authority:

```text
open bounded RF-auth hotfix PR
→ verify exact PR-head checks
→ separate merge decision
→ separately authorize/set RF_GATEWAY_ACCESS_KEY on Osteoporosis Render service
→ normal Render auto-deploy from merged main
→ bounded production smoke of RF form/history/create/PDF
```

Until then:

```text
NO PR
NO MERGE
NO production config/secret mutation
NO manual Render deploy
NO mutation of ortho-reception-backend-v2
NO claim of RF production-smoke verification
NO pilot claim
```

A future production smoke must confirm that RF form rendering, POST-only history lookup, create/PDF flow and credential non-exposure all work on the deployed runtime while the already-passed G4 ergonomics remain intact.
