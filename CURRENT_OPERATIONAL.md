# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-4 DEPLOYED / PRODUCTION SMOKE PARTIAL; BOUNDED RF AUTH GATEWAY HOTFIX ACTIVE.
> **Updated:** 2026-09-02 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 release PR:** `#72` — squash merged.
> **G-4 merge/deployed source:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 Render deploy:** `dep-dac27kojo6nc739biu80` — product-owner supplied `live` evidence at exact merge source.
> **Hotfix branch:** `fix/module01-g4-rf-auth-gateway-2026-09-02`.
> **ACTIVE CANONICAL WRITER/LOCK:** this branch / bounded G4 RF auth gateway hotfix.
> **ACTIVE RUNTIME WRITER/LOCK:** this branch / osteoporosis-only RF gateway surface.

---

# 1. Verified G-4 release state

```text
G-4 DESIGN                         COMPLETE
G-4 IMPLEMENTED                    YES
G-4 TESTED                         YES
G-4 RELEASE-READINESS REVIEW       PASS
G-4 PR                             #72
G-4 MERGED                         YES
G-4 DEPLOYED                       YES
G-4 PRODUCTION-SMOKE-VERIFIED      NO — RF authorized usability blocker remains
G-4 PILOT-VALIDATED                NO
```

Product-owner production smoke after deploy verified:

```text
Σύνοψη ασθενούς collapse/expand       PASS
Σημερινή ροή independent collapse     PASS
sticky patient summary                PASS
physiotherapy utility navigation       PASS
RF navigation reaches intended service PASS
RF authorized form usability           FAIL
```

Observed RF result after navigation:

```text
{"detail":"Απαιτείται εξουσιοδοτημένη πρόσβαση."}
```

The failure is an authentication-integration gap, not an RF form/PDF-engine failure.

---

# 2. Root cause / security boundary

The G-4 browser link currently navigates directly to:

```text
https://ortho-reception-backend-v2.onrender.com/rf
```

That endpoint is intentionally protected by the existing RF service. A browser arriving from the Osteoporosis Cockpit has neither the RF service's `rf_session` cookie nor an RF access credential, so the RF service correctly fails closed with HTTP 401.

Hard security constraints:

```text
NO RF access key in public JavaScript
NO long-lived secret in query string / browser history
NO patient/clinical cookie forwarded cross-service
NO RF template/PDF/business-state duplication
NO weakening/removal of RF authorization
```

---

# 3. Cross-repository authority result

The RF runtime lives in `athpapachr-cmd/ortho-reception-backend-v2`, whose canonical authority is `athpapachr-cmd/ortho-reception-ops`.

Fresh ops bootstrap on `a9b637e47cd91c6d421ec8a7c6fcb12f8ebf1044` found an active Call Causal Trace v1 slice with Backend PR #131 on frozen production base `92ae6c8857a5abf0967926772840cc141b731727` and:

```text
runtime writer             NONE pending independent review
merge authority            NONE
config/secret authority    NONE
production-smoke authority NONE
```

Therefore this hotfix MUST NOT mutate the reception Backend runtime or its configuration while that gate is active.

---

# 4. Approved bounded hotfix design

Implement an **Osteoporosis-side authenticated reverse gateway** only:

```text
authenticated Cockpit browser
→ /clinical/clinic-utilities/rf
→ existing ClinicalCookieMiddleware / X-Clinical-Key gate
→ osteoporosis server injects RF gateway credential server-to-server
→ fixed upstream https://ortho-reception-backend-v2.onrender.com
→ existing protected /rf workflow
```

Gateway surface is intentionally limited to the current RF workflow:

```text
GET  /clinical/clinic-utilities/rf
GET  /clinical/clinic-utilities/rf/history
POST /clinical/clinic-utilities/rf/create
GET  /clinical/clinic-utilities/rf/pdf/{application_id}
```

The gateway:

- uses fixed upstream origin; no user-controlled proxy target / SSRF seam;
- injects `X-RF-Key` only server-to-server from environment variable `RF_GATEWAY_ACCESS_KEY`;
- never renders/logs/returns that credential;
- never forwards the Cockpit's `clinical_session`, `Authorization` or `X-Clinical-Key` to the RF service;
- rewrites only RF route references in the upstream form so subsequent browser actions remain on the protected same-origin gateway;
- preserves upstream validation/PDF behavior and does not duplicate RF templates, coordinates, rules or persistence;
- applies bounded request size/timeouts and sanitized upstream-auth/network failures;
- retains `missing != negative`, patient-data and existing clinical-state boundaries because no RF payload enters an osteoporosis encounter.

Configuration is separate from code implementation:

```text
RF_GATEWAY_ACCESS_KEY on Osteoporosis Render service
= same existing RF access credential already accepted by the RF service
```

No reception service config mutation is required. Secret/config mutation and release remain separate gates and are not implied by implementation authority.

---

# 5. In-scope runtime/test files

Expected bounded surface:

```text
clinic_utilities/rf_gateway.py                  NEW
main.py                                         include router
static/baseline-audit/g4-workspace-ergonomics.js RF link → same-origin gateway
test_g4_rf_auth_gateway.py                      NEW focused server regression
test_g4_workspace_ergonomics.js                 update navigation ownership assertion
```

Workflow changes only if the existing G-4 test workflow cannot execute the focused regression without modification.

No changes to:

```text
G-2 evidence rules / thresholds
G-3 longitudinal summary derivation
G-1 VisitPlan semantics
C1 Finish/finalization
patient API / DB schema
C2
PR-1 / PR-2
RF backend code/templates/PDF logic/persistence
Ortho-Reception runtime/configuration
```

---

# 6. Acceptance gate

Implementation evidence must prove at minimum:

1. gateway routes require existing clinical authentication;
2. RF credential never reaches browser-visible HTML/URL or returned headers;
3. only fixed RF upstream paths are reachable;
4. form action/history fetch are rewritten to same-origin protected gateway;
5. POST body/content-type are forwarded without altering RF business semantics;
6. upstream 303 PDF redirect is rewritten to same-origin gateway;
7. PDF/content-disposition is returned intact;
8. absent RF gateway secret fails closed without contacting upstream;
9. upstream authorization/network failure is sanitized;
10. existing G-4 collapse/sticky/physio behavior remains unchanged.

---

# 7. Exact next action / STOP gate

```text
implement bounded osteoporosis-only RF gateway
→ run focused G4/RF regression + directly affected inherited G4 checks
→ exact-head review
→ STOP before PR/merge/config/deploy unless separately authorized
```

Forbidden now:

```text
NO mutation of ortho-reception-backend-v2
NO reception config/secret mutation
NO merge
NO production config mutation
NO deploy
NO claim of PRODUCTION-SMOKE-VERIFIED
```
