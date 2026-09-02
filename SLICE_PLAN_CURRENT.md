# SLICE_PLAN_CURRENT.md — G-4 RF Authorized Gateway Hotfix v1

> **STATUS:** IMPLEMENTED / TESTED / EXACT-HEAD REVIEW PASS — CANONICAL CLOSEOUT COMPLETE; RELEASE HOLD.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** Clinical Excellence workspace + cross-module Clinic Utilities.
> **Slice ID:** `M01-G4-RF-AUTH-GATEWAY-HOTFIX-v1`.
> **Production base:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **Branch:** `fix/module01-g4-rf-auth-gateway-2026-09-02`.
> **Exact tested runtime head:** `29140a6cd4c9f57b454daa6e4a2883ec0345b53f`.
> **Exact test workflow:** `G3 guidance salience longitudinal summary`, run `33640110048` — SUCCESS.
> **Runtime writer:** NONE.
> **Product-owner implementation authority:** consumed.
> **PR / merge / config / deploy authority:** NONE unless separately granted.

---

# 1. Trigger / first real failure

G-4 PR #72 was squash-merged to `main` as:

```text
338830340f6fed2ae1a3f08f6fdb0b8059932a66
```

and Render auto-deploy:

```text
dep-dac27kojo6nc739biu80
```

reached `live` at that exact source according to product-owner production evidence.

The same smoke verified the workspace ergonomics and physiotherapy utility path, but direct RF navigation returned:

```text
{"detail":"Απαιτείται εξουσιοδοτημένη πρόσβαση."}
```

Observed causal chain:

```text
Cockpit authenticated on Osteoporosis origin
→ browser opens external RF origin
→ browser has no RF-service session/access credential
→ existing RF authorization correctly rejects with 401
```

Generalized failure class:

```text
CROSS-SERVICE AUTHENTICATION INTEGRATION GAP
```

This was not evidence of RF PDF/template/business-rule failure and was not solved by weakening RF authorization.

---

# 2. Cross-repository authority boundary

The RF runtime lives in `athpapachr-cmd/ortho-reception-backend-v2`, whose mandatory operational control plane is `athpapachr-cmd/ortho-reception-ops`.

The implementation investigation bootstrapped that control plane and found an independently governed active Backend slice with no authority for an unrelated RF runtime/config/secret mutation. Therefore this hotfix deliberately did **not** change:

```text
ortho-reception-backend-v2 runtime
RF backend authorization contract
RF backend configuration/secrets
RF templates/PDF engine/persistence
```

A cross-repository short-lived token/session handoff remains architecturally possible later, but it was not authorized or required for this bounded correction.

---

# 3. Final implemented architecture

Chosen design:

```text
authenticated Cockpit browser
        ↓
ClinicalCookieMiddleware
        ↓
/clinical/clinic-utilities/rf[/...]
        ↓
existing protected clinical-key contract
        ↓
Osteoporosis RF gateway
        ↓  server-only X-RF-Key from RF_GATEWAY_ACCESS_KEY
fixed upstream https://ortho-reception-backend-v2.onrender.com
        ↓
existing protected RF implementation
```

The browser never receives the RF credential and never needs an RF-service cookie.

The RF backend remains the source of truth for:

- live form HTML/content;
- indications/consumables/application locations;
- repeat-use validation;
- upload validation;
- RF request/history persistence;
- official PDF templates/coordinates;
- generated PDF output.

The gateway owns only bounded transport/auth adaptation.

---

# 4. Final route contract

Browser-facing gateway routes:

```text
GET  /clinical/clinic-utilities/rf
POST /clinical/clinic-utilities/rf/history
POST /clinical/clinic-utilities/rf/create
GET  /clinical/clinic-utilities/rf/pdf/{application_id}
```

Fixed upstream mappings:

```text
GET  local /rf
→ GET  upstream /rf

POST local /rf/history
→ parse exactly three form-urlencoded fields
→ GET upstream /rf/history with those named query params

POST local /rf/create
→ POST upstream /rf/create with raw multipart body + Content-Type only

GET local /rf/pdf/{id}
→ GET upstream /rf/pdf/{validated-id}
```

No arbitrary path or user-controlled upstream host exists.

---

# 5. Privacy/security invariants

```text
NO RF secret in JS / HTML / browser URL / browser history
NO forwarding of clinical_session upstream
NO forwarding of browser Authorization upstream
NO forwarding of browser X-Clinical-Key upstream
NO user-controlled upstream URL
NO open reverse-proxy surface
NO RF payload persisted into osteoporosis encounter state
NO RF template/rule duplication
NO RF auth weakening
```

Server-side credential source:

```text
RF_GATEWAY_ACCESS_KEY
```

The upstream origin is a code constant:

```text
https://ortho-reception-backend-v2.onrender.com
```

The gateway injects only:

```text
X-RF-Key: <server-side value>
Accept: */*
Content-Type only when required
```

The key is fail-closed when absent and is never deliberately returned in browser-visible response content/headers.

---

# 6. POST-only RF history privacy correction

Independent review identified a material privacy issue in the first gateway candidate: retaining a local browser `GET /history?...` would create a second Osteoporosis access-log/URL surface containing identity and GeSY identifiers.

The final runtime therefore changed the **browser-facing history transport** to POST:

```text
browser
→ POST /clinical/clinic-utilities/rf/history
→ application/x-www-form-urlencoded
→ maximum body 4096 bytes
→ accepted names only:
   identity_number
   gesy_number
   application_location
```

The gateway then performs the existing RF backend's required upstream GET server-to-server.

Consequences:

- identifier values do not appear in the Osteoporosis browser URL/history;
- local GET `/history` is not implemented and therefore returns framework 405;
- duplicate values, oversized fields/body, malformed encoding and wrong content type fail closed;
- unexpected form fields are ignored and are never forwarded upstream.

This is transport privacy hardening, not new RF business logic.

---

# 7. Form adaptation contract

The existing upstream RF form owns the HTML. The gateway adapts only two expected transport seams:

```text
action="/rf/create"
→ action="/clinical/clinic-utilities/rf/create"

fetch('/rf/history?' + query.toString(), { credentials: 'same-origin' })
→ same-origin POST /clinical/clinic-utilities/rf/history
   with application/x-www-form-urlencoded body
```

If either expected upstream form seam disappears or changes, the gateway fails closed with a sanitized 502 compatibility error instead of serving a silently broken/privacy-unsafe form.

No RF form/template copy is stored in this repository.

---

# 8. Response/failure behavior

Successful behavior:

- form HTML returned with `Cache-Control: no-store`;
- history JSON/status relayed without credential headers;
- upstream `303 /rf/pdf/{id}` rewritten to same-origin protected gateway PDF URL;
- PDF bytes, media type and `Content-Disposition` relayed.

Fail-closed behavior:

```text
missing RF_GATEWAY_ACCESS_KEY
→ local 503
→ no upstream request

upstream 401/403
→ sanitized 502

upstream 5xx
→ sanitized 502

network failure
→ sanitized 502

timeout
→ sanitized 504

invalid PDF application id/path
→ no arbitrary upstream path
```

Create requests are bounded to 24 MiB around the existing RF upload requirement.

---

# 9. Implemented surface

Runtime/test changes are bounded to:

```text
clinic_utilities/rf_gateway.py                    NEW
main.py                                           include RF gateway router
static/baseline-audit/g4-workspace-ergonomics.js  RF link → same-origin gateway
test_g4_rf_auth_gateway.py                        NEW focused gateway regression
test_g4_workspace_ergonomics.js                   updated RF ownership/navigation assertions
.github/workflows/g3-guidance-summary-tests.yml    run focused RF Python gate on branch/PR
```

Explicitly unchanged:

```text
G-2 evidence rules / thresholds
G-3 longitudinal summary derivation
G-1 VisitPlan semantics
C1 Finish/finalization
patient API / DB schema
C2
PR-1 / PR-2
RF backend runtime/templates/PDF logic/persistence
Ortho-Reception configuration/secrets
```

---

# 10. Exact executable evidence

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

Passed on that exact head:

1. JavaScript syntax;
2. RF gateway Python syntax;
3. focused G4 RF authenticated-gateway regression;
4. POST-only history privacy regression;
5. G4 workspace ergonomics/RF navigation regression;
6. G3 salience and longitudinal-summary regressions;
7. G3 production visibility/cache regression;
8. frozen G2 evidence contract + G2 runtime regressions;
9. G1 core/wiring/UI/WHY-NOW regressions;
10. C1 authoritative Finish browser regression;
11. server finalization lifecycle regression.

Synthetic tests use no identifiable patient data and make no real RF patient request.

---

# 11. Independent exact-head review

The final source/security/scope review of `29140a6c...` found no remaining release-blocking defect in the bounded hotfix surface.

Verified properties include:

```text
fixed upstream host/path family
server-only RF credential
no browser credential propagation upstream
POST-only local identifier history transport
bounded request bodies / application ids
sanitized auth/config/network failure behavior
no RF business/template duplication
no osteoporosis clinical-state write
no G1/G2/G3/C1 semantic leakage
```

The privacy finding discovered during review was corrected **before** the final tested runtime head and is covered by the successful exact-head workflow.

---

# 12. Completion / release matrix

```text
HOTFIX DESIGN                         COMPLETE
HOTFIX IMPLEMENTED                    YES
HOTFIX TESTED                         YES
HOTFIX EXACT-HEAD REVIEW              PASS
HOTFIX CANONICAL CLOSEOUT             PASS
FINAL POST-RUNTIME DOCS-ONLY DRIFT    PASS subject to immediate branch compare verification
PR                                    NONE
MERGED                                NO
RF_GATEWAY_ACCESS_KEY CONFIGURED      NO / NOT VERIFIED
DEPLOYED                              NO
PRODUCTION-SMOKE-VERIFIED             NO
G-4 PILOT-VALIDATED                   NO
ACTIVE RUNTIME WRITER                 NONE
```

`IMPLEMENTED != TESTED != MERGED != CONFIGURED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED != PILOT-VALIDATED` remains explicit.

---

# 13. Release/config gate

The hotfix implementation/test writer is closed.

Next possible release sequence requires new explicit product-owner authority:

```text
open bounded RF-auth hotfix PR
→ verify exact PR-head checks
→ separate merge decision
→ securely configure RF_GATEWAY_ACCESS_KEY on Osteoporosis Render service
→ allow normal Render auto-deploy from merged main
→ production smoke RF form/history/create/PDF path
```

No RF credential value belongs in GitHub, chat output or client-side code.

Until separately authorized:

```text
NO PR
NO MERGE
NO production config mutation
NO manual deploy
NO claim of production smoke completion
NO mutation of ortho-reception-backend-v2
```

---

# 14. Production-smoke acceptance after future release

A later authorized production smoke must establish at minimum:

1. authenticated Cockpit RF navigation opens the same-origin gateway rather than the external unauthenticated page;
2. the live RF form renders;
3. history lookup works without identifier-bearing local URL/query strings;
4. create flow reaches existing RF validation/PDF generation;
5. generated PDF route/download works through the gateway;
6. no RF credential appears in browser-visible URL/content;
7. existing G4 collapse/sticky/physiotherapy behavior remains intact.

Only then may the RF blocker be marked production-smoke-verified. This remains distinct from pilot validation.
