# SLICE_PLAN_CURRENT.md — G-4 RF Authorized Gateway Hotfix v1

> **STATUS:** IMPLEMENTED / TESTED / REVIEWED / MERGED / AUTO-DEPLOYED — PRODUCTION CONFIG + SMOKE PENDING.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** Clinical Excellence workspace + cross-module Clinic Utilities.
> **Slice ID:** `M01-G4-RF-AUTH-GATEWAY-HOTFIX-v1`.
> **Original production base:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **Implementation branch:** `fix/module01-g4-rf-auth-gateway-2026-09-02`.
> **Exact tested runtime head:** `29140a6cd4c9f57b454daa6e4a2883ec0345b53f`.
> **Final PR head:** `63ad2f5f392bc4d6dadec84704757ba4520ea83f`.
> **Release PR:** `#73`.
> **Merge SHA / current production code source:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **Render auto-deploy:** `dep-dac43leq1p3s73a04s00` — `live`.
> **Runtime writer:** NONE.
> **Config/secret authority:** NONE unless separately granted.

---

# 1. Trigger / failure class

Original G-4 PR #72 was merged/deployed and workspace smoke passed collapse/expand, sticky summary and physiotherapy navigation. Direct RF navigation reached the intended existing RF service but returned:

```text
{"detail":"Απαιτείται εξουσιοδοτημένη πρόσβαση."}
```

Causal class:

```text
CROSS-SERVICE AUTHENTICATION INTEGRATION GAP
```

The Osteoporosis browser session did not authorize the separately protected RF service. This was not evidence of RF PDF/template/business-rule failure and was not solved by weakening RF authorization.

---

# 2. Cross-repository authority boundary

The RF runtime remains owned by `athpapachr-cmd/ortho-reception-backend-v2`, governed through `athpapachr-cmd/ortho-reception-ops`.

This hotfix does **not** change:

```text
ortho-reception-backend-v2 runtime
RF backend authorization contract
RF backend configuration/secrets
RF templates/PDF engine/persistence
```

The existing RF service remains source of truth for form HTML/content, history/persistence, validation, official PDF templates/coordinates and generated PDF output.

---

# 3. Released architecture

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

The browser never receives the RF credential and does not require an RF-service cookie.

The gateway owns only bounded transport/auth adaptation.

---

# 4. Final route contract

Browser-facing routes:

```text
GET  /clinical/clinic-utilities/rf
POST /clinical/clinic-utilities/rf/history
POST /clinical/clinic-utilities/rf/create
GET  /clinical/clinic-utilities/rf/pdf/{application_id}
```

Fixed upstream mappings:

```text
GET local root
→ GET upstream /rf

POST local /history
→ parse exactly three form-urlencoded fields
→ GET upstream /rf/history with those named params

POST local /create
→ POST upstream /rf/create with raw multipart body + Content-Type

GET local /pdf/{id}
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

Upstream origin is fixed in code:

```text
https://ortho-reception-backend-v2.onrender.com
```

The gateway injects only the server-side `X-RF-Key`, `Accept`, and required `Content-Type`. Missing key fails closed with local 503 before upstream access.

---

# 6. POST-only history privacy correction

Independent review identified that a browser-facing local `GET /history?...` would create an additional Osteoporosis URL/access-log surface containing identity and GeSY identifiers.

The final runtime therefore uses:

```text
browser
→ POST /clinical/clinic-utilities/rf/history
→ application/x-www-form-urlencoded
→ maximum body 4096 bytes
→ accepted fields only:
   identity_number
   gesy_number
   application_location
```

The gateway converts that request server-to-server to the existing upstream GET contract.

Consequences:

- identifiers do not appear in the Osteoporosis browser URL/history;
- local GET `/history` is not implemented;
- duplicate values, oversized fields/body, malformed encoding and wrong content type fail closed;
- unexpected fields are not forwarded upstream.

---

# 7. Form adaptation / failure behavior

The upstream RF form remains authoritative. The gateway adapts only the expected create/history transport seams. If either expected seam changes, the gateway fails closed with sanitized 502 rather than serving a silently broken or privacy-unsafe form.

Failure contract:

```text
missing RF_GATEWAY_ACCESS_KEY → 503, no upstream request
upstream 401/403             → sanitized 502
upstream 5xx                 → sanitized 502
network failure              → sanitized 502
timeout                      → sanitized 504
invalid PDF application id   → rejected; no arbitrary upstream path
```

Create bodies are bounded to 24 MiB. Successful generated-PDF redirects are rewritten to the protected same-origin gateway PDF route.

---

# 8. Implemented surface

Runtime/test changes were bounded to:

```text
clinic_utilities/rf_gateway.py                    NEW
main.py                                           include RF gateway router
static/baseline-audit/g4-workspace-ergonomics.js  RF link → same-origin gateway
test_g4_rf_auth_gateway.py                        NEW focused gateway regression
test_g4_workspace_ergonomics.js                   updated ownership/navigation assertions
.github/workflows/g3-guidance-summary-tests.yml    focused RF Python gate
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

# 9. Exact executable evidence

Pre-PR exact tested runtime head:

```text
29140a6cd4c9f57b454daa6e4a2883ec0345b53f
```

Workflow:

```text
G3 guidance salience longitudinal summary
run 33640110048
SUCCESS
```

Final PR head:

```text
63ad2f5f392bc4d6dadec84704757ba4520ea83f
```

Final-head PR-triggered checks:

```text
CU-1 focused tests
run 33648758303
SUCCESS

G3 guidance salience longitudinal summary
run 33648758293
SUCCESS
```

The final G3/G4 gate passed JavaScript syntax, RF gateway Python syntax, authenticated-gateway regression, POST-only history privacy regression, G4 workspace regression, G3 salience/summary/visibility gates, frozen G2 contract/runtime, G1 core/wiring/UI/WHY-NOW and C1 Finish/finalization regressions.

No runtime/test/workflow change occurred after that final checked head before merge.

---

# 10. Release evidence

PR #73 was squash-merged with expected-head guard at:

```text
8aa8b38e3fa9a8f8ba0618868b452b1835be0d47
```

Render auto-deployed that exact commit without a manual deploy:

```text
deploy  dep-dac43leq1p3s73a04s00
trigger new_commit
status  live
```

This proves code deployment only. It does not prove production RF authorization/usability because `RF_GATEWAY_ACCESS_KEY` is not configured or verified on the Osteoporosis service.

---

# 11. Completion matrix

```text
HOTFIX DESIGN                         COMPLETE
HOTFIX IMPLEMENTED                    YES
HOTFIX TESTED                         YES
HOTFIX EXACT-HEAD REVIEW              PASS
HOTFIX PR                             #73
HOTFIX PR FINAL-HEAD CHECKS           PASS
HOTFIX MERGED                         YES
HOTFIX MERGE SHA                      8aa8b38e3fa9a8f8ba0618868b452b1835be0d47
HOTFIX DEPLOYED                       YES — dep-dac43leq1p3s73a04s00 LIVE
RF_GATEWAY_ACCESS_KEY CONFIGURED      NO / NOT VERIFIED
PRODUCTION-SMOKE-VERIFIED             NO
G-4 PILOT-VALIDATED                   NO
ACTIVE RUNTIME WRITER                 NONE
```

`IMPLEMENTED != TESTED != MERGED != CONFIGURED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED != PILOT-VALIDATED` remains explicit.

---

# 12. Current config/smoke gate

The release code is merged and live. The only unresolved product blocker in this slice is production configuration + end-to-end verification.

Next possible sequence requires separate product-owner authority for the production secret/config mutation:

```text
securely configure RF_GATEWAY_ACCESS_KEY on Osteoporosis Render
→ verify service remains healthy
→ authenticated production smoke:
   form render
   POST-only history lookup
   create/PDF path
   credential non-exposure
   inherited G4 workspace behavior
```

No RF credential value belongs in GitHub, chat output or client-side code.

Until separately authorized:

```text
NO production secret/config mutation
NO manual deploy
NO mutation of ortho-reception-backend-v2
NO claim of production-smoke completion
NO pilot claim
```

Only after successful production smoke may the RF blocker and G-4 whole-slice production-smoke gate be closed.
