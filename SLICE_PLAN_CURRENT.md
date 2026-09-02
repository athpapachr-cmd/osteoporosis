# SLICE_PLAN_CURRENT.md — G-4 RF Authorized Gateway Hotfix v1

> **STATUS:** APPROVED / FROZEN — IMPLEMENTATION AUTHORIZED; RELEASE HOLD.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Module:** Clinical Excellence workspace + cross-module Clinic Utilities.
> **Slice ID:** `M01-G4-RF-AUTH-GATEWAY-HOTFIX-v1`.
> **Production base:** `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **Branch:** `fix/module01-g4-rf-auth-gateway-2026-09-02`.
> **Product-owner direction:** 2026-09-02 Asia/Nicosia — `προχώρα bounded G4 RF auth hotfix`.
> **Runtime writer:** this branch, osteoporosis-only RF gateway scope.

---

# 1. Trigger / first real failure

G-4 PR #72 merged and Render auto-deploy `dep-dac27kojo6nc739biu80` reached live at exact merge source `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.

Product-owner production smoke passed the G-4 workspace behavior and reached the intended RF service, but the RF target returned:

```text
{"detail":"Απαιτείται εξουσιοδοτημένη πρόσβαση."}
```

Observed causal chain:

```text
Cockpit authenticated on Osteoporosis origin
→ browser opens external https://ortho-reception-backend-v2.onrender.com/rf
→ browser has no RF-service rf_session/access credential
→ existing RF authorization correctly rejects with HTTP 401
```

Generalized failure class:

```text
CROSS-SERVICE AUTHENTICATION INTEGRATION GAP
```

This is not evidence of RF PDF/template/business-rule failure and must not be solved by weakening RF authorization.

---

# 2. Canonical/repository authority constraint discovered during investigation

The RF service runtime repository points to `athpapachr-cmd/ortho-reception-ops` as its mandatory control plane.

Fresh six-canonical ops bootstrap at:

```text
ortho-reception-ops/main
a9b637e47cd91c6d421ec8a7c6fcb12f8ebf1044
```

found active Backend PR #131 / Call Causal Trace v1 on frozen production base `92ae6c8857a5abf0967926772840cc141b731727`, with no config/secret/deploy authority.

Therefore the hotfix may inspect but MUST NOT mutate:

```text
athpapachr-cmd/ortho-reception-backend-v2
its Render configuration/secrets
its RF authorization contract
```

This discovery rules out the initially considered cross-repository token-handoff endpoint for the current slice.

---

# 3. Corrected bounded architecture

## 3.1 Chosen design — Osteoporosis-side authenticated reverse gateway

```text
browser with valid clinical_session
        ↓
ClinicalCookieMiddleware
        ↓
/clinical/clinic-utilities/rf[/...]
        ↓
existing X-Clinical-Key dependency
        ↓
RF gateway (Osteoporosis server only)
        ↓   server-side X-RF-Key: <RF_GATEWAY_ACCESS_KEY>
fixed upstream RF service
        ↓
existing /rf form/history/create/pdf implementation
```

The browser never receives the RF credential and never needs the RF service's cookie.

The RF backend remains source of truth for:

- HTML/template content;
- indications/consumables/application locations;
- repeat-use validation;
- upload validation;
- RF persistence/history;
- official PDF templates/coordinates;
- generated PDF output.

The gateway is transport/auth adaptation only.

## 3.2 Gateway routes

```text
GET  /clinical/clinic-utilities/rf
GET  /clinical/clinic-utilities/rf/history
POST /clinical/clinic-utilities/rf/create
GET  /clinical/clinic-utilities/rf/pdf/{application_id}
```

Allowed upstream mapping is fixed one-to-one:

```text
/clinical/clinic-utilities/rf
→ /rf

/clinical/clinic-utilities/rf/history
→ /rf/history

/clinical/clinic-utilities/rf/create
→ /rf/create

/clinical/clinic-utilities/rf/pdf/{id}
→ /rf/pdf/{id}
```

No arbitrary path or host proxying is allowed.

---

# 4. Security/data invariants

```text
NO RF secret in JS/HTML/query string/browser history
NO forwarding of clinical_session cookie upstream
NO forwarding of browser Authorization/X-Clinical-Key upstream
NO user-controlled upstream URL
NO broad open reverse proxy
NO RF payload persisted into osteoporosis encounter state
NO RF template/rule duplication
NO auth weakening on RF service
```

Server-side credential source:

```text
RF_GATEWAY_ACCESS_KEY
```

It is required at runtime and fail-closed when absent.

The credential is injected only as upstream `X-RF-Key`. It is never returned in response body, response header or logging by gateway code.

The upstream origin is a code constant:

```text
https://ortho-reception-backend-v2.onrender.com
```

This prevents environment/user input from turning the feature into an SSRF proxy.

The gateway forwards only the RF request content necessary for the mapped operation:

- history: named query parameters only;
- create: raw multipart body + Content-Type only;
- PDF: application id after strict existing-compatible validation.

Browser cookies and auth headers are not proxied.

---

# 5. Form-route adaptation

The existing upstream RF form contains absolute references such as:

```text
action="/rf/create"
fetch('/rf/history?...')
```

The gateway may rewrite only quoted RF route references in the returned HTML:

```text
"/rf/..." / '/rf/...'
→
"/clinical/clinic-utilities/rf/..." / '/clinical/clinic-utilities/rf/...'
```

This is transport adaptation, not a copied/template-owned RF form. All form content still comes live from the RF source of truth.

---

# 6. Response/failure behavior

## Upstream success

- HTML remains `text/html` with `Cache-Control: no-store`.
- history JSON/status propagates without credential headers.
- create upstream `303 /rf/pdf/{id}` is rewritten to the same-origin protected gateway PDF route.
- PDF bytes, content type and content disposition propagate.

## Fail closed

```text
RF_GATEWAY_ACCESS_KEY absent
→ 503 local service-unavailable error
→ no upstream request

upstream 401/403/503 auth/config failure
→ sanitized 502/503 gateway error
→ no upstream secret/config detail exposed

network/timeout
→ sanitized 502/504
```

Upstream clinical/form validation errors may be returned because they are required user-facing RF workflow feedback, but credential/config internals must not be exposed.

A bounded request-body limit is retained around the existing 20 MB RF PDF upload requirement so the gateway does not create an unbounded memory ingress.

---

# 7. Alternatives considered

## A. Put RF key in browser URL/JavaScript — REJECTED

Would expose a long-lived credential in public JS, page source, browser history, logs or referrers.

## B. Add short-lived handoff endpoint to RF backend — ARCHITECTURALLY VALID, CURRENTLY BLOCKED

Would allow direct RF-domain session establishment, but requires mutation/config in a repository currently frozen by independent Ortho-Reception governance. It is therefore not authorized in this hotfix.

## C. Osteoporosis-side reverse gateway — SELECTED

Advantages:

- no RF backend mutation;
- no client-side secret;
- reuses existing clinical browser auth;
- reuses existing RF authorization and business implementation;
- narrow fixed route map;
- one reversible osteoporosis-only release.

Trade-off:

- Osteoporosis service must hold one server-side copy of the existing RF access credential;
- RF traffic traverses the Osteoporosis service;
- form route references need narrow transport rewriting.

For the current authority constraints this is the smallest coherent secure design.

---

# 8. Planned implementation seams

```text
NEW clinic_utilities/rf_gateway.py
  → fixed upstream transport
  → existing _require_clinical_key dependency
  → HTML route rewrite
  → history/create/pdf mapped forwarding
  → sanitized failure behavior

main.py
  → include RF gateway router

static/baseline-audit/g4-workspace-ergonomics.js
  → RF_URL becomes /clinical/clinic-utilities/rf
  → new-tab isolation retained

test_g4_rf_auth_gateway.py
  → focused FastAPI transport/auth/security regressions

test_g4_workspace_ergonomics.js
  → same-origin gateway target assertion
```

No schema/database/evidence/clinical-rule changes.

---

# 9. Acceptance evidence

Focused executable evidence must prove:

1. missing clinical auth is rejected by the existing protected-route contract;
2. missing `RF_GATEWAY_ACCESS_KEY` fails closed before transport;
3. upstream receives `X-RF-Key` and does not receive clinical cookie/browser auth;
4. HTML route references are rewritten and secret is absent from HTML;
5. history forwards only allowed query fields;
6. multipart create forwards body/content type and does not follow upstream redirect automatically;
7. upstream RF PDF redirect becomes same-origin protected redirect;
8. PDF content/type/disposition propagate;
9. upstream auth/network failures are sanitized;
10. application id/path is bounded and cannot become arbitrary upstream path;
11. existing G-4 collapse/sticky/Clinic Utilities behavior remains green;
12. inherited directly affected application import/startup sanity remains green.

Evidence must not require a real RF patient payload or identifiable patient data.

---

# 10. Release/config gate

Code implementation/test authority is granted by the product-owner hotfix direction.

Still separate:

```text
IMPLEMENTED != TESTED != PR != MERGED != CONFIGURED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED
```

After exact-head implementation/review:

```text
STOP before PR/merge/config/deploy unless separately authorized.
```

Production enablement will require the existing RF credential to be copied securely into the Osteoporosis Render service as:

```text
RF_GATEWAY_ACCESS_KEY
```

No credential value belongs in GitHub, chat output or client-side code.

---

# 11. Replan triggers

STOP/replan if implementation proves any of the following:

- RF form requires additional undocumented upstream state/cookie semantics that cannot be transported safely;
- browser must receive the RF secret for the flow to work;
- arbitrary upstream proxying becomes necessary;
- RF backend mutation becomes unavoidable;
- the existing RF workflow requires an additional unapproved stateful endpoint;
- the gateway would duplicate RF clinical/business rules/templates;
- active repository authority changes materially before completion.
