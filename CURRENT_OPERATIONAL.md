# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-4 RF AUTH HOTFIX MERGED / AUTO-DEPLOYED; PRODUCTION CONFIG + RF SMOKE PENDING.
> **Updated:** 2026-09-02 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **G-4 original release PR:** `#72` — squash merged as `338830340f6fed2ae1a3f08f6fdb0b8059932a66`.
> **G-4 original Render deploy:** `dep-dac27kojo6nc739biu80` — prior product-owner smoke established workspace behavior and isolated RF auth failure.
> **RF auth hotfix PR:** `#73` — squash merged.
> **RF auth hotfix final PR head:** `63ad2f5f392bc4d6dadec84704757ba4520ea83f`.
> **RF auth hotfix merge SHA:** `8aa8b38e3fa9a8f8ba0618868b452b1835be0d47`.
> **RF auth hotfix Render auto-deploy:** `dep-dac43leq1p3s73a04s00` — `live`, exact merge source, trigger `new_commit`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only closeout branch only; no runtime writer.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **PR authority:** consumed.
> **Merge authority:** granted by product owner and consumed for PR #73.
> **Production config/secret authority:** NONE.
> **Manual deploy authority:** NONE / not required.

---

# 1. G-4 production state

Original G-4 workspace release remains verified as follows:

```text
G-4 DESIGN                         COMPLETE
G-4 IMPLEMENTED                    YES
G-4 TESTED                         YES
G-4 ORIGINAL PR                    #72
G-4 ORIGINAL MERGED                YES
G-4 ORIGINAL DEPLOYED              YES
G-4 WORKSPACE SMOKE                PASS
G-4 WHOLE-SLICE PRODUCTION-SMOKE   NO — RF authorized workflow still requires config + smoke
G-4 PILOT-VALIDATED                NO
```

The original product-owner smoke passed collapse/expand, independent current-flow collapse, sticky patient summary, physiotherapy utility navigation and navigation to the intended RF service. The direct external RF path failed authorization with `Απαιτείται εξουσιοδοτημένη πρόσβαση.`, correctly isolating a cross-service authentication integration gap.

---

# 2. RF auth hotfix runtime boundary

Released architecture:

```text
authenticated Cockpit browser
→ /clinical/clinic-utilities/rf
→ existing ClinicalCookieMiddleware / clinical-key gate
→ Osteoporosis server-side RF gateway
→ X-RF-Key from server-only RF_GATEWAY_ACCESS_KEY
→ fixed existing RF service
```

Browser-facing routes:

```text
GET  /clinical/clinic-utilities/rf
POST /clinical/clinic-utilities/rf/history
POST /clinical/clinic-utilities/rf/create
GET  /clinical/clinic-utilities/rf/pdf/{application_id}
```

The local history route is POST-only so identity/GeSY identifiers are carried in a bounded form-urlencoded body rather than an Osteoporosis browser URL/query string. The gateway maps only the allowed fields server-to-server to the existing RF history contract.

Hard boundaries remain:

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

# 3. Exact executable / PR evidence

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

PR-triggered exact-final-head checks:

```text
CU-1 focused tests
run 33648758303
SUCCESS

G3 guidance salience longitudinal summary
run 33648758293
SUCCESS
```

The final G3/G4 workflow passed RF gateway Python syntax, authenticated-gateway regression, POST-only RF history privacy regression, G4 workspace regression, G3 salience/summary and production-visibility gates, frozen G2 contract/runtime gates, G1 core/wiring/UI/WHY-NOW and C1 authoritative Finish/finalization regressions.

No runtime/test/workflow change followed the final checked PR head before squash merge.

---

# 4. Merge / deploy evidence

PR #73 was squash-merged with expected-head protection:

```text
merge SHA 8aa8b38e3fa9a8f8ba0618868b452b1835be0d47
```

Fresh GitHub verification confirmed `main` at exactly that SHA.

Render then auto-deployed from the new commit without a manual trigger:

```text
deploy  dep-dac43leq1p3s73a04s00
source  8aa8b38e3fa9a8f8ba0618868b452b1835be0d47
trigger new_commit
status  live
```

This proves deployment of the hotfix code. It does **not** prove authorized RF usability because the required server-side RF credential has not been configured or verified in the Osteoporosis service.

---

# 5. Current hotfix state matrix

```text
HOTFIX DESIGN                         COMPLETE
HOTFIX IMPLEMENTED                    YES
HOTFIX TESTED                         YES
HOTFIX EXACT-HEAD REVIEW              PASS
HOTFIX PR                             #73
HOTFIX PR CHECKS                      PASS @ 63ad2f5f392bc4d6dadec84704757ba4520ea83f
HOTFIX MERGED                         YES
HOTFIX MERGE SHA                      8aa8b38e3fa9a8f8ba0618868b452b1835be0d47
HOTFIX DEPLOYED                       YES — dep-dac43leq1p3s73a04s00 LIVE
RF_GATEWAY_ACCESS_KEY CONFIGURED      NO / NOT VERIFIED
HOTFIX PRODUCTION-SMOKE-VERIFIED      NO
HOTFIX PILOT-VALIDATED                NO
ACTIVE RUNTIME WRITER                 NONE
```

Hard distinction remains:

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

---

# 6. Cross-repository / secret constraint

The RF backend remains independently governed by `athpapachr-cmd/ortho-reception-ops`. PR #73 did not mutate RF backend runtime, authorization code or secrets/configuration.

Production RF enablement still requires a separately authorized server-side configuration on the Osteoporosis Render service:

```text
RF_GATEWAY_ACCESS_KEY
```

It must contain the existing RF access credential already accepted by the RF service. No credential value belongs in GitHub, client-side code or chat.

Until that key is configured, the gateway is designed to fail closed rather than bypass RF authorization.

---

# 7. Exact next action / STOP gate

Current exact next possible action requires **separate product-owner production config/secret authority**:

```text
securely configure RF_GATEWAY_ACCESS_KEY on the Osteoporosis Render service
→ verify resulting production service state
→ bounded authenticated RF production smoke
   form render
   POST-only history
   create/PDF path
   credential non-exposure
→ only then consider G-4 production-smoke-verified
```

Forbidden without new explicit authority:

```text
NO production secret/config mutation
NO manual Render deploy
NO mutation of ortho-reception-backend-v2
NO claim of RF production-smoke verification
NO pilot claim
```

C2 remains separately implemented/tested and unreleased. PR-1/PR-2 and later rich physiotherapy work remain outside this hotfix.
