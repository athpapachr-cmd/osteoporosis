# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — C1 + G-1 MERGED / DEPLOYED / PRODUCTION SMOKE NOT YET PROVEN.
> **Updated:** 2026-08-30 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main` after release:** `a6ba9ef1719a18a48a1756bf08bbd157d448a63e`.
> **Release PR:** `#64` — SQUASH-MERGED.
> **Release merge SHA:** `a6ba9ef1719a18a48a1756bf08bbd157d448a63e`.
> **Render deploy:** `dep-daa8iv0ae00c73b7eudg` — LIVE at exact release merge SHA.
> **ACTIVE CANONICAL WRITER/LOCK:** `docs/module01-g1-release-closeout-2026-08-30` — release closeout only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.

---

# 1. Released scope

The product owner authorized release of the accepted Module-01 ancestry through:

```text
C1 authoritative Finish
+
G-1 progressive guidance foundation
+
G1-R1 history-availability integrity correction
+
G1-R2 live-DOM-over-persisted state correction
```

Explicitly outside this release:

```text
PR-1 Heidi transcript extraction
PR-2 candidate review/population
new medication-specific milestone rules
physiotherapy/RF runtime changes
real 5-case or 30-case data
```

---

# 2. Release evidence

Fresh release bootstrap began from:

```text
main = 08ecd3ab33e98d567c47042a8a1de482df6952b9
```

The complete release compare contained only accepted Module-01 canonicals/contracts, C1 finalization runtime/tests and G-1 guidance runtime/tests. No parked physiotherapy/RF runtime was included.

PR #64 exact pre-merge head:

```text
e7f400b3a99810a5667cf89899f7db91424ea253
```

At that exact PR head all relevant checks passed:

```text
g1-guidance            SUCCESS — run 33331923695
baseline-finalization   SUCCESS — run 33331923682
```

The G-1 suite includes syntax, core, wiring, R1/R2 UI-state, C1 authoritative-Finish browser and C1 server-finalization lifecycle regressions.

PR #64 was then squash-merged with exact expected head into:

```text
a6ba9ef1719a18a48a1756bf08bbd157d448a63e
```

Fresh post-merge GitHub verification confirmed that exact SHA as `main`.

---

# 3. Deployment evidence

The Render `osteoporosis` service is configured:

```text
service:    srv-d5qfk31r0fns73di596g
branch:     main
autoDeploy: yes
trigger:    commit
```

No manual duplicate deploy was triggered.

Render created automatically:

```text
deploy: dep-daa8iv0ae00c73b7eudg
commit: a6ba9ef1719a18a48a1756bf08bbd157d448a63e
trigger: new_commit
status: LIVE
```

Therefore C1 + G-1 are MERGED and DEPLOYED.

---

# 4. Production-smoke boundary

Production browser behavior is **not yet marked smoke-verified**.

The assistant execution sandbox attempted direct requests to the production hostname, but DNS resolution failed before reaching Render. This is a tooling/network limitation and is not evidence of application failure.

No authenticated production-browser evidence was obtained for:

```text
C1 authoritative Finish
→ protected server completed/amended confirmation
→ reload/reopen persistence

G-1 interactive guidance
→ dropdown / quick context
→ WHY NOW rendering
→ loaded vs unavailable longitudinal-history presentation
```

Hard rule retained:

```text
MERGED != DEPLOYED != PRODUCTION-SMOKE-VERIFIED
```

---

# 5. Current status matrix

```text
C1 IMPLEMENTED / TESTED                    YES
C1 MERGED                                  YES
C1 DEPLOYED                                YES
C1 PRODUCTION-SMOKE-VERIFIED               NO
G-1 IMPLEMENTED / TESTED                   YES
G1-R1 / G1-R2                              CLOSED / TESTED
G-1 MERGED                                 YES
G-1 DEPLOYED                               YES
G-1 PRODUCTION-SMOKE-VERIFIED              NO
PR-1 HEIDI                                 NOT IMPLEMENTED
PR-2 REVIEW/POPULATION                     NOT IMPLEMENTED
REAL 5-CASE SYSTEM-ASSISTED PILOT          NOT STARTED
30-CASE SYSTEM-ASSISTED BASELINE           NOT STARTED
MODULE 01 CLOSED                           NO
```

---

# 6. Exact next action

Do not mutate G-1 runtime further merely for release polish.

Before real pilot collection, obtain one authenticated synthetic production smoke covering:

```text
1. open/load protected synthetic patient
2. confirm G-1 guidance bootstrap and coarse visit intent/context
3. verify an explicit WHY NOW path
4. verify longitudinal-history loaded/unavailable state is truthful
5. exercise authoritative Finish
6. confirm protected completed/amended response
7. reload/reopen and confirm persisted final state
```

If this smoke passes, record `PRODUCTION-SMOKE-VERIFIED` and release the production-readiness gate for the next authorized Module-01 slice. If it fails, classify the exact defect and re-open only the bounded affected seam.

No PR-1/PR-2, taxonomy expansion, physiotherapy/RF mutation or real pilot is authorized by this release closeout alone.