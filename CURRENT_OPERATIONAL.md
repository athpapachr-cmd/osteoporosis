# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 EVIDENCE-BACKED GUIDANCE CONTENT DESIGN-COMPLETE / RUNTIME IMPLEMENTATION NEXT.
> **Updated:** 2026-08-31 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main` before G-2 closeout:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Design branch:** `design/module01-g2-evidence-backed-guidance-2026-08-31`.
> **Design contract CI:** run `33358433732` — SUCCESS at `6a40a4a87882a4531c69ce9dff5e0ecd46011d84`.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE after design closeout; next bounded runtime branch must claim the lock.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — G-2 runtime has not started yet.

---

# 1. Closed production base

C1 authoritative Finish and G-1 progressive guidance are:

```text
IMPLEMENTED
TESTED
MERGED
DEPLOYED
PRODUCTION-SMOKE-VERIFIED
```

G-1 production readiness remains closed. The product owner confirmed visible `Γιατί τώρα:` and dynamic guidance behavior in production.

---

# 2. G-2 evidence/content design — COMPLETE

Slice:

```text
M01-G2-EVIDENCE-GUIDANCE-CONTENT-v1
```

Design artifacts:

```text
schemas/osteoporosis_guidance_evidence_registry_v1.yaml
schemas/osteoporosis_guidance_rules_v1.yaml
schemas/osteoporosis_guidance_profiles_v1.yaml
schemas/osteoporosis_therapy_milestones_v1.yaml
schemas/osteoporosis_guidance_contract_manifest_v1.yaml
M01_G2_EVIDENCE_GUIDANCE_REVIEW_V1.md
```

The machine contract passed exact contract CI and the human clinical/runtime review closed the remaining evidence-fidelity issues.

---

# 3. Frozen G-2 clinical boundaries

```text
GUIDANCE != AUTOMATIC TREATMENT DECISION
GUIDELINE A != GUIDELINE B
MISSING/UNKNOWN != NEGATIVE
SCHEDULED/PLANNED DOSE != ACTUAL DOSE
ADMINISTRATION COUNT != ELAPSED EXPOSURE
CHECKLIST GUIDANCE != SAFETY CLEARANCE
```

Specific frozen rules include:

- NOGG-specific thresholds require NOGG framework/scope guard;
- FRAX evidence rule requires explicit indication rather than initial-visit status alone;
- denosumab six-month due uses reliable exact actual dose date and remains ephemeral;
- specific >7-month NOGG rebound escalation requires ≥2 reliable actual denosumab doses;
- denosumab exit guidance does not silently write an agent choice;
- no automatic CTX 280/300 second-zoledronate rule;
- no generic Prolia 4th/8th/10th milestone;
- no automatic cardiology/vascular referral for romosozumab without approved clinic policy.

---

# 4. Runtime activation classes

The reviewed contract distinguishes:

```text
activate_v1
checklist_only
blocked_missing_structured_input
blocked_missing_reliable_linkage
design_only
```

Notably:

- `OST_G2_R15_DENOSUMAB_EXIT_CTX_FOLLOWUP` is evidence-valid but blocked for first runtime activation until the specific post-exit zoledronate actual event can be linked reliably to the denosumab-exit sequence;
- `OST_G2_R16_DENOSUMAB_EXIT_NO_CTX_OPTION` is evidence-valid but blocked until CTX-monitoring availability has an explicit structured representation.

Blank CTX data must not be interpreted as monitoring unavailable.

---

# 5. Runtime implementation boundary

The next bounded implementation should preserve the generic G-1 architecture:

```text
G-1 longitudinal projection
+ live current encounter snapshot
→ G-2 evidence context
→ pure deterministic G-2 rule evaluator
→ evidence contributions
→ merge with G-1 Visit Plan precedence
→ existing `Σημερινή ροή` / `Γιατί τώρα:` UI
```

Do not rewrite G-1 into a monolithic treatment-recommendation engine.

G1-R2 remains mandatory:

```text
LIVE CONTROL VALUE, INCLUDING BLANK
>
PERSISTED BROWSER CACHE
```

for every G-2 trigger field that has a live UI control.

---

# 6. Status matrix

```text
G-2 EVIDENCE REGISTRY                    DESIGN-COMPLETE
G-2 RULE REGISTRY                        DESIGN-COMPLETE
G-2 VISIT PROFILES                       DESIGN-COMPLETE
G-2 THERAPY MILESTONES                   DESIGN-COMPLETE
G-2 MACHINE CONTRACT CI                  PASS
G-2 HUMAN DESIGN REVIEW                  COMPLETE
G-2 RUNTIME IMPLEMENTED                  NO
G-2 RUNTIME TESTED                       NO
G-2 MERGED                               NO
G-2 DEPLOYED                             NO
G-2 PRODUCTION-SMOKE-VERIFIED            NO
PR-1 HEIDI                               NOT IMPLEMENTED
PR-2 INLINE REVIEW/POPULATION             NOT IMPLEMENTED
REAL 5-CASE SYSTEM-ASSISTED PILOT         NOT STARTED
MODULE 01 CLOSED                         NO
```

---

# 7. Exact next authorized action

The product owner instructed this session to proceed with evidence-backed osteoporosis guidance content. With the design contract now complete, the next bounded action under that authorization is:

```text
fresh verify remote main remains compatible
→ create `feat/module01-g2-evidence-guidance-runtime-2026-08-31`
   from the exact G-2 design-complete ancestry
→ claim canonical + runtime writer lock on that implementation branch
→ implement only the reviewed activation boundary
→ add focused synthetic/runtime tests + inherited G-1/C1 regressions
→ STOP at IMPLEMENTED / TESTED gate
```

Do **not** open a release PR, merge to `main`, deploy or claim production smoke without separate explicit release authorization.

Parked physiotherapy/RF work remains outside this slice.
