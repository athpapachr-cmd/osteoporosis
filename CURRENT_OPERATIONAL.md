# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** MODULE 01 — G-2 EVIDENCE-BACKED OSTEOPOROSIS GUIDANCE CONTENT — ACTIVE DESIGN / EVIDENCE REVIEW.
> **Updated:** 2026-08-31 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified remote `main`:** `5182d250e244b2ed9e086138cb3b2edcdb967e25`.
> **Active branch:** `design/module01-g2-evidence-backed-guidance-2026-08-31`.
> **ACTIVE CANONICAL WRITER/LOCK:** this ChatGPT session — G-2 evidence/content design and exact supporting canonicals only.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE — runtime activation is not started until the G-2 evidence/rule contract is frozen and reviewed.

---

# 1. Closed production base

C1 authoritative Finish and G-1 progressive guidance are IMPLEMENTED / TESTED / MERGED / DEPLOYED / PRODUCTION-SMOKE-VERIFIED.

G-1 production readiness is closed. The product owner directly confirmed visible `Γιατί τώρα:` and dynamic guidance behavior in production.

---

# 2. Active slice

```text
M01-G2-EVIDENCE-GUIDANCE-CONTENT-v1
```

Objective:

> Define the minimum evidence-backed osteoporosis guidance content needed before transcript-assisted capture and the five-case system-assisted pilot, using explicit source provenance and deterministic triggers without silently turning guidelines into automatic treatment decisions.

Current authorized work:

- verify current authoritative guideline/evidence sources;
- define source/version/freshness registry;
- define first-pass visit-profile guidance content;
- define event/safety overrides;
- define treatment-start / continuation / transition guidance content;
- define evidence-backed denosumab/time-critical timing rules only where provenance supports an exact rule;
- map rules to current G-1 domains/cards and current structured context;
- define machine-readable G-2 content contracts and synthetic acceptance fixtures;
- update G-2 canonicals.

---

# 3. Hard invariants

```text
GUIDANCE != AUTOMATIC TREATMENT DECISION
GUIDELINE A != GUIDELINE B — no silent hybridization
MISSING/UNKNOWN != NEGATIVE
SCHEDULED DOSE != ACTUAL DOSE
ADMINISTRATION COUNT != ELAPSED EXPOSURE
EXACT MILESTONE REQUIRES EXPLICIT REVIEWED SOURCE OR APPROVED CLINIC POLICY
EVENT/SAFETY OVERRIDE > ROUTINE VISIT DEFAULT
```

Every material active guidance rule must carry provenance sufficient to identify source, version/year, applicability and strength/certainty where available.

---

# 4. Explicitly out of scope during evidence-design pass

- PR-1 Heidi provider/API runtime;
- PR-2 provisional Accept/Edit/Reject population;
- real patient/transcript fixtures;
- real five-case pilot collection;
- KPI/performance feedback or Practice Review runtime;
- medication-specific rules unsupported by reviewed evidence;
- arbitrary Prolia 4th/8th/10th-dose rules;
- physiotherapy/RF mutation;
- merge/deploy of new clinical runtime before exact G-2 review.

---

# 5. Exact next action

```text
fresh evidence review
→ inspect existing G-1 domain/context seams
→ freeze G-2 source + rule + profile contracts
→ independent internal consistency/relevance review
→ only then decide whether bounded runtime activation can start under this authorization
```

If an evidence source conflicts with another framework, preserve both positions explicitly and do not manufacture a combined threshold.
