# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 PRODUCTION-SMOKE-VERIFIED / CLOSED FOR NOW — CLINICAL LEARNING HUB L-0 DESIGN ACTIVE
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Fresh verified production `main`:** `1d26195c77e186cff98086283252af2eb499dd17`.
> **Production deploy:** `dep-daeliaks728c7384f3fg` — LIVE.
> **Active design branch:** `docs/clinical-learning-hub-l0-main-2026-09-07`.
> **Current slice:** `CORE-LEARNING-HUB-L0-2026-09-07`.
> **Detailed design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE CANONICAL WRITER/LOCK:** ChatGPT — L-0 design/canonical reconciliation only until design merge completes.
> **RF runtime authority:** NONE — RF is closed for now unless new evidence demonstrates a material defect.
> **Learning design authority:** GRANTED by product owner.
> **Learning runtime implementation authority:** NONE.
> **Design PR/merge-to-main authority:** GRANTED by product owner for this canonical design step.
> **Production config/secret authority:** NONE.

---

# 1. Production truth — RF v2

Native RF v2 is now released through:

```text
PR #75 — native Category-A A.1/A.2 workflow
PR #76 — unilateral target + derived location + medication parser/capacity corrections
PR #77 — imaging-attachment semantic guard
```

Current production identity:

```text
main:
1d26195c77e186cff98086283252af2eb499dd17

Render:
dep-daeliaks728c7384f3fg
status: LIVE
```

Server-side fixed configuration is present for doctor profile and the confirmed Medikey / DIROS / Thermedico product catalog.

---

# 2. RF production smoke — PASS

Product-owner authenticated production smoke is now sufficient to close the current RF workstream.

Observed/proven in production:

```text
clinical authentication/session                    PASS
native Clinical Excellence RF UI                   PASS
Category A / A.1 / A.2                             PASS
product catalog                                     PASS
doctor profile                                      PASS
single unilateral target only                       PASS
derived exact location for fixed indications        PASS
medication capacity 0..3 per category               PASS
Narox / Melox / Panadol / Parcoten / Tramadex flow PASS
A.1 official PDF generation                         PASS
A.2 continuation flow                               PASS
uploaded imaging append                             PASS
obvious laboratory PDF rejection                    PASS
real imaging PDF path                               PASS
ambiguous/poorly extractable imaging confirmation   PASS
imaging semantic guard                              PASS
```

The clinician reports the workflow is working as intended after the final #77 deploy.

Lifecycle:

```text
RF v2 IMPLEMENTED                     YES
TESTED                                YES
MERGED                                YES
DEPLOYED                              YES
PRODUCTION-SMOKE-VERIFIED             YES
PILOT-VALIDATED                       NO / not required to resume primary Module-01 roadmap
```

No further RF refinement is active.

---

# 3. RF signature boundary

The official PDF still has a clinician-signature area. This is **not a release blocker** for the RF utility.

Current decision:

```text
automated signature image in public repository  FORBIDDEN
manual / external signing step                   ACCEPTED FOR NOW
```

A static signature PNG/SVG must not be committed to the public repository. If online signing is later required, it needs a separate security/e-signature design using protected private storage or an appropriate signing mechanism. It must not be implemented as a public version-controlled image asset.

This is deferred work, not an active RF defect.

---

# 4. Stop rule for RF

The RF workstream is closed unless one of the following occurs:

```text
new authoritative form change
material production defect
safety/data-integrity defect
new product-owner workflow requirement
```

OCR/vision enhancement for poorly encoded imaging reports is explicitly a later refinement. The current fallback — explicit clinician confirmation for ambiguous/unreadable but structurally valid imaging PDFs — is accepted.

---

# 5. Active program returns to Clinical Excellence learning/capture roadmap

The primary Module-01 objective remains:

```text
improve today's encounter
+
reduce duplicate/manual capture
+
review whether reasoning/decisions/communication were appropriate
+
improve clinician performance longitudinally
```

The next active design work is the reusable **Clinical Learning Hub**.

Approved conceptual architecture:

```text
Foundation Map
+
Clinical Challenges
+
Daily Real-Case Review
+
Signals
+
Learning Plan / spaced repetition
```

Detailed design is now being canonicalized from a fresh branch based on the current production `main`; the older planning branch is not being merged wholesale because it diverged from later RF releases.

---

# 6. L-0 current scope

Active slice:

```text
CORE-LEARNING-HUB-L0-2026-09-07
```

L-0 owns **design/contract freeze only**:

- `ClinicalLearningChallengeV1`;
- `LearningFactV1` / mandatory Fact Ledger;
- `DailyCaseReviewV1`;
- `FoundationDomainStateV1`;
- due-state / spaced-repetition semantics;
- challenge duplicate/revision semantics;
- PHI firewall and learning-record/patient-record separation;
- self-review-before-AI contract;
- reuse of the future PR-1 transcript owner rather than a parallel transcript stack;
- Signal/root-cause integration;
- baseline-intervention boundary;
- exact L-1 owner seams.

No learning runtime, API, database or background schedule is authorized by L-0.

---

# 7. Baseline methodology invariant

Visible systematic Daily Case Review coaching is an intervention.

Default sequence remains:

```text
build/test learning machinery
→ during 30-case scored system-assisted baseline:
     Practice Review / Daily Case Review may run in shadow
     routine clinician-facing AI critique hidden by default
→ baseline lock
→ activate visible daily coaching as a formal intervention
→ re-measure
```

If product owner later chooses visible daily AI coaching during the scored baseline, that requires explicit methodology REPLAN and cohort relabelling.

---

# 8. Exact next action

Current branch work:

```text
port approved detailed Learning Hub design onto fresh current main
→ reconcile TODO / phase plan / current slice / change log
→ verify docs-only diff and absence of runtime changes
→ open bounded design PR
→ squash merge design into main
→ allow normal Render auto-deploy if triggered by main commit; no manual redeploy
→ begin L-0 exact contract/design review
```

After L-0 design freeze:

```text
HOLD for separate L-1 implementation authority
```

Forbidden until separately authorized:

```text
NO learning runtime/database implementation
NO new external learning credential
NO patient-record writes from learning artifacts
NO raw Heidi transcript persistence
NO production config mutation
NO RF signature asset in repository
```
