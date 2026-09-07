# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** RF v2 PRODUCTION-SMOKE-VERIFIED / CLOSED FOR NOW — CLINICAL LEARNING HUB L-0 CANONICAL ON MAIN / DESIGN FREEZE ACTIVE
> **Updated:** 2026-09-07 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Latest runtime-bearing/design merge on `main`:** `6fa2099647513e4a6b0e71ec30eb5275164626ed` — PR #78 squash merge.
> **Render deploy for #78:** `dep-dafdb695efls73anlt6g` — LIVE.
> **Current slice:** `CORE-LEARNING-HUB-L0-2026-09-07`.
> **Detailed design:** `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **ACTIVE CANONICAL WRITER/LOCK:** post-merge docs closeout only; NONE after closeout merge.
> **RF runtime authority:** NONE — RF is closed unless new material evidence appears.
> **Learning design authority:** GRANTED / ACTIVE for L-0 contract freeze.
> **Learning runtime implementation authority:** NONE.
> **Production config/secret authority:** NONE.

---

# 1. RF v2 — production closed for now

Native RF v2 is released through:

```text
PR #75 — native Category-A A.1/A.2 workflow
PR #76 — unilateral target + derived location + medication parser/capacity corrections
PR #77 — imaging-attachment semantic guard
```

Product-owner authenticated production smoke has passed for:

```text
clinical authentication/session
native RF UI
Category A / A.1 / A.2
server-side doctor/product config
single unilateral target
system-derived exact location for fixed indications
0..3 medication capacity
Narox / Melox / Panadol / Parcoten / Tramadex flow
A.1 official PDF generation
A.2 continuation flow
imaging append
obvious laboratory PDF rejection
real imaging PDF path
ambiguous/poorly extractable imaging confirmation fallback
```

Lifecycle:

```text
RF v2 IMPLEMENTED                     YES
TESTED                                YES
MERGED                                YES
DEPLOYED                              YES
PRODUCTION-SMOKE-VERIFIED             YES
PILOT-VALIDATED                       NO
```

RF is not an active workstream.

---

# 2. RF signature boundary

The official PDF still contains a clinician-signature area.

Current product/security decision:

```text
automated signature image in public repository  FORBIDDEN
manual / external signing step                   ACCEPTED FOR NOW
```

A signature PNG/SVG must not be committed to the public repository. Future online signing, if needed, requires a separate security/e-signature design using protected private storage or an appropriate signing mechanism. This is deferred work, not an RF release blocker.

Reopen RF only for:

```text
new authoritative form change
material production defect
safety/data-integrity defect
explicit new workflow requirement
```

OCR/vision enhancement for poorly encoded imaging reports is also deferred; the current explicit clinician-confirmation fallback is accepted.

---

# 3. Clinical Learning Hub is now canonical design direction

PR #78 merged the approved Clinical Learning Hub design into `main`.

Core architecture:

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

The Hub keeps three instruments distinct:

```text
FOUNDATION MAP
what is structurally understood

CLINICAL CHALLENGE
controlled novel-case reasoning

DAILY REAL-CASE REVIEW
actual encounter performance under real constraints
```

They may feed the same Signal engine but do not collapse into one composite score.

---

# 4. Active L-0 design/contract freeze

Slice:

```text
CORE-LEARNING-HUB-L0-2026-09-07
```

L-0 owns design/contracts only:

- `ClinicalLearningChallengeV1`;
- `LearningFactV1` / mandatory Fact Ledger;
- `DailyCaseReviewV1`;
- `FoundationDomainStateV1`;
- `LearningDueStateV1` / spaced-repetition semantics;
- challenge duplicate/revision semantics;
- PHI firewall and learning-record/patient-record separation;
- self-review-before-AI contract;
- reuse of PR-1 transcript ownership rather than a parallel transcript stack;
- Signal/root-cause integration;
- baseline-intervention boundary;
- exact L-1 persistence/API/UI owners.

Hard learning boundary:

```text
REAL PATIENT FACT
!= SYNTHETIC / PROGRESSIVE-DISCLOSURE FACT
!= CLINICIAN HYPOTHESIS
!= AI INFERENCE
!= EDUCATIONAL COUNTERFACTUAL
```

No learning artifact may write hypothetical/counterfactual/AI-inferred facts into authoritative patient storage.

---

# 5. Daily real-case review direction

Target cadence:

```text
once per clinic day
→ if >=1 eligible osteoporosis/metabolic-bone encounter has usable Heidi/approved evidence
→ Daily Case Review due
→ clinician chooses a case or accepts a transparent recommendation
→ clinician self-review BEFORE AI critique
→ Practice Review / evidence review
→ clinician Accept / Modify / Dismiss observations
→ persist reviewed structured learning artifact
→ feed Signals + Foundation evidence
```

If no eligible case exists:

```text
state = no_eligible_case
```

No case is fabricated to satisfy cadence.

Raw Heidi transcript remains ephemeral by default and Daily Case Review must reuse the protected PR-1 transcript boundary.

---

# 6. Baseline methodology invariant

Visible systematic Daily Case Review coaching is an intervention.

Default policy:

```text
before scored baseline:
  design/test learning machinery

during 30-case system-assisted baseline:
  Practice Review / Daily Case Review may run in shadow
  routine AI critique/coaching hidden by default
  safety-critical feedback remains allowed

after baseline lock:
  activate visible Daily Case Review as formal intervention
  re-measure
```

If visible daily AI coaching is intentionally used during the scored baseline, that requires explicit methodology REPLAN and cohort relabelling.

---

# 7. Exact next action

The design is now on `main`. Next work is **L-0 exact contract review/freeze**, not runtime implementation.

Sequence:

```text
review object field-level contracts
→ freeze provenance / duplicate / revision semantics
→ freeze Foundation state-transition evidence
→ freeze Daily Case Review eligibility / disposition / due-state rules
→ freeze PHI firewall and storage boundaries
→ identify exact L-1 API / database / UI owners
→ independent design review
→ L-0 COMPLETE
→ HOLD for separate L-1 implementation authority
```

Forbidden until separately authorized:

```text
NO learning runtime/database implementation
NO new external learning credential
NO patient-record writes from learning artifacts
NO raw Heidi transcript persistence
NO background cron
NO production config mutation
NO RF signature asset in repository
```

The post-merge docs closeout may advance `main` by a documentation-only descendant commit; runtime behavior remains identical to the #78 code ancestry above.
