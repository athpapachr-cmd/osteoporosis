# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1B — PRODUCTION SMOKE DEFECT REMEDIATED / TESTED / RELEASE HOLD
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Hotfix branch:** `fix/clinical-learning-l1b-reference-title-phi-2026-09-09`.
> **Hotfix base:** `d83977464578727c0adb8f67e6778409b6d9f97c`.
> **Hotfix reviewed head before closeout:** `94563b24fd4896c1333e0cdf3c75b2586f056adc`.
> **PR:** #84 — DRAFT / RELEASE HOLD.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Production-smoke defect

Authenticated L-1B production smoke rejected the rich Challenge export with:

```text
phone_number_like_sequence_detected @ references[1].title
phone_number_like_sequence_detected @ references[2].title
phone_number_like_sequence_detected @ references[3].title
```

Root cause: rich imports store full bibliographic citations in `LearningReferenceV1.title`; year/volume/page/identifier number patterns were being passed through the generic phone-number-like heuristic.

This was a deterministic PHI false positive, not a schema failure and not evidence of patient identifiers.

---

# 2. Bounded remediation

Implemented only in `clinical_learning/privacy.py`:

- `references[*].title` now receives a **generic numeric phone-heuristic exclusion**;
- explicit phone/telephone/mobile phrases remain blocked;
- email, identity/GeSY, DOB and postal-address checks remain active;
- arbitrary Challenge facts, reasoning, observations and other learning text still use the generic phone-like detector.

Regression owner:

`test_clinical_learning_l1b_bibliographic_privacy.py`

Proves:

```text
bibliographic citation with year/volume/pages/PMID -> allowed
reference title with explicit phone phrase -> blocked
reference title with email -> blocked
ordinary non-reference phone-like sequence -> blocked
```

---

# 3. Verification

Exact tested runtime head before this docs-only closeout:

`94563b24fd4896c1333e0cdf3c75b2586f056adc`

- L1B regression gate run `34349129649` — **SUCCESS**.
- inherited L1 regression gate run `34349129608` — **SUCCESS**.
- L0 contract validation step — **SUCCESS**; overall L0 workflow failure is expected because its `design-only scope` guard correctly rejects this runtime hotfix.

Diff from base contains only:

```text
CURRENT_OPERATIONAL.md
clinical_learning/privacy.py
test_clinical_learning_l1b_bibliographic_privacy.py
```

No schema, patient-data, Signal/Foundation authority, production config/secret or adjacent RF/physio/CU-1 mutation.

---

# 4. Lifecycle state

```text
HOTFIX IMPLEMENTED = YES
HOTFIX TESTED = YES
FOCUSED REVIEW = PASS
PR #84 = DRAFT / RELEASE HOLD
MERGED = NO
DEPLOYED = NO
PRODUCTION SMOKE = BLOCKED UNTIL HOTFIX RELEASE
WRITER LOCK = NONE
```

Next allowed action: release decision for PR #84. After merge, rely on normal Render auto-deploy and repeat the same Advanced rich-import smoke that exposed this defect.