# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** CLINICAL LEARNING HUB L-1B — PRODUCTION SMOKE DEFECT / BOUNDED PHI HOTFIX ACTIVE
> **Updated:** 2026-09-09 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Hotfix branch:** `fix/clinical-learning-l1b-reference-title-phi-2026-09-09`.
> **Hotfix base:** `d83977464578727c0adb8f67e6778409b6d9f97c`.
> **Runtime squash-merge SHA:** `f15f854bbfca727356531d7b8ea896e3aedba437`.
> **Verified live deploy before smoke defect:** `0d86cd3bd5b61a23a0d7a73da29559e35cd335e5` / `dep-daghkeek1f9s73ah0fn0` LIVE.
> **ACTIVE RUNTIME WRITER/LOCK:** THIS bounded bibliographic PHI false-positive hotfix only.
> **Production config/secret authority:** NONE.
> **Patient-data mutation authority:** NONE.
> **Raw-transcript authority:** NONE.
> **DailyCase/PracticeReview/Signal authority:** NONE.
> **Physiotherapy/CU-1/RF authority:** NONE.

---

# 1. Production-smoke defect

During authenticated L-1B production smoke, Advanced rich-episode import was rejected with:

```text
phone_number_like_sequence_detected @ references[1].title
phone_number_like_sequence_detected @ references[2].title
phone_number_like_sequence_detected @ references[3].title
```

The rich adapter intentionally maps a full bibliographic citation into `LearningReferenceV1.title`. Journal/guideline citations can contain year/volume/page/identifier number patterns that trigger the generic phone-like numeric heuristic even though they are bibliographic content.

This is a deterministic PHI-guard false positive, not a schema/import failure and not evidence of patient identifiers.

---

# 2. Bounded correction

Authorized mutation scope:

```text
clinical_learning/privacy.py
test_clinical_learning_l1b_bibliographic_privacy.py
CURRENT_OPERATIONAL.md
```

Correction rule:

- `references[*].title` gets the same **numeric phone-heuristic exclusion** already used for PMID/DOI/URL bibliographic locator fields;
- email detection, explicit identity/GeSY phrases, DOB phrases, postal-address phrases and explicit phone/telephone/mobile phrases remain active;
- the exclusion does not apply to arbitrary Challenge text, reasoning, observations, facts or resource rationales.

Required regression:

```text
bibliographic citation with year/volume/pages -> allowed
reference title containing explicit "phone: 99123456" -> blocked
ordinary non-reference phone-like sequence -> blocked
```

---

# 3. Release state

```text
L1B MERGED = YES
L1B DEPLOYED = YES
PRODUCTION SMOKE = DEFECT FOUND
HOTFIX IMPLEMENTED = NO
HOTFIX TESTED = NO
HOTFIX MERGED = NO
HOTFIX DEPLOYED = NO
WRITER LOCK = ACTIVE
```

Exact next action:

```text
implement narrow privacy-path correction
→ focused regression + inherited L1/L1B gate
→ exact-head review
→ PR / RELEASE HOLD
```

No production secret/config change, no patient-data authority and no adjacent-owner mutation is authorized.