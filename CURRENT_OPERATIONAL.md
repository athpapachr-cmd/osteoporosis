# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified base main for this elbow design:** `9b13c53d0756174368882f1f508c7525cc52ba4d`.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile on active docs branch:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **ACTIVE CANONICAL WRITER/LOCK:** docs-only branch `docs/cu1-elbow-v1-design-2026-08-26` until elbow exact-head review/merge/handoff close.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational **NOW** for the active branch. Do not infer mutation authority from chat history.

---

# 1. Product boundary

```text
Clinical Excellence Core
→ reusable platform/workspace mechanics

Clinic Utilities / Clinical Operations
→ cross-module clinician workflow tools

Osteoporosis Module 01
→ osteoporosis-specific clinical standards/audit/workflows
```

CU-1 is a bounded cross-module Clinic Utility design detour. PR-1 remains paused intact, not abandoned.

---

# 2. Physiotherapy v2 architecture

```text
clinical problem
→ important findings
→ functional limitation
→ precautions/restrictions
→ goals
→ rehabilitation direction
→ structured ReferralDraft
→ short/detailed formatter
```

Hard rules:

```text
suggested finding != examined finding
selected goal != mandatory goal
condition profile != automatic diagnosis
subjective symptom != objective deficit
special/provocation test != diagnosis
imaging finding != automatically symptomatic diagnosis
not assessed != normal
adjunct technique != default primary treatment
clinician-entered diagnosis may be carried but not inferred
```

---

# 3. Frozen regional profiles

```text
cervical_v1_1 = FROZEN
lumbar_v1_1 = FROZEN
shoulder_v1_1 = FROZEN
elbow_v1_1 = FROZEN on docs branch pending exact-head review/merge
```

---

# 4. Elbow v1.1 — product-owner-approved design

Frozen default primary pathways:

```text
E1 lateral elbow tendinopathy / lateral epicondylalgia
E2 medial elbow tendinopathy / medial epicondylalgia
E3 ulnar neuropathy at elbow / cubital tunnel
E4 PIN / supinator syndrome
E5 distal biceps tendinopathy or established partial tear — conservative pathway
E6 elbow OA / degenerative painful stiffness
E7 ligament injury / instability rehabilitation
E8 post-traumatic elbow pain/stiffness after assessed injury
```

Rare/advanced/context decisions:

- radial tunnel syndrome is uncommon and remains secondary/coexisting context rather than a default primary pathway;
- olecranon bursitis is not a routine physiotherapy referral and is removed from default primary pathways; infection safety/context remains available;
- postoperative elbow is rare and remains an advanced/future-access route rather than a default MVP pathway;
- distal triceps and anconeus are rare selectable myotendinous entities rather than top-level pathways;
- anconeus epitrochlearis is distinct from ordinary anconeus pain/injury and is not automatically pathological;
- elbow fractures route to the shared fracture/post-immobilization profile.

Neural distinction:

```text
radial tunnel pain-predominant presentation
!=
PIN/supinator motor-neuropathy pathway
```

The literature contains nomenclature overlap, so the utility preserves clinician-entered diagnosis plus actual objective motor findings rather than inferring labels.

Adjunct decisions:

```text
manual therapy / soft tissue → optional
Dry needling → optional + competence safeguard
Acupuncture → optional
ESWT → optional evidence-sensitive adjunct for lateral/medial epicondylalgia
Counterforce/wrist orthosis → optional short-term/activity-specific support
Therapeutic ultrasound → not standard evidence-backed treatment
```

No runtime behavior changed.

---

# 5. Shared fracture boundary

Elbow-region fractures such as radial head/neck, olecranon/proximal ulna, distal humerus and coronoid route to the future shared fracture/post-immobilization profile.

```text
unresolved healing/loading/ROM context
→ warning
→ no unrestricted routine rehabilitation wording
```

---

# 6. Exact next action

```text
1. exact branch-vs-main review of elbow freeze
2. open docs-only elbow freeze PR if clean
3. independent exact-head review
4. merge only if exact head remains clean
5. clear canonical writer lock and record resulting main SHA
6. product owner selects next CU-1 region
```

---

# 7. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
START next regional mutation before elbow handoff closes
CREATE overlapping runtime writers
```

---

# 8. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = frozen v1.1
elbow = frozen v1.1 on docs branch pending review/merge
canonical writer = docs/cu1-elbow-v1-design-2026-08-26
runtime writer = none
runtime implementation = unauthorized
```
