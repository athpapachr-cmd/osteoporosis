# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / active-work lock

> **STATUS:** ACTIVE OPERATIONAL AUTHORITY.
> **Updated:** 2026-08-26 Asia/Nicosia.
> **Canonical home:** `athpapachr-cmd/osteoporosis`.
> **Verified main immediately before this source-identity reconciliation:** `644aca0d3d9f704949064ad5abe80deb98da2a6e`; this reconciliation commit advances `main` once written.
> **Current major phase:** Personal Clinical Excellence foundation with a bounded Clinic Utilities detour.
> **Active slice design:** `SLICE_PLAN_CURRENT.md` — CU-1 Physiotherapy Referral v2 clinical/content design.
> **Supporting detour plan:** `CLINIC_UTILITIES_PLAN.md`.
> **Frozen cervical profile:** `clinic_utilities/physio_profiles/cervical_v1_1.md`.
> **Frozen lumbar profile:** `clinic_utilities/physio_profiles/lumbar_v1_1.md`.
> **Frozen shoulder profile:** `clinic_utilities/physio_profiles/shoulder_v1_1.md`.
> **Frozen elbow profile:** `clinic_utilities/physio_profiles/elbow_v1_1.md`.
> **ELBOW FREEZE PR:** PR #41 squash-merged as `7fb085265e5c32a002c55c9f1c3046043ecaa49f`.
> **POST-MERGE CANONICAL ALIGNMENT:** `2c48bff35aa7128e6f360e17f31e99fd6432ac71` → `c0996601e65a42cebcc899aaff410fb87c152bab` → `644aca0d3d9f704949064ad5abe80deb98da2a6e` before this reconciliation.
> **ACTIVE CANONICAL WRITER/LOCK:** NONE.
> **ACTIVE RUNTIME WRITER/LOCK:** NONE in this repository.
> **RUNTIME IMPLEMENTATION:** NOT AUTHORIZED for CU-1; design only.
> **PR-1 TRANSCRIPT SLICE:** intentionally paused, preserved at `archive/slices/PR1_TRANSCRIPT_INTAKE_V3.md`.

This file is the sole owner of operational **NOW**. Do not infer mutation authority from chat history.

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
elbow_v1_1 = FROZEN
```

---

# 4. Elbow v1.1 — closed freeze

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

- radial tunnel syndrome remains secondary/coexisting context rather than a default primary pathway;
- olecranon bursitis is not a routine physiotherapy referral; infection safety/context remains available;
- postoperative elbow remains a rare advanced/future-access route rather than a default MVP pathway;
- distal triceps and anconeus remain rare selectable myotendinous entities;
- anconeus epitrochlearis remains distinct from ordinary anconeus pain/injury and is not automatically pathological;
- elbow fractures route to the shared fracture/post-immobilization profile.

Neural distinction:

```text
radial tunnel pain-predominant presentation
!=
PIN/supinator motor-neuropathy pathway
```

Adjunct decisions:

```text
manual therapy / soft tissue → optional
dry needling → optional + competence safeguard
acupuncture → optional
ESWT → optional evidence-sensitive adjunct for lateral/medial epicondylalgia
counterforce/wrist orthosis → optional short-term/activity-specific support
therapeutic ultrasound → not standard evidence-backed treatment
```

No runtime behavior changed.

---

# 5. Exact next action

```text
1. product owner selects the next remaining CU-1 regional profile
2. use the same taxonomy/findings/safety/goals/rehab/evidence method
3. likely next practical candidate is wrist / hand if product owner confirms
4. continue CU-1 design only
```

---

# 6. Explicitly forbidden now

```text
WRITE CU-1 production runtime code
WRITE PR-1 transcript runtime code
AUTO-PERSIST physiotherapy referrals
MUTATE RF/Secretary/Calendar/Setmore/Zadarma
COMMIT identifiable patient data
CREATE overlapping runtime writers
```

---

# 7. Handoff completeness

```text
active detour = Clinic Utilities
active slice = CU-1 Physio Referral v2 design
cervical = frozen v1.1
lumbar = frozen v1.1
shoulder = frozen v1.1
elbow = frozen v1.1
elbow PR = #41 merged
canonical writer = none
runtime writer = none
runtime implementation = unauthorized
next action = product owner selects next regional profile
```
