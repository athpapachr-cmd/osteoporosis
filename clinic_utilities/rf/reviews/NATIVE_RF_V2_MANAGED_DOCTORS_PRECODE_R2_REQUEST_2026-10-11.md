# Native RF v2 — managed doctors, ONE focused independent pre-code R2

**READ-ONLY · DESIGN REVIEW ONLY · no code/merge/deploy/real data.**

Repository \`athpapachr-cmd/osteoporosis\`; review design \`clinic_utilities/rf/RF_V2_MANAGED_DOCTORS_BOUND_DESIGN_2026-10-11.md\` on this exact design branch HEAD (fresh-resolve SHA). Parent is PR #146 reviewed exact HEAD \`cf69e1890dbe87d828b4d1c65482140da649809c\`. \`main\` at design activation \`8469e12e8803ab1f9a882437527ed69a59a5e9b6\`.

Fresh verify SHA/branch; read the six canonicals, PROCEDURES P1–P5 and RF CURRENT. Reuse independent JPEG P1-01 correction closure PASS (0:0:0) and existing native RF code/tests; they are **not** being re-reviewed.

**Decision:** Is this *one* bounded substitution—from two Render-fixed doctor profiles to clinician-managed protected doctor directory—correct and implementable **inside existing RF**, while JPEG remains unchanged?

Finite questions:
- R1: Existing protected RFBase/router/auth reuse; add/edit/deactivate/reactivate doctors in same UI, no second store/service or per-doctor Render config; static default backward compatibility.
- R2: Server resolves trusted doctor IDs and immutable issued-document snapshot; old records unattributed; existing A.2 prior procedure semantics unchanged; deactivate/edit do not change history; explicit-invalid fails closed.
- R3: Concurrent select/edit/deactivation and create flow; reject invalid without PDF/DB writes; authorized, validated profile changes, no extra patient data, no silent doctors.
- R4: Patient/work responsibility distinction: colleague RF creates **no automatic approval, scheduling, procedure, Home attention or clinician follow-up**. Future on-request rejected-RF revision deferred.
- R5: Exact protected UI scope, reuse prior JPEG/PDF/A1/A2/medicine evidence with no source changes; proportional synthetic tests and no unnecessary new milestone.

Return \`PRE-CODE R2 PASS | BLOCK | UNKNOWN\`, \`COMPLETE_FOR_DECLARED_SCOPE | PARTIAL\`, \`P0:P1:P2\`, smallest material findings (if any), exact HEAD and STOP. Do not implement, re-review unrelated source, merge, deploy or touch real data.
