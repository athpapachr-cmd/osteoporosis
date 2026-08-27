# Hip Physiotherapy Referral Profile v1 — SUPERSEDED DESIGN CANDIDATE

> **STATUS:** SUPERSEDED HISTORICAL CANDIDATE.
> **Parent slice:** CU-1 Physiotherapy Referral v2 design.
> **Authoritative frozen successor:** `clinic_utilities/physio_profiles/hip_v1_1.md`.
> **Product-owner review completed:** 2026-08-27.

This file records only the high-level pre-freeze candidate so obsolete keys or open decisions are not mistaken for active schema authority.

Original candidate explored:

```text
hip osteoarthritis
greater-trochanteric pain / GTPS / gluteal tendinopathy
FAIS / hip-related groin pain
symptomatic acetabular labral pathology
proximal hamstring tendinopathy
adductor-related groin pain
iliopsoas / internal snapping hip
post-traumatic hip pain/stiffness
postoperative hip rehabilitation
```

Product-owner review materially changed that candidate:

```text
hip OA → context only; not routinely referred
lateral hip / trochanteric pathway → retained, including clinician-entered trochanteric bursitis
FAIS + symptomatic labral pathology → combined into one nonarthritic intra-articular pathway
adductor-related groin pain → high-visibility routine pathway
proximal hamstring → rare/secondary
iliopsoas/internal snapping → rare/secondary
gluteal tendon tear → very rare/advanced
postoperative hip → removed from routine menu
acupuncture → excluded
dry needling → optional
ESWT → not generator-recommended; therapist-proposed use may be documented
no generic pediatric/adolescent Hip navigation group
```

Additional real-workflow needs identified during review:

```text
proximal rectus femoris / proximal quadriceps tendon injury in athletes
→ direct gateway to shared muscle/myotendinous profile

pelvic apophyseal avulsion fracture in children/adolescents, especially ASIS/AIIS
→ direct gateway to shared fracture/post-immobilization profile
```

Anatomical safeguard carried into v1.1:

```text
AIIS → rectus-femoris origin relationship
ASIS → classically sartorius-related traction
ASIS avulsion != proximal rectus-femoris injury by default
```

Do not implement from this file. Use `hip_v1_1.md` as the only active Hip/Groin clinical/content design.
