# CURRENT.md — Physio Referral product track

> **STATUS:** STEPS 1–4 FROZEN; STEP 5 PROTOTYPE TECHNICALLY PASSED; STEP 6A QUALIFIER REFINEMENT TECHNICALLY PASSED.
> **Updated:** 2026-09-11 Asia/Nicosia.
> **Branch:** `feat/physio-referral-knee-oa-prototype-v1-2026-09-11`.
> **Main:** `d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37`.
> **Step-4 parent:** `4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6`.
> **Step-5 tested head:** `6595bf4cc41388dbd796f9ba5b53ae7c49bafdee`.
> **Step-6A tested substantive head:** `243095ca9545bd2f96be8986520aeae8c3551c27`.
> **Writer:** NONE. Root `CURRENT_OPERATIONAL.md` remains sole operational authority.
> **Production registration / PR / merge / deploy:** NONE.

## What exists now

A runnable synthetic Knee-OA prototype with a small routine surface plus progressive clinical depth. The actual CU-1 engine remains validation/safety authority; the prototype runs only on `127.0.0.1` and stores no patient draft.

Routine phenotype now supports compact on-demand refinement:

```text
Pain       → anatomic location, including pes-anserine region
Stiffness  → morning / after inactivity → morning duration when relevant
Weakness   → generic / objective / quadriceps + optional visible atrophy
More       → examination findings including FFD, effusion, extension lag, focal tenderness
```

Clinical summary text adapts without treatment auto-selection. Pes-anserine location does not become bursitis; generic weakness does not become objective weakness; stiffness does not become fixed flexion deformity; >30-minute morning stiffness generates a non-blocking review clue rather than a treatment rule.

## Proven technical results

Step-6A substantive workflow `34627436841` at `243095ca...` passed:

```text
15 / 15 existing real-CU1/HTTP tests
8 / 8 qualifier projection tests
15 inherited exact-output fixtures inside existing suite
12 / 12 existing Chromium tests
6 / 6 qualifier Chromium tests
54 source-position Greek display summaries
packaged dependency-closure smoke
scope + syntax gates
```

Step-5 frozen outputs remain unchanged when no new qualifier is selected. The result record is `KNEE_OA_STEP6A_QUALIFIER_REFINEMENT_RESULT.md`.

## Not yet proven

```text
product-owner usability / clinical-copy acceptance       PENDING
actual iPhone Safari / VoiceOver                          NOT TESTED
complete accessibility / measured contrast audit          NOT PERFORMED
independent clinical / physio / UX / commercial review     NOT PERFORMED
independent source-to-claim audit                          NOT COMPLETED
willingness to pay / commercial pilot                      NOT VALIDATED
production release / public preview                       NOT AUTHORIZED
```

## Exact next action

Continue **Step 6 product-owner trial using synthetic cases**, now against the refined qualifier candidate. Judge whether the extra depth is actually useful without slowing the routine flow, whether the wording reads naturally, and especially what should be removed.

Do not add more clinical fields or a second diagnosis before that evidence-from-use step. Independent multi-axis review follows product-owner acceptance of the functional candidate.
