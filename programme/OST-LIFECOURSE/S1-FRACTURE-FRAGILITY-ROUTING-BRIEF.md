# Module-01 S1 — Fracture / fragility semantics correction routing brief

> **STATUS:** HISTORICAL PRE-CORRECTION ROUTING SNAPSHOT; S1 IS MERGED ON CURRENT `main` (AUTO-DEPLOY LIVE, PRODUCTION SMOKE NOT RUN).
> **MODE:** separate bounded correction task; not an OST-LIFECOURSE implementation.
> **ROOT LOCK:** PR-1 remains active and must not be mutated.

## Problem

Current Module-01 runtime has inconsistent semantics between a fracture event and a fragility fracture.

Verified current behaviours include:

1. structured events use fracture_history.events[].low_trauma;
2. app-core collectFractureEventsFromDom() can set risk_context.prior_fragility_fracture=true whenever any structured fracture event exists, regardless of low_trauma;
3. longitudinal summary reads event.fragility although current structured events use low_trauma;
4. longitudinal summary can derive prior_fragility_fracture from existence of any fracture event;
5. current G2 interval-fragility logic correctly requires low_trauma=yes;
6. historical vertebral-fragility logic currently treats missing/uncertain low_trauma as fragility unless explicitly no.

## Safety invariant

missing / uncertain / unspecified trauma mechanism != positive fragility fact

and

fracture fact != osteoporosis module interpretation of fragility unless criteria/evidence support that interpretation.

## Exact correction objective

Correct the existing owner paths so traumatic, unknown or uncertain fractures cannot silently become positive fragility facts merely because an event exists.

Preserve truthful historical data and existing event IDs.

## Required pre-code work

- fresh-bootstrap current main;
- identify all current readers/writers of prior_fragility_fracture, fracture_history.events, low_trauma and any stale event.fragility field;
- identify which values are authoritative versus derived;
- define backward-compatibility handling for existing records with prior_fragility_fracture=true but low_trauma absent/uncertain;
- define focused regression cases before implementation.

## Minimum regression cases

- low_trauma=yes fracture → may support fragility semantics;
- low_trauma=no fracture → must not become prior fragility fracture;
- low_trauma=uncertain → must not become positive fragility fact;
- low_trauma empty/missing → must not become positive fragility fact;
- traumatic vertebral fracture → must not be counted as vertebral fragility event merely because site=vertebral;
- current interval fracture with low_trauma=yes remains recognised;
- existing unrelated guidance and fracture-on-treatment mechanics remain preserved where semantically valid.

## Hard boundaries

- do not create a new fracture store;
- do not create a new event bus;
- do not redesign timeline/lifecourse architecture;
- do not mutate PR-1 transcript scope;
- do not resolve Product Constitution Q1–Q12;
- do not infer unknown trauma mechanism.

## Required handback

EXACT SOURCE IDENTITY
→ CURRENT READERS/WRITERS
→ CORRECTION CONTRACT
→ BACKWARD-COMPATIBILITY POLICY
→ TEST PLAN
→ IMPLEMENTATION BOUNDARY
→ STOP BEFORE CODE unless separately authorised.
