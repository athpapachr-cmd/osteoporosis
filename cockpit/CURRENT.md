# Cockpit Home CURRENT

> **STATUS:** ACTIVE / IMPLEMENTATION AUTHORIZED.
> **Workstream:** Clinical Excellence Cockpit Home v1.
> **Branch:** feat/cockpit-home-v1-2026-09-27.
> **Base main:** 88ad125f0a25a471b0151eeb26e68b8b8a93c84f.
> **Root writer lock:** unchanged; PR-1 Heidi-first transcript capture remains the repo-wide CURRENT_OPERATIONAL owner.
> **Overlap:** none; this slice is limited to global Cockpit navigation/home presentation and Osteoporosis sidebar cleanup.

## Product-owner decisions

- / must open the global Cockpit Home, not the Osteoporosis module.
- Osteoporosis is Module 01 / proving ground, not the whole Cockpit.
- Osteoporosis sidebar must contain only Module-01 navigation.
- Global Clinic Utilities belong on Cockpit Home.
- The duplicate physiotherapy referral entry must collapse to one global tool entry.
- The top-level Heidi AI navigation item must be removed from Osteoporosis; Heidi remains an encounter-capture capability inside the Module-01 workflow and future reusable Core.
- The separate Reception/calls dashboard remains owned by ortho-reception-backend-v2; Cockpit Home links to it rather than duplicating its implementation.
- Calendar/Cal.com reason ingestion is a separate follow-up integration slice because its source-of-truth boundary spans repositories.

## Home v1 information architecture

Cockpit Home
- Today
  - Clinical Calendar
- Clinical Modules
  - Osteoporosis · Module 01 · Active
- Learning & Improvement
  - Clinical Learning Hub
- Clinic Utilities
  - Παραπεμπτικό Φυσιοθεραπείας
  - Αναρρωτική άδεια
  - Ιατρικές εκθέσεις
  - Ραδιοκύματα
- Reception
  - Call / Reception dashboard (external owner)

Module 02+ appear as future placeholders only; no fake runtime is implied.

## Safety / scope

- no patient/clinical data model change;
- no Cal.com/Setmore/Digital Secretary mutation in this slice;
- no PR-1 transcript runtime mutation;
- no duplicate copy of the Reception dashboard;
- no new browser patient persistence;
- root Clinical Auth boundary remains unchanged.

## Exact next action

Implement Home v1 + Osteoporosis sidebar cleanup + deterministic navigation tests. Then run focused/current regression gates and checkpoint the tested head before release PR.
