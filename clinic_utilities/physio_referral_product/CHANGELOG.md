# CHANGELOG.md — Physio Referral productization

> **ROLE:** append-only supporting product-design history.
> **Parent repo authority:** root six canonicals remain authoritative for repo-wide operations.

---

## 2026-09-07 — Productization track activated

The existing CU-1 Physiotherapy Referral utility was selected as the fastest initial path toward modest recurring revenue, with an initial target of approximately €9.99/month and no immediate objective beyond roughly €1,000 monthly recurring revenue.

The intended longer-term architecture is a unified Clinical Cockpit with separately activatable modules; Physio Referral is the first focused module and should remain independently usable while being architecturally compatible with later cockpit entitlements.

The product must not be positioned as a simple text generator. Subscription value is expected to come from evidence-aware guidance, reviewed updates, clinical structure, flexibility and speed.

---

## 2026-09-07 — Single-diagnosis vertical slice selected

The product owner rejected parallel development of five conditions for the first productization experiment.

First complete vertical slice:

```text
Knee Osteoarthritis only
```

The single slice must prove the reusable architecture before a second diagnosis is added.

---

## 2026-09-07 — UX contract v1 frozen for prototype

The product owner approved a minimal, modern, mobile-first interaction model inspired by direct-manipulation first-party mobile software rather than conventional medical form UX.

Frozen principles include:

- smart evidence-aware starting plan instead of blank form;
- selectable rows/direct manipulation instead of routine checkbox grids;
- progressive disclosure by clinical context;
- compact `Περισσότερα` power-user layer with active-count memory when collapsed;
- live referral projection with no routine Generate button;
- color as an important evidence-state cue, never the only cue;
- green/strong emphasis for supported recommendations;
- neutral/blue-grey for context-dependent options;
- amber for limited/insufficient evidence;
- distinct caution state for recommendation against routine use;
- grey for not-yet-assessed evidence state;
- compact contextual evidence bubbles rather than warning boxes;
- small `i` control for concise rationale/source details;
- full bibliography only on deeper explicit request;
- evidence-backed suggestions with source/year and one-tap add;
- guideline publication year kept separate from product evidence-review date;
- normal final state shown simply as `Έτοιμη` rather than a quality score or dashboard of ticks;
- typical mobile routine path targeted at approximately 5–7 meaningful taps with little or no typing.

The UX contract is design-frozen for the Knee-OA prototype. It is not implemented or tested by this milestone.

Next step is a separate Knee-OA evidence knowledge-module contract before runtime implementation.
