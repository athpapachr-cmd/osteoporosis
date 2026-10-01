# Prototype 1 — Denosumab Longitudinal Adaptive Visit: Implementation Contract

> **Status:** bounded design/implementation contract, 2026-10-01 Asia/Nicosia. Runtime implementation has **not** started.  
> **Authority:** [single OST-UI synthesis](OST-PRODUCT-RECONSTRUCTION-DECISION.md), accepted R1–R4, existing G1/G2/S1 clinical meanings and root canonicals. This contract creates no medical rule and does not transfer the PR-1 writer or privacy gate.  
> **Goal:** test whether a clinician can safely prepare for, conduct and close a known patient's denosumab visit from a source-labelled longitudinal state, with the six existing Steps available as direct editors rather than a mandatory route.

## 1. Bounded vertical slice and exclusions

Prototype 1 covers **one known patient on a denosumab course** with a current appointment/visit, using synthetic or explicitly authorized protected data. It must exercise these five contexts without separate applications:

| Context | Entry behavior and required distinction |
|---|---|
| Reliable last recorded actual dose, reviewed guidance says due/on-time | Show *recorded actual date*, separately labelled *derived due*, relevant current checks and prior explicit decision. Do not auto-record an administration. |
| Reliable actual date with existing reviewed delayed-state guidance | Surface delay and its `Γιατί τώρα`/source; show the decision-changing questions and uncertainty. No new rescue/cadence logic. |
| Previous explicit stop/transition decision | Lead with transition pending/recorded state and its source, not a presumptive repeat. A successor plan is not an actual successor treatment. |
| Missing or conflicting actual date/history | Show “timing cannot be determined from available history”; expose source records and a verification action; suppress any confident timing-dependent conclusion. |
| Recorded missed/planned event or open task with uncertain continuity | Display it beside the actual chronology, marked unlinked/possibly duplicated; do not infer resolution, booking, administration or closure. |

The slice includes a compact milestone/visit trajectory, in-visit adaptive composition, a protected pre-visit summary **interface contract**, assisted clinician-confirmed historical reconciliation for the facts needed by these contexts, direct source-editor routing, and closeout/obligation visibility. It excludes new drugs, fracture reassessment (Prototype 2), new guideline thresholds, new treatment or safety clearance, automatic treatment selection, broad historical migration, Live Copilot, PR-1 changes, a second longitudinal database and production rollout.

## 2. Pre-implementation owner/interface freeze

Before a runtime writer begins, record the following contract in the bounded implementation slice and claim its non-overlapping scope in `CURRENT_OPERATIONAL.md` or obtain explicit coordination with its owner. If a required owner cannot be named, stop that portion rather than inventing one in UI code.

| Concern | Owner / interface rule |
|---|---|
| Patient and encounter identity | Existing protected patient and completed/amended encounter IDs/status. The active draft is distinct. Shared context passes IDs/status; no new patient owner. |
| Factual source | Each displayed event carries source owner, encounter/result ID where present, date precision, recorded/reviewed status and conflict/unknown state. Original rows remain accessible. |
| Treatment epoch | Existing Step-4 `treatment_episodes[]` is the proposed durable owner for any **explicitly confirmed** stable course link. Historical snapshots with no supported link stay separate. No UI-generated epoch inference. |
| Administration | Existing Step-4 `administrations[]` owns actual, scheduled, planned and missed assertions. Only a captured exact actual date is an actual action. Derived due is G2 output, never written as administration or appointment. |
| Transition / decision | Existing Step-4 transition and decision owners retain explicit stop/successor intent and final clinician choice. Option, recommendation, patient preference and final decision are distinct when captured; missing historical roles remain “not recorded.” |
| Obligation | Existing Step-4 task rows remain factual owners. A stable cross-encounter obligation link/disposition needs explicit clinician review and source. The legacy semantic tuple is not proof of continuity. Generic identity/provenance envelope may later be shared Core; clinical trigger/closure stays osteoporosis-local. |
| Investigations | Existing Step-3 and dated lab/DXA/FRAX factual owners. Prototype consumes qualified results; ambiguous dual rows show provenance/conflict and cannot silently become a single trend. |
| Clinical guidance | Existing reviewed G2 registry/executor and G1 context supply due/why-now; the visit projection only displays those outputs. No new threshold, dosing interval, rescue, discontinuation, contraindication or hard stop. |
| PR-1 candidates | Transient proposals only; separate clinician accept/reject gate before any authoritative owner update. Identifiable transcripts remain blocked by their own privacy/provider gate. |

The minimum read-only projection contract is: `patient/encounter identity + history availability + sourced actual/planned/missed events + asserted treatment/transition + explicit previous decision + recorded obligations + conflicts/unknowns + current reviewed guidance with rule/source identity`. A source-to-editor route and dependency invalidation must accompany every editable item. Field names and endpoint shape are implementation details; these meanings are fixed.

## 3. One adaptive interaction grammar

**Cockpit preparation.** A patient-specific appointment card is shown only inside an authenticated, authorized, privacy-reviewed context. Today's aggregate-only Cockpit Home does **not** itself authorize identifiable details. Where patient linkage or authorization is absent, show aggregate appointment state and a path to the protected record, not a guessed patient summary. Where available, the card reads the same qualified projection used in Module 01 and displays: likely visit focus; asserted treatment; last *recorded actual* dose/date or “unknown”; previous explicit final decision; one or a few relevant open recorded obligations; and a compact uncertainty cue. “No new fracture” or “stable DXA” is shown only if there is affirmative, sourced support, never from no entry. No clinical content is stored in a browser dashboard cache beyond existing approved boundaries.

**Visit entry.** Open in a small current-state workspace with patient/visit header, suggested focus, and a visible compact milestone/visit line. The clinician may change the focus or open any relevant editor. There is no “Step 1,” “2 of 4,” forced `Next`, or need to revisit completed domains. Suggested focus is a derived interaction hint, never a recorded diagnosis or decision.

**Milestone/visit trajectory.** Select source-backed visits and decision-changing milestones only: denosumab start/stop/transition, important actual administration, missed/planned event, significant investigation *only where existing reviewed comparability permits that wording*, explicit final decision and open/resolved obligation. Type and source/date appear with each point; a plan and derived due use visibly different labels from actual. Collapse routine detail, preserve source access and mark missing history or an uncertain link. Do not connect separated snapshots with an invented continuous line.

**Current visit.** State and present problem determine relevant blocks: what changed since the last visit, actual treatment chronology, unresolved verification, reviewed current guidance, decision and follow-up. A new fracture, preference, dental issue or other entered fact may reveal an existing appropriate editor/guidance branch, but Prototype 1 does not define new clinical interpretation for it. Safety/event/conflict cues override collapse. Guidance has nearby source/version and an expandable rationale; checklist display never reads as clearance. The clinician can move freely and correct an upstream fact directly. Pending recalculation withdraws stale timing/guidance; it does not erase recorded source facts.

**Decision and close.** Present recorded prior decision separately from today's options, recommendation, patient preference, final decision and actual action. None is preselected by a due calculation. Show recorded unresolved tasks, derived due and proposed next action as distinct. Closeout continues through existing authoritative editors/finalization, with explicit unknown and unresolved states rather than false completion.

## 4. Assisted historical reconciliation

The prototype may propose a **small set of candidate links** relevant to denosumab continuity: old treatment snapshots that may belong to one epoch, a missed/planned row and a later actual event, or apparently repeated obligations. Each proposal shows both original source records, dates, the reason for the match and the consequence for the visible trajectory. The clinician can **confirm link, correct source, keep separate or leave unresolved**. Confirmation records reviewer/time/source references through the existing owner's authorized write path once that path is designed; until then the prototype treats the link as transient and makes no durable continuity claim. A confirmed link does not itself certify clinical resolution or patient exposure. No bulk auto-merge, silent overwrite, deletion or manufactured negative history is allowed.

## 5. Implementation seams and order

1. Build/verify a read-only factual normalization seam over protected completed/amended encounters and dated results, reusing G1 actual-event/conflict logic and sharing source identities with G3. Preserve the present G1/G2/G3 responsibilities. Do not fork a second factual scanner with independent truth.
2. Expose the bounded projection and source-to-editor routes inside Module 01. Make history loading/unavailable, conflict, date precision and provisional/draft status explicit. Route edits to existing Step-2/3/4 owners; Steps 1–6 stay accessible.
3. Implement the compact in-visit trajectory and adaptive blocks using G4 disclosure and existing G2 guidance output. Invalidate/recompute dependent derived content on edit and suppress stale responses.
4. Add the protected pre-visit projection only after the Cockpit patient-link and privacy/authorization contract is satisfied. Reuse the same projection; keep the present aggregate Home behavior until that boundary is proven.
5. Add only the minimal clinician-confirmed continuity link path that existing owners can represent safely. If stable epoch/obligation persistence is not ready, show a labelled provisional reconciliation interaction and leave continuity unclaimed; this is a valid prototype limit, not a reason to invent a new store.

Likely reuse seams identified by R4 are `static/baseline-audit/patient-registry.js`, `longitudinal.js`, `progressive-guidance-core.js`/`progressive-guidance-ui.js`, `osteoporosis-longitudinal-summary-core.js`, Step-3/4 editors and `static/cockpit/app.js`. The implementation writer must verify file/API ownership against fresh `main` before touching them; this documentation branch is not a runtime base.

## 6. Acceptance evidence and stop conditions

Use synthetic cases for all five contexts in §1, including absent history, conflicting actual dates, prior stop decision, planned-without-actual and ambiguous task continuity. For each case verify:

- the entry state and pre-visit card (when authorized) agree on sourced facts and qualification;
- the milestone line exposes visit/event source and distinguishes actual, plan, decision, derived due and obligation;
- state change edits withdraw stale dependent guidance and immediately update or mark recalculation pending;
- an unknown or conflict blocks reliance on the affected timing conclusion but leaves source facts and unrelated editing available;
- no automatic administration, task closure, epoch link, treatment decision or unsupported “none” is written;
- the clinician reaches any relevant Step editor without numbered traversal, can return to the current view and resume from preserved protected state;
- guidance cites the existing reviewed rule/source and remains separate from recommendation/final choice;
- the protected pre-visit card cannot disclose patient-specific information through the aggregate-only Home or an unauthorized route.

Evaluation should include a short clinician walkthrough: find the last actual dose and prior decision, explain the current issue and uncertainty, revise an upstream date, inspect changed guidance, find the next obligation, and return to a prior milestone. Record observed navigation burden, misunderstanding and omissions; do not claim clinical benefit from synthetic correctness alone. Keep existing protected clinical, G1/G2/G3/G4, Cockpit privacy and finalization regressions green. Stop and replan for an owner collision, unsafe historical overwrite, a need for a new clinical rule, or a dashboard privacy boundary that cannot support the requested view. Roll back the projection/UI slice without deleting authoritative facts; any confirmed continuity data requires its own reversible migration/rollback design.

## 7. Handoff and authority

This contract is the next bounded implementation candidate, **not** implementation authority over the active root PR-1 lifecycle. Runtime work requires fresh-main bootstrap, explicit non-overlapping writer scope, concrete source/API ownership and focused tests. Prototype 2 receives a separate contract after Prototype 1's learning. PR-1 remains on release HOLD: H-05 and lifecycle engineering are complete at `0b45a4c96a7cb9a96893cfa3f14a1708f25e5345`; the identifiable-transcript privacy/provider gate is still open before its final release gate. No merge, deploy, identifiable transcript use or new clinical rule follows from this document.
