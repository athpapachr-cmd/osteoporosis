# Osteoporosis Product Reconstruction Decision

> **Workstream:** OST-UI · **Date:** 2026-10-01 Asia/Nicosia  
> **Disposition:** R1–R4 reconciled; one Phase-1 synthesis; product direction accepted for bounded prototype contracts, not a runtime release.  
> **Sources:** accepted [R1](R1-CURRENT-PRODUCT-CONSTITUTION-AUDIT.md), [R2](R2-LONGITUDINAL-CLINICAL-TRAJECTORY-REVIEW.md), [R3](R3-POINT-OF-CARE-INTERACTION-REVIEW.md), and [R4](R4-SHARED-CORE-MODULE-ARCHITECTURE-REUSE.md); Product Owner directions in the referenced coordinator conversation. This is reconciliation, not a fifth review.

## A. Product semantics confirmation

The product unit is the **patient's continuing osteoporosis care trajectory**. An encounter is the point at which sourced history, current observations, reviewed guidance, clinician judgment and future obligations meet. The clinician should understand both **what needs attention today** and **how the patient arrived here**. A useful screen must make that continuity apparent without exposing the machinery of Core, guidance, evidence, treatment epochs, provenance and audit as routine form burden.

The Product Owner's desired interaction is an **adaptive guided visit**: calm, clinically ordered and easy to enter, while the current patient state and today's problem determine which information and questions are prominent. Its progression follows the logic of the consultation but never forces a numbered wizard. A known patient with a denosumab visit, a delay, or a documented transition may therefore open in different current contexts. The system proposes a context; the clinician can correct facts, change focus and move directly to any relevant domain.

The longitudinal line of **visits and meaningful milestones** is a permanent part of this model. It is a selective, source-linked trajectory, not an exhaustive activity log or a claim of complete history. The pre-visit Cockpit summary and in-visit trajectory are two projections of the same protected, reconciled patient state. Cockpit owns appointment entry and a concise preparation view; Module 01 owns osteoporosis meaning and the visit interaction. Neither projection owns a second patient record or an independent clinical summary engine.

## B. Current-state diagnosis

R1 found a real protected patient, encounter and lab substrate, working current guidance and useful history-sensitive summaries, inside an encounter/audit-first six-step shell. R2 found that captured actual administration, some treatment and fracture facts, dated investigations and prior decisions can support a selective story, but a certified 10–15-year story cannot be inferred from unlinked treatment snapshots, ambiguous task identity, overlapping DXA/FRAX/lab representations, missing decision roles and unavailable historical evidence context. R3 found that eight clinical journeys do not share a sensible mandatory Step 1→6 route. R4 found that this calls for reuse and bounded semantic extensions, not another truth store or broad engine replacement.

The visible product problem is therefore **substantial interaction and information-architecture misalignment**, together with **targeted longitudinal semantic gaps**. The present six steps remain useful for detailed entry and closeout, yet their fixed navigation exposes internal structure too early. A compact patient/current-problem entry can already rely on qualified present facts; confident continuity claims require explicit reconciliation and provenance first.

## C. Preservation map

| Preserve | Role in target product |
|---|---|
| Protected patient/encounter/lab persistence and completed/amended history | Sole authoritative clinical substrate; current draft remains distinguishable from completed history. |
| Step 2–4 factual capture, Steps 5–6 communication/documentation, finalization and audit fields | Underlying editors and record completion surfaces, reachable directly from the current issue. |
| G1 longitudinal projection and EncounterContext | Guarded actual-event chronology, conflict handling and current visit context. |
| Reviewed G2 registry, deterministic guidance and `Γιατί τώρα` | Current evidence-linked guidance, separate from patient fact and clinician decision. |
| G3 selective summary and G4 disclosure/salience | Source-labelled history selection and restrained presentation. |
| S1 fracture/fragility semantics and actual-versus-planned administration guard | Existing fail-closed clinical/data integrity boundaries. |
| Cockpit Home and Physio interaction mechanics | Global entry; progressive disclosure, contextual controls, direct edit, stale-output withdrawal and evidence on demand, without importing Physio clinical meaning. |
| PR-1 proposal/review boundary | Candidate extraction remains transient and clinician reviewed before an existing owner changes. |

## D. Removal / relocation map

The six-step shell moves from **mandatory primary route** to **secondary source editor**. Remove numbered progress language such as “2 of 4” from the primary visit. Retire the duplicate global Physio link inside Module 01 and clean stale pilot/privacy or dead navigation copy in a later bounded UI change. Keep global navigation in Cockpit. Shared Core should own reusable patient/context identity, provenance display and a generic obligation identity envelope; osteoporosis retains fracture, treatment, goal, guideline, due and decision meaning. The tuple `type|due_date|timeframe_text` must cease to be authoritative task continuity only after legacy rows are reconciled to stable obligations. No unique historical row is discarded to make the display cleaner.

## E. Target product model

One protected patient/encounter/lab substrate feeds a **read-only, provenance-aware factual reconciliation**. From it, G1 derives continuity and current context; G2 evaluates existing reviewed rules; G3 selects history; and the adaptive visit presents the smallest decision-relevant view. The same state supplies a privacy-appropriate pre-visit Cockpit projection. A visit writes through existing clinical owners, never through the projection.

The model keeps separate: recorded actual administration; planned/scheduled/missed action; rule-derived due state; clinician recommendation; patient preference; final decision; and later observed action. A treatment epoch or obligation continuity link exists only when supported by explicit, clinician-confirmed source facts. Conflicting or absent history stays visible as unknown. Historical guidance “as seen then” cannot be reconstructed by rerunning today's registry. Current G2 guidance never becomes a historical patient fact.

**Assisted historical reconciliation** is a short clinician task, not autonomous reconstruction: show likely matching source rows and why they may match; permit confirm, correct, keep separate or leave unresolved; retain original sources and the reviewing clinician/time. A proposal cannot silently link an old missed plan to a later dose, merge treatment epochs, close an obligation, overwrite a DXA/FRAX/lab value or infer exposure. The safe default is an incomplete trajectory with explicit uncertainty.

## F. Target interaction model

1. **Before opening the visit:** a scheduled patient card in Cockpit shows a concise, source-qualified osteoporosis state, last *recorded actual* treatment event where reliable, previous explicit decision, relevant open recorded obligation, current uncertainty and likely visit focus. No fabricated “no new fracture” claim follows from silence; no identifiable clinical detail appears where the existing Cockpit privacy boundary disallows it. Opening the card passes patient/encounter context into Module 01.
2. **At visit entry:** a small header states patient, present treatment/state and today's likely issue. The clinician may accept or change the focus. No Step 1 or numbered wizard is required.
3. **In the visit:** the default view shows “what matters today” beside a persistent, compact **milestone/visit trajectory**. Only source-backed milestones appear: meaningful treatment start/stop/transition, actual administration when important, fracture, decision-changing DXA/result, major decision and obligation. Every item has date, fact/decision/plan type, source and a path to the encounter/editor; uncertain links are marked. The line expands for detail without becoming a complete activity log.
4. **Adaptive composition:** patient state plus current problem activate relevant blocks and questions. New information changes the composition in place. Safety/event/conflict signals override a collapsed preference. The clinician can jump freely, revise an upstream fact and see dependent derivations withdrawn or recalculated; no hidden “Next” gate controls clinical navigation.
5. **Decision and close:** facts, current interpretation, reviewed guidance, options, clinician recommendation, patient preference, final decision, actual action and future obligations remain visibly distinct. Evidence is available beside the statement it supports. Existing Steps 1–6 are direct-entry editors and documentation/closeout surfaces underneath this experience.

This is one interaction system with state-dependent composition, not a separate application for every drug or problem. Denosumab due/delayed/transition, new patient, fracture, dental concern, refusal and other future situations can share its grammar only when their clinical semantics are separately authorized.

## G. Reconstruction scope decision

**Decision: substantial UI/information-architecture restructuring plus bounded longitudinal semantic extension and factual consolidation.** The primary visit entry and navigation need reconstruction because the fixed shell cannot express the accepted interaction without persistent cognitive burden. The protected substrate, reviewed guidance, existing editors and useful projections are retained. Stable treatment and obligation identities, distinct decision roles and source reconciliation are limited data-contract extensions within existing owners; they do not justify a near-total product rewrite, new clinical database, new rules or automatic treatment choice.

Prototype findings may change presentation/component factoring. They may not create clinical meaning. Full reconstruction remains behind the programme's later two-prototype benefit and safe-longitudinal-state gates. R4's exact ownership/interface prerequisites are frozen per prototype before runtime work.

## H. Prototype recommendation and gates

Exactly two initial vertical slices test this model:

1. **Prototype 1 — Denosumab longitudinal adaptive visit.** Test sourced actual chronology, on-time/delayed/transition context using only existing reviewed guidance, a compact milestone line, pre-visit summary, clinician-confirmed reconciliation, state-dependent questions, direct source edits and honest future obligations. Its bounded implementation contract is [PROTO1-DENOSUMAB-LONGITUDINAL-ADAPTIVE-VISIT-CONTRACT.md](PROTO1-DENOSUMAB-LONGITUDINAL-ADAPTIVE-VISIT-CONTRACT.md).
2. **Prototype 2 — New fragility fracture / treatment reassessment.** Use existing S1 mechanism semantics and explicit treatment exposure, preserving unknowns and keeping options/recommendation/preference/final choice distinct. A separate contract is needed before implementation.

Both must show whether the clinician can orient, find uncertainty, revise a source, understand a derived change, choose a next action and resume later without traversing irrelevant Steps. Qualitative usability and safe interpretation are prototype outcomes, not claims already proven by R1–R4. No new cadence, threshold, contraindication, failure or switch rule is authorized.

## Coordination boundary — PR-1 release engineering

The PR-1 handback was reconciled read-only at verified remote branch head `0b45a4c96a7cb9a96893cfa3f14a1708f25e5345`, after current-main merge checkpoint `5376482f78c731d24dd4810f796e60398a62ae19`. The accepted H13 semantic files and frozen inputs remained unchanged; H-05 provider offload and browser request invalidation/lifecycle cleanup were completed. The handback reports the final exact-head PR-1 gate [`36861040681`](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/36861040681) as passed; the local checkout confirms the head and scope, while the GitHub Actions API was unavailable for independent live inspection in this session. The **identifiable-transcript privacy/provider gate is the only identified prerequisite remaining before the final release gate**. It is open; production PHI use, merge/deploy and pilot are not claimed or authorized by this synthesis. PR-1 and OST-UI retain separate owners.
