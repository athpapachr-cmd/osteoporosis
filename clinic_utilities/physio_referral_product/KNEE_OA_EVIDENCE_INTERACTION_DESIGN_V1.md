# Knee OA evidence interaction and traceability v1

> **Status:** DESIGN CANDIDATE, 2026-09-11. Runtime NOT IMPLEMENTED.
> **Scope:** Knee Osteoarthritis only; Step 4.
> **Parent:** `8489e2ee32f7aeae6f678c7db2838930c0759eb4`.
> **Machine owner:** `contracts/knee_oa_evidence_interaction_v1.yaml`.

## 1. Four independent meanings

The interface must keep these separate: clinician selection, reviewed evidence position, evidence provenance/availability, and clinical safety. A neutral check identifies selection. Green means reviewed support, not selected, effective for every patient, high certainty, or safe to proceed. Text weight is emphasis, never a numeric evidence grade.

The existing six evidence states are unchanged. Each has a distinct visible monochrome cue in addition to colour, with its full Greek meaning available through the same evidence control. A screen-reader-only label does not by itself solve colour-only communication for sighted users with colour-vision differences [W1].

| State | Compact Greek meaning | Semantic colour | Non-colour cue |
|---|---|---|---|
| recommended_or_supported | Συνιστάται | green | solid dot + short underline; semibold label |
| conditional_or_context_dependent | Υπό προϋποθέσεις | blue-grey | outlined diamond |
| limited_or_insufficient_evidence | Περιορισμένη τεκμηρίωση | amber | question mark in circle |
| guideline_conflict_or_mixed | Οι οδηγίες διαφέρουν | violet | opposing arrows |
| recommendation_against_routine_use | Δεν συνιστάται για συνήθη χρήση | muted red | minus in octagonal outline |
| not_yet_assessed | Δεν έχει αξιολογηθεί | grey | dotted ring |

These are semantic tokens, not approved final colour values or icon artwork. The prototype must prove they remain distinguishable in monochrome, forced colours and relevant display modes. No permanent wall of badges or six-colour legend on the work surface. `Ενδείξεις` inside the evidence sheet can reveal the legend.

## 2. Row interaction

The main row target toggles selection; a separate sibling `i` target opens evidence. No nested interactive buttons and no event propagation that selects treatment while opening information. Exactly one evidence-disclosure control per rendered item. A bubble references that control rather than adding another `i` beside it.

The icon can look small while its hit area is at least 44 by 44 CSS pixels. That is this web product's comfort target, not a claim that Apple's points are CSS pixels or that WCAG AA universally requires 44 pixels. Apple describes 44-point default controls for iOS; WCAG 2.5.8 has a 24-CSS-pixel minimum with exceptions [A1, W2].

Use native buttons, explicit accessible names (`Τεκμηρίωση για {item}`), a programmatic selected state, visible keyboard focus and Enter/Space activation. A title tooltip, hover or long press is never the only way to obtain evidence.

## 3. Calm contextual messages

At most one expanded evidence bubble is visible, anchored next to the most recently acted-on relevant item. It is non-modal, takes no focus, does not cover another target or the Copy action, and uses no auto-dismiss timer. It closes on explicit dismissal or relevant context change. In a narrow viewport it becomes a small inline message rather than overflowing the screen.

First render with normal defaults does not generate a stack of notifications. Selecting a mixed/limited/against/unassessed item may expose its brief state phrase. A conditional item exposes extra explanation only when relevant prerequisites need attention. Recommended items need no celebratory bubble.

Dismissing a bubble is not acknowledging a clinical warning, agreeing with a source, or removing evidence. The row cue and review summary remain available. Replacing the currently expanded bubble does not delete other notes. Deduplicate notes by item and reason; a five-source disagreement is not five warnings.

Apple recommends minimizing timed interface elements; relevant WCAG hover/focus content must remain dismissible, hoverable and persistent when that interaction is used [A1, W3]. Routine message changes use a polite status region without focus theft, not repeated assertive alerts [W4].

## 4. One evidence sheet, not nested popups

An explicit `i` tap opens one sheet. On mobile it is a bottom sheet; desktop may use the same modal sheet placed to the side. This first design uses modal semantics consistently: background inert, focus enters the heading, Tab stays inside, Escape and a visible close button close it, and focus returns to its invoker or the surviving corresponding row [W5]. Do not set aria-modal on a panel that leaves the background interactive.

Sheet order: intervention, plain evidence state, concise reviewed purpose, source attribution, last clinical review date, then `Τεκμηρίωση`. Detailed evidence expands inside this same sheet. An already-open referral sheet is replaced in the single sheet host, preserving its scroll and manual buffer; closing evidence restores the prior view. No overlay stack is required.

A material safety block is never hidden by the evidence sheet. If inherited safety state changes while the sheet is open, a persistent safety notice and action remain reachable within the current surface; exports remain blocked. Evidence dismissal never records a safety disposition.

For a mixed-guideline item, the first disclosure shows every reviewed source position in concise source-specific rows, including opposing and neutral positions. It must not show only a friendly selected source with the opposition hidden behind a second tap. Full native wording/strength and links belong in the deeper section. More scrolling is acceptable here; false consensus is not.

## 5. Faithful attribution

Each position has an immutable compound reference `{item_id}/{source_id}` within the pinned evidence package. Preserve source direction, native strength and support scope separately. `Limited` attached to a positive AAOS recommendation is not automatically the app state `limited_or_insufficient_evidence`. Do not convert EULAR grades, AAOS strength and GRADE terminology into one undocumented scale.

A named component of an exercise recommendation is labelled `Μέρος ευρύτερης σύστασης`. A task-specific clinical mapping is labelled `Κλινική προσαρμογή`, not `Ισχυρή σύσταση`. The deeper sheet retains the parent source's exact native strength together with the scope limitation, never as an isolated badge on the narrower item.

Clinical-purpose summaries are drawn from the frozen Step-2 item. Step 4 adds no effect size, dose, treatment promise or new medical recommendation. Translating or shortening source summaries for production needs faithful, reviewed Greek copy, not autonomous clinical reinterpretation.

Publication and review dates have different owners. Examples of display metadata from the inherited registry:
- `NICE NG226 · 2022`; separate `Κλινική ανασκόπηση: 11/09/2026`.
- `EULAR · ενημέρωση 2023 (δημ. 2024)`.
- `ACR/AF · 2019 (δημ. 2020)`.

Opening the application, checking a link or reviewing this UI does not refresh the clinical review date. Do not show `τελευταίες οδηγίες` or `ενημερωμένο σήμερα` without the corresponding clinical verification.

The inherited registry currently provides source-level URLs, not exact per-recommendation locators. Expose `Παραπομπή στην πηγή` and locator precision `source_level`; never invent a recommendation number, page, quotation or item-level verification. Source-level traceability is real but is not a completed independent source-to-claim audit.

## 6. Evidence availability is an overlay, not a seventh clinical verdict

Keep the last reviewed six-state verdict, its source set and its package version immutable for the current draft. If a material dependency is unverified, unavailable, due for review, withdrawn or superseded, show that status separately. Suppress fresh green endorsement, automatic promotion and evidence-backed defaults from the affected item until review resolves it. Preserve already-selected clinician choices and clearly show the limitation.

Do not recalculate a consensus after silently dropping an unavailable opposing guideline. Do not call unavailable material `not assessed` or `against`. Link availability is not evidence currency. No review interval is invented in Step 4; `review_due` requires explicit package metadata.

Evidence limitations alone do not override Step-3 Copy policy. Invalid clinical inputs and unresolved safety blocks still block export; a literature-access limitation is not falsely converted into a contraindication.

## 7. Evidence-backed suggestions

Suggestion eligibility comes from Step-2 `suggestion_policy`, filtered through Step-3 supported input scope. Only explicitly present/abnormal clinician-selected findings, or selected functional limitations, become positive trigger facts. Unselected, absent and not-assessed findings do not.

Equivalent finding/function aliases are explicit. `stairs_limitation` can support the existing `stairs` trigger; a generic stiffness tag cannot become measured ROM restriction and generic weakness cannot become objective weakness.

Core omissions rank before contextual candidates. Stable clinical order resolves ties, not a confidence score. Show one compact suggestion with the candidate, one faithful source/year cue, and one-tap add. `Προτάσεις · n` reveals the remaining list when needed. All candidates remain discoverable without queuing timed notifications.

Each candidate retains item ID, trigger reason(s), source-position references, claim scope and package/draft revision. A contextual recommendation's source caption explicitly says `Κλινική προσαρμογή` or `Μέρος ευρύτερης σύστασης` when required. Do not dress a clinician-designed link between stairs and task retraining as a stand-alone guideline mandate.

Adding requires an explicit user action and a still-valid current candidate. Revalidate the draft and package revision on activation; reject a stale candidate instead of adding it to a changed or new referral. Duplicate reasons produce one candidate. Already-selected items are not offered again. Step-2 no-auto-promotion exclusions remain intact.

A dismissed suggestion stays dismissed for the same item, reason signature and package version in that draft. A new relevant reason, changed package or new referral can make it eligible again. Dismissal never means clinically contraindicated or evidence disproved. No dismissal/trace data are persisted.

## 8. Power users without a larger default screen

One slim `Περισσότερα` disclosure remains collapsed by default. Its count is the number of selected canonical advanced items, not the number of warnings, sources or nested fields. Selected advanced choices remain selected and counted when collapsed. Explicit restrictions/notes remain preserved under Step-3 text ownership.

A collapsed section with an evidence note retains a compact non-colour note indicator and access through the final review summary. Expanding never resets choices, changes evidence, or preselects additional treatment. Context changes do not silently discard selected advanced data; invalid selections follow the existing fail-closed contract.

## 9. Readiness outranks surface calm

The Step-3 gate remains authoritative. The Step-1 `Έτοιμη · σημείο για έλεγχο` example applies only when export is actually allowed. It must never label missing diagnosis/laterality, invalid input or unresolved safety as ready.

Priority: inherited blocking safety/validation, missing or stale gate/preview revision, manual-buffer reconciliation, then ready-with-evidence-notes, then ready. All export actions share this guard: Copy, print, PDF and future share. The prototype must not preserve an old copyable referral after a blocking change.

An open evidence sheet, dismissed bubble or acknowledged literature conflict cannot clear a safety block. Evidence interactions do not rewrite referral text, structured facts or a manual buffer. A new structured revision makes an older projection non-exportable until reconciliation. Manual edits remain clinician-owned text, not automatically evidence-validated prose.

## 10. Privacy and release integrity

Traceability is an in-memory dependency graph, not a new patient audit database. Do not persist draft facts, trace events, selections, buffers or dismissed suggestions in localStorage, sessionStorage, URLs, logs, analytics, crash reports or server state. Reset and BFCache restoration clear patient-scoped state in the later implementation. Preferences, if later authorized, must be separate from clinical draft state.

External evidence links open only on deliberate activation and contain only reviewed static source URLs. No patient-specific query string, background literature request, third-party tracking script or AI service is introduced. Source links are restricted to HTTPS and vetted source metadata.

Package, evidence, template and projection versions travel together. A mid-draft update does not silently replace source positions or claim currency. Mismatched bundles fail the evidence-presentation check rather than falling back to a reassuring label.

## 11. Acceptance beyond machine checks

A design validator can establish identity, referential integrity and deterministic projection/reducer behavior. It cannot establish actual contrast, correct screen-reader behavior or an iPhone-like experience.

The Step-5/6 acceptance matrix must include: 320-CSS-pixel reflow, 200% text enlargement, keyboard-only operation, iPhone Safari/VoiceOver, reduced motion, forced colours, light/dark if offered, touch-target hit boxes, visible focus not covered by sheets/sticky UI, one-tap evidence without selection changes, long Greek labels, all-source conflict details, retention of collapsed selections, stale suggestion rejection, manual-edit protection and safety precedence. Standard text targets 4.5:1 contrast and meaningful controls/indicators 3:1; test actual combinations, not token names [W6, W7].

Tap-count and task-time goals remain hypotheses until observed. Independent clinical/physiotherapy/UX/commercial review is still required after the functional vertical slice, not claimed by this writer review.

## Design sources, accessed 2026-09-11

These support interface/accessibility choices, not Knee-OA clinical effectiveness.

- [A1] Apple HIG Accessibility: https://developer.apple.com/design/human-interface-guidelines/accessibility/
- [W1] W3C WCAG 1.4.1: https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html
- [W2] W3C WCAG 2.5.8: https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html
- [W3] W3C WCAG 1.4.13: https://www.w3.org/WAI/WCAG22/Understanding/content-on-hover-or-focus
- [W4] W3C WCAG 4.1.3: https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html
- [W5] WAI-ARIA modal dialog pattern: https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/
- [W6] W3C WCAG 1.4.3: https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html
- [W7] W3C WCAG 1.4.11: https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html
