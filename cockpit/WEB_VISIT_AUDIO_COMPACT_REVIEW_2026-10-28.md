# Web Visit Audio — compact clinician review correction

## Product Owner checkpoint

**Problem now:** the first successful fictional web-audio trial produced a clinically bounded draft, but the review surface exposed the entire internal candidate schema and presented provisional source units as separate confirmation actions. The clinician saw a very long form and experienced nine ordinary statements as nine questions.

**What changes after this step:** web-audio drafts show only populated clinical sections, with one grouped editable text area per section. Empty schema categories, per-item add/remove controls and provisional promotion questions are absent from the default audio review. One collapsed attention summary may identify source sentences worth rereading, but explicitly says they do not require individual confirmation. The clinician makes one final patient/content confirmation before Preview and the existing explicit Save.

**Concrete example:** `DEXA T-score -2.8 στο ισχίο` appears once under `Ιστορικό και ευρήματα`, not once in a field card and again in a nine-item risk list. `Συζητήθηκε alendronate...` remains visibly uncertain unless the clinician edits it; the UI does not ask a separate `Επιβεβαιώνω ότι ολοκληρώθηκε` question.

**Outside this step:** no provider, retention, transcription, classification, selected-patient context, Preview, CAS, Save, record schema or Render configuration change. The ordinary pasted-Dia and amendment editors retain their existing full controls.

**Why now / next milestone:** the fictional first-use test proved the provider path and exposed a blocking usability defect before real-patient use. After reviewed code, focused browser evidence and guarded release, the clinician repeats one short fictional recording without Save and confirms that the compact review is usable.

The Product Owner's screenshots and explicit statement that the nine confirmations and huge review list are counterproductive are the correction authority for this bounded slice.

## Reuse and authority

- **REBIND presentation only:** keep the existing `VisitCaptureCandidateV1` object, protected context, `recording_patient_confirmed`, Preview and one-shot Save.
- The web-audio draft uses a compact renderer; the ordinary/manual candidate renderer remains unchanged.
- No second candidate, write path, inference engine or persistence owner is introduced.

## Frozen behavior

1. Render `reason_for_visit` and only non-empty audio-supported arrays: history, findings, clinical impression, options discussed, follow-up, safety net, administrative notes and uncertainties.
2. Use one textarea per populated section. Each non-empty line represents one existing exact-source fact. Editing marks the fact `clinician_edit`; adding a line creates a clinician-confirmed fact of the same section; deleting a line removes that draft fact.
3. Hide empty categories and all per-item `Προσθήκη`, `Αφαίρεση`, `Μεταφορά...` and `Επιβεβαιώνω ότι ολοκληρώθηκε` actions in web-audio mode.
4. Provisional recommendation/decision/completed/preference units stay in `uncertainties`; the UI must not silently promote them.
5. Show review-unit warnings once in a collapsed attention disclosure, deduplicated by source text, with explicit wording that no item-by-item confirmation is required.
6. Retain exactly one final checkbox confirming the displayed patient and the reviewed compact text. Any edit clears it and invalidates Preview, as today.
7. Retain existing Preview and explicit Save gates. Unresolved uncertainty is allowed to remain explicit; no automatic clinical fact or obligation is created.

## Evidence questions

- The fictional browser path shows only populated grouped sections and no empty-schema/add/remove/per-item-confirmation controls.
- The same source sentence is not duplicated in the main editor and the visible default warning surface.
- A grouped edit reaches Preview/Save only after the single final checkbox and is the value persisted by the existing backend.
- Withdrawal, selected-patient context, temporary audio/transcript handling and explicit Save behavior remain covered by the existing focused audio test.

## Risk classification and release

R2: although this is a bounded UI correction, it removes provisional clinical-promotion controls and therefore changes review semantics. One independent pre-code design review and one independent exact-code post-code fidelity review are required. Synthetic data only; no agent recording, provider request or clinical write. Release remains separate and requires exact-head CI plus no-PHI production smoke.
