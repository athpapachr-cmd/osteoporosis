# Purpose-focused compact clinical capture

Base main458c29645545b72b1500997b71f17e7539b14c2f. Product Owner explicitly approved corrections, validation against an anonymized version of the supplied osteoporosis/MRI encounter, merge and deploy. Existing technical release is not product acceptance.

## Scope and reuse

EXTEND existing typed candidate, verbatim prepared-note store, protected draft/preview/atomic Save, recovery and audited amendment. No second store, database migration, provider/model/config change or additional model call for direct import. Existing raw audio/transcript retention and patient/identity binding remain unchanged. R2: audio neutral-purpose metadata plus clinical suggestion scope; one pre-code and one exact-code review.

## Compact display / complete optional record

All draft, latest-chart and selected-visit views display a compact source-derived projection by default. Complete prepared text or all typed fields remain under an explicit closed `Πλήρης κλινική καταγραφή` details control, with exact copy/export and editing retained. No source text is deleted or rewritten. New ChatGPT prompt produces a concise `Κλινική εικόνα` paragraph and short purpose-focused `Πλάνο και παρακολούθηση`; all additional details remain in ordinary headed sections in the same original note. No second AI call occurs at import/render.

Existing notes lacking the compact header receive labelled exact excerpts: reason, first complete paragraph of symptoms/status, first assessment paragraph, first plan paragraph and follow-up. Whole source spans are selected, never split mid-sentence; extra paragraphs/long spans remain explicitly available in full record. The projection is not a claim that omitted details are irrelevant. For typed audio, emphasize reason, source clinical impression, decisions/therapy and stated follow-up; long findings/history/uncertainties remain complete in optional detail. Compact projection is display only; Save still persists the complete candidate.

Completed/resolved/cancelled tasks leave the default active-obligations panel. They remain available in a closed task-history view and existing lifecycle/audit/history controls. Never change actual task status or delete clinical data based on this chat. Completed administrative referral in the synthetic example must not be presented as the clinical outcome or an active task. Explicit safety-net data remains visible; significant source caveats must not be rewritten into certainty.

## Audio reason_missing correction

A classification with identity consistent and valid explicit clinical facts may build a candidate even if no clause is tagged reason. Use neutral `Κλινική καταγραφή` metadata, mark `reason_is_unspecified=true`, certainty uncertain, and show a neutral title rather than claiming a chief complaint. No history/finding is invented as a visit reason. Preserve every source-derived group and transcript; permit existing Preview/one explicit Save. No clinical content or unresolved identity still blocks as before. The metadata flag defaults false for old notes and AI manual extraction. Clear the flag when the clinician explicitly supplies a reason. Contextual relevance recommendations are absent when a purpose cannot be established; missing purpose itself does not trigger a task or extra confirmation.

## Relevant next-visit considerations

Primary considerations must be linked to today's explicit purpose and the treating orthopaedic clinician's role. Extend the existing optional whole-visit considerations call (no extra call) with explicit purpose evidence and clinician-role context. Each consideration carries source IDs, purpose source IDs, role scope and impact; the deterministic checker retains only within-role items with purpose IDs drawn from reason facts and an eligible management/continuity/safety impact. Require a short rationale explaining the decision or follow-up it may affect. This checks declared linkage, not medical semantic correctness; prompt instruction plus representative actual/synthetic evaluation establishes the example behavior, not a guarantee for every possible case.

Ordinary uncertainty or incidental MRI/kidney findings must not automatically become clinical tasks/questions. Handoff-only/incidental/no-impact items are excluded from primary suggestions. Existing explicit instructions to another clinician remain recorded in full, not silently reassigned. Separate uncertain reported fact from an unresolved clinical question that affects management. For the supplied purpose (osteoporosis and vertebral-fracture assessment), suggestions should concern DEXA follow-through and endocrine treatment decision; no automatic clarification demand about disc extrusion or renal pelvis. If those are the explicit purpose in a different encounter, do not globally blacklist their words. Generic legacy typed reminders remain optional/collapsed, never primary clinical focus; no automatic adoption.

Purpose-scoped audio considerations must remain available after Save/reopen. Add a bounded optional `next_visit_focus` list (max3) in existing candidate JSON: text, rationale, exact validated source excerpts, explicit purpose text, within-role impact and provenance; origin is fixed ai_suggested and status fixed suggestion. Render it separately as unadopted AI preparation, never as stated facts, decisions, prescriptions or pending tasks. Existing single clinician Save retains these labelled preparation notes; no automatic adoption or extra write. Full clinical facts and sources remain separate. The edit/review surface permits removing suggestions; direct prepared-note suggestions remain verbatim labelled text. No new persistence owner or schema migration. This explicitly supersedes the earlier display-only contextual retention boundary for reviewed labelled planning notes only. New schema/source validation and persistence/reload/exclusion regressions are part of R2.

## Prompt

Replace the exhaustive repetitive instruction with a concise result-focused prompt: short overview, short plan, full clinical details without repetition, explicit active obligations only, completed actions separately, at most3 purpose/role-relevant next-visit ideas with clear decision impact. Keep numbers, negation, uncertainty and conditional therapy exact. Omit empty/absent checklist fields. No blanket statement that no medication is needed from absence of a new prescription. Prompt is in-app and downloadable; no identifiable case content enters public fixtures.

## Finite acceptance and release

1. Supplied-style long anonymized legacy note: compact visible card, full body closed yet exact original/copy/export retained, desktop/mobile; all relevant history/negation/conditions remain available.
2. New prompt-shaped note: concise overview/plan visible, incidental uncertainty/full imaging optional, DEXA/endocrine follow-through visible, completed admin task absent from active focus but accessible.
3. No-reason valid classification succeeds with neutral metadata, all facts retained, one Save; identity conflict/empty clinical content still rejected; retry/recovery/auth/stale invariants hold.
4. Purpose-scoped considerations: DEXA/endocrine fixture retained; declared incidental/handoff/no-impact/unlinked items rejected or omitted; disc/kidney tokens not globally banned for other purposes. No automatic task creation or treatment.
5. Existing direct no-provider import, one-shot Save/replay/stale guards, full edits/amendments, audio success/recovery and explicit Save remain.
6. Source/prompt tests plus one actual provider synthetic proof through existing authenticated test harness if available; never send the user's actual note/audio, modify patient records or expose credentials. Record actual/mocked limits separately.

After independent closure and exact-head CI: guarded merge, one native Render deploy, exact assets/auth/log/no-write browser smoke, durable release checkpoints. After notes with new metadata exist, retain their reader compatibility on rollback; disable new behavior rather than hiding saved data.
