# Web Visit Audio — source completeness and separate next-visit suggestions

## Product Owner checkpoint

**Problem now:** the fictional web recording reaches the provider and the compact review is usable, but the draft loses clinically material structure. The screenshot shows history/symptoms/treatment/plan/obligations in the Dia-generated comparison while the Cockpit result places a compound demographic-history sentence in `Λόγος επίσκεψης`, retains only fragments of history/findings, leaves treatment in `Αβεβαιότητες`, creates no obligations and offers no source-informed next-visit suggestions. The displayed one-month follow-up also conflicts with the Dia-generated three-month wording and therefore cannot be accepted without source review.

**After this step:** the provider may identify multiple exact excerpts inside one source unit and classify them into source-backed history/symptoms, findings, treatment/recommendations, explicit pending steps, follow-up and uncertainty. The server validates every excerpt against the transcript and deterministically builds the candidate. A second, visibly separate `Προτάσεις AI για την επόμενη επίσκεψη` block may infer bounded questions/checks from cited source units; it is not part of the candidate and cannot be saved unless the clinician performs one bulk `Προσθήκη στο πλάνο` action, after which the existing final confirmation/Preview/Save still apply.

**Concrete example:** an exact spoken clause about thoracic pain appears under `Ιστορικό και συμπτώματα`; an exact spoken X-ray plan becomes one editable `Επόμενο βήμα / υποχρέωση`; an AI suggestion such as checking adherence or adverse effects appears separately with a rationale and is absent from the record unless deliberately added.

**Outside this step:** no new provider, model upgrade, patient lookup, raw-transcript UI, automatic diagnosis, medication change, test order, coding, background write, configuration/retention change or real-patient agent test. Dia-generated prose is comparison evidence, not source truth, and is never copied into the record.

**Why now / next milestone:** the Product Owner's fictional comparison proves that simple single-category ID classification is not clinically complete enough. This bounded semantic correction precedes the deferred native-ASR qualification. After reviewed implementation/release, one new fictional no-Save recording checks completeness and suggestion usefulness.

The screenshot and Product Owner request are explicit authority for this replan.

## Observed cause in the released code

1. `source_units()` produces sentence-sized units; `checked_labels()` permits each entire ID in at most one category.
2. `build_candidate()` maps only history/finding/impression/option/follow-up/safety/administrative into typed sections. Recommendation, decision, completed and preference are deliberately routed to uncertainties.
3. The released contract creates no medication structure or durable pending obligations and has no suggestion object.
4. Exact-source copying prevents hallucination but one-category sentence assignment drops semantic clauses. The correction must preserve the exact-source safety property while allowing multiple exact excerpts from one unit.

## V2 provider contract

Reuse the existing `gpt-transcribe` and `gpt-6-luna` calls, timeouts, project headers, `store:false`, no-tools rule and temporary-content lifecycle. Replace only the classifier Structured Output with `visit_audio_analysis_v2`:

```text
identity_state: consistent | uncertain | conflict
facts[]:
  source_id: integer
  excerpt: exact non-empty substring of that source unit
  kind: reason | history | symptom | finding | impression | option |
        therapy | decision | completed |
        preference_accepted | preference_declined | preference_undecided |
        obligation_investigation | obligation_referral |
        obligation_follow_up | obligation_other |
        follow_up | safety | administrative | uncertainty
suggested_next_checks[]:
  text: generated check/question, max 240 chars
  rationale: generated reason tied to the supplied source, max 360 chars
  source_ids: one or more distinct IDs from the current batch
unclassified_ids[]: supplied IDs with no safe fact
```

Validation is fail closed:

- IDs must belong to the current batch; unknown IDs and duplicate entries in `source_ids` or `unclassified_ids` fail.
- Each normalized excerpt must resolve to exactly one character range in its normalized source unit. Zero matches or an ambiguous repeated match fails. The server copies the matched source substring, never provider prose, into the clinical candidate.
- A matched source range has exactly one `kind`. Fact ranges belonging to the same source unit must not overlap at all, even when their kinds differ. This deliberately rejects lifecycle contradictions such as the same words being treated as both an `option` and a `decision`, `completed` action or `obligation_*`; the provider must instead return distinct, non-overlapping spoken clauses. Any overlap or duplicate range fails the entire provider result.
- One source ID may yield several facts only when their exact matched ranges are distinct and non-overlapping. Every supplied ID must appear in at least one valid fact or exactly once in `unclassified_ids`, never both.
- Identity uncertainty/conflict still blocks the draft.
- Suggestions are limited to checks/questions for a later review. They may not diagnose, assert an unspoken fact, recommend a medication/test/treatment change, or claim an action occurred. Each cites current-batch source IDs. Invalid suggestions fail the provider result rather than entering the UI.
- Merge suggestions across batches by normalized text, keep at most six, and retain cited IDs/rationale only in transient browser response state.

## Deterministic candidate mapping

- All `reason` excerpts join deterministically into `reason_for_visit`; absence still fails.
- `history` and `symptom` become exact-source `history` facts; the compact audio label is `Ιστορικό και συμπτώματα`.
- `finding`, `impression`, `option`, `completed`, follow-up, safety and administrative map to their existing candidate arrays.
- `therapy` becomes a source-backed recommendation with deterministic type `therapy`; `decision` becomes a source-backed decision with deterministic type `advice` and no responsible-role inference.
- Preference kinds map to their explicit deterministic status.
- Explicit obligation kinds create source-backed pending candidates using the exact excerpt as both source excerpt and editable description, `status=open`, `responsible_role=unspecified`, no inferred review date/trigger/external dependency, and deterministic item type.
- `uncertainty` and all unclassified source units remain visible uncertainties. Numbers, dose, laterality, negation, dates/timing and uncertainty remain in the compact attention disclosure.
- No structured medication object, date, code, diagnosis, responsible person or completed status is inferred.

## Suggestion UI and authority

- Show at most six suggestions in one compact block after source-backed sections and before the final patient/content confirmation.
- Label: `Προτάσεις AI για την επόμενη επίσκεψη — δεν είναι καταγραφή όσων ειπώθηκαν και δεν αποθηκεύονται αυτόματα.`
- One editable textarea contains one suggested check per line. A closed `Γιατί προτείνονται` disclosure shows bounded rationales.
- One action `Προσθήκη στο πλάνο` adopts all remaining lines into `follow_up` with `clinician_edit` provenance. The action clears the current confirmation/Preview exactly like any other edit. Deleting lines before adoption is the selection mechanism; there are no per-item confirmation questions.
- Without that action, suggestions are excluded from candidate/Preview/Save and disappear with the transient draft.

## Evidence questions

1. A synthetic compound sentence yields multiple exact excerpts across history/symptom/therapy/obligation/follow-up without invented candidate text.
2. Invalid/non-substring excerpts, ambiguous repeated matches, overlapping or duplicate spans, conflicting option/decision/completed/obligation classifications, unknown IDs, incomplete accounting, identity conflict and unsupported suggestion sources fail closed.
3. Explicit source-backed plan produces compact recommendations and pending obligations; unspoken medication structure, dates and roles remain absent.
4. Suggestions are visible but absent from candidate/Preview/Save until one bulk adoption; adoption creates clinician-edited follow-up, clears confirmation and persists only after the existing explicit Save.
5. The compact UI retains one final confirmation and does not regress manual Dia/amendment or native/local audio confirmation paths.

## Review and release

R2: this changes clinical semantics, permits bounded model-generated suggestions and can create source-backed pending obligations. One independent pre-code review and one independent exact-code fidelity review are mandatory. Synthetic data only. Exact-head CI and guarded Render/no-PHI smoke precede Product Owner fictional acceptance; no agent recording/provider request/clinical write.
