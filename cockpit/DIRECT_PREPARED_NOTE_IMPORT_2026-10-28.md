# Complete prepared-note import, without a second AI call

Base: `9711b920120a267b5efb806f831bea84da5619f8`. Product Owner explicitly requested implementation, merge and deployment, retaining both input paths and one final Save.

## Contract / reuse

EXTEND the existing patient-bound Visit Capture candidate, preview, atomic one-shot Save, encounter JSON and chart reader. No second persistence owner, database migration or provider is needed. Existing recording/AI/recovery remains available as the second explicit choice.

The default choice is ready clinical text. Pasting/typing automatically builds a draft through a protected deterministic `/visit-capture/import` route; it never invokes provider configuration or AI. The complete supplied prepared clinical note (not raw audio transcript) is retained verbatim in a typed optional `prepared_note` containing `text` (1–30000 characters). Sections are derived from this same text at rendering time, not independently editable/cached clinical truth. Unknown/repeated headings, free preamble, blank sections, punctuation and all body text are preserved. No clinical rewriting, normalization, truncation or inference. Markdown headings, bold headings and plain `Heading:` forms are recognized for display; unrecognized material remains visible. The original full text is copyable and is the GeSY copy/export for this path. Empty or oversized input fails privately and is never silently truncated.

`prepared_note` uses a distinct source role `prepared_clinical_note`. It is a clinician-selected manual insertion into the displayed chart, not an identity-verification claim by a model. No new checkbox. The existing selected-chart/context validation and final explicit Save remain authoritative; provenance is source-stated until Save. The reason-for-visit projection uses an explicit recognized reason heading or the neutral title `Κλινική σημείωση` without invented clinical content. No medication/test-result/diagnosis semantics are inferred into other fields.

## Obligations and AI suggestions

Only explicit unchecked `- [ ] description` lines beneath the exact `Υποχρεώσεις` heading create first-class pending items, with status=open and responsible_role=unspecified. Conditions, stated owner and timing remain in the exact description; no semantic guessing. Ordinary prose, completed `[x]` items, empty checklists, absent data and AI-proposed checks never create tasks. The prompt reserves this syntax for explicitly agreed unfulfilled actions. Duplicates are preserved, not silently merged. An unsupported count/length rejects the whole draft with a private helpful message, preserving the input. Other headings (including AI suggestions) only render as note content. AI-suggestion headings are labelled as suggestions, not clinician decisions. No automatic medication initiation or task completion.

## UI and lifecycle

Two choices: default `Έτοιμο κείμενο · χωρίς AI`; secondary `Ηχογράφηση / επεξεργασία AI`. Ready text is shown immediately as text-safe content while the deterministic server draft/preview is prepared automatically. There is no Prepare/Preview step for the clinician. One Save performs fresh preview of the exact current candidate and the existing atomic Save. Before successful import there are zero encounter/pending writes. Patient/date/input/mode changes invalidate the draft and delayed callbacks; no obsolete draft may Save. Failed import retains the supplied text; errors do not replace a prior completed audio draft. Switching input paths never triggers a provider call: AI runs only on explicit recording/Stop or the existing explicit text processing action. Recording recovery remains transient, retained on failure and unchanged.

The same full sections appear in the provisional card, saved overview, stored visit reader and exact text export. The prepared note editor uses the complete text, not generic JSON fields. Stored-note amendments reuse existing audited amendment owner and preserve identical pending bindings; no silent rewrite of older visits. Source-derived sections cannot diverge from the stored original.

Prompt is supplied in app with Copy and readable fallback, and as a local Markdown artifact. It requests complete clinical coverage, exact numbers/negation/temporality/conditions, no fabrication, no identifiers, distinct current and proposed medication, explicit agreed tasks only, and separately labelled contextual AI proposals. Empty/unmentioned headings are omitted; arbitrary additional clinical headings remain allowed.

## Review and finite evidence

R2: new deterministic entry into authoritative clinical content, retaining existing write authority. One independent pre-code design review and one post-code exact-code review, with bounded correction closure only if needed.

Acceptance: (1) no provider invoked with AI disabled; (2) comprehensive synthetic note and unknown/repeated headings survive exact import→preview→Save→independent read→amend/copy; (3) numbers, negative findings and conditional medication remain exact; (4) tasks only from reserved syntax/heading, suggestions never tasks; (5) no writes before Save, replay/stale/auth rejected; (6) actual browser desktop/mobile paste renders automatically, editing invalidates old Save and one click saves; (7) mode/date/patient/late-response races cannot cross authority; (8) existing audio success/recovery/provider-failure and manual-AI path retain behavior.

No real patient content, production clinical writes, provider calls, new secrets, database grants or storage. Rollback: before any prepared note is saved, revert the bounded runtime commit. After prepared notes exist, retain the optional note schema/render/export support while disabling new imports; never roll back to a renderer that hides them. Existing historical encounters remain readable; no data rewrite or deletion.
