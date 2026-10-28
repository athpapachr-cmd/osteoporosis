# Complete prepared clinical notes — release receipt

- Product Owner explicitly authorized both input paths, full content preservation, one final Save, ChatGPT prompt, merge and deploy.
- PR #254 final head `3cb8104a9171d2b8437738d7c449fe8170f1cc78`: all17 final checks PASS; Visit Capture run40759494872 SUCCESS.
- Squash merge `a0c9ad8f024009df7dfca9a53979b9ecf10be52a`.
- Native Render `dep-dbh5adcs728c739smeog`, exact merge, LIVE2026-10-28T20:06:10Z. No duplicate manual deploy.
- Independent pre-code PASS and corrected exact-code R2 PASS / ALL_MATERIAL_FINDINGS_CLOSED / COMPLETE / 0:0:0. Source receipts in `cockpit/reviews/DIRECT_PREPARED_NOTE_*`.
- Local79 affected tests PASS; focused heading correction7 PASS; CP2 visibility-only test1 PASS. Desktop/mobile1280/390 automatic paste, full text after independent reload, exact copy/export, one Save and one amendment Save, identical task IDs, stale mode response rejection, and existing AI/audio/recovery covered. No provider calls.

## Production smoke

Three served assets SHA-256 match exact merged bytes:

- index.html: `a894715a093052cba696771971ebdb17710da84a698d07323c225f10f2ea1615`
- patient-chart.js: `defda2be565437ff853d30c67036d619a5e73a117a56993f21f23a8ad9f64477`
- doctor-shell.css: `4ed6141265911f56075bec4361338d52bffc4077bd636424958b3e6bef774584`

Anonymous import/extract/preview/save/audio-draft/audio-reprocess each401. Fresh production browser finds both input choices, ChatGPT prompt and unrestricted input retention; JavaScript errors0, capture write/provider requests0. The existing anonymous RF search request is unchanged. Release-window app ERROR and HTTP5xx queries empty. No production patient read, recording, provider call or clinical write. Production clinical Save was not exercised; functional persistence was verified against disposable synthetic local storage.

## Behavior and boundaries

Ready text imports deterministically without AI and retains the complete supplied prepared clinical note, arbitrary/repeated headings, numbers, negation and conditional instructions. Only explicit unchecked obligations under their heading become tasks; AI ideas remain labelled note content. Secondary AI recording/text processing and transient recovery remain. One final explicit Save owns clinical persistence.

Input limit30000characters; at most60 tasks of1000characters. Overlimits/ambiguous task indentation reject privately while retaining input, never truncate. No schema/config/secret/grant changes. After new notes exist, rollback must retain note schema/render/export support while disabling imports; do not revert to a reader that hides prepared notes.

Full prompt: `cockpit/CHATGPT_CLINICAL_NOTE_PROMPT.md`, exactly equal to the in-app copyable template. Technical release complete; next clinician fictional-case acceptance through either chosen input path. No claim of real-use or model clinical-quality validation.
