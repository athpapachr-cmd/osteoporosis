# Visit Capture — Product Owner smoke UX correction, independent R1

Date: 2026-10-10. **MODE: fresh independent READ-ONLY, ONE bounded post-code implementation-fidelity review.**

Repository: `athpapachr-cmd/osteoporosis`
Implementation PR: [#140](https://github.com/athpapachr-cmd/osteoporosis/pull/140) (DRAFT, no merge/deploy authority)
Branch: `fix/cockpit-visit-capture-smoke-ux-2026-10-10`
Base/frozen production main at correction activation: `1febcfa27096df20f6f4bdeb3ccee15158634fc3`
Exact tested **implementation+test+CI head**: `4c395b08e60334d1801dba6f611472a62238faa9`
Further checkpoint `f6eb5772b643768181403c65e7ab44d9c8ac1ca2` only prepended workstream CURRENT.

## Task

Review only the bounded UI correction prompted by first Product Owner smoke findings:
1. Authenticated login panel remains visible; ensure proper hidden attribute vs CSS, automatic collapse, and usable side toggle to reopen.
2. Demo was blocked by nonexistent `SYN-001`; make **default no-patient preview** work using synthetically generated or de-identified plaintext, without creating a patient, obtaining a clinical context, or invoking a clinical write. Confirm demo switches off Save even after a prior record-mode patient context.
3. Copy-and-paste Greek Dia prompt is accessible, instructs three stable section headings, and parsing safely displays Snapshot/Visit Brief/Encounter Detail without fabricated content or HTML execution.

Review directly affected real-record reuse as an affected regression: new protected list picker selects only an existing patient, and the established `/clinical/visit-capture/context` + server-side `/preview` + explicit `/save` retain patient/context/version, auth and signed immutability semantics. Do not reinterpret first-code as a live Dia/Heidi/GESY processor. The new picker currently browses up to 100 latest patients from the existing endpoint and filters locally; report any material UX mismatch without inventing missing server search features.

## Immutable changed file blobs

- `static/cockpit/visit-capture/index.html` — `c702544e6f241d23b8cdd8fde9fbb1ff028f6d63`
- `static/cockpit/visit-capture/styles.css` — `1b46620816deb5442be57bd3ca5ae4c1bfb923fa`
- `static/cockpit/visit-capture/app.js` — `36cbb1ae8a7d7e7e2b43406dcf7e1eb2445b0526`
- `test_visit_capture_ui.py` — `e4359506635d84e5b42f53d9fbf054984e86fbaf`
- `test_visit_capture_smoke_ui.js` — `6012061e663e65f62b4a9505334e17a97eaed64e`
- `.github/workflows/visit-capture-tests.yml` — `5a54a3fda44fb4369330bf996787331e1138b1cb`

No `clinical_data.py`, API schema, backend auth, provider or persistence changes.

## Evidence to reuse — do not rerun merely for reassurance

- Focused Visit Capture **SUCCESS** at exact implementation head: GitHub Actions [38019110930](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38019110930) — Python and JS syntax, Node no-patient/auth/picker simulation, focused Visit Capture + Home/UI tests, diff hygiene.
- After docs-only workstream CURRENT update: Visit Capture [38019150559](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38019150559) **SUCCESS**, canonical-impact [38019150520](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38019150520) **SUCCESS**, Home [38019150534](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/38019150534) SUCCESS.
- Clinical Learning L0 design-only scope assertion may FAIL for legitimate non-L0 PR file changes, an inherited inapplicable diff-scope check; inspect actual failures separately, do not label all CI green. Other owner gates are unaffected.

## Authority and exclusions

This is an R1 because it only changes browser affordance and local text preview; **patient selection authority and clinical mutation stay with pre-existing protected backend**, but their UI invocation is directly affected and must be checked. If source proves a changed identity/write authority or unsafe path, return material BLOCK with smallest correction and reclassification as needed. The prior Visit Capture pre/post-code R2 and P2-F2-01 closure stay CLOSED and their unaffected PASS reused.

No implementation, edits, patient data, live provider actions, merge, deployment, general re-architecture, unrelated clinical review or repetition of parent R2.

## Finite outcome

For each of **A1 authenticated collapse**, **A2 no-patient preview/no write**, **A3 Dia text flow**, **A4 existing-patient record mode regression**, return PASS/BLOCK/UNKNOWN with exact evidence. State ONE global finite verdict: `PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 0:0:0` if no material findings, otherwise bounded BLOCK or PARTIAL/UNKNOWN. STOP.
