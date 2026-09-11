# SLICE_PLAN_CURRENT.md — Step 5 functional Knee-OA prototype

> **STATUS:** IMPLEMENTATION AUTHORIZED / ACTIVE; SYNTHETIC NON-PRODUCTION ONLY.
> **Slice:** CU1-PRODUCT-KNEE-OA-PROTOTYPE-V1-20260911.
> **Branch:** feat/physio-referral-knee-oa-prototype-v1-2026-09-11.
> **Frozen parent:** 4e0e3206dd1e12a2e55a6abd18d2a6f7dbc3f7c6.
> **Verified main:** d9f312f6d2d596ec0bd4f35f6de56ad98dc34b37.
> **Authority:** product-owner progression to the explicitly named Step-5 functional prototype; no release authority.

## Scope and entrypoint

Only new files in `clinic_utilities/physio_referral_product/prototype/`, a focused workflow and supporting canonical/progress records. The runnable entrypoint is `python clinic_utilities/physio_referral_product/prototype/server.py`, binding exclusively to `127.0.0.1`. No production FastAPI/router/static registration; existing CU-1, Learning Hub, RF, database and clinical contracts remain read-only.

Read-only dependencies: real CU-1 engine validation; its existing Greek label owner; pinned Step-3 template and composition functions; pinned Step-4 evidence/suggestion functions. Reusing design functions is an explicit synthetic-prototype coupling, not a production architecture claim. Later productionization requires a separate bounded extraction/integration review. No silently substituted mock safety clearance is permitted in integration tests or delivered server.

## Data contract and isolation

`draft_id` UUID + nonnegative `revision` + fixed `package_version` + `synthetic_only=true` + bounded `state` + draft-local `dismissed` identities. Unsupported keys/types/IDs and stale candidates fail closed with sanitized errors. The product phenotype remains exactly stiffness and generic weakness; no inference of measured ROM or objective weakness. Restrictions and clinician note remain ephemeral. No raw request values in logs, URLs, analytics, browser storage or files.

Transport is local JSON POST only with a custom header, bounded body, exact loopback Host and same-origin enforcement. Static GET is allowlisted. No CORS, directory listing, remote backend, credentials or automatic source requests. Source URLs are static reviewed HTTPS links activated explicitly.

## UI and projection

Mobile-first Greek interface; neutral selection checks are distinct from the six evidence cues. Laterality and explicit diagnosis assertion remain necessary. Three evidence-backed defaults; small clinical-picture choices and progressive function disclosure; compact advanced groups. Selected optional interventions appear in the visible plan rather than being duplicated in advanced menus. Counts retain unique advanced canonical selections.

Server validation precedes text projection. Every clinical state change invalidates export immediately. A response applies only to the same draft/revision/package/request sequence. All exports share one guard; printed/copied text carries an explicit synthetic/non-clinical marker. PDF is browser print-to-PDF, not a separate server PDF generator.

One non-timed evidence message; one native modal dialog host. Info controls do not toggle selection. The mixed-source first disclosure preserves all source directions and bounded faithful Greek summaries, with original recorded wording and native strength inspectable. Greek display translation does not constitute new clinical source review. Other sources retain original recorded wording beneath explicit source/scope labels.

Manual edited text has an independent in-memory buffer. Structured changes preserve it but disable export until explicit reconciliation. Evidence viewing/dismissal does not edit it. Reset/pagehide/BFCache handling clears draft-scoped state.

## Focused evidence plan

New integration acceptance, not reassurance reruns of unchanged owners:

1. Real CU-1 validation adapter: default valid route, absent assertion/laterality, actual safety block, forged gate/ack rejection, unsupported selections, input type/size hygiene.
2. Every inherited Step-3 supported render fixture through the new adapter, checking exact Greek output; hidden walking-aid/dry-needling/weight-management remain excluded.
3. Suggestions: explicit add, changed revision/draft/package rejection, deduplication, source captions, no auto-treatment, inactive-source suppression.
4. Browser against the actual server: routine path, isolated info tap, all-source disagreement, advanced persistence, source-backed add, manual reconciliation, safety and network failure export blocking, reset and no storage.
5. Layout: 320/390/desktop widths, 200% text sizing, touch targets, focus/modal behavior, reduced motion and forced colours. Distinguish automated Chromium coverage from real Safari/VoiceOver and clinical/product-owner acceptance.
6. Scope/identity check: frozen inputs unchanged, no production source touched, exact-head focused gate and explicit writer review limitations.

An isolated mocked transport may help local visual inspection when network access prevents a checkout. Its screenshots/measurements are visual-only and must never be called real CU-1 integration evidence. CI must use the real repository/server.

## Completion and HOLD

Record implemented versus actually tested versus pending. Release writer at a clean checkpoint. Step 6 is product-owner usability/clinical-copy acceptance; Step 7 is independent multi-axis review. Neither is inherited from author tests. No PR, merge, deployment, production smoke, real clinical use, subscription system, second diagnosis, autonomous evidence updates or patient persistence.
