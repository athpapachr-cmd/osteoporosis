STATUS: MERGED / DEPLOYED / AUTHENTICATED PRODUCTION SMOKE PARTIALLY VERIFIED — TARGETED RESEARCH RE-SMOKE REQUIRED
PR: 104
Release commit: 2b6ccf56f77c646fc475318e4d76396dd95a2bee
Render deploy: dep-daj3l00ae00c73drs850 LIVE
Initial smoke run: 35052415555 — FAILURE, stage not isolated
Diagnostic smoke run: 35052740386 — FAILURE, exact findings isolated
Diagnostic smoke harness head: 7bf9a283abcef5bf25e1702859cca5f24d18b615
Writer: none — diagnostic production verification only; no runtime/config mutation authorized at this checkpoint.

Verified live in diagnostic run 35052740386:
- protected contract/auth: unauthenticated 401, authenticated 200;
- contract version `medical_report_v1_1`, AI enabled/configured/PHI-approved, clinician profile configured;
- no patient-case DB/browser case storage/autosave/source-file/signature/refinement-thread persistence;
- deployed V1.1 source/refinement assets and no browser persistence APIs in tested assets;
- live GPT-5.6 draft: HTTP 200, 6122 total tokens, structured source/evidence/timeline/report sections returned;
- referral-only synthetic MRI request did not become an `imaging_finding`;
- live research endpoint: HTTP 200, non-empty 1788-character synthesis, GPT-5.6, 5309 total tokens, but this referral-only analysis produced 0 citations and 0 captured web-search queries;
- live refinement: HTTP 200, GPT-5.6, 7683 total tokens, one proposed/accepted clinician resolution; evidence items and source summaries remained unchanged;
- preview PDF and final PDF: HTTP 200, valid PDF signature, 789144 bytes each.

Harness-only findings:
- the diagnostic workflow contained two malformed jq precedence expressions for research/refinement usage checks; the emitted model/token metadata proves these were harness assertion defects, not runtime usage failures.

Unresolved production-verification question:
- `0 citations / 0 queries` on the referral-only synthetic analysis is not yet classified as a product defect because that analysis may contain no material prognosis/research question. It must not be silently accepted either.

Next: run one targeted authenticated synthetic research smoke using an explicit generalized diagnosis/prognosis question. Require HTTP 200, non-empty synthesis, at least one real web-search call/query, at least one URL citation, and positive usage. If that passes, correct the two harness-only jq assertions and close Medical Report V1.1 production smoke/canonicals. If targeted research still returns no web-search/citation evidence, checkpoint as a real research-path defect before any runtime correction.
