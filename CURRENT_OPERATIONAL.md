STATUS: MERGED / DEPLOYED / AUTHENTICATED PRODUCTION SMOKE FAILED — DIAGNOSTIC RE-SMOKE REQUIRED
PR: 104
Release commit: 2b6ccf56f77c646fc475318e4d76396dd95a2bee
Render deploy: dep-daj3l00ae00c73drs850 LIVE
Smoke run: 35052415555 — FAILURE
Smoke harness head: 769c5ec5b138e63d27d8f695fb311aeed0d6ab13
Writer: none — diagnostic production verification only; no runtime/config mutation authorized or required.

Observed smoke state:
- authenticated synthetic-only Medical Report V1.1 production smoke was executed against the protected live service;
- the combined production-flow job failed with exit code 1 after entering the Medical Report verification step;
- the first harness did not emit per-stage markers, so the exact failing assertion is not established by the durable logs;
- no secret value or real patient data was printed or committed;
- Medical Report V1.1 MUST NOT be labelled PRODUCTION-SMOKE-VERIFIED from this run.

Next: run one diagnostic authenticated synthetic re-smoke with explicit per-stage markers/status capture for contract/auth/privacy, V1.1 UI assets, live AI draft, live literature research, refinement evidence immutability, preview PDF and final PDF. Do not advance to another product slice until the smoke outcome is durably checkpointed.
