# Clinical Inbox B1 — ONE independent affected POST-CODE R2 (synthetic persistence)

**Date:** 2026-10-25 Asia/Nicosia. **MODE:** fresh independent READ-ONLY affected POST-CODE R2, one bounded pass and STOP. No implementation/fixes, additional architecture review chain, production patient data, DDL/grants execution, Aiven/Render/config, provider login, merge/deploy, credentials or external communications.

**Repository:** `athpapachr-cmd/osteoporosis`
**Implementation:** [PR #215](https://github.com/athpapachr-cmd/osteoporosis/pull/215), DRAFT / UNMERGED, **stacked on B0 PR #214** (also DRAFT/unmerged).
**Production main at implementation start:** `3204bc7722b5bb45241c5b23d788f6f51c6fb502`.
**Frozen B0 base / PR #214 head:** `816d6dc9079909ddf82bce082ca8c8441dd38756`, design blob `16fcdf12d41553d93f7a75b934e4c0b722605094`.
**Frozen B1 implementation + tests + migration + CI head:** `eb21a23282075842842a8ddee3596906f504bb76`. Review the EXACT source blobs:
- `clinical_attention_b1.py` = `8fa9a52fcbbfc09644beeb88771fcab52eb9bf55`
- `test_clinical_attention_b1_postgres.py` = `e05fbcdc1862c24c66b00745d102cc83ce6e1fdd`
- `cockpit/migrations/20261025_clinical_attention_b1_empty.sql` = `94278b73e62acc3372ec870b9d7a52451487520b`
- `.github/workflows/clinical-inbox-b1-postgres.yml` = `ed42c32f5e0d4be9a706559071f40a24e503acf2`

Later PR HEAD may contain **review request and CURRENT-only docs**, not substantive B1 changes. Compare exact latest blobs before handback; any change to code/tests/migration/CI means re-evaluate the affected delta rather than silently reusing old CI.

## Independent evidence to reuse

- B0 independently reviewed affected PRE-CODE: **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2=0:0:0** (the Product Owner then said “Προχωράμε” specifically to activate B1). Reuse the settled earlier #210/Phase-A/#208/#209/#212/#213 safety and UX oracles. Do not reopen unchanged synthetic clinical UI or Visit Brief behavior.
- [B1 synthetic PostgreSQL 17 push run 40157316649](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40157316649): **SUCCESS**, including Python compile, disposable PostgreSQL migration rollback / least grant, two-actor concurrency, CAS/replay, source revision reset, role audit, storage and diff hygiene.
- [B1 matching PR run 40157319711](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40157319711): **SUCCESS**. [Canonical guard 40157319644](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40157319644): **SUCCESS**.
- Two earlier failing runs are **superseded author correction evidence**, not proof of failure on the frozen head: `40157154155` identified a parent-row/event FK flush ordering; `40157254705` identified duplicate-source integrity raised at flush instead of being mapped to a deterministic conflict. Both were corrected inside `clinical_attention_b1.py` by commit `eb21a23`; focus on the final exact source and CI.
- All B1 tests use *fresh disposable local PostgreSQL 17* and fabricated identifiers. No live patient, GESY, mailbox, Aiven or Render run is claimed.

## Exactly six affected questions

1. **Zero production surface:** Is `clinical_attention_b1.py` absent from normal `main.py` imports/routers; does it avoid `ClinicalBase.metadata.create_all`, runtime migrations and any feature-on route? Does it truly remain OFF even if production is restarted?
2. **Synthetic-only hard fence:** Does constructor require isolated local test environment, fresh explicit schema/app-role and synthetic test opt-in, with PostgreSQL CHECK preventing any `patient_id` or non-synthetic source identity? No arbitrary PDF/email/free-text, no ADT/referral evidence or live patient record can leak into B1.
3. **Actor and role boundary:** Does server storage code refuse role-wide `clinic-clinician`, legacy shared key, secretary and non-attributable identities; audit attributable view/denial; preserve existing separate secretary Visit Brief read unchanged? Do not misrepresent the synthetic actor as qualified for B2 real patients.
4. **Persistence/grant fidelity:** Does one explicit transactional migration start with strict role/DB/schema preflight, create only empty additive tables, verify privileges, forbid app CREATE/DELETE and enforce append-only event R/I without startup DDL? Does malformed migration roll back, and does a separate SQLAlchemy owner (not ClinicalBase) keep app bootstrap unchanged?
5. **State and concurrency:** Is each allowed mutation + audit atomic, PostgreSQL serializable via row lock/CAS or equivalent under concurrent tabs, idempotent for exact replay but 409 for old/conflicting keys, correct same-subject source revision invalidation, and durable across new sessions/processes? Do mixed sources and weak/unlinked items remain unprojected, without medical-result acceptance, ClinicalPending mutation or sending?
6. **Practical release stop:** Is B1 limited to disposable Postgres/CI feature OFF, no real identity/retention approval, no schema execution in production, no Inbox/Brief/UI/provider binding? Are B2 identifiable real actor, real issuer proof, storage/backups/retention, new integration gate and clinician activation explicitly separate? Does the stacked draft PR remain unmerged?

## Output and STOP

Return `PASS|BLOCK|UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE|PARTIAL`, `P0:P1:P2`, the SIX question dispositions, current freshly verified main/PR #214 base/PR #215 HEAD, all FOUR exact substantive blobs, focused source/test/CI evidence and whether **B1 is independently safe as an offline synthetic-only implementation candidate for subsequent separate release decision**.

If BLOCK: identify **one concrete reachable material flaw** and smallest affected correction, not a speculative future B2 redesign. If UNKNOWN: say exact missing evidence and narrow check. STOP READ-ONLY. Independent reviewer cannot grant merge/deploy, production migration, real patient identity, staff auth activation or mailbox connections. A PASS on this B1 branch is **not** authority to run B2 or enable real Inbox.
