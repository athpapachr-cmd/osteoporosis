# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / repo-wide writer lock

> **Visit Capture UI checkpoint (2026-10-08):** dedicated Cockpit surface added through `b823fe227dcc308d14a8b618c32e3b73774470c5` / `f80efdb9b886dbe8111554456192b072f4c02b3b` / `54cdddef26e9c04ccbf2c409ae2f8e29dc1b0010`. It performs explicit protected-patient confirmation, accepts one transient structured candidate, requests server preview automatically, renders Snapshot / Visit Brief / Encounter Detail as projections, and exposes one explicit Save. No live source/provider access. Runtime/test evidence is still NOT CLAIMED. Writer remains ACTIVE. Exact next action: wire the bounded Cockpit entry point and add focused synthetic backend/UI tests; then run CI/evidence.


> **Visit Capture backend checkpoint (2026-10-08):** first implementation transition committed at `9e780de18383fd244862187cb9be967c0a2c87cd`. `clinical_data.py` now contains the bounded VisitCaptureCandidateV1 contract, transient patient-bound capture contexts, deterministic 3-level preview, one explicit Save into existing `clinical_encounters`, separate `clinical_pending_items`, and server-side immutability for signed Visit Capture encounters. No frontend/runtime evidence yet; no tests claimed; writer remains ACTIVE for the reviewed slice. Exact next action: implement only the dedicated Cockpit Visit Capture UI and focused synthetic tests, then run finite evidence.


> **2026-10-08 Visit Capture implementation START — Product Owner APPROVED:** direct approval “Εγκρίνω” received after independent delta R2 PASS / 0:0:0. Active bounded writer: this coordinator on branch `feat/cockpit-visit-capture-v1-2026-10-08`, based on reviewed docs head `5bc2a505758de6b24e82667bb84580d144f96b9c`. Allowed mutation: Visit Capture synthetic/manual first-code only — existing protected encounter owner, stale patient/context guard, transient VisitCaptureCandidateV1 preview, one explicit Save, server-side post-signoff immutability for Visit-Capture-signed encounters, and the minimum bounded first-class pending persistence required by the reviewed contract; focused tests and implementation checkpoint docs. Forbidden: live Dia/Heidi/GESY/Gmail/Zadarma, patient data, lab acceptance, Calendar/Reception mutation, merge/deploy/release. Exact next action: inspect the existing runtime seams, implement only the reviewed first-code boundary, run finite focused evidence, checkpoint candidate, release writer into post-code R2 HOLD.


> **2026-10-08 Visit Capture independent R2 closeout:** Product Owner supplied independent PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0 for exact frozen delta. Receipt: `cockpit/reviews/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_R2_PASS_RECEIPT_2026-10-08.md`. Review chain CLOSED; no correction/re-review. Next: Product Owner first-code checkpoint then separately authorized narrow synthetic/manual implementation writer. NO current runtime write/merge/deploy/live patient authority. This statement supersedes prior pending-delta-R2 next-action checkpoint.


> **Cockpit Visit Capture delta checkpoint (2026-10-08):** the independent Visit Brief + Clinical Inbox R2 handback has been received as **PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0** and checkpointed at `cockpit/reviews/VISIT_BRIEF_CLINICAL_INBOX_PRECODE_R2_PASS_RECEIPT_2026-10-08.md`. The prior Q1–Q6 review is CLOSED and must not be reopened without a direct new contradiction.
> **New material delta:** Product Owner clarified the real post-visit workflow inside Dia: `Heidi today + GESY today + previous GESY → Dia candidate insertion into Cockpit → clinician review → ONE Save`. This introduces a protected clinical encounter write boundary not covered by the prior read-oriented A boundary.
> **Frozen delta design:** `cockpit/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_2026-10-08.md`, blob `7e19ce29a3c33a0d06e6c0ec485952f41c0ecf72`.
> **Prepared delta R2 request:** `cockpit/reviews/VISIT_CAPTURE_DIA_HEIDI_GESY_DELTA_PRECODE_REVIEW_REQUEST_2026-10-08.md`, blob `2e169a870d5ed04823c09d810f305dd3881defac`.
> **Writer/authority:** design/checkpoint writer RELEASED after freezing the delta. No runtime/schema/test/provider writer is active for Visit Capture. No Gmail/Zadarma/live Dia/Heidi/GESY/patient-data authority is granted.
> **Exact next Cockpit action:** ONE fresh independent read-only R2 of the Visit Capture delta request above. If PASS / 0:0:0, stop review and return for separate first-code writer authorization. If BLOCK, one smallest correction + one affected closure review only.
> **Supersession:** the older 2026-10-07 top checkpoint below is historical for next-action purposes; its original parent R2 request has now been completed and PASS-received.

> **Cockpit checkpoint (2026-10-07):** Calendar Unification = RELEASED / PRODUCT OWNER SMOKE VERIFIED. Evidence is the Product Owner's direct continuation instruction and prior smoke handback, not an agent-run production test. Independent Calendar review chain remains CLOSED. Visit Brief + independent Clinical Inbox design v2 is prepared for one R2 pre-code review; request PREPARED / NOT DISPATCHED / NO VERDICT; runtime is NOT STARTED.
> **Design branch/base:** `docs/cockpit-visit-brief-clinical-inbox-2026-10-07` / `045798dfa28612b268f16a99262c6ecc9ca4829d`; scope is Cockpit design/checkpoints plus affected roadmap/phase/constitution/history. No runtime/schema/test/config/provider write authority; no overlap with PR-1.
> **Design/request identities:** design blob `8ee79a96350f9ac42059eeb2a3e836cbe9d11245`; request blob `5bb6de5fce76170ca65d0422a4d6ed8272aabc20`. Author consistency audit is not an independent R2 verdict.
> **Design PR:** [#138](https://github.com/athpapachr-cmd/osteoporosis/pull/138) OPEN / DRAFT / R2 PRE-CODE HOLD; design-bearing published head `e305a9678183cab84df7825a4f59c5d4a841e113`. Any later checkpoint commit may change only publication metadata; exact design/request blobs above remain the review target. NOT MERGED / NOT DEPLOYED.
> **Exact next Cockpit action:** one independent read-only R2 of `cockpit/reviews/VISIT_BRIEF_CLINICAL_INBOX_PRECODE_REVIEW_REQUEST_2026-10-07.md` on the declared blobs. Request is not dispatched by this task. No implementation follows without required contract/gate closure and separate authority.
> **Outgoing checkpoint provenance:** exact previous root and Cockpit CURRENT bytes plus prior design/request are preserved under `cockpit/archive/checkpoints/2026-10-07-pre-clinical-inbox-replan/`, from the verified base above. They are historical, not current next-action authority.

> **STATUS:** PR-1 transcript extraction release engineering complete; identifiable-transcript privacy/provider gate OPEN; release HOLD.
> **Reconciled:** 2026-10-07 Asia/Nicosia; Calendar smoke closed and Cockpit design candidate checkpointed in R2 pre-code HOLD. The independent PR-1 branch checkpoint is unchanged.
> **D1 activation/release base `main`:** `87aedad3ad512e4b17a1eb737f0ff8302857aff2`; every fresh session must verify the current remote `main` per `AGENTS.md` before mutation.
> **Active primary slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`; design owner: `SLICE_PLAN_CURRENT.md`.
> **PR-1 branch:** `feat/pr1-transcript-capture-v1-2026-09-16` at `0b45a4c96a7cb9a96893cfa3f14a1708f25e5345` (last recorded remote branch check; PR-1 not rechecked by this bounded Cockpit task).
> **Writer lock:** Cockpit design writer RELEASED into R2 pre-code HOLD; no Cockpit or PR-1 runtime writer. PR-1 remains on its separate privacy/provider release HOLD. Visit Brief/Clinical Inbox/Gmail/Zadarma implementation is not authorized.
> **Release:** no PR-1 release PR observed among open PRs on 2026-10-01; not merged, deployed or enabled for identifiable transcripts.


## Current source and evidence

The previous root NOW was the 2026-09-16 activation checkpoint and incorrectly said the runtime branch had not been created. PR-1's later branch-local `CURRENT_OPERATIONAL.md` at `0b45a4c` is the detailed PR-1 evidence record. This root reconciliation brings its current state onto the verified `main`-based governance branch without copying its long implementation diary. The branch's full history remains evidence, not a competing repo-wide lock.

```text
H13 semantic promotion review: PASS_TO_RELEASE_ENGINEERING
H13 qualified synthetic provider run: 22/22 PASS; PHI approval=false
release-engineering implementation head: c16c81053c8596bb106555b9c8e5ef93bc686034
complete PR-1 gate on that head: 36860809239 SUCCESS
final docs checkpoint branch head: 0b45a4c96a7cb9a96893cfa3f14a1708f25e5345
current blocker: identifiable-transcript privacy/provider decision still OPEN
release gate: HOLD
```

The implementation branch has reconciled current `main`; `main.py` retained Cockpit root/surgery ownership and PR-1's protected transcript router. Its H-05 provider offload and transient browser lifecycle evidence are branch-local. These facts do not mean PR-1 is in production or authorized for real-patient use.

## Authority and exact next action

The Product Owner authorized bounded PR-1 implementation and synthetic qualification. That does not authorize identifiable transcript processing, release PR approval, merge, deployment, PR-2 authoritative writes or a real-patient pilot.

**Next lawful PR-1 lifecycle action:** keep release HOLD while the identifiable-transcript provider/privacy gate is decided with explicit Product Owner authority. Once that gate is closed, fresh-check `main`, PR-1 branch head, applicable CI and merge/release scope; checkpoint the decision here before opening or advancing the release PR. If the gate remains open, do not advance release. No new independent semantic review is required solely for this docs/canonical reconciliation; a material new behavior or risk receives its own bounded classification under `PROCEDURES.md`.

**Parallel Cockpit D1 authority:** on 2026-10-01 the Product Owner confirmed the plain-language D1 step and instructed “ξεκίνα με D1”. D1 is classified **R1** under `PROCEDURES.md`: bounded authenticated read-only Home projection over the existing Clinical Calendar endpoint, with no new clinical/identity/write authority. The required review chain is now CLOSED: final runtime/product behavior was accepted, the two stale `cockpit/CURRENT.md` statements were corrected as an R0 canonical-only delta, and the automatic CI on that docs-only head is clean for every applicable D1/Cockpit gate. The Product Owner has now explicitly instructed lawful PR #130 merge/release if that state remains clean. D1 release is complete: PR #130 squash-merged as `db42903e08b11dfe52d427a410f8b15547fb5507`; Render deploy `dep-davbs48473hc73eust00` is LIVE from `794d96dc4e0676691a4d6333b8f47bdd335cdb6c`. No D1 production smoke beyond deploy health is claimed here. That D1-release checkpoint originally proposed D2 next (historical). That D1.1 priority is historical. Current Cockpit priority is **Visit Brief + independent Clinical Inbox design-only replan** after Calendar Unification smoke; standalone D2 remains deferred and has no implementation authority. Do not start D2 implementation before D1 release is durable.

**Parallel Cockpit D1.1 release:** Reception PR #157 merged as `8d2827514457a1da54f33c6109b22e6106532301`, deployed LIVE as `dep-db25omh7lnhs73dn82o0` on Render web service `srv-d5rsvj14tr6s738bp67g`. Cockpit PR #136 merged as `12800be3a51f716b2ccd72715c6e0e637f2e43ab`, deployed LIVE as `dep-db25uae7bikc73dkrku0` on `srv-d5qfk31r0fns73di596g`. Cockpit `RECEPTION_SCHEDULE_CONTEXT_URL` was installed for the protected Reception route and config deploy `dep-db25qh7lot8c73duu8vg` was LIVE before code merge; the previously configured matching server-to-server ingest-secret pair was retained without reading or rotating secret values. All six reviewed Cockpit runtime/test blobs and Reception's full reviewed tree match merge results. D1.1 technical review PASS / 0:0:0 and CLOSED; historical daily-source HOLD request remains historical with no verdict. D1.1 is superseded as the current follow-up by the released Calendar Unification runtime. The Product Owner has now confirmed Home + weekly Calendar smoke; no separate agent smoke is claimed. Current next action is the Cockpit design-only replan above.

## Parallel workstreams — pointers, not lock transfer

- **Visit Intelligence P0-V0:** `programme/OST-VISIT-INTELLIGENCE/CURRENT.md` exists on local branch head `b22fd5c610d20baf0f8ef3384ca16472cd2d1f92`, not yet on `main`, and owns its contract/projector checkpoint there. A cumulative PASS was reported in the Product Owner handback; its branch-local CURRENT still requests that review, so its coordinator must reconcile the result and make a separate release decision. This root PR-1 HOLD does not grant Visit Intelligence release or UI authority.
- **OST-UI / Osteoporosis Product Reconstruction:** `programme/OST-UI/PROJECT-INDEX.md` and `CURRENT.md` are on the branch-local R4 lineage, absent from `main`. R4 is complete at `5cf4e98bfeb439c3cef6aecae70d48efd07612b4`; one programme coordinator synthesis is pending. No OST-UI runtime writer or prototype implementation authority exists. The synthesis must reconcile the already expressed Product Owner direction before any prototype decision; it does not replace this PR-1 root NOW.
- **Cockpit:** `cockpit/CURRENT.md` owns Calendar Unification RELEASED / PRODUCT OWNER SMOKE VERIFIED and the Visit Brief + Clinical Inbox design replan. D2 standalone implementation and all new runtime remain deferred. No lock transfers to Reception/Ops or other workstreams.
- **PHYSIO:** `programme/PHYSIO/PROJECT-INDEX.md` and `programme/PHYSIO/CURRENT.md` own the read-only P1 evidence work; no PHYSIO runtime writer is recorded there.
- **OST-CLINICAL:** `programme/OST-CLINICAL/CURRENT.md` owns the completed S1 checkpoint and any local follow-through.

`programme/MASTER-PROJECT-REGISTRY.md` provides navigation across these workstreams. It is never an authority for writer locks, releases or product state over the owning CURRENT files.
