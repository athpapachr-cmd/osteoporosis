# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / repo-wide writer lock

> **STATUS:** PR-1 transcript extraction release engineering complete; identifiable-transcript privacy/provider gate OPEN; release HOLD.
> **Reconciled:** 2026-10-03 Asia/Nicosia from fresh remote `main` for D1.1; the existing 2026-10-01 PR-1 branch checkpoint remains unchanged.
> **D1 activation/release base `main`:** `87aedad3ad512e4b17a1eb737f0ff8302857aff2`; every fresh session must verify the current remote `main` per `AGENTS.md` before mutation.
> **Active primary slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`; design owner: `SLICE_PLAN_CURRENT.md`.
> **PR-1 branch:** `feat/pr1-transcript-capture-v1-2026-09-16` at `0b45a4c96a7cb9a96893cfa3f14a1708f25e5345` (fresh remote branch check).
> **Writer lock:** bounded parallel **COCKPIT D1.1 GLOBAL APPOINTMENT CONTEXT** implementation candidate in POST-CODE REVIEW / RELEASE HOLD; implementation writer released after focused tests, branch `feat/cockpit-d1-1-global-context-2026-10-03`, base fresh remote main `d5da6271567a4141b2708d9fa12e673dfce37131`. Product Owner explicitly authorized implementation against corrected design blob `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33` and supplied independent pre-code closure PASS on 2026-10-03. Scope: design §11 runtime/tests and D1.1 checkpoint/review artifacts. PR-1 release HOLD remains. No merge/deploy or Visit Brief runtime authority.
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

**Parallel Cockpit D1 authority:** on 2026-10-01 the Product Owner confirmed the plain-language D1 step and instructed “ξεκίνα με D1”. D1 is classified **R1** under `PROCEDURES.md`: bounded authenticated read-only Home projection over the existing Clinical Calendar endpoint, with no new clinical/identity/write authority. The required review chain is now CLOSED: final runtime/product behavior was accepted, the two stale `cockpit/CURRENT.md` statements were corrected as an R0 canonical-only delta, and the automatic CI on that docs-only head is clean for every applicable D1/Cockpit gate. The Product Owner has now explicitly instructed lawful PR #130 merge/release if that state remains clean. D1 release is complete: PR #130 squash-merged as `db42903e08b11dfe52d427a410f8b15547fb5507`; Render deploy `dep-davbs48473hc73eust00` is LIVE from `794d96dc4e0676691a4d6333b8f47bdd335cdb6c`. No D1 production smoke beyond deploy health is claimed here. That D1-release checkpoint originally proposed D2 next (historical). Current priority is **D1.1 → floating Visit Brief / Patient Context**; D2 is deferred as supporting context and has no implementation authority. Do not start D2 implementation before D1 release is durable.

**Parallel Cockpit D1.1 authority:** Product Owner supplied independent pre-code closure **PASS / COMPLETE_FOR_DECLARED_SCOPE / PRE-01 CLOSED / PRE-02 CLOSED / P0:P1:P2=0:0:0 / IMPLEMENTATION MAY START=YES** against corrected design blob `261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33` (design branch head `ba1f7eafb500986ad07b2e6b4320ac982896eb77`). Q1–Q3 supported, material findings NONE, missing evidence NONE, stop reason Q1–Q3 disposed. Evidence: `cockpit/reviews/D1_1_PRECODE_CLOSURE_PASS_2026-10-03.md`. The pre-code review chain is CLOSED; no repeat design review. R2 implementation is now authorized by the Product Owner's current instruction, exactly to unchanged corrected design. Reuse producer/feed/auth, extend normalized store, preserve weekly filter and legacy discard, minimize every effective other write, add bounded global projection and rebind Home. D1.1 implementation and focused §12 checks are complete. Evidence: 30 focused Calendar/snapshot/Home/shared Surgery UI tests PASS (Python 3.12.2, 1.16s); Python/JavaScript syntax, G4 navigation and diff hygiene PASS. Runtime/test candidate is this checkpoint commit; its immutable SHA will be recorded in the following docs-only review request checkpoint. Next: prepare one independent post-code exact-implementation-head fidelity request and hold runtime bytes stable. No review has been launched and no post-code PASS is claimed. Merge/deploy require that review PASS plus separately confirmed release authority. Visit Brief starts only after D1.1 closure/release.

## Parallel workstreams — pointers, not lock transfer

- **Visit Intelligence P0-V0:** `programme/OST-VISIT-INTELLIGENCE/CURRENT.md` exists on local branch head `b22fd5c610d20baf0f8ef3384ca16472cd2d1f92`, not yet on `main`, and owns its contract/projector checkpoint there. A cumulative PASS was reported in the Product Owner handback; its branch-local CURRENT still requests that review, so its coordinator must reconcile the result and make a separate release decision. This root PR-1 HOLD does not grant Visit Intelligence release or UI authority.
- **OST-UI / Osteoporosis Product Reconstruction:** `programme/OST-UI/PROJECT-INDEX.md` and `CURRENT.md` are on the branch-local R4 lineage, absent from `main`. R4 is complete at `5cf4e98bfeb439c3cef6aecae70d48efd07612b4`; one programme coordinator synthesis is pending. No OST-UI runtime writer or prototype implementation authority exists. The synthesis must reconcile the already expressed Product Owner direction before any prototype decision; it does not replace this PR-1 root NOW.
- **Cockpit:** `cockpit/CURRENT.md` owns released Home/Calendar/Surgery state and the deployed D1 checkpoint. Product Owner production use on 2026-10-02 identified a D1 global-source/day-window defect: Cockpit reads the Osteoporosis Clinical Calendar while Reception already holds the global schedule. **D1.1 global appointment-context source/window correction is next; then Visit Brief / Patient Context. D2 communication is deferred as supporting Visit Brief context.** D1.1 has the bounded runtime writer/authority above; Visit Brief and D2 remain deferred.
- **PHYSIO:** `programme/PHYSIO/PROJECT-INDEX.md` and `programme/PHYSIO/CURRENT.md` own the read-only P1 evidence work; no PHYSIO runtime writer is recorded there.
- **OST-CLINICAL:** `programme/OST-CLINICAL/CURRENT.md` owns the completed S1 checkpoint and any local follow-through.

`programme/MASTER-PROJECT-REGISTRY.md` provides navigation across these workstreams. It is never an authority for writer locks, releases or product state over the owning CURRENT files.
