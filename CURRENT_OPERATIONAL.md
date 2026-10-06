# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / repo-wide writer lock

> **STATUS:** PR-1 transcript extraction release engineering complete; identifiable-transcript privacy/provider gate OPEN; release HOLD.
> **Reconciled:** 2026-10-04 Asia/Nicosia from fresh remote `main` and the exact D1.1 implementation branch; the existing 2026-10-01 PR-1 branch checkpoint remains unchanged.
> **D1 activation/release base `main`:** `87aedad3ad512e4b17a1eb737f0ff8302857aff2`; every fresh session must verify the current remote `main` per `AGENTS.md` before mutation.
> **Active primary slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`; design owner: `SLICE_PLAN_CURRENT.md`.
> **PR-1 branch:** `feat/pr1-transcript-capture-v1-2026-09-16` at `0b45a4c96a7cb9a96893cfa3f14a1708f25e5345` (fresh remote branch check).
> **Writer lock:** bounded Cockpit D1.1 release writer active for PR/merge/deploy/status only under direct Product Owner authorization of 2026-10-06. No active overlapping PR-1 or Cockpit runtime writer. Reviewed Osteoporosis runtime/test head `9dc14bbde1005aab60922e88e021518197f0654f`; Reception `b4739678016c1cd10f3afd8381e7c4de222f38ba`. Fresh Osteoporosis main `b9beca7b1e233245f9ea3429a247a1c80a69ac7c` has disjoint PHYSIO document drift; Reception main `bcfa57e0c1ca7358fb898b393e1dafa7b042c238`. Independent exact-head R2 post-code fidelity: PASS / COMPLETE_FOR_DECLARED_SCOPE / P0:P1:P2 = 0:0:0. Review chain CLOSED; old daily-source request stays historical HOLD/no verdict. Ops release-scope checkpoint `41dbf97f51dbce10d09502206c967c26a716f387`. Focused author tests: Cockpit 31 PASS; Reception 54 tests + 33 subtests PASS. Reception merge/deploy must precede Cockpit. Production smoke reserved for Product Owner; no Visit Brief authority.
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

**Parallel Cockpit D1.1 authority:** The corrected freshness design blob `00e3b4527a8ccbfc6df1a635494da136ccb9dd33` has independent R2 pre-code PASS / 0:0:0. Exact-head post-code fidelity is now PASS / COMPLETE_FOR_DECLARED_SCOPE / 0:0:0 on Reception `b4739678016c1cd10f3afd8381e7c4de222f38ba` and Cockpit `9dc14bbde1005aab60922e88e021518197f0654f`; the review chain is CLOSED. Product Owner authorized Reception-first merge/deploy, existing shared-key/URL configuration, then Cockpit merge/deploy on 2026-10-06, with production smoke reserved for Product Owner. The former daily-snapshot source/request remains historical HOLD/no verdict. Exact next action: open/check Reception release PR; proceed through P8 only if reviewed runtime bytes and required gates remain clean. Visit Brief remains deferred.

## Parallel workstreams — pointers, not lock transfer

- **Visit Intelligence P0-V0:** `programme/OST-VISIT-INTELLIGENCE/CURRENT.md` exists on local branch head `b22fd5c610d20baf0f8ef3384ca16472cd2d1f92`, not yet on `main`, and owns its contract/projector checkpoint there. A cumulative PASS was reported in the Product Owner handback; its branch-local CURRENT still requests that review, so its coordinator must reconcile the result and make a separate release decision. This root PR-1 HOLD does not grant Visit Intelligence release or UI authority.
- **OST-UI / Osteoporosis Product Reconstruction:** `programme/OST-UI/PROJECT-INDEX.md` and `CURRENT.md` are on the branch-local R4 lineage, absent from `main`. R4 is complete at `5cf4e98bfeb439c3cef6aecae70d48efd07612b4`; one programme coordinator synthesis is pending. No OST-UI runtime writer or prototype implementation authority exists. The synthesis must reconcile the already expressed Product Owner direction before any prototype decision; it does not replace this PR-1 root NOW.
- **Cockpit:** `cockpit/CURRENT.md` owns released Home/Calendar/Surgery state and D1.1 release checkpoints. D1.1 exact-head technical review is PASS / 0:0:0 and CLOSED; Product Owner authorized Reception-first release through deploy. Production smoke remains Product Owner pending. D2 and Visit Brief implementation are deferred.
- **PHYSIO:** `programme/PHYSIO/PROJECT-INDEX.md` and `programme/PHYSIO/CURRENT.md` own the read-only P1 evidence work; no PHYSIO runtime writer is recorded there.
- **OST-CLINICAL:** `programme/OST-CLINICAL/CURRENT.md` owns the completed S1 checkpoint and any local follow-through.

`programme/MASTER-PROJECT-REGISTRY.md` provides navigation across these workstreams. It is never an authority for writer locks, releases or product state over the owning CURRENT files.
