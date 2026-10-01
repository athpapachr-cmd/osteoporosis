# CURRENT_OPERATIONAL.md — Clinical Excellence operational NOW / repo-wide writer lock

> **STATUS:** PR-1 transcript extraction release engineering complete; identifiable-transcript privacy/provider gate OPEN; release HOLD.
> **Reconciled:** 2026-10-01 Asia/Nicosia from fresh remote `main` and the exact PR-1 remote branch checkpoint.
> **Verified remote `main`:** `63e903e05c1bfe22ca925374b8994355f6c92baf`.
> **Active primary slice:** `PR-1-TRANSCRIPT-INTAKE-CANDIDATE-EXTRACTION-V1-2026-09-16`; design owner: `SLICE_PLAN_CURRENT.md`.
> **PR-1 branch:** `feat/pr1-transcript-capture-v1-2026-09-16` at `0b45a4c96a7cb9a96893cfa3f14a1708f25e5345` (fresh remote branch check).
> **Writer lock:** no active overlapping PR-1 runtime writer at that branch checkpoint. The bounded release-engineering writer was released. A new writer must claim an exact scope here before overlap.
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

## Parallel workstreams — pointers, not lock transfer

- **Visit Intelligence P0-V0:** `programme/OST-VISIT-INTELLIGENCE/CURRENT.md` exists on local branch head `b22fd5c610d20baf0f8ef3384ca16472cd2d1f92`, not yet on `main`, and owns its contract/projector checkpoint there. A cumulative PASS was reported in the Product Owner handback; its branch-local CURRENT still requests that review, so its coordinator must reconcile the result and make a separate release decision. This root PR-1 HOLD does not grant Visit Intelligence release or UI authority.
- **OST-UI / Osteoporosis Product Reconstruction:** `programme/OST-UI/PROJECT-INDEX.md` and `CURRENT.md` are on the branch-local R4 lineage, absent from `main`. R4 is complete at `5cf4e98bfeb439c3cef6aecae70d48efd07612b4`; one programme coordinator synthesis is pending. No OST-UI runtime writer or prototype implementation authority exists. The synthesis must reconcile the already expressed Product Owner direction before any prototype decision; it does not replace this PR-1 root NOW.
- **Cockpit:** `cockpit/CURRENT.md` owns released Home/Calendar/Surgery state and the next bounded Today Context Strip V1 read projection. Visit Brief / What Changed remain downstream of P0-V0 release and affected OST-UI decisions. `cockpit/PRODUCT_CONSTITUTION.md` owns the Cockpit product boundary.
- **PHYSIO:** `programme/PHYSIO/PROJECT-INDEX.md` and `programme/PHYSIO/CURRENT.md` own the read-only P1 evidence work; no PHYSIO runtime writer is recorded there.
- **OST-CLINICAL:** `programme/OST-CLINICAL/CURRENT.md` owns the completed S1 checkpoint and any local follow-through.

`programme/MASTER-PROJECT-REGISTRY.md` provides navigation across these workstreams. It is never an authority for writer locks, releases or product state over the owning CURRENT files.

## Parallel OST-UI Prototype 1 runtime writer — 2026-10-01

Product Owner authorized the bounded Denosumab Longitudinal Adaptive Visit implementation after accepting the OST-UI synthesis/contract at documentation head `fc97c1ffe6743895098d627f2798b50fb166fff4`. Fresh remote `main` is `87aedad3ad512e4b17a1eb737f0ff8302857aff2`. Implementation branch `feat/ost-ui-proto1-denosumab-adaptive-visit-2026-10-01` starts from that exact main.

**Active writer scope:** OST-UI Prototype 1 in-visit read-only projection, adaptive presentation, source-editor navigation, transient assisted reconciliation, focused synthetic tests and this checkpoint. The writer may make bounded changes to the existing G1/G2/G3 UI handoff only to expose already computed state. PR-1 transcript code, clinical rule registries, protected persistence APIs, Cockpit patient disclosure and other workstreams remain outside scope. The PR-1 release HOLD and privacy gate remain unchanged.

**Owner/interface freeze:** protected patient/encounter/lab IDs and completed/amended rows remain authoritative; Step-4 episodes, administrations, decisions and tasks remain source owners. G1 and G2 retain derived chronology/guidance authority. Plans, actual events, derived due, decisions and obligations stay visibly distinct. Historical continuity suggestions are provisional unless an authorized existing owner persists clinician-confirmed links. The aggregate Cockpit Home remains aggregate; only a shared projection interface and synthetic protected-context test are allowed before the separate patient-link/privacy contract.

**Status:** bounded implementation candidate and deterministic synthetic tests complete; rendered browser walkthrough unverified. Writer scope released for handback. **Next:** run a protected synthetic browser walkthrough in an approved environment, then clinician usability assessment. Durable historical-link confirmation remains an owner-path dependency. No merge or deploy.
