# Independent final-code R2 — BLOCK

Exact candidate `9cb58b21296786203a4aa15d11e260ff40659b8c`; reviewer `icloud_triage_design_r2`, independent of author. **BLOCK / P0:P1:P2=0:0:1 / COMPLETE_FOR_DECLARED_SCOPE=YES / ALL_MATERIAL_FINDINGS_CLOSED=NO.**

## P2-CODE-01 — Pending Save discards later unsaved input

Per-message `static/cockpit/icloud-triage.js:106–118` and rule forms135–144 disable only Save while POST is pending. Later input records a newer volatile draft and resets pending. Earlier success unconditionally deletes that draft and dispatches projection invalidation/rebuild via confirmed78–80, removing newer choices. Reachable category/rule sequence: choose -> Save -> edit while POST waits -> old Save succeeds -> newer unsaved input disappears. This violates approved uncommitted-choice preservation.

Smallest correction: lock affected edit controls/revision refresh for the entire pending operation OR use a submitted draft generation and clear only that exact generation. Account for duplicate rule forms sharing a draft key. Add actual synthetic browser delayed-preferences-POST regression for both message and rule forms; list/peek delay tests do not cover this write path.

Finite dispositions: Q1 operational store/auth/ownership/no implicit DDL SUPPORTED; Q2 CAS/rollback/replay/undo/capacity SUPPORTED; Q3 source/future rules/privacy/degraded All SUPPORTED; Q4 visible UX BLOCK for P2-CODE-01; Q5 both cache consumers and delayed-response coherence SUPPORTED; Q6 meaningful tests/provisioning SUPPORTED WITH SAME CORRECTION. JUnits inspected89PASS/1skip, backend20PASS/1skip, browser4PASS, adjacent55PASS; overlaps not added. No reviewer edits, reruns, DB/provider/production changes or publication. Stop at this bounded finding; smallest correction and ONE independent delta+affected cumulative closure next.
