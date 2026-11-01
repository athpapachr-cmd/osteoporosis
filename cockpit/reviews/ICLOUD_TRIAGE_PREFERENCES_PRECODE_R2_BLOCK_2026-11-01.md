# Independent pre-code R2 — BLOCK

Target `65b665d3eccffc31bd48826a2953fd4d30815930`; request/status checkpoint `32f395c` preserves identical design/runtime bytes. Reviewer `icloud_triage_design_r2`, 2026-11-01, read-only.

**BLOCK / P0:P1:P2=0:0:1 / COMPLETE_FOR_DECLARED_SCOPE=YES / ALL_MATERIAL_FINDINGS_CLOSED=NO.**

## P2-01 — Bind cached category projections to preference revisions

The design requires saved category moves and truthful category visibility but omits the independent shell unread cache and a shared cache/revision contract. Reachable case: peek has cached an empty clinical list because X is Administrative; the doctor saves X as Clinical in full inbox; reopening the icon within its TTL still shows cached empty results/badge. Authentication alone does not establish current preference state. An older in-flight response can also repopulate an invalidated cache.

Evidence: design lines54,76,82; `static/cockpit/doctor-shell.js:412–417,532–576`; `static/cockpit/icloud-inbox.js:510–538`.

Smallest correction: mailbox-scoped preference revisions on cached projections; confirmed Save/reset/undo/rule edit invalidates BOTH caches and advances their read generations/aborts stale requests. Require next rendered projections at confirmed revision or newer; qualify revision before cache reuse on reopen/tab return. Failed qualification uses unknown/degraded visibility. Include doctor-shell.js seam and synthetic regression that primes both caches, changes category/rule, reopens within TTL and delivers an old delayed response.

## Finite question dispositions

1. Store ownership/provisioning/rollback: SUPPORTED (design28–50,106–108; Constitution10).
2. CAS/replay/eviction/conditional undo: SUPPORTED (35–39,73–75,95); disposable two-connection implementation proof remains required.
3. Account/alias/folder/UID epoch and future-only watermark: SUPPORTED (34,41,65–67; mail source110–122,343–350,409–417).
4. Precedence, review vs reset, opt-in conflicts, terms and filtering: SUPPORTED (54–67,82,90–94,97).
5. Auth/privacy/failure/cache: BLOCK only for P2-01; other reuse contracts supported.
6. Finite clinician-visible acceptance: SUPPORTED WITH SAME BLOCKING CORRECTION; add explicit cache regression.

No implementation, edits, test reruns, live provider/database access, publication, merge or deploy by reviewer. Original gate stops at this bounded finding. Next bounded design correction then ONE independent delta plus affected cumulative closure; supported unrelated questions reused.
