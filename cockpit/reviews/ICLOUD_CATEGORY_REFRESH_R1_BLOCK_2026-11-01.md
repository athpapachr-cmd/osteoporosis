# Independent R1 exact-code review — BLOCK

Reviewer: independent `icloud_triage_design_r2`, acting as the R1 frontend fidelity reviewer. Exact target c5403052317a5c9c653ea70836b3dabbb1b6e5f8; fresh remote main d545b1fc4834d0017a864ab8a4c432eca394b6c5.

P0:P1:P2=0:0:1. COMPLETE_FOR_DECLARED_SCOPE=YES. NO_ADDITIONAL_MATERIAL_FINDING=NO (one finding). No reviewer edits/tests/DB/provider/production/publication/merge/deploy. Original PR279 R2 remains closed.

## P2-REFRESH-01

New post-await category confirmation at icloud-triage.js138–139 reads mutable pending.rule/pending.expected_revision. Already-running revision-refresh GET handler143–144 can finish during POST, reset pending=null despite currently disabled controls, then the POST commits but confirmation throws before invalidation/one-item refresh. Reachable sequence: start revision GET; start Save; earlier GET finishes/reset pending; successful POST returns.

Smallest correction: capture immutable submitted operation before await, use it for request and hint. Add actual delayed qualification-GET/across-successful-Save browser regression, release GET before successful POST response; prove confirmation and one-item refresh.

Q1 BLOCK on this race, other consecutive/replay/revision predicates supported. Q2 page/cursor/filter/selection/reset/undo and no freshness renewal SUPPORTED. Q3 hidden peek, remote stripping, fallback/generation/auth SUPPORTED. Q4 scan/concurrent/replay/late/auth tests SUPPORTED except missing earlier GET race. Both inspected XML receipts12 PASS; overlap not summed. Exactly one bounded correction and one delta+cumulative closure next.
