from pathlib import Path

FREEZE_HEAD = 'f947b12ce77db2ad1e5ff9117d7a4f794e224b60'
FREEZE_RUN = '34148333413'

# CURRENT_OPERATIONAL
p = Path('CURRENT_OPERATIONAL.md')
s = p.read_text(encoding='utf-8')
s = s.replace(
    '> **STATUS:** CLINICAL LEARNING HUB L-0 — INDEPENDENT CLOSURE PASS / CANONICAL CLOSEOUT ACTIVE / FINAL EXACT-HEAD GATE PENDING',
    '> **STATUS:** CLINICAL LEARNING HUB L-0 — CONTRACT FROZEN / COMPLETE / PR #80 OPEN — RELEASE-DESIGN HOLD',
    1,
)
s = s.replace(
    '> **ACTIVE DESIGN/CANONICAL WRITER:** ChatGPT — canonical closeout only; no material contract mutation authorized without re-review.',
    '> **ACTIVE DESIGN/CANONICAL WRITER:** NONE — L-0 contract freeze is closed; PR #80 is release-design HOLD.',
    1,
)
s = s.replace(
    '> **L-0 merge authority:** NONE in this step; bounded design PR may be opened after final exact-head gate, then HOLD.',
    '> **L-0 merge authority:** NONE — PR #80 is open and must not be merged without a separate explicit product-owner decision.',
    1,
)
s = s.replace('# 6. Final closeout gate still required', '# 6. Final clean freeze gate — PASS', 1)
s = s.replace(
    'The independent substantive review is complete, but L-0 is not yet declared COMPLETE until the status/canonical closeout commits themselves pass the same exact-head contract gate.\n\nNo material contract change may be introduced during closeout. If one is needed, the independent CLOSURE PASS must be reopened.',
    f'The clean post-closeout/pre-PR branch head `{FREEZE_HEAD}` passed the complete `Clinical Learning L0 contract gate`, run `{FREEZE_RUN}` — SUCCESS. Temporary closeout helpers were absent from that clean head. This closes the L-0 contract/design freeze evidence gate.\n\nAny future material contract change requires reopening review; status-only PR metadata does not authorize semantic mutation.',
    1,
)
old = '''PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             CANDIDATE COMPLETE
SUBSTANTIVE CONTRACT GATE            PASS — run 34147429373
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
CANONICAL CLOSEOUT                   IN PROGRESS
FINAL EXACT-HEAD GATE                PENDING
L-0 CONTRACT FROZEN / COMPLETE       NO — final gate pending
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME IMPLEMENTATION           NOT AUTHORIZED'''
new = f'''PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             FROZEN / COMPLETE
SUBSTANTIVE CONTRACT GATE            PASS — run 34147429373
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
CANONICAL CLOSEOUT                   COMPLETE
CLEAN FREEZE GATE                    PASS — {FREEZE_HEAD} / run {FREEZE_RUN}
PR #80                               OPEN / MERGEABLE / RELEASE-DESIGN HOLD
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME IMPLEMENTATION           NOT AUTHORIZED'''
if old not in s:
    raise SystemExit('CURRENT lifecycle anchor missing')
s = s.replace(old, new, 1)
start = s.index('# 8. Exact next action / HOLD')
s = s[:start] + '''# 8. Exact next action / HOLD

```text
PR #80 OPEN
→ HOLD for separate product-owner merge decision
→ if merged later: docs/design-only Render auto-deploy may follow main
→ L-1 remains separately unauthorized until explicit product-owner implementation authority
```

Current hold:

```text
NO merge of PR #80
NO L-1 runtime/database implementation
NO learning API runtime routes
NO Learning Hub production UI
NO external learning credential
NO Daily Case Review runtime
NO raw transcript persistence
NO patient-record mutation
NO Signal promotion
NO background cron
NO production config mutation
```
'''
p.write_text(s, encoding='utf-8')

# SLICE_PLAN_CURRENT
p = Path('SLICE_PLAN_CURRENT.md')
s = p.read_text(encoding='utf-8')
s = s.replace(
    '> **STATUS:** INDEPENDENT CLOSURE PASS / CONTRACT FREEZE CLOSEOUT ACTIVE / FINAL EXACT-HEAD GATE PENDING',
    '> **STATUS:** CONTRACT FROZEN / COMPLETE / INDEPENDENT CLOSURE PASS / PR #80 OPEN — RELEASE-DESIGN HOLD',
    1,
)
s = s.replace(
    '> **Runtime implementation authority:** NONE.',
    '> **Runtime implementation authority:** NONE.\n> **Design PR:** #80 OPEN; merge authority NONE.',
    1,
)
old = '''PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             CANDIDATE COMPLETE
SUBSTANTIVE MACHINE GATE             PASS — run 34147429373
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
L-0 CONTRACT FROZEN / COMPLETE       NO — final canonical exact-head gate pending
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME AUTHORITY                NONE'''
new = f'''PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             FROZEN / COMPLETE
SUBSTANTIVE MACHINE GATE             PASS — run 34147429373
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
CLEAN FREEZE GATE                    PASS — {FREEZE_HEAD} / run {FREEZE_RUN}
DESIGN PR #80                        OPEN / RELEASE-DESIGN HOLD
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME AUTHORITY                NONE'''
if old not in s:
    raise SystemExit('SLICE lifecycle anchor missing')
s = s.replace(old, new, 1)
marker = '''Exact next action:

```text
canonical closeout
→ remove temporary closeout helpers
→ final exact-head L-0 gate
→ if PASS, mark L-0 CONTRACT FROZEN / COMPLETE
→ open bounded design PR to main
→ RELEASE / DESIGN HOLD
→ HOLD for separate product-owner L-1 runtime implementation decision
```'''
replacement = '''Exact next action:

```text
PR #80 OPEN
→ RELEASE / DESIGN HOLD
→ separate product-owner merge decision required
→ after any later merge, separate product-owner L-1 runtime implementation decision required
```'''
if marker not in s:
    raise SystemExit('SLICE next-action anchor missing')
s = s.replace(marker, replacement, 1)
p.write_text(s, encoding='utf-8')

# TODO
p = Path('TODO.md')
s = p.read_text(encoding='utf-8')
s = s.replace(
    '## 1.10A Clinical Learning Hub — L-0 INDEPENDENT CLOSURE PASS / FINAL GATE PENDING',
    '## 1.10A Clinical Learning Hub — L-0 CONTRACT FROZEN / COMPLETE — PR #80 HOLD',
    1,
)
s = s.replace(
    '- [ ] L-0: complete canonical closeout + final exact-head gate + bounded design PR/HOLD before declaring merge-ready lifecycle closure.',
    f'- [x] L-0: canonical closeout complete; clean freeze head `{FREEZE_HEAD}` passed run `{FREEZE_RUN}`; bounded design PR #80 is OPEN in release-design HOLD.',
    1,
)
p.write_text(s, encoding='utf-8')

# Append-only changelog
p = Path('osteoporosis-change-log.md')
s = p.read_text(encoding='utf-8')
marker = '## 2026-09-07 — Clinical Learning Hub L-0 contract frozen; PR #80 opened in HOLD'
if marker not in s:
    s = s.rstrip() + f'''\n\n---\n\n{marker}\n\nAfter the independent L-0 CLOSURE PASS, canonical closeout was completed and all temporary closeout helpers were removed. The clean freeze head was:\n\n```text\n{FREEZE_HEAD}\n```\n\nThe full `Clinical Learning L0 contract gate` passed on that clean head at run `{FREEZE_RUN}`. The L-0 field-level learning contracts are therefore **FROZEN / COMPLETE** as a design milestone.\n\nBounded design PR #80 (`Clinical Learning Hub: freeze L-0 contracts and L-1 boundaries`) was then opened against `main`. It remains unmerged and is in **RELEASE / DESIGN HOLD**. Opening the PR does not authorize merge and does not authorize L-1 runtime implementation.\n\nLifecycle at this milestone:\n\n```text\nL-0 CONTRACT FROZEN / COMPLETE   YES\nINDEPENDENT REVIEW               CLOSURE PASS\nCLEAN FREEZE GATE                PASS\nPR #80                           OPEN\nMERGED                           NO\nL-1 IMPLEMENTATION               NOT AUTHORIZED\n```\n'''
p.write_text(s, encoding='utf-8')
