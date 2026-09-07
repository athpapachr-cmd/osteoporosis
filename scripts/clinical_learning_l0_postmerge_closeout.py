from pathlib import Path

MERGE_SHA = "43ed3090c5d4fe849e5f500edf22182fade36cd8"
DEPLOY_ID = "dep-daffttrbc2fs73d7v45g"
PR = "#80"

# CURRENT_OPERATIONAL.md
p = Path("CURRENT_OPERATIONAL.md")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "> **STATUS:** CLINICAL LEARNING HUB L-0 — CONTRACT FROZEN / COMPLETE / PR #80 OPEN — RELEASE-DESIGN HOLD",
    "> **STATUS:** CLINICAL LEARNING HUB L-0 — CONTRACT FROZEN / COMPLETE / MERGED TO MAIN — L-1 AUTHORIZATION HOLD",
    1,
)
s = s.replace(
    "> **Fresh verified `main` / merge base:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`.",
    f"> **Current verified `main`:** `{MERGE_SHA}` — PR #80 squash merge.\n> **Render auto-deploy:** `{DEPLOY_ID}` — LIVE; design/contracts/canonical change only, no new learning runtime.",
    1,
)
s = s.replace(
    "> **Active branch:** `design/clinical-learning-l0-contract-freeze-2026-09-07`.",
    "> **Active branch:** NONE — L-0 merged and closed; no L-1 writer exists.",
    1,
)
s = s.replace(
    "> **ACTIVE DESIGN/CANONICAL WRITER:** NONE — L-0 contract freeze is closed; PR #80 is release-design HOLD.",
    "> **ACTIVE DESIGN/CANONICAL WRITER:** NONE — L-0 contract freeze is merged/closed.",
    1,
)
s = s.replace(
    "> **L-0 merge authority:** NONE — PR #80 is open and must not be merged without a separate explicit product-owner decision.",
    "> **L-0 merge:** COMPLETE — PR #80 squash-merged by explicit product-owner authority.",
    1,
)
s = s.replace(
    "PR #80                               OPEN / MERGEABLE / RELEASE-DESIGN HOLD\nL-0 MERGED TO MAIN                   NO",
    "PR #80                               MERGED\nL-0 MERGED TO MAIN                   YES — " + MERGE_SHA + "\nRENDER AUTO-DEPLOY                   LIVE — " + DEPLOY_ID,
    1,
)
start = s.index("# 8. Exact next action / HOLD")
s = s[:start] + '''# 8. Exact next action / HOLD

```text
L-0 is complete and merged
→ HOLD for separate product-owner L-1 implementation authority
→ when authorized, create a fresh L-1 implementation slice/branch from current main
```

Current hold:

```text
NO L-1 runtime/database implementation yet
NO learning API runtime routes yet
NO Learning Hub production UI yet
NO external learning credential
NO Daily Case Review runtime
NO raw transcript persistence
NO patient-record mutation
NO Signal promotion
NO background cron
NO production config mutation
```
'''
p.write_text(s, encoding="utf-8")

# SLICE_PLAN_CURRENT.md
p = Path("SLICE_PLAN_CURRENT.md")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "> **STATUS:** CONTRACT FROZEN / COMPLETE / INDEPENDENT CLOSURE PASS / PR #80 OPEN — RELEASE-DESIGN HOLD",
    "> **STATUS:** CONTRACT FROZEN / COMPLETE / INDEPENDENT CLOSURE PASS / MERGED TO MAIN — SLICE CLOSED",
    1,
)
s = s.replace(
    "> **Fresh `main` / merge base:** `46bbb2fa00ad7c77482aab5fa84ef54e049154c8`.",
    f"> **Merged `main`:** `{MERGE_SHA}` — PR #80 squash merge.",
    1,
)
s = s.replace(
    "> **Design PR:** #80 OPEN; merge authority NONE.",
    "> **Design PR:** #80 MERGED.\n> **Render auto-deploy:** `" + DEPLOY_ID + "` LIVE; no learning runtime introduced.",
    1,
)
s = s.replace(
    "DESIGN PR #80                        OPEN / RELEASE-DESIGN HOLD\nL-0 MERGED TO MAIN                   NO",
    "DESIGN PR #80                        MERGED\nL-0 MERGED TO MAIN                   YES — " + MERGE_SHA + "\nRENDER AUTO-DEPLOY                   LIVE — " + DEPLOY_ID,
    1,
)
s = s.replace(
    "Exact next action:\n\n```text\nPR #80 OPEN\n→ RELEASE / DESIGN HOLD\n→ separate product-owner merge decision required\n→ after any later merge, separate product-owner L-1 runtime implementation decision required\n```",
    "Exact next action:\n\n```text\nL-0 CLOSED / MERGED\n→ HOLD for separate product-owner L-1 runtime implementation authority\n→ when granted, freeze a fresh bounded L-1 implementation slice before coding\n```",
    1,
)
p.write_text(s, encoding="utf-8")

# TODO.md
p = Path("TODO.md")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "## 1.10A Clinical Learning Hub — L-0 CONTRACT FROZEN / COMPLETE — PR #80 HOLD",
    "## 1.10A Clinical Learning Hub — L-0 CONTRACT FROZEN / COMPLETE / MERGED",
    1,
)
s = s.replace(
    "- [x] L-0: canonical closeout complete; clean freeze head `f947b12ce77db2ad1e5ff9117d7a4f794e224b60` passed run `34148333413`; bounded design PR #80 is OPEN in release-design HOLD.",
    "- [x] L-0: canonical closeout complete; clean freeze head `f947b12ce77db2ad1e5ff9117d7a4f794e224b60` passed run `34148333413`; bounded design PR #80 squash-merged to `main` as `" + MERGE_SHA + "`.",
    1,
)
p.write_text(s, encoding="utf-8")

# osteoporosis-change-log.md — append only
p = Path("osteoporosis-change-log.md")
s = p.read_text(encoding="utf-8")
marker = "## 2026-09-07 — Clinical Learning Hub L-0 contracts merged to main"
if marker not in s:
    s = s.rstrip() + f'''\n\n---\n\n{marker}\n\nAfter explicit product-owner merge authority, PR #80 (`Clinical Learning Hub: freeze L-0 contracts and L-1 boundaries`) was squash-merged to `main` as:\n\n```text\n{MERGE_SHA}\n```\n\nThe merge contains the independently reviewed/frozen L-0 learning contracts, machine-validation fixtures/tests, Foundation Map v1, L-1 ownership boundary and canonical closeout. It contains no Learning Hub runtime/database/API/UI implementation and no patient/transcript write path.\n\nRender auto-deploy triggered from the `main` commit and reached `LIVE` as:\n\n```text\n{DEPLOY_ID}\n```\n\nThis deploy is lifecycle evidence for the merged repository state; it does not convert L-0 design contracts into an implemented Learning Hub runtime.\n\nCurrent lifecycle:\n\n```text\nL-0 CONTRACT FROZEN / COMPLETE   YES\nINDEPENDENT REVIEW               CLOSURE PASS\nPR #80                           MERGED\nMAIN                              {MERGE_SHA}\nRENDER                            LIVE\nL-1 IMPLEMENTATION               NOT AUTHORIZED\n```\n\nThe next gate is a separate product-owner decision to authorize the bounded L-1 Challenge Import + History + Foundation Map MVP implementation slice.\n'''
p.write_text(s, encoding="utf-8")
