# Osteoporosis — Clinical Excellence Module 01

This repository is the canonical project home for **Module 01 — Osteoporosis** and the reusable **Personal Clinical Excellence System** being developed through it.

The existing Osteoporosis Cockpit is the point-of-care Clinical Practice layer. The broader system connects clinical standards, evidence, learning, testing, patient feedback, audits, safety, benchmarking and improvement into a continuous feedback loop.

## Start here

Read the canonical control files in this order before substantial work:

1. [`AGENTS.md`](./AGENTS.md) — permanent operating rules.
2. [`HANDOFF_CURRENT.md`](./HANDOFF_CURRENT.md) — exact current state and next action.
3. [`TODO.md`](./TODO.md) — long-range roadmap.
4. [`CLINICAL_EXCELLENCE_PLAN.md`](./CLINICAL_EXCELLENCE_PLAN.md) — active detailed blueprint/phase plan.
5. [`osteoporosis-change-log.md`](./osteoporosis-change-log.md) — append-only historical logbook.

These five files are the active canonical documentation set. This README is navigation only.

## Current application files

- `index.html` — current Osteoporosis Cockpit UI.
- `main.py` — current FastAPI backend / clinical logic.
- `osteoporosis-qa-handout.html` — patient-facing Q&A handout asset.
- `Dockerfile`, `requirements.txt` — runtime/dependency files.

## Current phase

**Clinical Excellence Blueprint + Baseline/Audit foundation.**

The immediate next design task is **Baseline Osteoporosis Audit v1 + KPI Dictionary v1**. No composite improvement score should be treated as real before that baseline is defined and measured.

## Privacy

This repository is public. **Do not commit identifiable patient data, GeSY/EMR identifiers, transcripts containing identifiers, clinical documents with personal data, secrets or credentials.** Use synthetic or fully anonymized examples only.
