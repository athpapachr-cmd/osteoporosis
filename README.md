# Osteoporosis — Clinical Excellence Module 01

This repository is the canonical project home for **Module 01 — Osteoporosis** and the reusable **Personal Clinical Excellence System** being developed through it.

The system connects clinical standards, evidence, learning, real encounters, audit, Clinical Practice Review, patient feedback, safety and improvement into a continuous feedback loop.

## Start here — canonical bootstrap

Before substantial project work, follow `AGENTS.md` and read the active canonicals in this order:

1. [`AGENTS.md`](./AGENTS.md) — permanent operating rules.
2. [`TODO.md`](./TODO.md) — long-range roadmap/checklist.
3. [`CLINICAL_EXCELLENCE_PLAN.md`](./CLINICAL_EXCELLENCE_PLAN.md) — active detailed phase architecture.
4. [`SLICE_PLAN_CURRENT.md`](./SLICE_PLAN_CURRENT.md) — exact approved design of the one active slice.
5. [`CURRENT_OPERATIONAL.md`](./CURRENT_OPERATIONAL.md) — sole operational NOW / writer lock / exact next action.
6. [`osteoporosis-change-log.md`](./osteoporosis-change-log.md) — append-only durable history.

`HANDOFF_CURRENT.md` is now a compatibility redirect only and is **not** operational authority.

This README is navigation only.

## Current product direction

The immediate development program is:

```text
Baseline/pilot integrity
→ transcript-assisted structured capture
→ Clinical Practice Review
→ Signal / Learning / Improvement loop
→ adaptive osteoporosis consultation flow
→ Clinical Excellence Home
```

Calendar/Setmore/Digital Secretary integration is currently paused and does not block the above work.

## Current application areas

- `static/baseline-audit/` — prospective Baseline Audit / patient encounter workspace.
- `static/clinical-calendar/` — Clinical Calendar foundation; live Secretary feed deferred.
- `main.py` + clinical routers — FastAPI runtime / protected clinical data layer.
- `index.html` / legacy code — historical Osteoporosis Cockpit baseline, not the final Clinical Excellence Home.

## Privacy

This repository is public. **Do not commit identifiable patient data, GeSY/EMR identifiers, real transcripts, unredacted clinical documents, secrets or credentials.** Use synthetic or fully anonymized fixtures/examples only.
