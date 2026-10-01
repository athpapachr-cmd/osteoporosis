# Prototype 1 implementation evidence — synthetic only

## Five journeys

| Context | Deterministic result |
|---|---|
| Known denosumab / due today | Last recorded actual 2026-01-01; prior explicit continue; G2 R12 derived due 2026-07-01; no administration created. |
| Delayed | 2026-08-15 focus is delay; G2 R24 >7-month escalation retains reviewed source references; no rescue choice created. |
| Prior transition | Explicit denosumab-exit switch changes focus to transition; planned zoledronate remains a plan milestone and no zoledronate actual is inferred. |
| Conflicting actual date | Same administration ID with two dates withholds reliable last actual and derived due; both source events remain in trajectory. |
| Ambiguous plan/task | Planned row and later actual are proposed as a possible link; two same-type task rows with changed dates remain distinct and open; no link or closure is persisted. |

All five cases run against the existing G1/G2/G3 functions before the Prototype 1 projection. An upstream actual-date correction from 2026-01-01 to 2026-02-01 withdraws the displayed due rule for the July visit. Source rows remain unchanged by projection. Unknown/unavailable history does not yield a timing conclusion. The `previsit` function is a view of the same projection and is available only inside the protected module; aggregate Cockpit Home was not changed.

## Synthetic clinician walkthrough trace

This is a deterministic source/interaction trace, not an observed browser session or UX score.

1. Last actual dose is exposed by the G1 event projection and labelled as recorded actual, with source encounter route.
2. The current focus is selected from existing G2 due/delay outputs and prior explicit decision; the clinician may override the suggestion.
3. Conflict/missing history produces an uncertainty cue and withholds a confident due date.
4. Source action routes directly to the Step-4 administration editor; the existing protected encounter remains the owner.
5. An edited upstream date causes the workspace to replace old content with a recalculation message, then G2 and the visit projection rebuild. The synthetic correction test proves the due output disappears.
6. Raw Step-4 task rows are visible as open obligations with original source routes; tuple equality is not used as identity.
7. A prior milestone carries date precision, type, source and editor route. No continuous line is drawn across unsupported gaps.
8. Return action restores the current encounter pointer and visit workspace, or returns to the same active case when no protected encounter ID exists.

**Navigation burden:** one source action and one return action are designed for a correction; rendered click count was not observed. **Potential confusion:** the secondary source editor still exposes legacy numbered tabs and old labels; its visibility is confined to the editor view. **Missing context:** no durable continuity confirmation path and no authorized patient-specific Cockpit pre-visit card. **Repeated clicks:** not measured in a browser. **Six-step leakage:** the secondary editor retains Steps 1–6; the primary adaptive workspace has no numbered progression or forced Next.

## Limits and disposition

- Clinical rule changes: none.
- Authoritative store/schema changes: none.
- Historical link confirmation: held on existing owner path; suggestions remain provisional.
- Pre-visit projection: interface only; patient-specific Cockpit card held on privacy/authorization gate.
- Browser rendering and clinician usability: unverified in this execution environment. Prototype 1 usability evaluation ready: **NO** until a protected synthetic browser walkthrough verifies the actual UI and direct-return behavior.
