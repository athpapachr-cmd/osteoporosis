# CU-1 Physiotherapy Referral v2 — production browser smoke

> **Date:** 2026-08-28 Asia/Nicosia
> **Evidence source:** product-owner authenticated browser session
> **Runtime merge:** `c1da07f581cf8ccf1159d18bb63c23b674cbe9bd`
> **Render runtime deploy:** `dep-da8afeuk1f9s73f5sr6g`
> **Classification:** PRODUCTION-SMOKE-VERIFIED

The product owner executed the authenticated CU-1 browser smoke against production and reported all requested checks as passing.

Verified workflow:

```text
1. Clinical Excellence authentication/session succeeded.
2. Clinic Utilities → Physiotherapy Referral loaded successfully.
3. A representative Knee → Knee OA referral path was selectable and usable.
4. Minimum required fields could be completed.
5. Validate completed without unexpected blocking error.
6. Short referral generation completed successfully.
7. Detailed referral generation completed successfully.
8. Copy action worked.
9. Print action worked.
10. Refresh cleared the prior referral state as expected, confirming the intended ephemeral/no-browser-persistence behavior.
```

No defect was reported during this smoke.

Interpretation:

```text
IMPLEMENTED = PROVEN
TESTED = PROVEN
MERGED = PROVEN
DEPLOYED = PROVEN
PRODUCTION-SMOKE-VERIFIED = PROVEN by authenticated product-owner browser smoke
PILOT-VALIDATED = NOT CLAIMED
```

This smoke does not expand CU-1 scope, authorize persistence, reopen clinical taxonomy, authorize CU-2, or resume PR-1.
