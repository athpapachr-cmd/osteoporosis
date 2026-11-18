# Compact Day Home — affected correction evidence

The two independent code-review findings are corrected: the protected surgery summary now exposes an accurate count/link for source rows beyond the seven-day named projection, including the later-only case; and panel redraws restore focus to the corresponding stable control or a descriptive heading without storing patient identifiers in the DOM. Expand controls expose `aria-expanded`. Invalid room time is suppressed rather than treated as trusted display text.

Synthetic Node tests cover the single later surgery, exact list navigation, keyboard-focus restoration on expand/privacy and a source update, plus existing source/auth/identity regressions. The actual Chromium desktop/half/mobile suite passes real Tab/Enter expansion focus and long-description date visibility; Python/Office Cases/browser and earlier focused tests remain passing. No production patient/provider access or write. One independent affected R2 closure is required.
