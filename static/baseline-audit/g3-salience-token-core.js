(() => {
  "use strict";

  const HIGH_VALUE_REASONS = new Set(["NEW_EVENT", "UNRESOLVED_PRIOR", "EXPLICIT_DUE_STATE", "TREATMENT_CONTEXT"]);
  const asArray = value => Array.isArray(value) ? value : [];
  const clean = value => String(value || "").trim();

  function itemTokens(item) {
    const domain = clean(item?.card_id);
    if (!domain) return [];
    const tokens = [];
    asArray(item?.evidence_rules).forEach(rule => {
      const ruleId = clean(rule?.rule_id);
      if (ruleId) tokens.push(`E|${domain}|${ruleId}`);
    });
    asArray(item?.reason_codes).forEach(reason => {
      if (HIGH_VALUE_REASONS.has(reason)) tokens.push(`R|${domain}|${reason}`);
    });
    return tokens;
  }

  function tokenDomain(token) {
    return clean(token).split("|")[1] || "";
  }

  function advance({ previousTokens = null, retainedNewTokens = [], items = [], initialize = false } = {}) {
    const normalizedItems = asArray(items).filter(item => clean(item?.card_id));
    const currentTokens = new Set(normalizedItems.flatMap(itemTokens));
    const currentDomains = new Set(normalizedItems.map(item => clean(item.card_id)));
    const retained = new Set(asArray(retainedNewTokens).filter(Boolean));

    if (!initialize && previousTokens !== null) {
      const previous = new Set(asArray(previousTokens).filter(Boolean));
      currentTokens.forEach(token => {
        if (!previous.has(token)) retained.add(token);
      });
    } else {
      retained.clear();
    }

    Array.from(retained).forEach(token => {
      const domain = tokenDomain(token);
      if (!currentTokens.has(token) || !currentDomains.has(domain)) retained.delete(token);
    });

    return {
      current_tokens: Array.from(currentTokens),
      retained_new_tokens: Array.from(retained),
      newly_surfaced_domains: Array.from(new Set(Array.from(retained).map(tokenDomain).filter(Boolean)))
    };
  }

  window.G3SalienceTokenCore = Object.freeze({ itemTokens, tokenDomain, advance });
})();
