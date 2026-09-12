'use strict';
// Production bridge for the reviewed Knee-OA product UI. The shared projection
// keeps a frozen synthetic compatibility envelope for the local test transport;
// Cockpit network requests are explicitly marked as clinical production usage
// and the protected server adapter converts them to that internal envelope only
// after validating the boundary.
const PHYSIO_PRODUCT_API = '/clinical/clinic-utilities/physio-referral/api/product';
const productionFetch = window.fetch.bind(window);
window.fetch = function(input, init = {}) {
  const original = typeof input === 'string' ? input : input?.url;
  let mapped = input;
  if (original === '/api/bootstrap') {
    mapped = PHYSIO_PRODUCT_API + '/bootstrap';
  } else if (original === '/api/project') {
    mapped = PHYSIO_PRODUCT_API + '/project';
    if (init?.body) {
      try {
        const body = JSON.parse(init.body);
        body.synthetic_only = false;
        init = {...init, body: JSON.stringify(body)};
      } catch (_error) {}
    }
  }
  return productionFetch(mapped, {credentials:'same-origin', ...init});
};
