'use strict';
// Production bridge for the tested Knee-OA product UI. The UI keeps its
// synthetic-prototype internal API vocabulary; this bridge maps only transport
// to the existing authenticated Cockpit physiotherapy router.
const PHYSIO_PRODUCT_API = '/clinical/clinic-utilities/physio-referral/api/product';
const productionFetch = window.fetch.bind(window);
window.fetch = function(input, init = {}) {
  const original = typeof input === 'string' ? input : input?.url;
  let mapped = input;
  if (original === '/api/bootstrap') mapped = PHYSIO_PRODUCT_API + '/bootstrap';
  else if (original === '/api/project') mapped = PHYSIO_PRODUCT_API + '/project';
  return productionFetch(mapped, {credentials:'same-origin', ...init});
};
