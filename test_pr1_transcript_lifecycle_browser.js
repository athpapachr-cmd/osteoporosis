const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Node {
  constructor() {
    this.listeners = new Map();
    this.children = [];
    this.dataset = {};
    this.value = '';
    this.textContent = '';
    this.open = false;
    this.disabled = false;
    this.named = new Map();
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  dispatch(name, event = {}) { this.listeners.get(name)?.(event); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; this.dispatch('close'); }
  querySelector(selector) { return this.named.get(selector) || null; }
  set innerHTML(value) {
    this.html = value;
    for (const selector of [
      '[data-transcript-input]', '[data-transcript-status]', '[data-transcript-results]',
      '[data-transcript-submit]', '[data-transcript-close]', '[data-transcript-discard]',
      '[data-transcript-logout]',
    ]) this.named.set(selector, new Node());
  }
}

const documentListeners = new Map();
const windowListeners = new Map();
const document = {
  head: new Node(), body: new Node(),
  createElement: () => new Node(),
  querySelector: () => null,
  addEventListener: (name, callback) => documentListeners.set(name, callback),
};
const window = {
  addEventListener: (name, callback) => windowListeners.set(name, callback),
  alert: () => { throw new Error('unexpected logout failure'); },
};
const pending = [];
let logoutCalls = 0;
const fetch = (url, options) => {
  if (url === '/clinical/logout') {
    logoutCalls += 1;
    assert.equal(options.credentials, 'same-origin');
    return Promise.resolve({ ok: true });
  }
  assert.equal(url, '/clinical/transcript/extract');
  return new Promise(resolve => pending.push({ resolve, signal: options.signal }));
};
vm.runInNewContext(
  fs.readFileSync('static/baseline-audit/transcript-capture.js', 'utf8'),
  { document, window, fetch, AbortController, JSON, String },
);
const dialog = document.body.children[0];
const input = dialog.querySelector('[data-transcript-input]');
const results = dialog.querySelector('[data-transcript-results]');
const submit = dialog.querySelector('[data-transcript-submit]');

function open() {
  documentListeners.get('click')({
    target: { closest: selector => selector === '[data-nav-action="heidi"]' ? {} : null },
    preventDefault() {},
  });
}
function start() {
  open();
  input.value = 'SYNTHETIC TRANSCRIPT';
  submit.dispatch('click');
  assert.equal(pending.length > 0, true);
}
async function settleLate(index) {
  pending[index].resolve({
    ok: true,
    json: async () => ({ candidates: [{ semantic_type: 'late-result' }] }),
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(results.children.length, 0);
  assert.equal(input.value, '');
}

(async () => {
  start();
  windowListeners.get('pagehide')();
  assert.equal(dialog.open, false);
  assert.equal(pending[0].signal.aborted, true);
  await settleLate(0);
  windowListeners.get('pageshow')();
  assert.equal(results.children.length, 0);

  start();
  dialog.querySelector('[data-transcript-logout]').dispatch('click');
  assert.equal(dialog.open, false);
  assert.equal(logoutCalls, 1);
  assert.equal(pending[1].signal.aborted, true);
  await settleLate(1);

  start();
  documentListeners.get('click')({ target: { closest: selector => selector === '.side-item:not([data-nav-action="heidi"])' ? {} : null } });
  assert.equal(pending[2].signal.aborted, true);
  await settleLate(2);
  process.stdout.write('PR-1 transient lifecycle: pagehide/pageshow, logout, navigation, late response PASS\n');
})().catch(error => { process.stderr.write(String(error.stack || error)); process.exitCode = 1; });
