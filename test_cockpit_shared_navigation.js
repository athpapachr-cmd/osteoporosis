'use strict';

// Real shared-nav script in a deterministic synthetic DOM; no network or PHI.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const entrypoints = [
  'static/cockpit/index.html',
  'static/cockpit/visit-capture/index.html',
  'static/clinical-calendar/index.html',
  'static/baseline-audit/index.html',
  'static/clinical-learning/index.html',
  'static/clinic-utilities/rf/index.html',
  'static/clinic-utilities/physio-referral/index.html',
  'static/clinic-utilities/sick-leave/index.html',
  'static/clinic-utilities/medical-report/index.html',
];
for (const path of entrypoints) {
  const content = fs.readFileSync(path, 'utf8');
  assert(content.includes('src="/static/cockpit/shared-nav.js"'), path + ': missing shared navigation');
  assert(content.includes('href="/static/cockpit/shared-nav.css"'), path + ': missing shared style');
}
const source = fs.readFileSync('static/cockpit/shared-nav.js','utf8');
const style = fs.readFileSync('static/cockpit/shared-nav.css','utf8');
assert(style.includes('position:sticky') && style.includes('z-index:3000'));
assert(!source.includes('localStorage') && !source.includes('sessionStorage'));
assert(!source.includes('document.cookie'));
assert(source.includes('/clinical/login') && source.includes('/clinical/status'));
const elements = [];
const events = [];
function node(tag = '') {
  const listeners = {};
  const el = {
    tag, className: '', children: [], hidden: false, textContent: '', value: '',
    dataset: {}, attributes: {},
    append(...children) { this.children.push(...children); },
    setAttribute(name,value) { this.attributes[name] = value; },
    getAttribute(name) { return this.attributes[name]; },
    addEventListener(name,cb) { listeners[name] = cb; },
    contains(target) {
      if (this === target) return true;
      return this.children.some(x => x === target || x.contains?.(target));
    },
    focus() {},
    async fire(name, e = {}) { return listeners[name]({preventDefault() {},target:this,...e}); },
  };
  elements.push(el);
  return el;
}
const body = node('body');
body.firstChild = null;
body.insertBefore = function(child) {this.children.unshift(child);};
const document = {
  readyState: 'complete', body,
  createElement: node,
  createTextNode: text => ({textContent:text}),
  getElementById: id => elements.find(el=>el.id===id)||null,
  addEventListener(name,cb) {events.push({name,cb});},
};
let mode = 'unauthenticated';
const pendingStaff = [];
const requests = [];
const winEvents = [];
const window = {
  dispatchEvent(e) {winEvents.push(e.type);},
  addEventListener(name,cb) {events.push({name,cb});}
};
async function fetch(url,options = {}) {
  requests.push({url, method:options.method||'GET', options});
  if (url==='/clinical/status') return {ok:mode==='authenticated',status:mode==='authenticated'?200:401};
  if (url==='/clinical/staff-session')
    return new Promise(resolve=>pendingStaff.push(resolve));
  if(url==='/clinical/login') {
    assert.strictEqual(options.credentials,'same-origin');
    assert.strictEqual(options.method,'POST');
    assert.strictEqual(options.cache,'no-store');
    assert.strictEqual(options.headers['Content-Type'],'application/json');
    assert.strictEqual(JSON.parse(options.body).key,'synthetic-key');
    if(mode==='invalid')return {ok:false,status:401};
    mode='authenticated';return {ok:true,status:200,json:async()=>({authenticated:true,session:"browser",staff_role:"secretary"})};
  }
  throw Error('Unexpected shared-nav network request: '+url);
}
const context={document,window,fetch,Event:class {constructor(type){this.type=type;}}};
vm.runInNewContext(source,context,{filename:'shared-nav.js'});
const get=id=>document.getElementById(id);
assert(get('ceGlobalNav') && get('ceGlobalShortcuts') && get('ceGlobalKey'));
assert.strictEqual(body.children[0],get('ceGlobalNav'));
const nav = get('ceGlobalNav');
const panel = get('ceGlobalShortcutsPanel'), access = get('ceGlobalAccessPanel');
const home=nav.children[0].children[0];
assert.strictEqual(home.href,'/static/cockpit/');
(async()=>{
  await get('ceGlobalShortcuts').fire('click');
  assert.strictEqual(panel.hidden,false);
  const anchors = panel.children.filter(el=>el.tag==='a');
  assert.strictEqual(anchors.length,8);
  assert(anchors.every(x=>x.href?.startsWith('/')));
  assert(anchors.some(a=>a.href==='/clinical/clinic-utilities/rf'));
  assert(anchors.some(a=>a.href==='/static/clinical-calendar/'));
  await get('ceGlobalKey').fire('click');
  assert.strictEqual(panel.hidden,true);
  assert.strictEqual(access.hidden,false);
  await Promise.resolve();await Promise.resolve();
  assert(get('ceGlobalAuthStatus').textContent.includes('Απαιτείται σύνδεση'));
  const input=get('ceGlobalKeyInput'),form=get('ceGlobalLoginForm');
  assert.strictEqual(input.type,'password');
  input.value='synthetic-key';
  mode='invalid';
  await form.fire('submit');
  assert.strictEqual(input.value,'','key must not remain in input');
  assert(get('ceGlobalAuthStatus').textContent.includes('Λανθασμένο'));
  assert(!winEvents.includes('clinical:authenticated'));
  mode='unauthenticated';
  input.value='synthetic-key';
  await form.fire('submit');
  assert(get('ceGlobalAuthStatus').textContent.includes('Συνδέθηκες'));
  assert.strictEqual(window.CockpitStaffRole,'secretary');
  // A delayed clinician bootstrap/probe must not override the new secretary login.
  assert(pendingStaff.length>=1,'expected unresolved staff probe');
  for(const resolve of pendingStaff)
    resolve({ok:true,status:200,json:async()=>({staff_role:'clinician',principal_id:'clinic-clinician'})});
  await new Promise(resolve=>setImmediate(resolve));
  assert.strictEqual(window.CockpitStaffRole,'secretary','old staff probe replaced current role');
  assert.deepStrictEqual(winEvents,['clinical:authenticated']);
  assert(!JSON.stringify(elements.map(x=>x.value)).includes('synthetic-key'));
  assert.strictEqual(requests.filter(r=>r.url==='/clinical/login').length,2);
  const escape = events.find(e=>e.name==='keydown');
  escape.cb({key:'Escape'});
  assert.strictEqual(access.hidden,true);
  assert.strictEqual(get('ceGlobalShortcuts').getAttribute('aria-expanded'),'false');
  assert.strictEqual(get('ceGlobalKey').getAttribute('aria-expanded'),'false');
  // Idempotence: additional boot/load should not create another navigation.
  assert.strictEqual(elements.filter(x=>x.id==='ceGlobalNav').length,1);
  console.log('Cockpit shared sticky shortcuts / secure login UI: PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});

// The Cockpit daily source and weekly Calendar use the same private provider
// and never quietly substitute osteoporosis-only entries for full-day coverage.
const clinical=fs.readFileSync('clinical_calendar.py','utf8');
assert(clinical.includes('params={"target_date": today_local.isoformat()}'));
assert(clinical.includes('source.coverage_start > today_local'));
assert(clinical.includes('source.coverage_end < today_local'));
const homeJS=fs.readFileSync('static/cockpit/app.js','utf8');
assert(homeJS.includes('error.status === 401 || error.status === 403'));
assert(homeJS.includes('Το πρόγραμμα της Γραμματείας'));
assert(homeJS.includes('cockpitCalendarRetry'));
assert(homeJS.includes('clinical:authenticated'));
console.log('Cockpit local-day coverage, source-auth distinction and retry wiring: PASS');
