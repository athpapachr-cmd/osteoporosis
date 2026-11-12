'use strict';

// Exact start-page runtime in a deterministic browser-like context. The
// preference contains only an allowlisted id; no network or clinical data.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const source = fs.readFileSync('static/cockpit/start-page.js', 'utf8');

function run({path='/static/cockpit/', search='', hash='', stored=null, storageError=false,
              mainHidden=false, homeHidden=false} = {}) {
  const values = new Map(stored === null ? [] : [['clinical-excellence.start-page.v1', stored]]);
  const replacements = [];
  const events = [];
  const localStorage = {
    getItem(key) {if(storageError) throw Error('denied'); return values.get(key) ?? null;},
    setItem(key,value) {if(storageError) throw Error('denied'); values.set(key,value);},
  };
  const location = {pathname:path, search, hash, replace(value){replacements.push(value);}};
  const document = {getElementById(id) {
    if(id==='doctorMainView') return {hidden:mainHidden};
    if(id==='visitWorkspaceHome') return {hidden:homeHidden};
    return null;
  }};
  const window = {localStorage, location, dispatchEvent(event){events.push(event);}};
  class CustomEvent {constructor(type,options={}){this.type=type;this.detail=options.detail;}}
  vm.runInNewContext(source, {window,document,CustomEvent,URLSearchParams,Object}, {filename:'start-page.js'});
  return {api:window.CockpitStartPage,replacements,events,values,location};
}

{
  const result=run();
  assert.strictEqual(result.api.read(),'day');
  assert.strictEqual(result.api.current(),'day');
  assert.deepStrictEqual(result.replacements,[]);
  assert.strictEqual(result.api.href(),'/static/cockpit/?view=day');
}
{
  const result=run({stored:'inbox'});
  assert.deepStrictEqual(result.replacements,['/static/cockpit/#icloud-mail']);
  assert.strictEqual(result.api.href(),'/static/cockpit/#icloud-mail');
  assert.strictEqual(result.api.write('calendar'),true);
  assert.strictEqual(result.api.read(),'calendar');
  assert.strictEqual(result.events.at(-1).type,'cockpit:start-page-changed');
  assert.strictEqual(result.api.write('https://example.invalid/'),false);
  assert.strictEqual(result.api.read(),'calendar');
}
{
  const result=run({stored:'javascript:alert(1)'});
  assert.strictEqual(result.api.read(),'day');
  assert.deepStrictEqual(result.replacements,[]);
}
{
  const result=run({stored:'cases',storageError:true});
  assert.strictEqual(result.api.read(),'day');
  assert.strictEqual(result.api.write('cases'),false);
  assert.deepStrictEqual(result.replacements,[]);
}
{
  const result=run({stored:'calendar',search:'?chart=SYNTHETIC'});
  assert.deepStrictEqual(result.replacements,[],'deep link must not be redirected');
  assert.strictEqual(result.api.current(),null);
}
{
  const result=run({path:'/static/clinical-calendar/',stored:'day'});
  assert.strictEqual(result.api.current(),'calendar');
  assert.deepStrictEqual(result.replacements,[]);
}
{
  const result=run({hash:'#office-cases',stored:'day',mainHidden:true});
  assert.strictEqual(result.api.current(),'cases');
  assert.deepStrictEqual(result.replacements,[]);
}
{
  const result=run({stored:'day',homeHidden:true});
  assert.strictEqual(result.api.current(),null,'selected patient must not become Home');
}

console.log('Cockpit selectable start-page allowlist and neutral routing: PASS');
