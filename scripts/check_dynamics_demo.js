// Run the viewer logic against a minimal DOM/canvas harness; no browser UI.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const folder = process.argv[2];
const payload = JSON.parse(fs.readFileSync(folder + '/inline-data.json', 'utf8'));
let now = 1000, paintCalls = 0, persisted = null;
const callbacks = new Map();
let nextRaf = 1;
class Element {
  constructor() { this.style={}; this.events={}; this.children=[]; this.value='0'; this.textContent=''; this.clientWidth=320; }
  addEventListener(name,callback) { this.events[name]=callback; }
  appendChild(node) { this.children.push(node); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren() { this.children=[]; }
  remove() {}
}
const ids = {};
for(const id of ['ngff-demo-scene','ngff-demo-play','ngff-demo-time','ngff-demo-clock',
                 'ngff-demo-instant','ngff-demo-scores','ngff-demo-objects','ngff-demo-checkpoint'])ids[id]=new Element();
const ctx = new Proxy({}, {get(target,key){
  if(key in target)return target[key];
  return (...args)=>{ if(['arc','moveTo','lineTo','fillText'].includes(key))assert(args.slice(key==='fillText'?1:0).filter(x=>typeof x==='number').every(Number.isFinite));paintCalls++; };
}});
const canvases = Array.from({length:3},()=>Object.assign(new Element(),{
  width:0,height:0,parentElement:{clientWidth:320},getContext:()=>ctx,setPointerCapture:()=>{},
}));
const root = new Element();
root.querySelector = selector => {assert(ids[selector.slice(1)],selector);return ids[selector.slice(1)];};
root.querySelectorAll = selector => {assert.equal(selector,'canvas');return canvases;};
const style = {font:'14px Arial',color:'rgb(70,100,140)'};
const context = {
  document:{getElementById:id=>id==='ngff-actual-demo'?root:id==='ngff-demo-data'?{textContent:JSON.stringify(payload)}:null,
            createElement:()=>new Element(),createTextNode:value=>({textContent:value}),documentElement:{}},
  window:{devicePixelRatio:1,openai:{widgetState:null,setWidgetState:async x=>{persisted=x;}},addEventListener:()=>{}},
  getComputedStyle:()=>style,atob:global.atob,Uint8Array,Float32Array,DataView,Math,Number,Object,JSON,
  performance:{now:()=>now},
  requestAnimationFrame:fn=>{const id=nextRaf++;callbacks.set(id,fn);return id;},
  cancelAnimationFrame:id=>callbacks.delete(id),
  ResizeObserver:class{observe(){}},MutationObserver:class{observe(){}},
};
vm.runInNewContext(fs.readFileSync(folder+'/viewer-script.js','utf8'),context,{timeout:10000});
assert(paintCalls>1000);
assert.equal(ids['ngff-demo-scene'].children.length,2);
assert(ids['ngff-demo-scores'].textContent.includes(payload.scenes[0].metrics.ngff.rmse.toFixed(4)));
ids['ngff-demo-time'].value='40';ids['ngff-demo-time'].events.input();
assert(ids['ngff-demo-clock'].textContent.startsWith('1.58'));
ids['ngff-demo-time'].events.change();assert.equal(persisted.modelContent.frameIndex,79);
ids['ngff-demo-scene'].value='1';ids['ngff-demo-scene'].events.change();
assert(ids['ngff-demo-scores'].textContent.includes(payload.scenes[1].metrics.ngff.rmse.toFixed(4)));
assert.equal(persisted.modelContent.sceneSource,payload.scenes[1].source);
ids['ngff-demo-play'].events.click();assert.equal(ids['ngff-demo-play'].textContent,'暂停');
const tick=[...callbacks.values()].pop();callbacks.clear();now=1800;tick(now);
assert(!ids['ngff-demo-clock'].textContent.startsWith('0.00'));
ids['ngff-demo-play'].events.click();assert(ids['ngff-demo-play'].textContent.startsWith('播放'));
assert(persisted.modelContent.timeSeconds>0);
console.log('PASS: base64 trajectory decode, finite projection, scene picker, time slider, play/pause, state persistence');
