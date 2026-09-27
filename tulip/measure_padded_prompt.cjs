// Paired short/long-prompt timing windows, preserving competing creative edits.
const fs=require('node:fs'),path=require('node:path');
const W=require('node:module').createRequire(path.resolve(__dirname,'../../etherea-ai/package.json'))('ws');
const [file,seedFile]=process.argv.slice(2);
if(!file||fs.existsSync(file)||!seedFile)throw Error('NEW.json PROMPT_SEED.json');
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const get=async url=>{const r=await fetch(url,{signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error(`${url}: ${r.status}`);return r.json();};
(async()=>{
 const targets=await get('http://127.0.0.1:9334/json/list');
 const target=targets.find(t=>t.type==='page'&&t.url.includes(':7860/'));
 const ws=new W(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.once('open',r);ws.once('error',j);});
 const pending=new Map();let seq=0;
 ws.on('message',b=>{const r=JSON.parse(b),p=pending.get(r.id);if(p){pending.delete(r.id);r.error?p.reject(Error(JSON.stringify(r.error))):p.resolve(r.result);}});
 const evaluate=async(fn,arg)=>{
  const id=++seq;const v=await new Promise((resolve,reject)=>{
   const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout'));},15000);
   pending.set(id,{resolve:v=>{clearTimeout(timer);resolve(v);},reject:e=>{clearTimeout(timer);reject(e);}});
   ws.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression:`(${fn.toString()})(${JSON.stringify(arg)??''})`,awaitPromise:true,returnByValue:true}}));
  });if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value;
 };
 const read=()=>evaluate(()=>{const c=window.ethereaMultiStreamManager.streams.find(s=>s.type==='joyai')?.runtime;return{prompt:c?.prompt,running:c?.running,busy:!!(c?.control||c?.controlTask)};});
 const update=(expected,next)=>evaluate(({expected,next})=>{
  const c=window.ethereaMultiStreamManager.streams.find(s=>s.type==='joyai')?.runtime;
  if(!c?.running||c.prompt!==expected||c.control||c.controlTask)return false;
  void c.queuePrompt(next,false);return true;
 },{expected,next});
 let seed,owned;const samples=[];let complete=false,restore=null,error=null;
 const save=()=>fs.writeFileSync(file,JSON.stringify({complete,error,restore,seedFile,samples},null,2));
 try{
  const c=await read();if(!c.running||c.busy)throw Error('Client unavailable/busy; preserved');
  if(fs.existsSync(seedFile))seed=JSON.parse(fs.readFileSync(seedFile,'utf8'));
  else{
   const sentences=c.prompt.split(/(?<=[.!?])\s+/);const extra=sentences.at(-1);
   seed={original:c.prompt,variant:c.prompt+'\n'+extra};
   if(extra.length<30)throw Error('Need a substantive final sentence for the padding probe');
   fs.writeFileSync(seedFile,JSON.stringify(seed,null,2),{flag:'wx'});
  }
  if(c.prompt!==seed.original)throw Error('Scene changed since baseline; preserve the new prompt');
  owned=seed.original;
  for(const phase of ['original','variant']){
   if(phase==='variant'){
    if(!await update(owned,seed.variant))throw Error('Creative update superseded probe');
    owned=seed.variant;
   }
   const start=Date.now();
   while(Date.now()-start<65000){
    const c=await read();if(c.prompt!==owned)throw Error('Creative update superseded probe');
    const debug=await get('http://127.0.0.1:7873/debug');
    const health=await get('http://127.0.0.1:7873/health');
    const listing=await get('http://127.0.0.1:8793/control/multistream/list');
    const tile=listing.streams.find(s=>s.type==='joyai');
    const row={at:Date.now()/1000,phase,phaseSeconds:(Date.now()-start)/1000,
      stage:debug.stage_timings,profile:debug.session?.last_chunk_profile,memory:debug.memory,
      gpu:health.upscale?.gpu_video,stats:tile?.runtime?.stats,status:tile?.status};
    samples.push(row);save();
    if(samples.length%15===0)console.log(JSON.stringify({phase,seconds:row.phaseSeconds,tokens:row.profile?.graph_prompt_tokens,
      capacity:row.profile?.graph_prompt_capacity,active:row.profile?.graph_active_text_rows,underruns:row.gpu?.underruns}));
    await pause(1000);
   }
  }
  complete=true;
 }catch(e){error=String(e);throw e;}
 finally{
  if(seed&&owned===seed.variant)restore=await update(seed.variant,seed.original).catch(e=>({error:String(e)}));
  save();ws.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
