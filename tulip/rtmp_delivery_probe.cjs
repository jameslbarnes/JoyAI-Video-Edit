// Read-only remux probe: timestamp compressed video packet delivery after MediaMTX.
const {spawn}=require('node:child_process'),fs=require('node:fs');
const [file,seconds='180']=process.argv.slice(2);
if(!file||fs.existsSync(file))throw Error('Expected NEW JSONL path');
const log=value=>fs.appendFileSync(file,JSON.stringify({at:Date.now(),...value})+'\n');
const child=spawn('C:/Users/Admin/src/easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe',
 ['-hide_banner','-loglevel','error','-nostdin','-probesize','32768','-analyzeduration','0','-i','rtmp://127.0.0.1:1935/joyai-gpu','-map','0:v:0','-c:v','copy','-f','framecrc','-flush_packets','1','pipe:1'],{windowsHide:true,stdio:['ignore','pipe','pipe']});
let buffer='',last=null,count=0,maxGap=0;
child.stdout.on('data',data=>{buffer+=data;let i;while((i=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,i);buffer=buffer.slice(i+1);if(line.startsWith('#'))continue;const now=performance.now();const gap=last===null?0:now-last;last=now;count++;maxGap=Math.max(maxGap,gap);if(gap>62.5)log({event:'gap',gapMs:gap,line});}});
child.stderr.on('data',data=>log({event:'stderr',text:String(data)}));
const timer=setTimeout(()=>child.kill(),Number(seconds)*1000);
const health=setInterval(async()=>{try{const h=await(await fetch('http://127.0.0.1:7873/health')).json();const g=h.upscale.gpu_video;log({event:'status',count,maxGap,muxPid:g.mux_pid,underruns:g.underruns,gaps:g.delivery_gaps});}catch(error){log({event:'error',message:String(error)});}},1000);
child.on('close',code=>{clearTimeout(timer);clearInterval(health);log({event:'summary',count,maxGap,code});console.log(JSON.stringify({count,maxGap,code}));});
