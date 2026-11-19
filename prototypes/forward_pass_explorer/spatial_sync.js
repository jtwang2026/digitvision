/* A follower consumes snapshots; it never advances a second spatial clock. */
(function(root){
const session=crypto.randomUUID(),publisher=location.pathname.includes('forward_pass_explorer'),key='digitvision-spatial-owner';
let channel=null,latest=null,seq=0,owner=null,lastSeq=-1,received=0,listener=null,active=true;
try{channel=new BroadcastChannel('digitvision-spatial-v2');}catch{}
function lease(){try{const old=JSON.parse(localStorage.getItem(key)||'null');if(old&&old.session!==session&&Date.now()-old.time<2500)return false;localStorage.setItem(key,JSON.stringify({session,time:Date.now()}));return true;}catch{return true;}}
function send(){if(!publisher||!latest||!active||!lease())return;channel?.postMessage({version:2,session,seq:++seq,time:Date.now(),snapshot:latest});}
if(channel)channel.onmessage=e=>{const m=e.data;if(m?.type==='request'){send();return;}if(publisher||m?.version!==2||Date.now()-m.time>2500)return;let current;try{current=JSON.parse(localStorage.getItem(key)||'null');}catch{}if(current&&current.session!==m.session)return;if(owner!==m.session){owner=m.session;lastSeq=-1;}if(m.seq<=lastSeq)return;lastSeq=m.seq;received=Date.now();listener?.({...m.snapshot,session:m.session,sequence:m.seq,connected:true});};
const timer=setInterval(()=>{if(publisher)send();else if(received&&Date.now()-received>2500){received=0;listener?.({connected:false});}},500);
root.DigitVisionSpatialSync={publish(snapshot){latest=snapshot;send();},subscribe(fn){listener=fn;channel?.postMessage({type:'request'});},close(){active=false;clearInterval(timer);channel?.close();try{const old=JSON.parse(localStorage.getItem(key)||'null');if(old?.session===session)localStorage.removeItem(key);}catch{}}};
addEventListener('pagehide',()=>root.DigitVisionSpatialSync.close(),{once:true});
})(window);
