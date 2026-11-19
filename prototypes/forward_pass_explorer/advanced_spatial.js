/* Conv2, ReLU2 and Pool2 share the existing Forward Explorer RAF clock. */
(function(root){
"use strict";
const SPECS=Object.freeze({conv2:{input:"pool1",output:"conv2_pre",side:11,sourceSide:13,kernel:3,stride:1,channels:32},relu2:{input:"conv2_pre",output:"conv2_relu",side:11,sourceSide:11,kernel:1,stride:1,channels:64},pool2:{input:"conv2_relu",output:"pool2",side:5,sourceSide:11,kernel:2,stride:2,channels:64}});
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function point(spec,position){
 const count=spec.side*spec.side,p=clamp(position,0,count-1),n=Math.floor(p),f=p-n;
 const at=i=>({row:Math.floor(i/spec.side),col:i%spec.side});
 const a=at(n),b=at(Math.min(count-1,n+1));
 return {index:a.row*spec.side+a.col,step:n,row:a.row,col:a.col,visualRow:a.row+(b.row-a.row)*f,visualCol:a.col+(b.col-a.col)*f,phase:f,inputRow:a.row*spec.stride,inputCol:a.col*spec.stride,outputRow:a.row,outputCol:a.col,kernel:spec.kernel,stride:spec.stride};
}
function create(host){
 const {$,S,DATA,fmt,near,convAt,poolAt,mapFor,drawMap,drawBank,drawContributions,color,advanceClock,metrics,queueFrame,complete,pause,updateControls,onRender}=host;
 const handles=key=>Object.hasOwn(SPECS,key),state={key:"",mode:"fast",position:0,fraction:0,last:null,manualIndex:null,manualPhase:0,metrics:null,direction:1};
 const modes={conv2:"fast",relu2:"fast",pool2:"fast"};let cache=null,dom=null,detailKey="",controlKey="";
 function spec(){return SPECS[S.current.key];}
 function durationFor(mode,speed){return S.current.key==='pool2'&&mode==='detailed'?75000/Math.max(.5,speed):host.duration(mode,speed);}
 function reset(key=S.current.key){state.direction=1;state.fastClock=null;state.key=key;state.mode=modes[key]||"fast";state.position=0;state.fraction=0;state.last=null;state.manualIndex=null;state.manualPhase=0;state.metrics=null;cache=null;detailKey="";controlKey="";}
 function text(id,value){const el=typeof id==="string"?$(id):id;if(el.textContent!==value)el.textContent=value;}
 function ensureCache(){
  const s=spec(),key=S.sample.index+"/"+S.current.key+"/"+S.selection.out+"/"+S.selection.input;
  if(cache?.key===key)return;
  const source=mapFor(s.input,S.current.key==="conv2"?S.selection.input:S.selection.out),target=mapFor(s.output,S.selection.out);
  const range=Math.max(Math.abs(source.min),Math.abs(source.max),Math.abs(target.min),Math.abs(target.max),1e-9);
  let best=0;for(let i=1;i<target.values.length;i++)if(target.values[i]>target.values[best])best=i;
  const atlas=document.createElement("canvas");atlas.width=480;atlas.height=240;
  if(S.current.key==="conv2")drawBank(atlas,"pool1",S.selection.input);
  cache={key,source,target,range,atlas,best,painted:0};detailKey="";
 }
 function ensureDOM(){
  if(dom&&$("operation-content").contains(dom.root))return;
  const panel=$("operation-content");panel.innerHTML='<div class="advanced-spatial"><div class="advanced-title"></div><div class="advanced-grid"></div><div class="advanced-summary"></div><div class="advanced-phase"></div></div>';
  dom={root:panel.firstElementChild,title:panel.querySelector(".advanced-title"),grid:panel.querySelector(".advanced-grid"),summary:panel.querySelector(".advanced-summary"),phase:panel.querySelector(".advanced-phase")};
  $("term-grid").replaceChildren(...Array.from({length:4},()=>{const el=document.createElement("div");el.className="term revealed";return el;}));
  if(state.metrics)state.metrics.domBuilds++;
  detailKey="";
 }
 function drawOutput(p,selected){
  const s=spec(),canvas=$("target-canvas"),ctx=canvas.getContext("2d"),target=cache.target,source=cache.source,cell=canvas.width/s.side;
  ctx.fillStyle="#080f1c";ctx.fillRect(0,0,canvas.width,canvas.height);
  for(let row=0;row<s.side;row++)for(let col=0;col<s.side;col++){
   const sequence=row*s.side+col,index=row*s.side+col,done=sequence<S.local||S.local===S.limit;
   if(!done&&s!==SPECS.relu2)continue;
   const value=done?target.values[index]:source.values[index];
   ctx.fillStyle=color(value,cache.range,"feature");ctx.fillRect(col*cell+.4,row*cell+.4,cell-.8,cell-.8);
   if(!done&&s===SPECS.relu2&&value<0){ctx.fillStyle="#fb7185";ctx.globalAlpha=.12;ctx.fillRect(col*cell+.4,row*cell+.4,cell-.8,cell-.8);ctx.globalAlpha=1;}
  }
  const visual=state.manualIndex===null&&S.local<S.limit?p:{visualRow:selected.row,visualCol:selected.col};
  if(S.current.key==="relu2"&&S.local<S.limit){
   const i=p.row*s.side+p.col,raw=source.values[i];
   if(raw<0){ctx.fillStyle="#080f1c";ctx.fillRect(p.col*cell,p.row*cell,cell,cell);ctx.globalAlpha=1-p.phase;ctx.fillStyle=color(raw,cache.range,"feature");ctx.fillRect(p.col*cell+.4,p.row*cell+.4,cell-.8,cell-.8);ctx.globalAlpha=1;}
  }
  ctx.strokeStyle="#f6c76b";ctx.lineWidth=2.5;ctx.strokeRect(visual.visualCol*cell+1,visual.visualRow*cell+1,cell-2,cell-2);
  canvas.dataset.written=String(S.local===S.limit);canvas.dataset.progress=String(state.position/S.limit);canvas.dataset.cursor=JSON.stringify({row:selected.row,col:selected.col,visualRow:visual.visualRow,visualCol:visual.visualCol});
 }
 function drawAtlas(p){
  if(S.current.key!=="conv2")return;
  const canvas=$("input-bank"),ctx=canvas.getContext("2d");canvas.hidden=false;ctx.drawImage(cache.atlas,0,0);
  const side=13,w=canvas.width/8,h=canvas.height/4,cw=(w-8)/side,ch=(h-17)/side;
  ctx.strokeStyle="#5ee1e8";ctx.lineWidth=1.25;
  for(let i=0;i<32;i++){
   const x=i%8*w+4+p.visualCol*cw,y=Math.floor(i/8)*h+3+p.visualRow*ch;
   ctx.strokeRect(x+.5,y+.5,3*cw-1,3*ch-1);
  }
  canvas.dataset.cursor=JSON.stringify({row:p.row,col:p.col,visualRow:p.visualRow,visualCol:p.visualCol,windows:32});
 }
 function render(){
  if(!handles(S.current.key))return;
  if(state.key!==S.current.key)reset();
  let reverse=$("spatial-reverse");if(!reverse){reverse=document.createElement("button");reverse.id="spatial-reverse";reverse.className="button";reverse.textContent="方向：正向";$("layer-animation-controls").append(reverse);reverse.onclick=()=>{state.direction*=-1;state.last=null;reverse.textContent=state.direction===1?"方向：正向":"方向：反向";updateControls();};}
  ensureCache();ensureDOM();const s=spec(),playing=S.globalStatus==="PLAYING"||S.localStatus==="PLAYING",complete=S.local===S.limit;
  const scan=point(s,state.position),selected=state.manualIndex===null?complete?{row:Math.floor(cache.best/s.side),col:cache.best%s.side}:scan:{row:Math.floor(state.manualIndex/s.side),col:state.manualIndex%s.side};
  S.selection.row=selected.row;S.selection.col=selected.col;
  $("layer-animation-controls").hidden=false;$("layer-animation-mode").value=state.mode;
  text("layer-animation-state",playing?state.mode==="fast"?"Fast Overview · 连续空间扫描":"Detailed · 连续空间扫描":"Detailed Inspection · 真实数值");
  const duration=durationFor(state.mode,Number($(S.globalStatus==="PLAYING"?"global-speed":"local-speed").value));
  text("layer-animation-duration",duration===null?"手动单步":(duration/1000).toFixed(1)+" s · 教学播放时长");
  $("input-bank").hidden=S.current.key!=="conv2";$("kernel-bank").hidden=S.current.key!=="conv2";$("contributions").hidden=S.current.key!=="conv2";
  const src=$("source-canvas"),window=$("receptive-window");drawMap(src,cache.source);
  const vr=state.manualIndex===null&&!complete?scan.visualRow:selected.row,vc=state.manualIndex===null&&!complete?scan.visualCol:selected.col;
  window.hidden=false;window.style.left="0";window.style.top="0";window.style.width=s.kernel/s.sourceSide*100+"%";window.style.height=s.kernel/s.sourceSide*100+"%";
  window.style.transition="none";window.style.transform=`translate3d(${vc*s.stride/s.kernel*100}%,${vr*s.stride/s.kernel*100}%,0)`;
  src.dataset.cursor=JSON.stringify({row:selected.row*s.stride,col:selected.col*s.stride,visualRow:vr*s.stride,visualCol:vc*s.stride,size:s.kernel});
  drawOutput(scan,selected);drawAtlas(scan);
  const key=cache.key+"/"+selected.row+"/"+selected.col;
  if(detailKey!==key){detailKey=key;if(state.metrics)state.metrics.detailUpdates++;}
  const ch=S.selection.out,r=selected.row,c=selected.col;
  if(S.current.key==="conv2"){
   const op=convAt(DATA,S.sample,"conv2",ch,r,c,S.selection.input);
   text(dom.title,"32 个同步 3×3 窗口 → Filter "+ch);text(dom.summary,"Σ32 = "+fmt(op.contributions.reduce((a,b)=>a+b,0),8)+" · Bias = "+fmt(op.bias,8)+" · 输出 = "+fmt(op.expected,9));
   text(dom.phase,"输入窗口 ["+r+","+c+"] → 输出 ["+ch+","+r+","+c+"] · 64 个 Filter 独立响应");
   dom.grid.textContent="当前输入通道 C"+S.selection.input+" 的贡献 "+fmt(op.contributions[S.selection.input],8);
   drawContributions(op.contributions,S.selection.input,32);
   text("calculation-formula","Σ32 局部贡献 + 一次 Bias = "+fmt(op.raw,9)+"；真实导出 "+fmt(op.expected,9));
   text("result-value",fmt(op.expected,9));$("consistency").dataset.passed=String(near(op.raw,op.expected));
   text("operation-note","青色窗口在 32 张输入图同步移动。32 路贡献的顺序只是教学展示，64 个 Filter 可并行计算。");
   text("local-heading","Conv2 · 32 路真实贡献汇聚");
  }else if(S.current.key==="relu2"){
   const raw=cache.source.values[r*11+c],value=cache.target.values[r*11+c];
   text(dom.title,"逐元素 ReLU · 64 通道同位置");text(dom.summary,fmt(raw,8)+" → max(0,x) → "+fmt(value,8));
   text(dom.phase,raw<0?"负响应逐渐归零":raw===0?"零响应保持为零":"正响应保持原值");
   dom.grid.textContent="通道 "+ch+" · ["+r+","+c+"]";
   text("calculation-formula","max(0,"+fmt(raw,9)+") = "+fmt(value,9));text("result-value",fmt(value,9));$("consistency").dataset.passed=String(near(Math.max(0,raw),value));
   text("operation-note","每个通道独立逐元素激活；右侧按扫描位置形成真实 ReLU2 输出。");text("local-heading","ReLU2 · 负值归零，正值保留");
  }else{
   const op=poolAt(S.sample,"pool2",ch,r,c),v=op.values,a=Math.max(v[0],v[1]),b=Math.max(v[2],v[3]),phase=state.mode==="step"?state.manualPhase:Math.min(8,Math.floor((state.dwellPhase??scan.phase)*9));
   text(dom.title,"2×2 · A/B、C/D、两组胜出者");
   dom.grid.innerHTML=v.map((value,i)=>`<span class="advanced-candidate ${value===op.raw&&phase>=7?"winner":""}">${"ABCD"[i]} ${fmt(value,6)}</span>`).join("");
   const message=phase<2?"四个候选值":phase<4?`A ${v[0]<v[1]?"<":v[0]>v[1]?">":"="} B → ${fmt(a,7)}`:phase<6?`C ${v[2]<v[3]?"<":v[2]>v[3]?">":"="} D → ${fmt(b,7)}`:phase<8?`Winner 1 ${a<b?"<":a>b?">":"="} Winner 2 → ${fmt(op.raw,7)}`:"FINAL MAXIMUM → 输出 ["+ch+","+r+","+c+"]";
   text(dom.phase,"比较 "+phase+" / 8 · "+message);
   text(dom.summary,phase>=7?"FINAL MAXIMUM "+fmt(op.raw,9):"CURRENT MAXIMUM "+fmt(phase<4?a:phase<6?b:op.raw,9));
   text("calculation-formula","max(max(A,B),max(C,D)) = "+fmt(op.raw,9)+"；真实导出 "+fmt(op.expected,9));
   text("result-value",fmt(op.expected,9));$("consistency").dataset.passed=String(near(op.raw,op.expected));
   text("operation-note","输入起点 ["+r*2+","+c*2+"]，输出 ["+r+","+c+"]；末行末列不参与完整 2×2 窗口。");text("local-heading","Pool2 · 四值比较与最大值传递");
   let flight=$("advanced-pool-flight");if(!flight){flight=document.createElement("span");flight.id="advanced-pool-flight";flight.style.cssText="position:fixed;z-index:20;padding:6px 10px;background:#174b40;color:#82efbe;border-radius:5px;pointer-events:none;font:12px Consolas";document.body.append(flight);}
   const t=phase===8?state.mode==="step"?1:((state.dwellPhase??scan.phase)*9)%1:-1;flight.hidden=t<0;
   if(t>=0){const a=dom.summary.getBoundingClientRect(),b=$("target-canvas").getBoundingClientRect(),x=a.left+a.width/2,y=a.top+a.height/2,tx=b.left+(c+.5)*b.width/5,ty=b.top+(r+.5)*b.height/5;flight.textContent=fmt(op.raw,6);flight.style.left=(x+(tx-x)*t)+"px";flight.style.top=(y+(ty-y)*t)+"px";}
  }
  text("result-label","真实 PYTORCH 输出");text("result-detail","通道 "+ch+" · 空间位置 ["+r+","+c+"]");
  text("consistency","✓ 与真实导出一致 · abs 1e−4 + rel 1e−5");
  text("source-title",S.current.key==="conv2"?"32 张 Pool1 输入 · 当前 C"+S.selection.input:S.current.key==="relu2"?"Pre-ReLU2 · 11×11":"Pool2 输入 · 11×11");
  text("target-title",S.current.key==="pool2"?"Pool2 输出 · 5×5":S.current.key==="relu2"?"Post-ReLU2 · 11×11":"Conv2 输出 · Filter "+ch);
  text("source-caption",s.channels+" 个通道在同一空间坐标独立处理");text("target-caption","已处理 "+S.local+" / "+S.limit+" · 真实输出逐位置显现");
  text("channel-strip-title",(S.current.key==="conv2"?64:64)+" REAL OUTPUT CHANNELS · 独立并行响应");
  $("row").value=r;$("col").value=c;$("channel").value=ch;$("input-channel").value=S.selection.input;
  host.setViews(cache.source,cache.target,{source:cache.source,target:cache.target,sourceHighlight:{row:r*s.stride,col:c*s.stride,size:s.kernel},targetHighlight:{row:r,col:c}});
  const status=S.globalStatus+"/"+S.localStatus+"/"+state.mode+"/"+S.local;if(controlKey!==status){controlKey=status;updateControls();}
  text("local-position",S.current.key.toUpperCase()+" · "+S.local+" / "+S.limit+(state.mode==="step"&&S.current.key==="pool2"?" · COMPARE "+state.manualPhase+" / 8":""));$("local-progress").style.width=state.position/S.limit*100+"%";
  onRender?.();
 }
 function frame(now){
  if(!handles(S.current.key)||state.mode==="step"||S.globalStatus!=="PLAYING"&&S.localStatus!=="PLAYING")return;
  const duration=durationFor(state.mode,Number($(S.globalStatus==="PLAYING"?"global-speed":"local-speed").value));
  if(!state.metrics)state.metrics=metrics(state.mode,duration,state.position);
  const prior=state.position,delta=advanceClock(state,now,S.limit,duration);if(state.direction===-1){state.position=Math.max(0,prior-delta.advance);state.fraction=state.position%1;}if(delta.dt!==null){state.metrics.activeMs+=delta.dt;state.metrics.intervals.push(delta.dt);}
  state.dwellPhase=null;
  if(S.current.key==="pool2"&&state.mode==="fast"){
   state.fastClock=clamp((state.fastClock??prior/S.limit)+(delta.dt??0)/duration*state.direction,0,1);
   const u=state.fastClock,focus=Math.floor(cache.best/5)*5+cache.best%5;
   if(u<.25)state.position=u/.25*focus;
   else if(u<.75){state.position=focus;state.dwellPhase=(u-.25)/.5;}
   else state.position=focus+(u-.75)/.25*(S.limit-focus);
   state.fraction=state.position%1;
  }
  S.local=Math.min(S.limit,Math.floor(state.position+1e-8));if(S.local===S.limit)S.localStatus="COMPLETED";
  const start=performance.now();render();state.metrics.renderMs.push(performance.now()-start);state.metrics.frames++;
  if(S.local===S.limit||state.direction===-1&&state.position===0){state.metrics.completed=true;state.last=null;S.localStatus=state.direction===-1?"PAUSED":"COMPLETED";render();complete();}else queueFrame(frame);
 }
 function resume(){state.manualIndex=null;state.manualPhase=0;state.last=null;}
 function selected(){const side=spec().side;state.manualIndex=S.selection.row*side+S.selection.col;state.manualPhase=0;cache=null;render();}
 function step(direction){
  state.last=null;
  if(state.manualIndex!==null){state.position=state.manualIndex;state.manualIndex=null;}
  if(S.current.key==="pool2"&&state.mode==="step"){
   const micro=clamp(Math.floor(state.position)*9+state.manualPhase+direction,0,S.limit*9);state.position=Math.floor(micro/9);state.manualPhase=micro%9;
  }else state.position=clamp(Math.floor(state.position)+direction,0,S.limit);
  state.fraction=state.position%1;S.local=Math.floor(state.position);S.localStatus=S.local===S.limit?"COMPLETED":S.local?"PAUSED":"IDLE";render();
 }
 function modeChanged(mode){pause();state.mode=mode;modes[S.current.key]=mode;state.last=null;state.manualPhase=0;state.metrics=null;render();}
 function restart(){reset();S.local=0;S.localStatus="IDLE";render();}
 function openZoom(which){const map=which==="source"?cache.source:cache.target;drawMap($("zoom-canvas"),map,{highlight:which==="source"?{row:S.selection.row*spec().stride,col:S.selection.col*spec().stride,size:spec().kernel}:{row:S.selection.row,col:S.selection.col}});text("zoom-title",which==="source"?"真实输入":"真实输出");text("zoom-caption","通道 "+S.selection.out+" · 真实 Tensor 值");$("zoom-dialog").showModal();}
 return {state,handles,reset,resume,pauseTransfer:()=>{},resumeTransfer:()=>false,parallelChannel:()=>{cache=null;render();},frame,render,selected,step,restart,modeChanged,openZoom,invalidateGeometry:()=>{},snapshot:()=>({key:state.key,mode:state.mode,position:state.position,fraction:state.fraction,manualIndex:state.manualIndex,manualPhase:state.manualPhase})};
}
root.DigitVisionAdvancedSpatial={create,SPECS,point};if(typeof module!=="undefined"&&module.exports)module.exports={SPECS,point};
})(typeof window!=="undefined"?window:globalThis);




