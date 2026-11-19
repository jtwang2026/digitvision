/* Real ReLU1 / Pool1 rendering. The host owns the single RAF scheduler and clock. */
(function(root){
"use strict";
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),mix=(a,b,t)=>a+(b-a)*t;
const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
function scanPoint(position,side=13){
 const p=clamp(position,0,side*side-1),step=Math.floor(p),f=p-step;
 const coord=n=>{const row=Math.floor(n/side),c=n%side;return {row,col:row%2?side-1-c:c};};
 const a=coord(step),b=coord(Math.min(step+1,side*side-1));
 return {row:mix(a.row,b.row,f),col:mix(a.col,b.col,f),index:a.row*side+a.col,step,fraction:f};
}
function traversalIndex(index,side=13){const r=Math.floor(index/side),c=index%side;return r*side+(r%2?side-1-c:c);}
function create(host){
 const {$,S,DATA,fmt,near,poolAt}=host;
 const handles=key=>key==="relu1"||key==="pool1";
 const state={key:"",mode:"fast",position:0,fraction:0,last:null,manualIndex:null,manualPhase:0,transfer:null,transferring:false,transferPaused:false,metrics:null};
 const modes={relu1:"fast",pool1:"fast"};let cache=null,dom=null,detailKey="",controlsKey="",geometry=null,activeCell=-2,tourney=null;
 function text(id,value){const el=typeof id==="string"?$(id):id;if(el.textContent!==value)el.textContent=value;}
 function clearPoolComparison(){state.fastFocus=null;state.inspectClock=null;state.inspectPlaying=false;state.inspectPaused=false;}
 function reset(key=S.current.key){clearPoolComparison();state.key=key;state.mode=modes[key]||"fast";state.position=0;state.fraction=0;state.last=null;state.manualIndex=null;state.manualPhase=0;state.transfer=null;state.transferring=false;state.transferPaused=false;state.metrics=null;cache=null;detailKey="";controlsKey="";geometry=null;}
 function color(value,range){if(value===0)return "#080f1c";const level=clamp(Math.abs(value)/range,0,1),rgb=value<0?[251,113,133]:[94,225,232];return "rgba("+rgb+","+(.08+.92*level)+")";}
 function canvas(){const c=document.createElement("canvas");c.width=c.height=360;return c;}
 function paintPlane(canvas,values,side,range,filter=()=>true){const ctx=canvas.getContext("2d"),cell=canvas.width/side;ctx.clearRect(0,0,canvas.width,canvas.height);for(let i=0;i<values.length;i++)if(filter(values[i])){ctx.fillStyle=color(values[i],range);ctx.fillRect(i%side*cell+.35,Math.floor(i/side)*cell+.35,cell-.7,cell-.7);}}
 function ensureCache(){
  const key=S.sample.index+"/"+S.current.key+"/"+S.selection.out;
  if(cache&&cache.key===key)return;
  const relu=S.current.key==="relu1",source=host.mapFor(relu?"conv1_pre":"conv1_relu",S.selection.out),target=host.mapFor(relu?"conv1_relu":"pool1",S.selection.out);
  const range=Math.max(...source.values.map(Math.abs),1e-9),sourceImage=canvas(),positive=canvas(),negative=canvas(),buffer=canvas();
  paintPlane(sourceImage,source.values,26,range);paintPlane(positive,source.values,26,range,v=>v>0);paintPlane(negative,source.values,26,range,v=>v<0);
  let representative=0;for(let i=1;i<target.values.length;i++)if(target.values[i]>target.values[representative])representative=i;
  if(relu){representative=source.values.indexOf(Math.min(...source.values));}
  cache={key,source,target,range,sourceImage,positive,negative,buffer,painted:0,representative,negativeCount:source.values.filter(v=>v<0).length,positiveCount:source.values.filter(v=>v>0).length,zeroCount:source.values.filter(v=>v===0).length};
  const ctx=buffer.getContext("2d");ctx.fillStyle="#080f1c";ctx.fillRect(0,0,360,360);detailKey="";geometry=null;
 }
 function ensureDOM(){
  if(dom&&$("operation-content").contains(dom.root))return;
  $("operation-content").innerHTML='<div class="spatial-operation"><div class="activation-gate"><span>POINTWISE ACTIVATION</span><strong>max(0, x)</strong><div class="activation-pair"><span id="activation-before"></span><i>→</i><span id="activation-after"></span></div><div class="activation-meter"><i></i></div><p id="activation-counts"></p></div><div class="pool-comparison"><span class="field-label">2 × 2 · STRIDE 2</span><div class="pool-values">'+Array.from({length:4},(_,i)=>'<div class="pool-value" data-cell="'+i+'"><small>x'+i+'</small><strong>0</strong></div>').join("")+'</div><div class="pool-winner"><span>MAXIMUM</span><strong></strong></div><p id="pool-comparison-phase"></p></div><div class="spatial-scale"><div class="response-scale"></div><div class="response-labels"><span>负值</span><span>零</span><span>正值</span></div><small></small></div></div>';
  $("term-grid").replaceChildren(...Array.from({length:4},()=>{const el=document.createElement("div");el.className="term revealed";return el;}));
  dom={root:$("operation-content").firstElementChild,values:[...$("operation-content").querySelectorAll(".pool-value")],terms:[...$("term-grid").children],winner:$("operation-content").querySelector(".pool-winner strong"),meter:$("operation-content").querySelector(".activation-meter i"),scale:$("operation-content").querySelector(".spatial-scale small")};
  if(!$("pool-flight")){const el=document.createElement("div");el.id="pool-flight";el.hidden=true;el.setAttribute("aria-hidden","true");document.querySelector(".stage-grid").append(el);}
  detailKey="";controlsKey="";activeCell=-2;
  if(state.metrics)state.metrics.domBuilds++;
 }
 function geometryFor(){
  if(geometry)return geometry;const grid=document.querySelector(".stage-grid").getBoundingClientRect(),target=$("target-canvas").getBoundingClientRect();
  const result=dom.root.querySelector(".pool-winner").getBoundingClientRect();
  geometry={result:{x:result.left+result.width/2-grid.left,y:result.top+result.height/2-grid.top},target:{x:target.left-grid.left,y:target.top-grid.top,w:target.width,h:target.height},cells:dom.values.map(el=>{const r=el.getBoundingClientRect();return {x:r.left+r.width/2-grid.left,y:r.top+r.height/2-grid.top};})};return geometry;
 }
 function frame(now){
  if(state.key==="pool1"&&S.current.key==="pool1"&&state.inspectPlaying){
   host.advanceClock(state.inspectClock,now,1,state.mode==="fast"?4000:6000);
   if(state.inspectClock.position>=1)state.inspectPlaying=false;
   render();if(state.inspectPlaying)host.queueFrame(frame);return;
  }
  if(handles(S.current.key)&&state.key==="pool1"&&state.mode==="step"&&state.manualPhase===8&&state.transferring){host.advanceClock(state.transfer,now,1,650);render();if(state.transfer.position<1)host.queueFrame(frame);else{state.transferring=false;render();}return;}
  if(!handles(S.current.key)||state.mode==="step"||S.globalStatus!=="PLAYING"&&S.localStatus!=="PLAYING")return;
  const duration=host.duration(state.mode,Number($(S.globalStatus==="PLAYING"?"global-speed":"local-speed").value));
  if(!state.metrics)state.metrics=host.metrics(state.mode,duration,state.position);
  const m=state.metrics,delta=host.advanceClock(state,now,S.limit,duration);if(delta.dt!==null){m.activeMs+=delta.dt;m.intervals.push(delta.dt);}
  S.local=Math.min(S.limit,Math.floor(state.position+1e-8));if(S.local===S.limit)S.localStatus="COMPLETED";
  const start=performance.now();render();m.renderMs.push(performance.now()-start);m.frames++;
  if(S.local===S.limit){m.completed=true;state.last=null;host.complete();}else host.queueFrame(frame);
 }
 function pauseTransfer(){if(state.key==="pool1"&&state.inspectPlaying){state.inspectPlaying=false;state.inspectPaused=true;state.inspectClock.last=null;}if(state.transferring){state.transferring=false;state.transferPaused=true;state.transfer.last=null;}}
 function resumeTransfer(){if(state.key==="pool1"&&state.inspectPaused){state.inspectPaused=false;state.inspectPlaying=true;state.inspectClock.last=null;render();host.queueFrame(frame);return true;}if(state.mode!=="step"||!state.transferPaused||!state.transfer||state.transfer.position>=1)return false;state.transferPaused=false;state.transferring=true;state.transfer.last=null;render();host.queueFrame(frame);return true;}
 function resume(){if(state.key==="pool1"){state.fastFocus=state.manualIndex??state.fastFocus??cache?.representative??0;state.inspectClock=null;state.inspectPlaying=false;state.inspectPaused=false;}state.manualIndex=null;state.manualPhase=0;detailKey="";}
 function modeChanged(mode){host.pause();if(state.key==="pool1")clearPoolComparison();state.mode=mode;modes[state.key]=mode;state.manualPhase=0;state.fraction=state.position%1;state.transfer=null;state.transferring=false;state.transferPaused=false;state.metrics=null;state.last=null;render();}
 function selected(){
  if(state.key==="pool1")clearPoolComparison();
  state.transferring=false;state.transferPaused=false;state.transfer=null;
  if(cache&&cache.target.channel!==S.selection.out){reset(S.current.key);S.local=0;S.localStatus="IDLE";}
  else{state.manualIndex=S.selection.row*(S.current.key==="relu1"?26:13)+S.selection.col;state.manualPhase=0;}
  render();
 }
 function step(direction){
  if(state.key==="pool1"){state.inspectClock=null;state.inspectPlaying=false;state.inspectPaused=false;}
  state.transferring=false;state.transferPaused=false;state.transfer=null;
  if(state.key==="pool1"&&state.mode==="step"){
   if(state.manualIndex!==null){state.position=traversalIndex(state.manualIndex);state.manualIndex=null;state.manualPhase=0;}
   let micro=Math.floor(state.position)*9+state.manualPhase+direction;micro=clamp(micro,0,169*9);
   state.position=Math.floor(micro/9);state.manualPhase=micro%9;state.fraction=state.manualPhase/9;S.local=Math.min(169,Math.floor(state.position));
  }else{state.manualIndex=null;state.position=clamp(Math.floor(state.position)+direction,0,S.limit);state.fraction=0;state.manualPhase=0;S.local=Math.floor(state.position);}
  if(state.key==="pool1"&&state.mode==="step"&&state.manualPhase===8){state.transfer={position:direction>0?0:1,last:null,fraction:0};state.transferring=direction>0;}
  state.last=null;S.localStatus=S.local===S.limit?"COMPLETED":S.local?"PAUSED":"IDLE";render();if(state.transferring)host.queueFrame(frame);
 }
 /** Switch detail channel without rewinding the parallel tensor scan. */
 function parallelChannel(){state.transferring=false;state.transferPaused=false;state.transfer=null;state.inspectClock=null;state.inspectPlaying=false;state.inspectPaused=false;state.manualIndex=null;cache=null;detailKey="";geometry=null;render();}
 function restart(){reset(S.current.key);S.local=0;S.localStatus="IDLE";render();}
 function replayComparison(){
  host.pause();state.manualIndex=S.selection.row*13+S.selection.col;
  state.inspectClock={position:0,last:null,fraction:0};state.inspectPlaying=true;state.inspectPaused=false;
  state.manualPhase=0;state.transferring=false;state.transferPaused=false;state.transfer=null;render();host.queueFrame(frame);
 }
 function ensureTournament(){
  if(tourney&&dom.root.contains(tourney.root))return;
  tourney=root.DigitVisionPoolComparison.create(dom.root.querySelector(".pool-comparison"),{format:fmt,replay:replayComparison});geometry=null;
 }
 function render(){
  if(!handles(S.current.key))return;
  if(state.key!==S.current.key)reset(S.current.key);
  ensureCache();ensureDOM();
  const relu=state.key==="relu1",playing=S.globalStatus==="PLAYING"||S.localStatus==="PLAYING",complete=S.local===S.limit;
  const point=scanPoint(state.position),phase=state.mode==="step"?state.manualPhase/9:state.fraction;
  if(!relu&&state.mode==="fast"&&playing&&state.fastFocus==null)state.fastFocus=cache.representative;
  const overview=!relu&&state.mode==="fast"&&state.fastFocus!=null&&state.manualIndex===null&&!state.inspectClock;
  const index=state.manualIndex!==null?state.manualIndex:overview?state.fastFocus:relu||complete?cache.representative:point.index;
  let poolView=null;
  if(!relu){
   ensureTournament();
   poolView=state.inspectClock?root.DigitVisionPoolComparison.timeline(state.inspectClock.position):complete?{stage:8,progress:1}:state.mode==="step"||state.manualIndex!==null?{stage:state.manualPhase,progress:[2,4,6].includes(state.manualPhase)?.6:state.manualPhase===8?(state.transfer?.position??1):1}:root.DigitVisionPoolComparison.timeline(overview?state.position/169:phase);
  }
  const side=relu?26:13,r=Math.floor(index/side),c=index%side;
  S.selection.row=r;S.selection.col=c;
  document.body.classList.toggle("relu1-view",relu);document.body.classList.toggle("pool1-view",!relu);
  $("layer-animation-controls").hidden=false;$("layer-animation-mode").value=state.mode;
  text("layer-animation-state",playing?"真实记录回放 · "+(state.mode==="fast"?"Fast Overview":"Detailed"):"Detailed Inspection · 真实数值");
  const duration=host.duration(state.mode,Number($(S.globalStatus==="PLAYING"?"global-speed":"local-speed").value));text("layer-animation-duration",duration===null?"手动检查":duration/1000+" s · 动画时长，非 CNN 推理耗时");
  const oldLegend=$("conv1-color-legend");if(oldLegend)oldLegend.remove();$("conv1-input-detail").hidden=true;
  text("source-title",relu?"Pre-ReLU Feature Map · 26 × 26":"输入 Feature Map · 26 × 26");text("target-title",relu?"Post-ReLU Feature Map · 26 × 26":"Max Pooling 输出 · 13 × 13");
  text("source-caption",relu?"粉色：负响应 · 青色：正响应 · 深色：零":"真实 ReLU1 输出；每个窗口覆盖 2×2 响应。");
  text("operation-label",relu?"NEGATIVE → ZERO · POSITIVE UNCHANGED":"FOUR VALUES → ONE MAXIMUM");
  text("operation-note",relu?"仅颜色透明度作视觉过渡；检查面板始终显示真实输入和最终输出。32 个通道可并行处理。":"每个通道有169个池化位置；32个通道可并行处理。金色标记局部最大值；并列最大值同样标记，传递第一个最大位置。");
  text("channel-strip-title","32 REAL OUTPUT CHANNELS · 并行处理，选择通道浏览");
  text(dom.scale,"共同色阶："+fmt(-cache.range,4)+" / 0 / "+fmt(cache.range,4)+" · 前后使用相同范围");
  $("row").value=r;$("col").value=c;$("channel").value=S.selection.out;
  const src=$("source-canvas"),dst=$("target-canvas"),sctx=src.getContext("2d"),ctx=dst.getContext("2d"),bg="#080f1c";
  sctx.fillStyle=bg;sctx.fillRect(0,0,360,360);sctx.drawImage(cache.sourceImage,0,0);
  ctx.fillStyle=bg;ctx.fillRect(0,0,360,360);
  let sourceHighlight,targetHighlight;
  if(relu){
   const progress=clamp(state.position/100,0,1);ctx.drawImage(cache.positive,0,0);ctx.globalAlpha=1-progress;ctx.drawImage(cache.negative,0,0);ctx.globalAlpha=1;
   dom.meter.style.width=progress*100+"%";
   text("activation-counts",cache.negativeCount+" 负值归零 · "+cache.positiveCount+" 正值不变 · "+cache.zeroCount+" 零值不变");
   text("target-caption",complete?"完整真实 ReLU 输出 · 所有负值均已归零":"视觉过渡 "+Math.round(progress*100)+"% · 最终数值以检查面板为准");
   sourceHighlight={row:r,col:c};targetHighlight={row:r,col:c};host.setWindow(null);
   for(const context of [sctx,ctx]){context.strokeStyle="#f6c76b";context.lineWidth=2;context.strokeRect(c*360/26+1,r*360/26+1,360/26-2,360/26-2);}
   $("pool-flight").hidden=true;
  }else{
   const bctx=cache.buffer.getContext("2d"),cw=360/13;
   if(cache.painted>S.local){bctx.fillStyle=bg;bctx.fillRect(0,0,360,360);cache.painted=0;}
   for(let i=cache.painted;i<S.local;i++){const p=scanPoint(i);bctx.fillStyle=color(cache.target.values[p.index],cache.range);bctx.fillRect(p.col*cw+.5,p.row*cw+.5,cw-1,cw-1);if(state.metrics)state.metrics.revealedPositions++;}
   cache.painted=S.local;ctx.drawImage(cache.buffer,0,0);
   const moving=playing&&state.manualIndex===null,followCursor=state.mode!=="step"&&state.manualIndex===null&&!complete,visual=state.mode==="fast"?point:scanPoint(Math.floor(state.position)+smooth((phase-.82)/.18));
   const vr=followCursor?visual.row:r,vc=followCursor?visual.col:c;
   host.setWindow({row:vr*2,col:vc*2,size:2});ctx.strokeStyle=overview?"#5ee1e8":"#f6c76b";ctx.lineWidth=2;ctx.strokeRect(vc*cw+1,vr*cw+1,cw-2,cw-2);if(overview){ctx.strokeStyle="#f6c76b";ctx.strokeRect(c*cw+1,r*cw+1,cw-2,cw-2);sctx.strokeStyle="#f6c76b";sctx.lineWidth=2;sctx.strokeRect(c*2*360/26+1,r*2*360/26+1,2*360/26-2,2*360/26-2);}
   if(complete||poolView.stage===8&&poolView.progress>=.98){ctx.globalAlpha=1;ctx.fillStyle=color(cache.target.values[index],cache.range);ctx.fillRect(c*cw+.5,r*cw+.5,cw-1,cw-1);ctx.globalAlpha=1;}
   if(complete){host.setWindow(null);sctx.strokeStyle="#f6c76b";sctx.lineWidth=2;sctx.strokeRect(c*2*360/26+1,r*2*360/26+1,2*360/26-2,2*360/26-2);}
   sourceHighlight={row:r*2,col:c*2,size:2};targetHighlight={row:r,col:c};
   text("target-caption",complete?"完整169/169 · 26×26 压缩为13×13":"已写入 "+S.local+" / 169 · 检查焦点不改变播放进度");
  }
  src.dataset.inspectIndex=dst.dataset.inspectIndex=index;dst.dataset.written=String(complete);dst.dataset.progress=String(state.position/S.limit);src.dataset.scale=dst.dataset.scale=cache.range;
  host.setViews(cache.source,cache.target,{source:cache.source,target:cache.target,sourceHighlight,targetHighlight});
  const key=cache.key+"/"+index;
  if(detailKey!==key){
   detailKey=key;if(state.metrics)state.metrics.detailUpdates++;
   text("local-heading",relu?"逐元素检查：负值截断为零":"局部检查：四值比较与最大值位置");
   if(relu){
    const raw=cache.source.values[index],expected=cache.target.values[index];text("activation-before",fmt(raw,7));text("activation-after",fmt(expected,7));$("activation-before").classList.toggle("negative",raw<0);
    text(dom.terms[0],"原始响应："+fmt(raw,9));text(dom.terms[1],"操作：max(0, x)");text(dom.terms[2],"最终输出："+fmt(expected,9));text(dom.terms[3],raw<0?"负值 → 0":raw===0?"零值 → 0":"正值保持不变");
    text("calculation-formula","Y["+S.selection.out+","+r+","+c+"] = max(0, "+fmt(raw,9)+") = "+fmt(expected,9));text("result-value",fmt(expected,9));$("consistency").dataset.passed=String(near(Math.max(0,raw),expected));
   }else{
    const op=poolAt(S.sample,"pool1",S.selection.out,r,c),winner=op.values.indexOf(op.raw);cache.op=op;cache.winner=winner;
    dom.values.forEach((el,i)=>{text(el.querySelector("strong"),fmt(op.values[i],6));el.title=String(op.values[i]);el.dataset.winner=String(op.values[i]===op.raw);});
    op.values.forEach((v,i)=>text(dom.terms[i],"X["+(r*2+Math.floor(i/2))+","+(c*2+i%2)+"] = "+fmt(v,8)+(v===op.raw?" ← max":"")));
    text("calculation-formula","max("+op.values.map(v=>fmt(v,7)).join(", ")+") = "+fmt(op.raw,9)+" · 首个最大值位置 ["+Math.floor(winner/2)+","+winner%2+"]");text("result-value",fmt(op.expected,9));$("consistency").dataset.passed=String(near(op.raw,op.expected));
   }
   text("result-label","真实 PYTORCH 输出");text("result-detail","通道 "+S.selection.out+" · 位置 ["+r+", "+c+"] · 点击任意位置检查");text("consistency","✓ 与真实导出值一致 · abs 1e−4 + rel 1e−5");
  }
  if(!relu){
   const view=tourney.render(cache.op.values,{...poolView,row:r,col:c,step:state.mode==="step",overview,inspection:!!state.inspectClock,inspectPlaying:state.inspectPlaying});
   const names=["A","B","C","D"],descriptions=["四个真实候选值","选中 A","A vs B","确定第一组最大值","C vs D","确定第二组最大值","Winner 1 vs Winner 2","确定最终最大值","写入输出 Feature Map"];
   text("pool-comparison-phase","STEP "+view.stage+" / 8 · "+descriptions[view.stage]+([2,4,6].includes(view.stage)?poolView.progress<.22?" · START":" · COMPARE":[3,5,7].includes(view.stage)?" · RESULT":""));
   $("pool-comparison-phase").dataset.phase=String(view.stage);
   cache.op.values.forEach((value,i)=>text(dom.terms[i],names[i]+" = X["+(r*2+Math.floor(i/2))+","+(c*2+i%2)+"] = "+fmt(value,8)+(view.final&&view.winners.includes(i)?" ← 并列/最终最大值":"")));
   text("result-value",view.final?fmt(cache.op.expected,9):"—");
   text("calculation-formula",view.stage===0?"max(max(A,B), max(C,D)) · 等待比较":view.stage<=3?"第1组：max(A,B)"+(view.stage===3?" = "+fmt(view.row1,9):""):view.stage<=5?"第2组：max(C,D)"+(view.stage===5?" = "+fmt(view.row2,9):""):view.final?"max("+fmt(view.row1,9)+", "+fmt(view.row2,9)+") = "+fmt(view.maximum,9):"最终比较：Winner 1 vs Winner 2");
   text("operation-note","A/B、C/D 分组比较，再比较两组胜出者。相等值保留并列标记，仅传递首个最大值；32个通道可并行处理。");
   if(overview){text("source-caption","青色：169位置扫描 · 金色：中央教学窗口，三轮比较独立慢放。");text("target-caption","扫描 "+S.local+" / 169 · 金色标记当前教学窗口 ["+r+","+c+"]。");}
   if(state.inspectClock)text("target-caption","独立比较演示 · 全图扫描进度保持 "+S.local+" / 169；写入为当前窗口预览。");
   const travel=view.stage===8?poolView.progress:-1,flight=$("pool-flight");flight.hidden=travel<0;
   if(!flight.hidden){const geo=geometryFor(),from=geo.result,to={x:geo.target.x+(c+.5)*geo.target.w/13,y:geo.target.y+(r+.5)*geo.target.h/13};text(flight,fmt(view.maximum,6));flight.dataset.source=names[view.firstIndex];flight.style.transform="translate3d("+mix(from.x,to.x,travel)+"px,"+mix(from.y,to.y,travel)+"px,0) translate(-50%,-50%)";flight.style.opacity="1";}
  }

  const controlKey=S.globalStatus+"/"+S.localStatus+"/"+state.mode+"/"+(S.local===0)+"/"+complete+(relu?"":"/"+state.inspectPlaying+"/"+state.inspectPaused);
  if(controlsKey!==controlKey){controlsKey=controlKey;host.updateControls();}
  if(!relu&&state.mode==="step"){$("local-previous").disabled=state.position===0&&state.manualPhase===0;$("local-play").disabled=!state.transferPaused;$("local-pause").disabled=!state.transferring;}
  if(!relu&&(state.inspectPlaying||state.inspectPaused)){$("local-play").disabled=state.inspectPlaying;$("local-pause").disabled=!state.inspectPlaying;}
  text("local-position",relu?"TRANSITION "+Math.round(state.position)+"%":"POSITION "+S.local+" / 169"+(state.mode==="step"?" · COMPARISON "+state.manualPhase+" / 8":""));$("local-progress").style.width=state.position/S.limit*100+"%";
  $("channel-strip").querySelectorAll("button").forEach(button=>button.classList.toggle("active",Number(button.dataset.channel)===S.selection.out));
  host.onRender?.();
 }
/** Magnify the actual selected neighborhood, retaining the common signed color range. */
 function openZoom(which){
  ensureCache();
  const map=which==="source"?cache.source:cache.target,canvas=$("zoom-canvas"),ctx=canvas.getContext("2d"),poolInput=state.key==="pool1"&&which==="source";
  const selectedRow=S.selection.row*(poolInput?2:1),selectedCol=S.selection.col*(poolInput?2:1),size=poolInput?2:Math.min(5,map.w);
  const row=poolInput?selectedRow:clamp(selectedRow-Math.floor(size/2),0,map.h-size),col=poolInput?selectedCol:clamp(selectedCol-Math.floor(size/2),0,map.w-size),cw=canvas.width/size;
  ctx.fillStyle="#080f1c";ctx.fillRect(0,0,canvas.width,canvas.height);
  const local=[];
  for(let r=0;r<size;r++)for(let c=0;c<size;c++){
   const value=map.values[(row+r)*map.w+col+c];local.push(value);ctx.fillStyle=color(value,cache.range);ctx.fillRect(c*cw+1,r*cw+1,cw-2,cw-2);
   ctx.fillStyle="#dbe8f6";ctx.font="15px Consolas,monospace";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(fmt(value,5),(c+.5)*cw,(r+.5)*cw);
   ctx.fillStyle="#8d9bb3";ctx.font="10px Consolas,monospace";ctx.fillText("["+(row+r)+","+(col+c)+"]",(c+.5)*cw,r*cw+17);
  }
  ctx.strokeStyle="#f6c76b";ctx.lineWidth=3;
  if(poolInput){const maximum=Math.max(...local);local.forEach((value,i)=>{if(value===maximum)ctx.strokeRect(i%2*cw+2,Math.floor(i/2)*cw+2,cw-4,cw-4);});}
  else ctx.strokeRect((selectedCol-col)*cw+2,(selectedRow-row)*cw+2,cw-4,cw-4);
  canvas.dataset.localValues=JSON.stringify(local);canvas.dataset.region=JSON.stringify({row,col,size,selectedRow,selectedCol});
  text("zoom-title",(which==="source"?"真实输入":"真实最终输出")+" · 局部 "+size+"×"+size+" · 通道 "+S.selection.out);
  text("zoom-caption","共同色阶 ±"+fmt(cache.range,6)+" · 数值未经动画插值 · 金色标记"+(poolInput?"全部并列最大值":"选中位置 ["+selectedRow+","+selectedCol+"]"));
  $("zoom-dialog").showModal();
 }
 window.addEventListener("resize",()=>{geometry=null;});
 return {state,handles,reset,resume,parallelChannel,pauseTransfer,resumeTransfer,frame,render,selected,step,restart,modeChanged,openZoom,invalidateGeometry:()=>{geometry=null;if(tourney)tourney.invalidateGeometry();},snapshot:()=>({key:state.key,mode:state.mode,position:state.position,fraction:state.fraction,manualIndex:state.manualIndex,manualPhase:state.manualPhase,transferring:state.transferring,transferPaused:state.transferPaused,transferProgress:state.transfer?state.transfer.position:0,inspectionPlaying:!!state.inspectPlaying,inspectionPaused:!!state.inspectPaused,inspectionProgress:state.inspectClock?.position??null,fastFocus:state.fastFocus??null})};
}
const API={create,scanPoint,traversalIndex};if(typeof module!=="undefined"&&module.exports)module.exports=API;root.DigitVisionSpatialLayers=API;
})(typeof window!=="undefined"?window:globalThis);
