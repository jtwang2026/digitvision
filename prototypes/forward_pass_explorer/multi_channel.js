/* Canvas tensor overview. Read-only rendering; the existing layer clocks own computation display. */
(function(root){
"use strict";
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function responses(t,row,col){const [b,c,h,w]=t.shape;if(b!==1||row<0||row>=h||col<0||col>=w)throw RangeError("Tensor coordinate out of bounds");return Array.from({length:c},(_,ch)=>t.values[(ch*h+row)*w+col]);}
function poolResponses(sample,row,col){const source=sample.tensors.conv1_relu,target=sample.tensors.pool1;return responses(target,row,col).map((value,ch)=>{const values=[];for(let m=0;m<2;m++)for(let n=0;n<2;n++)values.push(source.values[(ch*26+row*2+m)*26+col*2+n]);return {channel:ch,values,maximum:Math.max(...values),output:value};});}
function create(host){
 const {S,DATA}=host,$=id=>document.getElementById(id),panel=$("multi-channel-overview");
 panel.innerHTML='<div class="mc-heading"><div><p class="eyebrow">THE WHOLE TENSOR</p><h3>Multi-Channel Overview</h3><p id="mc-shape"></p></div><div><button id="mc-detail" class="button">查看单通道细节 ↓</button><button id="mc-collapse" class="button ghost">折叠总览</button></div></div><div id="mc-body"><div class="mc-controls"><label>同位置 r <input id="mc-row" type="number" min="0"></label><label>c <input id="mc-col" type="number" min="0"></label><span id="mc-position"></span><span id="mc-status"></span></div><p class="mc-teaching" id="mc-teaching"></p><div class="mc-atlases"><article id="mc-source-pane"><h4 id="mc-source-title"></h4><canvas id="mc-source" width="1024" height="568" tabindex="0" aria-label="All input channels; click a channel"></canvas></article><article><h4 id="mc-target-title"></h4><canvas id="mc-target" width="1024" height="568" tabindex="0" aria-label="All output channels; click a channel"></canvas></article></div><div class="mc-response"><h4 id="mc-response-title">同一空间位置的真实响应</h4><canvas id="mc-responses" width="1100" height="200" aria-label="Per-channel response values; no cross-channel pooling"></canvas><output id="mc-value"></output></div><section id="mc-conv2" hidden><h4>Conv2 · 32 组真实 3×3 权重 → 跨通道加权融合</h4><canvas id="mc-weights" width="1024" height="380" aria-label="32 true kernels for the selected Conv2 output channel"></canvas><canvas id="mc-fusion" width="1100" height="380" aria-label="32 signed contributions converge to one accumulator; bias added once"></canvas><p id="mc-fusion-summary"></p></section></div>';
 let cache=null,weightsCache=null,fusionCache=null,raf=0,dirty=true,lastPaint=-Infinity,visible=true,collapsed=false,locked=null,lastContext="";
 const metrics={frames:0,paintMs:[],cacheBuilds:0};
 const text=(id,value)=>{if($(id).textContent!==value)$(id).textContent=value;};
 const range=t=>Math.max(Math.abs(t.min),Math.abs(t.max),1e-9);
 const color=(v,max,weight=false)=>v===0?"#080f1c":"rgba("+(v<0?"251,113,133":weight?"167,139,250":"94,225,232")+","+(.06+.94*Math.min(1,Math.abs(v)/max))+")";
 function blank(w,h){const c=document.createElement("canvas");c.width=w;c.height=h;return c;}
 /** One atlas for all channels, with one tensor-global range; no per-channel normalization. */
 function atlas(t,max,weight=false){
  const cols=innerWidth<760?4:8,rows=Math.ceil(t.shape[1]/cols),tw=1024/cols,th=tw+14,c=blank(1024,rows*th),ctx=c.getContext("2d"),positive=blank(c.width,c.height),negative=blank(c.width,c.height),base=blank(c.width,c.height),rects=[];
  const h=t.shape[2],w=t.shape[3];ctx.fillStyle="#080f1c";ctx.fillRect(0,0,c.width,c.height);base.getContext("2d").drawImage(c,0,0);
  for(let ch=0;ch<t.shape[1];ch++){
   const x=ch%cols*tw,y=Math.floor(ch/cols)*th,rect={x:x+7,y:y+6,w:tw-14,h:tw-14,labelY:y+th-5};rects.push(rect);
   for(const canvas of [c,base]){const cc=canvas.getContext("2d");cc.strokeStyle="#1d2c42";cc.strokeRect(x+2,y+2,tw-4,th-4);cc.fillStyle="#94a7c3";cc.font="11px Consolas";cc.textAlign="left";cc.fillText("ch"+String(ch).padStart(2,"0"),x+8,rect.labelY);}
   for(let r=0;r<h;r++)for(let col=0;col<w;col++){
    const v=t.values[(ch*h+r)*w+col],px=rect.x+col*rect.w/w,py=rect.y+r*rect.h/h,pw=rect.w/w,ph=rect.h/h;ctx.fillStyle=color(v,max,weight);ctx.fillRect(px+.2,py+.2,pw-.4,ph-.4);
    if(v!==0){const cc=(v>0?positive:negative).getContext("2d");cc.fillStyle=color(v,max,weight);cc.fillRect(px+.2,py+.2,pw-.4,ph-.4);}
   }
  }
  return {tensor:t,range:max,full:c,positive,negative,base,rects,cols,tw,th,buffer:null,painted:0};
 }
 function ensure(){
  const t=S.sample.tensors[S.current.tensor],applicable=t?.shape.length===4&&t.shape[1]>1;panel.hidden=!applicable;if(!applicable)return false;
  const key=S.sample.index+"/"+S.current.key+"/"+(innerWidth<760?4:8);
  if(cache?.key===key)return true;
  const kind=S.current.key,paired=["relu1","pool1","conv2"].includes(kind),source=S.sample.tensors[kind==="conv2"?"pool1":kind==="pool1"?"conv1_relu":"conv1_pre"];
  const shared=paired&&kind!=="conv2"?Math.max(range(source),range(t)):range(t);
  cache={key,kind,paired,source:paired?atlas(source,kind==="conv2"?range(source):shared):null,target:atlas(t,shared),positions:0};metrics.cacheBuilds++;weightsCache=null;fusionCache=null;locked=null;
  $("mc-source-pane").hidden=!paired;$("mc-conv2").hidden=kind!=="conv2";panel.classList.toggle("mc-single",!paired);
  text("mc-shape",S.current.label+" · ["+t.shape.join(", ")+"] · "+t.shape[1]+" 个通道同时存在");
  text("mc-source-title",kind==="pool1"?"32 张输入 · [1,32,26,26]":kind==="relu1"?"32 通道 · 激活前":"32 张输入 · [1,32,13,13]");
  text("mc-target-title",t.shape[1]+" 张真实 "+S.current.label+" 输出 · 共同色阶 ±"+host.fmt(shared,4));
  if(kind==="relu1")text("mc-target-title","32通道同步激活过渡 → 真实 ReLU 输出 · 同一色阶");
  text("mc-teaching",kind==="relu1"?"负值颜色同步淡至零背景，正值保持不变。中间透明度仅用于视觉过渡；下方响应面板始终显示真实最终 Tensor。空间坐标对齐不表示通道语义相同。":kind==="pool1"?"32 个通道各自取 2×2 最大值，不混合通道。青色窗口在全部通道同步移动；下方仅比较响应差异，绝不对32个通道的最大值再次取最大。":kind==="conv2"?"Conv2 融合32个输入通道：每个通道产生局部加权贡献，全部贡献相加，再加入一次 Bias。逐条高亮仅为教学顺序，实际运算可并行。":"同一空间坐标仅表示空间对齐，不表示不同通道学习了相同语义。32/64个通道在真实 CNN 中可并行计算。");
  for(const [id,a] of [["mc-source",cache.source],["mc-target",cache.target]])if(a){$(id).width=a.full.width;$(id).height=a.full.height;}
  return true;
 }
 function coordinate(play){
  const h=cache.target.tensor.shape[2],w=cache.target.tensor.shape[3];
  if(locked&&!play.playing)return {...locked};
  if(play.playing)locked=null;
  if(cache.kind==="pool1"&&play.playing){const p=root.DigitVisionSpatialLayers.scanPoint(play.position,13);return {r:Math.floor(p.index/13),c:p.index%13,vr:p.row,vc:p.col};}
  if(cache.kind==="conv1"&&play.playing){const p=host.conv1ScanPoint(play.position);return {r:Math.floor(p.index/26),c:p.index%26,vr:p.row,vc:p.col};}
  return {r:clamp(S.selection.row,0,h-1),c:clamp(S.selection.col,0,w-1)};
 }
 function prefix(a,count,side){
  if(!a.buffer||count<a.painted){a.buffer=blank(a.full.width,a.full.height);a.buffer.getContext("2d").drawImage(a.base,0,0);a.painted=0;}
  const ctx=a.buffer.getContext("2d"),w=a.tensor.shape[3],h=a.tensor.shape[2];
  for(let step=a.painted;step<count;step++){const p=root.DigitVisionSpatialLayers.scanPoint(step,side),r=Math.floor(p.index/side),c=p.index%side;for(let ch=0;ch<a.rects.length;ch++){const rect=a.rects[ch];ctx.fillStyle=color(a.tensor.values[(ch*h+r)*w+c],a.range);ctx.fillRect(rect.x+c*rect.w/w+.2,rect.y+r*rect.h/h+.2,rect.w/w-.4,rect.h/h-.4);}}
  a.painted=count;return a.buffer;
 }
 function drawAtlas(id,a,coord,options={}){
  const canvas=$(id),ctx=canvas.getContext("2d"),h=a.tensor.shape[2],w=a.tensor.shape[3];
  if(options.relu!==undefined){ctx.drawImage(a.base,0,0);ctx.drawImage((options.reluSource||a).positive,0,0);ctx.globalAlpha=1-options.relu;ctx.drawImage((options.reluSource||a).negative,0,0);ctx.globalAlpha=1;}
  else ctx.drawImage(options.prefix!==undefined?prefix(a,options.prefix,options.side):a.full,0,0);
  const values=responses(a.tensor,coord.r,coord.c),maxima=[],winnerIndices=[];
  for(let ch=0;ch<a.rects.length;ch++){
   const rect=a.rects[ch],size=options.size||1,r=coord.vr??coord.r,c=coord.vc??coord.c;
   if(options.preview){ctx.fillStyle=color(values[ch],a.range);ctx.fillRect(rect.x+coord.c*rect.w/w+.2,rect.y+coord.r*rect.h/h+.2,rect.w/w-.4,rect.h/h-.4);}
   ctx.fillStyle="rgba(94,225,232,.15)";ctx.fillRect(rect.x+c*rect.w/w,rect.y+r*rect.h/h,size*rect.w/w,size*rect.h/h);ctx.strokeStyle="#5ee1e8";ctx.lineWidth=1.5;ctx.strokeRect(rect.x+c*rect.w/w+.5,rect.y+r*rect.h/h+.5,size*rect.w/w-1,size*rect.h/h-1);
   if(options.poolStage!==undefined){const input=[];for(let m=0;m<2;m++)for(let n=0;n<2;n++)input.push(a.tensor.values[(ch*h+coord.r+m)*w+coord.c+n]);const view=root.DigitVisionPoolComparison.comparisonState(input,options.poolStage);maxima.push(view.maximum);winnerIndices.push(view.winners);for(let i=0;i<4;i++){const px=rect.x+(coord.c+i%2)*rect.w/w,py=rect.y+(coord.r+Math.floor(i/2))*rect.h/h;if(view.eliminated.includes(i)){ctx.fillStyle="rgba(8,15,28,.65)";ctx.fillRect(px,py,rect.w/w,rect.h/h);}if(view.final&&view.winners.includes(i)||view.comparing.includes(i)||!view.final&&view.stage>=3&&view.winners1.includes(i)||!view.final&&view.stage>=5&&view.winners2.includes(i)){ctx.strokeStyle=view.final&&view.winners.includes(i)?"#34d399":view.comparing.includes(i)?"#5ee1e8":"#f6c76b";ctx.lineWidth=1.3;ctx.strokeRect(px+.5,py+.5,rect.w/w-1,rect.h/h-1);}}}
   ctx.fillStyle="#a8bbd3";ctx.font="10px Consolas";ctx.textAlign="right";ctx.fillText(host.fmt(values[ch],4),rect.x+rect.w,rect.labelY);
   const selected=options.input?S.selection.input:S.selection.out;if(ch===selected){ctx.strokeStyle="#f6c76b";ctx.lineWidth=3;ctx.strokeRect(ch%a.cols*a.tw+2,Math.floor(ch/a.cols)*a.th+2,a.tw-4,a.th-4);}
  }
  if(maxima.length){canvas.dataset.maxima=JSON.stringify(maxima);canvas.dataset.winnerIndices=JSON.stringify(winnerIndices);}
  canvas.dataset.channels=a.rects.length;canvas.dataset.row=coord.r;canvas.dataset.col=coord.c;canvas.dataset.scale=a.range;canvas.dataset.region=JSON.stringify({row:coord.r,col:coord.c,size:options.size||1});canvas.dataset.revealed=options.prefix??a.tensor.shape[2]*a.tensor.shape[3];
 }
 function chart(values,coord){
  const c=$("mc-responses"),ctx=c.getContext("2d"),max=Math.max(...values.map(Math.abs),1e-9),span=(c.width-60)/values.length,base=95;
  ctx.fillStyle="#0b1423";ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle="#34475f";ctx.beginPath();ctx.moveTo(45,base);ctx.lineTo(c.width,base);ctx.stroke();
  values.forEach((v,ch)=>{const height=v/max*70,x=50+ch*span;ctx.fillStyle=color(v,cache.target.range);ctx.fillRect(x,Math.min(base,base-height),span-4,Math.max(1,Math.abs(height)));if(ch===S.selection.out){ctx.strokeStyle="#f6c76b";ctx.strokeRect(x-1,Math.min(base,base-height)-1,span-2,Math.max(3,Math.abs(height)+2));}ctx.fillStyle="#8fa5c1";ctx.font="10px Consolas";ctx.textAlign="center";if(values.length===32||ch%2===0)ctx.fillText("P"+ch,x+span/2,183);});
  ctx.textAlign="left";ctx.fillStyle="#8096b2";ctx.fillText(host.fmt(max,4),2,25);ctx.fillText("0",20,98);ctx.fillText(host.fmt(-max,4),2,165);
  c.dataset.values=JSON.stringify(values);c.dataset.row=coord.r;c.dataset.col=coord.c;text("mc-response-title","同位置跨通道响应 · ("+coord.r+", "+coord.c+") · 真实最终 Tensor 值");text("mc-value","P"+S.selection.out+"("+coord.r+","+coord.c+") = "+host.fmt(values[S.selection.out],9)+" · 仅对比强度，不进行跨通道取最大值");
 }
 function fusion(coord,now){
  const context=S.sample.index+"/conv2";const key=context+"/"+S.selection.out+"/"+coord.r+"/"+coord.c;
  if(fusionCache?.key!==key)fusionCache={key,op:host.convAt(DATA,S.sample,"conv2",S.selection.out,coord.r,coord.c)};
  const op=fusionCache.op,weightKey=context+"/"+S.selection.out;
  if(document.body.dataset.explorerView!=="main"){
  if(weightsCache?.key!==weightKey){const start=S.selection.out*32*9,values=DATA.weights.conv2.weight.values.slice(start,start+32*9),max=Math.max(...values.map(Math.abs),1e-9);weightsCache={key:weightKey,atlas:atlas({shape:[1,32,3,3],values,min:-max,max},max,true)};$("mc-weights").width=weightsCache.atlas.full.width;$("mc-weights").height=weightsCache.atlas.full.height;$("mc-weights").dataset.weights=JSON.stringify(values);$("mc-weights").dataset.outputChannel=S.selection.out;}
  drawAtlas("mc-weights",weightsCache.atlas,{r:1,c:1},{input:true});
  }
  const c=$("mc-fusion");if(c.height!==430)c.height=430;const ctx=c.getContext("2d"),count=clamp(S.local-3,0,32),biasCount=S.local>=36?1:0,partial=op.contributions.slice(0,count).reduce((a,b)=>a+b,0)+biasCount*op.bias;
  ctx.fillStyle="#0a1322";ctx.fillRect(0,0,c.width,c.height);ctx.font="11px Consolas";
  for(let ch=0;ch<32;ch++){
   const y=17+ch*12,x=8,sx=205,active=ch<count,selected=ch===S.selection.input;
   ctx.fillStyle=selected?"#f6c76b":op.contributions[ch]<0?"#fb7185":"#a9bfd9";ctx.textAlign="left";ctx.fillText("c"+String(ch).padStart(2,"0")+"  "+(document.body.dataset.explorerView==="main"?"":host.fmt(op.contributions[ch],6)),x,y);
   ctx.strokeStyle=active?"#5ee1e8":"#31435d";ctx.globalAlpha=active?.65:.35;ctx.beginPath();ctx.moveTo(sx,y-4);ctx.bezierCurveTo(sx+130,y-4,630,210,755,210);ctx.stroke();ctx.globalAlpha=1;
   if(active&&host.playback().playing){const t=(now%900)/900,u=1-t,px=u*u*u*sx+3*u*u*t*(sx+130)+3*u*t*t*630+t*t*t*755,py=u*u*u*(y-4)+3*u*u*t*(y-4)+3*u*t*t*210+t*t*t*210;ctx.fillStyle="#5ee1e8";ctx.fillRect(px-2,py-2,4,4);}
  }
  ctx.fillStyle="#1c293f";ctx.strokeStyle="#5ee1e8";ctx.beginPath();ctx.arc(790,210,40,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle="#5ee1e8";ctx.font="22px Consolas";ctx.textAlign="center";ctx.fillText("Σ",790,218);
  ctx.font="12px Consolas";ctx.fillText("Σ "+count+"/32 = "+host.fmt(partial-biasCount*op.bias,7),790,275);ctx.fillStyle="#f6c76b";ctx.fillText("Bias × "+biasCount+" : "+host.fmt(op.bias,6),900,80);ctx.fillStyle="#34d399";ctx.fillText("→ "+host.fmt(partial,8),965,215);ctx.fillStyle="#8ea4bf";ctx.fillText("真实输出 "+host.fmt(op.expected,8),915,320);ctx.fillText("32路贡献均已计算；顺序高亮仅为教学",815,395);
  c.dataset.contributions=JSON.stringify(op.contributions);c.dataset.bias=op.bias;c.dataset.biasCount=biasCount;c.dataset.partial=partial;c.dataset.expected=op.expected;c.dataset.count=count;text("mc-fusion-summary","输出通道 "+S.selection.out+" · ["+coord.r+","+coord.c+"] · Σ32局部贡献 + 一次 Bias = "+host.fmt(op.raw,9)+" · PyTorch = "+host.fmt(op.expected,9));
 }
 function draw(now){
  if(!ensure())return;const play=host.playback(),coord=coordinate(play),kind=cache.kind,paired=cache.paired,sourceCoord={...coord};
  if(kind==="pool1"){sourceCoord.r*=2;sourceCoord.c*=2;if(sourceCoord.vr!==undefined){sourceCoord.vr*=2;sourceCoord.vc*=2;}}
  const poolStage=kind==="pool1"?(play.playing?root.DigitVisionPoolComparison.timeline(play.position%1).stage:JSON.parse($("pool-tournament")?.dataset.state||"{}").stage??0):undefined;
  if(paired)drawAtlas("mc-source",cache.source,sourceCoord,{...(kind==="pool1"?{poolStage}:{}),size:kind==="pool1"?2:kind==="conv2"?3:1,input:kind==="conv2"});
  const progressive=kind==="pool1"||kind==="conv1",total=kind==="pool1"?169:676,count=progressive?clamp(S.local,0,total):null;
  let preview=false;if(kind==="pool1"&&!play.playing){const view=JSON.parse($("pool-tournament")?.dataset.state||"{}");preview=view.stage===8&&view.phaseProgress>=.98;}
  drawAtlas("mc-target",cache.target,coord,{...(progressive?{prefix:count,side:kind==="pool1"?13:26}:{}),...(kind==="relu1"?{relu:clamp(play.position/100,0,1),reluSource:cache.source}:{}),preview});
  chart(responses(cache.target.tensor,coord.r,coord.c),coord);if(kind==="conv2")fusion(coord,now);
  $("mc-row").max=cache.target.tensor.shape[2]-1;$("mc-col").max=cache.target.tensor.shape[3]-1;if(document.activeElement!==$("mc-row"))$("mc-row").value=coord.r;if(document.activeElement!==$("mc-col"))$("mc-col").value=coord.c;
  text("mc-position","全部 "+cache.target.tensor.shape[1]+" 通道同步坐标 ["+coord.r+","+coord.c+"]");text("mc-status",play.playing?"PARALLEL · 同步回放":"ALIGNED · 同位置检查");panel.dataset.coordinate=JSON.stringify(coord);panel.dataset.layer=kind;panel.dataset.channels=cache.target.tensor.shape[1];
 }
 function canPaint(){return document.body.dataset.explorerView!=="parallel"?S.current.key==="conv2":!collapsed&&visible&&!panel.hidden;}
 function paint(now){raf=0;if(!canPaint())return;const playback=host.playback(),playing=playback.playing||playback.animating;
  if(now-lastPaint>=33||!playing){const start=performance.now();if(document.body.dataset.explorerView!=="parallel")fusion({r:S.selection.row,c:S.selection.col},now);else draw(now);metrics.paintMs.push(performance.now()-start);metrics.frames++;lastPaint=now;dirty=false;}
  if(playing||dirty)raf=requestAnimationFrame(paint);
 }
 function update(){dirty=true;const current=S.sample.index+"/"+S.current.key;if(lastContext!==current){lastContext=current;cache=null;locked=null;panel.hidden=!(S.sample.tensors[S.current.tensor]?.shape.length===4&&S.sample.tensors[S.current.tensor].shape[1]>1);}if(!raf&&canPaint())raf=requestAnimationFrame(paint);}
 function pick(event,kind){const a=kind==="source"?cache?.source:kind==="weight"?weightsCache?.atlas:cache?.target;if(!a)return;const rect=event.currentTarget.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*a.full.width,y=(event.clientY-rect.top)/rect.height*a.full.height,ch=Math.floor(y/a.th)*a.cols+Math.floor(x/a.tw);if(ch<a.tensor.shape[1]){host.selectChannel(ch,kind==="weight"||kind==="source"&&cache.kind==="conv2"?"input":"output");update();}}
 $("mc-source").addEventListener("click",e=>pick(e,"source"));$("mc-target").addEventListener("click",e=>pick(e,"target"));$("mc-weights").addEventListener("click",e=>pick(e,"weight"));$("mc-responses").addEventListener("click",e=>{const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*1100;host.selectChannel(clamp(Math.floor((x-50)/((1100-60)/cache.target.tensor.shape[1])),0,cache.target.tensor.shape[1]-1),"output");update();});
 for(const id of ["mc-row","mc-col"])$(id).addEventListener("change",()=>{const h=cache.target.tensor.shape[2],w=cache.target.tensor.shape[3],r=clamp(Math.trunc(Number($("mc-row").value))||0,0,h-1),c=clamp(Math.trunc(Number($("mc-col").value))||0,0,w-1);locked={r,c};host.selectPosition(r,c);update();});
 $("mc-collapse").addEventListener("click",()=>{collapsed=!collapsed;$("mc-body").hidden=collapsed;text("mc-collapse",collapsed?"展开总览":"折叠总览");if(!collapsed)update();});
 if(typeof IntersectionObserver!=="undefined")new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)update();},{rootMargin:"200px"}).observe(panel);
 window.addEventListener("resize",()=>{cache=null;update();});
 return {update,report:()=>({...metrics,paintMs:[...metrics.paintMs]}),snapshot:()=>({layer:cache?.kind,channels:cache?.target.tensor.shape[1],coordinate:panel.dataset.coordinate?JSON.parse(panel.dataset.coordinate):null,responseValues:$("mc-responses").dataset.values?JSON.parse($("mc-responses").dataset.values):[],revealed:cache?.target.painted??0})};
}
const API={responses,poolResponses,create};if(typeof module!=="undefined"&&module.exports)module.exports=API;root.DigitVisionMultiChannel=API;
})(typeof window!=="undefined"?window:globalThis);
