/* DigitVision Forward Pass Explorer: real recorded model data, offline rendering. */
(function(root){
"use strict";
const LAYERS=[
 {key:"input",label:"Input",tensor:"input",source:"input",op:"input",steps:6,title:"From pixels to a tensor",description:"真实 MNIST 像素逐行显现；ToTensor 将灰度转为 [0,1]，不做额外 Normalize。"},
 {key:"conv1",label:"Conv1",tensor:"conv1_pre",source:"input",op:"conv",steps:676,title:"从原始像素中提取局部特征",description:"不同卷积核对局部模式产生不同响应。完整扫描 26×26 个位置，逐点形成真实特征图。"},
 {key:"relu1",label:"ReLU1",tensor:"conv1_relu",source:"conv1_pre",op:"relu",steps:100,title:"保留正向响应，抑制负向响应",description:"ReLU 将负数截断为零，正数保持不变，为网络引入非线性。"},
 {key:"pool1",label:"Pool1",tensor:"pool1",source:"conv1_relu",op:"pool",steps:169,title:"保留局部最强响应",description:"每个 2×2 区域只保留最大响应，降低特征图的空间尺寸。"},
 {key:"conv2",label:"Conv2",tensor:"conv2_pre",source:"pool1",op:"conv",steps:121,title:"32 channels. One shared conclusion.",description:"32 个输入通道的 3×3 窗口同步扫描 121 个位置；每个输出 Filter 汇聚 32 路贡献及一次 bias。"},
 {key:"relu2",label:"ReLU2",tensor:"conv2_relu",source:"conv2_pre",op:"relu",steps:121,title:"Activate the deeper features",description:"64 个通道在相同空间位置独立执行 ReLU；负值归零，正值保持。"},
 {key:"pool2",label:"Pool2",tensor:"pool2",source:"conv2_relu",op:"pool",steps:25,title:"A smaller map, a stronger summary",description:"64 个通道的 2×2 窗口同步扫描 5×5 个输出位置，末行和末列不参与池化。"},
 {key:"flatten",label:"Flatten",tensor:"flatten",source:"pool2",op:"flatten",steps:64,title:"Unfold the tensor, preserve every value",description:"按通道、行、列的 NCHW 连续顺序展开。每一步将一个通道的 25 个元素送入向量。"},
 {key:"fc1",label:"FC1",tensor:"fc1_relu",source:"flatten",op:"dense",steps:34,title:"1600 features become 128 activations",description:"按 32 组、每组 50 个真实乘积展示完整加权和，再加入 bias 并执行 ReLU。"},
 {key:"output",label:"Output",tensor:"probabilities",source:"fc1_relu",op:"output",steps:12,title:"From evidence to a prediction",description:"128 个激活形成十个原始 logits；随后显示真实 Softmax 概率和最终预测。"}
];
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const sum=values=>values.reduce((a,b)=>a+b,0);
function at(t,ch,row,col){const h=t.shape[t.shape.length-2],w=t.shape[t.shape.length-1];return t.values[(ch*h+row)*w+col];}
function softmax(logits){const max=Math.max(...logits),exp=logits.map(v=>Math.exp(v-max)),total=sum(exp);return exp.map(v=>v/total);}
function flatCoordinate(index){return {channel:Math.floor(index/25),row:Math.floor(index%25/5),col:index%5};}
function near(a,b,tolerance={absolute:1e-4,relative:1e-5}){return Math.abs(a-b)<=tolerance.absolute+tolerance.relative*Math.abs(b);}
function convAt(bundle,sample,layer,out,row,col,inputChannel=0){
 const second=layer==="conv2",source=sample.tensors[second?"pool1":"input"],target=sample.tensors[second?"conv2_pre":"conv1_pre"],activated=sample.tensors[second?"conv2_relu":"conv1_relu"];
 const weights=bundle.weights[layer],inChannels=source.shape[1],outChannels=target.shape[1];
 if(out<0||out>=outChannels||row<0||row>=target.shape[2]||col<0||col>=target.shape[3]||inputChannel<0||inputChannel>=inChannels)throw new RangeError("Convolution selection out of range");
 const contributions=[],terms=[],patch=[],kernel=[];
 for(let c=0;c<inChannels;c++){
  let partial=0;
  for(let m=0;m<3;m++)for(let n=0;n<3;n++){
   const x=at(source,c,row+m,col+n),w=weights.weight.values[((out*inChannels+c)*3+m)*3+n],product=x*w;
   partial+=product;
   if(c===inputChannel){terms.push({x,w,product});patch.push(x);kernel.push(w);}
  }
  contributions.push(partial);
 }
 const bias=weights.bias.values[out],raw=sum(contributions)+bias;
 return {terms,patch,kernel,contributions,bias,raw,activated:Math.max(0,raw),
  expected:at(target,out,row,col),expectedActivated:at(activated,out,row,col),row,col,out,inputChannel,sourceKey:second?"pool1":"input"};
}
function poolAt(sample,layer,channel,row,col){
 const source=sample.tensors[layer==="pool1"?"conv1_relu":"conv2_relu"],target=sample.tensors[layer];
 if(channel<0||channel>=target.shape[1]||row<0||row>=target.shape[2]||col<0||col>=target.shape[3])throw new RangeError("Pooling selection out of range");
 const values=[];
 for(let m=0;m<2;m++)for(let n=0;n<2;n++)values.push(at(source,channel,row*2+m,col*2+n));
 return {values,raw:Math.max(...values),expected:at(target,channel,row,col),row,col,channel};
}
function denseAt(bundle,sample,layer,neuron,group=0){
 const first=layer==="fc1",source=sample.tensors[first?"flatten":"fc1_relu"].values,weights=bundle.weights[first?"fc1":"fc2"];
 const count=source.length,groupSize=first?50:8,groups=Math.ceil(count/groupSize),contributions=Array(groups).fill(0),terms=[];
 if(neuron<0||neuron>=weights.bias.values.length)throw new RangeError("Neuron out of range");
 for(let i=0;i<count;i++){
  const x=source[i],w=weights.weight.values[neuron*count+i],product=x*w,g=Math.floor(i/groupSize);
  contributions[g]+=product;if(g===group)terms.push({index:i,x,w,product});
 }
 const bias=weights.bias.values[neuron],raw=sum(contributions)+bias;
 return {contributions,terms,bias,raw,activated:Math.max(0,raw),expected:sample.tensors[first?"fc1_pre":"logits"].values[neuron],
  expectedActivated:first?sample.tensors.fc1_relu.values[neuron]:raw,count,groupSize,groups,neuron,group};
}
function validateBundle(bundle){
 if(!bundle||bundle.schemaVersion!==1||!Array.isArray(bundle.samples)||!bundle.samples.length)throw new Error("Missing real model export");
 const shapes={input:[1,1,28,28],conv1_pre:[1,32,26,26],conv1_relu:[1,32,26,26],pool1:[1,32,13,13],conv2_pre:[1,64,11,11],conv2_relu:[1,64,11,11],pool2:[1,64,5,5],flatten:[1,1600],fc1_pre:[1,128],fc1_relu:[1,128],dropout:[1,128],logits:[1,10],probabilities:[1,10]};
 for(const sample of bundle.samples)for(const [key,shape]of Object.entries(shapes)){
  const t=sample.tensors[key];
  if(!t||JSON.stringify(t.shape)!==JSON.stringify(shape)||t.values.length!==shape.reduce((a,b)=>a*b,1)||t.values.some(v=>!Number.isFinite(v)))throw new Error("Invalid real tensor: "+key);
 }
 return true;
}
/** Timer-independent dual-level state. Local progress is reversible, never inferred from frames. */
class ExplorerState{
 constructor(bundle){this.bundle=bundle;this.samplePosition=0;this.visited=new Set();this.restart();}
 get sample(){return this.bundle.samples[this.samplePosition];}
 get current(){return LAYERS[this.layer];}
 get limit(){return this.current.steps;}
 resetSelection(){
  const d=this.current,t=this.sample.tensors[d.tensor];
  this.selection={out:0,input:0,row:0,col:0,neuron:0,flatIndex:0,group:0};
  if(t.shape.length===4){
   const relevant=d.key==="conv1"?this.sample.tensors.conv1_relu:d.key==="conv2"?this.sample.tensors.conv2_relu:t;
   let best=0;for(let i=1;i<relevant.values.length;i++)if(relevant.values[i]>relevant.values[best])best=i;
   const h=t.shape[2],w=t.shape[3];this.selection.out=Math.floor(best/(h*w));this.selection.row=Math.floor(best%(h*w)/w);this.selection.col=best%w;
  }
  if(d.op==="dense")this.selection.neuron=this.sample.tensors.fc1_relu.values.indexOf(Math.max(...this.sample.tensors.fc1_relu.values));
  if(d.op==="output")this.selection.neuron=this.sample.prediction;
 }
 restart(){this.layer=0;this.local=0;this.globalStatus="IDLE";this.localStatus="IDLE";this.visited=new Set([0]);this.resetSelection();}
 pause(){if(this.globalStatus==="PLAYING")this.globalStatus="PAUSED";if(this.localStatus==="PLAYING")this.localStatus="PAUSED";}
 jump(index,automatic=false){
  if(index<0||index>=LAYERS.length)return;
  if(!automatic)this.pause();
  this.layer=index;this.local=0;this.localStatus=automatic?"PLAYING":"IDLE";
  this.globalStatus=automatic?"PLAYING":"PAUSED";this.visited.add(index);this.resetSelection();
 }
 switchSample(position){if(position<0||position>=this.bundle.samples.length)throw new RangeError("Unknown exported sample");this.samplePosition=position;this.restart();}
 advance(){
  if(this.local>=this.limit)return;
  this.local++;
  if(this.current.key==="conv2"&&this.local>=4)this.selection.input=clamp(this.local-4,0,31);
  if(this.current.op==="flatten")this.selection.flatIndex=(this.local-1)*25+this.selection.flatIndex%25;
  if(this.current.op==="dense")this.selection.group=clamp(this.local-1,0,31);
  if(this.current.op==="output")this.selection.neuron=this.local<=10?this.local-1:this.sample.prediction;
  if(this.local===this.limit)this.localStatus="COMPLETED";
 }
 previous(){this.pause();this.local=Math.max(0,this.local-1);this.localStatus=this.local?"PAUSED":"IDLE";
  if(this.current.key==="conv2")this.selection.input=clamp(this.local-4,0,31);
  if(this.current.op==="flatten")this.selection.flatIndex=Math.max(0,this.local-1)*25+this.selection.flatIndex%25;
  if(this.current.op==="dense")this.selection.group=clamp(this.local-1,0,31);
 }
}
function representativeConv1(bundle,sample,channel){
 const raw=sample.tensors.conv1_pre,act=sample.tensors.conv1_relu,input=sample.tensors.input,candidates=[];
 for(let i=0;i<676;i++){
  const r=Math.floor(i/26),c=i%26;let energy=0;
  for(let m=0;m<3;m++)for(let n=0;n<3;n++)energy+=at(input,0,r+m,c+n);
  if(energy>0)candidates.push({index:i,energy,activation:at(act,channel,r,c),raw:at(raw,channel,r,c)});
 }
 const hasPositive=candidates.some(v=>v.activation>0);
 let best={index:0,energy:0,score:-Infinity};
 for(const v of candidates){const score=hasPositive?v.activation:Math.abs(v.raw);if(score>best.score||(score===best.score&&v.energy>best.energy))best={index:v.index,energy:v.energy,score};}
 return {...best,criterion:hasPositive?"maximum real ReLU response overlapping ink":"maximum absolute raw response overlapping ink"};
}
/** Display duration only; deliberately unrelated to model execution time. */
function conv1Duration(mode="fast",speed=1){
 if(mode==="step")return null;
 return mode==="detailed"?clamp(20000/speed,10000,20000):clamp(4000/speed,2000,4000);
}
/** Continuous serpentine display path; recorded tensor coordinates are never reordered. */
function conv1ScanPoint(position){
 const p=clamp(position,0,675),i=Math.floor(p),fraction=p-i;
 const coordinate=n=>{const row=Math.floor(n/26),column=n%26;return {row,col:row%2?25-column:column};};
 const a=coordinate(i),b=coordinate(Math.min(i+1,675));
 return {row:a.row+(b.row-a.row)*fraction,col:a.col+(b.col-a.col)*fraction,index:a.row*26+a.col,step:i,fraction};
}
/** Shared timestamp clock: renderers never wait per tensor position. */
function advanceVisualClock(visual,now,limit,duration){
 const dt=visual.last===null?null:Math.max(0,now-visual.last);
 const advance=dt===null?0:dt*limit/duration;
 visual.position=Math.min(limit,visual.position+advance);if(limit-visual.position<1e-8)visual.position=limit;visual.last=now;visual.fraction=visual.position%1;
 return {dt,advance};
}
function createFrameMetrics(mode,targetMs,startPosition){return {mode,targetMs,startPosition,activeMs:0,frames:0,intervals:[],renderMs:[],completed:false,domBuilds:0,detailUpdates:0,revealedPositions:0,maxMotionExcess:0};}
const API={advanceVisualClock,conv1ScanPoint,conv1Duration,representativeConv1,LAYERS,ExplorerState,at,convAt,poolAt,denseAt,softmax,flatCoordinate,near,validateBundle};
if(typeof module!=="undefined"&&module.exports)module.exports=API;
if(typeof document==="undefined")return;
root.DigitVisionForwardMath=API;

const $=id=>document.getElementById(id);
const DATA=root.DIGITVISION_DATA;
try{validateBundle(DATA);}catch(error){$("app-error").hidden=false;$("app-error").textContent="无法加载真实模型数据："+error.message;return;}
const S=new ExplorerState(DATA);
let informationView=null;
let clock=0,lastView=null,sourceView=null,targetView=null,lastWriteAnimation="";
let convRAF=0;const convVisual={fraction:0,last:null,key:"",manualIndex:null,mode:"fast",metrics:null,mapKey:"",representative:null,position:0,dom:null,painted:0,buffer:null,controlKey:"",detailKey:"",staticKey:""};
const layerAnim=root.DigitVisionUnifiedSpatial.create({$,S,DATA,fmt:(...args)=>fmt(...args),near,convAt,poolAt,mapFor,drawMap,drawBank,drawContributions,color,advanceClock:advanceVisualClock,metrics:createFrameMetrics,duration:conv1Duration,queueFrame:queueSpatialFrame,complete:()=>{if(S.globalStatus==="PLAYING")clock=setTimeout(tick,0);},pause:pauseAll,updateControls,onRender:()=>{publishSpatial();multiChannel.update();informationView?.update();},setViews:(source,target,view)=>{sourceView=source;targetView=target;lastView=view;},setWindow:region=>{const el=$("receptive-window");el.hidden=!region;if(!region)return;el.style.left=el.style.top="0";el.style.width=el.style.height=region.size/26*100+"%";el.style.transition="none";el.style.transform="translate3d("+region.col/region.size*100+"%,"+region.row/region.size*100+"%,0)";}});
function isStepMode(){return S.current.key==="conv1"?convVisual.mode==="step":layerAnim.handles(S.current.key)&&layerAnim.state.mode==="step";}
function queueSpatialFrame(callback){if(new URLSearchParams(location.search).get("selftest")!=="1")convRAF=requestAnimationFrame(callback);}
const multiChannel=root.DigitVisionMultiChannel.create({S,DATA,fmt:(...args)=>fmt(...args),convAt,conv1ScanPoint,playback:()=>({playing:S.globalStatus==="PLAYING"||S.localStatus==="PLAYING",animating:layerAnim.handles(S.current.key)&&(layerAnim.state.inspectPlaying||layerAnim.state.transferring),position:S.current.key==="conv1"?convVisual.position:layerAnim.handles(S.current.key)?layerAnim.state.position:S.local}),selectPosition:(row,col)=>{pauseAll();S.selection.row=row;S.selection.col=col;selectionChanged();},selectChannel:(channel,kind)=>{if(kind==="input"){pauseAll();S.selection.input=channel;}else{S.selection.out=channel;if(layerAnim.handles(S.current.key))layerAnim.parallelChannel();}renderAll();}});
informationView=root.DigitVisionViews.create({S,invalidate:()=>layerAnim.invalidateGeometry(),refreshParallel:()=>multiChannel.update(),render:()=>renderDynamic()});
const fmt=(value,digits=5)=>{if(value===0||Object.is(value,-0))return "0";return Number(value.toPrecision(digits)).toString().replace("-","−");};
const shape=t=>"["+t.shape.join(", ")+"]";
function stopClock(){clearTimeout(clock);clock=0;cancelAnimationFrame(convRAF);convRAF=0;convVisual.last=null;layerAnim.state.last=null;}
function pauseAll(){stopClock();layerAnim.pauseTransfer();S.pause();if(S.current.key==="conv1")renderConv1();else if(layerAnim.handles(S.current.key))layerAnim.render();else updateControls();}
function schedule(){
 stopClock();
 if(layerAnim.handles(S.current.key)){if(isStepMode()){S.pause();layerAnim.render();}else if(S.globalStatus==="PLAYING"||S.localStatus==="PLAYING")queueSpatialFrame(layerAnim.frame);return;}
 if(isStepMode()){S.pause();renderConv1();return;}
 if(S.current.key==="conv1"&&(S.globalStatus==="PLAYING"||S.localStatus==="PLAYING")){queueSpatialFrame(conv1Frame);return;}
 if(S.globalStatus==="PLAYING"||S.localStatus==="PLAYING"){
  const delay=S.globalStatus==="PLAYING"?220/Number($("global-speed").value):360/Number($("local-speed").value);
  clock=setTimeout(tick,delay);
 }
}
function tick(){
 clock=0;
 if(S.globalStatus!=="PLAYING"&&S.localStatus!=="PLAYING")return;
 try{
  if(S.local<S.limit){S.advance();renderDynamic();}
  else if(S.globalStatus==="PLAYING"){
   if(S.layer===LAYERS.length-1){S.globalStatus="COMPLETED";S.localStatus="COMPLETED";updateControls();return;}
   S.jump(S.layer+1,true);renderAll();
  }else{S.localStatus="COMPLETED";updateControls();return;}
  schedule();
 }catch(error){pauseAll();$("app-error").hidden=false;$("app-error").textContent=error.message;}
}
function globalPlay(){if(layerAnim.handles(S.current.key)&&layerAnim.resumeTransfer())return;if(layerAnim.handles(S.current.key)){if(isStepMode()){pauseAll();return;}layerAnim.resume();}if(isStepMode()){pauseAll();return;}if(S.current.key==="conv1"){convVisual.manualIndex=null;convVisual.key="";}if(S.globalStatus==="COMPLETED")return;stopClock();S.globalStatus="PLAYING";if(S.local<S.limit)S.localStatus="PLAYING";updateControls();schedule();}
function localPlay(){if(layerAnim.handles(S.current.key)&&layerAnim.resumeTransfer())return;if(layerAnim.handles(S.current.key)){if(isStepMode()){pauseAll();return;}layerAnim.resume();}if(isStepMode()){pauseAll();return;}if(S.current.key==="conv1"){convVisual.manualIndex=null;convVisual.key="";}pauseAll();if(S.local>=S.limit&&layerAnim.state.direction!==-1)return;S.globalStatus="PAUSED";S.localStatus="PLAYING";updateControls();schedule();}
function jump(index){globalEngine?.pause();layerAnim.reset(LAYERS[index]?.key);convVisual.position=0;convVisual.metrics=null;convVisual.manualIndex=null;convVisual.fraction=0;convVisual.key="";stopClock();S.jump(index);renderAll();if(new URLSearchParams(location.search).get("selftest")!=="1")globalEngine?.jumpUi(LAYERS[index].key);}
function localNext(){if(layerAnim.handles(S.current.key)){pauseAll();layerAnim.step(1);return;}convVisual.manualIndex=null;convVisual.fraction=0;convVisual.key="";pauseAll();if(S.local<S.limit){S.localStatus="PAUSED";S.advance();renderDynamic();}}
function localPrevious(){if(layerAnim.handles(S.current.key)){pauseAll();layerAnim.step(-1);return;}convVisual.manualIndex=null;convVisual.fraction=0;convVisual.key="";pauseAll();S.previous();renderDynamic();}
function localRestart(){if(layerAnim.handles(S.current.key)){pauseAll();layerAnim.restart();return;}convVisual.position=0;convVisual.metrics=null;convVisual.manualIndex=null;convVisual.fraction=0;convVisual.key="";pauseAll();S.local=0;S.localStatus="IDLE";renderAll();}
function globalRestart(){globalEngine?.restart();layerAnim.reset("input");convVisual.position=0;convVisual.metrics=null;convVisual.manualIndex=null;convVisual.fraction=0;convVisual.key="";stopClock();S.restart();renderAll();}
function selectionChanged(){
 if(globalEngine?.snapshot().state==="PLAYING")globalEngine.inspect();stopClock();S.pause();if(layerAnim.handles(S.current.key)){layerAnim.selected();return;}convVisual.key="";
 const isConv1=S.current.key==="conv1",changedChannel=isConv1&&convVisual.channel!==S.selection.out;
 const selected=S.selection.row*26+S.selection.col;
 if(isConv1){
  convVisual.manualIndex=changedChannel?null:selected;
  if(changedChannel){S.local=0;convVisual.position=0;convVisual.fraction=0;convVisual.metrics=null;}
 }else S.local=0;
 convVisual.channel=S.selection.out;S.localStatus=S.local===S.limit?"COMPLETED":S.local?"PAUSED":"IDLE";renderAll();
}
function mapFor(key,channel=0){
 const t=S.sample.tensors[key];
 if(t.shape.length===4){const h=t.shape[2],w=t.shape[3];return {values:t.values.slice(channel*h*w,(channel+1)*h*w),h,w,min:t.min,max:t.max,key,channel};}
 const count=t.values.length,w=count===1600?40:count===128?16:5,h=Math.ceil(count/w);
 return {values:t.values,h,w,min:t.min,max:t.max,key,channel:0};
}
function color(value,range,palette){
 const level=clamp(Math.abs(value)/Math.max(range,1e-9),0,1),rgb=value<0?[251,113,133]:palette==="input"?[77,163,255]:palette==="output"?[52,211,153]:[167,139,250];
 return "rgba("+rgb.join(",")+","+(0.025+level*.94)+")";
}
function drawPlane(ctx,map,x,y,width,height,options={}){
 const cellW=width/map.w,cellH=height/map.h,range=Math.max(Math.abs(map.min),Math.abs(map.max),1e-9);
 for(let r=0;r<map.h;r++)for(let c=0;c<map.w;c++){
  const index=r*map.w+c,value=map.values[index]||0;
  ctx.globalAlpha=(options.reveal !== undefined && index >= options.reveal) ? 0.14 : 1;
  if(options.ghost)ctx.globalAlpha=index===options.activeIndex?(options.written?1:0.12):0.42;
  ctx.fillStyle=color(value,range,options.palette||"feature");
  const gap=map.w<=40?Math.min(.6,cellW*.07):0;
  ctx.fillRect(x+c*cellW+gap,y+r*cellH+gap,Math.max(.1,cellW-gap*2),Math.max(.1,cellH-gap*2));
 }
 ctx.globalAlpha=1;
}
function drawMap(canvas,map,options={}){
 const ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#080f1c";ctx.fillRect(0,0,canvas.width,canvas.height);
 drawPlane(ctx,map,0,0,canvas.width,canvas.height,options);
 if(options.highlight){
  const {row,col,size=1}=options.highlight,cw=canvas.width/map.w,ch=canvas.height/map.h;
  ctx.strokeStyle=options.highlightColor||"#f6c76b";ctx.lineWidth=2.5;ctx.strokeRect(col*cw+1,row*ch+1,size*cw-2,size*ch-2);
 }
}
function setWindow(map,region){
 const overlay=$("receptive-window");
 overlay.hidden=!region;
 if(!region)return;
 overlay.style.left=(region.col/map.w*100)+"%";overlay.style.top=(region.row/map.h*100)+"%";
 overlay.style.width=(region.size/map.w*100)+"%";overlay.style.height=(region.size/map.h*100)+"%";
}
function effectiveCoordinates(){
 const q=S.selection;
 if(S.current.op==="conv"&&S.local>=1&&S.local<=3)return {row:q.row,col:Math.max(0,q.col-(3-S.local))};
 return {row:q.row,col:q.col};
}
function operation(){
 const d=S.current,q=S.selection,p=effectiveCoordinates();
 if(d.op==="conv")return convAt(DATA,S.sample,d.key,q.out,p.row,p.col,d.key==="conv2"?q.input:0);
 if(d.op==="pool")return poolAt(S.sample,d.key,q.out,q.row,q.col);
 if(d.op==="dense"||d.op==="output")return denseAt(DATA,S.sample,d.key,q.neuron,q.group);
 if(d.op==="relu")return {raw:at(S.sample.tensors[d.source],q.out,q.row,q.col),expected:at(S.sample.tensors[d.tensor],q.out,q.row,q.col)};
 if(d.op==="flatten"){const coord=flatCoordinate(q.flatIndex);return {...coord,raw:S.sample.tensors.flatten.values[q.flatIndex],expected:at(S.sample.tensors.pool2,coord.channel,coord.row,coord.col)};}
 const raw=at(S.sample.tensors.input,0,q.row,q.col);return {raw,expected:raw};
}
function matrixMarkup(values,kind="patch",active=-1,four=false){
 return '<div class="tiny-matrix '+(four?"four":"")+'">'+values.map((value,index)=>'<div title="'+value+'" class="tiny-cell '+(kind==="weight"?"weight ":kind==="product"?"product ":"")+(value<0?"negative ":"")+(index===active?"selected ":"")+'">'+fmt(value,4)+'</div>').join("")+'</div>';
}
function drawContributions(values,active,completed){
 const canvas=$("contributions"),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height,span=w/values.length,max=Math.max(...values.map(Math.abs),1e-9);
 ctx.clearRect(0,0,w,h);ctx.fillStyle="#0a1423";ctx.fillRect(0,0,w,h);
 const baseline=h*.53;ctx.strokeStyle="#39506b";ctx.beginPath();ctx.moveTo(8,baseline);ctx.lineTo(w-8,baseline);ctx.stroke();
 for(let i=0;i<values.length;i++){
  const value=values[i],height=Math.abs(value)/max*(h*.34),x=i*span+span*.17,y=value>=0?baseline-height:baseline;
  ctx.globalAlpha=completed===0?0.48:(i<completed?1:.24);ctx.fillStyle=value<0?"#fb7185":"#5ee1e8";
  ctx.fillRect(x,y,span*.66,Math.max(1,height));ctx.globalAlpha=1;
  if(i===active){ctx.strokeStyle="#f6c76b";ctx.lineWidth=2;ctx.strokeRect(x-2,Math.min(y,baseline)-3,span*.66+4,Math.max(6,height+6));}
  if(i%4===0||i===values.length-1){ctx.fillStyle=i===active?"#f6c76b":"#7188a6";ctx.font="11px Consolas";ctx.textAlign="center";ctx.fillText(i,x+span*.33,h-7);}
 }
 ctx.fillStyle="#8da4c1";ctx.font="11px Segoe UI";ctx.textAlign="left";ctx.fillText("signed contribution",8,14);
}
function drawBank(canvas,key,selected,weights=false){
 const ctx=canvas.getContext("2d"),cols=8,rows=4,cw=canvas.width/cols,ch=canvas.height/rows;
 ctx.fillStyle="#0a1322";ctx.fillRect(0,0,canvas.width,canvas.height);
 let weightRange=1;
 if(weights){const w=DATA.weights.conv2.weight.values,start=S.selection.out*32*9;weightRange=Math.max(...w.slice(start,start+32*9).map(Math.abs),1e-9);}
 for(let c=0;c<32;c++){
  const x=(c%cols)*cw,y=Math.floor(c/cols)*ch;
  let map;
  if(weights){const start=(S.selection.out*32+c)*9;map={values:DATA.weights.conv2.weight.values.slice(start,start+9),h:3,w:3,min:-weightRange,max:weightRange};}
  else map=mapFor(key,c);
  drawPlane(ctx,map,x+4,y+3,cw-8,ch-17);
  if(c===selected){ctx.strokeStyle="#f6c76b";ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,cw-2,ch-2);}
  ctx.fillStyle=c===selected?"#f6c76b":"#7e92af";ctx.font="10px Consolas";ctx.textAlign="center";ctx.fillText("c"+c,x+cw/2,y+ch-3);
 }
}
function denseSVG(values,selected){
 const max=Math.max(...values.map(Math.abs),1e-9);
 return '<svg class="dense-svg" viewBox="0 0 320 180" aria-label="32 grouped real weighted contributions">'+values.map((v,i)=>{
  const y=12+i*4.8,color=i===selected?"#f6c76b":v<0?"#fb7185":"#a78bfa",opacity=i===selected?1:.15+.5*Math.abs(v)/max;
  return '<path d="M 24 '+y+' C 145 '+y+', 170 90, 260 90" fill="none" stroke="'+color+'" opacity="'+opacity+'" stroke-width="'+(i===selected?2:1)+'"><title>Group '+i+': '+v+'</title></path><circle cx="24" cy="'+y+'" r="2" fill="'+color+'"/>';
 }).join("")+'<circle cx="270" cy="90" r="22" fill="#292342" stroke="#a78bfa"/><text x="270" y="95" text-anchor="middle" fill="#dfd3ff" font-family="Consolas" font-size="13">'+S.selection.neuron+'</text></svg>';
}
function fillOptions(id,count,value,prefix=""){
 const select=$(id);select.replaceChildren();
 for(let i=0;i<count;i++){const option=document.createElement("option");option.value=i;option.textContent=prefix+i;select.append(option);}
 select.value=value;
}
function stripTensor(){if(S.current.op==="flatten")return "pool2";const t=S.sample.tensors[S.current.tensor];return t.shape.length===4&&t.shape[1]>1?S.current.tensor:null;}
function renderStrip(){
 const key=stripTensor(),wrap=$("channel-strip-wrap"),strip=$("channel-strip");
 wrap.hidden=!key;strip.replaceChildren();if(!key)return;
 const count=S.sample.tensors[key].shape[1];
 $("channel-strip-title").textContent=count+" REAL "+(S.current.op==="flatten"?"SOURCE":"OUTPUT")+" CHANNELS";
 for(let c=0;c<count;c++){
  const button=document.createElement("button");button.className="channel-thumb";button.dataset.channel=c;
  button.setAttribute("aria-label","Inspect output channel "+c);
  const canvas=document.createElement("canvas");canvas.width=56;canvas.height=56;
  const label=document.createElement("span");label.textContent=String(c).padStart(2,"0");
  button.append(canvas,label);drawMap(canvas,mapFor(key,c));strip.append(button);
  button.addEventListener("click",()=>{if(S.current.op==="flatten")S.selection.flatIndex=c*25+S.selection.flatIndex%25;else S.selection.out=c;selectionChanged();});
 }
}
function buildPipeline(){
 $("pipeline").replaceChildren();
 for(let i=0;i<LAYERS.length;i++){
  const d=LAYERS[i],button=document.createElement("button");button.className="layer-node";button.dataset.layer=d.key;
  button.innerHTML='<span class="node-title">'+d.label+'<span class="node-order">'+String(i+1).padStart(2,"0")+'</span></span><span class="node-shape">'+S.sample.tensors[d.tensor].shape.join("×")+'</span>';
  button.setAttribute("aria-label","Go to "+d.label);button.addEventListener("click",()=>jump(i));$("pipeline").append(button);
 }
}
function updateControls(){
 publishSpatial();
 document.body.classList.toggle("global-running",S.globalStatus==="PLAYING");
 document.body.classList.toggle("local-running",S.localStatus==="PLAYING");
 $("global-status").textContent=S.globalStatus;$("global-status").dataset.state=S.globalStatus;
 $("local-status").textContent=S.localStatus;$("local-status").dataset.state=S.localStatus;
 $("global-play").disabled=S.globalStatus==="PLAYING"||S.globalStatus==="COMPLETED"||isStepMode();
 $("global-pause").disabled=S.globalStatus!=="PLAYING";
 $("global-previous").disabled=S.layer===0;$("global-next").disabled=S.layer===LAYERS.length-1;
 $("local-play").disabled=S.localStatus==="PLAYING"||(S.local===S.limit&&layerAnim.state.direction!==-1)||isStepMode();
 $("local-pause").disabled=S.localStatus!=="PLAYING";$("local-previous").disabled=S.local===0;$("local-next").disabled=S.local===S.limit;
 $("local-position").textContent="STEP "+S.local+" / "+S.limit;$("local-progress").style.width=(S.local/S.limit*100)+"%";
 $("network-progress").textContent=S.current.label+" / "+LAYERS.length+" layers";
 [...$("pipeline").children].forEach((button,i)=>{button.classList.toggle("active",i===S.layer);button.classList.toggle("done",i<S.layer);button.setAttribute("aria-current",i===S.layer?"step":"false");});
}
function renderAll(){multiChannel.update();const spatial=layerAnim.handles(S.current.key);$("layer-animation-controls").hidden=!spatial;$("conv1-input-detail").hidden=S.current.key!=="conv1";document.body.classList.toggle("relu1-view",S.current.key==="relu1");document.body.classList.toggle("pool1-view",S.current.key==="pool1");if(!spatial&&$("pool-flight"))$("pool-flight").hidden=true;layerAnim.invalidateGeometry();$("conv1-mode-controls").hidden=S.current.key!=="conv1";document.body.classList.toggle("conv1-view",S.current.key==="conv1");if(S.current.key==="conv1"){convVisual.channel=S.selection.out;convVisual.key="";}
 const d=S.current,q=S.selection,t=S.sample.tensors[d.tensor],isMap=t.shape.length===4;
 $("sample").value=S.samplePosition;$("true-label").textContent=S.sample.groundTruth;$("predicted-label").textContent=S.sample.prediction;
 drawMap($("sample-preview"),mapFor("input"),{palette:"input"});
 $("stage-category").textContent=String(S.layer+1).padStart(2,"0")+" / "+d.label.toUpperCase();
 $("stage-title").textContent=d.title;$("stage-description").textContent=d.description;
 $("stage-input-shape").textContent=shape(S.sample.tensors[d.source]);$("stage-output-shape").textContent=shape(t);
 $("channel-control").hidden=!isMap||t.shape[1]===1;
 if(isMap)fillOptions("channel",t.shape[1],q.out,"ch ");
 const hasInput=d.key==="conv2"||d.op==="dense"||d.op==="output";
 $("input-channel-control").hidden=!hasInput;
 $("input-channel-control").childNodes[0].textContent=d.key==="conv2"?"Input channel ":"Input group ";
 if(hasInput)fillOptions("input-channel",d.key==="conv2"?32:d.op==="dense"?32:16,d.key==="conv2"?q.input:q.group,"");
 $("row-control").hidden=$("col-control").hidden=!isMap;
 if(isMap){$("row").max=t.shape[2]-1;$("col").max=t.shape[3]-1;}
 $("neuron-control").hidden=!(d.op==="dense"||d.op==="output");$("neuron").max=d.op==="dense"?127:9;
 $("flat-control").hidden=d.op!=="flatten";
 $("input-bank").hidden=d.key!=="conv2";$("kernel-bank").hidden=d.key!=="conv2";
 $("contributions").hidden=!(d.key==="conv2"||d.op==="dense"||d.op==="output");
 $("target-canvas").hidden=d.op==="output";$("probability-chart").hidden=d.op!=="output";
 renderStrip();if(d.key==="conv1")$("channel-strip").querySelectorAll("button").forEach(button=>button.classList.toggle("active",Number(button.dataset.channel)===q.out));renderDynamic();
 const active=$("pipeline").children[S.layer],bar=$("pipeline");
 if(active.offsetLeft<bar.scrollLeft||active.offsetLeft+active.offsetWidth>bar.scrollLeft+bar.clientWidth)bar.scrollLeft=Math.max(0,active.offsetLeft-bar.clientWidth/2);
}


/** Only this clock owns the continuous visual cursor. S.local is a reveal count, not a cursor. */
function conv1Frame(now){
 convRAF=0;
 if(S.current.key!=="conv1"||(S.globalStatus!=="PLAYING"&&S.localStatus!=="PLAYING"))return;
 if(convVisual.mode==="step"){pauseAll();return;}
 const speed=Number($(S.globalStatus==="PLAYING"?"global-speed":"local-speed").value);
 const duration=conv1Duration(convVisual.mode,speed);
 if(!convVisual.metrics)convVisual.metrics=createFrameMetrics(convVisual.mode,duration,convVisual.position);
 const metrics=convVisual.metrics,previous=conv1ScanPoint(convVisual.position),delta=advanceVisualClock(convVisual,now,676,duration);
 if(delta.dt!==null){const current=conv1ScanPoint(convVisual.position);metrics.maxMotionExcess=Math.max(metrics.maxMotionExcess,Math.hypot(current.row-previous.row,current.col-previous.col)-delta.advance);metrics.activeMs+=delta.dt;metrics.intervals.push(delta.dt);}
 S.local=Math.min(676,Math.floor(convVisual.position+1e-8));
 if(S.local===676){S.localStatus="COMPLETED";convVisual.fraction=0;}
 const start=performance.now();renderConv1();metrics.renderMs.push(performance.now()-start);metrics.frames++;
 if(S.local===676){metrics.completed=true;convVisual.last=null;if(S.globalStatus==="PLAYING")clock=setTimeout(tick,0);return;}
 queueSpatialFrame(conv1Frame);
}
/** Build once per layer visit. All animated content below is updated in place. */
function conv1DOM(){
 if(convVisual.dom&&$("operation-content").contains(convVisual.dom.kernel))return convVisual.dom;
 $("operation-content").innerHTML='<div class="conv-kernel">'+matrixMarkup(Array(9).fill(0),"weight")+'</div><div class="conv-link"><i></i><span>输入感受野 → 当前输出响应</span></div><div class="conv-explanation"><strong></strong><p class="explanation-summary"></p><span></span><p class="explanation-result"></p></div>';
 $("conv1-input-detail").innerHTML=["真实 3×3 输入","真实权重","逐元素乘积"].map((name,i)=>'<div><span>'+name+'</span>'+matrixMarkup(Array(9).fill(0),i===1?"weight":i===2?"product":"patch")+'</div>').join("");
 $("term-grid").replaceChildren(...Array.from({length:9},()=>{const el=document.createElement("div");el.className="term revealed";return el;}));
 let legend=$("conv1-color-legend");if(!legend){legend=document.createElement("div");legend.id="conv1-color-legend";$("target-caption").after(legend);}
 legend.innerHTML='<div class="response-scale"></div><div class="response-labels"><span>负响应</span><span>接近零</span><span>正响应</span></div><small></small>';
 const content=$("operation-content"),kernel=content.querySelector(".conv-kernel");
 convVisual.dom={kernel,kernelCells:[...kernel.querySelectorAll(".tiny-cell")],detailCells:[...$("conv1-input-detail").querySelectorAll(".tiny-cell")],terms:[...$("term-grid").children],title:content.querySelector(".conv-explanation strong"),summary:content.querySelector(".explanation-summary"),coordinate:content.querySelector(".conv-explanation span"),result:content.querySelector(".explanation-result"),dot:content.querySelector(".conv-link i"),legend:legend.querySelector("small")};
 convVisual.staticKey="";convVisual.detailKey="";convVisual.controlKey="";convVisual.activeWeight=null;
 if(convVisual.metrics)convVisual.metrics.domBuilds++;
 return convVisual.dom;
}
function conv1Text(id,value){const el=typeof id==="string"?$(id):id;if(el.textContent!==value)el.textContent=value;}
function renderConv1(){
 publishSpatial();
 informationView?.update();
 multiChannel.update();
 document.body.classList.add("conv1-view");
 const q=S.selection,playing=S.globalStatus==="PLAYING"||S.localStatus==="PLAYING";
 // Synchronize explicit previous/next/reset and deterministic test state, never round the visual cursor each frame.
 if(Math.floor(convVisual.position+1e-8)!==S.local)convVisual.position=S.local+convVisual.fraction;
 if(S.local===676)convVisual.position=676;
 const mapKey=S.sample.index+"/"+q.out;
 if(convVisual.mapKey!==mapKey){
  convVisual.mapKey=mapKey;convVisual.representative=representativeConv1(DATA,S.sample,q.out);convVisual.source=mapFor("input",0);convVisual.target=mapFor("conv1_pre",q.out);
  convVisual.buffer=document.createElement("canvas");convVisual.buffer.width=convVisual.buffer.height=360;convVisual.painted=0;convVisual.bufferKey="";
 }
 const point=conv1ScanPoint(convVisual.position),representative=convVisual.representative;
 const resting=S.local===676||(S.local===0&&!playing&&convVisual.position===0);
 const index=convVisual.manualIndex!==null&&!playing?convVisual.manualIndex:resting?representative.index:point.index;
 const r=Math.floor(index/26),c=index%26,overview=playing&&convVisual.mode==="fast";
 q.row=r;q.col=c;sourceView=convVisual.source;targetView=convVisual.target;
 lastView={source:sourceView,target:targetView,sourceHighlight:{row:r,col:c,size:3},targetHighlight:{row:r,col:c}};
 document.body.classList.toggle("conv1-overview",overview);
 const dom=conv1DOM(),speed=Number($(S.globalStatus==="PLAYING"?"global-speed":"local-speed").value),duration=conv1Duration(convVisual.mode,speed);
 conv1Text("conv1-duration",duration===null?"手动单步，无自动播放":(duration/1000).toFixed(1)+" s / 完整扫描 · 仅展示时长");
 conv1Text("conv1-inspection-status",playing?(overview?"Fast Overview · 连续扫描":"Detailed · 逐位置检查"):"Detailed Inspection · 当前真实计算");
 const staticKey=mapKey+"/"+overview+"/"+(S.local===676?index:"scan");
 if(convVisual.staticKey!==staticKey){
  convVisual.staticKey=staticKey;
  drawMap($("source-canvas"),sourceView,{palette:"input",highlight:S.local===676?{row:r,col:c,size:3}:null});
  conv1Text("source-title","真实 MNIST 输入 · 28 × 28");conv1Text("target-title","真实特征图 · 26 × 26");
  conv1Text("source-caption","青色感受野沿连续蛇形路径移动；蓝色亮度为真实像素。");
  conv1Text("operation-label","真实训练权重 · 通道 "+q.out);
  conv1Text("operation-note","32 个输出通道可并行计算；这里只展示所选通道。动画不是推理计时。暂停或点击任意位置检查，检查焦点不会改变播放进度。");
  conv1Text("local-heading","当前窗口的九项乘法与一次 bias");
  conv1Text(dom.title,overview?"Fast Overview":"局部模式 → 特征响应");
  conv1Text(dom.summary,overview?"连续扫描全部 676 个位置；暂停查看完整乘加。":"九个乘积求和，只加入一次 bias。");
  const kernel=convAt(DATA,S.sample,"conv1",q.out,0,0).kernel,max=Math.max(...kernel.map(Math.abs),1e-9);
  dom.kernelCells.forEach((el,i)=>{conv1Text(el,fmt(kernel[i],4));el.title="真实权重 "+kernel[i];el.dataset.sign=Math.abs(kernel[i])<=max*.08?"near-zero":kernel[i]<0?"negative":"positive";});
  const range=Math.max(Math.abs(targetView.min),Math.abs(targetView.max),1e-9);
  conv1Text(dom.legend,"对称显示色阶："+fmt(-range,4)+" / 0 / "+fmt(range,4)+" · 只映射颜色，不改变数值");
 }
 const detailKey=mapKey+"/"+index+"/"+overview;
 if(!overview&&convVisual.detailKey!==detailKey){
  convVisual.detailKey=detailKey;const op=convAt(DATA,S.sample,"conv1",q.out,r,c);
  if(convVisual.metrics)convVisual.metrics.detailUpdates++;
  [op.patch,op.kernel,op.terms.map(v=>v.product)].flat().forEach((v,i)=>{conv1Text(dom.detailCells[i],fmt(v,4));dom.detailCells[i].title=String(v);dom.detailCells[i].classList.toggle("negative",v<0);});
  op.terms.forEach((term,i)=>{conv1Text(dom.terms[i],fmt(term.x,5)+" × "+fmt(term.w,5)+" = "+fmt(term.product,5));dom.terms[i].title=JSON.stringify(term);});
  conv1Text("calculation-formula","Σ9 products = "+fmt(sum(op.terms.map(v=>v.product)),9)+"; bias = "+fmt(op.bias,9)+"; z = "+fmt(op.raw,10)+"（互相关，不翻转卷积核）");
  conv1Text("result-label","真实 PyTorch 原始响应");conv1Text("result-value",fmt(op.expected,9));$("result-value").classList.toggle("negative",op.expected<0);
  conv1Text("result-detail","ReLU 预览 "+fmt(op.expectedActivated,9)+"；这是下一节点。");
  conv1Text("consistency","✓ |Δ| "+Math.abs(op.raw-op.expected).toExponential(3)+"；容差 abs 1e−4 + rel 1e−5");$("consistency").dataset.passed=String(near(op.raw,op.expected));
  conv1Text(dom.coordinate,"当前位置 ["+r+", "+c+"]");conv1Text(dom.result,"z = "+fmt(op.expected,8)+" · ReLU = "+fmt(op.expectedActivated,8));
 }
 if(overview){conv1Text(dom.coordinate,"已显现 "+S.local+" / 676");conv1Text(dom.result,"暂停或点击位置进入 Detailed Inspection");}
 $("operation-content").dataset.inspectIndex=index;
 $("row").value=r;$("col").value=c;$("channel").value=q.out;
 const useCursor=!resting&&convVisual.manualIndex===null,vr=useCursor?point.row:r,vc=useCursor?point.col:c;
 const windowEl=$("receptive-window");windowEl.hidden=S.local===676;windowEl.style.transition="none";
 windowEl.style.left=windowEl.style.top="0";windowEl.style.width=windowEl.style.height=(3/28*100)+"%";
 windowEl.style.transform="translate3d("+(vc/3*100)+"%,"+(vr/3*100)+"%,0)";
 const active=!overview&&!resting&&playing?Math.min(8,Math.floor(convVisual.fraction*9)):-1;
 if(convVisual.activeWeight!==active){convVisual.activeWeight=active;dom.kernelCells.forEach((el,i)=>el.classList.toggle("weight-active",i===active));dom.terms.forEach((el,i)=>el.classList.toggle("weight-active",i===active));}
 dom.dot.style.left=(resting?100:convVisual.position/676*100)+"%";
 const canvas=$("target-canvas"),ctx=canvas.getContext("2d"),cw=canvas.width/26,ch=canvas.height/26,buffer=convVisual.buffer,bctx=buffer.getContext("2d");
 const preview=S.local===0&&!playing&&convVisual.position===0,range=Math.max(Math.abs(targetView.min),Math.abs(targetView.max),1e-9);
 const bufferKey=mapKey+"/"+preview;
 if(convVisual.bufferKey!==bufferKey||convVisual.painted>S.local){bctx.fillStyle="#080f1c";bctx.fillRect(0,0,360,360);convVisual.painted=0;convVisual.bufferKey=bufferKey;if(preview){bctx.globalAlpha=.22;for(let i=0;i<676;i++){bctx.fillStyle=color(targetView.values[i],range,"output");bctx.fillRect(i%26*cw+.3,Math.floor(i/26)*ch+.3,cw-.6,ch-.6);}bctx.globalAlpha=1;}}
 // Each real response is painted once in traversal order, even if one RAF spans several positions.
 for(let step=convVisual.painted;step<S.local;step++){const i=conv1ScanPoint(step).index;bctx.fillStyle=color(targetView.values[i],range,"output");bctx.fillRect(i%26*cw+.3,Math.floor(i/26)*ch+.3,cw-.6,ch-.6);if(convVisual.metrics)convVisual.metrics.revealedPositions++;}
 convVisual.painted=S.local;ctx.drawImage(buffer,0,0);
 if(!playing||convVisual.position<676){ctx.globalAlpha=playing?convVisual.fraction:1;ctx.fillStyle=color(targetView.values[index],range,"output");ctx.fillRect(c*cw+.3,r*ch+.3,cw-.6,ch-.6);ctx.globalAlpha=1;}
 ctx.strokeStyle="#f6c76b";ctx.lineWidth=2;ctx.strokeRect(vc*cw+1,vr*ch+1,cw-2,ch-2);
 canvas.dataset.written=String(S.local===676);canvas.dataset.inspectIndex=$("source-canvas").dataset.inspectIndex=index;
 conv1Text("target-caption",S.local===676?"完整 676/676 · 代表性响应 ["+r+", "+c+"]。":preview?"真实响应预览；播放从第一个位置开始。":"已显现 "+S.local+" / 676 · 蛇形扫描，不改变真实坐标或数值。");
 const controlKey=S.globalStatus+"/"+S.localStatus+"/"+convVisual.mode+"/"+(S.local===0)+"/"+(S.local===676);
 if(convVisual.controlKey!==controlKey){convVisual.controlKey=controlKey;updateControls();}
 else{conv1Text("local-position","STEP "+S.local+" / 676");$("local-progress").style.width=(convVisual.position/676*100)+"%";}
}

function renderDynamic(){
 if(S.current.key!=="pool2"&&$("advanced-pool-flight"))$("advanced-pool-flight").hidden=true;
 informationView?.update();multiChannel.update();
 if(layerAnim.handles(S.current.key)){layerAnim.render();return;}
 if(S.current.key==="conv1"){renderConv1();return;}
 $("receptive-window").style.transform="";
 const oldLegend=$("conv1-color-legend");if(oldLegend)oldLegend.remove();
 const d=S.current,q=S.selection,t=S.sample.tensors[d.tensor],op=operation(),coords=effectiveCoordinates();
 const sourceChannel=d.op==="conv"?(d.key==="conv2"?q.input:0):d.op==="flatten"?flatCoordinate(q.flatIndex).channel:t.shape.length===4?q.out:0;
 sourceView=mapFor(d.source,sourceChannel);targetView=mapFor(d.tensor,t.shape.length===4?q.out:0);
 let sourceHighlight=null,targetHighlight=null,windowRegion=null,reveal;
 if(d.op==="conv"){windowRegion={row:coords.row,col:coords.col,size:3};targetHighlight={row:coords.row,col:coords.col};}
 else if(d.op==="pool"){windowRegion={row:q.row*2,col:q.col*2,size:2};targetHighlight={row:q.row,col:q.col};}
 else if(d.op==="flatten"){const c=flatCoordinate(q.flatIndex);sourceHighlight={row:c.row,col:c.col};targetHighlight={row:Math.floor(q.flatIndex/40),col:q.flatIndex%40};reveal=S.local*25;}
 else if(d.op==="dense"||d.op==="output"){const start=(d.op==="dense"?50:8)*q.group;sourceHighlight={row:Math.floor(start/sourceView.w),col:start%sourceView.w};if(d.op==="dense")targetHighlight={row:Math.floor(q.neuron/16),col:q.neuron%16};}
 else{sourceHighlight=targetHighlight={row:q.row,col:q.col};if(d.op==="input")reveal=S.local?Math.ceil(784*S.local/S.limit):196;if(d.op==="relu")reveal=S.local?Math.ceil(targetView.values.length*S.local/S.limit):0;}
 drawMap($("source-canvas"),sourceView,{palette:d.source==="input"?"input":"feature",highlight:sourceHighlight});
 setWindow(sourceView,windowRegion);
 const isSpatialWrite=d.op==="conv"||d.op==="pool";
 const writeStep=d.key==="conv1"?15:d.key==="conv2"?38:d.op==="pool"?6:0;
 const written=!isSpatialWrite||S.local>=writeStep;
 drawMap($("target-canvas"),targetView,{palette:d.op==="input"?"input":d.op==="output"?"output":"feature",highlight:targetHighlight,reveal,
  ghost:isSpatialWrite&&S.local>0&&S.local<S.limit,written,activeIndex:targetHighlight?targetHighlight.row*targetView.w+targetHighlight.col:-1});
 $("target-canvas").dataset.written=String(written);
 const animationKey=S.sample.index+"/"+S.layer+"/"+S.local;
 if(isSpatialWrite&&S.local===writeStep&&lastWriteAnimation!==animationKey){
  lastWriteAnimation=animationKey;
  if(!matchMedia("(prefers-reduced-motion: reduce)").matches)$("target-canvas").animate([{opacity:.55},{opacity:1}],{duration:240,easing:"ease-out"});
 }
 lastView={source:sourceView,target:targetView,sourceHighlight,targetHighlight};
 $("source-title").textContent=d.op==="input"?"Original MNIST image":d.source==="pool1"?"Pool1 · input channel "+q.input:d.op==="flatten"?"Pool2 · channel "+sourceChannel:d.op==="dense"?"1600 input features":d.op==="output"?"128 hidden activations":d.source+" · channel "+sourceChannel;
 $("target-title").textContent=d.op==="output"?"Ten class probabilities":d.op==="flatten"?"1600-element vector":d.op==="dense"?"128 hidden activations":d.tensor+" · channel "+q.out;
 $("source-caption").textContent=d.source+" "+shape(S.sample.tensors[d.source])+(d.op==="dense"||d.op==="output"?" · display reshape only":"")+"\nrange "+fmt(sourceView.min)+" … "+fmt(sourceView.max);
 $("target-caption").textContent=d.tensor+" "+shape(t)+(t.shape.length===2?" · display reshape only":"")+"\nRecorded PyTorch values · click to inspect";
 $("channel").value=q.out;$("input-channel").value=d.key==="conv2"?q.input:q.group;
 $("row").value=q.row;$("col").value=q.col;$("neuron").value=q.neuron;$("flat-index").value=q.flatIndex;
 const activeChannel=d.op==="flatten"?flatCoordinate(q.flatIndex).channel:q.out;
 $("channel-strip").querySelectorAll("button").forEach(button=>button.classList.toggle("active",Number(button.dataset.channel)===activeChannel));
 const strip=$("channel-strip"),activeThumb=strip.querySelector(".active");
 if(activeThumb&&(activeThumb.offsetLeft<strip.scrollLeft||activeThumb.offsetLeft+activeThumb.offsetWidth>strip.scrollLeft+strip.clientWidth))strip.scrollLeft=Math.max(0,activeThumb.offsetLeft-strip.clientWidth/2);
 let terms=[],formula="",value=op.expected,label="RECORDED PYTORCH VALUE",detail="",error=0;
 $("local-heading").textContent=d.label+" · "+(d.op==="conv"?"weighted sum + bias":d.op==="pool"?"2 × 2 maximum":d.op==="dense"?"grouped weighted sum":d.op==="flatten"?"NCHW index mapping":d.op==="output"?"logits → Softmax":d.op==="relu"?"max(0, z)":"pixel → tensor");
 if(d.op==="conv"){
  const prefix=d.key==="conv1"?clamp(S.local-3,0,9):clamp(S.local-3,0,32);
  const partial=d.key==="conv1"?sum(op.terms.slice(0,prefix).map(term=>term.product)):sum(op.contributions.slice(0,prefix));
  const includeBias=d.key==="conv1"?S.local>=13:S.local>=36;
  $("operation-label").textContent=d.key==="conv1"?"3 × 3 CROSS-CORRELATION":"32 INPUT CHANNELS → ONE OUTPUT";
  $("operation-content").innerHTML='<div class="equation-block">'+matrixMarkup(op.patch)+
   '<span class="op-symbol">×</span>'+matrixMarkup(op.kernel,"weight")+
   '<span class="op-symbol">=</span>'+matrixMarkup(op.terms.map(term=>term.product),"product")+
   '</div><div class="operation-summary">'+(S.local>=1&&S.local<=3?'Scanning real window ['+coords.row+', '+coords.col+']':'Σ '+prefix+' / '+(d.key==="conv1"?9:32)+(d.key==="conv1"?' products':' channel contributions'))+
   '<br><strong>'+fmt(partial+(includeBias?op.bias:0),6)+'</strong>'+(includeBias?' · bias included':' · before bias')+'</div>';
  terms=op.terms.map((term,i)=>({text:fmt(term.x,4)+" × "+fmt(term.w,4)+" = "+fmt(term.product,4),full:JSON.stringify(term),revealed:d.key==="conv2"||S.local===0||i<prefix}));
  formula="z["+q.out+","+coords.row+","+coords.col+"] = Σ "+(d.key==="conv2"?"32 channel contributions":"9 products")+" + bias("+fmt(op.bias,8)+") = "+fmt(op.raw,9);
  detail="PyTorch z = "+fmt(op.expected,9)+"; next ReLU = "+fmt(op.expectedActivated,9);
  label="CONVOLUTION OUTPUT · BEFORE ReLU";error=Math.abs(op.raw-op.expected);
  $("operation-note").textContent=(d.key==="conv2"?"Kernel bank below: 32 real 3×3 filters for output "+q.out+". Current input channel "+q.input+". ":"")+"位置 ["+coords.row+", "+coords.col+"]；bias 只加一次。ReLU 预览属于下一个独立节点。";
  if(d.key==="conv2"){drawContributions(op.contributions,q.input,prefix);drawBank($("input-bank"),"pool1",q.input);drawBank($("kernel-bank"),null,q.input,true);}
 }else if(d.op==="pool"){
  const winner=op.values.indexOf(op.raw),active=S.local>=5?winner:clamp(S.local-1,0,3);
  $("operation-label").textContent="COMPARE FOUR REAL VALUES";
  $("operation-content").innerHTML='<div class="equation-block">'+matrixMarkup(op.values,"patch",active,true)+'<span class="op-symbol">→</span><div class="relu-block">max<br><strong>'+fmt(op.raw,6)+'</strong></div></div><div class="operation-summary">2 × 2 window · stride 2</div>';
  terms=op.values.map((v,i)=>({text:"x"+i+" = "+fmt(v,7)+(i===winner?"  ← max":""),revealed:S.local===0||i<S.local}));
  formula="max("+op.values.map(v=>fmt(v,7)).join(", ")+") = "+fmt(op.raw,9);
  detail="Y["+q.out+","+q.row+","+q.col+"] ← maximum; ties may share the maximum.";error=Math.abs(op.raw-op.expected);
  $("operation-note").textContent=S.local>=6?"最大值已写入当前池化输出位置。":"依次比较四个输入，再保留最大响应；每个通道独立处理。";
 }else if(d.op==="relu"){
  $("operation-label").textContent="POINTWISE NONLINEARITY";
  $("operation-content").innerHTML='<div class="relu-block">z = <span class="'+(op.raw<0?"negative":"")+'">'+fmt(op.raw,7)+'</span><br>max(0, z)<br><strong>'+fmt(op.expected,7)+'</strong></div>';
  const source=S.sample.tensors[d.source],r=Math.min(q.row,source.shape[2]-3),c=Math.min(q.col,source.shape[3]-3);
  for(let m=0;m<3;m++)for(let n=0;n<3;n++){const z=at(source,q.out,r+m,c+n);terms.push({text:"max(0, "+fmt(z,4)+") = "+fmt(Math.max(0,z),4),revealed:true});}
  formula="a = max(0, z); 每个元素独立计算。";detail="Selected ["+q.out+","+q.row+","+q.col+"]: "+fmt(op.raw,9)+" → "+fmt(op.expected,9);
  error=Math.abs(Math.max(0,op.raw)-op.expected);
  $("operation-note").textContent="负值响应以粉色表示；ReLU 后真实输出为零。动画按当前通道分组显现。";
 }else if(d.op==="flatten"){
  const coord=flatCoordinate(q.flatIndex);
  $("operation-label").textContent="SAME VALUES · NEW ADDRESS";
  $("operation-content").innerHTML='<div class="flatten-info">64 × 5 × 5 → 1600</div><div class="flatten-map">'+Array.from({length:64},(_,i)=>'<span class="'+(i<S.local?"done ":"")+(i===coord.channel?"active":"")+'" title="Source channel '+i+'"></span>').join("")+'</div><div class="operation-summary">channel '+coord.channel+' · row '+coord.row+' · col '+coord.col+'<br><strong>vector['+q.flatIndex+']</strong></div>';
  const start=Math.min(q.flatIndex,1591);
  terms=Array.from({length:9},(_,i)=>({text:"v["+(start+i)+"] = "+fmt(S.sample.tensors.flatten.values[start+i],6),revealed:true}));
  formula="index = channel × 25 + row × 5 + column = "+coord.channel+"×25 + "+coord.row+"×5 + "+coord.col+" = "+q.flatIndex;
  detail="Pool2["+coord.channel+","+coord.row+","+coord.col+"] = vector["+q.flatIndex+"] = "+fmt(value,9);
  error=Math.abs(op.raw-op.expected);
  $("operation-note").textContent="40×40 只是便于绘制的向量排布，不是新的网络空间维度。每个小块代表一个源通道。";
 }else if(d.op==="dense"){
  const prefix=clamp(S.local,0,32),partial=sum(op.contributions.slice(0,prefix))+(S.local>=33?op.bias:0);
  $("operation-label").textContent="32 EXACT GROUPS × 50 FEATURES";
  $("operation-content").innerHTML=denseSVG(op.contributions,q.group)+'<div class="dense-caption">Each line = Σ of 50 actual weighted inputs</div><div class="operation-summary">Σ '+prefix+' / 32 groups<br><strong>'+fmt(partial,7)+'</strong></div>';
  drawContributions(op.contributions,q.group,prefix);
  terms=op.terms.slice(0,9).map(term=>({text:"v["+term.index+"] × "+fmt(term.w,4)+" = "+fmt(term.product,4),full:JSON.stringify(term),revealed:true}));
  formula="Group "+q.group+" includes inputs "+q.group*50+"…"+(q.group*50+49)+". 显示该组前9项；完整50项之和 = "+fmt(op.contributions[q.group],8)+". Bias = "+fmt(op.bias,8);
  value=op.expectedActivated;label="HIDDEN NEURON "+q.neuron+" · AFTER ReLU";
  detail="Σ1600 weighted inputs + bias = "+fmt(op.raw,9)+"; ReLU = "+fmt(op.expectedActivated,9);
  error=Math.max(Math.abs(op.raw-op.expected),Math.abs(op.activated-op.expectedActivated));
  $("operation-note").textContent="只绘制32条真实分组贡献连接，而非204,800条权重线。Dropout 在 eval 模式中是恒等映射。";
 }else if(d.op==="output"){
  const probabilities=S.sample.tensors.probabilities.values,logits=S.sample.tensors.logits.values;
  $("operation-label").textContent="128 → 10 LOGITS → SOFTMAX";
  $("operation-content").innerHTML='<div class="output-equation">Selected digit <strong>'+q.neuron+'</strong><br>z = '+fmt(op.expected,8)+'<br>exp(z − max) / Σ exp(logits − max)<br><strong>'+fmt(probabilities[q.neuron]*100,7)+'%</strong></div>';
  drawContributions(op.contributions,q.group,op.groups);
  const chart=$("probability-chart");chart.replaceChildren();
  probabilities.forEach((probability,i)=>{
   const row=document.createElement("div");row.className="prob-row "+(i===S.sample.prediction?"winner":"");
   row.innerHTML='<span class="digit">'+i+'</span><div class="track"><div class="fill" style="width:'+(S.local>=11?probability*100:0)+'%"></div></div><span>'+fmt(probability*100,5)+'%</span><span class="logit" style="opacity:'+(S.local===0||i<S.local?1:.3)+'">logit '+fmt(logits[i],7)+'</span>';
   row.title="digit "+i+"; logit="+logits[i]+"; probability="+probability;chart.append(row);
  });
  terms=op.terms.map(term=>({text:"h["+term.index+"] × "+fmt(term.w,4)+" = "+fmt(term.product,4),full:JSON.stringify(term),revealed:true}));
  formula="Selected logit = Σ128(h×w) + bias("+fmt(op.bias,8)+"). Inspector group "+q.group+" displays its 8 actual inputs.";
  value=probabilities[q.neuron];label="PROBABILITY · DIGIT "+q.neuron;
  detail="True "+S.sample.groundTruth+" / predicted "+S.sample.prediction+"; selected logit "+fmt(op.expected,9);
  const computed=softmax(logits);
  error=Math.max(Math.abs(op.raw-op.expected),Math.abs(computed[q.neuron]-probabilities[q.neuron]));
  $("operation-note").textContent=S.local>=12?"十类概率来自同一份真实推理记录，预测类别已高亮。":"先产生10个 logits，再在第11步呈现 Softmax，第12步突出最终预测。";
 }else{
  $("operation-label").textContent="GRAYSCALE → FLOAT32 TENSOR";
  $("operation-content").innerHTML='<div class="input-conversion"><span class="tensor-bracket">[</span><div class="tensor-description">1 × 1 × 28 × 28<small>batch · channel · height · width</small></div><span class="tensor-bracket">]</span></div><div class="operation-summary">pixel['+q.row+', '+q.col+'] = <strong>'+fmt(op.raw,6)+'</strong></div>';
  const input=S.sample.tensors.input,r=Math.min(q.row,25),c=Math.min(q.col,25);
  for(let m=0;m<3;m++)for(let n=0;n<3;n++)terms.push({text:"X["+(r+m)+","+(c+n)+"] = "+fmt(at(input,0,r+m,c+n),5),revealed:true});
  formula="ToTensor: uint8 / 255 → float32 in [0,1]. 这里只展示真实输入，没有卷积或额外归一化。";
  detail="Test index "+S.sample.index+"; ground truth "+S.sample.groundTruth+"; local pixel ["+q.row+","+q.col+"].";
  $("operation-note").textContent="左侧为完整输入，右侧按行分组显现对应 Tensor。浏览器没有运行 PyTorch。";
 }
 const grid=$("term-grid");grid.replaceChildren();
 for(const term of terms){const item=document.createElement("div");item.className="term "+(term.revealed?"revealed":"future");item.textContent=term.text;item.title=term.full||term.text;grid.append(item);}
 $("calculation-formula").textContent=formula;$("result-label").textContent=label;$("result-value").textContent=fmt(value,8);
 $("result-value").classList.toggle("negative",value<0);$("result-detail").textContent=detail;
 const expectedForCheck=d.op==="relu"?op.expected:d.op==="flatten"?op.expected:op.expected;
 const actualForCheck=d.op==="relu"?Math.max(0,op.raw):op.raw;
 const good=near(actualForCheck,expectedForCheck,DATA.metadata.tolerance);
 $("consistency").textContent=(good?"✓ Within tolerance":"✕ Outside tolerance")+" · max |Δ| "+error.toExponential(2)+"\nabs 1e−4 + rel 1e−5";
 $("consistency").dataset.passed=String(good);
 updateControls();
}

/** Publish the exact displayed coordinate; the 3D follower never extrapolates. */
function publishSpatial(){
 const key=S.current.key;if(!["conv1","conv2","relu1","relu2","pool1","pool2"].includes(key))return;
 const side=S.sample.tensors[S.current.tensor].shape[3],stride=key.startsWith("pool")?2:1,kernel=key.startsWith("conv")?3:stride;
 const position=key==="conv1"?convVisual.position:layerAnim.state.position;
 const count=side*side,progress=key==="relu1"?position/100:position/count;
 const p=root.DigitVisionAdvancedSpatial.SPECS[key]?root.DigitVisionAdvancedSpatial.point(root.DigitVisionAdvancedSpatial.SPECS[key],progress*count):null; const point=p?{index:p.index,step:p.step,fraction:p.phase,row:p.visualRow,col:p.visualCol}:root.DigitVisionSpatialLayers.scanPoint(progress*count,side);
 const manual=key==="conv1"?convVisual.manualIndex:layerAnim.state.manualIndex;
 const row=manual!==null?Math.floor(manual/side):Math.floor(point.index/side),col=manual!==null?manual%side:point.index%side;
 root.DigitVisionSpatialSync?.publish({layer:key,sample:S.samplePosition,progress,traversal:root.DigitVisionAdvancedSpatial.SPECS[key]?"row-major":"snake",operation:key,phase:point.fraction,speed:Number($("local-speed").value),direction:layerAnim.state.direction||1,playbackState:S.localStatus,selectedOutputChannel:S.selection.out,selectedChannel:S.selection.input,inputRow:row*stride,inputCol:col*stride,outputRow:row,outputCol:col,visualInputRow:(manual!==null?row:point.row)*stride,visualInputCol:(manual!==null?col:point.col)*stride,visualOutputRow:manual!==null?row:point.row,visualOutputCol:manual!==null?col:point.col,kernelHeight:kernel,kernelWidth:kernel,stride,index:point.step});
}

function openZoom(which){if(layerAnim.handles(S.current.key)){layerAnim.openZoom(which);return;}
 const map=which==="source"?sourceView:targetView;
 $("zoom-title").textContent=(which==="source"?$("source-title").textContent:$("target-title").textContent)+" · enlarged";
 drawMap($("zoom-canvas"),map,{palette:map.key==="input"?"input":map.key==="probabilities"?"output":"feature",highlight:which==="source"?lastView.sourceHighlight:lastView.targetHighlight});
 $("zoom-caption").textContent=map.key+" | channel "+map.channel+" | display "+map.h+"×"+map.w+" | real range "+fmt(map.min,8)+" … "+fmt(map.max,8);
 $("zoom-dialog").showModal();
}
function selectMapPosition(event,which){
 const canvas=event.currentTarget,rect=canvas.getBoundingClientRect(),map=which==="source"?sourceView:targetView;
 const row=clamp(Math.floor((event.clientY-rect.top)/rect.height*map.h),0,map.h-1),col=clamp(Math.floor((event.clientX-rect.left)/rect.width*map.w),0,map.w-1);
 const d=S.current,q=S.selection,t=S.sample.tensors[d.tensor];
 if(d.op==="flatten"){
  if(which==="target")q.flatIndex=clamp(row*40+col,0,1599);
  else q.flatIndex=Math.floor(q.flatIndex/25)*25+row*5+col;
 }else if(d.op==="dense"||d.op==="output"){
  if(which==="target"&&d.op==="dense")q.neuron=clamp(row*16+col,0,127);
  else if(which==="source")q.group=clamp(Math.floor((row*map.w+col)/(d.op==="dense"?50:8)),0,d.op==="dense"?31:15);
 }else{
  const scale=which==="source"&&d.op==="pool"?2:1;
  q.row=clamp(Math.floor(row/scale),0,t.shape[2]-1);q.col=clamp(Math.floor(col/scale),0,t.shape[3]-1);
 }
 selectionChanged();
}
$("layer-animation-mode").addEventListener("change",()=>layerAnim.modeChanged($("layer-animation-mode").value));
$("conv1-mode").addEventListener("change",()=>{pauseAll();convVisual.mode=$("conv1-mode").value;convVisual.metrics=null;convVisual.key="";renderConv1();});
$("global-play").addEventListener("click",()=>new URLSearchParams(location.search).get("selftest")==="1"?globalPlay():globalEngine.play());$("global-pause").addEventListener("click",()=>new URLSearchParams(location.search).get("selftest")==="1"?pauseAll():globalEngine.pause());
$("global-next").addEventListener("click",()=>new URLSearchParams(location.search).get("selftest")==="1"?jump(S.layer+1):globalEngine.move(1));$("global-previous").addEventListener("click",()=>new URLSearchParams(location.search).get("selftest")==="1"?jump(S.layer-1):globalEngine.move(-1));$("global-restart").addEventListener("click",globalRestart);
$("local-play").addEventListener("click",localPlay);$("local-pause").addEventListener("click",pauseAll);
$("local-next").addEventListener("click",localNext);$("local-previous").addEventListener("click",localPrevious);$("local-restart").addEventListener("click",localRestart);
for(const kind of ["global","local"])$(kind+"-speed").addEventListener("input",()=>{$(kind+"-speed-label").textContent=$(kind+"-speed").value+"×";schedule();});
$("sample").addEventListener("change",()=>{globalEngine.pause();stopClock();convVisual.manualIndex=null;convVisual.fraction=0;convVisual.key="";convVisual.position=0;convVisual.metrics=null;layerAnim.reset("input");S.switchSample(Number($("sample").value));buildPipeline();renderAll();globalEngine.sampleChanged();});
for(const [id,key]of [["channel","out"],["input-channel","input"],["row","row"],["col","col"],["neuron","neuron"],["flat-index","flatIndex"]]){
 $(id).addEventListener("focus",pauseAll);
 $(id).addEventListener("change",()=>{
  const value=Math.trunc(Number($(id).value));
  if(!Number.isFinite(value)){renderAll();return;}
  const target=S.sample.tensors[S.current.tensor];
  let bounded=value;
  if(id==="row")bounded=clamp(value,0,target.shape[2]-1);
  if(id==="col")bounded=clamp(value,0,target.shape[3]-1);
  if(id==="flat-index")bounded=clamp(value,0,1599);
  if(id==="neuron")bounded=clamp(value,0,S.current.op==="dense"?127:9);
  if(id==="input-channel"&&S.current.key!=="conv2")S.selection.group=value;
  else S.selection[key]=bounded;
  selectionChanged();
 });
}
$("source-canvas").addEventListener("click",event=>selectMapPosition(event,"source"));
$("target-canvas").addEventListener("click",event=>selectMapPosition(event,"target"));
$("zoom-source").addEventListener("click",()=>openZoom("source"));$("zoom-target").addEventListener("click",()=>openZoom("target"));$("zoom-close").addEventListener("click",()=>$("zoom-dialog").close());
for(const id of ["input-bank","kernel-bank"])$(id).addEventListener("click",event=>{
 const rect=event.currentTarget.getBoundingClientRect();
 S.selection.input=clamp(Math.floor((event.clientY-rect.top)/rect.height*4)*8+Math.floor((event.clientX-rect.left)/rect.width*8),0,31);selectionChanged();
});
$("contributions").addEventListener("click",event=>{
 const rect=event.currentTarget.getBoundingClientRect(),count=S.current.key==="conv2"?32:S.current.op==="dense"?32:16;
 const index=clamp(Math.floor((event.clientX-rect.left)/rect.width*count),0,count-1);
 if(S.current.key==="conv2")S.selection.input=index;else S.selection.group=index;
 selectionChanged();
});
document.addEventListener("keydown",event=>{
 if(["INPUT","SELECT","TEXTAREA","BUTTON"].includes(event.target.tagName)||event.ctrlKey||event.metaKey||event.altKey||$("zoom-dialog").open)return;
 if(event.key===" "){event.preventDefault();if(new URLSearchParams(location.search).get("selftest")==="1")S.globalStatus==="PLAYING"?pauseAll():globalPlay();else globalEngine.snapshot().state==="PLAYING"?globalEngine.pause():globalEngine.play();}
 else if(event.key==="ArrowRight"){event.preventDefault();localNext();}
 else if(event.key==="ArrowLeft"){event.preventDefault();localPrevious();}
 else if(event.key.toLowerCase()==="r")new URLSearchParams(location.search).get("selftest")==="1"?globalRestart():globalEngine.restart();
});
DATA.samples.forEach((sample,index)=>{const option=document.createElement("option");option.value=index;option.textContent="Index "+sample.index+" · digit "+sample.groundTruth;$("sample").append(option);});
$("provenance").textContent="Real PyTorch "+DATA.metadata.torch+"; checkpoint epoch "+DATA.metadata.checkpointEpoch+"; SHA256 "+DATA.metadata.checkpointSha256+". Exported batch "+DATA.metadata.batchShape.join("×")+" in "+DATA.metadata.forwardCalls+" model call. Float comparison: |a-b| ≤ 1e−4 + 1e−5|b|.";
buildPipeline();renderAll();
const globalEngine=root.DigitVisionGlobalEngine.create({S,DATA,view:()=>informationView.snapshot().view,stepMode:isStepMode,restore:()=>renderDynamic(),
 enter:(ui,stage)=>{stopClock();const index=LAYERS.findIndex(v=>v.key===ui);if(index!==S.layer){layerAnim.reset(ui);convVisual.position=0;convVisual.key="";S.jump(index,true);renderAll();}S.globalStatus="PAUSED";S.localStatus="PAUSED";},
 phase:(stage,progress,localElapsed,playbackState)=>{const max=S.limit,next=stage.stageId==="S12"?11:stage.stageId==="S13"?12:stage.stageId==="S10"?max:Math.min(max,Math.floor(progress*max));S.localStatus=playbackState==="PLAYING"?"PLAYING":playbackState==="COMPLETED"?"COMPLETED":"PAUSED";if(stage.stageId==="S02"){convVisual.position=progress*676;convVisual.fraction=convVisual.position%1;convVisual.last=null;}
  if(layerAnim.handles(S.current.key)){layerAnim.state.position=progress*max;layerAnim.state.fraction=layerAnim.state.position%1;layerAnim.state.last=null;}
  if(S.local!==next){S.local=next;if(S.current.key==="flatten")S.selection.flatIndex=Math.min(1599,Math.floor(progress*1600));if(S.current.key==="conv2")S.selection.input=Math.min(31,Math.floor(progress*32));renderDynamic();}
  if(stage.stageId==="S10")$("operation-note").textContent="Inference Mode · Dropout(p=0.5) 为 Identity；128 个真实值保持不变。";
  if(stage.stageId==="S11"||stage.stageId==="S12"||stage.stageId==="S13"){$("operation-note").textContent=stage.stageName+" · "+stage.inputTensorKey+" → "+stage.outputTensorKey+" · 真实模型导出记录";}
 }});
root.ForwardExplorer={globalSnapshot:()=>globalEngine.snapshot(),globalReport:()=>globalEngine.report(),globalSeek:ms=>globalEngine.seek(ms),setView:view=>informationView.setView(view),viewSnapshot:()=>informationView.snapshot(),multiChannelSnapshot:()=>multiChannel.snapshot(),multiChannelReport:()=>multiChannel.report(),spatialAnimationReport:()=>layerAnim.state.metrics?JSON.parse(JSON.stringify(layerAnim.state.metrics)):null,animationReport:()=>convVisual.metrics?JSON.parse(JSON.stringify(convVisual.metrics)):null,snapshot:()=>({sample:S.sample.index,layer:S.current.key,localStep:S.local,localLimit:S.limit,globalStatus:S.globalStatus,localStatus:S.localStatus,selection:{...S.selection},spatial:layerAnim.snapshot(),animation:{position:convVisual.position,fraction:convVisual.fraction,mode:convVisual.mode,inspectIndex:convVisual.manualIndex,cursor:conv1ScanPoint(convVisual.position)}})};
document.body.dataset.qaSelftest=new URLSearchParams(location.search).get("selftest")==="1"?"1":"0";
const params=new URLSearchParams(location.search);
if(["fast","detailed","step"].includes(params.get("mode"))){convVisual.mode=params.get("mode");$("conv1-mode").value=convVisual.mode;}
if(params.has("sample")){const pos=DATA.samples.findIndex(sample=>sample.index===Number(params.get("sample")));if(pos>=0){S.switchSample(pos);buildPipeline();renderAll();}}
if(params.has("layer")){const index=LAYERS.findIndex(d=>d.key===params.get("layer"));if(index>=0)jump(index);}
if(params.has("step")&&/^\d+$/.test(params.get("step"))){for(let i=0,n=Math.min(Number(params.get("step")),S.limit);i<n;i++)S.advance();S.localStatus=S.local===S.limit?"COMPLETED":"PAUSED";renderDynamic();}

async function runBrowserChecks(){
 const results=[],assert=(condition,name)=>{if(!condition)throw new Error(name);results.push({name,passed:true});};
 const wait=ms=>new Promise(resolve=>setTimeout(()=>{if((S.current.key==="conv1"||layerAnim.handles(S.current.key))&&(S.globalStatus==="PLAYING"||S.localStatus==="PLAYING")){cancelAnimationFrame(convRAF);if(S.current.key==="conv1"){convVisual.last=0;conv1Frame(ms);}else{layerAnim.state.last=0;layerAnim.frame(ms);}}resolve();},ms));
 const click=id=>$(id).click();
 const set=(id,value)=>{$(id).value=value;$(id).dispatchEvent(new Event("change",{bubbles:true}));};
 const layer=key=>document.querySelector('[data-layer="'+key+'"]').click();
 try{
  globalRestart();
  assert(S.sample.index===0&&S.sample.groundTruth===7&&S.sample.prediction===7,"Default real sample 0: 7 -> 7");
  assert(validateBundle(DATA),"All recorded tensor shapes and values valid");
  assert(location.protocol==="file:"||["127.0.0.1","localhost"].includes(location.hostname),"Offline file or local static server, no inference backend");
  assert([...document.scripts].every(script=>!script.src||(new URL(script.src).protocol==="file:"||new URL(script.src).origin===location.origin)),"All script resources local");
  layer("conv1");set("channel",2);set("row",5);set("col",6);
  let op=operation();assert(near(op.raw,op.expected),"Conv1 selected 9 products plus bias match PyTorch");
  const startPosition=S.local;click("local-next");assert(S.local===startPosition+1,"Local next advances");
  click("local-previous");assert(S.local===startPosition,"Local previous restores progress");
  $("local-speed").value=8;$("local-speed").dispatchEvent(new Event("input",{bubbles:true}));
  click("local-play");await wait(190);click("local-pause");const paused=S.local;const frozenPhase=convVisual.fraction;await wait(190);
  assert(convVisual.fraction===frozenPhase,"Conv1 pause preserves exact visual phase");
  assert(paused>0&&S.local===paused&&S.current.key==="conv1","Local playback and pause, no global layer advance");
  click("local-play");
  for(let i=0;i<550&&S.local<S.limit;i++)await wait(80);
  assert(S.local===S.limit&&S.current.key==="conv1","Local resume completes only current operation");
  assert($("target-canvas").dataset.written==="true","Conv1 result write is revealed at operation completion");
  click("local-restart");assert(S.local===0,"Local restart clears operation progress");
  assert($("target-canvas").dataset.written==="false","Local restart resets output write state");
  document.querySelector('.channel-thumb[data-channel="1"]').click();
  assert(S.selection.out===1,"Feature channel thumbnail selection");
  click("zoom-target");assert($("zoom-dialog").open,"Feature map enlargement opens");click("zoom-close");assert(!$("zoom-dialog").open,"Enlargement closes");
  layer("pool1");op=operation();assert(op.raw===op.expected,"Pool1 maximum matches recorded output");
  const canvas=$("target-canvas"),rect=canvas.getBoundingClientRect();
  canvas.dispatchEvent(new MouseEvent("click",{bubbles:true,clientX:rect.left+rect.width*.15,clientY:rect.top+rect.height*.15}));
  assert(S.selection.row===1&&S.selection.col===1,"Canvas spatial position selection");
  layer("conv2");set("channel",3);set("input-channel",5);set("row",4);set("col",4);
  op=operation();assert(op.contributions.length===32&&op.kernel.length===9&&near(op.raw,op.expected),"Conv2 all 32 channel contributions plus one bias match PyTorch");
  assert(!$("input-bank").hidden&&!$("kernel-bank").hidden,"Conv2 exposes real 32-channel input and kernel banks");
  for(let i=0;i<20;i++)click("local-next");
  assert(S.local===68&&S.selection.input===5,"Conv2 advances twenty spatial positions from selected [4,4], preserving inspected input channel");
  click("local-previous");assert(S.local===67&&S.selection.input===5,"Conv2 spatial progress can rewind without changing inspected channel");
  layer("pool2");op=operation();assert(op.raw===op.expected,"Pool2 maximum matches recorded output");
  layer("flatten");set("flat-index",624);op=operation();
  assert(op.channel===24&&op.row===4&&op.col===4&&op.raw===op.expected,"Flatten exact C,H,W mapping at index 624");
  layer("fc1");set("neuron",9);set("input-channel",2);op=operation();
  assert(op.contributions.length===32&&op.terms.length===50&&near(op.raw,op.expected)&&near(op.activated,op.expectedActivated),"FC1 full 1600-input sum, bias and ReLU");
  layer("output");set("neuron",7);op=operation();
  assert(near(op.raw,op.expected),"Output selected 128-input logit");
  const probabilities=softmax(S.sample.tensors.logits.values);
  assert(probabilities.every((v,i)=>near(v,S.sample.tensors.probabilities.values[i])),"All ten Softmax probabilities agree");
  const oldPixels=$("sample-preview").toDataURL();
  layer("conv1");click("local-play");await wait(5);
  set("sample",1);
  assert(convVisual.fraction===0,"Sample switch resets old Conv1 visual phase");
  assert(S.sample.index===1&&S.sample.groundTruth===2&&S.sample.prediction===2&&$("predicted-label").textContent==="2"&&S.layer===0,"Switch to new real sample 1, reset state");
  assert($("sample-preview").toDataURL()!==oldPixels,"Sample pixels really change, no stale image");
  layer("conv2");assert(near(operation().raw,operation().expected),"New sample has its own correct Conv2 tensors");
  set("sample",2);assert(S.sample.index===2&&S.sample.prediction===1,"Third real sample loads");
  set("sample",0);
  $("global-speed").value=8;$("global-speed").dispatchEvent(new Event("input",{bubbles:true}));
  click("global-play");await wait(190);click("global-pause");const frozen=[S.layer,S.local].join(",");await wait(190);
  assert([S.layer,S.local].join(",")===frozen&&S.globalStatus==="PAUSED","Global play and pause stop local clock");
  click("global-restart");click("global-next");assert(S.layer===1,"Next layer");
  click("global-previous");assert(S.layer===0,"Previous layer");
  layer("output");assert(S.layer===9,"Direct layer jump");click("global-restart");
  assert(S.layer===0&&S.local===0&&S.globalStatus==="IDLE","Global restart");
  click("global-play");
  for(let i=0;i<500&&S.globalStatus!=="COMPLETED";i++)await wait(100);
  assert(S.globalStatus==="COMPLETED"&&S.visited.size===10,"Play All visits all ten nodes and completes local operations");
  assert(document.querySelectorAll(".prob-row").length===10&&S.sample.prediction===7,"Final probabilities and real prediction displayed");
  click("global-restart");
  document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowRight",bubbles:true}));assert(S.local===1,"Keyboard local next");
  document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowLeft",bubbles:true}));assert(S.local===0,"Keyboard local previous");
  document.body.dispatchEvent(new KeyboardEvent("keydown",{key:" ",bubbles:true}));assert(S.globalStatus==="PLAYING","Keyboard global play");
  document.body.dispatchEvent(new KeyboardEvent("keydown",{key:" ",bubbles:true}));assert(S.globalStatus==="PAUSED","Keyboard global pause");
  document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"r",bubbles:true}));assert(S.layer===0&&S.local===0,"Keyboard global restart");
  assert(document.documentElement.scrollWidth<=innerWidth,"Responsive layout has no page-wide horizontal overflow");
  assert($("app-error").hidden,"No application runtime error");
  layer("conv2");for(let i=0;i<20;i++)click("local-next");
  const report={passed:true,clockValidation:"Conv1 uses deterministic timestamp injection in headless virtual time; production uses native requestAnimationFrame",checks:results.length,protocol:location.protocol,viewport:{width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth},samples:DATA.samples.map(s=>({index:s.index,true:s.groundTruth,prediction:s.prediction})),results,final:root.ForwardExplorer.snapshot()};
  $("qa-report").textContent=JSON.stringify(report,null,2);document.documentElement.dataset.qaStatus="passed";
 }catch(error){pauseAll();$("qa-report").textContent=JSON.stringify({passed:false,error:error.message,results,snapshot:root.ForwardExplorer.snapshot()},null,2);document.documentElement.dataset.qaStatus="failed";}
 $("qa-panel").hidden=false;
}
if(S.current.key==="conv1"&&params.has("phase")){convVisual.fraction=clamp(Number(params.get("phase"))||0,0,.999);renderConv1();}
if(params.get("selftest")==="1")runBrowserChecks().then(()=>{
 if(document.documentElement.dataset.qaStatus!=="passed")return;
 try{
  jump(1);const rep=representativeConv1(DATA,S.sample,S.selection.out);
  if(Number($("target-canvas").dataset.inspectIndex)!==rep.index||rep.energy<=0)throw new Error("Representative default mismatch");
  for(let i=0;i<676;i++)S.advance();renderConv1();
  if(!$("receptive-window").hidden||Number($("target-canvas").dataset.inspectIndex)!==rep.index)throw new Error("Completion representative focus mismatch");
  if(S.local!==676||$("target-canvas").dataset.written!=="true")throw new Error("Full scan lost");
  const report=JSON.parse($("qa-report").textContent);
  report.refinement={passed:true,positions:S.local,representative:rep,channel:S.selection.out,sourceOutputLinked:$("source-canvas").dataset.inspectIndex===$("target-canvas").dataset.inspectIndex,negativeZeroPositiveLegend:!!$("conv1-color-legend")};
  $("qa-report").textContent=JSON.stringify(report,null,2);document.documentElement.dataset.refinementStatus="passed";
 }catch(e){document.documentElement.dataset.qaStatus="failed";$("qa-report").textContent=JSON.stringify({passed:false,error:e.message});}
});
})(typeof window!=="undefined"?window:globalThis);












