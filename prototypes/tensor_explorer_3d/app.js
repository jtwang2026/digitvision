import {SceneManager} from "./renderer/scene_manager.js?v=122";
import {tensorFor,viewForKey,tensorValue,channelStats,conv1At,conv2At} from "./core/tensor_adapter.js?v=12";
import {responseColor,colorScale} from "./core/color_mapper.js";
import {TensorAnimationEngine} from "./core/animation_engine.js?v=12";
import {OPERATIONS,processingSpec,cursorAt} from "./core/processing_cursor.js?v=12";

const data=window.DIGITVISION_DATA;
const controller=document.querySelector('.animation-bar');document.getElementById('scene').before(controller);
const layoutStyle=document.createElement('style');layoutStyle.textContent='#scene{height:clamp(320px,52vh,600px)}.animation-bar{position:sticky;top:0;z-index:5;background:#101a2b}';document.head.append(layoutStyle);
if(!data?.samples?.length)throw Error("Real DigitVision tensor export unavailable");
const $=id=>document.getElementById(id), state={sample:0,layer:"conv1",activation:"relu",scale:"global",view:"split",side:"output",outputChannel:16,inputChannel:0,row:20,col:13};
const scene=new SceneManager($("scene"),pick=>{stopDemo();animation.pause();state.side=pick.side;if(pick.side==="input")state.inputChannel=pick.channel;else state.outputChannel=pick.channel;state.row=pick.row;state.col=pick.col;renderSelection();});
let input=null,output=null,processing=null,lastCursor=null,demoPlaying=false,demoStart=0;
let following2D=false;
const fmt=v=>Number(v).toFixed(6);
const current=()=>state.side==="input"&&input?input:output;
const channel=()=>state.side==="input"?state.inputChannel:state.outputChannel;
const animationNotes={assembly:"真实通道平面逐层组装；位置移动仅用于观察 Tensor 结构。",scan:"沿 Z 轴观察通道。Inspector 与二维特征图读取同一真实 Tensor。",tracking:"同一 Row / Column 在所有通道对齐；曲线显示真实响应值。",separation:"选中通道临时浮出，索引及像素坐标保持不变。",flow:"青色扫描光带只是教学观察方向，不代表 Tensor 元素在物理空间移动。",spatial:"窗口在 XY 平面扫描；所有输入通道同步，输出坐标与真实计算一致。教学顺序不代表 GPU 执行顺序。",flatten:"按 C×25+Row×5+Column 保留元素身份；高亮元素沿二维到一维轨迹移动。"};
const animation=new TensorAnimationEngine(frame=>{
  if(!output)return;
  const view=current();
  if((frame.mode==="scan"||frame.mode==="tracking")&&(frame.status==="playing"||frame.status==="complete")){
    const index=Math.min(view.channels-1,Math.floor(frame.progress*view.channels));
    const next=frame.direction===1?index:view.channels-1-index;
    if(channel()!==next){if(state.side==="input")state.inputChannel=next;else state.outputChannel=next;renderSelection();}
  }
  if(frame.mode==="separation"&&frame.status==="idle")renderSelection();
  if(frame.mode==="spatial"&&processing){
    lastCursor={operation:state.layer,...cursorAt(processing,frame.progress,frame.direction),activeChannels:processing.inputChannels,selectedOutputChannel:state.outputChannel};
    if(state.row!==lastCursor.outputRow||state.col!==lastCursor.outputCol||state.side!=="output"){
      state.side="output";state.row=lastCursor.outputRow;state.col=lastCursor.outputCol;renderSelection();
    }
  }else if(lastCursor){lastCursor=null;scene.setProcessingCursor(null,state.outputChannel);}
  if(frame.mode==="flatten"&&state.layer==="flatten"){
    const order=Math.min(1599,Math.floor(frame.progress*1600)),index=frame.direction===1?order:1599-order;
    const ch=Math.floor(index/25),row=Math.floor(index%25/5),col=index%5;
    if(state.outputChannel!==ch||state.row!==row||state.col!==col){state.outputChannel=ch;state.row=row;state.col=col;renderSelection();}
    drawFlatten(frame,index);scene.setFlatten(frame.progress,index,data.samples[state.sample].tensors.flatten.values[index]);
  }
  scene.applyAnimation(frame.status==="idle"?"idle":frame.mode,frame.progress,state.side,channel(),frame.direction);
  if(lastCursor){scene.setProcessingCursor(lastCursor,state.outputChannel);renderProcessingDetail(lastCursor);}
  $("anim-progress").style.width=(frame.progress*100).toFixed(1)+"%";
  $("anim-status").textContent=frame.status.toUpperCase()+" · "+Math.round(frame.progress*100)+"%";
  $("anim-pause").disabled=frame.status!=="playing";
  $("anim-return").disabled=frame.mode!=="separation"||frame.progress===0;
  $("anim-direction").disabled=frame.mode!=="scan"&&frame.mode!=="tracking"&&frame.mode!=="spatial"&&frame.mode!=="flatten";
  $("response-panel").hidden=frame.mode!=="tracking";
  $("processing-panel").hidden=frame.mode!=="spatial"||!processing;
  $("anim-note").textContent=animationNotes[frame.mode];
  if(frame.mode==="tracking"&&frame.status==="complete")drawResponse();
});
scene.onFrame=now=>{if(!following2D){animation.tick(now);runDemo(now);}};
function setViews(resetCamera=false){
  processing=OPERATIONS[state.layer]?processingSpec(data,state.sample,state.layer,state.activation):null;
  output=state.layer==="flatten"?viewForKey(data,state.sample,"pool2"):processing?viewForKey(data,state.sample,processing.outputKey):tensorFor(data,state.sample,state.layer,state.activation,"output");
  input=processing?viewForKey(data,state.sample,processing.inputKey):null;
  if(!input)state.side="output";
  state.outputChannel=Math.min(output.channels-1,state.outputChannel);
  state.inputChannel=input?Math.min(input.channels-1,state.inputChannel):0;
  const v=current();state.row=Math.min(v.height-1,state.row);state.col=Math.min(v.width-1,state.col);
  lastCursor=animation.mode==="spatial"&&processing?{...cursorAt(processing,animation.progress,animation.direction),activeChannels:processing.inputChannels,selectedOutputChannel:state.outputChannel}:null;
  scene.setViews(input,output,state.scale,resetCamera);
  if(state.layer.startsWith('relu')&&input)scene.setReluSource(input,state.scale);
  document.body.dataset.layer=state.layer;
  $("shape-chip").textContent=state.layer==="flatten"?"1×64×5×5 → 1×1600":input?input.tensor.shape.join("×")+" → "+output.tensor.shape.join("×"):output.tensor.shape.join("×");
  $("flatten-panel").hidden=state.layer!=="flatten";
  $("conv2-panel").hidden=state.layer!=="conv2";
  $("stack-side").hidden=!input;
  if(input){$("stack-side").querySelector('[data-side="input"]').textContent=input.channels+" input channels";$("stack-side").querySelector('[data-side="output"]').textContent=output.channels+" output channels";}
  $("scene-status").textContent=input?input.channels+" INPUT + "+output.channels+" OUTPUT PLANES":output.channels+" OUTPUT PLANES";
  renderSelection();
}
function drawDetail(){
  const view=current(),ch=channel(),canvas=$("detail-map"),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height,scale=colorScale(view,state.scale,ch);
  const building=animation.mode==="spatial"&&lastCursor&&state.side==="output";
  const visited=(row,col)=>{const n=row*view.width+(row%2?view.width-1-col:col);return lastCursor.direction===1?n<=lastCursor.index:n>=view.height*view.width-1-lastCursor.index;};
  ctx.fillStyle="#091422";ctx.fillRect(0,0,w,h);
  for(let row=0;row<view.height;row++)for(let col=0;col<view.width;col++){
    if(building&&!visited(row,col))continue;
    const rgba=responseColor(tensorValue(view,ch,row,col),scale);ctx.fillStyle=`rgb(${rgba[0]},${rgba[1]},${rgba[2]})`;
    ctx.fillRect(col*w/view.width,row*h/view.height,Math.ceil(w/view.width),Math.ceil(h/view.height));
  }
  const x=state.col*w/view.width,y=state.row*h/view.height;
  ctx.strokeStyle="#f6c76b";ctx.lineWidth=2.5;ctx.strokeRect(x+1,y+1,w/view.width-2,h/view.height-2);
}
function drawFlatten(frame,index){
  const values=data.samples[state.sample].tensors.flatten.values;
  const canvas=$("flatten-canvas"),ctx=canvas.getContext("2d"),width=canvas.width;
  const ch=Math.floor(index/25),row=Math.floor(index%25/5),col=index%5;
  const actual=tensorValue(output,ch,row,col),flat=values[index];
  if(actual!==flat)throw Error("Flatten identity mismatch at "+index);
  ctx.fillStyle="#0b1728";ctx.fillRect(0,0,width,canvas.height);
  const scale=colorScale(output,"global",0);
  for(let r=0;r<5;r++)for(let c=0;c<5;c++){
    const rgb=responseColor(tensorValue(output,ch,r,c),scale);
    ctx.fillStyle=`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;ctx.fillRect(24+c*19,20+r*19,17,17);
  }
  ctx.fillStyle="#8fb4c8";ctx.font="12px Consolas";ctx.fillText("C"+ch+" 5×5",24,133);ctx.fillText("1600 维真实向量",190,37);
  const visited=Math.min(1600,Math.floor(frame.progress*1600)+1);
  for(let order=0;order<visited;order++){
    const i=frame.direction===1?order:1599-order,v=values[i],rgb=responseColor(v,scale);
    ctx.fillStyle=`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;ctx.fillRect(190+i*590/1600,66,Math.max(.5,590/1600),35);
  }
  const phase=frame.progress*1600-Math.floor(frame.progress*1600),t=phase*phase*(3-2*phase);
  const x=33+col*19+(190+index*590/1600-(33+col*19))*t,y=28+row*19+(83-(28+row*19))*t;
  ctx.strokeStyle="#f6c76b";ctx.lineWidth=2;ctx.strokeRect(23+col*19,19+row*19,19,19);
  ctx.fillStyle="#f6c76b";ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.fill();
  ctx.fillText("index="+index+"  value="+fmt(flat),190,126);
  $("flatten-label").textContent=`C${ch}[${row},${col}] → Vector[${index}]`;
}
function drawResponse(){
  const frame=animation.snapshot();if(frame.mode!=="tracking")return;
  const view=current(),values=Array.from({length:view.channels},(_,i)=>tensorValue(view,i,state.row,state.col));
  const canvas=$("response-chart"),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
  const lo=Math.min(0,...values),hi=Math.max(0,...values),range=Math.max(1e-9,hi-lo);
  const x=i=>20+i*(w-32)/Math.max(1,view.channels-1),y=v=>h-16-(v-lo)/range*(h-30);
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#0b1728";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#3b5368";ctx.beginPath();ctx.moveTo(16,y(0));ctx.lineTo(w-8,y(0));ctx.stroke();
  const n=frame.status==="complete"?view.channels:Math.min(view.channels,Math.floor(frame.progress*view.channels)+1);
  const revealed=i=>frame.direction===1?i<n:i>=view.channels-n;
  ctx.strokeStyle="#5ee1e8";ctx.lineWidth=1.5;ctx.beginPath();let started=false;
  for(let i=0;i<view.channels;i++)if(revealed(i)){if(!started){ctx.moveTo(x(i),y(values[i]));started=true;}else ctx.lineTo(x(i),y(values[i]));}
  ctx.stroke();
  for(let i=0;i<view.channels;i++){
    ctx.fillStyle=i===channel()?"#f6c76b":revealed(i)?"#5ee1e8":"#334b5d";
    ctx.beginPath();ctx.arc(x(i),y(values[i]),i===channel()?4:2,0,2*Math.PI);ctx.fill();
  }
  ctx.fillStyle="#9eb8cc";ctx.font="10px Consolas";ctx.fillText("CH 0",8,h-3);ctx.fillText("CH "+(view.channels-1),w-44,h-3);
  $("response-count").textContent=n+" / "+view.channels+" · ["+state.row+","+state.col+"]";
}
function renderProcessingDetail(cursor){
  if(!processing||!cursor)return;
  const ch=Math.min(state.outputChannel,processing.inputChannels-1),r=cursor.inputRow,c=cursor.inputCol;
  $("processing-step").textContent=(cursor.index+1)+" / "+processing.count;
  $("processing-coordinates").textContent=`全部 ${processing.inputChannels} 输入通道窗口 [${r}:${r+processing.kernel}, ${c}:${c+processing.kernel}] → Output[${state.outputChannel},${cursor.outputRow},${cursor.outputCol}]`;
  const candidates=$("processing-candidates");candidates.hidden=state.layer==="conv2";$("processing-stage").textContent="";
  if(state.layer==="conv1"){
    const op=conv1At(data,state.sample,state.outputChannel,cursor.outputRow,cursor.outputCol);
    const actual=tensorValue(output,state.outputChannel,cursor.outputRow,cursor.outputCol);
    $("processing-stage").textContent=`Filter ${state.outputChannel} · 1 个输入通道 · 9 项真实乘积`;
    $("processing-math").textContent=`Σ (${op.products.map(fmt).join(" + ")}) + Bias ${fmt(op.bias)} = Pre-ReLU ${fmt(op.expected)} · 当前输出 ${fmt(actual)} · 误差 ${op.error.toExponential(2)}`;
  }else if(state.layer==="conv2"){
    const op=conv2At(data,state.sample,state.outputChannel,cursor.outputRow,cursor.outputCol,state.inputChannel);
    const actual=tensorValue(output,state.outputChannel,cursor.outputRow,cursor.outputCol);
    $("processing-math").textContent=`32 路贡献 Σ ${fmt(op.calculated-op.bias)} + Bias ${fmt(op.bias)} = Pre-ReLU ${fmt(op.expected)} · 当前输出 ${fmt(actual)} · 误差 ${op.error.toExponential(2)}`;
  }else if(state.layer.startsWith("pool")){
    const a=tensorValue(input,ch,r,c),b=tensorValue(input,ch,r,c+1),d=tensorValue(input,ch,r+1,c+1),cc=tensorValue(input,ch,r+1,c);
    const w1=Math.max(a,b),w2=Math.max(cc,d),result=Math.max(w1,w2),actual=tensorValue(output,ch,cursor.outputRow,cursor.outputCol);
    const vals=[a,b,cc,d],stage=Math.min(3,Math.floor(cursor.phase*4)),cells=[...candidates.children];
    cells.forEach((el,i)=>{el.hidden=false;el.style.opacity="1";el.textContent="ABCD"[i]+" "+fmt(vals[i]);el.className=stage===0&&i<2||stage===1&&i>=2?"comparing":stage>=2&&vals[i]===result?"winner":stage>=2?"eliminated":"";});
    $("processing-stage").textContent=[`① A ${a===b?"=":a>b?">":"<"} B → ${fmt(w1)}`,`② C ${cc===d?"=":cc>d?">":"<"} D → ${fmt(w2)}`,`③ Row1 ${w1===w2?"=":w1>w2?">":"<"} Row2 → ${fmt(result)}`,`④ ${fmt(result)} 写入 Output[${ch},${cursor.outputRow},${cursor.outputCol}]`][stage];
    $("processing-math").textContent=`C${ch}: max(A ${fmt(a)}, B ${fmt(b)}) = ${fmt(w1)}；max(C ${fmt(cc)}, D ${fmt(d)}) = ${fmt(w2)}；max(${fmt(w1)}, ${fmt(w2)}) = ${fmt(result)}；真实输出 ${fmt(actual)}`;
  }else{
    const before=tensorValue(input,ch,r,c),after=tensorValue(output,ch,r,c),visual=before<0?before*(1-cursor.phase):before;
    const cells=[...candidates.children];cells.forEach((el,i)=>{el.hidden=i>0;});cells[0].className=before<0?"relu-negative":"relu-positive";cells[0].style.opacity=before<0?String(1-cursor.phase*.85):"1";cells[0].textContent=`${fmt(before)} → ${fmt(visual)} → ${fmt(after)}`;
    $("processing-stage").textContent=before<0?"负响应逐渐归零；真实 Post-ReLU 始终为 0":"正响应保持不变";
    $("processing-math").textContent=`C${ch}: Pre-ReLU ${fmt(before)} → 视觉过渡 ${fmt(visual)} → max(0, x) = ${fmt(after)}（真实输出）`;
  }
}
function drawAtlas(){
  if(state.view!=="2d")return;
  const view=current(),canvas=$("atlas"),ctx=canvas.getContext("2d"),cols=8,tile=110,rows=Math.ceil(view.channels/cols);
  canvas.width=cols*tile;canvas.height=rows*tile;ctx.fillStyle="#091422";ctx.fillRect(0,0,canvas.width,canvas.height);
  const global=colorScale(view,"global",0);
  for(let ch=0;ch<view.channels;ch++){
    const baseX=ch%cols*tile,baseY=Math.floor(ch/cols)*tile,margin=11,size=82,scale=state.scale==="channel"?colorScale(view,"channel",ch):global;
    ctx.fillStyle=ch===channel()?"#f6c76b":"#7899af";ctx.font="11px Consolas";ctx.fillText("CH "+String(ch).padStart(2,"0"),baseX+margin,baseY+12);
    for(let row=0;row<view.height;row++)for(let col=0;col<view.width;col++){
      const rgba=responseColor(tensorValue(view,ch,row,col),scale);ctx.fillStyle=`rgb(${rgba[0]},${rgba[1]},${rgba[2]})`;
      ctx.fillRect(baseX+margin+col*size/view.width,baseY+18+row*size/view.height,Math.ceil(size/view.width),Math.ceil(size/view.height));
    }
    ctx.strokeStyle=ch===channel()?"#f6c76b":"#36566c";ctx.lineWidth=ch===channel()?2:1;ctx.strokeRect(baseX+margin-.5,baseY+17.5,size+1,size+1);
    ctx.fillStyle="#f6c76b";ctx.fillRect(baseX+margin+(state.col+.5)*size/view.width-2,baseY+18+(state.row+.5)*size/view.height-2,4,4);
  }
}
function renderChannels(){
  const view=current(),chosen=channel(),selector=$("channel"),nav=$("channel-nav");
  if(selector.options.length!==view.channels){selector.replaceChildren();for(let i=0;i<view.channels;i++)selector.add(new Option("Channel "+i,i));}
  selector.value=String(chosen);
  if(nav.children.length!==view.channels){nav.replaceChildren();for(let i=0;i<view.channels;i++){const button=document.createElement("button");button.type="button";button.textContent=String(i).padStart(2,"0");button.onclick=()=>{stopDemo();animation.pause();if(state.side==="input")state.inputChannel=i;else state.outputChannel=i;renderSelection();};nav.append(button);}}
  for(let i=0;i<nav.children.length;i++)nav.children[i].classList.toggle("active",i===chosen);
  $("navigation-title").textContent=view.channels+" "+(state.side==="input"?"input":"output")+" channels";
}
function renderConv2(){
  if(state.layer!=="conv2")return;
  const row=Math.min(state.row,10),col=Math.min(state.col,10);
  const op=conv2At(data,state.sample,state.outputChannel,row,col,state.inputChannel);
  $("filter-label").textContent="Filter "+state.outputChannel+" ← 32 inputs";
  $("sum-label").textContent="Σ contribution = "+fmt(op.calculated-op.bias);
  $("bias-label").textContent="Bias once = "+fmt(op.bias);
  $("output-label").textContent="Output["+state.outputChannel+","+row+","+col+"] = "+fmt(op.expected);
  $("conv2-check").textContent=state.row>10||state.col>10?"所选输入位置超出输出范围；局部计算展示边界内的最近输出坐标。":"PyTorch 对照误差 "+op.error.toExponential(2);
  for(const [id,values] of [["kernel-grid",op.kernel],["patch-grid",op.patch]]){const target=$(id);target.replaceChildren();for(const value of values){const el=document.createElement("span");el.textContent=fmt(value);el.style.color=value<0?"#fb7185":value>0?"#8be3ed":"#9aacc2";target.append(el);}}
  const list=$("contribution-list");list.replaceChildren();for(let i=0;i<32;i++){const b=document.createElement("button");b.type="button";b.textContent="C"+i+" "+op.contributions[i].toFixed(3);b.classList.toggle("active",i===state.inputChannel);b.onclick=()=>{animation.pause();state.inputChannel=i;state.side="input";renderSelection();};list.append(b);}
}
function renderSelection(){
  const view=current(),ch=channel();state.row=Math.min(view.height-1,Math.max(0,state.row));state.col=Math.min(view.width-1,Math.max(0,state.col));
  $("tensor-key").textContent=view.key;
  $("channel-label").textContent=ch+" / "+view.channels;
  $("position-label").textContent="["+state.row+","+state.col+"]";
  $("value-label").textContent=fmt(tensorValue(view,ch,state.row,state.col));
  const stats=channelStats(view,ch);
  $("min-label").textContent="MIN "+fmt(stats.min);$("max-label").textContent="MAX "+fmt(stats.max);$("mean-label").textContent="MEAN "+fmt(stats.mean);
  $("row").max=String(view.height-1);$("col").max=String(view.width-1);$("row").value=String(state.row);$("col").value=String(state.col);
  $("scale-note").textContent=state.scale==="global"?"全通道统一色标":"各通道独立归一化";
  for(const b of document.querySelectorAll("[data-side]"))b.classList.toggle("active",b.dataset.side===state.side);
  scene.setSelection(state.side,ch,state.row,state.col,state.outputChannel);
  renderChannels();drawDetail();drawAtlas();renderConv2();drawResponse();
  const frame=animation.snapshot();scene.applyAnimation(frame.status==="idle"?"idle":frame.mode,frame.progress,state.side,ch,frame.direction);
  if(frame.mode==="spatial"&&lastCursor){scene.setProcessingCursor(lastCursor,state.outputChannel);renderProcessingDetail(lastCursor);}
}
function stopDemo(){demoPlaying=false;demoStart=0;$("demo-filters").textContent="播放 64 Filter 教学顺序";}
function runDemo(timestamp){
  if(!demoPlaying)return;
  if(!demoStart)demoStart=timestamp;const elapsed=timestamp-demoStart;
  state.outputChannel=Math.min(63,Math.floor(elapsed/3600*64));state.side="output";renderSelection();
  if(elapsed>=3600)stopDemo();
}
$("demo-filters").onclick=()=>{if(demoPlaying){stopDemo();return;}animation.stop();demoPlaying=true;demoStart=0;$("demo-filters").textContent="停止教学顺序";};
for(let i=0;i<data.samples.length;i++){const s=data.samples[i];$("sample").add(new Option("Index "+s.index+" · true "+s.groundTruth+" · predict "+s.prediction,i));}
$("activation").value=state.activation;
$("sample").onchange=e=>{stopDemo();animation.stop();state.sample=Number(e.target.value);setViews();};
$("layer").onchange=e=>{stopDemo();animation.stop();state.layer=e.target.value;state.side="output";state.row=state.layer==="conv1"?12:state.layer==="flatten"?0:5;state.col=state.row;state.outputChannel=state.layer==="conv1"?16:state.layer==="pool1"||state.layer==="relu1"?16:state.layer==="flatten"?0:49;setViews(true);if(state.layer==="flatten"){animation.select("flatten");$("anim-mode").value="flatten";}else if(animation.mode==="flatten"){animation.select("spatial");$("anim-mode").value="spatial";}};
$("activation").onchange=e=>{state.activation=e.target.value;setViews();};
$("scale").onchange=e=>{state.scale=e.target.value;setViews();};
$("channel").onchange=e=>{stopDemo();animation.pause();if(state.side==="input")state.inputChannel=Number(e.target.value);else state.outputChannel=Number(e.target.value);renderSelection();};
for(const id of ["row","col"])$(id).onchange=e=>{stopDemo();animation.pause();state[id]=Number(e.target.value);renderSelection();};
for(const button of document.querySelectorAll("[data-side]"))button.onclick=()=>{stopDemo();animation.pause();state.side=button.dataset.side;renderSelection();};
for(const button of document.querySelectorAll("[data-view]"))button.onclick=()=>{state.view=button.dataset.view;$("workspace").dataset.view=state.view;document.body.dataset.view=state.view;for(const b of document.querySelectorAll("[data-view]"))b.classList.toggle("active",b===button);drawAtlas();scene.resize();};
for(const button of document.querySelectorAll("[data-camera]"))button.onclick=()=>scene.setPreset(button.dataset.camera);
$("reset-camera").onclick=()=>scene.setPreset("three-quarter");
$("low-power").onchange=e=>scene.setLowPower(e.target.checked);
$("detail-map").onclick=e=>{animation.pause();const rect=e.target.getBoundingClientRect(),v=current();state.col=Math.max(0,Math.min(v.width-1,Math.floor((e.clientX-rect.left)/rect.width*v.width)));state.row=Math.max(0,Math.min(v.height-1,Math.floor((e.clientY-rect.top)/rect.height*v.height)));renderSelection();};
$("atlas").onclick=e=>{animation.pause();const rect=e.target.getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width*e.target.width,y=(e.clientY-rect.top)/rect.height*e.target.height,ch=Math.floor(y/110)*8+Math.floor(x/110);if(ch<current().channels){if(state.side==="input")state.inputChannel=ch;else state.outputChannel=ch;renderSelection();}};
$("anim-mode").onchange=e=>{stopDemo();if(e.target.value==="flatten"&&state.layer!=="flatten"){state.layer="flatten";$("layer").value="flatten";state.side="output";state.outputChannel=0;state.row=state.col=0;setViews(true);}else if(e.target.value==="spatial"&&!processing){state.layer="conv2";$("layer").value="conv2";state.row=state.col=5;setViews(true);}animation.select(e.target.value);renderSelection();};
$("anim-play").onclick=()=>{stopDemo();animation.play();};
$("anim-pause").onclick=()=>animation.pause();
$("anim-restart").onclick=()=>{stopDemo();animation.restart();};
$("anim-direction").onclick=()=>{animation.setDirection(animation.direction===1?-1:1);$("anim-direction").textContent=animation.direction===1?"正向 0→末":"反向 末→0";};
$("anim-return").onclick=()=>animation.returnSeparated();
$("anim-speed").oninput=e=>{animation.setSpeed(Number(e.target.value));$("anim-speed-label").textContent=e.target.value+"×";};
let lastStatsTime=performance.now(),lastRenderedFrames=0;
const statsTimer=setInterval(()=>{const now=performance.now(),count=scene.renderedFrames;$("fps").textContent="Render "+((count-lastRenderedFrames)*1000/Math.max(1,now-lastStatsTime)).toFixed(1)+" FPS";lastRenderedFrames=count;lastStatsTime=now;},1000);
window.addEventListener("beforeunload",()=>{clearInterval(statsTimer);stopDemo();animation.stop();scene.dispose();},{once:true});
document.body.dataset.view=state.view;
setViews(true);
animation.select("assembly");animation.play();
const follow=document.createElement("button");follow.textContent="跟随 2D 空间进度";follow.type="button";$("anim-mode").parentElement.append(follow);
follow.onclick=()=>{following2D=!following2D;for(const id of ["layer","sample","anim-mode","anim-play","anim-restart","anim-direction"])$(id).disabled=following2D;animation.pause();follow.textContent=following2D?"退出 2D 跟随":"跟随 2D 空间进度";};
window.DigitVisionSpatialSync?.subscribe(s=>{
 if(s.connected===false){$("anim-status").textContent="2D DISCONNECTED · 已冻结最后画面";return;}
 if(!following2D||!OPERATIONS[s.layer])return;
 const changed=state.layer!==s.layer||state.sample!==s.sample;
 state.layer=s.layer;state.sample=s.sample;state.activation="pre";state.side="output";state.outputChannel=s.selectedOutputChannel;state.inputChannel=s.selectedChannel;
 if(changed)setViews();
 for(const id of ['layer','sample','anim-mode','anim-play','anim-restart','anim-direction']){$(id).disabled=true;$(id).title='2D 跟随中，由主控决定播放状态';}
 $("layer").value=state.layer;$("sample").value=String(state.sample);
 animation.mode="spatial";animation.progress=s.progress;animation.status=s.playbackState==="PLAYING"?"playing":"paused";
 state.row=s.outputRow;state.col=s.outputCol;
 lastCursor={...s,activeChannels:processing.inputChannels};renderSelection();scene.setProcessingCursor(lastCursor,state.outputChannel);renderProcessingDetail(lastCursor);
 $("anim-status").textContent="2D FOLLOW · "+s.playbackState+" · "+Math.round(s.progress*100)+"%";
});
window.TensorExplorer={snapshot:()=>({state:{...state},animation:animation.snapshot(),cursor:lastCursor?{...lastCursor}:null,inputShape:input?.tensor.shape||null,outputShape:output.tensor.shape,selectedValue:tensorValue(current(),channel(),state.row,state.col),scene:scene.metrics()}),setSample:i=>{animation.stop();state.sample=i;$("sample").value=String(i);setViews();},setLayer:l=>{animation.stop();state.layer=l;$("layer").value=l;state.side="output";state.row=state.col=l==="conv1"?12:5;setViews(true);},setCoordinate:(r,c)=>{animation.pause();state.row=r;state.col=c;renderSelection();},setChannel:(c,side="output")=>{animation.pause();state.side=side;if(side==="input")state.inputChannel=c;else state.outputChannel=c;renderSelection();},setView:v=>document.querySelector(`[data-view="${v}"]`).click(),setScale:v=>{state.scale=v;$("scale").value=v;setViews();},animation,camera:scene.camera,sceneManager:scene,dispose:()=>{clearInterval(statsTimer);animation.stop();scene.dispose();}};




