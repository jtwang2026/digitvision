/* One global RAF and timestamp-driven 13-stage teaching timeline. */
(function(root){"use strict";
const data=root.DigitVisionStageRegistry,stages=data.stages,total=stages.reduce((n,s)=>n+s.duration,0);
if(stages.length!==13||total!==70000)throw Error("Invalid stage contract");
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function locate(ms){ms=clamp(ms,0,total);let start=0;for(let i=0;i<stages.length;i++){let end=start+stages[i].duration;if(ms<end||i===stages.length-1)return {index:i,stage:stages[i],local:Math.min(ms-start,stages[i].duration),progress:Math.min(1,(ms-start)/stages[i].duration),start,end};start=end;}}
function create(host){
 const {S,DATA}=host,$=id=>document.getElementById(id),container=$("stage"),tabs=$("explorer-views");
 const band=document.createElement("div");band.id="global-story";band.innerHTML='<div class="global-story-top"><span id="story-stage"></span><strong id="story-phase"></strong><span id="story-time"></span></div><div class="global-story-track"><i id="story-fill"></i></div><p id="story-handoff"></p><canvas id="story-canvas" width="1080" height="260" aria-label="真实前向传播阶段动画"></canvas><div class="global-story-controls"><button type="button" id="story-play">▶ Play All</button><button type="button" id="story-pause" disabled>Ⅱ Pause</button><button type="button" id="story-restart">↺ Global Restart</button><button type="button" id="story-previous">← 阶段</button><button type="button" id="story-next">阶段 →</button><button type="button" id="story-inspect">检查当前计算</button><button type="button" id="story-resume" hidden>Resume Global Playback</button><button type="button" id="story-return" hidden>返回主视图</button><label>速度 <input id="story-speed" type="range" min="0.5" max="2" step="0.5" value="1"><output id="story-speed-label">1×</output></label></div>';
 tabs.before(band);const canvas=$("story-canvas"),ctx=canvas.getContext("2d");
 let elapsed=0,state="IDLE",raf=0,last=null,entered=-1,frameCount=0,intervals=[],drawMs=[],inspection=false,savedElapsed=null;
 function tensor(key){return S.sample.tensors[key]||null;}
 function title(s){return s.stageId+" · "+s.stageName;}
 function draw(stage,progress){
  let start=performance.now(),t=tensor(stage.outputTensorKey),input=tensor(stage.inputTensorKey),w=canvas.width,h=canvas.height;
  ctx.fillStyle="#0a1424";ctx.fillRect(0,0,w,h);ctx.font="14px Consolas";ctx.textAlign="left";ctx.fillStyle="#9bb2cf";
  ctx.fillText(stage.inputTensorKey+"  "+stage.inputShape.join("×"),20,28);ctx.textAlign="right";ctx.fillText(stage.outputTensorKey+"  "+stage.outputShape.join("×"),w-20,28);ctx.textAlign="center";ctx.fillStyle="#5ee1e8";ctx.font="18px Consolas";ctx.fillText(stage.animationType.toUpperCase().replaceAll('-',' '),w/2,29);
  const values=t?.values||S.sample.tensors.probabilities.values,src=input?.values||values;
  if(stage.stageId==="S01"){const px=S.sample.tensors.input.values,side=196,x=65,y=48;for(let r=0;r<28;r++)for(let c=0;c<28;c++){const v=px[r*28+c];ctx.fillStyle=`rgb(${Math.round(40+v*170)},${Math.round(70+v*150)},${Math.round(110+v*145)})`;ctx.fillRect(x+c*side/28,y+r*side/28,side/28-.2,side/28-.2);}const pos=Math.min(783,Math.floor(progress*784)),row=Math.floor(pos/28),col=pos%28,v=px[pos];ctx.strokeStyle="#f6c76b";ctx.lineWidth=2;ctx.strokeRect(x+col*side/28,y+row*side/28,side/28,side/28);ctx.fillStyle="#d6e6f7";ctx.font="17px Consolas";ctx.textAlign="left";ctx.fillText("MNIST test #"+S.sample.index+" · real 28×28 grayscale",310,82);ctx.fillText("pixel["+row+","+col+"] = "+v.toFixed(6),310,125);ctx.fillText("uint8 ≈ "+Math.round(v*255)+" → /255 → Float32",310,166);ctx.fillText("[C,H,W] → [B,C,H,W] = [1,1,28,28]",310,209);drawMs.push(performance.now()-start);return;}
  if(stage.stageId==="S13"){const probs=S.sample.tensors.probabilities.values,truth=S.sample.groundTruth,pred=S.sample.prediction;ctx.fillStyle="#a9bdd6";ctx.font="15px Consolas";ctx.textAlign="left";ctx.fillText("Classification Result · real probabilities",35,55);for(let i=0;i<10;i++){let y=76+i*14;ctx.fillStyle="#9aafc8";ctx.textAlign="right";ctx.fillText(String(i),65,y+9);ctx.fillStyle="#1b2940";ctx.fillRect(83,y,560,10);ctx.fillStyle=i===pred?"#34d399":"#64748b";ctx.fillRect(83,y,560*probs[i]*Math.min(1,progress*2),10);ctx.textAlign="left";ctx.fillStyle="#b6c9df";ctx.fillText(probs[i]*100<0.01?(probs[i]*100).toExponential(2)+"%":(probs[i]*100).toFixed(2)+"%",660,y+9);}ctx.textAlign="left";ctx.font="38px system-ui";ctx.fillStyle="#34d399";if(progress>.55)ctx.fillText(String(pred),855,143);ctx.font="15px system-ui";ctx.fillStyle="#a9bdd6";if(progress>.75)ctx.fillText("预测 "+pred+" · 真实 "+truth+" · "+(pred===truth?"正确":"错误"),825,186);drawMs.push(performance.now()-start);return;}
  const max=Math.max(...values.map(Math.abs),1e-9),count=stage.stageId==='S02'?676:stage.stageId==='S04'?169:stage.stageId==='S07'?25:stage.stageId==='S08'?64:stage.stageId==='S05'?32:Math.min(values.length,128);
  const shown=clamp(Math.floor(progress*count),0,count),left=50,right=w-50,top=66,bottom=218,bars=Math.min(count,128),gap=(right-left)/bars;
  for(let i=0;i<bars;i++){
   const index=Math.floor(i*count/bars),v=values[index%values.length],positive=v>=0;
   const amplitude=Math.max(2,Math.abs(v)/max*115);ctx.fillStyle=positive?"#34d399":"#fb7185";ctx.globalAlpha=index<shown?.82:.16;
   ctx.fillRect(left+i*gap,positive?bottom-amplitude:bottom-2,Math.max(2,gap-2),amplitude);ctx.globalAlpha=1;
  }
  if(stage.stageId==='S05'){
   const o=S.selection.out,r=clamp(S.selection.row,0,10),c=clamp(S.selection.col,0,10),op=root.DigitVisionForwardMath.convAt(DATA,S.sample,'conv2',o,r,c);
   ctx.strokeStyle="#5ee1e8";ctx.lineWidth=1;for(let k=0;k<32;k++){let y=56+k*5;ctx.globalAlpha=k<shown?1:.2;ctx.beginPath();ctx.moveTo(360,y);ctx.lineTo(530,135);ctx.stroke();}ctx.globalAlpha=1;ctx.fillStyle="#f6c76b";ctx.fillText("Σ 32 contributions + bias "+op.bias.toFixed(5)+" → "+op.expected.toFixed(5),w/2,248);
  }else if(stage.stageId==='S04'||stage.stageId==='S07'){
   const key=stage.stageId==='S04'?'pool1':'pool2',side=key==='pool1'?13:5,coord=Math.min(Math.floor(progress*side*side),side*side-1),op=root.DigitVisionForwardMath.poolAt(S.sample,key,S.selection.out,Math.floor(coord/side),coord%side);
   ctx.fillStyle="#f6c76b";ctx.fillText("max("+op.values.map(v=>v.toFixed(3)).join(', ')+") = "+op.expected.toFixed(3),w/2,248);
  }else if(stage.stageId==='S08'){
   let j=Math.min(1599,shown*25);ctx.fillStyle="#f6c76b";ctx.fillText("channel "+Math.floor(j/25)+" · vector["+j+"] = "+values[j].toFixed(5),w/2,248);
  }else if(stage.stageId==='S10'){
   ctx.fillStyle="#34d399";ctx.fillText("eval() · Dropout(p=0.5) = Identity · 128 values unchanged",w/2,248);
  }else if(stage.stageId==='S12'){
   ctx.fillStyle="#34d399";ctx.fillText("exp(logit − max) / Σ exp(logit − max) · Σp = "+values.reduce((a,b)=>a+b,0).toFixed(6),w/2,248);
  }else if(stage.stageId==='S13'){
   ctx.fillStyle="#34d399";ctx.fillText("prediction "+S.sample.prediction+" · true label "+S.sample.groundTruth,w/2,248);
  }else{ctx.fillStyle="#8fb9d0";ctx.fillText("real tensor · "+values.length+" values · "+shown+" / "+count+" revealed",w/2,248);}
  drawMs.push(performance.now()-start);
 }
 function sync(force=false){let p=locate(elapsed),s=p.stage;
  if(force||entered!==p.index){entered=p.index;host.enter(s.uiNode,s);}
  document.body.dataset.globalStage=s.stageId;document.body.dataset.globalProgress=Math.floor(p.progress*100);host.phase(s,p.progress,p.local,state);
  $('story-stage').textContent=title(s);$('story-phase').textContent=(inspection?'LOCAL INSPECTION · ':'')+s.animationType+' · '+Math.round(p.progress*100)+'%';$('story-time').textContent=(elapsed/1000).toFixed(1)+' / 70.0 s · 教学动画';$('story-fill').style.width=(elapsed/total*100)+'%';
  $('story-handoff').textContent=s.inputTensorKey+' → '+s.outputTensorKey+' · '+(s.canonicalStageId||'模型外 Argmax')+' · '+S.sample.index+'号真实样本';
  $('story-play').disabled=state==='PLAYING'||state==='COMPLETED';$('story-pause').disabled=state!=='PLAYING';
  band.dataset.state=state;band.dataset.stage=s.stageId;document.body.dataset.globalStage=s.stageId;document.body.dataset.globalProgress=Math.floor(p.progress*100);band.dataset.mode=inspection?'local':'global';document.body.dataset.globalMode=inspection?'local':'global';$('story-play').textContent=state==='PAUSED'?'▶ Resume':'▶ Play All';$('story-inspect').hidden=inspection;$('story-resume').hidden=!inspection;$('story-return').hidden=!inspection;draw(s,p.progress);
 }
 function frame(now){raf=0;if(state!=='PLAYING')return;if(last!==null){let dt=Math.max(0,Math.min(50,now-last));elapsed=Math.min(total,elapsed+dt*Number($('story-speed').value));intervals.push(dt);}last=now;frameCount++;if(elapsed>=total){state='COMPLETED';last=null;sync();return;}sync();raf=requestAnimationFrame(frame);}
 function pause(){if(state!=='PLAYING')return;state='PAUSED';cancelAnimationFrame(raf);raf=0;last=null;sync();}
 function play(){if(inspection)return resumeInspection();if(state==='COMPLETED'||state==='PLAYING')return;if(host.stepMode()){pause();return;}state='PLAYING';last=null;sync();raf=requestAnimationFrame(frame);}
 function seek(ms){inspection=false;savedElapsed=null;cancelAnimationFrame(raf);raf=0;last=null;elapsed=clamp(ms,0,total);state=elapsed===total?'COMPLETED':'PAUSED';sync();}
 function restart(){seek(0);state='IDLE';sync();}
 function move(delta){let p=locate(elapsed);seek(delta<0?p.index===0?0:locate(p.start-1).start:p.index===12?total:p.end);}
 function sampleChanged(){restart();entered=-1;sync(true);}
 function jumpUi(key){let p=stages.findIndex(v=>v.uiNode===key);if(p<0)return;seek(stages.slice(0,p).reduce((n,v)=>n+v.duration,0));}
 function inspect(){if(inspection)return;savedElapsed=elapsed;pause();inspection=true;sync();}
 function returnInspection(resume=false){if(!inspection)return;inspection=false;const prior=savedElapsed;savedElapsed=null;seek(prior);host.restore();if(resume)play();}
 function resumeInspection(){returnInspection(true);}
 function snapshot(){const p=locate(elapsed);return {state,mode:inspection?"LOCAL_INSPECTION":"GLOBAL",elapsed,stageId:p.stage.stageId,uiNode:p.stage.uiNode,subStage:p.stage.stageName,localElapsed:p.local,progress:p.progress,view:host.view(),sample:S.sample.index,selection:{...S.selection},scanCursor:p.progress,revealCount:Math.floor(p.progress*(p.stage.stageId==='S02'?676:p.stage.stageId==='S04'?169:p.stage.stageId==='S07'?25:100))};}
 $('story-inspect').onclick=inspect;$('story-resume').onclick=resumeInspection;$('story-return').onclick=()=>returnInspection(false);$('story-play').onclick=play;$('story-pause').onclick=pause;$('story-restart').onclick=restart;$('story-previous').onclick=()=>move(-1);$('story-next').onclick=()=>move(1);$('story-speed').oninput=()=>{$('story-speed-label').textContent=$('story-speed').value+'×';};
 document.addEventListener('visibilitychange',()=>{last=null;});
 sync(true);return {play,pause,restart,seek,move,jumpUi,sampleChanged,inspect,returnInspection,snapshot,report:()=>({frames:frameCount,intervals:[...intervals],drawMs:[...drawMs]})};
}
root.DigitVisionGlobalEngine={create,locate,stages,total};if(typeof module!=='undefined'&&module.exports)module.exports={locate,stages,total};
})(typeof window!=='undefined'?window:globalThis);


