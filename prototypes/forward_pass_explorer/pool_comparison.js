/* Pool1-only tournament teaching view. All values are real selected tensor values. */
(function(root){
"use strict";
const LETTERS=["A","B","C","D"],clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
/** Preserve every tied winner; firstIndex follows PyTorch's row-major tie policy. */
function tournament(values){
 if(!Array.isArray(values)||values.length!==4||values.some(v=>!Number.isFinite(v)))throw new TypeError("Expected four finite Pool1 values");
 const row1=Math.max(values[0],values[1]),row2=Math.max(values[2],values[3]),maximum=Math.max(row1,row2);
 const winners1=[0,1].filter(i=>values[i]===row1),winners2=[2,3].filter(i=>values[i]===row2),winners=[0,1,2,3].filter(i=>values[i]===maximum);
 return {values:[...values],row1,row2,maximum,winners1,winners2,winners,firstIndex:winners[0],relations:[values[0]<values[1]?"<":values[0]>values[1]?">":"=",values[2]<values[3]?"<":values[2]>values[3]?">":"=",row1<row2?"<":row1>row2?">":"="]};
}
/** Nine stable states: initial plus the eight user-controlled teaching steps. */
function comparisonState(values,stage){
 const t=tournament(values);stage=clamp(Math.floor(stage),0,8);
 const comparing=stage===2?[0,1]:stage===4?[2,3]:stage===6?[...t.winners1,...t.winners2]:[];
 const eliminated=[];if(stage>=3)for(const i of [0,1])if(values[i]<t.row1)eliminated.push(i);if(stage>=5)for(const i of [2,3])if(values[i]<t.row2)eliminated.push(i);if(stage>=7)for(let i=0;i<4;i++)if(values[i]<t.maximum&&!eliminated.includes(i))eliminated.push(i);
 const current=stage===0?null:stage<=2?values[0]:stage<=4?t.row1:stage===5?t.row2:stage===6?t.row1:t.maximum;
 const label=stage===0?"MAX POOLING":stage<=2?"CURRENT MAXIMUM":stage<=4?"ROW 1 WINNER":stage===5?"ROW 2 WINNER":stage===6?"COMPARING WINNERS":"FINAL MAXIMUM";
 return {...t,stage,comparing,eliminated,current,label,final:stage>=7,write:stage===8};
}
const BOUNDS=[0,.06,.12,.25,.37,.50,.62,.77,.86,1];
function timeline(progress){
 progress=clamp(progress,0,1);let stage=0;while(stage<8&&progress>=BOUNDS[stage+1])stage++;
 return {stage,progress:progress===1?1:clamp((progress-BOUNDS[stage])/(BOUNDS[stage+1]-BOUNDS[stage]),0,1)};
}
function create(panel,options){
 const format=options.format,cells=[...panel.querySelectorAll(".pool-value")],box=panel.querySelector(".pool-winner"),number=box.querySelector("strong"),label=box.querySelector("span");
 const bracket=document.createElement("div");bracket.id="pool-tournament";
 bracket.innerHTML='<div class="pool-demo-caption"></div><div class="pool-row-matches">'+["A vs B","C vs D"].map((name,i)=>'<section class="pool-match" data-match="'+i+'"><span>'+name+'</span><div class="pool-relation">尚未比较</div><div class="pool-row-winner"><small>Winner '+(i+1)+'</small><strong>—</strong></div></section>').join("")+'</div><div class="pool-final-match"><span>Winner 1</span><b>VS</b><span>Winner 2</span><div class="pool-final-relation">等待两组胜出者</div></div><button class="button ghost pool-replay" type="button">↻ 慢放当前窗口的三轮比较</button>';
 box.before(bracket);const meta=document.createElement("small");meta.className="pool-winner-meta";box.append(meta);
 const matches=[...bracket.querySelectorAll(".pool-match")],slots=[...bracket.querySelectorAll(".pool-row-winner")],finalMatch=bracket.querySelector(".pool-final-match"),finalRelation=bracket.querySelector(".pool-final-relation"),caption=bracket.querySelector(".pool-demo-caption"),replay=bracket.querySelector(".pool-replay");
 replay.addEventListener("click",options.replay);let geometry=null;
 function text(el,value){if(el.textContent!==value)el.textContent=value;}
 function relation(a,b,operator){let left=format(a,5),right=format(b,5);if(a!==b&&left===right){left=String(a);right=String(b);}return left+" "+operator+" "+right+(operator==="="?" · 并列":"");}
 function dimensions(){if(geometry)return geometry;const rect=panel.getBoundingClientRect();geometry={cells:cells.map(el=>{const r=el.getBoundingClientRect();return {x:r.left+r.width/2-rect.left,y:r.top+r.height/2-rect.top};}),slots:slots.map(el=>{const r=el.getBoundingClientRect();return {x:r.left+r.width/2-rect.left,y:r.top+r.height/2-rect.top};})};return geometry;}
 function render(values,view){
  const s=comparisonState(values,view.stage),p=clamp(view.progress,0,1),g=dimensions();
  cells.forEach((el,i)=>{
   text(el.querySelector("small"),LETTERS[i]+" · ["+(view.row*2+Math.floor(i/2))+","+(view.col*2+i%2)+"]");
   el.classList.remove("comparing","winner","compared");
   el.classList.toggle("is-comparing",s.comparing.includes(i));el.classList.toggle("is-selected",s.stage===1&&i===0);
   el.classList.toggle("is-eliminated",s.eliminated.includes(i));el.classList.toggle("is-final",s.final&&s.winners.includes(i));
   el.classList.toggle("is-candidate",!s.final&&(s.stage>=3&&s.winners1.includes(i)||s.stage>=5&&s.winners2.includes(i)));
  });
  const pairValues=[[values[0],values[1]],[values[2],values[3]]],results=[s.row1,s.row2],winnerIndices=[s.winners1,s.winners2];
  matches.forEach((el,i)=>{
   const compareStage=i?4:2,resultStage=i?5:3,ready=s.stage>=resultStage,active=s.stage===compareStage,slot=slots[i];
   el.classList.toggle("is-comparing",active);el.dataset.status=active?(p<.22?"start":"compare"):ready?"result":"waiting";
   text(el.querySelector(".pool-relation"),s.stage>=compareStage&&(!active||p>=.22)?relation(...pairValues[i],s.relations[i]):active?"开始比较 · VS":"尚未比较");
   text(slot.querySelector("strong"),ready?format(results[i],7):"—");slot.classList.toggle("is-ready",ready);slot.classList.toggle("is-comparing",s.stage===6);slot.classList.toggle("is-final",s.final&&results[i]===s.maximum);
   // The row winner travels from its real A/B/C/D card into its winner slot.
   const enter=ready?(s.stage===resultStage?p:1):0,origin=g.cells[winnerIndices[i][0]],destination=g.slots[i];
   const converge=s.stage>=6?(s.stage===6?p:1):0;
   slot.style.setProperty("--pool-origin-x",origin.x-destination.x+"px");slot.style.setProperty("--pool-origin-y",origin.y-destination.y+"px");slot.classList.toggle("step-enter",view.step&&s.stage===resultStage);slot.style.transition=view.step?"opacity .2s":"none";
   slot.style.transform="translate("+((origin.x-destination.x)*(ready?1-enter:0)+(i? -12:12)*converge)+"px,"+((origin.y-destination.y)*(ready?1-enter:0)+8*converge)+"px)";
   slot.style.opacity=ready?String(.25+.75*enter):".35";
  });
  finalMatch.classList.toggle("is-comparing",s.stage===6);finalMatch.classList.toggle("is-final",s.final);
  text(finalRelation,s.stage>=6&&(s.stage!==6||p>=.22)?relation(s.row1,s.row2,s.relations[2]):s.stage===6?"两组胜出者进入最终比较":"等待两组胜出者");
  text(label,s.label);text(number,s.current===null?"—":format(s.current,8));box.classList.toggle("is-final",s.final);box.classList.toggle("is-comparing",s.comparing.length>0);
  const origins=s.winners.map(i=>LETTERS[i]+"["+(view.row*2+Math.floor(i/2))+","+(view.col*2+i%2)+"]").join("、");
  text(meta,s.final?(s.winners.length>1?"并列最大：":"原始位置：")+origins+" · "+(s.stage===8?"仅传递 "+LETTERS[s.firstIndex]+" 的最大值":"三轮比较完成"):s.stage===0?"等待四个候选值参与比较":s.stage<=2?"选中 A，比较 A 与 B":s.stage<=4?"上排胜出者已确定；继续比较 C 与 D":s.stage===5?"下排胜出者已确定；等待最终比较":"Winner 1 vs Winner 2");
  text(caption,view.overview?"快速扫描 ∥ 教学窗口 ["+view.row+","+view.col+"] 慢放":view.inspection?"独立比较演示 · 扫描进度保持不变":"当前 2×2 窗口的淘汰赛比较");
  replay.hidden=view.step;replay.disabled=view.inspectPlaying;
  bracket.dataset.state=JSON.stringify({...s,row:view.row,col:view.col,phaseProgress:p,overview:view.overview,inspection:view.inspection});
  return s;
 }
 return {render,invalidateGeometry:()=>{geometry=null;},root:bracket};
}
const API={tournament,comparisonState,timeline,create};if(typeof module!=="undefined"&&module.exports)module.exports=API;root.DigitVisionPoolComparison=API;
})(typeof window!=="undefined"?window:globalThis);
