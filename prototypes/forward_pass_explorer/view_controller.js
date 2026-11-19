/* Presentation ownership only: no inference, clock, or tensor mutation. */
(function(root){
"use strict";
function create(host){
 const {S}=host,$=id=>document.getElementById(id),stage=$("stage"),grid=stage.querySelector(".stage-grid"),inspector=document.querySelector(".inspector"),overview=$("multi-channel-overview"),fusion=$("mc-fusion"),fusionHome=fusion.parentNode;
 let view="main",layer="",changing=false;
 const element=(tag,id,parent)=>{const el=document.createElement(tag);el.id=id;parent.append(el);return el;};
 const tabs=element("nav","explorer-views",stage);tabs.setAttribute("aria-label","观察视图");tabs.setAttribute("role","tablist");
 for(const [key,label] of [["main","主视图"],["parallel","并行视图"],["details","查看计算详情"]]){const b=element("button","view-"+key,tabs);b.textContent=label;b.type="button";b.setAttribute("role","tab");b.setAttribute("aria-controls",key+"-view");b.onclick=()=>setView(key);b.onkeydown=e=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;e.preventDefault();const keys=["main","parallel","details"],i=keys.indexOf(view),next=e.key==="Home"?0:e.key==="End"?2:(i+(e.key==="ArrowRight"?1:2))%3;setView(keys[next]);$("view-"+keys[next]).focus();};}
 const panels={};for(const key of ["main","parallel","details"]){panels[key]=element("section",key+"-view",stage);panels[key].setAttribute("role","tabpanel");panels[key].setAttribute("aria-labelledby","view-"+key);}
 panels.main.append(grid);panels.parallel.append(overview);
 const empty=element("p","parallel-empty",panels.parallel);empty.innerHTML='<h3 id="parallel-title"></h3><p id="parallel-explanation"></p><canvas id="parallel-vector" width="1000" height="280" aria-label="真实向量与分类响应"></canvas>';
 const detailVisual=element("div","detail-visual",panels.details),fusionDetail=element("div","detail-fusion",panels.details);panels.details.append(inspector);
 const verify=element("div","detail-verification",panels.details);verify.append($("consistency"));
 const shelf=element("div","playback-shelf",stage);
 for(const node of [document.querySelector(".selection-bar"),$("conv1-mode-controls"),$("layer-animation-controls"),document.querySelector(".operation-toolbar")])shelf.append(node);
 const badge=element("span","observation-context",document.querySelector(".selection-bar"));
 const archive=element("details","channel-archive",panels.details);const summary=element("summary","channel-archive-title",archive);summary.textContent="通道快捷选择";archive.append($("channel-strip-wrap"));
 const mainFusion=element("article","main-fusion",grid);const title=element("h3","fusion-focus-title",mainFusion);title.textContent="32 路局部贡献 → 一次加权汇聚";
 const mainFusionCanvas=element("div","main-fusion-canvas",mainFusion);const fusion64=element("div","fusion64",mainFusion);fusion64.innerHTML="<strong>一套 Filter → 一张输出图</strong><p>64 套训练得到的不同权重，都读取全部 32 个输入通道。</p><div>Filter 0 → Map 0　 Filter 1 → Map 1　 …　 Filter 63 → Map 63</div>";
 const weights=element("details","fusion-weight-details",fusionHome),ws=element("summary","fusion-weight-title",weights);ws.textContent="展开 32 组真实 3×3 权重";weights.append($("mc-weights"));weights.addEventListener("toggle",()=>host.refreshParallel());
 const weighted=element("details","weighted-input-details",panels.details),weightedTitle=element("summary","weighted-input-title",weighted);weightedTitle.textContent="查看分组加权贡献 · 点击选择输入组";weighted.append($("contributions"));
 const global=document.querySelector(".global-controls"),globalDisclosure=document.createElement("details");globalDisclosure.id="network-playback";global.before(globalDisclosure);const gs=element("summary","network-playback-title",globalDisclosure);gs.textContent="完整网络播放与层间控制";globalDisclosure.append(global);
 $("mc-detail").textContent="返回主视图 · 保留选中通道";$("mc-detail").onclick=()=>{setView("main");$("view-main").focus();};
 const descriptions={input:["输入图像","将 28×28 灰度图保留为真实像素 Tensor。"],conv1:["Conv1 · 寻找局部模式","3×3 卷积核滑过输入图像，形成当前通道的响应。"],relu1:["ReLU1 · 保留正向响应","负数截断为零，正数保持不变，为网络引入非线性。"],pool1:["Pool1 · 保留局部最强响应","每个通道独立比较 2×2 区域，只将最大值传递到输出。"],conv2:["Conv2 · 融合多通道信息","32 个输入通道分别产生加权贡献，求和后加入一次 Bias。"],relu2:["ReLU2 · 抑制负向响应","负数归零，正向响应保持不变。"],pool2:["Pool2 · 压缩空间尺寸","每个 2×2 区域保留最强响应，通道之间互不混合。"],flatten:["Flatten · 从空间到向量","保持数值与顺序，将 64×5×5 的 Tensor 展开为 1600 维向量。"],fc1:["FC1 · 组合已提取的特征","将输入特征加权组合，再通过 ReLU 形成隐藏层响应。"],output:["Output · 从响应到预测","十类 Logits 转换为 Softmax 概率，最高概率对应预测数字。"]};
 /** Reparent existing components without seeking, pausing, or selecting anything. */
 function layout(){
  const key=S.current.key;document.body.dataset.explorerView=view;document.body.dataset.explorerLayer=key;
  for(const k of Object.keys(panels)){panels[k].hidden=k!==view;$("view-"+k).setAttribute("aria-selected",String(k===view));$("view-"+k).tabIndex=k===view?0:-1;}
  (view==="details"&&["pool1","output"].includes(key)?detailVisual:panels.main).append(grid);
  if(key==="conv2"&&view==="main")mainFusionCanvas.append(fusion);else fusionHome.insertBefore(fusion,$("mc-fusion-summary"));
  (key==="conv2"&&view==="details"?fusionDetail:$("mc-body")).append(fusionHome);
  mainFusion.hidden=key!=="conv2";fusionDetail.hidden=key!=="conv2";
  weighted.hidden=!["fc1","output"].includes(key);
  const t=S.sample.tensors[S.current.tensor];empty.hidden=t.shape.length===4&&t.shape[1]>1;
  host.invalidate();host.refreshParallel();if(view==="parallel")paintParallel();
 }
 function paintParallel(){const key=S.current.key,canvas=$("parallel-vector"),ctx=canvas.getContext("2d"),note=$("parallel-explanation"),title=$("parallel-title");const specs={input:["Single-channel Input","当前输入只有一个灰度通道；无需多通道 Feature Map。","input"],flatten:["64 张特征图 → 1600 维向量","每个真实 5×5 通道按行展开为连续 25 个值。","flatten"],fc1:["128 个隐藏神经元","这些是向量神经元，不是空间 Feature Maps。","fc1_relu"],output:["十个分类类别","按数字 0–9 对齐真实 Logits 与 Softmax 概率。","probabilities"]};const spec=specs[key];if(!spec)return;title.textContent=spec[0];note.textContent=spec[1];const values=S.sample.tensors[spec[2]].values;ctx.fillStyle="#0a1424";ctx.fillRect(0,0,canvas.width,canvas.height);if(key==="input"){const side=224,x=388,y=28;for(let r=0;r<28;r++)for(let c=0;c<28;c++){const v=values[r*28+c];ctx.fillStyle=`rgb(${Math.round(25+v*75)},${Math.round(48+v*125)},${Math.round(76+v*175)})`;ctx.fillRect(x+c*8,y+r*8,8,8);}ctx.fillStyle="#a8b9d0";ctx.font="14px Consolas";ctx.fillText("1 channel · 28×28 real pixels",x+6,270);canvas.dataset.tensor=spec[2];canvas.dataset.values=JSON.stringify(values);return;}const n=key==="input"?1:key==="flatten"?64:key==="fc1"?128:10;const max=Math.max(...values.map(Math.abs),1e-9);for(let i=0;i<n;i++){let value=key==="flatten"?values.slice(i*25,i*25+25).reduce((a,b)=>a+Math.abs(b),0)/25:key==="input"?Math.max(...values):values[i];let x=28+i*(canvas.width-56)/n,bh=Math.max(2,Math.abs(value)/max*180);ctx.fillStyle=i===S.selection.neuron||key==="output"&&i===S.sample.prediction?"#f6c76b":value<0?"#fb7185":"#5ee1e8";ctx.fillRect(x,220-bh,Math.max(3,(canvas.width-56)/n-3),bh);if(n<=64){ctx.fillStyle="#a8b9d0";ctx.font="11px Consolas";ctx.fillText(String(i),x,246);}}canvas.dataset.tensor=spec[2];canvas.dataset.values=JSON.stringify(values);}
 function setView(next){if(!["main","parallel","details"].includes(next))throw RangeError("Unknown observation view");if(changing||view===next)return;view=next;changing=true;layout();host.render();changing=false;}
 /** Cheap per-render presentation update; does not recreate tensor DOM. */
 function update(){
  if(layer!==S.current.key){layer=S.current.key;view="main";layout();}
  const text=descriptions[S.current.key];if(text){$("stage-title").textContent=text[0];$("stage-description").textContent=text[1];}
  const context="通道 "+S.selection.out+" · 坐标 ["+S.selection.row+", "+S.selection.col+"]";if(badge.textContent!==context)badge.textContent=context;
  fusion64.hidden=!(S.current.key==="conv2"&&document.body.dataset.globalStage==="S05"&&Number(document.body.dataset.globalProgress)>=65);if(view==="parallel")paintParallel();
  if(S.current.key==="pool1"&&$("pool-tournament")){
   const tournament=$("pool-tournament"),s=JSON.parse(tournament.dataset.state||"null");
   if(s){let summary=$("pool-round-summary");if(!summary){summary=element("p","pool-round-summary",tournament);tournament.querySelector(".pool-replay").before(summary);}
    const row=s.stage<=3?0:s.stage<=5?1:2,labels=["A vs B","C vs D","Winner 1 vs Winner 2"],numbers=row===0?s.values.slice(0,2):row===1?s.values.slice(2,4):[s.row1,s.row2];
    const display=n=>Number(n.toPrecision(5)).toString(),left=display(numbers[0]),right=display(numbers[1]),format=left===right&&numbers[0]!==numbers[1]?String:display;
    const text=s.stage===0?"四个候选值 → 两组比较 → 最终最大值":labels[row]+" · "+format(numbers[0])+" "+s.relations[row]+" "+format(numbers[1])+(s.relations[row]==="="?" · 并列最大值":"");if(summary.textContent!==text)summary.textContent=text;
   }
  }
 }
 return {update,setView,snapshot:()=>({view,layer:S.current.key})};
}
root.DigitVisionViews={create};
})(window);

