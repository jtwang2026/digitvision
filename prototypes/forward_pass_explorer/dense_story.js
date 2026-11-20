/* Pure rendering: progress comes from the existing global engine, never a new clock. */
(function(root){'use strict';
const clamp=x=>Math.max(0,Math.min(1,x));
function create(host){
 const {S,DATA,denseAt}=host;let cache=null;
 const panel=document.createElement('section');panel.id='dense-story';panel.hidden=true;
 panel.innerHTML='<div class="dense-story-heading"><strong></strong><span></span></div><canvas width="1120" height="410" aria-label="真实全连接加权汇聚与十类预测动画"></canvas><p class="dense-story-status" aria-live="off"></p><p class="dense-story-note">连线按真实乘积分组；高亮顺序和移动标记只用于教学，不代表模型实际串行执行或推理耗时。</p>';
 document.getElementById('main-view').prepend(panel);
 const canvas=panel.querySelector('canvas'),ctx=canvas.getContext('2d'),title=panel.querySelector('strong'),shape=panel.querySelector('span'),status=panel.querySelector('.dense-story-status');
 function label(text,x,y,color='#9fb4cd',size=13){ctx.fillStyle=color;ctx.font=size+'px "Segoe UI",Consolas,sans-serif';ctx.textAlign='left';ctx.fillText(text,x,y);}
 function grid(values,x,y,cols,cell,reveal=1,selected=-1){const max=Math.max(...values.map(Math.abs),1e-9);for(let i=0;i<values.length;i++){const a=Math.abs(values[i])/max;ctx.globalAlpha=i<reveal*values.length?1:.12;ctx.fillStyle=values[i]<0?`rgba(251,113,133,${.12+.88*a})`:`rgba(94,225,232,${.12+.88*a})`;ctx.fillRect(x+i%cols*cell,y+Math.floor(i/cols)*cell,cell-1,cell-1);}ctx.globalAlpha=1;if(selected>=0){ctx.strokeStyle='#f6c76b';ctx.lineWidth=2;ctx.strokeRect(x+selected%cols*cell-1,y+Math.floor(selected/cols)*cell-1,cell,cell);}}
 function line(x,y,tx,ty,color,alpha,t){ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,y);ctx.bezierCurveTo(x+90,y,tx-90,ty,tx,ty);ctx.stroke();if(t!==null){const u=1-t,px=u*u*u*x+3*u*u*t*(x+90)+3*u*t*t*(tx-90)+t*t*t*tx,py=u*u*u*y+3*u*u*t*y+3*u*t*t*ty+t*t*t*ty;ctx.fillStyle=color;ctx.beginPath();ctx.arc(px,py,3.5,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;}
 function render(stage,progress){
  const key=S.current.key;panel.hidden=!['fc1','output'].includes(key);document.body.classList.toggle('dense-story-active',!panel.hidden);if(panel.hidden)return;
  progress=clamp(progress);ctx.fillStyle='#0a1424';ctx.fillRect(0,0,1120,410);
  const context=S.sample.index+'/'+key+'/'+S.selection.neuron;
  if(cache?.key!==context)cache={key:context,op:denseAt(DATA,S.sample,key==='fc1'?'fc1':'output',S.selection.neuron,0)};
  if(key==='fc1'){
   const drop=stage==='S10',p=drop?1:progress,op=cache.op,n=Math.min(32,Math.floor(p/.76*32)),sum=op.contributions.slice(0,n).reduce((a,b)=>a+b,0),biased=p>=.83;
   title.textContent=drop?'Dropout · 推理时保持全部激活':'FC1 · 1600 个特征，加权形成 128 个激活';shape.textContent='[1,1600] → Linear → ReLU → [1,128]';
   grid(S.sample.tensors.flatten.values,25,70,40,5.7);label('1600 个真实输入',25,48,'#72b5ff',16);label('40×40 仅为向量显示排布',25,326);
   for(let k=0;k<32;k++){const y=67+k*7.6,active=k<n,current=k===n&&p<.76;line(292,y,558,190,op.contributions[k]<0?'#fb7185':'#5ee1e8',active||current?.85:.13,current?(p/.76*32)%1:null);}
   label('32 组 × 50 个真实乘积',285,45,'#a78bfa',15);
   ctx.fillStyle='#182d43';ctx.strokeStyle='#5ee1e8';ctx.beginPath();ctx.arc(585,190,31,0,Math.PI*2);ctx.fill();ctx.stroke();label('Σ',575,198,'#5ee1e8',25);
   label('累计 '+n+'/32 组',480,265);label('Σ = '+sum.toFixed(5),480,288,'#e3edf9',16);label('此处示范神经元 '+S.selection.neuron+'；128 个神经元各有独立权重',310,341,'#91a8c1',12);
   line(617,190,750,190,biased?'#f6c76b':'#58708b',1,p>=.76&&p<.88?(p-.76)/.12:null);label('+ bias',630,146,biased?'#f6c76b':'#58708b');label(op.bias.toFixed(5),625,169);
   label('ReLU',725,153,p>=.88?'#34d399':'#58708b',17);label(p>=.88?Math.max(0,op.raw).toFixed(5):'max(0, z)',715,224,'#34d399',15);
   grid(S.sample.tensors.fc1_relu.values,839,103,16,14,drop?1:clamp((p-.86)/.14),S.selection.neuron);label('128 个真实隐藏激活',835,48,'#34d399',16);
   label('选中神经元 '+S.selection.neuron,835,255,'#f6c76b');label('z = '+op.expected.toFixed(6),835,280);label('a = '+op.expectedActivated.toFixed(6),835,304,'#34d399');
   status.textContent=drop?'eval()：Dropout 为恒等映射，128 个数值保持不变。':p<.76?'输入 × 真实权重：已累计 '+n+' / 32 组，每组 50 项。':p<.88?'32 组贡献汇聚完成，只加入一次 Bias。':'ReLU 截断负值，形成真实的 128 维隐藏表示。';
   canvas.dataset.numeric=JSON.stringify({stage,progress:p,neuron:S.selection.neuron,partial:sum,bias:op.bias,raw:op.raw,expected:op.expected,activated:op.expectedActivated});
  }else{
   const logits=S.sample.tensors.logits.values,probs=S.sample.tensors.probabilities.values,pred=S.sample.prediction,soft=stage==='S12'||stage==='S13',done=stage==='S13',p=progress,max=Math.max(...logits.map(Math.abs),1e-9),maxLogit=Math.max(...logits),exps=logits.map(v=>Math.exp(v-maxLogit)),den=exps.reduce((a,b)=>a+b,0);
   title.textContent=done?'Prediction · 最终选择概率最大的类别':soft?'Softmax · 将十类响应归一化为概率':'FC2 · 128 个激活形成十个 Logits';shape.textContent='[1,128] → [1,10] → Softmax → Argmax';
   grid(S.sample.tensors.fc1_relu.values,26,112,16,12);label('128 个隐藏激活',25,54,'#72b5ff',16);label('每类使用自己的128个权重',25,249);
   for(let i=0;i<10;i++){const y=72+i*27,shown=soft?1:clamp(p*10-i),color=logits[i]<0?'#fb7185':'#5ee1e8';line(233,158,420,y,color,shown>0?.7:.12,shown>0&&shown<1?shown:null);ctx.fillStyle=shown>0?'#1c3349':'#111e31';ctx.fillRect(425,y-13,118,22);label(i+'  z='+logits[i].toFixed(3),433,y+3,shown>0?color:'#53647b',12);}
   label('10 个原始 Logits（不是概率）',319,45,'#a78bfa',14);
   const reveal=done?1:p;
   for(let i=0;i<10;i++){const y=66+i*27;label(String(i),701,y+5,i===pred?'#f6c76b':'#9fb4cd',15);ctx.fillStyle='#17253a';ctx.fillRect(726,y-10,242,16);if(soft){ctx.fillStyle=i===pred?'#34d399':'#5c94ad';ctx.fillRect(726,y-10,242*probs[i]*reveal,16);label((probs[i]*100).toFixed(3)+'%',979,y+4,i===pred?'#34d399':'#9fb4cd',12);}else{ctx.strokeStyle='#50627b';ctx.beginPath();ctx.moveTo(847,y-12);ctx.lineTo(847,y+9);ctx.stroke();ctx.fillStyle=logits[i]<0?'#fb7185':'#5ee1e8';const width=logits[i]/max*116*clamp(p*10-i);ctx.fillRect(width<0?847+width:847,y-9,Math.abs(width),15);label(logits[i].toFixed(3),979,y+4,'#9fb4cd',12);}}
   label(soft?'十类概率 · 总和最终为 1':'原始响应 · 可正可负',723,38,soft?'#34d399':'#5ee1e8',16);
   if(soft){label('pᵢ = exp(zᵢ − max z) / Σ exp(z − max z)',310,364,'#b6a2f0',16);label('Σ exp = '+den.toFixed(6),310,388);}
   if(done){label('预测 '+pred,750,384,'#34d399',25);label('真实标签 '+S.sample.groundTruth,925,382,'#f6c76b',16);}
   status.textContent=done?'Argmax 选择最大真实概率：预测 '+pred+'，真实标签 '+S.sample.groundTruth+'。':soft?'对真实 Logits 作稳定 Softmax。柱条的增长是视觉呈现；最终概率来自真实导出数据。':'每条连接表示该类别的真实加权和：Σ128(h×w) + 一次 Bias。';
   canvas.dataset.numeric=JSON.stringify({stage,progress:p,logits,probabilities:probs,softmax:exps.map(v=>v/den),prediction:pred});
  }
 }
 return {render};
}
root.DigitVisionDenseStory={create};
})(window);
