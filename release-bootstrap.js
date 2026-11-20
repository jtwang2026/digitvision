/* Explicit ordered startup with progress, retry and a visible failure state. */
(async function(){
 'use strict';const self=document.currentScript,root=new URL('./',self.src),mode=self.dataset.explorer;
 const cover=document.createElement('div');cover.id='release-loading';cover.setAttribute('role','status');cover.style.cssText='position:fixed;inset:0;z-index:9999;background:#080d18f5;display:grid;place-content:center;padding:30px;color:#e4effa;font:17px/1.8 system-ui';cover.innerHTML='<strong style="font-size:26px;color:#66d9df">DigitVision</strong><p id="release-loading-text">正在准备真实模型数据…</p><progress style="width:min(460px,80vw);accent-color:#5ee1e8"></progress><small>首次加载可能较慢。数据只用于浏览器回放，不需要安装软件。</small><button hidden style="margin-top:18px;padding:12px">重新加载</button>';document.body.append(cover);
 const message=cover.querySelector('p'),bar=cover.querySelector('progress');
 async function attempt(fn,name){let error;for(let i=0;i<3;i++){try{return await fn(i);}catch(e){error=e;message.textContent=name+'失败，正在重试 '+(i+1)+'/3…';await new Promise(r=>setTimeout(r,800*(i+1)));}}throw error;}
 async function bytes(url){const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),120000);try{const r=await fetch(url,{signal:controller.signal});if(!r.ok)throw Error('HTTP '+r.status);const total=Number(r.headers.get('content-length'))||0,reader=r.body.getReader(),chunks=[];let loaded=0;while(true){const p=await reader.read();if(p.done)break;chunks.push(p.value);loaded+=p.value.length;message.textContent='正在下载真实数据：'+(loaded/1048576).toFixed(2)+' MB'+(total?' / '+(total/1048576).toFixed(2)+' MB':'');if(total){bar.max=total;bar.value=loaded;}}const all=new Uint8Array(loaded);let start=0;for(const c of chunks){all.set(c,start);start+=c.length;}return all;}finally{clearTimeout(timeout);}}
 function script(path,module=false){return attempt(()=>new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=new URL(path,root).href;s.async=false;if(module)s.type='module';const timeout=setTimeout(()=>{s.remove();reject(Error('脚本加载超时：'+path));},60000);s.onload=()=>{clearTimeout(timeout);resolve();};s.onerror=()=>{clearTimeout(timeout);s.remove();reject(Error('无法加载：'+path));};document.head.append(s);}),path);}
 try{
  let data;
  if('DecompressionStream' in window){const packed=await attempt(()=>bytes(new URL('prototypes/forward_pass_explorer/data/model-data.bin',root)),'数据下载');message.textContent='正在解压真实 Tensor…';const plain=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).text();data=JSON.parse(plain);}
  else {const raw=await attempt(()=>bytes(new URL('prototypes/forward_pass_explorer/data/forward_pass_data.js',root)),'兼容数据下载');const s=new TextDecoder().decode(raw);data=JSON.parse(s.slice(s.indexOf('=')+1).trim().replace(/;$/,''));}
  if(data?.samples?.length!==3||!data.weights)throw Error('真实数据不完整');window.DIGITVISION_DATA=data;
  const base='prototypes/forward_pass_explorer/';
  const list=mode==='2d'?['pool_comparison.js','spatial_layers.js','advanced_spatial.js','unified_spatial.js','multi_channel.js','view_controller.js','stage_registry.js','global_engine.js','spatial_sync.js','dense_story.js','app.js']:[];
  for(let i=0;i<list.length;i++){message.textContent='正在启动动画模块 '+(i+1)+' / '+list.length;await script(base+list[i]+'?release=20261120b');}
  if(mode==='3d'){await script(base+'spatial_sync.js?release=20261120b');await script('prototypes/tensor_explorer_3d/app.js?release=20261120b',true);}
  if(mode==='2d'?!window.ForwardExplorer:!window.TensorExplorer)throw Error('动画初始化失败，请重新加载；若仍失败请检查浏览器控制台。');
  cover.remove();document.documentElement.dataset.releaseReady='true';
 }catch(e){message.textContent='加载未完成：'+e.message;bar.remove();const button=cover.querySelector('button');button.hidden=false;button.onclick=()=>location.reload();document.documentElement.dataset.releaseReady='failed';console.error('DigitVision startup:',e);}
})();
