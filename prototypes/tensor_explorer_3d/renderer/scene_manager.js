import * as THREE from "../vendor/three.module.min.js";
import { OrbitControls } from "../vendor/OrbitControls.js";
import { colorScale, textureBytes, responseColor } from "../core/color_mapper.js";
import { planePoint, uvToCoordinate } from "../core/coordinate_mapper.js";

export class SceneManager {
  constructor(container, onPick) {
    this.container=container;this.onPick=onPick;this.stacks={};this.pickables=[];this.frames=[];this.drawTimes=[];this.intervals=[];this.lastFrame=0;this.lowPower=false;this.needsRender=true;this.renderedFrames=0;
    this.renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"default",alpha:false});
    this.renderer.setClearColor(0x091422,1);
    container.append(this.renderer.domElement);
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0x091422);
    this.camera=new THREE.PerspectiveCamera(43,1,.1,200);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.enableDamping=true;this.controls.dampingFactor=.075;
    this.controls.minDistance=8;this.controls.maxDistance=85;this.controls.enablePan=true;
    this.raycaster=new THREE.Raycaster();
    this.geometry=new THREE.PlaneGeometry(1,1);
    this.markerGeometry=new THREE.RingGeometry(.075,.115,16);
    this.markerMaterial=new THREE.MeshBasicMaterial({color:0xf6c76b,side:THREE.DoubleSide,depthTest:false});
    this.otherMarkerMaterial=new THREE.MeshBasicMaterial({color:0x5ee1e8,side:THREE.DoubleSide,depthTest:false,transparent:true,opacity:.85});
    this.frameGeometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.5,-.5,.01),new THREE.Vector3(.5,-.5,.01),new THREE.Vector3(.5,.5,.01),new THREE.Vector3(-.5,.5,.01)]);
    this.frameMaterial=new THREE.LineBasicMaterial({color:0x3a6587,transparent:true,opacity:.7});
    this.selectedFrameMaterial=new THREE.LineBasicMaterial({color:0xf6c76b});
    this.relationGeometry=new THREE.BufferGeometry();
    this.relationGeometry.setAttribute("position",new THREE.BufferAttribute(new Float32Array(32*2*3),3));
    this.relationMaterial=new THREE.LineBasicMaterial({color:0x5ee1e8,transparent:true,opacity:.35,depthWrite:false});
    this.relation=new THREE.LineSegments(this.relationGeometry,this.relationMaterial);
    this.scene.add(this.relation);this.relation.visible=false;
    this.flowMaterial=new THREE.MeshBasicMaterial({color:0x5ee1e8,side:THREE.DoubleSide,transparent:true,opacity:.16,depthWrite:false,depthTest:false});
    this.flowBand=new THREE.Mesh(this.geometry,this.flowMaterial);this.flowBand.visible=false;this.scene.add(this.flowBand);
    this.flowFrameMaterial=new THREE.LineBasicMaterial({color:0x5ee1e8,transparent:true,opacity:.82,depthTest:false});
    this.flowFrame=new THREE.LineLoop(this.frameGeometry,this.flowFrameMaterial);this.flowFrame.visible=false;this.scene.add(this.flowFrame);
    this.processingMaterial=new THREE.LineBasicMaterial({color:0x5ee1e8,transparent:true,opacity:.94,depthTest:false});
    this.onFrame=null;this.animationPose={mode:"idle",progress:0};
    this.pointerDown=null;
    this.onPointerDown=e=>{this.pointerDown={x:e.clientX,y:e.clientY};};
    this.onPointerUp=e=>{if(!this.pointerDown)return;const d=Math.hypot(e.clientX-this.pointerDown.x,e.clientY-this.pointerDown.y);this.pointerDown=null;if(d<5)this.pick(e);};
    this.renderer.domElement.addEventListener("pointerdown",this.onPointerDown);
    this.renderer.domElement.addEventListener("pointerup",this.onPointerUp);
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(container);
    this.setPreset("three-quarter");this.resize();
    this.running=true;this.loop=this.loop.bind(this);this.raf=requestAnimationFrame(this.loop);
  }
  setLowPower(on){this.lowPower=Boolean(on);this.resize();}
  resize(){const w=Math.max(1,this.container.clientWidth),h=Math.max(1,this.container.clientHeight);this.renderer.setPixelRatio(this.lowPower?1:Math.min(devicePixelRatio||1,1.7));this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.needsRender=true;}
  setPreset(name){
    const center=this.stacks.output?.view.channels===64?10:5;
    const dual=Boolean(this.stacks.input);
    const poses={front:[0,0,center+(dual?33:23)],top:[0,dual?36:25,center+5],side:[dual?31:23,4,center+1],"three-quarter":dual?[18,12,center+29]:[12,9,center+20]};
    const p=poses[name]||poses["three-quarter"];this.camera.position.set(...p);this.controls.target.set(0,0,center);this.controls.update();this.needsRender=true;
  }
  removeStack(name){
    const stack=this.stacks[name];if(!stack)return;
    this.scene.remove(stack.group);for(const plane of stack.planes){plane.texture.dispose();plane.material.dispose();}
    delete this.stacks[name];
  }
  makeStack(name,view,offset,cell,step){
    const group=new THREE.Group();group.position.x=offset;this.scene.add(group);
    const width=view.width*cell,height=view.height*cell,planes=[];
    for(let ch=0;ch<view.channels;ch++){
      const bytes=new Uint8Array(view.width*view.height*4);
      const texture=new THREE.DataTexture(bytes,view.width,view.height,THREE.RGBAFormat);
      texture.minFilter=texture.magFilter=THREE.NearestFilter;texture.generateMipmaps=false;texture.colorSpace=THREE.SRGBColorSpace;texture.needsUpdate=true;
      const material=new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,transparent:true,opacity:.54,depthWrite:false});
      const reveal={value:-1},relu={value:0},snake={value:1};material.userData.spatial={reveal,relu,snake};
      material.onBeforeCompile=shader=>{
        Object.assign(shader.uniforms,{dvReveal:reveal,dvRelu:relu,dvSnake:snake,dvSize:{value:new THREE.Vector2(view.width,view.height)}});
        shader.fragmentShader='uniform float dvSnake; uniform float dvReveal; uniform float dvRelu; uniform vec2 dvSize;\n'+shader.fragmentShader;
        shader.fragmentShader=shader.fragmentShader.replace('#include <map_fragment>',`#include <map_fragment>
          float r=min(dvSize.y-1.0,floor((1.0-vMapUv.y)*dvSize.y)); float c=min(dvSize.x-1.0,floor(vMapUv.x*dvSize.x));
          float order=r*dvSize.x+((dvSnake>0.5 && mod(r,2.0)>0.5)?dvSize.x-1.0-c:c);
          if(dvReveal>=0.0){
            if(dvRelu<0.5 && order>=dvReveal) diffuseColor.rgb=vec3(0.018,0.035,0.06);
            if(dvRelu>0.5 && sampledDiffuseColor.a<0.75){
              diffuseColor.a=opacity; float fade=clamp(dvReveal-order,0.0,1.0); diffuseColor.rgb=mix(diffuseColor.rgb,vec3(0.018,0.035,0.06),fade);
            }
          }`);
      };
      const mesh=new THREE.Mesh(this.geometry,material);mesh.scale.set(width,height,1);mesh.position.z=ch*step;mesh.userData={side:name,channel:ch};group.add(mesh);
      const frame=new THREE.LineLoop(this.frameGeometry,this.frameMaterial);frame.scale.set(width,height,1);frame.position.z=ch*step+.015;group.add(frame);
      const marker=new THREE.Mesh(this.markerGeometry,this.otherMarkerMaterial);marker.position.z=.03;marker.scale.set(2/width,2/height,1);mesh.add(marker);
      const window=new THREE.LineLoop(this.frameGeometry,this.processingMaterial);window.position.z=.07;window.visible=false;mesh.add(window);
      planes.push({mesh,frame,marker,window,material,texture});
    }
    this.stacks[name]={group,view,planes,width,height,step};this.updatePickables();
  }
  updatePickables(){this.pickables=Object.values(this.stacks).flatMap(s=>s.planes.map(p=>p.mesh));}
  setViews(input,output,scaleMode,resetCamera=false){
    const definitions=input?[["input",input,-6.7,input.width>20?.32:.47,.36],["output",output,6.7,output.width<6?.7:output.width>20?.32:.47,.36]]:[["output",output,0,.32,.38]];
    for(const key of Object.keys(this.stacks))if(!definitions.some(d=>d[0]===key))this.removeStack(key);
    for(const [name,view,x,cell,step] of definitions){
      const old=this.stacks[name];
      if(old&&(old.view.channels!==view.channels||old.view.height!==view.height||old.view.width!==view.width)){this.removeStack(name);}
      if(!this.stacks[name])this.makeStack(name,view,x,cell,step);
      const stack=this.stacks[name];stack.view=view;
      const global=colorScale(view,"global",0);
      for(let ch=0;ch<view.channels;ch++){
        const scale=scaleMode==="channel"?colorScale(view,"channel",ch):global;
        stack.planes[ch].texture.image.data.set(textureBytes(view,ch,scale));
        stack.planes[ch].texture.needsUpdate=true;
      }
    }
    if(resetCamera)this.setPreset("three-quarter");
    this.applyAnimation("idle",0,"output",0,1);
  }
  /** Move preallocated per-plane XY windows. No Tensor or texture update occurs. */
  setReluSource(view,scaleMode){
    const stack=this.stacks.output;if(!stack)return;
    for(let ch=0;ch<stack.planes.length;ch++){const p=stack.planes[ch];const bytes=textureBytes(view,ch,colorScale(view,scaleMode,ch));for(let r=0;r<view.height;r++)for(let c=0;c<view.width;c++)if(view.values[(ch*view.height+r)*view.width+c]<0)bytes[((view.height-1-r)*view.width+c)*4+3]=128;p.texture.image.data.set(bytes);p.texture.needsUpdate=true;}
  }
  setFlatten(progress,index,value){
    if(!this.flatVector){
      this.flatMaterial=new THREE.MeshBasicMaterial({color:0x34d399,side:THREE.DoubleSide});
      this.flatVector=new THREE.InstancedMesh(this.geometry,this.flatMaterial,1600);this.flatVector.instanceMatrix.setUsage(THREE.DynamicDrawUsage);this.scene.add(this.flatVector);
      this.flatToken=new THREE.Mesh(this.geometry,this.markerMaterial);this.scene.add(this.flatToken);this.flatMatrix=new THREE.Matrix4();
      for(let i=0;i<1600;i++){this.flatMatrix.makeScale(.011,.5,1);this.flatMatrix.setPosition(-9+i*18/1600,-5,0);this.flatVector.setMatrixAt(i,this.flatMatrix);const rgb=responseColor(this.stacks.output.view.values[i],colorScale(this.stacks.output.view,"global",0));this.flatVector.setColorAt(i,new THREE.Color(rgb[0]/255,rgb[1]/255,rgb[2]/255));}this.flatVector.instanceMatrix.needsUpdate=true;
    }
    this.flatVector.visible=this.flatToken.visible=true;this.flatVector.count=Math.min(1600,Math.floor(progress*1600));
    if(this.flatSource!==this.stacks.output.view){this.flatSource=this.stacks.output.view;const scale=colorScale(this.flatSource,'global',0);for(let i=0;i<1600;i++){const rgb=responseColor(this.flatSource.values[i],scale);this.flatVector.setColorAt(i,new THREE.Color(rgb[0]/255,rgb[1]/255,rgb[2]/255));}this.flatVector.instanceColor.needsUpdate=true;}
    const stack=this.stacks.output,ch=Math.floor(index/25),r=Math.floor(index%25/5),c=index%5,p=stack.planes[ch];
    const start=new THREE.Vector3((c+.5)/5-.5,.5-(r+.5)/5,.05);p.mesh.localToWorld(start);
    const end=new THREE.Vector3(-9+index*18/1600,-5,0),phase=progress>=1?1:(progress*1600)%1,t=phase*phase*(3-2*phase);
    this.flatToken.position.copy(start).lerp(end,t);this.flatToken.scale.set(.22,.22,1);this.flatToken.userData={index,value,channel:ch,row:r,col:c};this.needsRender=true;
  }
  setProcessingCursor(cursor,selectedOutputChannel){
    for(const [name,stack] of Object.entries(this.stacks))for(let i=0;i<stack.planes.length;i++){
      const p=stack.planes[i],win=p.window;
      if(name==='output'){const u=p.material.userData.spatial;u.reveal.value=cursor?cursor.progress*stack.view.width*stack.view.height:-1;u.snake.value=cursor?.traversal==="row-major"?0:1;u.relu.value=cursor?.operation?.startsWith('relu')?1:0;}
      const active=Boolean(cursor)&&(name==="input"||i===selectedOutputChannel);
      win.visible=active;if(!active)continue;
      const row=name==="input"?cursor.visualInputRow:cursor.visualOutputRow;
      const col=name==="input"?cursor.visualInputCol:cursor.visualOutputCol;
      const kw=name==="input"?cursor.kernelWidth:1,kh=name==="input"?cursor.kernelHeight:1;
      win.position.set((col+kw/2)/stack.view.width-.5,.5-(row+kh/2)/stack.view.height,.07);
      win.scale.set(kw/stack.view.width,kh/stack.view.height,1);
      if(name==="input"){
        const point=planePoint(Math.min(stack.view.height-1,Math.floor(row)),Math.min(stack.view.width-1,Math.floor(col)),stack.view.height,stack.view.width,stack.width,stack.height);
        p.marker.position.x=point.x/stack.width;p.marker.position.y=point.y/stack.height;
      }
    }
    this.needsRender=true;
  }
  /** Only transforms and opacity change during playback. Tensor textures are untouched. */
  applyAnimation(mode,progress,side,activeChannel,direction=1){
    if(this.flatVector&&mode!=='flatten'){this.flatVector.visible=this.flatToken.visible=false;}
    this.needsRender=true;
    this.animationPose={mode,progress,side,activeChannel,direction};
    const active=this.stacks[side]||this.stacks.output;
    const ease=t=>t*t*(3-2*t),clamp=t=>Math.max(0,Math.min(1,t));
    for(const [name,stack] of Object.entries(this.stacks)){
      for(let i=0;i<stack.planes.length;i++){
        const p=stack.planes[i],base=i*stack.step;
        let x=0,y=0,z=base,scale=1;
        if(stack===active&&mode==="assembly"){
          const t=ease(clamp((progress-i/stack.planes.length*.55)/.45));
          x=(1-t)*2.8;y=(1-t)*((i%2?1:-1)*1.0);z=base+(1-t)*4.5;
        }
        if(stack===active&&mode==="separation"&&i===activeChannel){
          const t=ease(progress);z=base+t*2.2;scale=1+t*.16;
        }
        p.mesh.position.set(x,y,z);p.frame.position.set(x,y,z+.015);
        p.mesh.scale.set(stack.width*scale,stack.height*scale,1);
        p.frame.scale.set(stack.width*scale,stack.height*scale,1);
        if(stack===active&&(mode==="scan"||mode==="tracking"||mode==="flow"||mode==="separation"))p.material.opacity=i===activeChannel?1:.16;
      }
    }
    const bandVisible=active&&(mode==="scan"||mode==="tracking"||mode==="flow");
    this.flowBand.visible=this.flowFrame.visible=Boolean(bandVisible);
    if(bandVisible){
      const n=active.planes.length;
      const index=(direction===-1?1-progress:progress)*(n-1);
      this.flowBand.position.set(active.group.position.x,0,index*active.step+.1);
      this.flowBand.scale.set(active.width,active.height,1);
      this.flowFrame.position.set(active.group.position.x,0,index*active.step+.12);
      this.flowFrame.scale.set(active.width,active.height,1);
      this.flowMaterial.opacity=mode==="flow"?.18:.11;
    }
  }
  setSelection(side,channel,row,col,outChannel=channel){
    this.needsRender=true;
    for(const [name,stack] of Object.entries(this.stacks)){
      const rr=Math.min(row,stack.view.height-1),cc=Math.min(col,stack.view.width-1);
      const point=planePoint(rr,cc,stack.view.height,stack.view.width,stack.width,stack.height);
      stack.planes.forEach((p,i)=>{
        const selected=name===side&&i===channel;
        p.material.opacity=selected?1:.43;p.frame.material=selected?this.selectedFrameMaterial:this.frameMaterial;
        p.marker.material=selected?this.markerMaterial:this.otherMarkerMaterial;
        p.marker.position.x=point.x/stack.width;p.marker.position.y=point.y/stack.height;
      });
    }
    this.relation.visible=Boolean(this.stacks.input&&this.stacks.output);
    if(this.relation.visible){
      const input=this.stacks.input,output=this.stacks.output,targetZ=output.planes[Math.min(outChannel,output.planes.length-1)].mesh.position.z;
      const array=this.relationGeometry.attributes.position.array;
      for(let i=0;i<32;i++){let j=i*6;array[j]=input.group.position.x;array[j+1]=0;array[j+2]=input.planes[input.planes.length===1?0:i].mesh.position.z;array[j+3]=output.group.position.x;array[j+4]=0;array[j+5]=input.planes.length===1?output.planes[i].mesh.position.z:targetZ;}
      this.relationGeometry.attributes.position.needsUpdate=true;this.relationGeometry.computeBoundingSphere();
    }
  }
  pick(e){
    const rect=this.renderer.domElement.getBoundingClientRect();
    const pointer=new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);
    this.raycaster.setFromCamera(pointer,this.camera);
    const hit=this.raycaster.intersectObjects(this.pickables,false)[0];
    if(!hit)return;
    const info=hit.object.userData,stack=this.stacks[info.side],coordinate=uvToCoordinate(hit.uv.x,hit.uv.y,stack.view.height,stack.view.width);
    this.onPick({...info,...coordinate});
  }
  loop(now){
    if(!this.running)return;
    if(this.lastFrame)this.intervals.push(now-this.lastFrame);this.lastFrame=now;
    if(this.intervals.length>600)this.intervals.shift();
    const t=performance.now();if(this.onFrame)this.onFrame(now);const moved=this.controls.update();
    if((moved||this.needsRender)&&this.container.getClientRects().length){this.renderer.render(this.scene,this.camera);this.renderedFrames++;this.drawTimes.push(performance.now()-t);if(this.drawTimes.length>600)this.drawTimes.shift();this.needsRender=false;}
    this.raf=requestAnimationFrame(this.loop);
  }
  metrics(){
    const values=this.intervals.slice(30),sorted=[...values].sort((a,b)=>a-b);
    return {frames:this.intervals.length,renderedFrames:this.renderedFrames,avgFrameMs:values.length?values.reduce((a,b)=>a+b,0)/values.length:null,p95FrameMs:sorted.length?sorted[Math.floor(sorted.length*.95)]:null,estimatedFps:values.length?1000/(values.reduce((a,b)=>a+b,0)/values.length):null,avgRenderMs:this.drawTimes.length?this.drawTimes.reduce((a,b)=>a+b,0)/this.drawTimes.length:null,gpu:this.renderer.info.memory,stacks:Object.fromEntries(Object.entries(this.stacks).map(([k,v])=>[k,{planes:v.planes.length,textures:v.planes.length}]))};
  }
  dispose(){
    this.running=false;cancelAnimationFrame(this.raf);this.resizeObserver.disconnect();this.renderer.domElement.removeEventListener("pointerdown",this.onPointerDown);this.renderer.domElement.removeEventListener("pointerup",this.onPointerUp);
    this.controls.dispose();for(const key of Object.keys(this.stacks))this.removeStack(key);
    this.flatVector?.dispose();this.flatMaterial?.dispose();
    for(const item of [this.geometry,this.markerGeometry,this.frameGeometry,this.relationGeometry,this.markerMaterial,this.otherMarkerMaterial,this.frameMaterial,this.selectedFrameMaterial,this.relationMaterial,this.flowMaterial,this.flowFrameMaterial,this.processingMaterial])item.dispose();
    this.scene.remove(this.relation,this.flowBand,this.flowFrame);this.renderer.dispose();this.renderer.forceContextLoss();this.renderer.domElement.remove();
  }
}




