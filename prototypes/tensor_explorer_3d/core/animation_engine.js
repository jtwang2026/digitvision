/** One timestamp-driven state machine; the SceneManager's existing RAF calls tick(). */
export class TensorAnimationEngine {
  constructor(onFrame) {
    this.onFrame = onFrame;
    this.mode = "assembly";
    this.status = "idle";
    this.progress = 0;
    this.speed = 1;
    this.direction = 1;
    this.lastTime = null;
    this.durations = Object.freeze({ assembly: 3200, scan: 5600, tracking: 6400, separation: 1100, flow: 4200, spatial: 9000, flatten: 24000 });
  }
  snapshot() { return {mode:this.mode,status:this.status,progress:this.progress,speed:this.speed,direction:this.direction}; }
  emit() { this.onFrame(this.snapshot()); }
  select(mode) {
    if (!(mode in this.durations)) throw new RangeError("Unknown animation mode");
    this.mode=mode;this.status="idle";this.progress=0;this.lastTime=null;this.emit();
  }
  setSpeed(speed) {
    if (!Number.isFinite(speed)||speed<=0) throw new RangeError("Animation speed");
    this.speed=speed;this.emit();
  }
  setDirection(direction) {
    if (direction!==1&&direction!==-1) throw new RangeError("Animation direction");
    this.direction=direction;this.emit();
  }
  play() {
    if (this.status==="playing") return;
    if (this.status==="complete") this.progress=0;
    this.status="playing";this.lastTime=null;this.emit();
  }
  pause() { if(this.status==="playing") {this.status="paused";this.lastTime=null;this.emit();} }
  restart() {this.progress=0;this.status="playing";this.lastTime=null;this.emit();}
  stop() {this.status="idle";this.progress=0;this.lastTime=null;this.emit();}
  /** Reverse separation while preserving its current pose. */
  returnSeparated() {
    if(this.mode!=="separation")return;
    this.status="returning";this.lastTime=null;this.emit();
  }
  tick(now) {
    if(this.status!=="playing"&&this.status!=="returning")return;
    if(this.lastTime===null){this.lastTime=now;return;}
    const delta=Math.max(0,Math.min(100,now-this.lastTime));this.lastTime=now;
    const step=delta*this.speed/this.durations[this.mode];
    this.progress=Math.max(0,Math.min(1,this.progress+(this.status==="returning"?-step:step)));
    if(this.status==="returning"&&this.progress===0)this.status="idle";
    else if(this.status==="playing"&&this.progress===1)this.status="complete";
    this.emit();
  }
}
