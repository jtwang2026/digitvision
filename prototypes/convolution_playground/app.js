/* DigitVision Convolution Playground. No dependencies or network requests. */
(function (root) {
  "use strict";
  const INPUT = [[1,2,3,0,1],[0,1,2,1,0],[1,0,1,2,1],[2,1,0,1,2],[0,1,2,0,1]];
  const PRESETS = {
    vertical: [[1,0,-1],[1,0,-1],[1,0,-1]],
    horizontal: [[1,1,1],[0,0,0],[-1,-1,-1]],
    diagonal: [[2,1,0],[1,0,-1],[0,-1,-2]]
  };
  const copy = matrix => matrix.map(row => row.slice());
  const STATES = Object.freeze({ IDLE:"IDLE", PLAYING:"PLAYING", PAUSED:"PAUSED", COMPLETED:"COMPLETED" });

  /** Build a row-major cross-correlation plan; the kernel is never flipped. */
  function convolve(input, kernel, stride = 1, padding = 0) {
    if (![1,2].includes(stride) || ![0,1].includes(padding)) throw new RangeError("Unsupported stride/padding");
    if (!Array.isArray(input) || !input.length || !Array.isArray(input[0]) || !input[0].length) throw new TypeError("Invalid input matrix");
    const height = input.length, width = input[0].length;
    if (input.some(row => row.length !== width || row.some(value => !Number.isFinite(value)))) throw new TypeError("Input must be rectangular and finite");
    if (!Array.isArray(kernel) || kernel.length !== 3 || kernel.some(row => !Array.isArray(row) || row.length !== 3 || row.some(value => !Number.isFinite(value)))) throw new TypeError("Kernel must be finite and 3 x 3");
    const outHeight = Math.floor((height + 2 * padding - 3) / stride) + 1;
    const outWidth = Math.floor((width + 2 * padding - 3) / stride) + 1;
    if (outHeight < 1 || outWidth < 1) throw new RangeError("Kernel exceeds input");
    const padded = Array.from({length:height + 2 * padding}, (_,r) =>
      Array.from({length:width + 2 * padding}, (_,c) =>
        r >= padding && r < height + padding && c >= padding && c < width + padding ? input[r-padding][c-padding] : 0));
    const steps = [], output = Array.from({length:outHeight}, () => Array(outWidth).fill(0));
    for (let row = 0; row < outHeight; row++) {
      for (let col = 0; col < outWidth; col++) {
        const patch = [], products = [], terms = [];
        let sum = 0;
        for (let m = 0; m < 3; m++) {
          patch[m] = []; products[m] = [];
          for (let n = 0; n < 3; n++) {
            const x = padded[row * stride + m][col * stride + n], weight = kernel[m][n];
            const product = x * weight;
            patch[m][n] = x; products[m][n] = product;
            terms.push({x,weight,product}); sum += product;
          }
        }
        if (!Number.isFinite(sum)) throw new RangeError("Calculation overflow");
        output[row][col] = sum;
        steps.push({row,col,inputRow:row*stride-padding,inputCol:col*stride-padding,
          paddedRow:row*stride,paddedCol:col*stride,patch,products,terms,sum});
      }
    }
    return {height,width,outHeight,outWidth,stride,padding,padded,kernel:copy(kernel),output,steps};
  }

  /** A reversible state machine. Output is always derived from the completed prefix. */
  class PlaygroundState {
    constructor(plan) { this.plan = plan; this.restart(); }
    configure(plan) { this.plan = plan; this.restart(); }
    restart() { this.completed = 0; this.status = STATES.IDLE; }
    get current() { return this.plan.steps[Math.max(0, this.completed - 1)]; }
    get visibleOutput() {
      return this.plan.steps.map((step,index) => index < this.completed ? step.sum : null);
    }
    play() {
      if (this.completed < this.plan.steps.length) this.status = STATES.PLAYING;
    }
    pause() { if (this.status === STATES.PLAYING) this.status = STATES.PAUSED; }
    advance() {
      if (this.completed >= this.plan.steps.length) return;
      this.completed++;
      if (this.completed === this.plan.steps.length) this.status = STATES.COMPLETED;
      else if (this.status !== STATES.PLAYING) this.status = STATES.PAUSED;
    }
    next() { this.pause(); this.advance(); }
    previous() {
      if (this.completed === 0) return;
      this.completed--;
      this.status = this.completed === 0 ? STATES.IDLE : STATES.PAUSED;
    }
  }
  const API = {INPUT:copy(INPUT),PRESETS,STATES,convolve,PlaygroundState};
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof document === "undefined") return;
  root.DigitVisionPlayground = API;

  const $ = id => document.getElementById(id);
  let kernel = copy(PRESETS.vertical);
  const session = new PlaygroundState(convolve(INPUT,kernel));
  let frame = 0, invalid = false;
  const fmt = value => {
    if (Object.is(value,-0) || value === 0) return "0";
    if (Math.abs(value) >= 100000 || Math.abs(value) < .0001) return value.toExponential(2);
    return Number(value.toFixed(4)).toString().replace("-","−");
  };
  const signed = value => value < 0 ? "(" + fmt(value) + ")" : fmt(value);

  function stopClock() { clearTimeout(frame); frame = 0; }
  function scheduleTick() {
    stopClock();
    if (session.status === STATES.PLAYING) frame = setTimeout(tick,1400 / Number($("speed").value));
  }
  function pause() { stopClock(); session.pause(); render(); }
  function next() { stopClock(); if (!invalid) session.next(); render(); }
  function previous() { stopClock(); if (!invalid) session.previous(); render(); }
  function restart() { stopClock(); session.restart(); render(); }
  function play() {
    if (invalid || session.status === STATES.PLAYING || session.status === STATES.COMPLETED) return;
    stopClock(); session.play();
    if (session.completed === 0) session.advance();
    render();
    scheduleTick();
  }
  function tick() {
    frame = 0;
    if (session.status !== STATES.PLAYING) return;
    session.advance(); render(); scheduleTick();
  }

  function buildMatrices() {
    const plan = session.plan, size = plan.padded.length;
    $("input-grid").replaceChildren();
    $("input-grid").style.gridTemplateColumns = "repeat(" + size + ",1fr)";
    $("input-grid").dataset.size = size;
    plan.padded.forEach((row,r) => row.forEach((value,c) => {
      const cell = document.createElement("div");
      cell.className = "matrix-cell";
      cell.textContent = fmt(value); cell.dataset.row = r; cell.dataset.col = c;
      cell.setAttribute("role","gridcell");
      cell.setAttribute("aria-label","Input ["+(r-plan.padding)+", "+(c-plan.padding)+"] = "+value);
      if (r < plan.padding || c < plan.padding || r >= plan.height + plan.padding || c >= plan.width + plan.padding) cell.classList.add("padding-cell");
      $("input-grid").append(cell);
    }));
    const windowElement = document.createElement("div");
    windowElement.className = "kernel-window"; windowElement.id = "kernel-window";
    windowElement.setAttribute("aria-hidden","true"); $("input-grid").append(windowElement);
    $("output-grid").replaceChildren();
    $("output-grid").style.gridTemplateColumns = "repeat("+plan.outWidth+",1fr)";
    plan.steps.forEach((step,index) => {
      const cell = document.createElement("div");
      cell.className = "matrix-cell output-cell"; cell.dataset.outputIndex = index;
      cell.setAttribute("role","gridcell"); $("output-grid").append(cell);
    });
    $("input-shape").textContent = size+" × "+size;
    $("output-shape").textContent = $("output-dimension").textContent = plan.outHeight+" × "+plan.outWidth;
    $("padding-note").textContent = plan.padding ? "Zero padding · 1" : "No padding";
    $("size-formula").textContent = "⌊(5 + 2·"+plan.padding+" − 3) / "+plan.stride+"⌋ + 1 = "+plan.outHeight;
    $("stride-formula").textContent = "With stride/padding: X_pad[i·"+plan.stride+"+m, j·"+plan.stride+"+n] · K[m,n], m,n = 0…2";
  }
  function renderKernel() {
    $("kernel-grid").replaceChildren();
    kernel.forEach((row,r) => row.forEach((value,c) => {
      const field = document.createElement("input");
      field.type = "number"; field.step = "any"; field.min = "-1000"; field.max = "1000";
      field.value = value; field.dataset.row = r; field.dataset.col = c;
      field.setAttribute("aria-label","Kernel weight row "+r+" column "+c);
      $("kernel-grid").append(field);
    }));
  }
  function positionWindow() {
    const step = session.current, grid = $("input-grid");
    const first = grid.querySelector('[data-row="'+step.paddedRow+'"][data-col="'+step.paddedCol+'"]');
    const last = grid.querySelector('[data-row="'+(step.paddedRow+2)+'"][data-col="'+(step.paddedCol+2)+'"]');
    if (!first || !last) return;
    const overlay = $("kernel-window");
    overlay.style.transform = "translate("+first.offsetLeft+"px,"+first.offsetTop+"px)";
    overlay.style.width = (last.offsetLeft+last.offsetWidth-first.offsetLeft)+"px";
    overlay.style.height = (last.offsetTop+last.offsetHeight-first.offsetTop)+"px";
  }
  function fillMini(id,matrix) {
    const fragment = document.createDocumentFragment();
    matrix.flat().forEach(value => { const cell = document.createElement("div"); cell.textContent = fmt(value); fragment.append(cell); });
    $(id).replaceChildren(fragment);
  }
  function render() {
    const plan = session.plan, step = session.current, total = plan.steps.length;
    $("state-badge").textContent = session.status; $("state-badge").dataset.state = session.status;
    $("play").disabled = invalid || session.status === STATES.PLAYING || session.status === STATES.COMPLETED;
    $("pause").disabled = session.status !== STATES.PLAYING;
    $("previous").disabled = invalid || session.completed === 0;
    $("next").disabled = invalid || session.completed === total;
    $("written-count").textContent = session.completed+" / "+total+" written";
    $("progress-fill").style.width = (100*session.completed/total)+"%";
    $("current-position").textContent = "Y["+step.row+", "+step.col+"]";
    $("step-number").textContent = "STEP "+String(Math.max(1,session.completed)).padStart(2,"0")+" / "+String(total).padStart(2,"0");
    $("input-grid").querySelectorAll(".matrix-cell").forEach(cell => {
      const r = Number(cell.dataset.row), c = Number(cell.dataset.col);
      cell.classList.toggle("covered",r>=step.paddedRow && r<step.paddedRow+3 && c>=step.paddedCol && c<step.paddedCol+3);
    });
    const maxMagnitude = Math.max(1,...plan.steps.map(item => Math.abs(item.sum)));
    $("output-grid").querySelectorAll(".output-cell").forEach((cell,index) => {
      const filled = index < session.completed, item = plan.steps[index];
      cell.classList.toggle("filled",filled);
      cell.classList.toggle("current",filled && index === session.completed-1);
      cell.textContent = filled ? fmt(item.sum) : "·";
      cell.style.background = filled ? "rgba(62,174,115,"+(.11+.24*Math.abs(item.sum)/maxMagnitude)+")" : "";
      cell.setAttribute("aria-label","Output ["+item.row+", "+item.col+"]: "+(filled?item.sum:"not computed"));
    });
    fillMini("patch-grid",step.patch); fillMini("weight-grid",plan.kernel); fillMini("product-grid",step.products);
    $("sum-expression").textContent = step.terms.map(term => signed(term.product)).join(" + ");
    $("current-sum").textContent = fmt(step.sum);
    $("sum-destination").textContent = (session.completed === 0 ? "Preview → " : "Written → ")+"Y["+step.row+", "+step.col+"]";
    $("multiplication-terms").replaceChildren();
    step.terms.forEach(term => {
      const cell = document.createElement("div"); cell.className = "term";
      for (const [text,className] of [[fmt(term.x),"term-input"],[" × ",""],[signed(term.weight),"term-weight"],[" = ",""],[fmt(term.product),"term-product"]]) {
        const span = document.createElement("span"); span.textContent = text; span.className = className; cell.append(span);
      }
      $("multiplication-terms").append(cell);
    });
    $("progress-announcement").textContent = session.status+". "+session.completed+" of "+total+" outputs written. Current Y["+step.row+","+step.col+"] = "+step.sum;
    positionWindow();
  }
  function reconfigure() {
    stopClock();
    const fields = [...$("kernel-grid").querySelectorAll("input")];
    invalid = fields.some(field => field.value.trim() === "" || !field.validity.valid || !Number.isFinite(field.valueAsNumber));
    $("kernel-error").hidden = !invalid;
    if (invalid) {
      $("kernel-error").textContent = "请输入 −1000 到 1000 之间的有效数值。";
      session.pause(); render(); return;
    }
    kernel = Array.from({length:3},(_,r) => fields.slice(r*3,r*3+3).map(field => field.valueAsNumber));
    session.configure(convolve(INPUT,kernel,Number($("stride").value),Number($("padding").value)));
    buildMatrices(); render();
  }
  $("play").addEventListener("click",play);
  $("pause").addEventListener("click",pause);
  $("next").addEventListener("click",next);
  $("previous").addEventListener("click",previous);
  $("restart").addEventListener("click",restart);
  $("stride").addEventListener("change",reconfigure);
  $("padding").addEventListener("change",reconfigure);
  $("speed").addEventListener("input",() => {
    $("speed-value").textContent = $("speed").value+"×";
    document.documentElement.style.setProperty("--move-duration",(260/Number($("speed").value))+"ms");
    if (session.status === STATES.PLAYING) scheduleTick();
  });
  $("preset").addEventListener("change",() => {
    const preset = $("preset").value;
    if (preset !== "custom") {kernel = copy(PRESETS[preset]); renderKernel();}
    const descriptions = {vertical:"Left − right response",horizontal:"Top − bottom response",diagonal:"Diagonal response",custom:"Your own filter"};
    $("preset-description").textContent = descriptions[preset]; reconfigure();
  });
  $("kernel-grid").addEventListener("input",event => {
    if (event.target.tagName !== "INPUT") return;
    $("preset").value = "custom"; $("preset-description").textContent = "Your own filter"; reconfigure();
  });
  document.addEventListener("keydown",event => {
    if (["INPUT","SELECT","TEXTAREA","BUTTON"].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === " ") {event.preventDefault();session.status === STATES.PLAYING ? pause() : play();}
    else if (event.key === "ArrowRight") {event.preventDefault();next();}
    else if (event.key === "ArrowLeft") {event.preventDefault();previous();}
    else if (event.key.toLowerCase() === "r") restart();
  });
  renderKernel(); buildMatrices(); render();
  new ResizeObserver(positionWindow).observe($("input-grid"));
  const params = new URLSearchParams(location.search);
  if (params.has("step") && /^\d+$/.test(params.get("step"))) {
    const completed = Math.min(Number(params.get("step")),session.plan.steps.length);
    for (let i=0;i<completed;i++) session.next();
    render();
  }

  /** Real-browser UI checks, opt-in only; normal offline usage never runs these. */
  async function runBrowserChecks() {
    const results = [];
    const assert = (condition,name) => {if (!condition) throw new Error(name); results.push({name,passed:true});};
    const wait = ms => new Promise(resolve => setTimeout(resolve,ms));
    const change = (id,value) => {$(id).value = value; $(id).dispatchEvent(new Event("change",{bubbles:true}));};
    const visible = () => [...$("output-grid").children].map(cell => cell.textContent.replace("−","-"));
    const baseline = [-4,0,4,0,-2,0,0,-1,-1];
    try {
      $("restart").click();
      assert(session.status === STATES.IDLE && session.completed === 0,"Initial IDLE, no outputs");
      assert($("output-shape").textContent === "3 × 3","Default output dimensions");
      $("next").click();
      assert(session.status === STATES.PAUSED && session.completed === 1 && visible()[0] === "-4","Next writes first value -4");
      $("next").click(); $("previous").click();
      assert(session.completed === 1 && visible()[1] === "·","Previous clears undone output");
      $("previous").click();
      assert(session.status === STATES.IDLE && visible().every(value => value === "·"),"Previous restores initial prefix");
      $("speed").value = "3"; $("speed").dispatchEvent(new Event("input",{bubbles:true}));
      assert($("speed-value").textContent === "3×","Speed control updates");
      $("play").click();
      assert(session.status === STATES.PLAYING,"Play enters PLAYING");
      await wait(650);
      assert(session.completed >= 2,"Playback clock advances positions");
      $("pause").click(); const pausedAt = session.completed;
      await wait(650);
      assert(session.status === STATES.PAUSED && session.completed === pausedAt,"Pause freezes playback");
      $("play").click();
      for (let i=0;i<60 && session.status !== STATES.COMPLETED;i++) await wait(100);
      assert(session.status === STATES.COMPLETED,"Resume reaches COMPLETED");
      assert(JSON.stringify(visible().map(Number)) === JSON.stringify(baseline),"Complete default matrix matches independent reference");
      $("previous").click();
      assert(session.status === STATES.PAUSED && session.completed === 8 && visible()[8] === "·","Previous works after completion");
      $("restart").click();
      assert(session.status === STATES.IDLE && session.completed === 0,"Restart clears all outputs");
      change("stride","2");
      assert($("output-shape").textContent === "2 × 2","Stride 2, padding 0 dimensions");
      change("padding","1");
      assert($("output-shape").textContent === "3 × 3" && $("input-grid").querySelectorAll(".padding-cell").length === 24,"Stride 2, padding 1 dimensions and zero border");
      for (let i=0;i<9;i++) $("next").click();
      assert(JSON.stringify(visible().map(Number)) === JSON.stringify([-3,2,1,-2,-2,4,-2,1,1]),"Stride 2, padding 1 numeric values");
      change("stride","1");
      assert($("output-shape").textContent === "5 × 5","Stride 1, padding 1 dimensions");
      change("preset","horizontal");
      assert(JSON.stringify(session.plan.kernel) === JSON.stringify(PRESETS.horizontal) && session.completed === 0,"Horizontal preset and reset");
      change("preset","diagonal");
      assert(JSON.stringify(session.plan.kernel) === JSON.stringify(PRESETS.diagonal),"Diagonal preset");
      const weight = $("kernel-grid").querySelector("input");
      weight.value = "2.5"; weight.dispatchEvent(new Event("input",{bubbles:true}));
      assert($("preset").value === "custom" && session.plan.kernel[0][0] === 2.5,"Custom decimal weight");
      weight.value = ""; weight.dispatchEvent(new Event("input",{bubbles:true}));
      assert(!$("kernel-error").hidden && $("play").disabled && $("next").disabled,"Invalid weight blocks computation");
      weight.value = "2"; weight.dispatchEvent(new Event("input",{bubbles:true}));
      assert($("kernel-error").hidden && !$("play").disabled,"Valid weight recovers");
      change("preset","vertical"); change("stride","1"); change("padding","0");
      assert(document.documentElement.scrollWidth <= innerWidth,"No horizontal layout overflow");
      $("play").click(); $("restart").click(); await wait(550);
      assert(session.status === STATES.IDLE && session.completed === 0,"Restart cancels an active clock");
      document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowRight",bubbles:true}));
      assert(session.completed === 1,"Keyboard next");
      document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowLeft",bubbles:true}));
      assert(session.completed === 0,"Keyboard previous");
      document.body.dispatchEvent(new KeyboardEvent("keydown",{key:" ",bubbles:true}));
      assert(session.status === STATES.PLAYING,"Keyboard play");
      document.body.dispatchEvent(new KeyboardEvent("keydown",{key:" ",bubbles:true}));
      assert(session.status === STATES.PAUSED,"Keyboard pause");
      document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"r",bubbles:true}));
      assert(session.status === STATES.IDLE && session.completed === 0,"Keyboard restart");
      $("play").click(); change("padding","1"); await wait(550);
      assert(session.status === STATES.IDLE && session.completed === 0,"Parameter change cancels an active clock");
      change("padding","0");
      for (let i=0;i<9;i++) $("next").click();
      const report = {passed:true,checks:results.length,viewport:{width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth},results,defaultOutput:session.plan.output};
      $("qa-report").textContent = JSON.stringify(report,null,2);
      document.documentElement.dataset.qaStatus = "passed";
    } catch(error) {
      pause();
      $("qa-report").textContent = JSON.stringify({passed:false,error:error.message,results},null,2);
      document.documentElement.dataset.qaStatus = "failed";
    }
    $("qa-panel").hidden = false;
  }
  if (params.get("selftest") === "1") runBrowserChecks();
})(typeof window !== "undefined" ? window : globalThis);
