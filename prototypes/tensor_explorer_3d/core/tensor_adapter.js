export const TENSORS = Object.freeze({
  conv1: { input: null, pre: "conv1_pre", relu: "conv1_relu" },
  conv2: { input: "pool1", pre: "conv2_pre", relu: "conv2_relu" },
});

export function tensorFor(bundle, sampleIndex, layer, activation, side = "output") {
  if (!bundle?.samples?.[sampleIndex] || !TENSORS[layer]) throw new RangeError("Unknown sample or layer");
  const key = side === "input" ? TENSORS[layer].input : TENSORS[layer][activation];
  if (!key) throw new RangeError("This layer has no multi-channel input stack");
  return viewForKey(bundle,sampleIndex,key);
}

export function viewForKey(bundle,sampleIndex,key) {
  const tensor = bundle.samples[sampleIndex]?.tensors[key];
  if (!tensor || tensor.shape.length !== 4 || tensor.shape[0] !== 1) throw new Error("Invalid NCHW tensor: " + key);
  const [, channels, height, width] = tensor.shape;
  if (tensor.values.length !== channels * height * width) throw new Error("Tensor length mismatch: " + key);
  return { key, tensor, channels, height, width, values: tensor.values };
}

export function tensorValue(view, channel, row, col) {
  if (![channel,row,col].every(Number.isInteger) || channel < 0 || channel >= view.channels || row < 0 || row >= view.height || col < 0 || col >= view.width) throw new RangeError("Tensor coordinate");
  return view.values[(channel * view.height + row) * view.width + col];
}

export function channelStats(view, channel) {
  if (!Number.isInteger(channel) || channel < 0 || channel >= view.channels) throw new RangeError("Channel");
  let min = Infinity, max = -Infinity, sum = 0;
  const start = channel * view.height * view.width;
  for (let i = 0; i < view.height * view.width; i++) {
    const v = view.values[start + i]; min = Math.min(min, v); max = Math.max(max, v); sum += v;
  }
  return { min, max, mean: sum / (view.height * view.width) };
}

export function tensorRange(view, channel = null) {
  if (channel !== null) return channelStats(view, channel);
  let min = Infinity, max = -Infinity;
  for (const v of view.values) { min = Math.min(min, v); max = Math.max(max, v); }
  return { min, max };
}

/** Real Conv1 cross-correlation at one output position; bias is added once. */
export function conv1At(bundle,sampleIndex,outChannel,row,col){
  const input=viewForKey(bundle,sampleIndex,"input"),output=viewForKey(bundle,sampleIndex,"conv1_pre");
  if(![outChannel,row,col].every(Number.isInteger)||outChannel<0||outChannel>=32||row<0||row>=26||col<0||col>=26)throw new RangeError("Conv1 coordinate");
  const weights=bundle.weights.conv1.weight.values,patch=[],kernel=[],products=[];
  for(let m=0;m<3;m++)for(let n=0;n<3;n++){
    const x=tensorValue(input,0,row+m,col+n),w=weights[outChannel*9+m*3+n];
    patch.push(x);kernel.push(w);products.push(x*w);
  }
  const bias=bundle.weights.conv1.bias.values[outChannel],calculated=products.reduce((a,b)=>a+b,0)+bias,expected=tensorValue(output,outChannel,row,col);
  return {patch,kernel,products,bias,calculated,expected,error:Math.abs(calculated-expected)};
}

/** Recalculate all 32 genuine Conv2 local contributions without modifying tensors. */
export function conv2At(bundle, sampleIndex, outChannel, row, col, inputChannel = 0) {
  const input = tensorFor(bundle, sampleIndex, "conv2", "pre", "input");
  const output = tensorFor(bundle, sampleIndex, "conv2", "pre", "output");
  if (![outChannel,row,col,inputChannel].every(Number.isInteger) || outChannel < 0 || outChannel >= 64 || row < 0 || row >= 11 || col < 0 || col >= 11 || inputChannel < 0 || inputChannel >= 32) throw new RangeError("Conv2 coordinate");
  const w = bundle.weights.conv2.weight.values, biases = bundle.weights.conv2.bias.values;
  const contributions = [], patch = [], kernel = [];
  for (let ci = 0; ci < 32; ci++) {
    let sum = 0;
    for (let m = 0; m < 3; m++) for (let n = 0; n < 3; n++) {
      const x = tensorValue(input, ci, row + m, col + n);
      const weight = w[((outChannel * 32 + ci) * 3 + m) * 3 + n];
      sum += x * weight;
      if (ci === inputChannel) { patch.push(x); kernel.push(weight); }
    }
    contributions.push(sum);
  }
  const bias = biases[outChannel], calculated = contributions.reduce((a,b)=>a+b,0)+bias;
  const expected = tensorValue(output,outChannel,row,col);
  return { contributions, bias, calculated, expected, error: Math.abs(calculated-expected), patch, kernel, outChannel, row, col, inputChannel };
}
