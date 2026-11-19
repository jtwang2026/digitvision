export const OPERATIONS=Object.freeze({
  conv1:{input:"input",outputPre:"conv1_pre",outputRelu:"conv1_relu",kernel:3,stride:1},
  conv2:{input:"pool1",outputPre:"conv2_pre",outputRelu:"conv2_relu",kernel:3,stride:1},
  pool1:{input:"conv1_relu",output:"pool1",kernel:2,stride:2},
  pool2:{input:"conv2_relu",output:"pool2",kernel:2,stride:2},
  relu1:{input:"conv1_pre",output:"conv1_relu",kernel:1,stride:1},
  relu2:{input:"conv2_pre",output:"conv2_relu",kernel:1,stride:1},
});
export function processingSpec(bundle,sample,operation,activation="pre"){
  const op=OPERATIONS[operation],t=bundle.samples[sample]?.tensors;
  if(!op||!t)throw new RangeError("Processing operation/sample");
  const outputKey=op.output||op[activation==="relu"?"outputRelu":"outputPre"];
  const input=t[op.input],output=t[outputKey];
  if(!input||!output)throw Error("Exported Tensor unavailable");
  const [,ic,ih,iw]=input.shape,[,oc,oh,ow]=output.shape;
  if(Math.floor((ih-op.kernel)/op.stride)+1!==oh||Math.floor((iw-op.kernel)/op.stride)+1!==ow)throw Error("Processing shape mismatch");
  if(operation!=="conv2"&&operation!=="conv1"&&ic!==oc)throw Error("Channel mismatch");
  return {...op,operation,inputKey:op.input,outputKey,input,output,inputShape:input.shape,outputShape:output.shape,inputChannels:ic,outputChannels:oc,inputHeight:ih,inputWidth:iw,outputHeight:oh,outputWidth:ow,count:oh*ow};
}
export function spatialPosition(spec,index,direction=1){
  if(!Number.isInteger(index)||index<0||index>=spec.count)throw new RangeError("Spatial index");
  const n=direction===-1?spec.count-1-index:index;
  const row=Math.floor(n/spec.outputWidth),colInRow=n%spec.outputWidth;
  const col=row%2?spec.outputWidth-1-colInRow:colInRow;
  return {index:n,inputRow:row*spec.stride,inputCol:col*spec.stride,outputRow:row,outputCol:col,kernelHeight:spec.kernel,kernelWidth:spec.kernel,stride:spec.stride};
}
export function cursorAt(spec,progress,direction=1){
  const position=Math.min(spec.count-1,Math.max(0,progress*spec.count));
  const index=Math.floor(position),next=Math.min(spec.count-1,index+1),a=spatialPosition(spec,index,direction),b=spatialPosition(spec,next,direction),t=position-index;
  return {...a,visualInputRow:a.inputRow+(b.inputRow-a.inputRow)*t,visualInputCol:a.inputCol+(b.inputCol-a.inputCol)*t,visualOutputRow:a.outputRow+(b.outputRow-a.outputRow)*t,visualOutputCol:a.outputCol+(b.outputCol-a.outputCol)*t,phase:t,direction,progress};
}
