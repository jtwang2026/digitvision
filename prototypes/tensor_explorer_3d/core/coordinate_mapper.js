/** Axis contract: column → +X, row → −Y, channel → +Z. */
export function tensorIndex(channel, row, col, height, width) {
  if (![channel,row,col,height,width].every(Number.isInteger) || channel < 0 || row < 0 || col < 0 || row >= height || col >= width) throw new RangeError("NCHW index");
  return (channel * height + row) * width + col;
}

export function planePoint(row, col, height, width, physicalWidth, physicalHeight) {
  if (row < 0 || row >= height || col < 0 || col >= width) throw new RangeError("Spatial coordinate");
  return { x: ((col + 0.5) / width - 0.5) * physicalWidth, y: (0.5 - (row + 0.5) / height) * physicalHeight };
}

export function uvToCoordinate(u, v, height, width) {
  const col = Math.max(0, Math.min(width - 1, Math.floor(u * width)));
  const row = Math.max(0, Math.min(height - 1, Math.floor((1 - v) * height)));
  return { row, col };
}

export function atlasCoordinate(x, y, canvasWidth, canvasHeight, columns, channelCount) {
  const rows = Math.ceil(channelCount / columns), col = Math.floor(x / (canvasWidth / columns)), row = Math.floor(y / (canvasHeight / rows));
  const channel = row * columns + col;
  return channel >= 0 && channel < channelCount && col >= 0 && col < columns ? channel : null;
}
