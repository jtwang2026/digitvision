const clamp = v => Math.max(0, Math.min(1, v));
/** Color encodes signed response. Zero is a neutral dark cell. */
export function responseColor(value, scale) {
  const magnitude = clamp(Math.abs(value) / Math.max(scale, 1e-12));
  if (magnitude < 1e-10) return [18, 29, 46, 255];
  const base = value < 0 ? [251, 113, 133] : [67, 209, 231];
  return [Math.round(18 + (base[0]-18)*magnitude), Math.round(29+(base[1]-29)*magnitude), Math.round(46+(base[2]-46)*magnitude), 255];
}

export function colorScale(view, mode, channel) {
  const values = view.values, start = mode === "channel" ? channel * view.height * view.width : 0;
  const end = mode === "channel" ? start + view.height * view.width : values.length;
  let max = 0; for (let i = start; i < end; i++) max = Math.max(max, Math.abs(values[i]));
  return max || 1;
}

/** DataTexture.flipY=false: store visual row zero in the last texture row. */
export function textureBytes(view, channel, scale) {
  const out = new Uint8Array(view.width * view.height * 4);
  for (let row = 0; row < view.height; row++) for (let col = 0; col < view.width; col++) {
    const value = view.values[(channel * view.height + row) * view.width + col];
    out.set(responseColor(value, scale), ((view.height - 1 - row) * view.width + col) * 4);
  }
  return out;
}
