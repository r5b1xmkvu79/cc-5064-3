// 2D canvas charts: precision distribution curve and block-state map.

import { FORMATS, MODES, distanceStep, normalizeParams } from './depth-kernel.js';
import { BLOCK_STATE } from './block-store.js';

const COLORS = {
  [FORMATS.UNORM24]: '#ffb454',
  [FORMATS.FLOAT32]: '#5aa2ff',
  gap: '#ff5d5d',
  grid: '#2a3140',
  text: '#8b94a5',
};

function logRange(min, max, n) {
  const out = [];
  const a = Math.log10(min);
  const b = Math.log10(max);
  for (let i = 0; i < n; i++) out.push(10 ** (a + ((b - a) * i) / (n - 1)));
  return out;
}

export function drawPrecisionChart(canvas, state) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#10141a';
  ctx.fillRect(0, 0, W, H);

  const near = state.near;
  const maxD = state.infiniteFar ? Math.max(1e5, state.farDistance * 4) : state.far;
  const minD = near;
  const dists = logRange(minD, maxD, 240);

  const configs = [
    { format: FORMATS.UNORM24, mode: MODES.CONVENTIONAL, label: 'UNORM24 常规' },
    { format: FORMATS.UNORM24, mode: MODES.REVERSED, label: 'UNORM24 反向' },
    { format: FORMATS.FLOAT32, mode: MODES.CONVENTIONAL, label: 'FLOAT32 常规' },
    { format: FORMATS.FLOAT32, mode: MODES.REVERSED, label: 'FLOAT32 反向' },
  ];

  const steps = configs.map((c) => {
    const params = normalizeParams({ near, far: state.infiniteFar ? Infinity : state.far, mode: c.mode });
    return dists.map((d) => distanceStep(d, params, c.format));
  });
  const allSteps = steps.flat().filter((v) => Number.isFinite(v) && v > 0);
  const minS = Math.min(...allSteps);
  const maxS = Math.max(...allSteps);

  const x = (d) => ((Math.log10(d) - Math.log10(minD)) / (Math.log10(maxD) - Math.log10(minD))) * (W - 60) + 50;
  const y = (s) =>
    H - 24 - ((Math.log10(s) - Math.log10(minS)) / (Math.log10(maxS) - Math.log10(minS))) * (H - 44);

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = COLORS.text;
  ctx.font = '10px monospace';
  for (let i = 0; i <= 4; i++) {
    const yy = 20 + ((H - 44) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(50, yy);
    ctx.lineTo(W - 10, yy);
    ctx.stroke();
    const sv = 10 ** (Math.log10(maxS) - ((Math.log10(maxS) - Math.log10(minS)) * i) / 4);
    ctx.fillText(sv.toExponential(1), 4, yy + 3);
  }

  // Gap threshold line: surfaces closer than this in distance merge when the
  // step curve rises above it.
  const gapY = y(state.farGap);
  ctx.strokeStyle = COLORS.gap;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(50, gapY);
  ctx.lineTo(W - 10, gapY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.gap;
  ctx.fillText(`gap=${state.farGap}`, W - 90, gapY - 4);

  configs.forEach((c, ci) => {
    ctx.strokeStyle = ci < 2 ? COLORS[FORMATS.UNORM24] : COLORS[FORMATS.FLOAT32];
    ctx.lineWidth = ci % 2 === 0 ? 1.4 : 2.4;
    ctx.setLineDash(ci % 2 === 0 ? [3, 3] : []);
    ctx.beginPath();
    steps[ci].forEach((s, i) => {
      const px = x(dists[i]);
      const py = y(s);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Mark the actual plate distances.
  for (const [label, d] of [
    ['NEAR', state.nearDistance],
    ['FAR', state.farDistance],
  ]) {
    if (d < minD || d > maxD) continue;
    const px = x(d);
    ctx.strokeStyle = '#57d98a';
    ctx.beginPath();
    ctx.moveTo(px, 18);
    ctx.lineTo(px, H - 24);
    ctx.stroke();
    ctx.fillStyle = '#57d98a';
    ctx.fillText(label, px + 3, 16);
  }

  // Legend.
  let lx = 54;
  ctx.font = '11px sans-serif';
  configs.forEach((c, ci) => {
    ctx.strokeStyle = ci < 2 ? COLORS[FORMATS.UNORM24] : COLORS[FORMATS.FLOAT32];
    ctx.lineWidth = 2;
    ctx.setLineDash(ci % 2 === 0 ? [3, 3] : []);
    ctx.beginPath();
    ctx.moveTo(lx, H - 8);
    ctx.lineTo(lx + 16, H - 8);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.text;
    ctx.fillText(c.label, lx + 20, H - 4);
    lx += 20 + ctx.measureText(c.label).width + 14;
  });
}

const STATE_COLORS = {
  [BLOCK_STATE.FRESH]: '#57d98a',
  [BLOCK_STATE.REMAPPED]: '#5aa2ff',
  [BLOCK_STATE.UNDECIDED]: '#ffb454',
  [BLOCK_STATE.TIE]: '#ff5d5d',
  [BLOCK_STATE.EMPTY]: '#3a4150',
};

export function drawBlockMap(canvas, storeA, storeB) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#10141a';
  ctx.fillRect(0, 0, W, H);
  if (!storeA || !storeB || storeA.cols === 0) return;

  const drawOne = (store, ox, halfW, title) => {
    const bw = halfW / store.cols;
    const bh = (H - 30) / store.rows;
    ctx.fillStyle = COLORS.text;
    ctx.font = '11px sans-serif';
    ctx.fillText(title, ox + 4, 14);
    for (let r = 0; r < store.rows; r++) {
      for (let c = 0; c < store.cols; c++) {
        const block = store.get(c, r);
        const state = block ? block.state : BLOCK_STATE.EMPTY;
        ctx.fillStyle = STATE_COLORS[state] ?? STATE_COLORS[BLOCK_STATE.EMPTY];
        ctx.fillRect(ox + c * bw + 1, 20 + r * bh + 1, Math.max(1, bw - 2), Math.max(1, bh - 2));
      }
    }
  };
  drawOne(storeA, 0, W / 2 - 8, '路径 A 区块');
  drawOne(storeB, W / 2 + 8, W / 2 - 8, '路径 B 区块');

  let lx = 8;
  ctx.font = '10px sans-serif';
  Object.entries(STATE_COLORS).forEach(([name, color]) => {
    ctx.fillStyle = color;
    ctx.fillRect(lx, H - 12, 10, 10);
    ctx.fillStyle = COLORS.text;
    ctx.fillText(name, lx + 13, H - 3);
    lx += 13 + ctx.measureText(name).width + 10;
  });
}
