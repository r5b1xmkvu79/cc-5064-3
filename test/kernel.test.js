import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  FORMATS,
  MODES,
  UNORM24_MAX,
  projectDepth,
  reconstructDistance,
  quantize,
  depthCode,
  nextCode,
  distanceStep,
  analyze,
  failureDistance,
  sameCode,
  distinguishable,
  createBlockRecord,
  remapBlock,
  historicalSampleConvertible,
  sweep,
  recommend,
} from '../src/depth-kernel.js';

const approx = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} !~ ${b}`);

test('projection endpoints: near->0/1, far->1/0 (finite)', () => {
  const p = { near: 0.1, far: 100 };
  approx(projectDepth(0.1, { ...p, mode: MODES.CONVENTIONAL }), 0);
  approx(projectDepth(100, { ...p, mode: MODES.CONVENTIONAL }), 1);
  approx(projectDepth(0.1, { ...p, mode: MODES.REVERSED }), 1);
  approx(projectDepth(100, { ...p, mode: MODES.REVERSED }), 0);
});

test('projection endpoints with infinite far plane', () => {
  approx(projectDepth(0.1, { near: 0.1, far: Infinity, mode: MODES.CONVENTIONAL }), 0);
  approx(projectDepth(1e9, { near: 0.1, far: Infinity, mode: MODES.CONVENTIONAL }), 1 - 1e-10);
  approx(projectDepth(0.1, { near: 0.1, far: Infinity, mode: MODES.REVERSED }), 1);
  approx(projectDepth(1e9, { near: 0.1, far: Infinity, mode: MODES.REVERSED }), 1e-10);
});

test('depth->distance round trip inverts projection in every convention', () => {
  for (const mode of [MODES.CONVENTIONAL, MODES.REVERSED]) {
    for (const far of [100, Infinity]) {
      const p = { near: 0.25, far, mode };
      for (const d of [0.26, 1, 10, 75]) {
        approx(reconstructDistance(projectDepth(d, p), p), d, 1e-12);
      }
    }
  }
});

test('quantization: UNORM24 has 2^24 cells, FLOAT32 honors ulp near zero', () => {
  assert.equal(depthCode(0, FORMATS.UNORM24), 0);
  assert.equal(depthCode(1, FORMATS.UNORM24), UNORM24_MAX);
  assert.equal(depthCode(0.5, FORMATS.UNORM24), Math.round(0.5 * UNORM24_MAX));
  assert.equal(nextCode(0, FORMATS.FLOAT32, +1), 2 ** -149);
  // ulp just below 1.0 belongs to the [0.5,1) bin -> 2^-24
  assert.equal(nextCode(Math.fround(1 - 2 ** -24), FORMATS.FLOAT32, +1), 1);
  assert.equal(nextCode(1, FORMATS.FLOAT32, -1), Math.fround(1 - 2 ** -24));
  assert.equal(quantize(0.1, FORMATS.FLOAT32), Math.fround(0.1));
});

test('reconstruction from quantized depth stays inside the represented distance cell', () => {
  for (const mode of [MODES.CONVENTIONAL, MODES.REVERSED]) {
    const p = { near: 0.1, far: 1000, mode };
    const a = analyze(50, p, FORMATS.UNORM24);
    approx(a.reconstructed, reconstructDistance(a.storedDepth, p));
    // Reconstructed distance must lie within one distance-step cell of truth.
    assert.ok(Math.abs(a.errorDistance) <= a.stepDistance * 1.01);
  }
});

test('distance step grows ~d^2 on conventional UNORM24 (perspective non-linearity)', () => {
  const p = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const s1 = distanceStep(1, p, FORMATS.UNORM24);
  const s100 = distanceStep(100, p, FORMATS.UNORM24);
  assert.ok(s100 > s1 * 5000, `step at 100 (${s100}) should be ~1e4x step at 1 (${s1})`);
});

test('reversed + float32 keeps fine resolution at extreme far distance', () => {
  const near = 0.1;
  const gap = 1;
  const target = 1e5;
  const bad = { near, far: Infinity, mode: MODES.CONVENTIONAL };
  const good = { near, far: Infinity, mode: MODES.REVERSED };
  assert.equal(distinguishable(target, target + gap, bad, FORMATS.FLOAT32), false);
  assert.equal(distinguishable(target, target + gap, good, FORMATS.FLOAT32), true);
  assert.ok(failureDistance(good, FORMATS.FLOAT32, gap) > target);
});

test('reversed alone (UNORM24) does NOT improve far precision vs conventional', () => {
  const p1 = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const p2 = { near: 0.1, far: 1000, mode: MODES.REVERSED };
  const d = 500;
  const s1 = distanceStep(d, p1, FORMATS.UNORM24);
  const s2 = distanceStep(d, p2, FORMATS.UNORM24);
  approx(s1, s2, 1e-9);
});

test('near-coplanar plates: failure distance is the limit of resolvable gap', () => {
  const p = { near: 0.05, far: 100, mode: MODES.CONVENTIONAL };
  const fail = failureDistance(p, FORMATS.UNORM24, 0.01);
  assert.ok(fail > p.near && fail < p.far);
  assert.equal(distinguishable(fail * 0.5, fail * 0.5 + 0.01, p, FORMATS.UNORM24), true);
  assert.equal(sameCode(fail * 1.5, fail * 1.5 + 0.01, p, FORMATS.UNORM24), true);
});

test('distance-mode block remaps exactly after clip change; code-mode can go undecided', () => {
  const old = { near: 0.1, far: 100, mode: MODES.CONVENTIONAL };
  const next = { near: 1, far: 50, mode: MODES.CONVENTIONAL };
  const plates = [{ id: 'A', distance: 20 }, { id: 'B', distance: 20.0005 }];

  const distRecord = createBlockRecord({ candidates: plates, params: old, format: FORMATS.UNORM24, store: 'distance' });
  const remap1 = remapBlock(distRecord, next);
  assert.equal(remap1.convertible, true);
  assert.equal(remap1.winnerId, 'A');
  assert.equal(historicalSampleConvertible(distRecord, next), true);

  // Force both plates into one old code cell.
  const close = [{ id: 'A', distance: 20 }, { id: 'B', distance: 20.0000001 }];
  const codeRecord = createBlockRecord({ candidates: close, params: old, format: FORMATS.UNORM24, store: 'code' });
  const remap2 = remapBlock(codeRecord, next);
  assert.equal(remap2.status, 'undecided');
  assert.equal(remap2.convertible, false);
});

test('code-only block with disjoint new cells still yields a deterministic winner', () => {
  const old = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const next = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const plates = [{ id: 'A', distance: 50 }, { id: 'B', distance: 52 }];
  const rec = createBlockRecord({ candidates: plates, params: old, format: FORMATS.UNORM24, store: 'code' });
  const r = remapBlock(rec, next);
  assert.equal(r.status, 'remapped');
  assert.equal(r.winnerId, 'A');
});

test('sweep covers camera ranges and flags failures near the far plane', () => {
  const rows = sweep({
    nears: [0.01, 0.1, 1],
    ratios: [100, 10000],
    gap: 0.1,
    targetDistance: 9000,
    formats: [FORMATS.UNORM24, FORMATS.FLOAT32],
    modes: [MODES.CONVENTIONAL, MODES.REVERSED],
  });
  assert.equal(rows.length, 3 * 2 * 2 * 2 * 2);
  const worst = rows.reduce((m, r) => Math.max(m, r.relativeErrorAtTarget), 0);
  assert.ok(worst > 0);
  const anyFail = rows.some((r) => !r.passes);
  assert.ok(anyFail, '0.1m gap at 9km must fail for some configurations');
});

test('recommender picks reversed float32 for large depth range and states failure range', () => {
  const r = recommend({ near: 0.05, far: 5000, gap: 0.02, targetDistances: [1, 1000, 4000] });
  assert.equal(r.choice.format, FORMATS.FLOAT32);
  assert.equal(r.choice.mode, MODES.REVERSED);
  assert.ok(r.choice.passes);
  assert.match(r.failureRange, /d >/);
});
