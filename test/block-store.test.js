import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BlockStore, BLOCK_STATE } from '../src/block-store.js';
import { PLATE_IDS } from '../src/geometry.js';
import { FORMATS } from '../src/depth-kernel.js';

function fakeReadback(samples) {
  // samples: [{id, distance}] -> one pixel each, integer id in R32UI channel.
  const n = samples.length;
  const exact = new Float32Array(n);
  const hardware = new Float32Array(n);
  const ids = new Uint32Array(n);
  const color = new Uint8Array(n * 4);
  samples.forEach((s, i) => {
    exact[i] = s.distance;
    hardware[i] = 0.5;
    ids[i] = s.id;
    color[i * 4 + 3] = 255;
  });
  return { exact, hardware, color, ids, w: n, h: 1 };
}

function makeStore(storeMode, format = FORMATS.UNORM24) {
  const store = new BlockStore({ blockSize: 64, format, store: storeMode });
  store.setViewport(256, 256);
  return store;
}

test('distance-mode block survives a clip change and keeps the near plate winner', () => {
  const store = makeStore('distance');
  store.ingest(
    0, 0,
    fakeReadback([
      { id: PLATE_IDS.FAR_A, distance: 60 },
      { id: PLATE_IDS.FAR_B, distance: 60.5 },
    ]),
    { near: 0.1, far: 1000, mode: 'conventional' },
  );
  const summary = store.updateParams({ near: 1, far: 500, mode: 'conventional' });
  assert.equal(summary.undecided, 0);
  const block = store.get(0, 0);
  assert.equal(block.winnerId, PLATE_IDS.FAR_A);
  assert.equal(block.state, BLOCK_STATE.REMAPPED);
});

test('code-only block with two plates in one old cell goes undecided after clip change', () => {
  const store = makeStore('code');
  // Gap far smaller than one UNORM24 distance cell at d=60 (n=0.1, f=1000).
  store.ingest(
    0, 0,
    fakeReadback([
      { id: PLATE_IDS.FAR_A, distance: 60 },
      { id: PLATE_IDS.FAR_B, distance: 60.000001 },
    ]),
    { near: 0.1, far: 1000, mode: 'conventional' },
  );
  const summary = store.updateParams({ near: 5, far: 200, mode: 'conventional' });
  assert.equal(summary.undecided, 1);
  assert.equal(store.get(0, 0).state, BLOCK_STATE.UNDECIDED);
});

test('code-only block with disjoint cells remaps and names the nearer plate', () => {
  const store = makeStore('code');
  store.ingest(
    0, 0,
    fakeReadback([
      { id: PLATE_IDS.FAR_A, distance: 60 },
      { id: PLATE_IDS.FAR_B, distance: 63 },
    ]),
    { near: 0.1, far: 1000, mode: 'conventional' },
  );
  const summary = store.updateParams({ near: 0.1, far: 1000, mode: 'conventional' });
  assert.equal(summary.undecided, 0);
  assert.equal(store.get(0, 0).winnerId, PLATE_IDS.FAR_A);
});

test('FLOAT32 reversed block resolves a gap UNORM24 collapses', () => {
  const unorm = makeStore('code', FORMATS.UNORM24);
  const f32 = makeStore('code', FORMATS.FLOAT32);
  const samples = fakeReadback([
    { id: PLATE_IDS.FAR_A, distance: 1000 },
    { id: PLATE_IDS.FAR_B, distance: 1000.5 },
  ]);
  unorm.ingest(0, 0, samples, { near: 0.1, far: Infinity, mode: 'conventional' });
  f32.ingest(0, 0, samples, { near: 0.1, far: Infinity, mode: 'reversed' });
  // Under conventional infinite far, both samples collapse; reversed f32 keeps
  // two distinct codes and therefore a deterministic winner.
  const u = unorm.updateParams({ near: 0.1, far: Infinity, mode: 'conventional' });
  const f = f32.updateParams({ near: 0.1, far: Infinity, mode: 'reversed' });
  assert.ok(u.undecided + u.tie >= 1 || unorm.get(0, 0).winnerId !== PLATE_IDS.FAR_B);
  assert.equal(f.undecided, 0);
  assert.equal(f32.get(0, 0).winnerId, PLATE_IDS.FAR_A);
});
