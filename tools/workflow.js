// End-to-end simulation of the in-app workflow using only the kernel + block
// store: pick far plates -> dolly in (local refresh keeps history) -> clip
// jump (historical sample conversion). Run: node tools/workflow.js
import { BlockStore, BLOCK_STATE } from '../src/block-store.js';
import { PLATE_IDS } from '../src/geometry.js';
import {
  FORMATS,
  MODES,
  projectDepth,
  quantize,
  reconstructDistance,
  distinguishable,
  normalizeParams,
} from '../src/depth-kernel.js';

const rb = (samples) => ({
  exact: new Float32Array(samples.map((s) => s.distance)),
  hardware: new Float32Array(samples.length),
  ids: new Uint32Array(samples.map((s) => s.id)),
  color: new Uint8Array(samples.length * 4),
  w: samples.length,
  h: 1,
});

const scenarios = [
  { name: 'UNORM24 常规', format: FORMATS.UNORM24, mode: MODES.CONVENTIONAL },
  { name: 'FLOAT32 反向无限远', format: FORMATS.FLOAT32, mode: MODES.REVERSED, infinite: true },
];

for (const sc of scenarios) {
  console.log(`\n=== ${sc.name} ===`);
  const store = new BlockStore({ blockSize: 64, format: sc.format, store: 'code' });
  store.setViewport(512, 384);

  // 1. Cartographer picks the two far plates while camera is distant.
  const far = 60;
  const gap = 0.5;
  const p0 = normalizeParams({ near: 0.1, far: sc.infinite ? Infinity : 1000, mode: sc.mode });
  store.ingest(
    3, 2,
    rb([
      { id: PLATE_IDS.FAR_A, distance: far },
      { id: PLATE_IDS.FAR_B, distance: far + gap },
    ]),
    p0,
  );
  const block0 = store.get(3, 2);
  console.log(`远景拾取后状态: ${block0.state}, 板: ${block0.record.candidates.map((c) => c.id).join(',')}`);

  // Readback cross-check: hardware depth vs kernel quantization.
  for (const d of [far, far + gap]) {
    const z = projectDepth(d, p0);
    const stored = quantize(z, sc.format);
    const back = reconstructDistance(stored, p0);
    console.log(
      `  d=${d} z=${z.toPrecision(5)} 存储=${stored.toPrecision(5)} 重建=${back.toFixed(4)} 误差=${(
        back - d
      ).toExponential(2)}`,
    );
  }
  console.log(
    `  两板可分辨: ${distinguishable(far, far + gap, p0, sc.format)}（遮挡次序 A 在前）`,
  );

  // 2. Dolly in: only the center block refreshes; the far block keeps history.
  const center = store.blockAtPixel(256, 192);
  store.ingest(
    center.col,
    center.row,
    rb([{ id: PLATE_IDS.NEAR_A, distance: 12 }]),
    p0,
  );
  console.log(`推近局部刷新中心块；远景历史块仍为: ${store.get(3, 2).state}`);

  // 3. Clip jump: near and far move so old and new depths land in new ranges.
  const p1 = normalizeParams({ near: 1, far: sc.infinite ? Infinity : 200, mode: sc.mode });
  const summary = store.updateParams(p1);
  console.log(
    `裁面跳变 n=${p1.near} f=${p1.far}: 重映射${summary.remapped} 未决${summary.undecided} 抢占${summary.tie} 空${summary.empty}`,
  );
  const after = store.get(3, 2);
  console.log(`远景历史块新状态: ${after.state}, 获胜板: ${after.winnerId ?? '(未决)'}`);
  if (after.state === BLOCK_STATE.UNDECIDED) {
    console.log('  -> 仅量化码且新码格相交，拾取系统判定历史样本不可换算，需重绘');
  }
}

// Near-coplanar plates: gap below one UNORM24 distance cell must produce an
// undecided code-only block after the clip change, while distance mode remaps.
console.log('\n=== 近共面薄板：仅量化码 -> 未决；视空间距离 -> 可重映射 ===');
{
  const p0 = normalizeParams({ near: 0.1, far: 1000, mode: MODES.CONVENTIONAL });
  const p1 = normalizeParams({ near: 5, far: 200, mode: MODES.CONVENTIONAL });
  const tiny = rb([
    { id: PLATE_IDS.FAR_A, distance: 60 },
    { id: PLATE_IDS.FAR_B, distance: 60.000001 },
  ]);
  const codeStore = new BlockStore({ blockSize: 64, format: FORMATS.UNORM24, store: 'code' });
  codeStore.setViewport(256, 256);
  codeStore.ingest(0, 0, tiny, p0);
  const s1 = codeStore.updateParams(p1);
  console.log(`仅量化码: 未决=${s1.undecided} 抢占=${s1.tie} 状态=${codeStore.get(0, 0).state}`);

  const distStore = new BlockStore({ blockSize: 64, format: FORMATS.UNORM24, store: 'distance' });
  distStore.setViewport(256, 256);
  distStore.ingest(0, 0, tiny, p0);
  const s2 = distStore.updateParams(p1);
  console.log(
    `视距保存: 重映射=${s2.remapped} 抢占(tie)=${s2.tie} 状态=${distStore.get(0, 0).state}（距离仍可换算，但两板同码无法分胜负）`,
  );

  // A separable pair under a gentler clip change -> distance block remaps
  // exactly and keeps the physically nearer plate as winner.
  const separable = rb([
    { id: PLATE_IDS.FAR_A, distance: 60 },
    { id: PLATE_IDS.FAR_B, distance: 60.5 },
  ]);
  const p2 = normalizeParams({ near: 0.2, far: 900, mode: MODES.CONVENTIONAL });
  const distStore2 = new BlockStore({ blockSize: 64, format: FORMATS.UNORM24, store: 'distance' });
  distStore2.setViewport(256, 256);
  distStore2.ingest(0, 0, separable, p0);
  const s3 = distStore2.updateParams(p2);
  console.log(
    `温和裁面变化(可分辨板): 重映射=${s3.remapped} 获胜板=${distStore2.get(0, 0).winnerId}（应为 ${PLATE_IDS.FAR_A}）`,
  );
}
