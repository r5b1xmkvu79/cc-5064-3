// Camera-range sweep: depth format x projection convention x far-plane type.
// Pure kernel, run with: npm run sweep
import { FORMATS, MODES, sweep, recommend, failureDistance } from '../src/depth-kernel.js';

const gap = Number(process.env.GAP ?? 0.05);
const nears = [0.01, 0.1, 1];
const ratios = [100, 1000, 10000, 100000];

const rows = sweep({
  nears,
  ratios,
  gap,
  targetDistance: undefined,
  formats: [FORMATS.UNORM24, FORMATS.FLOAT32],
  modes: [MODES.CONVENTIONAL, MODES.REVERSED],
});

const fmt = (x) => (x === Infinity ? '   Inf   ' : x.toExponential(2));
console.log(`gap = ${gap} m. failureDistance = last d where d and d+gap get distinct depth codes\n`);
console.log('near     far        format   mode          far-plane  failureDistance   stepAtTarget');
for (const r of rows) {
  console.log(
    `${fmt(r.near)}  ${fmt(r.far)}  ${r.format.padEnd(7)}  ${r.mode.padEnd(12)} ${
      r.infinite ? 'infinite' : 'finite  '
    }  ${fmt(r.failureDistance).padStart(11)}    ${fmt(r.stepAtTarget)} ${r.passes ? '' : ' <-- FAIL @ far'}`,
  );
}

console.log('\nNear-plane behavior (finite far = near * 10000, conventional UNORM24):');
for (const near of nears) {
  const p = { near, far: near * 10000, mode: MODES.CONVENTIONAL };
  console.log(`  near=${fmt(near)}  failureDistance(0.05m)=${fmt(failureDistance(p, FORMATS.UNORM24, 0.05))}`);
}

console.log('\nRecommendation for near=0.05, far=5000, gap=0.02, targets [1, 1000, 4000]:');
const rec = recommend({ near: 0.05, far: 5000, gap: 0.02, targetDistances: [1, 1000, 4000] });
for (const c of rec.combos) {
  console.log(`  ${c.passes ? 'PASS' : 'FAIL'}  ${c.label.padEnd(42)} failureDistance=${fmt(c.failureDistance)}`);
}
console.log(`\nchoice: ${rec.choice.label}`);
console.log(`failure range: ${rec.failureRange}`);
