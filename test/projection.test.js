import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildProjection } from '../src/renderer.js';
import { projectDepth, FORMATS, MODES, normalizeParams } from '../src/depth-kernel.js';

const approx = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} !~ ${b}`);

// Window depth produced by a projection matrix for a view-space point (0,0,-d).
function matrixWindowDepth(proj, d) {
  const clipZ = proj[10] * -d + proj[14];
  const clipW = -proj[11] * d; // w = -z_view = d
  const ndcZ = clipZ / clipW;
  return ndcZ * 0.5 + 0.5; // depthRange [0,1]
}

test('projection matrix window depth matches kernel in every convention', () => {
  const cases = [
    { near: 0.1, far: 100, mode: MODES.CONVENTIONAL },
    { near: 0.1, far: 100, mode: MODES.REVERSED },
    { near: 0.05, far: Infinity, mode: MODES.CONVENTIONAL },
    { near: 0.05, far: Infinity, mode: MODES.REVERSED },
  ];
  for (const c of cases) {
    const proj = buildProjection({ fovy: 1, aspect: 1, ...c });
    const p = normalizeParams(c);
    for (const d of [c.near * 1.5, c.near * 50, c.far === Infinity ? 5000 : c.far * 0.9]) {
      // Matrix evaluation reorders the same arithmetic; allow ~1e-9 relative.
      approx(matrixWindowDepth(proj, d), projectDepth(d, p), 1e-8);
    }
  }
});

test('reversed matrix clears the expected end of the window range', () => {
  const rev = buildProjection({ fovy: 1, aspect: 1, near: 0.1, far: 100, mode: MODES.REVERSED });
  approx(matrixWindowDepth(rev, 0.1), 1);   // near -> 1
  approx(matrixWindowDepth(rev, 100), 0);   // far  -> 0
  const conv = buildProjection({ fovy: 1, aspect: 1, near: 0.1, far: 100, mode: MODES.CONVENTIONAL });
  approx(matrixWindowDepth(conv, 0.1), 0);
  approx(matrixWindowDepth(conv, 100), 1);
});

test('shader-side reconstruction formula equals kernel reconstructDistance', () => {
  // Mirrors the GLSL in renderer.js post-process pass.
  function shaderReconstruct(hw, near, far, reverse) {
    if (far <= 0) return reverse ? near / hw : near / (1 - hw);
    if (reverse) return (near * far) / (near + hw * (far - near));
    return (near * far) / (far - hw * (far - near));
  }
  for (const mode of [MODES.CONVENTIONAL, MODES.REVERSED]) {
    const reverse = mode === MODES.REVERSED;
    for (const far of [100, Infinity]) {
      const p = normalizeParams({ near: 0.2, far, mode });
      for (const d of [0.3, 5, 40, 90]) {
        if (far !== Infinity && d > far) continue;
        const z = projectDepth(d, p);
        const hw = z; // GLSL uses the same stored depth
        approx(shaderReconstruct(hw, 0.2, far === Infinity ? 0 : far, reverse), d, 1e-9);
      }
    }
  }
  void FORMATS;
});
