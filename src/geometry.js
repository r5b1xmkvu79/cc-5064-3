// Scene geometry: pairs of thin plates near-coplanar in depth, a ground grid.
// Each vertex carries the stable object id of its plate; physical plate
// thickness is never adjusted to hide z-fighting.

export const PLATE_IDS = {
  NEAR_A: 1,
  NEAR_B: 2,
  FAR_A: 3,
  FAR_B: 4,
  GROUND: 9,
};

export const ID_LABELS = {
  1: 'NEAR-A',
  2: 'NEAR-B',
  3: 'FAR-A',
  4: 'FAR-B',
  9: 'GROUND',
};

export const ID_COLORS = {
  1: [0.85, 0.35, 0.25],
  2: [0.95, 0.65, 0.2],
  3: [0.3, 0.6, 0.9],
  4: [0.35, 0.8, 0.75],
  9: [0.4, 0.42, 0.48],
};

// Output: positions Float32Array (xyz per vertex), ids Float32Array (one per
// vertex), indices Uint32Array (triangles).
function pushBox(out, cx, cy, cz, hx, hy, hz, id) {
  const verts = [
    [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
    [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
  ];
  const faces = [
    [0, 3, 2, 0, 2, 1], [4, 5, 6, 4, 6, 7],
    [0, 1, 5, 0, 5, 4], [3, 7, 6, 3, 6, 2],
    [1, 2, 6, 1, 6, 5], [0, 4, 7, 0, 7, 3],
  ];
  const base = out.positions.length / 3;
  for (const v of verts) {
    out.positions.push(cx + v[0], cy + v[1], cz + v[2]);
    out.ids.push(id);
  }
  for (const f of faces) for (const idx of f) out.indices.push(base + idx);
}

export function buildScene(state) {
  const out = { positions: [], ids: [], indices: [] };
  const t = 0.02; // plate half-thickness (physical geometry, fixed)
  const farHalf = state.farHalfExtent ?? 3;

  // Far pair: cartographer picks these first while the camera is distant.
  pushBox(out, 0, state.farY, -state.farDistance, farHalf, farHalf, t, PLATE_IDS.FAR_A);
  pushBox(out, 0, state.farY, -(state.farDistance + state.farGap), farHalf, farHalf, t, PLATE_IDS.FAR_B);

  // Near pair: becomes relevant after dollying in.
  pushBox(out, 4, state.nearY, -state.nearDistance, 2, 2, t, PLATE_IDS.NEAR_A);
  pushBox(out, 4, state.nearY, -(state.nearDistance + state.nearGap), 2, 2, t, PLATE_IDS.NEAR_B);

  const mesh = {
    positions: new Float32Array(out.positions),
    ids: new Float32Array(out.ids),
    indices: new Uint32Array(out.indices),
  };

  // Ground grid lines (LINES), each vertex tagged GROUND id.
  const gPos = [];
  const gIds = [];
  const extent = 600;
  const step = 10;
  const y = -3.05;
  const pushLine = (x0, z0, x1, z1) => {
    gPos.push(x0, y, z0, x1, y, z1);
    gIds.push(PLATE_IDS.GROUND, PLATE_IDS.GROUND);
  };
  for (let x = -extent; x <= extent; x += step) pushLine(x, 0, x, -extent);
  for (let z = 0; z >= -extent; z -= step) pushLine(-extent, z, extent, z);

  const ground = { positions: new Float32Array(gPos), ids: new Float32Array(gIds) };
  return { mesh, ground };
}
