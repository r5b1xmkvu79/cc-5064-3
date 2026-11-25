// Depth precision kernel — pure math, no DOM / WebGL dependency.
// All distances are positive view-space depths (distance from eye along -Z).
//
// Window-depth conventions (perspective, depthRange [0,1]):
//   conventional finite : z = f(d-n)/((f-n)d)        d=n -> 0, d=f -> 1
//   reversed     finite : z = n(f-d)/((f-n)d)        d=n -> 1, d=f -> 0
//   conventional infinite far (f=Inf): z = 1 - n/d
//   reversed     infinite far (f=Inf): z = n/d

export const FORMATS = Object.freeze({
  UNORM24: 'UNORM24',
  FLOAT32: 'FLOAT32',
});

export const MODES = Object.freeze({
  CONVENTIONAL: 'conventional',
  REVERSED: 'reversed',
});

export const UNORM24_MAX = 2 ** 24 - 1;
const F32_MIN_SUBNORMAL = 2 ** -149;

export function normalizeParams(params = {}) {
  const near = Number(params.near);
  const far = params.far === Infinity || params.far === undefined ? Infinity : Number(params.far);
  const reverse = params.mode === MODES.REVERSED || params.reverse === true;
  if (!(near > 0)) throw new Error('near must be > 0');
  if (far !== Infinity && !(far > near)) throw new Error('far must be > near');
  return { near, far, reverse, mode: reverse ? MODES.REVERSED : MODES.CONVENTIONAL };
}

// Exact (unquantized) window depth for a view-space distance.
export function projectDepth(distance, rawParams) {
  const { near: n, far: f, reverse } = normalizeParams(rawParams);
  if (!(distance >= n)) return reverse ? 1 : 0;
  let z;
  if (f === Infinity) {
    z = reverse ? n / distance : 1 - n / distance;
  } else if (reverse) {
    z = (n * (f - distance)) / ((f - n) * distance);
  } else {
    z = (f * (distance - n)) / ((f - n) * distance);
  }
  if (z < 0) z = 0;
  if (z > 1) z = 1;
  return z;
}

// Reconstruct view-space distance from a (possibly quantized) window depth.
export function reconstructDistance(z, rawParams) {
  const { near: n, far: f, reverse } = normalizeParams(rawParams);
  if (f === Infinity) {
    if (reverse) {
      if (z <= 0) return Infinity;
      return n / z;
    }
    if (z >= 1) return Infinity;
    return n / (1 - z);
  }
  if (reverse) {
    const den = n + z * (f - n);
    return den <= 0 ? Infinity : (n * f) / den;
  }
  const den = f - z * (f - n);
  return den <= 0 ? Infinity : (n * f) / den;
}

// ---- quantization -------------------------------------------------------

export function float32Ulp(z) {
  if (z <= 0) return F32_MIN_SUBNORMAL;
  if (!Number.isFinite(z)) return Infinity;
  const e = Math.floor(Math.log2(z));
  // Exact powers of two are the top of the lower bin; their ulp is 2^(e-23).
  const bin = 2 ** e === z ? e - 1 : e;
  return 2 ** (bin - 23);
}

// Numeric "code" of a window depth. UNORM24 -> integer 0..16777215, FLOAT32 -> f32 number.
export function depthCode(z, format) {
  if (format === FORMATS.UNORM24) {
    const k = Math.round(Math.min(1, Math.max(0, z)) * UNORM24_MAX);
    return Math.min(UNORM24_MAX, Math.max(0, k));
  }
  if (format === FORMATS.FLOAT32) return Math.fround(z);
  throw new Error(`unknown format ${format}`);
}

// Window depth value actually stored in the buffer.
export function quantize(z, format) {
  if (format === FORMATS.UNORM24) return depthCode(z, format) / UNORM24_MAX;
  return Math.fround(z);
}

export function nextCode(code, format, direction) {
  if (format === FORMATS.UNORM24) {
    const k = Math.round(code);
    return Math.min(UNORM24_MAX, Math.max(0, k + direction));
  }
  if (code === 0) return direction > 0 ? F32_MIN_SUBNORMAL : 0;
  const ulp = float32Ulp(code);
  const next = direction > 0 ? Math.fround(code + ulp) : Math.fround(code - ulp);
  return next < 0 ? 0 : next;
}

// Two view-space surfaces land in the same depth code cell.
export function sameCode(d1, d2, params, format) {
  const p = normalizeParams(params);
  return depthCode(projectDepth(d1, p), format) === depthCode(projectDepth(d2, p), format);
}

export function distinguishable(d1, d2, params, format) {
  return !sameCode(d1, d2, params, format);
}

// |dz/dd| of the projection curve; perspective hyperbola derivative.
function depthDerivative(distance, params) {
  const { near: n, far: f } = params;
  if (f === Infinity) return n / (distance * distance);
  return (n * n * f) / ((f - n) * distance * distance);
}

// Distance span covered by one depth code cell at the given distance.
export function distanceStep(distance, params, format) {
  const p = normalizeParams(params);
  const z = projectDepth(distance, p);
  const ulp = format === FORMATS.UNORM24 ? 1 / UNORM24_MAX : float32Ulp(z);
  // dd = dz / |dz/dd|
  return ulp / depthDerivative(distance, p);
}

export function analyze(distance, params, format) {
  const p = normalizeParams(params);
  const z = projectDepth(distance, p);
  const code = depthCode(z, format);
  const zStored = format === FORMATS.UNORM24 ? code / UNORM24_MAX : code;
  const reconstructed = reconstructDistance(zStored, p);
  const errorDistance = reconstructed - distance;
  return {
    distance,
    windowDepth: z,
    code,
    storedDepth: zStored,
    reconstructed,
    errorDistance,
    relativeError: Math.abs(errorDistance) / distance,
    stepDistance: distanceStep(distance, p, format),
  };
}

// Largest distance at which two surfaces separated by `gap` still get distinct codes.
// Beyond it the pair collapses into one code cell (precision failure).
export function failureDistance(params, format, gap) {
  const p = normalizeParams(params);
  const ok = (d) => distinguishable(d, d + gap, p, format);
  if (!ok(p.near)) return p.near;
  let hi = Math.max(p.near * 2, p.near + gap);
  const cap = p.far === Infinity ? 1e15 : p.far;
  while (hi < cap && ok(hi)) hi *= 2;
  if (ok(cap)) return cap;
  let lo = hi / 2;
  for (let i = 0; i < 64; i++) {
    const mid = Math.sqrt(lo * hi);
    if (ok(mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}

// ---- picking / block history --------------------------------------------
//
// A block record stores either exact view-space distances (store='distance')
// or only quantized depth codes (store='code') captured under oldParams.

export function createBlockRecord({ candidates, params, format, store }) {
  const p = normalizeParams(params);
  if (store === 'distance') {
    return {
      store,
      format,
      params: p,
      candidates: candidates.map((c) => ({ id: c.id, distance: c.distance })),
    };
  }
  return {
    store,
    format,
    params: p,
    candidates: candidates.map((c) => ({
      id: c.id,
      code: depthCode(projectDepth(c.distance, p), format),
    })),
  };
}

function codeDistanceInterval(code, format, params) {
  const here = reconstructDistance(format === FORMATS.UNORM24 ? code / UNORM24_MAX : code, params);
  const up = nextCode(code, format, +1);
  const there = reconstructDistance(format === FORMATS.UNORM24 ? up / UNORM24_MAX : up, params);
  return [Math.min(here, there), Math.max(here, there)];
}

function firstCodeAtOrAbove(z, format) {
  const c = depthCode(z, format);
  if (format === FORMATS.FLOAT32) {
    if (c < z) return nextCode(c, format, +1);
  }
  return c;
}

// Do the representable code sets of two [zLo,zHi] window-depth ranges share a cell?
function codeRangesOverlap(aLo, aHi, bLo, bHi, format) {
  const lo = Math.max(aLo, bLo);
  const hi = Math.min(aHi, bHi);
  if (lo > hi) return false;
  const first = firstCodeAtOrAbove(lo, format);
  if (format === FORMATS.UNORM24) return first <= depthCode(hi, format);
  return first <= hi || Math.fround(first) <= hi;
}

function winnerOf(candidates) {
  let best = candidates[0];
  let tie = false;
  for (let i = 1; i < candidates.length; i++) {
    const c = candidates[i];
    if (Math.abs(c.distance - best.distance) < Number.EPSILON * Math.max(1, c.distance)) {
      tie = true;
    } else if (c.distance < best.distance) {
      best = c;
      tie = false;
    }
  }
  return { winnerId: best.id, tie };
}

// Re-evaluate a historical block record under new clip parameters.
// Returns:
//   status 'remapped'  - distances re-quantized, winner still known
//          'tie'       - candidates now share a code cell (z-fight)
//          'undecided' - code-only block, old cells cannot be mapped back uniquely
//   convertible=false marks a historical sample the picking system must reject / redraw.
export function remapBlock(record, newParams) {
  const np = normalizeParams(newParams);
  const { format } = record;

  if (record.store === 'distance') {
    const evaluated = record.candidates.map((c) => ({
      id: c.id,
      distance: c.distance,
      code: depthCode(projectDepth(c.distance, np), format),
    }));
    const codes = new Set(evaluated.map((c) => c.code));
    if (codes.size < evaluated.length) {
      return { status: 'tie', convertible: true, candidates: evaluated, winnerId: null };
    }
    const { winnerId } = winnerOf(record.candidates);
    return { status: 'remapped', convertible: true, candidates: evaluated, winnerId };
  }

  // Code-only block: each old code only tells us a distance interval.
  const intervals = record.candidates.map((c) => ({
    id: c.id,
    oldCode: c.code,
    distInterval: codeDistanceInterval(c.code, format, record.params),
  }));
  const newRanges = intervals.map((it) => {
    const [d0, d1] = it.distInterval;
    const z0 = projectDepth(d0, np);
    const z1 = projectDepth(d1, np);
    return { id: it.id, zLo: Math.min(z0, z1), zHi: Math.max(z0, z1) };
  });
  for (let i = 0; i < newRanges.length; i++) {
    for (let j = i + 1; j < newRanges.length; j++) {
      const a = newRanges[i];
      const b = newRanges[j];
      if (codeRangesOverlap(a.zLo, a.zHi, b.zLo, b.zHi, format)) {
        return { status: 'undecided', convertible: false, candidates: newRanges, winnerId: null };
      }
    }
  }
  // Disjoint new ranges: nearest distance interval wins deterministically.
  let nearest = intervals[0];
  for (let i = 1; i < intervals.length; i++) {
    if (intervals[i].distInterval[0] < nearest.distInterval[0]) nearest = intervals[i];
  }
  return { status: 'remapped', convertible: true, candidates: newRanges, winnerId: nearest.id };
}

// Historical picked pixel still convertible under new camera parameters?
export function historicalSampleConvertible(record, newParams) {
  return remapBlock(record, newParams).convertible;
}

// ---- camera-range sweep & recommendation --------------------------------

export function sweep({ nears, ratios, gap, targetDistance, formats, modes, infinite = [false, true] }) {
  const rows = [];
  for (const near of nears) {
    for (const ratio of ratios) {
      for (const isInf of infinite) {
        const far = isInf ? Infinity : near * ratio;
        for (const format of formats) {
          for (const mode of modes) {
            const params = normalizeParams({ near, far, mode });
            const fail = failureDistance(params, format, gap);
            const target = targetDistance ?? (isInf ? near * ratio : far);
            const atTarget = analyze(Math.min(target, isInf ? target : far * (1 - 1e-9)), params, format);
            rows.push({
              near,
              far,
              infinite: isInf,
              format,
              mode,
              gap,
              failureDistance: fail,
              passes: fail >= target,
              stepAtTarget: atTarget.stepDistance,
              relativeErrorAtTarget: atTarget.relativeError,
            });
          }
        }
      }
    }
  }
  return rows;
}

// Choose a depth representation for a required gap over required distances.
export function recommend({ near, far, gap, targetDistances }) {
  const combos = [];
  for (const format of [FORMATS.UNORM24, FORMATS.FLOAT32]) {
    for (const mode of [MODES.CONVENTIONAL, MODES.REVERSED]) {
      for (const infinite of [false, true]) {
        const params = normalizeParams({
          near,
          far: infinite ? Infinity : far,
          mode,
        });
        const fail = failureDistance(params, format, gap);
        const worstTarget = Math.max(...targetDistances);
        const passes = fail >= worstTarget;
        combos.push({
          label: `${format} / ${mode}${infinite ? ' / infinite-far' : ' / finite-far'}`,
          format,
          mode,
          infinite,
          failureDistance: fail,
          passes,
        });
      }
    }
  }
  combos.sort((a, b) => Number(b.passes) - Number(a.passes) || b.failureDistance - a.failureDistance);
  const choice = combos[0];
  return {
    combos,
    choice,
    failureRange: `d > ${choice.failureDistance.toExponential(3)} gap=${gap} becomes unresolvable`,
  };
}
