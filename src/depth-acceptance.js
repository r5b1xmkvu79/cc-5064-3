// Executable acceptance oracle for depth relations.
//
// The reference formulas in this file are deliberately independent from
// src/depth-kernel.js. The kernel is imported only as the implementation under
// test (projection/reconstruction). SoftwareDepthAttachment replays the
// observable part of a depth attachment: clear value, quantized storage,
// LEQUAL/GEQUAL and object/distance MRT writes. Browser runs additionally feed
// real WebGL readPixels into evaluateActualReadback().

import {
  FORMATS,
  MODES,
  projectDepth as implementationProjectDepth,
  quantize as implementationQuantize,
  depthCode as implementationDepthCode,
  reconstructDistance as implementationReconstructDistance,
} from './depth-kernel.js';

export { FORMATS, MODES };

export const UNORM24_MAX = 2 ** 24 - 1;

function clamp01(z) {
  return Math.min(1, Math.max(0, z));
}

export function normalizeReferenceParams(params = {}) {
  const near = Number(params.near);
  const far = params.far === Infinity || params.far == null ? Infinity : Number(params.far);
  const reverse = params.mode ? params.mode === MODES.REVERSED : params.reverse === true;
  if (!(near > 0)) throw new Error('reference near must be > 0');
  if (far !== Infinity && !(far > near)) throw new Error('reference far must be > near');
  return { near, far, reverse, mode: reverse ? MODES.REVERSED : MODES.CONVENTIONAL };
}

// Independent analytic oracle. Do not call the production kernel here.
export const referenceDepth = {
  normalize: normalizeReferenceParams,

  project(distance, rawParams) {
    const { near: n, far: f, reverse } = normalizeReferenceParams(rawParams);
    const d = Number(distance);
    let z;
    if (f === Infinity) {
      z = reverse ? n / d : 1 - n / d;
    } else if (reverse) {
      z = (n * (f - d)) / ((f - n) * d);
    } else {
      z = (f * (d - n)) / ((f - n) * d);
    }
    return clamp01(z);
  },

  reconstruct(z, rawParams) {
    const { near: n, far: f, reverse } = normalizeReferenceParams(rawParams);
    z = Number(z);
    if (f === Infinity) {
      if (reverse) return z <= 0 ? Infinity : n / z;
      return z >= 1 ? Infinity : n / (1 - z);
    }
    if (reverse) {
      const den = n + z * (f - n);
      return den <= 0 ? Infinity : (n * f) / den;
    }
    const den = f - z * (f - n);
    return den <= 0 ? Infinity : (n * f) / den;
  },

  code(z, format = FORMATS.UNORM24) {
    z = clamp01(z);
    if (format === FORMATS.UNORM24) {
      return Math.min(UNORM24_MAX, Math.max(0, Math.round(z * UNORM24_MAX)));
    }
    if (format === FORMATS.FLOAT32) return Math.fround(z);
    throw new Error(`unsupported format ${format}`);
  },

  stored(z, format = FORMATS.UNORM24) {
    const code = this.code(z, format);
    return format === FORMATS.UNORM24 ? code / UNORM24_MAX : code;
  },

  clearValue(params, format = FORMATS.UNORM24) {
    const p = normalizeReferenceParams(params);
    return this.stored(p.reverse ? 0 : 1, format);
  },

  codeValue(code, format = FORMATS.UNORM24) {
    return format === FORMATS.UNORM24 ? Number(code) / UNORM24_MAX : Number(code);
  },

  ulp(z) {
    if (z <= 0) return 2 ** -149;
    if (!Number.isFinite(z)) return Infinity;
    const e = Math.floor(Math.log2(z));
    const bin = 2 ** e === z ? e - 1 : e;
    return 2 ** (bin - 23);
  },

  // Round-to-nearest code bin, converted back to view distance. The returned
  // interval is the prescribed acceptable tie/reconstruction range.
  distanceInterval(distanceOrCode, rawParams, format = FORMATS.UNORM24, input = 'distance') {
    const p = normalizeReferenceParams(rawParams);
    const code =
      input === 'code'
        ? distanceOrCode
        : this.code(this.project(distanceOrCode, p), format);
    const zc = this.codeValue(code, format);
    let halfZ;
    if (format === FORMATS.UNORM24) halfZ = 0.5 / UNORM24_MAX;
    else halfZ = this.ulp(zc) * 0.5;
    const zLo = clamp01(zc - halfZ);
    const zHi = clamp01(zc + halfZ);
    const d0 = this.reconstruct(zLo, p);
    const d1 = this.reconstruct(zHi, p);
    const center = this.reconstruct(zc, p);
    const low = Math.min(d0, d1);
    const high = Math.max(d0, d1);
    return { code, zCenter: zc, centerDistance: center, low, high };
  },

  tolerance(distance, rawParams, format = FORMATS.UNORM24) {
    const interval = this.distanceInterval(distance, rawParams, format, 'distance');
    const geometric = Math.max(
      Math.abs(interval.centerDistance - distance),
      Number.isFinite(interval.low) ? Math.abs(interval.centerDistance - interval.low) : Infinity,
      Number.isFinite(interval.high) ? Math.abs(interval.centerDistance - interval.high) : Infinity,
    );
    const numerical = 1e-8 * Math.max(1, Math.abs(distance), Math.abs(interval.centerDistance));
    return {
      ...interval,
      toleranceM: geometric + numerical,
    };
  },
};

// Construct two distinct exact surfaces that quantize to one target code.
// High precision exact distances still establish A is nearer.
export function makeSameCodePair({
  params,
  format = FORMATS.UNORM24,
  targetDistance = 60,
  idPrefix = 'same',
  ids = [91, 92],
  spanFraction = 0.7,
} = {}) {
  const p = normalizeReferenceParams(params);
  const z0 = referenceDepth.project(targetDistance, p);
  const code = referenceDepth.code(z0, format);
  const zc = referenceDepth.codeValue(code, format);
  const halfZ =
    format === FORMATS.UNORM24 ? 0.5 / UNORM24_MAX : referenceDepth.ulp(zc) * 0.5;
  const za = clamp01(zc - halfZ * spanFraction);
  const zb = clamp01(zc + halfZ * spanFraction);
  const da = referenceDepth.reconstruct(za, p);
  const db = referenceDepth.reconstruct(zb, p);
  const nearDistance = Math.min(da, db);
  const farDistance = Math.max(da, db);
  return {
    code,
    zCode: zc,
    highPrecisionWinner: ids[0],
    plates: [
      { id: ids[0], numericId: ids[0], distance: nearDistance, role: 'near' },
      { id: ids[1], numericId: ids[1], distance: farDistance, role: 'far' },
    ],
    gap: farDistance - nearDistance,
    targetSameCode: true,
  };
}

// Solve near plane at which a fixed distance crosses a quantization boundary.
export function nearPlaneAtDepthBoundary(distance, boundaryZ, rawParams) {
  const { far: f, reverse } = normalizeReferenceParams(rawParams);
  const d = Number(distance);
  const z = Number(boundaryZ);
  let near;
  if (f === Infinity) {
    near = reverse ? d * z : d * (1 - z);
  } else if (reverse) {
    near = (z * d * f) / (f - d + z * d);
  } else {
    near = (f * d * (1 - z)) / (f - z * d);
  }
  if (!(near > 0)) throw new Error('boundary near plane must be > 0');
  return near;
}

export function makeClipBoundaryCases({
  distance = 60,
  params,
  format = FORMATS.UNORM24,
  codeIndex = null,
} = {}) {
  const initial = normalizeReferenceParams(params);
  const z = referenceDepth.project(distance, initial);
  const code = codeIndex ?? referenceDepth.code(z, format);
  const maxCode = format === FORMATS.UNORM24 ? UNORM24_MAX : null;
  const boundaryCode = maxCode == null ? code : Math.min(maxCode - 1, code);
  const boundaryZ =
    format === FORMATS.UNORM24 ? (boundaryCode + 0.5) / UNORM24_MAX : z;
  const nBoundary = nearPlaneAtDepthBoundary(distance, boundaryZ, initial);

  let nearBefore = nBoundary;
  let nearAfter = nBoundary;
  for (const fraction of [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]) {
    const lo = nBoundary * (1 - fraction);
    const hi = nBoundary * (1 + fraction);
    if (!(lo > 0) || (initial.far !== Infinity && hi >= initial.far)) continue;
    const codeLo = referenceDepth.code(
      referenceDepth.project(distance, { ...initial, near: lo }),
      format,
    );
    const codeHi = referenceDepth.code(
      referenceDepth.project(distance, { ...initial, near: hi }),
      format,
    );
    if (codeLo !== codeHi) {
      nearBefore = lo;
      nearAfter = hi;
      break;
    }
  }

  return {
    distance,
    format,
    boundaryZ,
    boundaryNear: nBoundary,
    cases: [
      { label: 'near-below-boundary', near: nearBefore },
      { label: 'near-above-boundary', near: nearAfter },
    ],
  };
}

function defaultProject(d, p) {
  return implementationProjectDepth(d, p);
}
function defaultQuantize(z, format) {
  return implementationQuantize(z, format);
}
function defaultReconstruct(z, p) {
  return implementationReconstructDistance(z, p);
}

// Small observable depth attachment. Every plate covers every pixel; object
// identity and exact distance are MRT channels analogous to renderer.js.
export class ReferenceDepthPipeline {
  constructor({
    width = 3,
    height = 3,
    format = FORMATS.UNORM24,
    params,
    mapping = 'correct', // 'correct' | 'inverted'
    comparator = 'correct', // 'correct' | 'wrong'
    clearValue = undefined,
    project = defaultProject,
    quantizeDepth = defaultQuantize,
    codeDepth = implementationDepthCode,
    reconstruct = defaultReconstruct,
  } = {}) {
    if (!params) throw new Error('params required');
    this.width = width;
    this.height = height;
    this.format = format;
    this.params = normalizeReferenceParams(params);
    this.mapping = mapping;
    this.comparator = comparator;
    this.project = project;
    this.quantizeDepth = quantizeDepth;
    this.codeDepth = codeDepth;
    this.reconstruct = reconstruct;
    this.expectedClear = referenceDepth.clearValue(this.params, format);
    this.clearValue = clearValue === undefined ? this.expectedClear : clearValue;
    const n = width * height;
    this.depth = new Float64Array(n);
    this.depthCodes = new Float64Array(n);
    this.objectId = new Uint32Array(n);
    this.exactDistance = new Float64Array(n);
    this.trace = [];
    this.clear();
  }

  index(x, y) {
    return y * this.width + x;
  }

  clear() {
    this.depth.fill(this.clearValue);
    this.depthCodes.fill(referenceDepth.code(this.clearValue, this.format));
    this.objectId.fill(0);
    this.exactDistance.fill(0);
    this.trace = [];
  }

  compare(incoming, stored) {
    const correct = this.params.reverse ? incoming >= stored : incoming <= stored;
    if (this.comparator !== 'wrong') return correct;
    // Deliberately install the opposite named GL comparison direction. Equal
    // depths still pass under both LEQUAL and GEQUAL.
    return this.params.reverse ? incoming <= stored : incoming >= stored;
  }

  compareFuncName() {
    const correctName = this.params.reverse ? 'GEQUAL' : 'LEQUAL';
    if (this.comparator !== 'wrong') return correctName;
    return this.params.reverse ? 'LEQUAL' : 'GEQUAL';
  }

  draw(plate) {
    const p = this.params;
    if (plate.distance < p.near || (p.far !== Infinity && plate.distance > p.far)) {
      this.trace.push({
        id: plate.id,
        numericId: plate.numericId ?? plate.id,
        distance: plate.distance,
        clipped: true,
        passed: false,
        depthFunc: this.compareFuncName(),
      });
      return this;
    }
    let z = this.project(plate.distance, p);
    if (this.mapping === 'inverted') z = 1 - z;
    const incoming = this.quantizeDepth(z, this.format);
    const incomingCode = this.codeDepth
      ? this.codeDepth(z, this.format)
      : referenceDepth.code(incoming, this.format);
    const center = this.index(Math.floor(this.width / 2), Math.floor(this.height / 2));
    const storedBefore = this.depth[center];
    const storedCodeBefore = this.depthCodes[center];
    const passed = this.compare(incoming, storedBefore);
    if (passed) {
      for (let i = 0; i < this.depth.length; i++) {
        // Compare per fragment against its own current attachment value.
        const ok = this.compare(incoming, this.depth[i]);
        if (!ok) continue;
        this.depth[i] = incoming;
        this.depthCodes[i] = incomingCode;
        this.objectId[i] = plate.numericId ?? plate.id;
        this.exactDistance[i] = plate.distance;
      }
    }
    this.trace.push({
      id: plate.id,
      numericId: plate.numericId ?? plate.id,
      distance: plate.distance,
      clipped: false,
      projectedDepth: z,
      incomingDepth: incoming,
      incomingCode,
      storedBefore,
      storedCodeBefore,
      passed,
      storedAfter: this.depth[center],
      storedCodeAfter: this.depthCodes[center],
      depthFunc: this.compareFuncName(),
    });
    return this;
  }

  drawAll(plates) {
    for (const plate of plates) this.draw(plate);
    return this;
  }

  pick(x = Math.floor(this.width / 2), y = Math.floor(this.height / 2)) {
    const i = this.index(x, y);
    const hardwareDepth = this.depth[i];
    return {
      x,
      y,
      objectId: this.objectId[i],
      hardwareDepth,
      hardwareCode: this.depthCodes[i],
      exactViewDist: this.exactDistance[i],
      reconstructedDistance: this.reconstruct(hardwareDepth, this.params),
      depthReadbackChannel: 'reference depth attachment',
    };
  }

  pixels() {
    const out = [];
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const i = this.index(x, y);
        out.push({
          x,
          y,
          objectId: this.objectId[i],
          hardwareDepth: this.depth[i],
          exactViewDist: this.exactDistance[i],
        });
      }
    }
    return out;
  }
}

function idMatches(actual, expected) {
  return String(actual) === String(expected) || Number(actual) === Number(expected);
}

function plateMatchesId(plate, actual) {
  return idMatches(plate.id, actual) || (plate.numericId != null && idMatches(plate.numericId, actual));
}

export function deriveReferenceExpectation(spec) {
  const p = normalizeReferenceParams(spec.params);
  const format = spec.format ?? FORMATS.UNORM24;
  const eps = spec.clipEpsilon ?? 1e-9;
  const evaluated = spec.plates.map((plate) => {
    const z = referenceDepth.project(plate.distance, p);
    const code = referenceDepth.code(z, format);
    return {
      ...plate,
      windowDepth: z,
      code,
      storedDepth: referenceDepth.codeValue(code, format),
      visibleToClip:
        plate.distance >= p.near * (1 - eps) &&
        (p.far === Infinity || plate.distance <= p.far * (1 + eps)),
    };
  });
  const visible = evaluated.filter((plate) => plate.visibleToClip);
  const front = visible.length
    ? visible.reduce((a, b) => (a.distance <= b.distance ? a : b))
    : null;
  const codes = [...new Set(evaluated.map((plate) => plate.code))];
  const resolvable = visible.length >= 2 && new Set(visible.map((plate) => plate.code)).size === visible.length;
  const tieExpected = visible.length >= 2 && !resolvable;
  return {
    params: p,
    format,
    plates: evaluated,
    visible,
    expectedFront: front,
    highPrecisionWinnerId: front?.id ?? null,
    resolvable,
    tieExpected,
    targetCodeCount: codes.length,
    expectedClear: referenceDepth.clearValue(p, format),
    expectedDepthFunc: p.reverse ? 'GEQUAL' : 'LEQUAL',
  };
}

// Evaluate either a ReferenceDepthPipeline result or a browser readPixel().
export function evaluateDepthRelation(result, spec = {}) {
  const expectation = deriveReferenceExpectation(spec);
  const pick = result.pick ?? result;
  const trace = result.trace ?? [];
  const failures = [];
  const evidence = {
    specName: spec.name ?? null,
    params: expectation.params,
    format: expectation.format,
    expectedFrontId: expectation.expectedFront?.id ?? null,
    highPrecisionWinnerId: expectation.highPrecisionWinnerId,
    targetResolvable: expectation.resolvable,
    expectedClear: expectation.expectedClear,
    actualClear: result.clearValue,
    expectedDepthFunc: expectation.expectedDepthFunc,
    actualDepthFunc: trace[0]?.depthFunc ?? result.depthFunc ?? null,
    pick,
    trace,
    pixels: result.pixels ?? undefined,
  };

  if (result.clearValue !== undefined && result.clearValue !== expectation.expectedClear) {
    failures.push({
      code: 'CLEAR_VALUE_ERROR',
      expected: expectation.expectedClear,
      actual: result.clearValue,
    });
  }
  if (
    typeof result.depthFunc === 'string' &&
    (result.depthFunc === 'LEQUAL' || result.depthFunc === 'GEQUAL') &&
    result.depthFunc !== expectation.expectedDepthFunc
  ) {
    failures.push({
      code: 'COMPARE_FUNCTION_CONFIG_ERROR',
      expected: expectation.expectedDepthFunc,
      actual: result.depthFunc,
    });
  }

  // Mapping is checked from each incoming fragment, independently of whether
  // the comparator later rejects it.
  for (const fragment of trace) {
    if (fragment.clipped) continue;
    const expectedPlate = expectation.plates.find((plate) => plateMatchesId(plate, fragment.id));
    if (!expectedPlate) continue;
    const expectedCode = expectedPlate.code;
    const actualCode =
      fragment.incomingCode ?? referenceDepth.code(fragment.incomingDepth, expectation.format);
    if (actualCode !== expectedCode) {
      const complementDepth = clamp01(1 - expectedPlate.windowDepth);
      const reversedLike =
        Math.abs(fragment.incomingDepth - complementDepth) <
        Math.abs(fragment.incomingDepth - expectedPlate.storedDepth);
      failures.push({
        code: reversedLike ? 'DEPTH_MAPPING_DIRECTION_REVERSED' : 'DEPTH_MAPPING_ERROR',
        plateId: fragment.id,
        expectedCode,
        actualCode,
        expectedDepth: expectedPlate.storedDepth,
        actualDepth: fragment.incomingDepth,
      });
    }
  }

  const drawnFragments = trace.filter((fragment) => !fragment.clipped && fragment.incomingCode != null);
  if (drawnFragments.length >= 2) {
    const byDistance = expectation.plates
      .map((plate) => ({
        plate,
        fragment: drawnFragments.find((fragment) => plateMatchesId(plate, fragment.id)),
      }))
      .filter((x) => x.fragment);
    byDistance.sort((a, b) => a.plate.distance - b.plate.distance);
    for (let i = 1; i < byDistance.length; i++) {
      const nearDepth = byDistance[i - 1].fragment.incomingDepth;
      const farDepth = byDistance[i].fragment.incomingDepth;
      const orientationCorrect = expectation.params.reverse
        ? nearDepth >= farDepth
        : nearDepth <= farDepth;
      if (!orientationCorrect) {
        failures.push({
          code: 'DEPTH_MAPPING_DIRECTION_REVERSED',
          nearId: byDistance[i - 1].plate.id,
          farId: byDistance[i].plate.id,
          nearDepth,
          farDepth,
          expectedOrientation: expectation.params.reverse ? 'near >= far' : 'near <= far',
        });
        break;
      }
    }
  }

  // Comparator semantics against the attachment value actually present before
  // each draw. This catches a swapped GL depth function even before identity.
  for (const fragment of trace) {
    if (fragment.clipped) continue;
    const shouldPass = expectation.params.reverse
      ? fragment.incomingDepth >= fragment.storedBefore
      : fragment.incomingDepth <= fragment.storedBefore;
    if (fragment.passed !== shouldPass) {
      failures.push({
        code: 'COMPARE_FUNCTION_ERROR',
        plateId: fragment.id,
        depthFunc: fragment.depthFunc,
        incomingDepth: fragment.incomingDepth,
        storedBefore: fragment.storedBefore,
        expectedPass: shouldPass,
        actualPass: fragment.passed,
      });
    }
  }

  if (!expectation.expectedFront) {
    if (Number(pick.objectId) !== 0) {
      failures.push({
        code: 'CLIP_OCCLUSION_IDENTITY',
        message: 'geometry beyond the clip planes must not be visible',
        expectedId: 0,
        actualId: pick.objectId,
      });
    }
    return {
      ok: failures.length === 0,
      verdict: failures.length === 0 ? 'expected-absence-pass' : 'fail',
      ...evidence,
      failures,
      distanceErrorM: null,
      distanceToleranceM: null,
      acceptedDistanceIntervalM: null,
    };
  }

  const pickedPlate = expectation.plates.find((plate) => plateMatchesId(plate, pick.objectId));
  const frontIsPicked =
    expectation.expectedFront && plateMatchesId(expectation.expectedFront, pick.objectId);
  if (Number(pick.objectId) === 0 || !pickedPlate) {
    failures.push({
      code: 'OCCLUSION_IDENTITY',
      message: 'expected a visible plate but object picking returned background/unknown id',
      expectedFrontId: expectation.expectedFront.id,
      actualId: pick.objectId,
      reconstructedDistance: pick.reconstructedDistance,
    });
  } else if (expectation.resolvable && !frontIsPicked) {
    failures.push({
      code: 'OCCLUSION_ORDER',
      message: 'target format resolves the pair, but visible pixels show the farther object',
      expectedFrontId: expectation.expectedFront.id,
      actualVisibleId: pick.objectId,
      actualVisibleOrder: trace.filter((t) => t.passed).map((t) => t.id),
    });
  }

  let distanceErrorM = null;
  let distanceToleranceM = null;
  let acceptedDistanceIntervalM = null;
  const extraDistanceToleranceM = Number(spec.extraDistanceToleranceM ?? 0);
  const codeTolerance = Number(spec.codeTolerance ?? 0);
  if (pickedPlate) {
    const interval = referenceDepth.tolerance(
      pickedPlate.distance,
      expectation.params,
      expectation.format,
    );
    distanceToleranceM = interval.toleranceM + extraDistanceToleranceM;
    if (codeTolerance > 0) {
      let zLow;
      let zHigh;
      if (expectation.format === FORMATS.UNORM24) {
        zLow = clamp01((pickedPlate.code - codeTolerance - 0.5) / UNORM24_MAX);
        zHigh = clamp01((pickedPlate.code + codeTolerance + 0.5) / UNORM24_MAX);
      } else {
        const codeUlp = referenceDepth.ulp(Number(pickedPlate.code));
        zLow = clamp01(Number(pickedPlate.code) - codeTolerance * codeUlp);
        zHigh = clamp01(Number(pickedPlate.code) + codeTolerance * codeUlp);
      }
      const codeBoundToleranceM = Math.max(
        Math.abs(referenceDepth.reconstruct(zLow, expectation.params) - pickedPlate.distance),
        Math.abs(referenceDepth.reconstruct(zHigh, expectation.params) - pickedPlate.distance),
      );
      distanceToleranceM = Math.max(
        distanceToleranceM,
        codeBoundToleranceM + extraDistanceToleranceM,
      );
    }
    acceptedDistanceIntervalM = [interval.low, interval.high];
    distanceErrorM = Math.abs(pick.reconstructedDistance - pickedPlate.distance);
    if (
      !Number.isFinite(distanceErrorM) ||
      distanceErrorM > distanceToleranceM ||
      pickedPlate.distance < interval.low - interval.toleranceM * 0.01 ||
      pickedPlate.distance > interval.high + interval.toleranceM * 0.01
    ) {
      failures.push({
        code: 'DISTANCE_RECONSTRUCTION_ERROR',
        pickedId: pickedPlate.id,
        referenceDistanceM: pickedPlate.distance,
        reconstructedDistanceM: pick.reconstructedDistance,
        distanceErrorM,
        distanceToleranceM,
        acceptedDistanceIntervalM,
      });
    }

    const expectedPickedCode = pickedPlate.code;
    const actualPickedCode =
      pick.hardwareCode ?? referenceDepth.code(pick.hardwareDepth, expectation.format);
    let codeMatches = actualPickedCode === expectedPickedCode;
    if (!codeMatches && codeTolerance > 0) {
      if (expectation.format === FORMATS.UNORM24) {
        codeMatches = Math.abs(Number(actualPickedCode) - Number(expectedPickedCode)) <= codeTolerance;
      } else {
        const ulp = referenceDepth.ulp(Number(expectedPickedCode));
        codeMatches = Math.abs(Number(actualPickedCode) - Number(expectedPickedCode)) <= ulp * codeTolerance;
      }
    }
    if (!codeMatches) {
      failures.push({
        code: 'DEPTH_ATTACHMENT_MISMATCH',
        pickedId: pickedPlate.id,
        expectedCode: expectedPickedCode,
        actualCode: actualPickedCode,
        expectedDepth: pickedPlate.storedDepth,
        actualDepth: pick.hardwareDepth,
      });
    }
  } else {
    const referenceDistanceM = expectation.expectedFront.distance;
    distanceErrorM = Math.abs(pick.reconstructedDistance - referenceDistanceM);
    failures.push({
      code: 'DISTANCE_RECONSTRUCTION_ERROR',
      pickedId: null,
      referenceDistanceM,
      reconstructedDistanceM: pick.reconstructedDistance,
      distanceErrorM,
      distanceToleranceM: Infinity,
    });
  }

  const cleanTie =
    expectation.tieExpected &&
    pickedPlate != null &&
    failures.every((f) => !['OCCLUSION_ORDER', 'OCCLUSION_IDENTITY'].includes(f.code));
  const tieAccepted = cleanTie && failures.length === 0;

  return {
    ok: failures.length === 0 || tieAccepted,
    verdict:
      failures.length === 0
        ? expectation.tieExpected
          ? 'acceptable-tie'
          : 'pass'
        : tieAccepted
          ? 'acceptable-tie'
          : 'fail',
    tieAccepted,
    ...evidence,
    pickedReferenceDistanceM: pickedPlate?.distance ?? expectation.expectedFront.distance,
    distanceErrorM,
    distanceToleranceM,
    acceptedDistanceIntervalM,
    failures,
  };
}

export function runDepthPipeline(spec, options = {}) {
  const pipeline = new ReferenceDepthPipeline({
    width: spec.width ?? 3,
    height: spec.height ?? 3,
    format: spec.format,
    params: spec.params,
    ...options,
  });
  const drawOrder = options.drawOrder ?? spec.drawOrder ?? spec.plates;
  pipeline.drawAll(drawOrder);
  const result = {
    ...pipeline.pick(),
    clearValue: pipeline.clearValue,
    depthFunc: pipeline.compareFuncName(),
    trace: pipeline.trace,
    pixels: pipeline.pixels(),
  };
  const verdict = evaluateDepthRelation(result, spec);
  return { pipeline, result, verdict };
}

export function canonicalAcceptanceScenarios() {
  const sameCodeParams = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const samePair = makeSameCodePair({
    params: sameCodeParams,
    format: FORMATS.UNORM24,
    targetDistance: 60,
    idPrefix: 'same',
  });

  const farParams = { near: 0.1, far: Infinity, mode: MODES.REVERSED };
  const extremeFarPlates = [
    { id: 'far-A', numericId: 3, distance: 10000 },
    { id: 'far-B', numericId: 4, distance: 10002 },
  ];

  const nearParams = { near: 1, far: 1000, mode: MODES.REVERSED };
  const nearPair = [
    { id: 'graze-A', numericId: 1, distance: 1.01 },
    { id: 'graze-B', numericId: 2, distance: 1.012 },
  ];

  return [
    {
      name: 'same-code-readback-accepts-interval-tie',
      params: sameCodeParams,
      format: FORMATS.UNORM24,
      plates: samePair.plates,
      samePair,
      drawOrder: [samePair.plates[0], samePair.plates[1]],
    },
    {
      name: 'extreme-far-reversed-float32-identity-and-distance',
      params: farParams,
      format: FORMATS.FLOAT32,
      plates: extremeFarPlates,
      drawOrder: [extremeFarPlates[0], extremeFarPlates[1]],
    },
    {
      name: 'extreme-far-unorm24-is-an-explicit-tie-not-order-error',
      params: farParams,
      format: FORMATS.UNORM24,
      plates: extremeFarPlates,
      drawOrder: [extremeFarPlates[0], extremeFarPlates[1]],
    },
    {
      name: 'near-plane-grazing-positive',
      params: nearParams,
      format: FORMATS.FLOAT32,
      plates: nearPair,
      drawOrder: [nearPair[0], nearPair[1]],
    },
    {
      name: 'near-plane-grazing-clipped-negative',
      params: nearParams,
      format: FORMATS.FLOAT32,
      plates: [{ id: 'clipped', numericId: 7, distance: 0.99 }],
      drawOrder: [{ id: 'clipped', numericId: 7, distance: 0.99 }],
    },
  ];
}

function coarseQuantize(z) {
  // Controlled precision bug: only 16 depth codes instead of FLOAT32/UNORM24.
  const levels = 15;
  return Math.round(clamp01(z) * levels) / levels;
}

function biasedReconstruct(z, params) {
  return implementationReconstructDistance(z, params) * 1.05 + 0.01;
}

export function mutationChecks() {
  const scenarios = canonicalAcceptanceScenarios();
  const far = scenarios.find((s) => s.name.includes('float32-identity'));
  const clip = makeClipBoundaryCases({
    distance: 60,
    params: { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL },
    format: FORMATS.UNORM24,
  });

  const mutations = [
    {
      id: 'mapping-direction-inverted',
      target: far,
      options: { mapping: 'inverted' },
      expectCodes: ['DEPTH_MAPPING_DIRECTION_REVERSED'],
    },
    {
      id: 'compare-direction-swapped',
      target: far,
      options: { comparator: 'wrong' },
      expectCodes: ['COMPARE_FUNCTION_ERROR', 'OCCLUSION_IDENTITY'],
    },
    {
      id: 'precision-coarse-16-levels',
      target: far,
      options: { quantizeDepth: coarseQuantize, codeDepth: null },
      expectAnyCodes: [
        'DEPTH_ATTACHMENT_MISMATCH',
        'OCCLUSION_ORDER',
        'OCCLUSION_IDENTITY',
        'DISTANCE_RECONSTRUCTION_ERROR',
      ],
    },
    {
      id: 'reconstruction-bias-5-percent',
      target: far,
      options: { reconstruct: biasedReconstruct },
      expectCodes: ['DISTANCE_RECONSTRUCTION_ERROR'],
    },
    {
      id: 'clear-value-swapped',
      target: far,
      options: { clearValue: 1 },
      expectAnyCodes: ['CLEAR_VALUE_ERROR', 'COMPARE_FUNCTION_ERROR', 'OCCLUSION_IDENTITY'],
    },
  ];

  return mutations.map((mutation) => {
    const run = runDepthPipeline(mutation.target, mutation.options);
    const codes = run.verdict.failures.map((f) => f.code);
    const detected =
      !run.verdict.ok &&
      (mutation.expectCodes
        ? mutation.expectCodes.every((code) => codes.includes(code))
        : mutation.expectAnyCodes.some((code) => codes.includes(code)));
    return {
      mutation: mutation.id,
      detected,
      expectedCodes: mutation.expectCodes ?? mutation.expectAnyCodes,
      actualCodes: codes,
      verdict: run.verdict.verdict,
      actualVisiblePixel: {
        objectId: run.result.objectId,
        hardwareDepth: run.result.hardwareDepth,
        reconstructedDistance: run.result.reconstructedDistance,
      },
      evidence: run.verdict,
    };
  });
}

export function runDepthAcceptanceSuite() {
  const scenarios = canonicalAcceptanceScenarios();
  const caseResults = scenarios.map((spec) => {
    const run = runDepthPipeline(spec);
    return { name: spec.name, ...run.verdict, result: run.result };
  });

  const clipSpecBase = {
    name: 'clip-quantization-boundary-replay',
    format: FORMATS.UNORM24,
    plates: [
      { id: 'clip-near', numericId: 3, distance: 60 },
      { id: 'clip-far', numericId: 4, distance: 60.01 },
    ],
  };
  const boundary = makeClipBoundaryCases({
    distance: 60,
    params: { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL },
  });
  const clipResults = boundary.cases.map((clipCase) => {
    const spec = {
      ...clipSpecBase,
      params: { near: clipCase.near, far: 1000, mode: MODES.CONVENTIONAL },
    };
    const correct = runDepthPipeline(spec);
    const inverted = runDepthPipeline(spec, { mapping: 'inverted' });
    const wrongCompare = runDepthPipeline(spec, { comparator: 'wrong' });
    return {
      label: clipCase.label,
      near: clipCase.near,
      correctCode: referenceDepth.code(
        referenceDepth.project(60, spec.params),
        FORMATS.UNORM24,
      ),
      correct: correct.verdict,
      mappingInvertedDetected: !inverted.verdict.ok,
      mappingFailureCodes: inverted.verdict.failures.map((f) => f.code),
      compareWrongDetected: !wrongCompare.verdict.ok,
      compareFailureCodes: wrongCompare.verdict.failures.map((f) => f.code),
      actualVisiblePixel: {
        objectId: wrongCompare.result.objectId,
        hardwareDepth: wrongCompare.result.hardwareDepth,
        reconstructedDistance: wrongCompare.result.reconstructedDistance,
      },
    };
  });

  const mutations = mutationChecks();
  const allCasesOk = caseResults.every((c) => c.ok);
  const clipOk = clipResults.every(
    (c) =>
      c.correct.ok &&
      c.mappingInvertedDetected &&
      c.compareWrongDetected,
  );
  const mutationsOk = mutations.every((m) => m.detected);
  return {
    ok: allCasesOk && clipOk && mutationsOk,
    cases: caseResults,
    clipBoundary: { ...boundary, results: clipResults, ok: clipOk },
    mutations,
  };
}
