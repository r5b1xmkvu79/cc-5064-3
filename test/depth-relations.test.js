import { test } from 'node:test';
import assert from 'node:assert/strict';

import { DepthRenderer } from '../src/renderer.js';
import { MODES, FORMATS } from '../src/depth-kernel.js';
import {
  referenceDepth,
  makeSameCodePair,
  makeClipBoundaryCases,
  runDepthPipeline,
  runDepthAcceptanceSuite,
  mutationChecks,
  evaluateDepthRelation,
} from '../src/depth-acceptance.js';

function numericOrder(trace) {
  const near = trace.find((t) => t.numericId === 3);
  const far = trace.find((t) => t.numericId === 4);
  return { near: near.incomingDepth, far: far.incomingDepth };
}

test('same-code thin plates: high precision oracle orders them, target format accepts interval tie', () => {
  const params = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const pair = makeSameCodePair({
    params,
    format: FORMATS.UNORM24,
    targetDistance: 60,
    ids: [3, 4],
  });

  assert.ok(pair.plates[0].distance < pair.plates[1].distance);
  const zA = referenceDepth.project(pair.plates[0].distance, params);
  const zB = referenceDepth.project(pair.plates[1].distance, params);
  assert.notEqual(zA, zB, 'full precision reference must separate the surfaces');
  assert.equal(referenceDepth.code(zA, FORMATS.UNORM24), pair.code);
  assert.equal(referenceDepth.code(zB, FORMATS.UNORM24), pair.code);

  const spec = {
    name: 'same-code',
    params,
    format: FORMATS.UNORM24,
    plates: pair.plates,
    // Geometry draws the physically near plate first. Equal-code LEQUAL then
    // lets the later far fragment win; that is a tie, not a known order error.
    drawOrder: pair.plates,
  };
  const run = runDepthPipeline(spec);
  const v = run.verdict;

  assert.equal(v.ok, true);
  assert.equal(v.verdict, 'acceptable-tie');
  assert.equal(v.highPrecisionWinnerId, 3);
  assert.ok([3, 4].includes(Number(run.result.objectId)));
  assert.equal(v.failures.some((f) => f.code === 'OCCLUSION_ORDER'), false);
  assert.ok(v.distanceErrorM <= v.distanceToleranceM);
  assert.ok(pair.plates.every((p) => p.distance >= v.acceptedDistanceIntervalM[0]));
  assert.ok(pair.plates.every((p) => p.distance <= v.acceptedDistanceIntervalM[1]));
});

test('reverse depth endpoints, clear values and independent reconstruction agree', () => {
  const conv = { near: 0.25, far: 100, mode: MODES.CONVENTIONAL };
  const rev = { near: 0.25, far: Infinity, mode: MODES.REVERSED };

  assert.equal(referenceDepth.clearValue(conv, FORMATS.UNORM24), 1);
  assert.equal(referenceDepth.clearValue(rev, FORMATS.FLOAT32), 0);
  assert.equal(referenceDepth.project(0.25, conv), 0);
  assert.equal(referenceDepth.project(100, conv), 1);
  assert.equal(referenceDepth.project(0.25, rev), 1);
  assert.equal(referenceDepth.project(1e6, rev), 0.25 / 1e6);

  for (const p of [conv, { near: 0.25, far: 100, mode: MODES.REVERSED }]) {
    for (const d of [0.26, 1, 25, 90]) {
      const z = referenceDepth.project(d, p);
      assert.ok(Math.abs(referenceDepth.reconstruct(z, p) - d) <= 1e-9 * d);
    }
  }
});

function createSpyGl() {
  const calls = { clearDepth: [], depthFunc: [], enable: [] };
  const gl = {
    VERTEX_SHADER: 1,
    FRAGMENT_SHADER: 2,
    COMPILE_STATUS: 3,
    LINK_STATUS: 4,
    ARRAY_BUFFER: 10,
    ELEMENT_ARRAY_BUFFER: 11,
    STATIC_DRAW: 12,
    FLOAT: 61,
    DEPTH_TEST: 72,
    createShader: () => ({}),
    shaderSource() {},
    compileShader() {},
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    createProgram: () => ({}),
    attachShader() {},
    linkProgram() {},
    getProgramParameter: () => true,
    getProgramInfoLog: () => '',
    getUniformLocation: () => ({}),
    getExtension: () => ({}),
    createVertexArray: () => ({}),
    bindVertexArray() {},
    createBuffer: () => ({}),
    bindBuffer() {},
    bufferData() {},
    enableVertexAttribArray() {},
    vertexAttribPointer() {},
    clearDepth(value) { calls.clearDepth.push(value); },
    depthFunc(value) { calls.depthFunc.push(value); },
    enable(value) { calls.enable.push(value); },
    depthRange() {},
  };
  return { gl, calls };
}

test('DepthRenderer configures reverse clear=0/GEQUAL and conventional clear=1/LEQUAL', () => {
  const { gl, calls } = createSpyGl();
  const canvas = { width: 64, height: 64, getContext: () => gl };
  const renderer = new DepthRenderer(canvas);

  renderer._configureDepth(MODES.REVERSED);
  assert.deepEqual(calls.clearDepth.at(-1), 0);
  assert.equal(calls.depthFunc.at(-1), gl.GEQUAL);
  assert.ok(calls.enable.includes(gl.DEPTH_TEST));

  renderer._configureDepth(MODES.CONVENTIONAL);
  assert.deepEqual(calls.clearDepth.at(-1), 1);
  assert.equal(calls.depthFunc.at(-1), gl.LEQUAL);
});

test('clip plane crosses a quantization boundary; replay separates mapping and comparator faults', () => {
  const params = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const boundary = makeClipBoundaryCases({
    distance: 60,
    params,
    format: FORMATS.UNORM24,
  });
  const codes = boundary.cases.map((c) =>
    referenceDepth.code(referenceDepth.project(60, { ...params, near: c.near }), FORMATS.UNORM24),
  );
  assert.notEqual(codes[0], codes[1]);

  for (const clipCase of boundary.cases) {
    const clipParams = { ...params, near: clipCase.near };
    const spec = {
      name: 'clip-boundary',
      params: clipParams,
      format: FORMATS.UNORM24,
      plates: [
        { id: 3, numericId: 3, distance: 60 },
        { id: 4, numericId: 4, distance: 60.01 },
      ],
    };

    const correct = runDepthPipeline(spec);
    assert.equal(correct.verdict.ok, true, JSON.stringify(correct.verdict.failures));
    assert.equal(correct.result.objectId, 3);

    const inverted = runDepthPipeline(spec, { mapping: 'inverted' });
    assert.equal(inverted.verdict.ok, false);
    assert.ok(inverted.verdict.failures.some((f) => f.code === 'DEPTH_MAPPING_DIRECTION_REVERSED'));
    assert.equal(inverted.verdict.failures.some((f) => f.code === 'COMPARE_FUNCTION_ERROR'), false);

    const wrongCompare = runDepthPipeline(spec, { comparator: 'wrong' });
    assert.equal(wrongCompare.verdict.ok, false);
    assert.ok(wrongCompare.verdict.failures.some((f) => f.code === 'COMPARE_FUNCTION_ERROR'));
    assert.equal(wrongCompare.verdict.failures.some((f) => f.code.startsWith('DEPTH_MAPPING')), false);
    assert.ok(wrongCompare.result.pixels.every((p) => p.objectId === 0));
  }
});

test('distant ordered thin layers: perturbed clip and wrong comparator produce visible-order evidence', () => {
  const plates = [
    { id: 3, numericId: 3, distance: 10000 },
    { id: 4, numericId: 4, distance: 10002 },
  ];

  for (const near of [0.05, 0.1, 0.2]) {
    const params = { near, far: Infinity, mode: MODES.REVERSED };
    const spec = { name: 'extreme-far', params, format: FORMATS.FLOAT32, plates };

    const correct = runDepthPipeline(spec);
    assert.equal(correct.verdict.ok, true, JSON.stringify(correct.verdict.failures));
    assert.equal(correct.result.objectId, 3);
    assert.ok(correct.verdict.distanceErrorM <= correct.verdict.distanceToleranceM);
    assert.ok(correct.verdict.distanceErrorM < 0.01);

    const wrong = runDepthPipeline(spec, { comparator: 'wrong' });
    assert.equal(wrong.verdict.ok, false);
    assert.ok(wrong.verdict.failures.some((f) => f.code === 'COMPARE_FUNCTION_ERROR'));
    assert.ok(wrong.verdict.failures.some((f) => f.code === 'OCCLUSION_IDENTITY'));

    // Numeric incoming depths still have the expected reversed ordering. A
    // buffer-only ordering test can therefore pass while every visible pixel is
    // background because the comparator rejected both fragments.
    const order = numericOrder(wrong.result.trace);
    assert.ok(order.near > order.far);
    assert.ok(wrong.result.pixels.every((p) => p.objectId === 0));
    assert.equal(wrong.result.objectId, 0);
    assert.equal(wrong.result.reconstructedDistance, Infinity);
    assert.ok(wrong.verdict.distanceErrorM > 1000);
  }
});

test('far-plane perturbation clips the farther layer without falsifying nearest identity', () => {
  const plates = [
    { id: 3, numericId: 3, distance: 10000 },
    { id: 4, numericId: 4, distance: 10002 },
  ];
  for (const far of [Infinity, 50000, 10001]) {
    const params = { near: 0.1, far, mode: MODES.REVERSED };
    const spec = { name: 'far-perturbation', params, format: FORMATS.FLOAT32, plates };
    const correct = runDepthPipeline(spec);
    assert.equal(correct.verdict.ok, true, JSON.stringify(correct.verdict.failures));
    assert.equal(correct.result.objectId, 3);

    const wrong = runDepthPipeline(spec, { comparator: 'wrong' });
    assert.equal(wrong.verdict.ok, false);
    assert.ok(wrong.result.pixels.every((p) => p.objectId === 0));
    assert.ok(wrong.verdict.failures.some((f) => f.code === 'COMPARE_FUNCTION_ERROR'));
  }
});

test('depth relations acceptance has extreme-far and near-grazing positive/negative evidence', () => {
  const suite = runDepthAcceptanceSuite();
  assert.equal(suite.ok, true, JSON.stringify(suite.mutations, null, 2));

  const byName = Object.fromEntries(suite.cases.map((c) => [c.name, c]));
  const far = byName['extreme-far-reversed-float32-identity-and-distance'];
  assert.equal(far.verdict, 'pass');
  assert.equal(Number(far.pick.objectId), 3);
  assert.ok(far.distanceErrorM < 0.01);

  const unormTie = byName['extreme-far-unorm24-is-an-explicit-tie-not-order-error'];
  assert.equal(unormTie.verdict, 'acceptable-tie');
  assert.equal(unormTie.failures.some((f) => f.code === 'OCCLUSION_ORDER'), false);

  const near = byName['near-plane-grazing-positive'];
  assert.equal(near.verdict, 'pass');
  assert.equal(Number(near.pick.objectId), 1);
  assert.ok(near.distanceErrorM < 1e-6);

  const clipped = byName['near-plane-grazing-clipped-negative'];
  assert.equal(clipped.verdict, 'expected-absence-pass');
  assert.equal(Number(clipped.pick.objectId), 0);
});

test('controlled precision, mapping, comparator, reconstruction and clear errors are all caught', () => {
  const checks = mutationChecks();
  for (const check of checks) {
    assert.equal(check.detected, true, `${check.mutation}: ${check.actualCodes.join(',')}`);
  }

  const bias = checks.find((c) => c.mutation === 'reconstruction-bias-5-percent');
  assert.ok(bias.actualCodes.includes('DISTANCE_RECONSTRUCTION_ERROR'));
  assert.equal(bias.actualCodes.includes('OCCLUSION_ORDER'), false);

  const coarse = checks.find((c) => c.mutation === 'precision-coarse-16-levels');
  assert.ok(
    coarse.actualCodes.includes('OCCLUSION_ORDER') ||
      coarse.actualCodes.includes('OCCLUSION_IDENTITY'),
  );
});

test('browser code tolerance extends distance allowance only over the declared code range', () => {
  const params = { near: 0.1, far: 1000, mode: MODES.CONVENTIONAL };
  const distance = 60.48;
  const plates = [{ id: 3, numericId: 3, distance }];
  const expectedCode = referenceDepth.code(referenceDepth.project(distance, params), FORMATS.UNORM24);

  function verdictFor(offset, codeTolerance) {
    const hardwareCode = expectedCode + offset;
    const hardwareDepth = referenceDepth.codeValue(hardwareCode, FORMATS.UNORM24);
    return evaluateDepthRelation(
      {
        objectId: 3,
        hardwareCode,
        hardwareDepth,
        reconstructedDistance: referenceDepth.reconstruct(hardwareDepth, params),
        clearValue: 1,
        depthFunc: 'LEQUAL',
      },
      {
        name: 'bounded-code-tolerance',
        params,
        format: FORMATS.UNORM24,
        plates,
        codeTolerance,
      },
    );
  }

  const strict = verdictFor(4, 0);
  assert.equal(strict.ok, false);
  assert.ok(strict.failures.some((f) => f.code === 'DEPTH_ATTACHMENT_MISMATCH'));
  assert.ok(strict.failures.some((f) => f.code === 'DISTANCE_RECONSTRUCTION_ERROR'));

  const allowed = verdictFor(4, 4);
  assert.equal(allowed.ok, true, JSON.stringify(allowed.failures));

  const beyondBound = verdictFor(5, 4);
  assert.equal(beyondBound.ok, false);
  assert.ok(beyondBound.failures.some((f) => f.code === 'DEPTH_ATTACHMENT_MISMATCH'));
  assert.ok(beyondBound.failures.some((f) => f.code === 'DISTANCE_RECONSTRUCTION_ERROR'));
});
