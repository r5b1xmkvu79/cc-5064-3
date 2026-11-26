// Application wiring: kernel-driven controls, two raster viewports, picking,
// block-level local refresh, readback display, error/chart rendering, sweeps.

import { OrbitCamera } from './mat4.js';
import { DepthRenderer, buildProjection } from './renderer.js';
import { buildScene, ID_LABELS, PLATE_IDS } from './geometry.js';
import { BlockStore } from './block-store.js';
import { drawPrecisionChart, drawBlockMap } from './charts.js';
import {
  FORMATS,
  MODES,
  normalizeParams,
  projectDepth,
  quantize,
  reconstructDistance,
  analyze,
  distinguishable,
  failureDistance,
  sweep,
  recommend,
} from './depth-kernel.js';
import { evaluateDepthRelation, referenceDepth } from './depth-acceptance.js';

const $ = (id) => document.getElementById(id);

const state = {
  near: 0.1,
  far: 1000,
  infiniteFar: false,
  cameraDist: 80,
  farDistance: 60,
  farGap: 0.5,
  farHalfExtent: 3,
  nearDistance: 12,
  nearGap: 0.2,
  farY: 0,
  nearY: 0,
  displayMode: 0,
  pathA: { format: FORMATS.UNORM24, mode: MODES.CONVENTIONAL },
  pathB: { format: FORMATS.FLOAT32, mode: MODES.REVERSED },
  storeMode: 'distance',
};

const fovy = Math.PI / 4;
const camera = new OrbitCamera([0, 0, 0]);
const refs = { A: { renderer: null, store: null }, B: { renderer: null, store: null } };
let scene;
let activeRegressionCase = null;
const logEl = $('log');

function log(msg, cls = '') {
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function showUiError(error) {
  const el = $('uiError');
  if (!el) return;
  el.hidden = false;
  el.textContent = `深度精度界面异常：${error?.stack || error?.message || String(error)}`;
  try {
    log(error?.message || String(error), 'e');
  } catch {
    // The error panel is the primary failure surface if logging is unavailable.
  }
}

window.addEventListener('error', (event) => showUiError(event.error || event.message));
window.addEventListener('unhandledrejection', (event) => showUiError(event.reason));

function effectiveFar() {
  return state.infiniteFar ? Infinity : state.far;
}

function currentParams(mode) {
  return normalizeParams({ near: state.near, far: effectiveFar(), mode });
}

// ---------------------------------------------------------------- setup

function initRenderer(canvas) {
  const r = new DepthRenderer(canvas);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || canvas.parentElement.clientWidth || 480;
  const w = Math.max(2, Math.round(cssW * dpr));
  const h = Math.max(2, Math.round(w * 0.75));
  r.resize(w, h);
  return r;
}

function rebuildScene() {
  scene = buildScene(state);
  refs.A.renderer.uploadMesh(scene.mesh);
  refs.A.renderer.uploadGround(scene.ground);
  refs.B.renderer.uploadMesh(scene.mesh);
  refs.B.renderer.uploadGround(scene.ground);
  resetStores();
}

function makeStore(format) {
  return new BlockStore({ blockSize: 64, format, store: state.storeMode });
}

function resetStores() {
  refs.A.store = makeStore(state.pathA.format);
  refs.B.store = makeStore(state.pathB.format);
  refs.A.store.setViewport(refs.A.renderer.width, refs.A.renderer.height);
  refs.B.store.setViewport(refs.B.renderer.width, refs.B.renderer.height);
}

function renderPath(key, blockRect = null) {
  const { renderer } = refs[key];
  const path = state[`path${key}`];
  camera.distance = state.cameraDist;
  const view = camera.view();
  const params = currentParams(path.mode);
  const proj = buildProjection({
    fovy,
    aspect: renderer.width / renderer.height,
    near: params.near,
    far: params.far,
    mode: path.mode,
  });
  renderer.renderFbo(path.format, { view, proj, mode: path.mode, blockRect });
  renderer.present(path.format, {
    near: params.near,
    far: params.far,
    mode: path.mode,
    displayMode: state.displayMode,
  });
  return params;
}

function fullRender() {
  renderPath('A');
  renderPath('B');
  drawBlockMap($('blockCanvas'), refs.A.store, refs.B.store);
  drawPrecisionChart($('chartCanvas'), state);
  updateRecommend();
  updateOcclusionVerdict();
}

// ------------------------------------------------------- block refresh

function refreshBlockAtPixel(key, x, y) {
  const { renderer, store } = refs[key];
  const path = state[`path${key}`];
  const { col, row } = store.blockAtPixel(x, y);
  const rect = store.rectOf(col, row);
  renderPath(key, rect);
  const readback = renderer.readBlock(path.format, rect[0], rect[1], rect[2], rect[3]);
  if (readback) store.ingest(col, row, readback, currentParams(path.mode));
  drawBlockMap($('blockCanvas'), refs.A.store, refs.B.store);
}

// ------------------------------------------------------------- picking

function pick(key, canvas, ev) {
  const { renderer, store } = refs[key];
  const path = state[`path${key}`];
  const rect = canvas.getBoundingClientRect();
  const sx = Math.floor((ev.clientX - rect.left) * (renderer.width / rect.width));
  const sy = Math.floor((ev.clientY - rect.top) * (renderer.height / rect.height));
  const sample = renderer.readPixel(path.format, sx, sy);
  if (!sample || !(sample.exactViewDist > 0)) {
    log('拾取落空（背景）', 'w');
    return;
  }
  const params = currentParams(path.mode);
  const kernelZ = projectDepth(sample.exactViewDist, params);
  const expectedStored = quantize(kernelZ, path.format);
  const reconstructed = reconstructDistance(sample.hardwareDepth, params);
  const id = sample.objectId;

  refreshBlockAtPixel(key, sx, sy);

  const a = analyze(sample.exactViewDist, params, path.format);
  const match = Math.abs(sample.hardwareDepth - expectedStored) <= 2e-7;
  const rows = [
    ['选中板号', `${id == null ? '?' : ID_LABELS[id] ?? id} (id=${id ?? '-'})`],
    ['精确视距 d', sample.exactViewDist.toFixed(5) + ' m'],
    ['内核投影 z', kernelZ.toPrecision(8)],
    ['硬件缓冲 z (MRT)', sample.hardwareDepth.toPrecision(8)],
    ['内核量化期望', expectedStored.toPrecision(8)],
    ['深度附件直读', sample.directDepthRead == null ? 'WebGL2不允许(经R32F读回)' : sample.directDepthRead.toPrecision(8)],
    ['读回通道', sample.depthReadbackChannel],
    ['z 读回一致', match ? '是' : '偏差(检查量化)'],
    ['由深度重建 d\'', reconstructed.toFixed(5) + ' m'],
    ['重建误差', (reconstructed - sample.exactViewDist).toExponential(3) + ' m'],
    ['相对误差', a.relativeError.toExponential(3)],
    ['码格跨度 Δd', a.stepDistance.toExponential(3) + ' m'],
    ['该区块保存', store.storeMode === 'distance' ? '视空间距离' : '仅量化码'],
  ];
  $(
    'readback',
  ).innerHTML = rows.map(([k, v]) => `<span class="k">${k}</span><span class="v mono">${v}</span>`).join('');

  log(
    `路径${key} 拾取 ${ID_LABELS[id] ?? id}: d=${sample.exactViewDist.toFixed(3)} 硬件z=${sample.hardwareDepth.toPrecision(
      4,
    )} 重建误差=${(reconstructed - sample.exactViewDist).toExponential(2)}m`,
    'g',
  );
  updateOcclusionVerdict();
}

// Verify both raster paths against the known geometric occlusion order.
function updateOcclusionVerdict() {
  const pairs = [
    [PLATE_IDS.FAR_A, PLATE_IDS.FAR_B, state.farDistance, state.farGap],
    [PLATE_IDS.NEAR_A, PLATE_IDS.NEAR_B, state.nearDistance, state.nearGap],
  ];
  const lines = [];
  for (const [idA, idB, dist, gap] of pairs) {
    for (const key of ['A', 'B']) {
      const path = state[`path${key}`];
      const params = currentParams(path.mode);
      const resolvable = distinguishable(dist, dist + gap, params, path.format);
      lines.push(
        `<span class="pill ${resolvable ? 'good' : 'bad'}">${key}: ${ID_LABELS[idA]}&gt;${ID_LABELS[idB]} ${
          resolvable ? '次序可分辨' : '同码抢占(z-fight)'
        }</span>`,
      );
    }
  }
  $('rec').innerHTML = lines.join(' ');
}

// ------------------------------------------------- graphical acceptance
const FAR_PLATE_THICKNESS = 0.02;

const REGRESSION_PRESETS = {
  extreme: {
    label: '极远薄层',
    apply(floatSupported) {
      Object.assign(state, {
        near: 0.1,
        far: 100000,
        infiniteFar: true,
        cameraDist: 0.5,
        farDistance: 10000,
        farGap: 2,
        farHalfExtent: 300,
        farY: 0,
        displayMode: 0,
      });
      state.pathA = { format: FORMATS.UNORM24, mode: MODES.CONVENTIONAL };
      state.pathB = {
        format: floatSupported ? FORMATS.FLOAT32 : FORMATS.UNORM24,
        mode: MODES.REVERSED,
      };
    },
  },
  sameCode: {
    label: '同码薄片',
    apply(floatSupported) {
      Object.assign(state, {
        near: 0.1,
        far: 1000,
        infiniteFar: false,
        cameraDist: 0.5,
        farDistance: 60,
        farGap: 0.001,
        farHalfExtent: 3,
        farY: 0,
        displayMode: 0,
      });
      state.pathA = { format: FORMATS.UNORM24, mode: MODES.CONVENTIONAL };
      state.pathB = {
        format: floatSupported ? FORMATS.FLOAT32 : FORMATS.UNORM24,
        mode: MODES.REVERSED,
      };
    },
  },
  nearGraze: {
    label: '近面擦边正例',
    apply(floatSupported) {
      Object.assign(state, {
        near: 1,
        far: 1000,
        infiniteFar: false,
        cameraDist: 0.5,
        farDistance: 0.53,
        farGap: 0.002,
        farHalfExtent: 3,
        farY: 0,
        displayMode: 0,
      });
      state.pathA = { format: FORMATS.UNORM24, mode: MODES.CONVENTIONAL };
      state.pathB = {
        format: floatSupported ? FORMATS.FLOAT32 : FORMATS.UNORM24,
        mode: MODES.REVERSED,
      };
    },
  },
  clipped: {
    label: '近面擦边反例',
    apply(floatSupported) {
      Object.assign(state, {
        near: 1,
        far: 1000,
        infiniteFar: false,
        cameraDist: 0.5,
        farDistance: 0.45,
        farGap: 0.002,
        farHalfExtent: 3,
        farY: 0,
        displayMode: 0,
      });
      state.pathA = { format: FORMATS.UNORM24, mode: MODES.CONVENTIONAL };
      state.pathB = {
        format: floatSupported ? FORMATS.FLOAT32 : FORMATS.UNORM24,
        mode: MODES.REVERSED,
      };
    },
  },
};

function setLogControl(id, valueId, value, digits = 3) {
  $(id).value = String(Math.log10(value));
  if (valueId) $(valueId).textContent = value.toPrecision(digits);
}

function syncControlValues() {
  setLogControl('near', 'nearVal', state.near, 3);
  setLogControl('far', 'farVal', state.far, 3);
  setLogControl('cam', 'camVal', state.cameraDist, 4);
  setLogControl('farDist', 'farDistVal', state.farDistance, 4);
  setLogControl('farGap', 'farGapVal', state.farGap, 3);
  setLogControl('nearDist', 'nearDistVal', state.nearDistance, 4);
  $('infiniteFar').checked = state.infiniteFar;
  $('fmtA').value = state.pathA.format;
  $('fmtB').value = state.pathB.format;
  for (const key of ['A', 'B']) {
    $(`mode${key}`).querySelectorAll('button').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.mode === state[`path${key}`].mode);
    });
  }
  $('displayMode').querySelectorAll('button').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.m) === state.displayMode);
  });
  camera.distance = state.cameraDist;
  camera.yaw = 0;
  camera.pitch = 0;
  setBadges();
}

// Independent ray/plane derivation. Presets use the x/y-centered far plates;
// this intentionally does not read sample.exactViewDist to choose the expected
// distance.
function independentFarPlates() {
  const eye = camera.eye();
  const target = camera.target;
  const length = Math.hypot(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]);
  const forward = [
    (target[0] - eye[0]) / length,
    (target[1] - eye[1]) / length,
    (target[2] - eye[2]) / length,
  ];
  if (!(Math.abs(forward[2]) > 1e-8)) throw new Error('相机方向与薄板平面平行，无法独立推导纵深');
  return [PLATE_IDS.FAR_A, PLATE_IDS.FAR_B].map((id, index) => {
    const centerDistance = state.farDistance + index * state.farGap;
    const faceZ = -centerDistance + FAR_PLATE_THICKNESS;
    const travel = (faceZ - eye[2]) / forward[2];
    const hit = [
      eye[0] + forward[0] * travel,
      eye[1] + forward[1] * travel,
      eye[2] + forward[2] * travel,
    ];
    const half = state.farHalfExtent ?? 3;
    return {
      id,
      numericId: id,
      distance: travel,
      hit,
      insideCoverage: Math.abs(hit[0]) <= half && Math.abs(hit[1]) <= half,
    };
  });
}

function formatRegressionNumber(value, digits = 6) {
  if (value == null) return '—';
  if (value === Infinity) return 'Infinity';
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  return Math.abs(value) >= 1000 || Math.abs(value) < 0.001
    ? value.toExponential(3)
    : value.toPrecision(digits);
}

function renderRegressionReport(caseId, reports) {
  const preset = REGRESSION_PRESETS[caseId];
  const allOk = reports.every((r) => r.verdict.ok);
  const header = `<div class="pill ${allOk ? 'good' : 'bad'}">${preset?.label ?? caseId}：${allOk ? '实际管线验收通过' : '发现坏结果'}</div>`;
  const rows = reports
    .map(({ key, path, verdict, sample, depthState, mrtReferenceError }) => {
      const cls = verdict.ok ? 'good' : 'bad';
      const failures = verdict.failures.length
        ? `<div class="fail">${verdict.failures.map((f) => f.code).join(' · ')}</div>`
        : '';
      const interval = verdict.acceptedDistanceIntervalM
        ? `[${formatRegressionNumber(verdict.acceptedDistanceIntervalM[0])}, ${formatRegressionNumber(verdict.acceptedDistanceIntervalM[1])}] m`
        : '—';
      const targetCode = referenceDepth.code(sample.hardwareDepth, path.format);
      return `<div class="verdict ${cls}">
        <strong>路径 ${key}</strong> · ${path.format} · ${path.mode} · ${verdict.verdict}<br>
        <span class="mono">可见表面 ID=${sample.objectId} (${ID_LABELS[sample.objectId] ?? '背景/未知'}) · 期望前片=${verdict.expectedFrontId ?? 0}</span><br>
        <span class="mono">clear=${depthState.clearValue ?? '?'} · depthFunc=${depthState.depthFunc ?? '?'} · DEPTH_TEST=${depthState.depthTest ?? '?'}</span><br>
        <span class="mono">实际 z=${formatRegressionNumber(sample.hardwareDepth, 8)} · 目标码=${targetCode}</span><br>
        <span class="mono">还原距离=${formatRegressionNumber(sample.reconstructedDistance)} m · 独立参照=${formatRegressionNumber(verdict.pickedReferenceDistanceM)} m</span><br>
        <span class="mono">距离误差=${formatRegressionNumber(verdict.distanceErrorM)} m · 容差=${formatRegressionNumber(verdict.distanceToleranceM)} m · 接受区间=${interval}</span><br>
        <span class="mono">MRT 精确视距互校误差=${formatRegressionNumber(mrtReferenceError)} m</span>
        ${failures}
      </div>`;
    })
    .join('');
  $('regressionReport').innerHTML = header + rows;
}

function runActualDepthAcceptance(caseId = activeRegressionCase) {
  if (!caseId || !refs.A.renderer || !refs.B.renderer) return null;
  const plates = independentFarPlates();
  const reports = [];
  for (const key of ['A', 'B']) {
    const renderer = refs[key].renderer;
    const path = state[`path${key}`];
    const params = currentParams(path.mode);
    renderPath(key);
    const x = Math.floor(renderer.width / 2);
    const y = Math.floor(renderer.height / 2);
    const sample = renderer.readPixel(path.format, x, y);
    if (!sample) throw new Error(`${key}: ${path.format} FBO 不完整或深度读回失败`);
    sample.reconstructedDistance = reconstructDistance(sample.hardwareDepth, params);
    let depthState = {};
    try {
      depthState = renderer.depthState();
    } catch (err) {
      depthState = { depthStateError: err.message };
    }
    const spec = {
      name: `${caseId}-${key}`,
      params,
      format: path.format,
      plates,
      // Geometry positions and projection uniforms are Float32 on the GPU.
      // This prescribed bound is still far below the multi-metre controlled
      // precision failures.
      extraDistanceToleranceM: Math.max(
        ...plates.map((p) => {
          const e = Math.floor(Math.log2(Math.max(1, p.distance)));
          return 8 * 2 ** (e - 23);
        }),
      ),
      codeTolerance: 4,
    };
    const verdict = evaluateDepthRelation({ ...sample, ...depthState }, spec);
    const independentPlate = plates.find((p) => p.numericId === Number(sample.objectId));
    const independentDistance = independentPlate?.distance ?? verdict.pickedReferenceDistanceM;
    const mrtReferenceError =
      sample.exactViewDist > 0 && independentDistance != null
        ? Math.abs(sample.exactViewDist - independentDistance)
        : null;
    reports.push({ key, path, verdict, sample, depthState, mrtReferenceError });
  }
  renderRegressionReport(caseId, reports);
  const allOk = reports.every((r) => r.verdict.ok);
  if (!allOk) {
    const codes = reports.flatMap((r) => r.verdict.failures.map((f) => `${r.key}:${f.code}`));
    showUiError(new Error(`深度管线验收未通过：${codes.join(', ')}`));
  } else {
    $('uiError').hidden = true;
  }
  return reports;
}

function refreshRegressionIfActive() {
  if (!activeRegressionCase) return;
  try {
    runActualDepthAcceptance(activeRegressionCase);
  } catch (err) {
    showUiError(err);
  }
}

function openDepthRegression() {
  const caseId = $('regressionCase').value;
  const preset = REGRESSION_PRESETS[caseId];
  if (!preset) throw new Error(`未知深度回归样例：${caseId}`);
  activeRegressionCase = caseId;
  const floatSupported = refs.A.renderer?.availableFormats().includes(FORMATS.FLOAT32) ?? false;
  preset.apply(floatSupported);
  syncControlValues();
  rebuildScene();
  fullRender();
  try {
    runActualDepthAcceptance(caseId);
    log(`已从光栅调试视窗打开深度精度回归样例：${preset.label}`, 'g');
  } catch (err) {
    showUiError(err);
  }
}

function bindRegression() {
  $('openRegression').addEventListener('click', openDepthRegression);
  $('runActualAcceptance').addEventListener('click', () => {
    activeRegressionCase = $('regressionCase').value;
    try {
      runActualDepthAcceptance(activeRegressionCase);
    } catch (err) {
      showUiError(err);
    }
  });
}

// -------------------------------------------------------- clip changes

function onClipChanged() {
  const sumA = refs.A.store.updateParams(currentParams(state.pathA.mode));
  const sumB = refs.B.store.updateParams(currentParams(state.pathB.mode));
  log(
    `裁面变更 n=${state.near} f=${effectiveFar()} | A[重映射${sumA.remapped} 未决${sumA.undecided} 抢占${sumA.tie}] B[重映射${sumB.remapped} 未决${sumB.undecided} 抢占${sumB.tie}]`,
    sumA.undecided + sumB.undecided + sumA.tie + sumB.tie > 0 ? 'w' : 'g',
  );
  fullRender();
  refreshRegressionIfActive();
}

// ------------------------------------------------------------- sweeps

function runSweep() {
  const rows = sweep({
    nears: [0.01, 0.1, 1],
    ratios: [100, 1000, 10000],
    gap: state.farGap,
    formats: [FORMATS.UNORM24, FORMATS.FLOAT32],
    modes: [MODES.CONVENTIONAL, MODES.REVERSED],
    infinite: [false, true],
  });
  const head =
    '<tr><th>near</th><th>far</th><th>格式</th><th>约定</th><th>远面</th><th>失效距离</th><th>Δd@目标</th></tr>';
  const body = rows
    .map((r) => {
      const target = r.infinite ? r.near * 10000 : r.far;
      return `<tr class="${r.failureDistance >= target ? 'ok' : 'fail'}">
        <td>${r.near}</td><td>${r.far === Infinity ? '∞' : r.far}</td>
        <td>${r.format}</td><td>${r.mode === MODES.REVERSED ? '反向' : '常规'}</td>
        <td>${r.infinite ? '无限' : '有限'}</td>
        <td>${r.failureDistance.toExponential(2)}</td>
        <td>${r.stepAtTarget.toExponential(2)}</td></tr>`;
    })
    .join('');
  $('sweepTable').innerHTML = head + body;
  log(`扫描完成：${rows.length} 组组合，红色=gap ${state.farGap}m 在远端失效`, 'g');
}

function updateRecommend() {
  const target = state.farDistance;
  const rec = recommend({
    near: state.near,
    far: Math.max(state.far, target * 2),
    gap: state.farGap,
    targetDistances: [state.nearDistance, target],
  });
  const failAt = failureDistance(currentParams(state.pathA.mode), state.pathA.format, state.farGap);
  const lines = rec.combos
    .map(
      (c) =>
        `<div class="${c.passes ? 'ok' : 'fail'}">${c.passes ? '✓' : '✗'} ${c.label} · 失效距离 ${c.failureDistance.toExponential(
          2,
        )}m</div>`,
    )
    .join('');
  $('recommend').innerHTML = `
    <div>当前路径A失效范围：<b>d &gt; ${failAt.toExponential(3)} m</b>（间距 ${state.farGap}m 起不可分辨）</div>
    <div style="margin:6px 0">推荐：<span class="pill good">${rec.choice.label}</span> ${rec.failureRange}</div>
    ${lines}`;
}

// --------------------------------------------------------------- UI

function bindSliders() {
  const logSliders = [
    ['near', 'nearVal', 'near', 3],
    ['far', 'farVal', 'far', 3],
    ['farGap', 'farGapVal', 'farGap', 3],
    ['cam', 'camVal', 'cameraDist', 4],
    ['farDist', 'farDistVal', 'farDistance', 4],
    ['nearDist', 'nearDistVal', 'nearDistance', 4],
  ];
  for (const [id, valId, key, digits] of logSliders) {
    const el = $(id);
    const apply = () => {
      state[key] = 10 ** Number(el.value);
      $(valId).textContent = state[key].toPrecision(digits);
    };
    el.addEventListener('input', () => {
      apply();
      if (key === 'near' || key === 'far') onClipChanged();
      else if (key === 'cam') {
        camera.distance = state.cameraDist;
        fullRender();
        refreshRegressionIfActive();
      } else if (key === 'farDist' || key === 'nearDist') {
        rebuildScene();
        fullRender();
        refreshRegressionIfActive();
      } else {
        fullRender();
      }
    });
    apply();
  }
}

function bindSegments() {
  const bindMode = (containerId, key) => {
    $(containerId).querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        $(containerId).querySelectorAll('button').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state[`path${key}`].mode = btn.dataset.mode;
        resetStores();
        setBadges();
        fullRender();
      });
    });
  };
  bindMode('modeA', 'A');
  bindMode('modeB', 'B');

  $('storeMode').querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', () => {
      $('storeMode').querySelectorAll('button').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.storeMode = btn.dataset.s;
      resetStores();
      fullRender();
      log(
        state.storeMode === 'distance'
          ? '区块保存视空间距离：裁面变化可精确重映射'
          : '区块仅存量化码：两候选落入同码格时标为未决/重绘',
      );
    });
  });

  $('displayMode').querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', () => {
      $('displayMode').querySelectorAll('button').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.displayMode = Number(btn.dataset.m);
      fullRender();
    });
  });
}

function bindFormatSelects() {
  const fill = (sel) => {
    const available = refs.A.renderer.availableFormats();
    sel.innerHTML = available.map((f) => `<option value="${f}">${f}</option>`).join('');
  };
  fill($('fmtA'));
  fill($('fmtB'));
  $('fmtA').value = state.pathA.format;
  $('fmtB').value = state.pathB.format;
  $('fmtA').addEventListener('change', (e) => {
    state.pathA.format = e.target.value;
    resetStores();
    fullRender();
  });
  $('fmtB').addEventListener('change', (e) => {
    state.pathB.format = e.target.value;
    resetStores();
    fullRender();
  });
}

function bindButtons() {
  $('dollyIn').addEventListener('click', () => {
    state.cameraDist = Math.max(0.05, state.cameraDist * 0.45);
    $('cam').value = Math.log10(state.cameraDist);
    $('camVal').textContent = state.cameraDist.toPrecision(4);
    for (const key of ['A', 'B']) {
      const r = refs[key].renderer;
      refreshBlockAtPixel(key, Math.floor(r.width / 2), Math.floor(r.height / 2));
    }
    log('推近相机：仅对中心区块执行 scissor 局部刷新（其余保留历史样本）', 'g');
    fullRender();
  });
  $('dollyOut').addEventListener('click', () => {
    state.cameraDist *= 1.8;
    $('cam').value = Math.log10(state.cameraDist);
    $('camVal').textContent = state.cameraDist.toPrecision(4);
    fullRender();
  });
  $('fullRefresh').addEventListener('click', () => {
    resetStores();
    fullRender();
    log('全量重绘，区块历史已清空');
  });
  $('clipShift').addEventListener('click', () => {
    state.near *= 4;
    state.far = Math.max(10, state.far * 0.4);
    $('near').value = Math.log10(state.near);
    $('far').value = Math.log10(state.far);
    $('nearVal').textContent = state.near.toPrecision(3);
    $('farVal').textContent = state.far.toPrecision(3);
    onClipChanged();
  });
  $('sweepBtn').addEventListener('click', runSweep);
  $('infiniteFar').addEventListener('change', (e) => {
    state.infiniteFar = e.target.checked;
    resetStores();
    onClipChanged();
  });
}

function bindCanvasInteraction(key, canvas) {
  let dragging = false;
  let downPos = null;
  canvas.addEventListener('pointerdown', (ev) => {
    dragging = true;
    downPos = [ev.clientX, ev.clientY];
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!dragging || !downPos) return;
    const dx = ev.clientX - downPos[0];
    const dy = ev.clientY - downPos[1];
    downPos = [ev.clientX, ev.clientY];
    camera.yaw += dx * 0.005;
    camera.pitch = Math.max(-1.3, Math.min(1.3, camera.pitch + dy * 0.005));
    fullRender();
  });
  canvas.addEventListener('pointerup', (ev) => {
    const moved = downPos ? Math.hypot(ev.clientX - downPos[0], ev.clientY - downPos[1]) : 99;
    dragging = false;
    if (moved < 5) pick(key, canvas, ev);
  });
  canvas.addEventListener(
    'wheel',
    (ev) => {
      ev.preventDefault();
      camera.dolly(ev.deltaY > 0 ? 1.12 : 0.89);
      state.cameraDist = camera.distance;
      $('cam').value = Math.log10(state.cameraDist);
      $('camVal').textContent = state.cameraDist.toPrecision(4);
      fullRender();
    },
    { passive: false },
  );
}

function setBadges() {
  $('badgeA').textContent = `${state.pathA.format} · ${state.pathA.mode === MODES.REVERSED ? '反向' : '常规'}`;
  $('badgeB').textContent = `${state.pathB.format} · ${state.pathB.mode === MODES.REVERSED ? '反向' : '常规'}`;
}

// --------------------------------------------------------------- boot

function sizeCharts() {
  for (const id of ['chartCanvas', 'blockCanvas']) {
    const c = $(id);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(200, Math.round(c.clientWidth * dpr));
    const h = id === 'chartCanvas' ? 180 : 150;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }
}

function boot() {
  try {
    refs.A.renderer = initRenderer($('canvasA'));
    refs.B.renderer = initRenderer($('canvasB'));
  } catch (err) {
    $('caps').innerHTML = `<span class="pill bad">${err.message}</span>`;
    log(`初始化失败：${err.message}`, 'e');
    return;
  }
  const caps = [
    `FLOAT32 颜色附件: ${refs.A.renderer.extFloat ? '可用' : '不可用'}`,
    `FBO 格式: ${refs.A.renderer.availableFormats().join(', ')}`,
  ];
  $('caps').innerHTML = caps.map((c) => `<span class="pill">${c}</span>`).join(' ');

  // Fall back any path whose format is incomplete on this driver.
  for (const key of ['A', 'B']) {
    const path = state[`path${key}`];
    if (!refs[key].renderer.availableFormats().includes(path.format)) {
      path.format = FORMATS.UNORM24;
      log(`路径${key} 的 FLOAT32 FBO 不可用，已回退到 UNORM24`, 'w');
    }
  }

  sizeCharts();
  window.addEventListener('resize', () => {
    sizeCharts();
    fullRender();
  });
  bindSliders();
  rebuildScene();
  bindSegments();
  bindFormatSelects();
  bindButtons();
  bindRegression();
  bindCanvasInteraction('A', $('canvasA'));
  bindCanvasInteraction('B', $('canvasB'));
  setBadges();
  fullRender();
  runSweep();
  log('启动：同场景双路径。先在远景 FAR-A/FAR-B 上点选两层薄板，再推近相机局部刷新', 'g');
  log('调整 near/far 触发历史样本换算；切到“仅量化码”观察同码格未决；误差图查看重建误差', 'g');
}

if (typeof window !== 'undefined') {
  window.depthLabRegression = {
    presets: Object.keys(REGRESSION_PRESETS),
    open: openDepthRegression,
    run: runActualDepthAcceptance,
  };
}

try {
  boot();
} catch (err) {
  const caps = document.getElementById('caps');
  if (caps) caps.innerHTML = `<span class="pill bad">${err.message}</span>`;
  showUiError(err);
}
