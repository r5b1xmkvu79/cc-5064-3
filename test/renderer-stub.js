// Minimal WebGL2 stub: enough to construct DepthRenderer and exercise FBO
// setup, mesh upload, renderFbo, present and readPixel without a GPU. This
// catches GLSL typo-level / uniform-wiring / attachment-config mistakes in
// Node; real rasterization still requires a browser (see README).
import { DepthRenderer, buildProjection } from '../src/renderer.js';
import { buildScene } from '../src/geometry.js';
import { MODES } from '../src/depth-kernel.js';

function makeCanvas(w = 320, h = 240) {
  return {
    width: w,
    height: h,
    clientWidth: w,
    getContext: () => gl,
  };
}

const textures = [];
const buffers = [];
const vaos = [];
const fbos = [];
const shaders = [];
const programs = [];

const gl = {
  VERTEX_SHADER: 1, FRAGMENT_SHADER: 2, COMPILE_STATUS: 3, LINK_STATUS: 4,
  ARRAY_BUFFER: 10, ELEMENT_ARRAY_BUFFER: 11, STATIC_DRAW: 12,
  TEXTURE_2D: 20, TEXTURE_MIN_FILTER: 21, TEXTURE_MAG_FILTER: 22, NEAREST: 23,
  FRAMEBUFFER: 30, COLOR_ATTACHMENT0: 40, COLOR_ATTACHMENT1: 41, COLOR_ATTACHMENT2: 42,
  COLOR_ATTACHMENT3: 43, DEPTH_ATTACHMENT: 44, FRAMEBUFFER_COMPLETE: 45,
  RGBA8: 50, R32F: 51, R32UI: 52, DEPTH_COMPONENT24: 53, DEPTH_COMPONENT32F: 54,
  UNSIGNED_INT: 60, FLOAT: 61, RGBA: 62, UNSIGNED_BYTE: 63, RED: 64, RED_INTEGER: 65,
  TEXTURE0: 100,
  TRIANGLES: 70, LINES: 71, DEPTH_TEST: 72, SCISSOR_TEST: 73,
  LEQUAL: 80, GEQUAL: 81, COLOR_BUFFER_BIT: 82, DEPTH_BUFFER_BIT: 83,
  READ_FRAMEBUFFER: 90, DRAW_FRAMEBUFFER: 91,
  EXT_color_buffer_float: true,
  canvas: null,
  getExtension(name) {
    return name === 'EXT_color_buffer_float' ? {} : null;
  },
  createShader: () => ({ id: shaders.push({}) }),
  shaderSource() {},
  compileShader() {},
  getShaderParameter: () => true,
  getShaderInfoLog: () => '',
  createProgram: () => {
    const p = { id: programs.length + 1, uniforms: {} };
    programs.push(p);
    return p;
  },
  attachShader() {},
  linkProgram() {},
  getProgramParameter: () => true,
  getProgramInfoLog: () => '',
  useProgram() {},
  getUniformLocation(p, name) {
    const loc = { p: p.id, name };
    p.uniforms[name] = loc;
    return loc;
  },
  uniform1i() {}, uniform1f() {}, uniform3f() {}, uniform3fv() {},
  uniformMatrix4fv() {},
  createTexture: () => ({ id: textures.length + 1 }),
  bindTexture() {},
  activeTexture() {},
  texStorage2D() {},
  texParameteri() {},
  createBuffer: () => ({ id: buffers.length + 1 }),
  bindBuffer() {},
  bufferData() {},
  createVertexArray: () => ({ id: vaos.length + 1 }),
  bindVertexArray() {},
  enableVertexAttribArray() {},
  disableVertexAttribArray() {},
  vertexAttribPointer() {},
  createFramebuffer: () => ({ id: fbos.length + 1 }),
  bindFramebuffer() {},
  framebufferTexture2D() {},
  checkFramebufferStatus() {
    return gl.FRAMEBUFFER_COMPLETE;
  },
  drawBuffers() {},
  viewport() {},
  enable() {},
  disable() {},
  depthFunc() {},
  depthRange() {},
  clearDepth() {},
  clear() {},
  scissor() {},
  drawElements() {},
  drawArrays() {},
  readBuffer() {},
  readPixels(target, x, y, w, h, format, type, buf) {
    if (buf && typeof buf === 'object') buf.fill(0);
  },
  blitFramebuffer() {},
  deleteTexture() {}, deleteBuffer() {}, deleteVertexArray() {}, deleteFramebuffer() {},
};

let failures = 0;
const check = (cond, msg) => {
  if (!cond) {
    failures++;
    console.error('FAIL:', msg);
  } else {
    console.log('ok:', msg);
  }
};

const canvas = makeCanvas();
gl.canvas = canvas;
const r = new DepthRenderer(canvas);
r.resize(320, 240);
check(r.availableFormats().includes('UNORM24'), 'UNORM24 FBO complete');
check(r.availableFormats().includes('FLOAT32'), 'FLOAT32 FBO complete with ext');

const { mesh, ground } = buildScene({
  farDistance: 60, farGap: 0.5, nearDistance: 12, nearGap: 0.2, farY: 0, nearY: 0,
});
r.uploadMesh(mesh);
r.uploadGround(mesh && ground);
check(r.mesh.count === mesh.indices.length, 'mesh index count uploaded');
check(r.ground.count === ground.positions.length / 3, 'ground vertex count uploaded');

for (const format of ['UNORM24', 'FLOAT32']) {
  for (const mode of [MODES.CONVENTIONAL, MODES.REVERSED]) {
    const params = { near: 0.1, far: 1000, mode };
    const proj = buildProjection({ fovy: 1, aspect: 4 / 3, ...params });
    const view = new Float32Array(16);
    const out = r.renderFbo(format, { view, proj, mode });
    check(!!out && !out.incomplete, `renderFbo ${format} ${mode}`);
    r.present(format, { near: 0.1, far: 1000, mode, displayMode: 1 });
    const px = r.readPixel(format, 10, 10);
    check(px && typeof px.hardwareDepth === 'number' && 'objectId' in px, `readPixel ${format} ${mode}`);
    const blk = r.readBlock(format, 0, 0, 64, 64);
    check(blk.ids.length === 64 * 64, `readBlock ids sized ${format}`);
  }
}

// Infinite far reversed projection must be finite at the near plane.
const inf = buildProjection({ fovy: 1, aspect: 1, near: 0.1, far: Infinity, mode: MODES.REVERSED });
check(Number.isFinite(inf[10]) && Number.isFinite(inf[14]), 'infinite reversed projection finite');

if (failures) {
  console.error(`\n${failures} stub check(s) failed`);
  process.exit(1);
}
console.log('\nAll renderer stub checks passed.');
