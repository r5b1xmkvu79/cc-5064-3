// WebGL2 raster renderer for the depth precision lab.
//
// For each depth-storage format one offscreen FBO is rendered with the same
// geometry. MRT attachments:
//   0 RGBA8  shaded color (object-tinted)
//   1 R32F   actual hardware window depth written by the rasterizer (gl_FragCoord.z)
//   2 R32F   exact view-space distance (pre-quantization ground truth)
//   3 R32UI  integer object id (stable picking / block identity, no color guess)
// WebGL2 cannot readPixels a depth attachment, so the R32F hardware-depth
// target is the real buffer readback channel.

import { mat4Perspective, mat4PerspectiveInfinite } from './mat4.js';
import { MODES } from './depth-kernel.js';
import { ID_COLORS } from './geometry.js';

const SCENE_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in float aId;
uniform mat4 uView;
uniform mat4 uProj;
flat out int vId;
out float vViewDist;
void main() {
  vec4 view = uView * vec4(aPos, 1.0);
  vViewDist = -view.z;
  gl_Position = uProj * view;
  vId = int(aId + 0.5);
}`;

const SCENE_FS = `#version 300 es
precision highp float;
flat in int vId;
in float vViewDist;
uniform vec3 uColorTable[10];
layout(location=0) out vec4 outColor;
layout(location=1) out float outHardwareDepth;
layout(location=2) out float outViewDist;
layout(location=3) out uint outId;
void main() {
  vec3 base = uColorTable[vId];
  float shade = 0.72 + 0.28 * fract(float(vId) * 0.6180339);
  outColor = vec4(base * shade, 1.0);
  outHardwareDepth = gl_FragCoord.z;
  outViewDist = vViewDist;
  outId = uint(vId);
}`;

const POST_VS = `#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main(){ vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const POST_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uColor;
uniform sampler2D uHwDepth;
uniform sampler2D uExactDist;
uniform float uNear;
uniform float uFar;       // 0 encodes infinite
uniform int uReverse;
uniform int uMode;        // 0 shaded, 1 error heatmap, 2 raw depth
out vec4 frag;
vec3 heat(float t) {
  t = clamp(t, 0.0, 1.0);
  return mix(vec3(0.1,0.2,0.9), mix(vec3(0.2,0.9,0.3), vec3(1.0,0.15,0.1), t), smoothstep(0.25, 1.0, t));
}
void main() {
  vec4 base = texture(uColor, vUv);
  float hw = texture(uHwDepth, vUv).r;
  float exact = texture(uExactDist, vUv).r;
  if (exact <= 0.0) { frag = base; return; }
  if (uMode == 2) {
    frag = vec4(vec3(hw), 1.0);
    return;
  }
  if (uMode == 1) {
    // Reconstruct distance from the stored hardware depth using the same
    // window-depth convention as the JS kernel. Epsilon guards match the
    // kernel's Infinity behavior at the far/near window-depth endpoints.
    float eps = 1e-30;
    float recon;
    if (uFar <= 0.0) {
      recon = uReverse == 1 ? uNear / max(hw, eps) : uNear / max(1.0 - hw, eps);
    } else if (uReverse == 1) {
      recon = (uNear * uFar) / max(uNear + hw * (uFar - uNear), eps);
    } else {
      recon = (uNear * uFar) / max(uFar - hw * (uFar - uNear), eps);
    }
    if (!isinf(recon) && !isnan(recon)) {
      float err = abs(recon - exact);
      float rel = err / exact;
      // Log-scaled heat: 1% relative error saturates red.
      float t = log(1.0 + rel * 1000.0) / log(1.0 + 10.0);
      frag = vec4(heat(t), 1.0);
    } else {
      frag = vec4(1.0, 0.15, 0.1, 1.0); // endpoint overflow -> red
    }
    return;
  }
  frag = base;
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function makeProgram(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

// Projection matrix matching depth-kernel window depth.
// Conventional: near->0, far->1. Reversed: near->1, far->0.
export function buildProjection({ fovy, aspect, near, far, mode }) {
  const reverse = mode === MODES.REVERSED;
  if (far === Infinity) {
    const m = mat4PerspectiveInfinite(fovy, aspect, near);
    if (reverse) {
      m[10] = 1;      // z_ndc = 1 - 2n/w  -> window z = n/d
      m[14] = 2 * near;
    }
    return m;
  }
  const m = mat4Perspective(fovy, aspect, near, far);
  if (reverse) {
    m[10] = -m[10];
    m[14] = -m[14];
  }
  return m;
}

export class DepthRenderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', { antialias: false });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.extFloat = gl.getExtension('EXT_color_buffer_float');
    this.canDirectDepthRead = null;
    this.sceneProg = makeProgram(gl, SCENE_VS, SCENE_FS);
    this.postProg = makeProgram(gl, POST_VS, POST_FS);
    this.loc = {
      view: gl.getUniformLocation(this.sceneProg, 'uView'),
      proj: gl.getUniformLocation(this.sceneProg, 'uProj'),
      colors: gl.getUniformLocation(this.sceneProg, 'uColorTable'),
      post: {
        color: gl.getUniformLocation(this.postProg, 'uColor'),
        hw: gl.getUniformLocation(this.postProg, 'uHwDepth'),
        exact: gl.getUniformLocation(this.postProg, 'uExactDist'),
        near: gl.getUniformLocation(this.postProg, 'uNear'),
        far: gl.getUniformLocation(this.postProg, 'uFar'),
        reverse: gl.getUniformLocation(this.postProg, 'uReverse'),
        mode: gl.getUniformLocation(this.postProg, 'uMode'),
      },
    };
    this._colorTable();
    this._quadVao();
    this.fbos = {};
    this.width = 0;
    this.height = 0;
  }

  _colorTable() {
    const table = new Float32Array(30);
    for (const idStr of Object.keys(ID_COLORS)) {
      const id = Number(idStr);
      const c = ID_COLORS[id];
      table.set(c, id * 3);
    }
    this.colorTable = table;
  }

  _quadVao() {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.quadVao = vao;
  }

  resize(w, h) {
    w = Math.max(2, Math.floor(w));
    h = Math.max(2, Math.floor(h));
    if (this.width === w && this.height === h) return;
    this.width = w; this.height = h;
    const gl = this.gl;
    gl.canvas.width = w; gl.canvas.height = h;
    for (const f of Object.values(this.fbos)) this._destroyFbo(f);
    this.fbos = {};
    this.fbos.UNORM24 = this._createFbo(gl.DEPTH_COMPONENT24, gl.UNSIGNED_INT, false);
    if (this.extFloat) this.fbos.FLOAT32 = this._createFbo(gl.DEPTH_COMPONENT32F, gl.FLOAT, true);
  }

  _createFbo(depthInternal, depthType, floatDepth) {
    const gl = this.gl;
    const { width: w, height: h } = this;
    const mkColor = (internal) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, internal, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      return t;
    };
    const color0 = mkColor(gl.RGBA8);
    const color1 = mkColor(gl.R32F);
    const color2 = mkColor(gl.R32F);
    const color3 = mkColor(gl.R32UI);
    const depth = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, depth);
    gl.texStorage2D(gl.TEXTURE_2D, 1, depthInternal, w, h);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color0, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, color1, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, color2, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, color3, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return {
      fbo, color0, color1, color2, color3, depth, depthType, floatDepth,
      incomplete: status !== gl.FRAMEBUFFER_COMPLETE, status,
    };
  }

  _destroyFbo(f) {
    if (!f) return;
    const gl = this.gl;
    [f.color0, f.color1, f.color2, f.color3, f.depth].forEach((t) => t && gl.deleteTexture(t));
    if (f.fbo) gl.deleteFramebuffer(f.fbo);
  }

  availableFormats() {
    return Object.keys(this.fbos).filter((k) => !this.fbos[k].incomplete);
  }

  uploadMesh(mesh) {
    const gl = this.gl;
    if (this.mesh) {
      gl.deleteVertexArray(this.mesh.vao);
      gl.deleteBuffer(this.mesh.pbo); gl.deleteBuffer(this.mesh.ibo); gl.deleteBuffer(this.mesh.idbo);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const pbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pbo);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const idbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, idbo);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.ids, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.mesh = { vao, pbo, idbo, ibo, count: mesh.indices.length };
  }

  uploadGround(ground) {
    const gl = this.gl;
    if (this.ground) {
      gl.deleteVertexArray(this.ground.vao);
      gl.deleteBuffer(this.ground.vbo);
      gl.deleteBuffer(this.ground.idbo);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, ground.positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const idbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, idbo);
    gl.bufferData(gl.ARRAY_BUFFER, ground.ids, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.ground = { vao, vbo, idbo, count: ground.positions.length / 3 };
  }

  _configureDepth(mode) {
    const gl = this.gl;
    // Keep the full [0,1] window-depth range so conventions are explicit.
    gl.depthRange(0, 1);
    if (mode === MODES.REVERSED) {
      gl.clearDepth(0);
      gl.depthFunc(gl.GEQUAL);
    } else {
      gl.clearDepth(1);
      gl.depthFunc(gl.LEQUAL);
    }
    gl.enable(gl.DEPTH_TEST);
  }

  // Render one FBO. blockRect [x,y,w,h] (top-left origin) scissor-refreshes
  // only that region; null renders fully.
  renderFbo(format, { view, proj, mode, blockRect = null }) {
    const gl = this.gl;
    const target = this.fbos[format];
    if (!target || target.incomplete) return null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, this.width, this.height);
    this._configureDepth(mode);
    gl.enable(gl.SCISSOR_TEST);
    if (blockRect) {
      const [x, topY, w, h] = blockRect;
      gl.scissor(x, this.height - topY - h, w, h);
    } else {
      gl.scissor(0, 0, this.width, this.height);
    }
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
      gl.COLOR_ATTACHMENT2,
      gl.COLOR_ATTACHMENT3,
    ]);

    gl.useProgram(this.sceneProg);
    gl.uniformMatrix4fv(this.loc.view, false, view);
    gl.uniformMatrix4fv(this.loc.proj, false, proj);
    gl.uniform3fv(this.loc.colors, this.colorTable);
    gl.bindVertexArray(this.mesh.vao);
    gl.drawElements(gl.TRIANGLES, this.mesh.count, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(this.ground.vao);
    gl.drawArrays(gl.LINES, 0, this.ground.count);
    gl.bindVertexArray(null);
    gl.disable(gl.SCISSOR_TEST);
    return target;
  }

  // Post-process an FBO onto the default framebuffer.
  // displayMode: 0 shaded, 1 error heatmap, 2 raw hardware depth.
  present(format, { near, far, mode, displayMode }) {
    const gl = this.gl;
    const target = this.fbos[format];
    if (!target || target.incomplete) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.DEPTH_TEST);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.postProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, target.color0);
    gl.uniform1i(this.loc.post.color, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, target.color1);
    gl.uniform1i(this.loc.post.hw, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, target.color2);
    gl.uniform1i(this.loc.post.exact, 2);
    gl.uniform1f(this.loc.post.near, near);
    gl.uniform1f(this.loc.post.far, far === Infinity ? 0 : far);
    gl.uniform1i(this.loc.post.reverse, mode === MODES.REVERSED ? 1 : 0);
    gl.uniform1i(this.loc.post.mode, displayMode);
    gl.bindVertexArray(this.quadVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  // Read actual buffers at one CSS pixel (top-left origin).
  readPixel(format, x, topY) {
    const gl = this.gl;
    const target = this.fbos[format];
    if (!target || target.incomplete) return null;
    const y = this.height - 1 - topY;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    const rgba = new Uint8Array(4);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    const hw = new Float32Array(1);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(x, y, 1, 1, gl.RED, gl.FLOAT, hw);
    const exact = new Float32Array(1);
    gl.readBuffer(gl.COLOR_ATTACHMENT2);
    gl.readPixels(x, y, 1, 1, gl.RED, gl.FLOAT, exact);
    const idBuf = new Uint32Array(1);
    gl.readBuffer(gl.COLOR_ATTACHMENT3);
    gl.readPixels(x, y, 1, 1, gl.RED_INTEGER, gl.UNSIGNED_INT, idBuf);

    // WebGL2 does not permit readPixels on a depth attachment. The R32F
    // attachment `color1` stores gl_FragCoord.z and IS the real hardware
    // window depth; report that the direct path is unavailable so the UI
    // never confuses the two readback channels.
    const direct = null;
    this.canDirectDepthRead = false;
    return {
      color: [rgba[0], rgba[1], rgba[2], rgba[3]],
      objectId: idBuf[0],
      hardwareDepth: hw[0],
      exactViewDist: exact[0],
      directDepthRead: direct,
      depthReadbackChannel: 'R32F MRT (gl_FragCoord.z); depth-attachment readPixels not allowed in WebGL2',
    };
  }

  depthState() {
    const gl = this.gl;
    const func = gl.getParameter(gl.DEPTH_FUNC);
    return {
      clearValue: gl.getParameter(gl.DEPTH_CLEAR_VALUE),
      depthFunc:
        func === gl.GEQUAL ? 'GEQUAL' : func === gl.LEQUAL ? 'LEQUAL' : `GLenum(${func})`,
      depthTest: gl.isEnabled(gl.DEPTH_TEST),
      directDepthRead: false,
    };
  }

  // Aggregate block readback for the block store / error statistics.
  readBlock(format, x, topY, w, h) {
    const gl = this.gl;
    const target = this.fbos[format];
    if (!target || target.incomplete) return null;
    const fy = this.height - topY - h;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    const hw = new Float32Array(w * h);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(x, fy, w, h, gl.RED, gl.FLOAT, hw);
    const exact = new Float32Array(w * h);
    gl.readBuffer(gl.COLOR_ATTACHMENT2);
    gl.readPixels(x, fy, w, h, gl.RED, gl.FLOAT, exact);
    const ids = new Uint32Array(w * h);
    gl.readBuffer(gl.COLOR_ATTACHMENT3);
    gl.readPixels(x, fy, w, h, gl.RED_INTEGER, gl.UNSIGNED_INT, ids);
    const color = new Uint8Array(w * h * 4);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(x, fy, w, h, gl.RGBA, gl.UNSIGNED_BYTE, color);
    return { hardware: hw, exact, color, ids, w, h };
  }
}
