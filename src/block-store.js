// Block store for scissored local depth refresh and historical sample
// conversion. The screen is divided into fixed blocks; each block remembers
// either exact view-space distances or only quantized depth codes, captured
// under the clip parameters in effect at refresh time.

import {
  FORMATS,
  normalizeParams,
  createBlockRecord,
  remapBlock,
  depthCode,
  projectDepth,
  quantize,
  reconstructDistance,
} from './depth-kernel.js';

export const BLOCK_STATE = Object.freeze({
  FRESH: 'fresh',
  REMAPPED: 'remapped',
  UNDECIDED: 'undecided',
  TIE: 'tie',
  EMPTY: 'empty',
});

export class BlockStore {
  constructor({ blockSize = 64, format = FORMATS.UNORM24, store = 'distance' }) {
    this.blockSize = blockSize;
    this.format = format;
    this.storeMode = store; // 'distance' | 'code'
    this.blocks = new Map();
    this.params = null;
    this.cols = 0;
    this.rows = 0;
  }

  setViewport(width, height) {
    this.cols = Math.ceil(width / this.blockSize);
    this.rows = Math.ceil(height / this.blockSize);
    this.blocks.clear();
  }

  key(col, row) {
    return row * this.cols + col;
  }

  blockAtPixel(x, y) {
    return {
      col: Math.floor(x / this.blockSize),
      row: Math.floor(y / this.blockSize),
    };
  }

  rectOf(col, row) {
    return [
      col * this.blockSize,
      row * this.blockSize,
      Math.min(this.blockSize, this.cols * this.blockSize - col * this.blockSize),
      Math.min(this.blockSize, this.rows * this.blockSize - row * this.blockSize),
    ];
  }

  // Ingest a readback (hardware depth + exact distance + integer object id)
  // for one block. Candidates = distinct visible object ids in the block,
  // each tagged with its nearest exact view-space distance.
  ingest(col, row, readback, params) {
    this.params = normalizeParams(params);
    const found = new Map(); // id -> nearest exact distance
    let anyGeometry = false;
    for (let i = 0; i < readback.exact.length; i++) {
      const d = readback.exact[i];
      if (!(d > 0)) continue;
      anyGeometry = true;
      const id = readback.ids[i];
      if (!id) continue;
      const prev = found.get(id);
      if (prev === undefined || d < prev) found.set(id, d);
    }
    const k = this.key(col, row);
    if (!anyGeometry || found.size === 0) {
      this.blocks.set(k, { state: BLOCK_STATE.EMPTY, record: null });
      return this.blocks.get(k);
    }
    const candidates = [...found.entries()].map(([id, distance]) => ({ id, distance }));
    const record = createBlockRecord({
      candidates,
      params: this.params,
      format: this.format,
      store: this.storeMode,
    });
    const block = { state: BLOCK_STATE.FRESH, record, candidates };
    this.blocks.set(k, block);
    return block;
  }

  get(col, row) {
    return this.blocks.get(this.key(col, row)) ?? null;
  }

  // Clip parameters changed: distance blocks remap exactly; code-only blocks
  // either remap, go undecided (overlapping new cells), or tie.
  updateParams(newParams) {
    const np = normalizeParams(newParams);
    const summary = { remapped: 0, undecided: 0, tie: 0, empty: 0, fresh: 0 };
    for (const block of this.blocks.values()) {
      if (block.state === BLOCK_STATE.EMPTY || !block.record) {
        summary.empty++;
        continue;
      }
      const wasFresh = block.state === BLOCK_STATE.FRESH;
      const result = remapBlock(block.record, np);
      block.lastResult = result;
      if (result.status === 'undecided') {
        block.state = BLOCK_STATE.UNDECIDED;
        summary.undecided++;
      } else if (result.status === 'tie') {
        block.state = BLOCK_STATE.TIE;
        summary.tie++;
      } else {
        // A clip change means any surviving block is a remap of a historical
        // sample; only a block ingested after this call stays FRESH.
        block.state = BLOCK_STATE.REMAPPED;
        block.record.params = np;
        if (block.record.store === 'distance') {
          block.record.candidates = result.candidates.map((c) => ({
            id: c.id,
            distance: c.distance,
          }));
        }
        block.winnerId = result.winnerId;
        if (wasFresh) summary.fresh++;
        summary.remapped++;
      }
    }
    this.params = np;
    return summary;
  }

  // Recompute a stored winner's expected code under current params for display.
  expectedCodes() {
    const out = [];
    for (const [k, block] of this.blocks) {
      if (!block.record) continue;
      const rows = block.record.candidates.map((c) => {
        if (block.record.store === 'distance') {
          const z = projectDepth(c.distance, this.params);
          return { id: c.id, distance: c.distance, code: depthCode(z, this.format), stored: quantize(z, this.format) };
        }
        return { id: c.id, code: c.code };
      });
      out.push({ key: k, state: block.state, rows, winnerId: block.winnerId ?? null });
    }
    return out;
  }

  // Reconstruct view distance from a hardware depth sample under current params.
  reconstruct(hardwareDepth) {
    return reconstructDistance(hardwareDepth, this.params);
  }
}
