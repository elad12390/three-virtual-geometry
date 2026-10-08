/**
 * Procedural showcase assets for `?scene=ruins`. All detail is real geometry: stone blocks with bevels and
 * chipped edges, fluted columns with carved capitals, paving where every stone is modelled, carved relief
 * panels, stratified rocks and a twisted bronze sculpture.
 *
 * Meshes are written straight into typed arrays as welded, indexed triangle lists with smooth, area-weighted
 * normals. three.js helpers (mergeVertices, BufferGeometry) are avoided: they take seconds at these sizes.
 */
import type { VirtualMeshSource } from '../../src/index';
import { Perlin, mulberry32, smoothstep } from './noise';

const TAU = Math.PI * 2;
const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);
const fract = (x: number) => x - Math.floor(x);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

export type RGB = [number, number, number];

/** sRGB hex to linear RGB (vertex colors multiply the linear material color). */
export function rgb(hex: number): RGB {
  return [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255].map((c) => Math.pow(c / 255, 2.2)) as RGB;
}

/** out = mix(out, c, t) */
function blend(out: RGB, c: RGB, t: number) {
  out[0] += (c[0] - out[0]) * t;
  out[1] += (c[1] - out[1]) * t;
  out[2] += (c[2] - out[2]) * t;
}

function scaleRGB(out: RGB, k: number) {
  out[0] *= k;
  out[1] *= k;
  out[2] *= k;
}

export const PALETTE = {
  limestone: rgb(0xc4ad88),
  travertine: rgb(0xcbae82),
  greyStone: rgb(0xa39581),
  darkStone: rgb(0x8a7964),
  marble: rgb(0xd8cab0),
  dirt: rgb(0x6b5538),
  moss: rgb(0x5e6a2e),
  lichen: rgb(0xa6974f),
  bronze: rgb(0x8a6034),
  verdigris: rgb(0x4f8c78),
};

// ---------------------------------------------------------------------------- mesh building

function growF(a: Float32Array) {
  const b = new Float32Array(a.length * 2);
  b.set(a);
  return b;
}

function growU(a: Uint32Array) {
  const b = new Uint32Array(a.length * 2);
  b.set(a);
  return b;
}

/** Growable vertex (position + color) and index buffers. */
export class MeshBuilder {
  positions: Float32Array;
  colors: Float32Array;
  indices: Uint32Array;
  vertexCount = 0;
  indexCount = 0;

  constructor(vertices = 1 << 16, triangles = 1 << 17) {
    this.positions = new Float32Array(vertices * 3);
    this.colors = new Float32Array(vertices * 3);
    this.indices = new Uint32Array(triangles * 3);
  }

  vertex(x: number, y: number, z: number, c: RGB) {
    const o = this.vertexCount * 3;
    if (o >= this.positions.length) {
      this.positions = growF(this.positions);
      this.colors = growF(this.colors);
    }
    this.positions[o] = x;
    this.positions[o + 1] = y;
    this.positions[o + 2] = z;
    this.colors[o] = c[0];
    this.colors[o + 1] = c[1];
    this.colors[o + 2] = c[2];
    return this.vertexCount++;
  }

  triangle(a: number, b: number, c: number) {
    if (this.indexCount + 3 > this.indices.length) this.indices = growU(this.indices);
    this.indices[this.indexCount++] = a;
    this.indices[this.indexCount++] = b;
    this.indices[this.indexCount++] = c;
  }

  /** Quad a-b-c-d, counter-clockwise seen from the front. */
  quad(a: number, b: number, c: number, d: number) {
    this.triangle(a, b, c);
    this.triangle(a, c, d);
  }

  finish(): VirtualMeshSource {
    const positions = this.positions.slice(0, this.vertexCount * 3);
    const indices = this.indices.slice(0, this.indexCount);
    return { positions, normals: computeNormals(positions, indices), colors: this.colors.slice(0, this.vertexCount * 3), indices };
  }
}

/** Smooth, area-weighted vertex normals. */
export function computeNormals(positions: Float32Array, indices: Uint32Array) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3;
    const b = indices[i + 1] * 3;
    const c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    normals[a] += nx;
    normals[a + 1] += ny;
    normals[a + 2] += nz;
    normals[b] += nx;
    normals[b + 1] += ny;
    normals[b + 2] += nz;
    normals[c] += nx;
    normals[c + 1] += ny;
    normals[c + 2] += nz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const l = Math.sqrt(normals[i] * normals[i] + normals[i + 1] * normals[i + 1] + normals[i + 2] * normals[i + 2]);
    if (l > 0) {
      normals[i] /= l;
      normals[i + 1] /= l;
      normals[i + 2] /= l;
    } else normals[i + 1] = 1;
  }
  return normals;
}

/** Recolors a finished mesh from position and normal (e.g. moss on upward faces). */
function recolor(mesh: VirtualMeshSource, fn: (i: number, p: Float32Array, n: Float32Array, out: RGB) => void) {
  const c: RGB = [0, 0, 0];
  const colors = mesh.colors!;
  for (let i = 0; i < colors.length / 3; i++) {
    c[0] = colors[i * 3];
    c[1] = colors[i * 3 + 1];
    c[2] = colors[i * 3 + 2];
    fn(i, mesh.positions, mesh.normals, c);
    colors[i * 3] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }
}

let latticeScratch = new Int32Array(0);

/**
 * Surface of an nx * ny * nz lattice (a subdivided cube), welded and wound outward. `vertexAt(i, j, k)` is called
 * once per surface lattice point and returns the index of the vertex it added.
 */
function cubeSurface(mb: MeshBuilder, nx: number, ny: number, nz: number, vertexAt: (i: number, j: number, k: number) => number) {
  const sy = nx + 1;
  const sz = (nx + 1) * (ny + 1);
  const size = sz * (nz + 1);
  const map = size > 1 << 22 ? new Map<number, number>() : null;
  if (!map) {
    if (latticeScratch.length < size) latticeScratch = new Int32Array(size);
    latticeScratch.fill(-1, 0, size);
  }
  const lattice = latticeScratch;
  const id = (i: number, j: number, k: number) => {
    const key = i + j * sy + k * sz;
    if (map) {
      let v = map.get(key);
      if (v === undefined) map.set(key, (v = vertexAt(i, j, k)));
      return v;
    }
    let v = lattice[key];
    if (v < 0) lattice[key] = v = vertexAt(i, j, k);
    return v;
  };
  const dims = [nx, ny, nz];
  const c = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) {
    const b = (axis + 1) % 3;
    const d = (axis + 2) % 3;
    for (let side = 0; side < 2; side++) {
      c[axis] = side ? dims[axis] : 0;
      for (let v = 0; v < dims[d]; v++) {
        for (let u = 0; u < dims[b]; u++) {
          c[b] = u;
          c[d] = v;
          const p00 = id(c[0], c[1], c[2]);
          c[b] = u + 1;
          const p10 = id(c[0], c[1], c[2]);
          c[d] = v + 1;
          const p11 = id(c[0], c[1], c[2]);
          c[b] = u;
          const p01 = id(c[0], c[1], c[2]);
          // Alternate the diagonal so rounded corners stay symmetric.
          if ((u + v) & 1) {
            if (side) {
              mb.triangle(p00, p10, p01);
              mb.triangle(p10, p11, p01);
            } else {
              mb.triangle(p00, p01, p10);
              mb.triangle(p10, p01, p11);
            }
          } else if (side) mb.quad(p00, p10, p11, p01);
          else mb.quad(p00, p01, p11, p10);
        }
      }
    }
  }
}

/**
 * Lattice coordinates along one axis of a rounded box with half size h and bevel radius r: `bevel` segments on
 * each rounded edge (evenly spaced in angle once rounded) and `flat` segments between them. With r = h and
 * flat = 0 this is an equal-angle cube sphere.
 */
function boxAxis(h: number, r: number, bevel: number, flat: number) {
  const inner = Math.max(0, h - r);
  const out: number[] = [];
  for (let k = 0; k < bevel; k++) out.push(-(inner + r * Math.tan(((1 - k / bevel) * Math.PI) / 4)));
  if (flat === 0) out.push(0);
  else for (let k = 0; k <= flat; k++) out.push(-inner + (2 * inner * k) / flat);
  for (let k = bevel - 1; k >= 0; k--) out.push(inner + r * Math.tan(((1 - k / bevel) * Math.PI) / 4));
  return out;
}

// ---------------------------------------------------------------------------- stone blocks

export interface StoneStyle {
  /** Bevel segments per rounded edge. */
  bevelSegments: number;
  /** Depth of the irregular erosion along edges (m). */
  wear: number;
  /** Surface undulation and pitting (m). */
  rough: number;
  /** Max chip planes per block. */
  chips: number;
  /** Max chip depth (m). */
  chipDepth: number;
  /** Extra darkening near the bottom of the object (dirt, damp), from object-space y. */
  dirt?: (y: number) => number;
}

export interface BlockSpec {
  center: [number, number, number];
  half: [number, number, number];
  yaw?: number;
  pitch?: number;
  roll?: number;
  bevel: number;
  /** Target lattice spacing (m). */
  step: number;
  seed: number;
  tint: RGB;
}

/** A rounded, weathered stone block with chipped edges and corners. */
export function addBlock(mb: MeshBuilder, b: BlockSpec, noise: Perlin, style: StoneStyle) {
  const rand = mulberry32(b.seed);
  const [hx, hy, hz] = b.half;
  const r = Math.min(b.bevel, hx * 0.9, hy * 0.9, hz * 0.9);
  const nb = style.bevelSegments;
  const flat = (h: number) => Math.max(1, Math.round((2 * (h - r)) / b.step));
  const ax = boxAxis(hx, r, nb, flat(hx));
  const ay = boxAxis(hy, r, nb, flat(hy));
  const az = boxAxis(hz, r, nb, flat(hz));
  const ix = hx - r;
  const iy = hy - r;
  const iz = hz - r;

  // rotation Ry(yaw) * Rx(pitch) * Rz(roll)
  const cyw = Math.cos(b.yaw ?? 0);
  const syw = Math.sin(b.yaw ?? 0);
  const cp = Math.cos(b.pitch ?? 0);
  const sp = Math.sin(b.pitch ?? 0);
  const cr = Math.cos(b.roll ?? 0);
  const sr = Math.sin(b.roll ?? 0);
  const m00 = cyw * cr + syw * sp * sr;
  const m01 = -cyw * sr + syw * sp * cr;
  const m02 = syw * cp;
  const m10 = cp * sr;
  const m11 = cp * cr;
  const m12 = -sp;
  const m20 = -syw * cr + cyw * sp * sr;
  const m21 = syw * sr + cyw * sp * cr;
  const m22 = cyw * cp;

  // Chips: planes cutting corners and edges (a point beyond a plane is projected back onto it).
  const chips: number[] = [];
  const chipCount = Math.floor(rand() * (style.chips + 1));
  const maxDepth = Math.min(style.chipDepth, Math.min(hx, hy, hz) * 0.7);
  for (let k = 0; k < chipCount; k++) {
    const s = [rand() < 0.5 ? -1 : 1, rand() < 0.5 ? -1 : 1, rand() < 0.5 ? -1 : 1];
    const n = [s[0] * (0.4 + rand()), s[1] * (0.4 + rand()), s[2] * (0.4 + rand())];
    const along = Math.floor(rand() * 4); // 0-2: an edge along that axis, 3: a corner
    if (along < 3) n[along] = (rand() - 0.5) * 0.3;
    const l = Math.hypot(n[0], n[1], n[2]);
    for (let i = 0; i < 3; i++) n[i] /= l;
    const support = Math.abs(n[0]) * ix + Math.abs(n[1]) * iy + Math.abs(n[2]) * iz + r;
    // Edge chips sit at a random place along the edge: shift the plane by the offset along that axis.
    const shift = along < 3 ? (rand() - 0.5) * 2 * b.half[along] * n[along] : 0;
    chips.push(n[0], n[1], n[2], support - maxDepth * (0.3 + 0.7 * rand()) + shift);
  }

  const ox = rand() * 200;
  const oy = rand() * 200;
  const oz = rand() * 200;
  const c: RGB = [0, 0, 0];
  const mottle = 0.12;

  cubeSurface(mb, ax.length - 1, ay.length - 1, az.length - 1, (i, j, k) => {
    const px = ax[i];
    const py = ay[j];
    const pz = az[k];
    const qx = clamp(px, -ix, ix);
    const qy = clamp(py, -iy, iy);
    const qz = clamp(pz, -iz, iz);
    let nx = px - qx;
    let ny = py - qy;
    let nz = pz - qz;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    nx /= l;
    ny /= l;
    nz /= l;
    let x = qx + nx * r;
    let y = qy + ny * r;
    let z = qz + nz * r;

    const edge = 1 - Math.max(Math.abs(nx), Math.abs(ny), Math.abs(nz));
    const wx = x + ox;
    const wy = y + oy;
    const wz = z + oz;
    let disp = style.rough * (noise.fbm3(wx * 1.3, wy * 1.3, wz * 1.3, 2) * 1.6 + noise.fbm3(wx * 5, wy * 5, wz * 5, 3) * 0.7 + noise.fbm3(wx * 24, wy * 24, wz * 24, 3) * 0.35);
    const pit = smoothstep(0.45, 0.75, noise.noise3(wx * 38, wy * 38, wz * 38));
    disp -= style.rough * 0.8 * pit;
    const erosion = smoothstep(0.02, 0.28, edge) * (0.55 + 0.45 * noise.noise3(wx * 7, wy * 7, wz * 7));
    disp -= style.wear * erosion;
    x += nx * disp;
    y += ny * disp;
    z += nz * disp;

    let fresh = 0;
    for (let q = 0; q < chips.length; q += 4) {
      const o = x * chips[q] + y * chips[q + 1] + z * chips[q + 2] - chips[q + 3];
      if (o > 0) {
        const rough = 0.006 * smoothstep(0, 0.01, o) * (0.5 + 0.5 * noise.noise3(wx * 30, wy * 30, wz * 30));
        x -= chips[q] * (o + rough);
        y -= chips[q + 1] * (o + rough);
        z -= chips[q + 2] * (o + rough);
        fresh = Math.max(fresh, smoothstep(0, 0.006, o));
      }
    }

    const wxp = b.center[0] + m00 * x + m01 * y + m02 * z;
    const wyp = b.center[1] + m10 * x + m11 * y + m12 * z;
    const wzp = b.center[2] + m20 * x + m21 * y + m22 * z;

    c[0] = b.tint[0];
    c[1] = b.tint[1];
    c[2] = b.tint[2];
    scaleRGB(c, 1 + mottle * noise.fbm3(wx * 1.6, wy * 1.6, wz * 1.6, 3));
    scaleRGB(c, 1 - 0.42 * erosion - 0.25 * pit);
    if (fresh > 0) blend(c, PALETTE.limestone, fresh * 0.45);
    const lich = smoothstep(0.35, 0.65, noise.fbm3(wx * 2.2 + 9, wy * 2.2, wz * 2.2, 3)) * (1 - fresh);
    if (lich > 0) blend(c, ny > 0.5 ? PALETTE.moss : PALETTE.lichen, lich * 0.45);
    if (style.dirt) blend(c, PALETTE.dirt, style.dirt(wyp));
    return mb.vertex(wxp, wyp, wzp, c);
  });
}

// ---------------------------------------------------------------------------- walls

export interface WallOptions {
  seed: number;
  length: number;
  height: number;
  thickness: number;
  /** Course height range (m). */
  course: [number, number];
  /** Block length range (m). */
  block: [number, number];
  gap: number;
  bevel: number;
  step: number;
  /** 0: intact top, 1: heavily ruined (broken, stepped top). */
  ruin: number;
  /** Blocks start exactly at the wall end (no running-bond offset); for beams on columns. */
  aligned?: boolean;
  /** Max blocks lying on the ground next to the wall. */
  fallen?: number;
  /** Irregular masonry: blocks sit unevenly (offsets and tilts, m and rad). */
  jitter?: number;
  palette: RGB[];
  style: StoneStyle;
}

/** A wall segment along x, centred on x = 0 and z = 0, from y = 0 up. Every block is its own geometry. */
export function makeWall(o: WallOptions): VirtualMeshSource {
  const rand = mulberry32(o.seed);
  const noise = new Perlin(o.seed);
  const mb = new MeshBuilder(1 << 20, 1 << 21);
  const half = o.length / 2;
  const profile = (x: number) => {
    if (o.ruin <= 0) return o.height + 1;
    const n = 0.5 + 0.75 * noise.fbm3(x / 3.2 + 3.1, 0.37, 0.71, 3);
    return o.height * clamp(1 - o.ruin * (0.15 + 0.85 * n), 0.06, 1);
  };
  const removed: { len: number; ch: number; depth: number; x: number }[] = [];
  // Blocks go through the whole thickness: walls of thin face blocks around a core look the same up close,
  // but their coarse LODs (voxel proxies of a hollow shell) are much worse.
  let seed = o.seed * 7919;
  let y = 0;
  while (y < o.height - 0.05) {
    const ch = Math.min(o.height - y, o.course[0] + rand() * (o.course[1] - o.course[0]));
    let x = o.aligned ? -half : -half - rand() * o.block[1] * 0.8;
    while (x < half - 0.05) {
      let len = o.block[0] + rand() * (o.block[1] - o.block[0]);
      if (half - (x + len) < o.block[0] * 0.4) len = half - x; // no slivers at the end
      const x0 = Math.max(x, -half);
      const x1 = Math.min(x + len, half);
      x += len;
      if (x1 - x0 < 0.12) continue;
      const xc = (x0 + x1) / 2;
      const depth = o.thickness - rand() * 0.03;
      const top = profile(xc);
      const yc = y + ch / 2;
      if (yc + ch * 0.3 * rand() > top || (yc + ch > top && rand() < 0.35)) {
        removed.push({ len: x1 - x0, ch, depth, x: xc });
        continue;
      }
      const settle = o.ruin * clamp(yc / o.height, 0, 1);
      const j = o.jitter ?? 0;
      addBlock(
        mb,
        {
          center: [xc + (rand() - 0.5) * 0.006, yc + (rand() - 0.5) * 0.004, (rand() - 0.5) * (0.012 + j) + (rand() - 0.5) * 0.03 * settle],
          half: [(x1 - x0 - o.gap) / 2, (ch - o.gap) / 2, depth / 2],
          pitch: (rand() - 0.5) * j * 1.2,
          yaw: (rand() - 0.5) * (0.01 + j + 0.04 * settle),
          roll: (rand() - 0.5) * (0.006 + j + 0.02 * settle),
          bevel: o.bevel * (0.7 + 0.6 * rand()),
          step: o.step,
          seed: seed++,
          tint: tintFrom(o.palette, rand),
        },
        noise,
        o.style
      );
    }
    y += ch;
  }

  // Some of the missing blocks lie on the ground next to the wall.
  const fallen = Math.min(o.fallen ?? 0, Math.floor(removed.length * 0.12));
  for (let k = 0; k < fallen; k++) {
    const b = removed[Math.floor(rand() * removed.length)];
    const side = rand() < 0.5 ? -1 : 1;
    const hy = Math.min(b.ch, b.depth) / 2;
    addBlock(
      mb,
      {
        // Close to the foot of the wall: far-off pieces would vanish at coarse LODs and make them jump.
        center: [clamp(b.x + (rand() - 0.5) * 1.5, -half + 0.5, half - 0.5), hy - 0.03, side * (o.thickness / 2 + 0.3 + rand() * 0.5)],
        half: [b.len / 2 - o.gap, hy - o.gap / 2, Math.max(b.ch, b.depth) / 2 - o.gap / 2],
        yaw: (rand() - 0.5) * 1.2,
        pitch: (rand() - 0.5) * 0.15,
        roll: (rand() - 0.5) * 0.15,
        bevel: o.bevel * 1.3,
        step: o.step,
        seed: seed++,
        tint: tintFrom(o.palette, rand),
      },
      noise,
      o.style
    );
  }
  return mb.finish();
}

function tintFrom(palette: RGB[], rand: () => number, spread = 0.14): RGB {
  const a = palette[Math.floor(rand() * palette.length)];
  const k = 1 - spread / 2 + rand() * spread;
  return [a[0] * k * (0.98 + rand() * 0.04), a[1] * k, a[2] * k * (0.96 + rand() * 0.05)];
}

// ---------------------------------------------------------------------------- height-field slabs

/** Separable box blur of a (nx+1) x (nz+1) grid (prefix sums, radius in cells). */
function blurGrid(src: Float32Array, nx: number, nz: number, radius: number) {
  const w = nx + 1;
  const h = nz + 1;
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const line = new Float64Array(Math.max(w, h) + 1);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) line[i + 1] = line[i] + src[j * w + i];
    for (let i = 0; i < w; i++) {
      const a = Math.max(0, i - radius);
      const b = Math.min(w - 1, i + radius);
      tmp[j * w + i] = (line[b + 1] - line[a]) / (b - a + 1);
    }
  }
  for (let i = 0; i < w; i++) {
    for (let j = 0; j < h; j++) line[j + 1] = line[j] + tmp[j * w + i];
    for (let j = 0; j < h; j++) {
      const a = Math.max(0, j - radius);
      const b = Math.min(h - 1, j + radius);
      out[j * w + i] = (line[b + 1] - line[a]) / (b - a + 1);
    }
  }
  return out;
}

/**
 * Closed slab whose top is a height field over [x0, x0 + width] x [z0, z0 + depth] ((nx+1) x (nz+1) samples),
 * with a skirt down to `bottom` and a fan underneath, so it is watertight and simplifies down to a box.
 */
function heightSlab(nx: number, nz: number, x0: number, z0: number, width: number, depth: number, heights: Float32Array, colors: Float32Array, bottom: number, under: RGB) {
  const mb = new MeshBuilder((nx + 1) * (nz + 1) + 4 * (nx + nz) + 8, nx * nz * 2 + 16 * (nx + nz));
  const w = nx + 1;
  const c: RGB = [0, 0, 0];
  for (let j = 0; j <= nz; j++) {
    for (let i = 0; i <= nx; i++) {
      const v = j * w + i;
      c[0] = colors[v * 3];
      c[1] = colors[v * 3 + 1];
      c[2] = colors[v * 3 + 2];
      mb.vertex(x0 + (i * width) / nx, heights[v], z0 + (j * depth) / nz, c);
    }
  }
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const a = j * w + i;
      const b = a + 1;
      const cc = a + w;
      const d = cc + 1;
      if ((i + j) & 1) {
        mb.triangle(a, cc, d);
        mb.triangle(a, d, b);
      } else {
        mb.triangle(a, cc, b);
        mb.triangle(b, cc, d);
      }
    }
  }
  // Border loop: +x along z0, +z along the far x, -x along the far z, -z back to the start.
  const loop: number[] = [];
  for (let i = 0; i < nx; i++) loop.push(i);
  for (let j = 0; j < nz; j++) loop.push(j * w + nx);
  for (let i = nx; i > 0; i--) loop.push(nz * w + i);
  for (let j = nz; j > 0; j--) loop.push(j * w);
  const base = loop.map((v) => mb.vertex(mb.positions[v * 3], bottom, mb.positions[v * 3 + 2], under));
  const center = mb.vertex(x0 + width / 2, bottom, z0 + depth / 2, under);
  for (let k = 0; k < loop.length; k++) {
    const k1 = (k + 1) % loop.length;
    mb.triangle(loop[k], base[k1], base[k]);
    mb.triangle(loop[k], loop[k1], base[k1]);
    mb.triangle(center, base[k], base[k1]);
  }
  return mb.finish();
}

// ---------------------------------------------------------------------------- paving

/**
 * Square paving tile (size x size, top around y = 0.05) where every stone is modelled: rounded, chipped,
 * worn and cracked flagstones in rows, or domed cobbles. The tile border is a joint, so tiles can be laid side
 * by side in any orientation.
 */
export function makePavingTile(kind: 'flagstones' | 'cobbles', seed: number, size = 6, res = 600): VirtualMeshSource {
  const rand = mulberry32(seed);
  const noise = new Perlin(seed);
  const n = res;
  const w = n + 1;
  const cell = size / n;
  const heights = new Float32Array(w * w);
  const stoneOf = new Int32Array(w * w);
  const gap = kind === 'flagstones' ? 0.012 : 0.02;
  const half = size / 2;

  interface Stone {
    x0: number;
    x1: number;
    z0: number;
    z1: number;
    top: number;
    sx: number;
    sz: number;
    tint: RGB;
    crack: number[] | null;
  }
  const stones: Stone[] = [];
  // Close tints: coarse LODs interpolate between a few stones' colours over large triangles.
  const palette = [PALETTE.limestone, PALETTE.travertine];
  const newStone = (x0: number, x1: number, z0: number, z1: number): Stone => ({
    x0,
    x1,
    z0,
    z1,
    top: (kind === 'flagstones' ? 0.06 : 0.07) + (rand() - 0.5) * 0.02,
    sx: (rand() - 0.5) * 0.02,
    sz: (rand() - 0.5) * 0.02,
    tint: tintFrom(palette, rand, 0.08),
    crack: rand() < 0.3 ? [x0 + rand() * (x1 - x0), z0 + rand() * (z1 - z0), rand() * Math.PI] : null,
  });

  if (kind === 'flagstones') {
    // Rows of random depth, stones of random length in each row (running bond).
    const rows: { z0: number; z1: number; cuts: number[]; ids: number[] }[] = [];
    let z = -half;
    while (z < half - 0.01) {
      let dz = 0.45 + rand() * 0.5;
      if (half - (z + dz) < 0.4) dz = half - z;
      const row = { z0: z, z1: z + dz, cuts: [] as number[], ids: [] as number[] };
      let x = -half;
      while (x < half - 0.01) {
        let dx = 0.55 + rand() * 0.8;
        if (half - (x + dx) < 0.4) dx = half - x;
        row.cuts.push(x + dx);
        row.ids.push(stones.length);
        stones.push(newStone(x + gap / 2, x + dx - gap / 2, z + gap / 2, z + dz - gap / 2));
        x += dx;
      }
      rows.push(row);
      z += dz;
    }
    let r = 0;
    for (let j = 0; j <= n; j++) {
      const pz = -half + j * cell;
      while (r < rows.length - 1 && pz > rows[r].z1) r++;
      const row = rows[r];
      let s = 0;
      for (let i = 0; i <= n; i++) {
        const px = -half + i * cell;
        while (s < row.cuts.length - 1 && px > row.cuts[s]) s++;
        stoneOf[j * w + i] = row.ids[s];
      }
    }
  } else {
    // Cobbles: jittered grid of feature points, each cell one stone (Voronoi).
    const cells = Math.round(size / 0.17);
    const cs = size / cells;
    const fx = new Float32Array(cells * cells);
    const fz = new Float32Array(cells * cells);
    for (let j = 0; j < cells; j++) {
      for (let i = 0; i < cells; i++) {
        fx[j * cells + i] = -half + (i + 0.2 + rand() * 0.6) * cs;
        fz[j * cells + i] = -half + (j + 0.2 + rand() * 0.6) * cs;
        stones.push(newStone(fx[j * cells + i] - cs / 2, fx[j * cells + i] + cs / 2, fz[j * cells + i] - cs / 2, fz[j * cells + i] + cs / 2));
      }
    }
    // Distance to the cell border ((d2 - d1) / 2 approximates the distance to the bisector).
    for (let j = 0; j <= n; j++) {
      const pz = -half + j * cell;
      const cj = Math.min(cells - 1, Math.floor((pz + half) / cs));
      for (let i = 0; i <= n; i++) {
        const px = -half + i * cell;
        const ci = Math.min(cells - 1, Math.floor((px + half) / cs));
        let d1 = Infinity;
        let d2 = Infinity;
        let best = 0;
        for (let b = Math.max(0, cj - 1); b <= Math.min(cells - 1, cj + 1); b++) {
          for (let a = Math.max(0, ci - 1); a <= Math.min(cells - 1, ci + 1); a++) {
            const k = b * cells + a;
            const d = Math.hypot(px - fx[k], pz - fz[k]);
            if (d < d1) {
              d2 = d1;
              d1 = d;
              best = k;
            } else if (d < d2) d2 = d;
          }
        }
        stoneOf[j * w + i] = best;
        heights[j * w + i] = (d2 - d1) / 2; // temporarily: distance to the stone border
      }
    }
  }

  // Heights.
  const bevel = kind === 'flagstones' ? 0.03 : 0.045;
  for (let j = 0; j <= n; j++) {
    const pz = -half + j * cell;
    for (let i = 0; i <= n; i++) {
      const px = -half + i * cell;
      const v = j * w + i;
      const id = stoneOf[v];
      const st = stones[id];
      const cx = (st.x0 + st.x1) / 2;
      const cz = (st.z0 + st.z1) / 2;
      let d: number;
      if (kind === 'flagstones') d = Math.min(px - st.x0, st.x1 - px, pz - st.z0, st.z1 - pz);
      else d = heights[v] - gap / 2;
      d = Math.min(d, half - Math.abs(px) - gap / 2, half - Math.abs(pz) - gap / 2);
      // irregular outline and chipped edges
      d += 0.012 * noise.noise3(px * 7, pz * 7, id * 0.37) + 0.004 * noise.noise3(px * 31, pz * 31, id);
      d -= 0.06 * smoothstep(0.5, 0.85, noise.noise3(px * 3.1, pz * 3.1, id * 1.71 + 5));
      const ground = 0.004 + 0.006 * noise.fbm3(px * 4, pz * 4, 0.5, 3);
      if (d <= 0) {
        heights[v] = ground;
        continue;
      }
      const t = Math.min(1, d / bevel);
      const round = Math.sqrt(1 - (1 - t) * (1 - t));
      let top = st.top + st.sx * (px - cx) + st.sz * (pz - cz);
      if (kind === 'flagstones') {
        const rx = (px - cx) / ((st.x1 - st.x0) / 2);
        const rz = (pz - cz) / ((st.z1 - st.z0) / 2);
        top -= 0.006 * Math.max(0, 1 - rx * rx - rz * rz); // worn hollow
        top += 0.003 * noise.fbm3(px * 6 + id, pz * 6, 1.5, 3) + 0.0012 * noise.fbm3(px * 40, pz * 40, id, 2);
      } else {
        const r2 = ((px - cx) ** 2 + (pz - cz) ** 2) / (0.1 * 0.1);
        top -= 0.025 * Math.min(1, r2) * 0.6; // domed
        top += 0.004 * noise.fbm3(px * 9 + id, pz * 9, 2.5, 3) + 0.0012 * noise.fbm3(px * 45, pz * 45, id, 2);
      }
      if (st.crack) {
        const [ax, az, ang] = st.crack;
        const dl = Math.abs((px - ax) * Math.sin(ang) - (pz - az) * Math.cos(ang) + 0.025 * noise.fbm3(px * 5, pz * 5, id, 2));
        top -= 0.012 * Math.exp(-(dl * dl) / (0.006 * 0.006));
      }
      heights[v] = ground + (top - ground) * round;
    }
  }

  // Colors: stone tint, mottling, dirt and moss in joints and hollows (from a blurred copy of the heights).
  const blur = blurGrid(heights, n, n, 5);
  const colors = new Float32Array(w * w * 3);
  const c: RGB = [0, 0, 0];
  const earth: RGB = [0, 0, 0];
  for (let j = 0; j <= n; j++) {
    const pz = -half + j * cell;
    for (let i = 0; i <= n; i++) {
      const px = -half + i * cell;
      const v = j * w + i;
      const st = stones[stoneOf[v]];
      const h = heights[v];
      const raised = smoothstep(0.012, 0.03, h);
      c[0] = st.tint[0];
      c[1] = st.tint[1];
      c[2] = st.tint[2];
      scaleRGB(c, 1 + 0.07 * noise.fbm3(px * 2.3, pz * 2.3, stoneOf[v] * 0.1, 3) + 0.04 * noise.noise3(px * 25, pz * 25, 0.5));
      const cavity = clamp((blur[v] - h) * 45, 0, 1);
      scaleRGB(c, 1 - 0.06 * cavity);
      // Joints: sandy soil only a little darker than the stone; the geometry itself draws the joint. Coarse
      // LODs keep mostly edge and joint vertices, so strong contrast here would blotch the paving from afar.
      earth[0] = st.tint[0] * 0.85;
      earth[1] = st.tint[1] * 0.85;
      earth[2] = st.tint[2] * 0.8;
      blend(earth, PALETTE.moss, smoothstep(0, 0.5, noise.fbm3(px * 1.3, pz * 1.3, 7.7, 3)) * 0.15);
      blend(c, earth, 1 - raised);
      const lich = smoothstep(0.3, 0.6, noise.fbm3(px * 2.5 + 3, pz * 2.5, 4.4, 3)) * raised;
      blend(c, PALETTE.lichen, lich * 0.08);
      colors.set(c, v * 3);
    }
  }
  // Shallow skirt in stone colour: coarse LODs fold the tile border onto it, which must not show from afar.
  const under: RGB = [PALETTE.limestone[0] * 0.95, PALETTE.limestone[1] * 0.95, PALETTE.limestone[2] * 0.93];
  return heightSlab(n, n, -half, -half, size, size, heights, colors, -0.03, under);
}

// ---------------------------------------------------------------------------- terrain

/** Height-field terrain over a square (closed underneath), heights from `heightAt`. Colors are left neutral. */
export function makeGroundTerrain(size: number, res: number, heightAt: (x: number, z: number) => number): VirtualMeshSource {
  const w = res + 1;
  const heights = new Float32Array(w * w);
  for (let j = 0; j <= res; j++) for (let i = 0; i <= res; i++) heights[j * w + i] = heightAt(-size / 2 + (i * size) / res, -size / 2 + (j * size) / res);
  const mesh = heightSlab(res, res, -size / 2, -size / 2, size, size, heights, new Float32Array(w * w * 3).fill(1), -50, [1, 1, 1]);
  mesh.colors = undefined;
  return mesh;
}

// ---------------------------------------------------------------------------- rocks

export interface RockOptions {
  /** Lattice segments per cube face edge: 6 * detail^2 * 2 triangles. */
  detail: number;
  stretch: [number, number, number];
  /** Number of strata layers over the height, and how far the ledges stand out. */
  strata: number;
  strataDepth: number;
  cracks: number;
  /** Large-scale irregularity of the silhouette. */
  lumpy: number;
  tint: RGB;
}

/** Boulder or cliff rock: displaced cube sphere with strata ledges, deep crevices and fine pitting. */
export function makeRock(seed: number, o: RockOptions): VirtualMeshSource {
  const noise = new Perlin(seed);
  const rand = mulberry32(seed);
  const n = o.detail;
  const axis = boxAxis(1, 1, n / 2, 0);
  const mb = new MeshBuilder(6 * (n + 1) * (n + 1), 12 * n * n);
  const crackOf = new Float32Array(6 * (n + 1) * (n + 1));
  const layerOf = new Float32Array(crackOf.length);
  const tiltX = (rand() - 0.5) * 0.5;
  const tiltZ = (rand() - 0.5) * 0.5;
  const c: RGB = [0, 0, 0];
  const floor = -0.45 * o.stretch[1];
  cubeSurface(mb, n, n, n, (i, j, k) => {
    let dx = axis[i];
    let dy = axis[j];
    let dz = axis[k];
    const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dx /= l;
    dy /= l;
    dz /= l;
    const warp = 0.35 * noise.fbm3(dx * 1.1 + 9, dy * 1.1, dz * 1.1, 2);
    let r = 1 + o.lumpy * noise.fbm3(dx * 0.8 + 4 + warp, dy * 0.8 + warp, dz * 0.8 - warp, 4);
    // strata: tilted layers, each one bulging at its base and stepping back at its top
    const sy = dy + dx * tiltX + dz * tiltZ + 0.06 * noise.fbm3(dx * 2, dy * 2, dz * 2 + 7, 3);
    const layerPos = (sy + 2) * o.strata + 0.6 * noise.noise3(dx * 3, dy * 1.5, dz * 3);
    const layer = Math.floor(layerPos);
    const f = layerPos - layer;
    const hard = 0.5 + fract(Math.sin(layer * 127.1 + seed) * 43758.5453);
    r += o.strataDepth * hard * (smoothstep(0, 0.88, f) - smoothstep(0.88, 1, f) - 0.5) * -1;
    // crevices
    // mostly vertical joints (stretched noise), plus a finer, patchy network
    const c1 = Math.abs(noise.noise3(dx * 2.4 + 11, dy * 0.7, dz * 2.4));
    const c2 = Math.abs(noise.noise3(dx * 5.5, dy * 5.5 + 3, dz * 5.5));
    const crack = (1 - smoothstep(0, 0.05, c1)) * o.cracks + (1 - smoothstep(0, 0.03, c2)) * o.cracks * 0.15 * smoothstep(0.1, 0.4, noise.noise3(dx * 2, dy * 2 + 5, dz * 2));
    r -= crack;
    r += 0.014 * noise.fbm3(dx * 14, dy * 14, dz * 14, 4) + 0.006 * (1 - Math.abs(noise.noise3(dx * 45, dy * 45, dz * 45)));
    const x = dx * r * o.stretch[0];
    let y = dy * r * o.stretch[1];
    const z = dz * r * o.stretch[2];
    if (y < floor) y = floor + (y - floor) * 0.12;
    const v = mb.vertex(x, y, z, o.tint);
    crackOf[v] = crack / Math.max(1e-6, o.cracks);
    layerOf[v] = layer;
    return v;
  });
  const mesh = mb.finish();
  recolor(mesh, (i, p, nrm, out) => {
    const x = p[i * 3];
    const y = p[i * 3 + 1];
    const z = p[i * 3 + 2];
    out[0] = o.tint[0];
    out[1] = o.tint[1];
    out[2] = o.tint[2];
    const band = fract(Math.sin(layerOf[i] * 78.233 + seed) * 12345.678);
    scaleRGB(out, 0.8 + 0.25 * band);
    if (band > 0.8) blend(out, PALETTE.travertine, 0.25);
    scaleRGB(out, 1 + 0.15 * noise.fbm3(x * 2, y * 2, z * 2, 3));
    scaleRGB(out, 1 - 0.65 * clamp(crackOf[i], 0, 1));
    // darker, weathered faces where the strata ledges overhang
    scaleRGB(out, 0.75 + 0.25 * smoothstep(-0.6, 0.4, nrm[i * 3 + 1]));
    const up = nrm[i * 3 + 1];
    const moss = smoothstep(0.55, 0.85, up) * smoothstep(-0.15, 0.35, noise.fbm3(x * 1.7, y * 1.7, z * 1.7, 3));
    blend(out, PALETTE.moss, moss * 0.75);
    const lich = smoothstep(0.4, 0.65, noise.fbm3(x * 4 + 2, y * 4, z * 4, 3));
    blend(out, PALETTE.lichen, lich * 0.25);
    blend(out, PALETTE.dirt, smoothstep(floor + 0.25, floor, y) * 0.6);
  });
  return mesh;
}

// ---------------------------------------------------------------------------- lathe

interface ProfileRow {
  r: number;
  y: number;
  part: number;
  /** Position within its part, 0..1. */
  t: number;
  /** Outward profile normal (radial, vertical). */
  nr: number;
  ny: number;
}

/** Profile of a surface of revolution, bottom to top (r >= 0, r = 0 rows become poles). */
class Profile {
  readonly rows: ProfileRow[] = [];

  point(r: number, y: number, part: number, t = 0) {
    this.rows.push({ r, y, part, t, nr: 0, ny: 0 });
    return this;
  }

  /** Straight segment to (r1, y1) in `n` steps (the start point is the current last row). */
  line(r1: number, y1: number, n: number, part: number) {
    const last = this.rows[this.rows.length - 1];
    for (let k = 1; k <= n; k++) this.point(mix(last.r, r1, k / n), mix(last.y, y1, k / n), part, k / n);
    return this;
  }

  /** Arc around (cr, cy) from angle a0 to a1 (radians, 0 = outward), excluding its first point. */
  arc(cr: number, cy: number, radius: number, a0: number, a1: number, n: number, part: number) {
    for (let k = 1; k <= n; k++) {
      const a = mix(a0, a1, k / n);
      this.point(cr + radius * Math.cos(a), cy + radius * Math.sin(a), part, k / n);
    }
    return this;
  }

  /** Rows of `part` spread over [y0, y1] at constant radius (t = 0..1); the caller reshapes them. */
  span(r: number, y0: number, y1: number, n: number, part: number, includeFirst = false) {
    for (let k = includeFirst ? 0 : 1; k <= n; k++) this.point(r, mix(y0, y1, k / n), part, k / n);
    return this;
  }

  finish() {
    const rows = this.rows;
    for (let v = 0; v < rows.length; v++) {
      // Neighbours within the same part (parts reshaped by the caller would give bogus tangents across).
      const prev = rows[Math.max(0, v - 1)];
      const next = rows[Math.min(rows.length - 1, v + 1)];
      let a = prev.part === rows[v].part ? prev : rows[v];
      let b = next.part === rows[v].part ? next : rows[v];
      if (a === b) {
        a = prev;
        b = next;
      }
      const dr = b.r - a.r;
      const dy = b.y - a.y;
      const l = Math.hypot(dr, dy) || 1;
      rows[v].nr = dy / l;
      rows[v].ny = -dr / l;
    }
    return rows;
  }
}

/**
 * Surface of revolution: `segments` around, one ring per profile row. Rows with r = 0 at either end become a
 * single pole vertex. `vertex(row, rowIndex, theta)` adds the vertex (it may reshape it freely) and returns its index.
 */
function lathe(mb: MeshBuilder, rows: ProfileRow[], segments: number, vertex: (row: ProfileRow, v: number, theta: number) => number) {
  const rings: number[][] = [];
  for (let v = 0; v < rows.length; v++) {
    const pole = rows[v].r === 0 && (v === 0 || v === rows.length - 1);
    if (pole) {
      const p = vertex(rows[v], v, 0);
      rings.push(new Array(segments).fill(p));
    } else {
      const ring: number[] = [];
      for (let u = 0; u < segments; u++) ring.push(vertex(rows[v], v, (u / segments) * TAU));
      rings.push(ring);
    }
  }
  for (let v = 0; v < rows.length - 1; v++) {
    const r0 = rings[v];
    const r1 = rings[v + 1];
    for (let u = 0; u < segments; u++) {
      const u1 = (u + 1) % segments;
      const a = r0[u];
      const b = r0[u1];
      const c = r1[u1];
      const d = r1[u];
      if (a !== b) mb.triangle(a, d, c);
      if (c !== d) mb.triangle(a, c, b);
    }
  }
}

// ---------------------------------------------------------------------------- columns

const enum Part {
  Base,
  Shaft,
  Bead,
  Neck,
  Echinus,
  Top,
  FractureTop,
  FractureBottom,
  Flat,
  ScrollFront,
  ScrollSide,
  ScrollBack,
  Band,
}

export type ColumnKind = 'full' | 'broken' | 'drum' | 'capital';

/**
 * Ionic column, ~9 m: square plinth, moulded base, shaft with 24 flutes and entasis, bead-and-reel astragal,
 * egg-and-dart echinus, volute scrolls with spiral faces and an abacus. All weathered: eroded arrises, chips,
 * pitting, lichen and dirt. 'broken' ends in a rough fracture, 'drum' is a fallen shaft piece with fractures at
 * both ends (lying along y, set it on its side), 'capital' is a fallen capital on its own.
 */
export function makeColumn(seed: number, kind: ColumnKind): VirtualMeshSource {
  const rand = mulberry32(seed);
  const noise = new Perlin(seed);
  const mb = new MeshBuilder(1 << 19, 1 << 20);
  const FLUTES = 24;
  const SEG = FLUTES * 16;
  const plinthH = 0.26;
  const y0 = plinthH;
  const R0 = 0.55;
  const R1 = 0.47;
  const ys0 = y0 + 0.47;
  const shaftLen = 7.4;
  const ys1 = ys0 + shaftLen;
  const fluteDepth = 0.032;
  const tint = tintFrom([PALETTE.marble, PALETTE.marble, PALETTE.limestone], rand);
  const style: StoneStyle = { bevelSegments: 3, wear: 0.012, rough: 0.003, chips: 3, chipDepth: 0.06, dirt: (y) => smoothstep(1.4, 0, y) * 0.35 };

  // Fracture surfaces: rough, tilted, a few big conchoidal steps.
  const breakY = kind === 'broken' ? ys0 + shaftLen * (0.2 + rand() * 0.45) : kind === 'drum' ? 1.4 + rand() * 1.2 : 0;
  const tilt = [(rand() - 0.5) * 0.5, (rand() - 0.5) * 0.5];
  const fracture = (x: number, z: number, base: number, sign: number) =>
    base +
    sign * (0.12 * noise.fbm3(x * 1.6, z * 1.6, base, 4) + tilt[0] * x + tilt[1] * z + 0.05 * (1 - Math.abs(noise.noise3(x * 4, z * 4, base + 3))) + 0.012 * noise.fbm3(x * 18, z * 18, base, 3));
  const bottomY = 0.0;

  // Shaft radius at height y (entasis) and flute profile around.
  const shaftR = (y: number) => {
    const t = clamp((y - ys0) / shaftLen, 0, 1);
    return R0 - (R0 - R1) * Math.pow(t, 1.6);
  };
  const fluteAt = (theta: number) => {
    const phi = fract((theta * FLUTES) / TAU);
    const w = 0.8;
    if (phi >= w) return 0;
    const s = (phi - w / 2) / (w / 2);
    return Math.sqrt(1 - s * s);
  };
  // Rounded flute ends near the base and the capital (only where the shaft is original, not at fractures).
  const fluteEnds = (y: number) => {
    const e = 0.11;
    const a = clamp((y - ys0) / e, 0, 1);
    const b = clamp((ys1 - y) / e, 0, 1);
    return Math.sqrt(1 - (1 - a) * (1 - a)) * Math.sqrt(1 - (1 - b) * (1 - b));
  };

  const p = new Profile();
  const hasBase = kind === 'full' || kind === 'broken';
  if (hasBase) {
    p.point(0, y0, Part.Base)
      .line(0.7, y0, 10, Part.Base)
      .arc(0.7, y0 + 0.075, 0.075, -Math.PI / 2, Math.PI / 2, 24, Part.Base)
      .line(0.665, y0 + 0.15, 2, Part.Base)
      .line(0.665, y0 + 0.17, 2, Part.Base)
      .arc(0.665, y0 + 0.235, 0.065, -Math.PI / 2, -Math.PI * 1.5, 20, Part.Base)
      .line(0.645, y0 + 0.3, 2, Part.Base)
      .arc(0.645, y0 + 0.35, 0.05, -Math.PI / 2, Math.PI / 2, 18, Part.Base)
      .line(0.6, y0 + 0.4, 3, Part.Base)
      .arc(0.6, ys0, 0.05, 0, Math.PI / 2, 8, Part.Base);
    // ^ apophyge: concave sweep into the shaft (ends at r = 0.55 = R0)
  }
  if (kind === 'full') {
    p.span(R0, ys0, ys1, 360, Part.Shaft);
    p.arc(R1 + 0.035, ys1, 0.035, Math.PI, Math.PI / 2, 6, Part.Neck)
      .arc(R1 + 0.035, ys1 + 0.075, 0.04, -Math.PI / 2, Math.PI / 2, 22, Part.Bead)
      .line(R1 + 0.01, ys1 + 0.115, 2, Part.Neck)
      .line(R1 + 0.01, ys1 + 0.17, 4, Part.Neck);
    const ye = ys1 + 0.17;
    for (let k = 1; k <= 40; k++) {
      const a = (k / 40) * (Math.PI / 2);
      p.point(R1 + 0.01 + 0.19 * Math.sin(a), ye + 0.2 * (1 - Math.cos(a)), Part.Echinus, k / 40);
    }
    p.line(0, ye + 0.2, 10, Part.Top);
  } else if (kind === 'broken') {
    p.span(R0, ys0, breakY, 260, Part.Shaft);
    p.span(0, 0, 0, 40, Part.FractureTop);
  } else if (kind === 'drum') {
    const len = breakY;
    p.point(0, bottomY, Part.FractureBottom, 1);
    for (let k = 39; k >= 1; k--) p.point(R0 * (1 - k / 40), bottomY, Part.FractureBottom, k / 40);
    p.span(R0, bottomY, len, Math.round(len * 48), Part.Shaft, true);
    p.span(0, 0, 0, 40, Part.FractureTop);
  } else {
    // capital on its own: neck from a flat bottom (drawn as fallen)
    p.point(0, 0, Part.Flat).line(R1 + 0.01, 0, 12, Part.Flat).line(R1 + 0.01, 0.06, 3, Part.Neck);
    p.arc(R1 + 0.035, 0.1, 0.04, -Math.PI / 2, Math.PI / 2, 22, Part.Bead).line(R1 + 0.01, 0.14, 2, Part.Neck).line(R1 + 0.01, 0.2, 4, Part.Neck);
    const ye = 0.2;
    for (let k = 1; k <= 40; k++) {
      const a = (k / 40) * (Math.PI / 2);
      p.point(R1 + 0.01 + 0.19 * Math.sin(a), ye + 0.2 * (1 - Math.cos(a)), Part.Echinus, k / 40);
    }
    p.line(0, ye + 0.2, 10, Part.Top);
  }
  const rows = p.finish();

  // Shaft ends (per theta) for fractured pieces, sampled at the shaft surface.
  const shaftTop = (theta: number) => {
    const r = shaftR(breakY);
    return fracture(r * Math.cos(theta), r * Math.sin(theta), breakY, 1);
  };
  const shaftBottom = (theta: number) => fracture(R0 * Math.cos(theta), R0 * Math.sin(theta), bottomY, -1);
  const fluted = (theta: number, y: number, ends: boolean) => shaftR(y) - fluteDepth * fluteAt(theta) * (ends ? fluteEnds(y) : 1);

  // A few deep chips (bites) out of the shaft and mouldings.
  const bites: number[] = [];
  const bitesN = 4 + Math.floor(rand() * 6);
  const topY = kind === 'full' ? ys1 + 0.6 : kind === 'broken' ? breakY : kind === 'drum' ? breakY : 0.5;
  for (let k = 0; k < bitesN; k++) {
    const th = rand() * TAU;
    const y = (hasBase ? y0 : 0) + rand() * (topY - (hasBase ? y0 : 0));
    const rr = kind === 'capital' ? 0.55 : shaftR(y);
    bites.push(rr * Math.cos(th), y, rr * Math.sin(th), 0.04 + rand() * 0.1);
  }

  const c: RGB = [0, 0, 0];
  const vertex = (row: ProfileRow, _v: number, theta: number) => {
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    let r = row.r;
    let y = row.y;
    let dn = 0;
    let cavity = 0;
    let fresh = 0;
    let weather = 1;
    switch (row.part) {
      case Part.Shaft: {
        if (kind === 'broken') y = mix(ys0, shaftTop(theta), row.t);
        else if (kind === 'drum') y = mix(shaftBottom(theta), shaftTop(theta), row.t);
        const flute = fluteAt(theta) * (kind === 'full' || (kind === 'broken' && row.t < 0.5) ? fluteEnds(y) : 1);
        r = shaftR(kind === 'drum' ? ys0 + 2 + y : y) - fluteDepth * flute;
        cavity = flute * 0.8;
        // eroded arrises: chips along the fillets between flutes
        const arris = 1 - flute;
        dn -= 0.008 * arris * smoothstep(0.5, 0.85, noise.noise3(ct * 3, y * 3.5, st * 3));
        break;
      }
      case Part.FractureTop:
      case Part.FractureBottom: {
        const top = row.part === Part.FractureTop;
        const ref = top ? breakY : bottomY;
        const yAt = kind === 'drum' ? ys0 + 2 + ref : ref;
        const edge = shaftR(kind === 'drum' ? yAt : breakY) - fluteDepth * fluteAt(theta);
        r = edge * (1 - row.t);
        const x = r * ct;
        const z = r * st;
        y = fracture(x, z, ref, top ? 1 : -1);
        fresh = 1;
        weather = 0;
        break;
      }
      case Part.Echinus: {
        // egg-and-dart, 16 eggs around
        const s = (theta * 16) / TAU;
        const u = fract(s) - 0.5;
        const v = row.t - 0.5;
        const e = (u / 0.33) ** 2 + (v / 0.4) ** 2;
        if (e < 1) dn += 0.02 * Math.sqrt(1 - e);
        else if (e < 1.25) {
          dn -= 0.008;
          cavity = 0.8;
        } else if (e < 1.9 && v > -0.48) dn += 0.012 * Math.sin(((e - 1.25) / 0.65) * Math.PI);
        else {
          cavity = 0.5;
          dn -= 0.004;
        }
        const dart = Math.abs(Math.abs(u) - 0.5);
        if (dart < 0.05 * (0.3 + row.t) && v > -0.45) dn += 0.012;
        break;
      }
      case Part.Bead: {
        // bead and reel, 44 beads around
        const s = fract((theta * 44) / TAU) - 0.5;
        const bead = Math.abs(s) < 0.32 ? Math.sqrt(1 - (s / 0.32) ** 2) : 0;
        const reel = Math.abs(Math.abs(s) - 0.42) < 0.035 ? 0.7 : 0;
        dn -= 0.016 * (1 - Math.max(bead, reel));
        cavity = (1 - Math.max(bead, reel)) * 0.7;
        break;
      }
    }

    let x = r * ct;
    let z = r * st;
    if (weather > 0) {
      dn += 0.004 * noise.fbm3(x * 3, y * 3, z * 3, 3) + 0.0018 * noise.fbm3(x * 20, y * 20, z * 20, 3);
      dn -= 0.003 * smoothstep(0.45, 0.75, noise.noise3(x * 45, y * 45, z * 45));
      for (let k = 0; k < bites.length; k += 4) {
        const ddx = x - bites[k];
        const ddy = y - bites[k + 1];
        const ddz = z - bites[k + 2];
        const d = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
        const rb = bites[k + 3];
        if (d < rb) {
          const depth = (rb - d) * 0.7 * (0.8 + 0.4 * noise.noise3(x * 25, y * 25, z * 25));
          dn -= depth;
          fresh = Math.max(fresh, smoothstep(0, 0.01, depth));
        }
      }
    }
    x += dn * row.nr * ct;
    z += dn * row.nr * st;
    y += dn * row.ny;

    c[0] = tint[0];
    c[1] = tint[1];
    c[2] = tint[2];
    // marble veins, mottling, flute shadows, dirt, black crust streaks and lichen
    const vein = Math.abs(Math.sin(x * 2.1 + y * 0.9 + z * 1.7 + 5 * noise.fbm3(x * 0.7, y * 0.5, z * 0.7, 3)));
    scaleRGB(c, 1 - 0.18 * (1 - smoothstep(0, 0.1, vein)));
    scaleRGB(c, 1 + 0.08 * noise.fbm3(x * 4, y * 4, z * 4, 2));
    scaleRGB(c, 1 - 0.28 * cavity);
    if (fresh > 0) blend(c, PALETTE.marble, fresh * 0.35);
    const streak = smoothstep(0.15, 0.55, noise.fbm3(x * 5, y * 0.35, z * 5, 3)) * (1 - fresh);
    scaleRGB(c, 1 - 0.3 * streak);
    const lich = smoothstep(0.38, 0.62, noise.fbm3(x * 2.4 + 5, y * 2.4, z * 2.4, 3)) * (1 - fresh);
    blend(c, PALETTE.lichen, lich * 0.4);
    blend(c, PALETTE.dirt, (kind === 'full' || kind === 'broken' ? smoothstep(1.6, 0.2, y) : 0.15) * 0.4);
    return mb.vertex(x, y, z, c);
  };
  lathe(mb, rows, SEG, vertex);

  if (hasBase) {
    addBlock(mb, { center: [0, plinthH / 2, 0], half: [0.76, plinthH / 2, 0.76], bevel: 0.03, step: 0.035, seed: seed * 13 + 1, tint }, noise, style);
  }
  if (kind === 'full' || kind === 'capital') {
    const ye = kind === 'full' ? ys1 + 0.37 : 0.4;
    addCapitalTop(mb, ye, seed, noise, tint, style);
  }
  return mb.finish();
}

/** Volute scrolls, cushion and abacus of the Ionic capital, above echinus height `ye`. */
function addCapitalTop(mb: MeshBuilder, ye: number, seed: number, noise: Perlin, tint: RGB, style: StoneStyle) {
  // cushion (canalis) between the scrolls, and the abacus on top
  addBlock(mb, { center: [0, ye + 0.08, 0], half: [0.66, 0.09, 0.5], bevel: 0.04, step: 0.03, seed: seed * 13 + 2, tint }, noise, { ...style, chips: 1 });
  addBlock(mb, { center: [0, ye + 0.2, 0], half: [0.7, 0.045, 0.7], bevel: 0.025, step: 0.03, seed: seed * 13 + 3, tint }, noise, style);

  // Two scrolls: lathes around the z axis (front to back), spiral faces at both ends, waisted in the middle.
  const R = 0.22;
  const L = 1.0;
  const eye = 0.17;
  const turns = 2.6;
  const b = Math.log(1 / eye) / (TAU * turns);
  const p = new Profile();
  p.point(0, -L / 2, Part.ScrollFront, 1);
  for (let k = 69; k >= 1; k--) p.point(R * (1 - k / 70), -L / 2, Part.ScrollFront, k / 70);
  p.span(R, -L / 2, L / 2, 60, Part.ScrollSide, true);
  for (let k = 1; k <= 69; k++) p.point(R * (1 - k / 70), L / 2, Part.ScrollBack, k / 70);
  p.point(0, L / 2, Part.ScrollBack, 1);
  const rows = p.finish();
  const c: RGB = [0, 0, 0];
  for (const side of [-1, 1]) {
    const cx = side * 0.6;
    const cy = ye - 0.02;
    lathe(mb, rows, 200, (row, _v, theta) => {
      const ct = Math.cos(theta);
      const st = Math.sin(theta);
      let r = row.r;
      let axial = row.y;
      let cavity = 0;
      if (row.part === Part.ScrollSide) {
        // waist (balteus) with mouldings, and fine flutes along the bolster
        const s = row.y / (L / 2);
        r = R * (1 - 0.16 * Math.exp(-((s / 0.28) ** 2))) + 0.008 * Math.exp(-(((Math.abs(s) - 0.3) / 0.04) ** 2));
        const fl = 0.5 + 0.5 * Math.cos(theta * 28);
        r -= 0.006 * fl * (1 - Math.exp(-((s / 0.35) ** 2))) * smoothstep(1, 0.85, Math.abs(s));
        cavity = fl * 0.3;
      } else {
        // spiral face: a log-spiral channel winding into a raised eye
        const rho = Math.max(1e-4, r / R);
        let relief = 0;
        if (rho < eye) relief = 0.028 * Math.sqrt(Math.max(0, 1 - (rho / eye) ** 2)) + 0.004;
        else {
          const ang = side * (row.part === Part.ScrollFront ? 1 : -1) * theta;
          const f = fract((Math.log(rho) / b - ang) / TAU);
          const ridge = f < 0.16 ? Math.sin((f / 0.16) * Math.PI) : 0;
          const channel = f >= 0.16 ? Math.sin(((f - 0.16) / 0.84) * Math.PI) : 0;
          relief = 0.006 * ridge - 0.022 * channel * smoothstep(eye, eye * 1.6, rho);
          cavity = channel * 0.7;
        }
        relief *= smoothstep(1, 0.9, rho);
        axial += (row.part === Part.ScrollFront ? -1 : 1) * relief;
      }
      let x = r * ct;
      let yy = r * st;
      const wear = 0.002 * noise.fbm3(x * 25 + cx, yy * 25, axial * 25, 3);
      x += wear * ct;
      yy += wear * st;
      c[0] = tint[0];
      c[1] = tint[1];
      c[2] = tint[2];
      scaleRGB(c, (1 - 0.3 * cavity) * (1 + 0.08 * noise.fbm3(x * 4 + cx, yy * 4, axial * 4, 2)));
      const lich = smoothstep(0.38, 0.62, noise.fbm3(x * 2.4 + cx, yy * 2.4 + cy, axial * 2.4, 3));
      blend(c, PALETTE.lichen, lich * 0.4);
      // local (x, axial, y) -> world: rotation about x by +90 degrees keeps the winding
      return mb.vertex(cx + x, cy - yy, axial, c);
    });
  }
}

// ---------------------------------------------------------------------------- relief panel

const MEANDER = ['111110', '100010', '101010', '101110', '111111'];

/** Height of a raised Greek-key ribbon at (u, v) in cell units (v = 0 at the band bottom). */
function meander(u: number, v: number) {
  const rowsN = MEANDER.length;
  const period = MEANDER[0].length;
  if (v < 0 || v >= rowsN) return 0;
  const on = (i: number, j: number) => j >= 0 && j < rowsN && MEANDER[rowsN - 1 - j][((i % period) + period) % period] === '1';
  const i = Math.floor(u);
  const j = Math.floor(v);
  if (!on(i, j)) return 0;
  // distance to the nearest off cell, for rounded ribbon edges
  let d = 1;
  for (let b = j - 1; b <= j + 1; b++) {
    for (let a = i - 1; a <= i + 1; a++) {
      if (on(a, b)) continue;
      const dx = Math.max(a - u, 0, u - (a + 1));
      const dy = Math.max(b - v, 0, v - (b + 1));
      d = Math.min(d, Math.hypot(dx, dy));
    }
  }
  return Math.sqrt(clamp(d / 0.3, 0, 1));
}

/**
 * Carved frieze panel, 2 m x 1 m, relief up to ~4 cm: egg-and-dart and bead-and-reel mouldings, a running vine
 * scroll with spirals, leaves and rosettes, and a Greek-key band. Front faces +z, bottom edge at y = 0.
 */
export function makeReliefPanel(seed: number, resX = 1000): VirtualMeshSource {
  const noise = new Perlin(seed);
  const rand = mulberry32(seed);
  const W = 2;
  const H = 1;
  const nx = resX;
  const ny = Math.round((resX * H) / W);
  const w = nx + 1;
  const heights = new Float32Array(w * (ny + 1));
  const cell = W / nx;

  const stemY = 0.47;
  const amp = 0.13;
  const lambda = 0.5;
  // spirals: one per half wave, alternating above/below the stem
  const spirals: { cx: number; cy: number; dir: number }[] = [];
  for (let k = -1; k <= 8; k++) {
    const x = -W / 2 + (k + 0.5) * (lambda / 2);
    const up = k % 2 === 0 ? 1 : -1;
    spirals.push({ cx: x, cy: stemY - up * 0.12, dir: up });
  }
  const spiralA = 0.022;
  const spiralTurns = 1.6;
  const spiralB = Math.log(0.12 / spiralA) / (TAU * spiralTurns);
  const leafSeeds = Array.from({ length: 40 }, () => [rand(), rand(), rand()]);

  const ribbon = (d: number, width: number, height: number) => (d < width ? height * Math.sqrt(1 - (d / width) ** 2) : 0);

  for (let j = 0; j <= ny; j++) {
    // row j is at v = H - j * cell (rows run top to bottom so the rotation below keeps the winding)
    const v = H - j * cell;
    for (let i = 0; i <= nx; i++) {
      const x = -W / 2 + i * cell;
      let h = 0;
      const frame = Math.min(x + W / 2, W / 2 - x, v, H - v);
      // outer frame with a rounded inner edge
      if (frame < 0.035) h = 0.03 * (frame < 0.03 ? 1 : Math.sqrt(1 - ((frame - 0.03) / 0.005) ** 2));
      else if (v > 0.84) {
        // egg and dart
        const u = fract((x + W / 2) / 0.08) - 0.5;
        const t = (v - 0.84) / (H - 0.035 - 0.84) - 0.5;
        const e = (u / 0.32) ** 2 + (t / 0.36) ** 2;
        h = 0.006;
        if (e < 1) h = 0.026 * Math.sqrt(1 - e) + 0.006;
        else if (e < 1.25) h = 0.002;
        else if (e < 1.9) h = 0.006 + 0.014 * Math.sin(((e - 1.25) / 0.65) * Math.PI);
        if (Math.abs(Math.abs(u) - 0.5) < 0.06 * (t + 0.6) && t < 0.4) h = Math.max(h, 0.018);
      } else if (v > 0.795) {
        // bead and reel
        const u = fract((x + W / 2) / 0.05) - 0.5;
        const t = (v - 0.795) / 0.045 - 0.5;
        const e = (u / 0.3) ** 2 + (t / 0.48) ** 2;
        h = 0.004;
        if (e < 1) h = 0.004 + 0.016 * Math.sqrt(1 - e);
        const reel = Math.abs(Math.abs(u) - 0.42) < 0.04 && Math.abs(t) < 0.35;
        if (reel) h = Math.max(h, 0.012);
      } else if (v < 0.16) {
        // Greek key band with fillets
        if (v < 0.045 || v > 0.145) h = 0.012;
        else h = 0.002 + 0.02 * meander((x + W / 2) / 0.02, (v - 0.045) / 0.02);
      } else if (v > 0.18 && v < 0.78) {
        // field: vine scroll
        const phase = (TAU * (x + W / 2)) / lambda;
        const sy = stemY + amp * Math.sin(phase);
        const slope = amp * Math.cos(phase) * (TAU / lambda);
        const dStem = Math.abs(v - sy) / Math.sqrt(1 + slope * slope);
        h = Math.max(h, ribbon(dStem, 0.013, 0.022) - 0.004 * Math.exp(-((dStem / 0.003) ** 2)));
        for (const s of spirals) {
          const dx = x - s.cx;
          const dy = v - s.cy;
          if (Math.abs(dx) > 0.15 || Math.abs(dy) > 0.15) continue;
          const rho = Math.hypot(dx, dy);
          const ang = s.dir * Math.atan2(dy, dx);
          // spiral arm: nearest turn of rho = a * exp(b * theta)
          if (rho > spiralA * 0.6) {
            const turn = Math.round((Math.log(rho / spiralA) / spiralB - ang) / TAU);
            let best = Infinity;
            for (let k = turn - 1; k <= turn + 1; k++) {
              const th = ang + k * TAU;
              if (th < 0 || th > spiralTurns * TAU + Math.PI * 0.5) continue;
              best = Math.min(best, Math.abs(rho - spiralA * Math.exp(spiralB * th)));
            }
            const width = 0.006 + 0.04 * rho;
            h = Math.max(h, ribbon(best, width * 0.5, 0.02) - 0.003 * Math.exp(-((best / 0.0025) ** 2)));
          }
          // rosette in the eye: 8 petals and a boss
          const R = 0.034;
          if (rho < R) {
            const a = Math.atan2(dy, dx);
            const petal = R * (0.55 + 0.45 * Math.abs(Math.cos(4 * a)) ** 0.6);
            if (rho < petal) h = Math.max(h, 0.024 * Math.sqrt(1 - rho / petal) + 0.006);
            if (rho < R * 0.28) h = Math.max(h, 0.034 * Math.sqrt(1 - (rho / (R * 0.28)) ** 2) + 0.006);
            h -= 0.002 * Math.exp(-(((Math.abs(Math.sin(4 * a)) * rho) / 0.0015) ** 2));
          }
        }
        // leaves along the stem: pointed ellipses with a central vein and side veins
        const k = Math.floor((x + W / 2) / (lambda / 4));
        for (let kk = k - 1; kk <= k + 1; kk++) {
          const ls = leafSeeds[(kk + 40) % 40];
          const lx = -W / 2 + (kk + 0.5) * (lambda / 4) + (ls[0] - 0.5) * 0.03;
          const lp = (TAU * (lx + W / 2)) / lambda;
          const ly = stemY + amp * Math.sin(lp);
          const dir = kk % 2 === 0 ? 1 : -1;
          const ang = Math.atan(amp * Math.cos(lp) * (TAU / lambda)) + dir * (0.7 + ls[1] * 0.4);
          const len = 0.06 + ls[2] * 0.02;
          const ox = lx + Math.cos(ang) * len * dir;
          const oy = ly + Math.sin(ang) * len * dir;
          const qx = (x - ox) * Math.cos(ang) + (v - oy) * Math.sin(ang);
          const qy = -(x - ox) * Math.sin(ang) + (v - oy) * Math.cos(ang);
          const t = qx / len;
          if (Math.abs(t) >= 1) continue;
          const halfW = 0.024 * (1 - t * t) * (1 - 0.3 * t);
          const lobes = 1 + 0.25 * Math.sin(t * 14);
          const wv = Math.abs(qy) / (halfW * lobes);
          if (wv < 1) {
            let lh = 0.018 * Math.sqrt(1 - wv * wv) * (1 - t * t * 0.5);
            lh -= 0.003 * Math.exp(-((qy / 0.0018) ** 2));
            lh -= 0.0015 * Math.max(0, Math.cos((t * 6 + Math.abs(qy) * 60) * Math.PI)) * wv;
            h = Math.max(h, lh);
          }
        }
      } else {
        // fillets between bands
        h = 0.012;
      }
      // weathering: erosion, pitting, a chipped corner
      h -= 0.0025 * (0.5 + 0.5 * noise.fbm3(x * 6, v * 6, 0.5, 3)) + 0.0012 * smoothstep(0.4, 0.8, noise.noise3(x * 60, v * 60, 2.5));
      const corner = Math.hypot(x - W / 2, v - H) - (0.08 + 0.04 * noise.noise3(x * 12, v * 12, 3.3));
      if (corner < 0) h += corner * 0.6;
      heights[j * w + i] = h;
    }
  }

  heights.set(blurGrid(heights, nx, ny, 1));
  const blur = blurGrid(heights, nx, ny, 6);
  const colors = new Float32Array(w * (ny + 1) * 3);
  const c: RGB = [0, 0, 0];
  const base = PALETTE.marble;
  for (let j = 0; j <= ny; j++) {
    const v = H - j * cell;
    for (let i = 0; i <= nx; i++) {
      const x = -W / 2 + i * cell;
      const k = j * w + i;
      c[0] = base[0];
      c[1] = base[1];
      c[2] = base[2];
      scaleRGB(c, 1 + 0.08 * noise.fbm3(x * 4, v * 4, 9.1, 3));
      const cavity = clamp((blur[k] - heights[k]) * 60, 0, 1);
      scaleRGB(c, 1 - 0.55 * cavity);
      blend(c, PALETTE.dirt, cavity * 0.3);
      const streak = smoothstep(0.2, 0.6, noise.fbm3(x * 9, v * 0.6, 4.2, 3));
      scaleRGB(c, 1 - 0.22 * streak);
      blend(c, PALETTE.lichen, smoothstep(0.4, 0.65, noise.fbm3(x * 3, v * 3, 1.7, 3)) * 0.4);
      colors.set(c, k * 3);
    }
  }
  // Build lying (height up, rows along +z) then stand it up: (x, y, z) -> (x, -z, y), with row j at z = -v.
  const mesh = heightSlab(nx, ny, -W / 2, -H, W, H, heights, colors, -0.1, PALETTE.marble);
  const pos = mesh.positions;
  for (let i = 0; i < pos.length; i += 3) {
    const y = pos[i + 1];
    const z = pos[i + 2];
    pos[i + 1] = -z;
    pos[i + 2] = y;
  }
  mesh.normals = computeNormals(pos, mesh.indices);
  return mesh;
}

// ---------------------------------------------------------------------------- hero sculpture

/**
 * Twisted bronze knot: a (2, 3) torus knot swept by a five-strand rope with fine carved ridges along each strand
 * and hatching on alternate strands. About `along * around * 2` triangles; lowest point at y = 0.
 */
export function makeTwistedKnot(seed: number, along = 2800, around = 360): VirtualMeshSource {
  const noise = new Perlin(seed);
  const P = 2;
  const Q = 3;
  const R = 1.0;
  const r = 0.45;
  const tube = 0.19;
  const scale = 1.15;
  const strands = 5;
  const twists = 9;
  const curve = (t: number, out: number[]) => {
    const rr = R + r * Math.cos(Q * t);
    out[0] = rr * Math.cos(P * t);
    out[1] = rr * Math.sin(P * t);
    out[2] = r * Math.sin(Q * t);
  };
  // Parallel-transport frames, with the closing twist spread evenly along the loop.
  const pts: number[][] = [];
  const tang: number[][] = [];
  for (let i = 0; i < along; i++) {
    const a = [0, 0, 0];
    const b = [0, 0, 0];
    const t = (i / along) * TAU;
    curve(t, a);
    curve(t + 1e-4, b);
    pts.push(a);
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const l = Math.hypot(d[0], d[1], d[2]);
    tang.push([d[0] / l, d[1] / l, d[2] / l]);
  }
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a: number[]) => {
    const l = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / l, a[1] / l, a[2] / l];
  };
  const normals: number[][] = [norm(cross(tang[0], [0, 0, 1]))];
  for (let i = 1; i <= along; i++) {
    const t = tang[i % along];
    const prev = normals[i - 1];
    const n = norm([prev[0] - t[0] * dot(prev, t), prev[1] - t[1] * dot(prev, t), prev[2] - t[2] * dot(prev, t)]);
    normals.push(n);
  }
  const n0 = normals[0];
  const nEnd = normals[along];
  const closing = Math.atan2(dot(cross(nEnd, n0), tang[0]), dot(nEnd, n0));

  const mb = new MeshBuilder(along * around + 8, along * around * 2 + 8);
  const c: RGB = [0, 0, 0];
  let minY = Infinity;
  const ring: number[] = [];
  for (let i = 0; i < along; i++) {
    const t = tang[i];
    const s = i / along;
    const corr = closing * s;
    const nn = normals[i];
    const bb = cross(t, nn);
    const nx = [nn[0] * Math.cos(corr) + bb[0] * Math.sin(corr), nn[1] * Math.cos(corr) + bb[1] * Math.sin(corr), nn[2] * Math.cos(corr) + bb[2] * Math.sin(corr)];
    const bx = cross(t, nx);
    const psi = twists * TAU * s;
    for (let k = 0; k < around; k++) {
      const phi = (k / around) * TAU;
      const alpha = phi - psi;
      const w = fract((alpha * strands) / TAU);
      const strandId = Math.floor((((alpha * strands) / TAU) % strands) + strands) % strands;
      const bulge = Math.sqrt(Math.max(0, 1 - (2 * w - 1) ** 2));
      let rad = tube * (0.76 + 0.24 * bulge);
      const ridges = 0.5 + 0.5 * Math.cos(TAU * w * 7);
      rad -= tube * 0.018 * ridges * bulge;
      if (strandId % 2 === 0) rad -= tube * 0.02 * Math.max(0, Math.cos(TAU * (s * 900 + w * 2.5))) ** 6 * bulge;
      const p0 = pts[i];
      const x0 = p0[0] + (nx[0] * Math.cos(phi) + bx[0] * Math.sin(phi)) * rad;
      const y0 = p0[1] + (nx[1] * Math.cos(phi) + bx[1] * Math.sin(phi)) * rad;
      const z0 = p0[2] + (nx[2] * Math.cos(phi) + bx[2] * Math.sin(phi)) * rad;
      const wear = 0.0015 * noise.fbm3(x0 * 12, y0 * 12, z0 * 12, 3);
      const k2 = 1 + wear / rad;
      const x = (p0[0] + (x0 - p0[0]) * k2) * scale;
      const y = (p0[1] + (y0 - p0[1]) * k2) * scale;
      const z = (p0[2] + (z0 - p0[2]) * k2) * scale;
      minY = Math.min(minY, y);
      // bronze with verdigris in the grooves and streaks
      c[0] = PALETTE.bronze[0];
      c[1] = PALETTE.bronze[1];
      c[2] = PALETTE.bronze[2];
      const groove = (1 - bulge) * 0.9 + ridges * 0.25 * bulge;
      const patina = clamp(groove + 0.35 * noise.fbm3(x * 3, y * 3, z * 3, 3), 0, 1);
      blend(c, PALETTE.verdigris, patina * 0.65);
      scaleRGB(c, 1 + 0.15 * noise.fbm3(x * 9, y * 9, z * 9, 2));
      ring.push(mb.vertex(x, y, z, c));
    }
  }
  for (let i = 0; i < along; i++) {
    const i1 = (i + 1) % along;
    for (let k = 0; k < around; k++) {
      const k1 = (k + 1) % around;
      const a = ring[i * around + k];
      const b = ring[i * around + k1];
      const cc = ring[i1 * around + k1];
      const d = ring[i1 * around + k];
      mb.quad(a, b, cc, d);
    }
  }
  const mesh = mb.finish();
  for (let i = 1; i < mesh.positions.length; i += 3) mesh.positions[i] -= minY;
  return mesh;
}

/** Round pedestal, 1.6 m tall: moulded base and cornice, a die with a carved Greek-key band. */
export function makePedestal(seed: number): VirtualMeshSource {
  const noise = new Perlin(seed);
  const mb = new MeshBuilder(1 << 18, 1 << 19);
  const p = new Profile();
  p.point(0, 0, Part.Base)
    .line(1.0, 0, 12, Part.Base)
    .line(1.0, 0.16, 8, Part.Base)
    .arc(0.94, 0.22, 0.06, -Math.PI / 2, Math.PI / 2, 18, Part.Base)
    .arc(0.84, 0.34, 0.06, -Math.PI / 2, -Math.PI * 1.5, 18, Part.Base)
    .line(0.82, 0.42, 3, Part.Base)
    .span(0.82, 0.42, 1.34, 180, Part.Band)
    .line(0.86, 1.36, 3, Part.Base)
    .arc(0.86, 1.42, 0.06, -Math.PI / 2, 0, 10, Part.Base)
    .arc(0.86 + 0.06 + 0.04, 1.42, 0.04, Math.PI, Math.PI / 2, 8, Part.Base)
    .line(0.97, 1.5, 3, Part.Base)
    .line(0.97, 1.58, 4, Part.Base)
    .line(0, 1.58, 14, Part.Base);
  const rows = p.finish();
  const c: RGB = [0, 0, 0];
  const tint = PALETTE.marble;
  lathe(mb, rows, 720, (row, _v, theta) => {
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    let dn = 0;
    let cavity = 0;
    if (row.part === Part.Band) {
      const y = row.y;
      const u = (theta / TAU) * 150; // 25 key periods around
      if (y > 0.72 && y < 1.02) {
        const m = meander(u, (y - 0.78) / 0.035);
        dn += 0.012 * m;
        cavity = y > 0.78 && y < 0.78 + 5 * 0.035 ? (1 - m) * 0.6 : 0;
        if (Math.abs(y - 0.75) < 0.012 || Math.abs(y - 0.985) < 0.012) dn += 0.01;
      }
    }
    const r = row.r;
    let x = r * ct;
    let y = row.y;
    let z = r * st;
    dn += 0.003 * noise.fbm3(x * 3, y * 3, z * 3, 3) + 0.0012 * noise.fbm3(x * 25, y * 25, z * 25, 2);
    x += dn * row.nr * ct;
    z += dn * row.nr * st;
    y += dn * row.ny;
    c[0] = tint[0];
    c[1] = tint[1];
    c[2] = tint[2];
    scaleRGB(c, (1 - 0.35 * cavity) * (1 + 0.08 * noise.fbm3(x * 4, y * 4, z * 4, 2)));
    blend(c, PALETTE.lichen, smoothstep(0.4, 0.65, noise.fbm3(x * 2.5, y * 2.5, z * 2.5, 3)) * 0.35);
    blend(c, PALETTE.dirt, smoothstep(0.5, 0, y) * 0.4);
    return mb.vertex(x, y, z, c);
  });
  return mb.finish();
}

// ---------------------------------------------------------------------------- the ruins asset set

const ashlarStyle: StoneStyle = { bevelSegments: 3, wear: 0.02, rough: 0.006, chips: 4, chipDepth: 0.08, dirt: (y) => smoothstep(1.2, 0, y) * 0.35 };
const rubbleStyle: StoneStyle = { bevelSegments: 2, wear: 0.022, rough: 0.008, chips: 3, chipDepth: 0.06, dirt: (y) => smoothstep(0.6, 0, y) * 0.3 };
const ashlarPalette = [PALETTE.limestone, PALETTE.travertine, PALETTE.limestone];
const rubblePalette = [PALETTE.greyStone, PALETTE.limestone, PALETTE.travertine, PALETTE.limestone];

export interface RuinsAsset {
  name: string;
  make: () => VirtualMeshSource;
}

/** Every asset of the ruins scene except the terrain (which depends on the scene's height function). */
export const RUINS_ASSETS: RuinsAsset[] = [
  { name: 'knot', make: () => makeTwistedKnot(7) },
  { name: 'pedestal', make: () => makePedestal(8) },
  { name: 'relief', make: () => makeReliefPanel(9) },
  { name: 'column', make: () => makeColumn(21, 'full') },
  { name: 'column-broken', make: () => makeColumn(22, 'broken') },
  { name: 'drum', make: () => makeColumn(23, 'drum') },
  { name: 'capital', make: () => makeColumn(24, 'capital') },
  {
    name: 'wall-ashlar',
    make: () => makeWall({ seed: 31, length: 8, height: 6, thickness: 0.9, course: [0.45, 0.58], block: [0.8, 1.4], gap: 0.008, bevel: 0.03, step: 0.03, ruin: 0, palette: ashlarPalette, style: ashlarStyle }),
  },
  {
    name: 'wall-ashlar-ruin',
    make: () =>
      makeWall({ seed: 32, length: 8, height: 6, thickness: 0.9, course: [0.45, 0.58], block: [0.8, 1.4], gap: 0.008, bevel: 0.03, step: 0.03, ruin: 0.75, fallen: 8, palette: ashlarPalette, style: ashlarStyle }),
  },
  {
    name: 'wall-rubble',
    make: () =>
      makeWall({ seed: 33, length: 8, height: 3.4, thickness: 0.7, course: [0.14, 0.24], block: [0.25, 0.52], gap: 0.014, bevel: 0.035, step: 0.03, ruin: 0.35, fallen: 6, jitter: 0.03, palette: rubblePalette, style: rubbleStyle }),
  },
  {
    name: 'wall-rubble-ruin',
    make: () =>
      makeWall({ seed: 34, length: 8, height: 3.4, thickness: 0.7, course: [0.14, 0.24], block: [0.25, 0.52], gap: 0.014, bevel: 0.035, step: 0.03, ruin: 0.85, fallen: 10, jitter: 0.03, palette: rubblePalette, style: rubbleStyle }),
  },
  {
    name: 'step',
    make: () => makeWall({ seed: 35, length: 12, height: 0.4, thickness: 1.1, course: [0.4, 0.4], block: [1.0, 1.7], gap: 0.01, bevel: 0.035, step: 0.035, ruin: 0, palette: ashlarPalette, style: ashlarStyle }),
  },
  {
    name: 'beam',
    make: () =>
      makeWall({ seed: 36, length: 13.6, height: 0.85, thickness: 1.15, course: [0.85, 0.85], block: [3.4, 3.4], aligned: true, gap: 0.01, bevel: 0.04, step: 0.04, ruin: 0, palette: ashlarPalette, style: ashlarStyle }),
  },
  { name: 'flagstones', make: () => makePavingTile('flagstones', 41) },
  { name: 'cobbles', make: () => makePavingTile('cobbles', 42) },
  { name: 'boulder', make: () => makeRock(51, { detail: 290, stretch: [1.3, 0.9, 1.1], strata: 7, strataDepth: 0.03, cracks: 0.08, lumpy: 0.4, tint: PALETTE.greyStone }) },
  { name: 'cliff', make: () => makeRock(52, { detail: 290, stretch: [1.8, 1.3, 1.4], strata: 7, strataDepth: 0.03, cracks: 0.12, lumpy: 0.5, tint: PALETTE.greyStone }) },
  { name: 'cypress', make: () => makeCypress(61) },
];

// ---------------------------------------------------------------------------- cypress

/** Mediterranean cypress, ~11 m: a dense, flame-shaped crown of clumpy foliage on a short trunk (~40k triangles). */
export function makeCypress(seed: number): VirtualMeshSource {
  const noise = new Perlin(seed);
  const mb = new MeshBuilder(1 << 16, 1 << 17);
  const n = 56;
  const axis = boxAxis(1, 1, n / 2, 0);
  const H = 10.5;
  const dark: RGB = rgb(0x1f3320);
  const light: RGB = rgb(0x4a6233);
  const c: RGB = [0, 0, 0];
  cubeSurface(mb, n, n, n, (i, j, k) => {
    let dx = axis[i];
    let dy = axis[j];
    let dz = axis[k];
    const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dx /= l;
    dy /= l;
    dz /= l;
    // flame profile: widest a third of the way up, pointed top, rounded base
    const t = (dy + 1) / 2;
    const width = 1.15 * Math.pow(Math.sin(Math.PI * Math.pow(t, 0.75)), 0.8) + 0.04;
    const clumps = 0.22 * noise.fbm3(dx * 3, t * 9, dz * 3, 3) + 0.1 * (1 - Math.abs(noise.noise3(dx * 7, t * 22, dz * 7)));
    const r = width * (1 + clumps);
    const x = dx * r;
    const y = 0.8 + t * H + clumps * 0.3 * dy;
    const z = dz * r;
    c[0] = dark[0];
    c[1] = dark[1];
    c[2] = dark[2];
    blend(c, light, clamp(0.5 + clumps * 2.2 + 0.25 * noise.noise3(x * 2, y * 0.5, z * 2), 0, 1) * 0.8);
    return mb.vertex(x, y, z, c);
  });
  // trunk
  const p = new Profile();
  p.point(0, 0, Part.Base).line(0.22, 0, 2, Part.Base).line(0.16, 1.6, 6, Part.Base).line(0, 1.6, 2, Part.Base);
  const bark = rgb(0x4a3a2a);
  lathe(mb, p.finish(), 12, (row, _v, theta) => mb.vertex(row.r * Math.cos(theta), row.y, row.r * Math.sin(theta), bark));
  return mb.finish();
}
