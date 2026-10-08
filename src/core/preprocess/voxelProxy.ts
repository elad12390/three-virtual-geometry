/**
 * Voxel proxies for coarse LODs, for foliage and other aggregate geometry.
 *
 * Triangle simplification breaks down on aggregate geometry such as a tree crown made of hundreds of
 * separate leaf clumps: below a few hundred triangles it can only delete or collapse whole clumps, so
 * the silhouette shrinks and the LOD error jumps. A voxel proxy keeps the volume instead: the mesh is
 * voxelized, gaps between clumps are closed, and a smooth closed surface is extracted (surface nets).
 * That surface is one connected shape, which simplifies gracefully to a handful of triangles.
 *
 * Every proxy vertex lies within about one cell of the original surface, and closing fills gaps of at most
 * two cells, so `VOXEL_ERROR_CELLS * cellSize` bounds how far the proxy deviates from the original.
 */

/** Deviation of a proxy from the source mesh, in cells (surface offset plus closed gaps). */
export const VOXEL_ERROR_CELLS = 2;

export interface VoxelProxy {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array | null;
  /** UV of a representative source sample near each vertex (null when the source has no UVs). */
  uvs: Float32Array | null;
  indices: Uint32Array;
  cellSize: number;
}

/**
 * Voxelizes the triangles `indices` of a mesh with `resolution` cells along its longest side and returns
 * a closed, smooth surface around the occupied volume (null if the mesh is empty).
 * With `uvs`, each proxy vertex takes the UV of the first source sample in its cell's voxels, so textured
 * geometry keeps plausible colors at a distance (UVs cannot be averaged across islands).
 */
export function buildVoxelProxy(
  positions: Float32Array,
  colors: Float32Array | null | undefined,
  indices: Uint32Array,
  resolution: number,
  uvs?: Float32Array | null
): VoxelProxy | null {
  if (indices.length === 0) return null;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < indices.length; i++) {
    const o = indices[i] * 3;
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], positions[o + k]);
      max[k] = Math.max(max[k], positions[o + k]);
    }
  }
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  if (!(extent > 0)) return null;
  const cell = extent / resolution;
  // Samples sit at cell centers; 3 cells of padding keep the closed surface away from the grid border.
  const PAD = 3;
  const n = [0, 1, 2].map((k) => Math.ceil((max[k] - min[k]) / cell) + 1 + PAD * 2);
  const origin = [0, 1, 2].map((k) => min[k] - PAD * cell);
  const [nx, ny, nz] = n;
  const sx = 1;
  const sy = nx;
  const sz = nx * ny;
  const total = nx * ny * nz;

  // ---- 1. occupancy (+ color) by sampling every triangle at half-cell spacing ----
  const occupied = new Uint8Array(total);
  const colorSum = colors ? new Float32Array(total * 3) : null;
  const colorCount = colors ? new Uint32Array(total) : null;
  // First UV sampled in each voxel (NaN: none yet).
  const voxelUv = uvs ? new Float32Array(total * 2).fill(NaN) : null;
  let sampleU = 0;
  let sampleV = 0;
  const mark = (x: number, y: number, z: number, r: number, g: number, b: number) => {
    const i = Math.floor((x - origin[0]) / cell) * sx + Math.floor((y - origin[1]) / cell) * sy + Math.floor((z - origin[2]) / cell) * sz;
    occupied[i] = 1;
    if (voxelUv && Number.isNaN(voxelUv[i * 2])) {
      voxelUv[i * 2] = sampleU;
      voxelUv[i * 2 + 1] = sampleV;
    }
    if (colorSum) {
      colorSum[i * 3] += r;
      colorSum[i * 3 + 1] += g;
      colorSum[i * 3 + 2] += b;
      colorCount![i]++;
    }
  };
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3;
    const b = indices[t + 1] * 3;
    const c = indices[t + 2] * 3;
    const longest = Math.max(dist(positions, a, b), dist(positions, b, c), dist(positions, c, a));
    const steps = Math.max(1, Math.ceil(longest / (cell * 0.5)));
    for (let i = 0; i <= steps; i++) {
      for (let j = 0; j <= steps - i; j++) {
        const u = i / steps;
        const v = j / steps;
        const w = 1 - u - v;
        const x = positions[a] * w + positions[b] * u + positions[c] * v;
        const y = positions[a + 1] * w + positions[b + 1] * u + positions[c + 1] * v;
        const z = positions[a + 2] * w + positions[b + 2] * u + positions[c + 2] * v;
        if (uvs) {
          const ta = indices[t] * 2;
          const tb = indices[t + 1] * 2;
          const tc = indices[t + 2] * 2;
          sampleU = uvs[ta] * w + uvs[tb] * u + uvs[tc] * v;
          sampleV = uvs[ta + 1] * w + uvs[tb + 1] * u + uvs[tc + 1] * v;
        }
        if (colors) {
          mark(
            x,
            y,
            z,
            colors[a] * w + colors[b] * u + colors[c] * v,
            colors[a + 1] * w + colors[b + 1] * u + colors[c + 1] * v,
            colors[a + 2] * w + colors[b + 2] * u + colors[c + 2] * v
          );
        } else mark(x, y, z, 0, 0, 0);
      }
    }
  }

  // ---- 2. morphological closing (dilate, then erode) fuses clumps into one volume ----
  const closed = erode(dilate(occupied, n), n);

  // ---- 3. smooth field: 3x3x3 box blur of the occupancy, iso level 0.5 ----
  const field = new Float32Array(total);
  for (let i = 0; i < total; i++) field[i] = closed[i];
  blurAxis(field, n, sx, nx);
  blurAxis(field, n, sy, ny);
  blurAxis(field, n, sz, nz);
  const ISO = 0.5;

  // ---- 4. surface nets: one vertex per cell crossing the iso surface, one quad per crossing edge ----
  const cellsX = nx - 1;
  const cellsY = ny - 1;
  const cellsZ = nz - 1;
  const cellVertex = new Int32Array(cellsX * cellsY * cellsZ).fill(-1);
  const outPositions: number[] = [];
  const outNormals: number[] = [];
  const outColors: number[] = [];
  const outUvs: number[] = [];
  const corner = new Float32Array(8);
  for (let z = 0; z < cellsZ; z++) {
    for (let y = 0; y < cellsY; y++) {
      for (let x = 0; x < cellsX; x++) {
        const base = x * sx + y * sy + z * sz;
        let inside = 0;
        for (let c = 0; c < 8; c++) {
          corner[c] = field[base + (c & 1) * sx + ((c >> 1) & 1) * sy + ((c >> 2) & 1) * sz];
          if (corner[c] >= ISO) inside++;
        }
        if (inside === 0 || inside === 8) continue;
        // Vertex: mean of the iso crossings on the cell's 12 edges.
        let px = 0;
        let py = 0;
        let pz = 0;
        let crossings = 0;
        for (let c = 0; c < 8; c++) {
          for (const bit of [1, 2, 4]) {
            if (c & bit) continue;
            const d = c | bit;
            const f0 = corner[c];
            const f1 = corner[d];
            if (f0 >= ISO === f1 >= ISO) continue;
            const s = (ISO - f0) / (f1 - f0);
            px += (c & 1) + (bit === 1 ? s : 0);
            py += ((c >> 1) & 1) + (bit === 2 ? s : 0);
            pz += ((c >> 2) & 1) + (bit === 4 ? s : 0);
            crossings++;
          }
        }
        cellVertex[x + y * cellsX + z * cellsX * cellsY] = outPositions.length / 3;
        // Sample (i, j, k) sits at origin + (i + 0.5) * cell.
        outPositions.push(origin[0] + (x + 0.5 + px / crossings) * cell, origin[1] + (y + 0.5 + py / crossings) * cell, origin[2] + (z + 0.5 + pz / crossings) * cell);
        // Normal: the field decreases outward, so the outward normal is minus its gradient.
        const gx = corner[1] + corner[3] + corner[5] + corner[7] - corner[0] - corner[2] - corner[4] - corner[6];
        const gy = corner[2] + corner[3] + corner[6] + corner[7] - corner[0] - corner[1] - corner[4] - corner[5];
        const gz = corner[4] + corner[5] + corner[6] + corner[7] - corner[0] - corner[1] - corner[2] - corner[3];
        const len = Math.hypot(gx, gy, gz) || 1;
        outNormals.push(-gx / len, -gy / len, -gz / len);
        if (colorSum) {
          let r = 0;
          let g = 0;
          let b = 0;
          let count = 0;
          for (let c = 0; c < 8; c++) {
            const i = base + (c & 1) * sx + ((c >> 1) & 1) * sy + ((c >> 2) & 1) * sz;
            if (!colorCount![i]) continue;
            r += colorSum[i * 3];
            g += colorSum[i * 3 + 1];
            b += colorSum[i * 3 + 2];
            count += colorCount![i];
          }
          if (count) outColors.push(r / count, g / count, b / count);
          else outColors.push(NaN, NaN, NaN); // filled in below from neighbours
        }
        if (voxelUv) {
          let u = NaN;
          let v = NaN;
          for (let c = 0; c < 8 && Number.isNaN(u); c++) {
            const i = base + (c & 1) * sx + ((c >> 1) & 1) * sy + ((c >> 2) & 1) * sz;
            u = voxelUv[i * 2];
            v = voxelUv[i * 2 + 1];
          }
          outUvs.push(u, v); // NaN: filled in below from neighbours
        }
      }
    }
  }

  const outIndices: number[] = [];
  const cellIndex = (x: number, y: number, z: number) => cellVertex[x + y * cellsX + z * cellsX * cellsY];
  const quad = (a: number, b: number, c: number, d: number, nx_: number, ny_: number, nz_: number) => {
    // Orient so the face normal agrees with the outward direction of the crossing edge.
    const ux = outPositions[c * 3] - outPositions[a * 3];
    const uy = outPositions[c * 3 + 1] - outPositions[a * 3 + 1];
    const uz = outPositions[c * 3 + 2] - outPositions[a * 3 + 2];
    const vx = outPositions[d * 3] - outPositions[b * 3];
    const vy = outPositions[d * 3 + 1] - outPositions[b * 3 + 1];
    const vz = outPositions[d * 3 + 2] - outPositions[b * 3 + 2];
    const cx = uy * vz - uz * vy;
    const cy = uz * vx - ux * vz;
    const cz = ux * vy - uy * vx;
    if (cx * nx_ + cy * ny_ + cz * nz_ >= 0) outIndices.push(a, b, c, a, c, d);
    else outIndices.push(a, c, b, a, d, c);
  };
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const i = x * sx + y * sy + z * sz;
        const here = field[i] >= ISO;
        // Edge to +x: the four cells around it share x and span y-1..y, z-1..z.
        if (here !== field[i + sx] >= ISO) {
          const s = here ? 1 : -1;
          quad(cellIndex(x, y - 1, z - 1), cellIndex(x, y, z - 1), cellIndex(x, y, z), cellIndex(x, y - 1, z), s, 0, 0);
        }
        if (here !== field[i + sy] >= ISO) {
          const s = here ? 1 : -1;
          quad(cellIndex(x - 1, y, z - 1), cellIndex(x, y, z - 1), cellIndex(x, y, z), cellIndex(x - 1, y, z), 0, s, 0);
        }
        if (here !== field[i + sz] >= ISO) {
          const s = here ? 1 : -1;
          quad(cellIndex(x - 1, y - 1, z), cellIndex(x, y - 1, z), cellIndex(x, y, z), cellIndex(x - 1, y, z), 0, 0, s);
        }
      }
    }
  }
  if (outIndices.length === 0) return null;

  let proxyColors: Float32Array | null = null;
  if (colorSum) {
    proxyColors = Float32Array.from(outColors);
    // Vertices of cells with no colored sample (volume added by closing) take the mean of their triangles' others.
    fillMissing(proxyColors, outIndices, 3);
  }
  let proxyUvs: Float32Array | null = null;
  if (voxelUv) {
    proxyUvs = Float32Array.from(outUvs);
    fillMissing(proxyUvs, outIndices, 2);
  }
  return {
    positions: Float32Array.from(outPositions),
    normals: Float32Array.from(outNormals),
    colors: proxyColors,
    uvs: proxyUvs,
    indices: Uint32Array.from(outIndices),
    cellSize: cell,
  };
}

function dist(p: Float32Array, a: number, b: number) {
  return Math.hypot(p[a] - p[b], p[a + 1] - p[b + 1], p[a + 2] - p[b + 2]);
}

/** 3x3x3 max (26-neighbourhood), separable per axis. */
function dilate(src: Uint8Array, n: number[]) {
  let a = src;
  for (let axis = 0; axis < 3; axis++) a = morph(a, n, axis, 1);
  return a;
}

/** 3x3x3 min, separable per axis. */
function erode(src: Uint8Array, n: number[]) {
  let a = src;
  for (let axis = 0; axis < 3; axis++) a = morph(a, n, axis, 0);
  return a;
}

function morph(src: Uint8Array, n: number[], axis: number, grow: number) {
  const [nx, ny, nz] = n;
  const stride = axis === 0 ? 1 : axis === 1 ? nx : nx * ny;
  const size = n[axis];
  const out = new Uint8Array(src.length);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const i = x + y * nx + z * nx * ny;
        const c = axis === 0 ? x : axis === 1 ? y : z;
        const prev = c > 0 ? src[i - stride] : 0;
        const next = c < size - 1 ? src[i + stride] : 0;
        out[i] = grow ? src[i] | prev | next : src[i] & prev & next;
      }
    }
  }
  return out;
}

/** In-place 3-tap box blur along one axis (borders treated as empty). */
function blurAxis(field: Float32Array, n: number[], stride: number, size: number) {
  const [nx, ny, nz] = n;
  const line = new Float32Array(size);
  const lines = (nx * ny * nz) / size;
  // Walk every line along this axis: its first sample index enumerates the other two axes.
  for (let l = 0; l < lines; l++) {
    let start: number;
    if (stride === 1) start = l * nx;
    else if (stride === nx) start = (l % nx) + Math.floor(l / nx) * nx * ny;
    else start = l;
    for (let k = 0; k < size; k++) line[k] = field[start + k * stride];
    for (let k = 0; k < size; k++) {
      const a = k > 0 ? line[k - 1] : 0;
      const b = k < size - 1 ? line[k + 1] : 0;
      field[start + k * stride] = (a + line[k] + b) / 3;
    }
  }
}

/** Fills NaN entries (`size` floats per vertex) with the mean of their triangles' known neighbours. */
function fillMissing(values: Float32Array, indices: number[], size: number) {
  const count = values.length / size;
  for (let pass = 0; pass < 8; pass++) {
    let missing = 0;
    const sum = new Float32Array(count * size);
    const hits = new Uint32Array(count);
    for (let t = 0; t < indices.length; t += 3) {
      for (let k = 0; k < 3; k++) {
        const v = indices[t + k];
        if (!Number.isNaN(values[v * size])) continue;
        for (let m = 1; m < 3; m++) {
          const u = indices[t + ((k + m) % 3)];
          if (Number.isNaN(values[u * size])) continue;
          for (let c = 0; c < size; c++) sum[v * size + c] += values[u * size + c];
          hits[v]++;
        }
      }
    }
    for (let v = 0; v < count; v++) {
      if (!Number.isNaN(values[v * size])) continue;
      if (hits[v]) {
        for (let c = 0; c < size; c++) values[v * size + c] = sum[v * size + c] / hits[v];
      } else missing++;
    }
    if (!missing) return;
  }
  for (let i = 0; i < values.length; i++) if (Number.isNaN(values[i])) values[i] = 0.5;
}
