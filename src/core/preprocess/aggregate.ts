/**
 * Simplification of aggregate geometry: foliage, grass and other meshes made of many small separate pieces.
 *
 * Edge collapses cannot merge pieces that share no vertex, so a group of leaf cards stops simplifying once each card
 * is down to its own few triangles, and the DAG would keep every card at full count up to the voxel levels. Here a
 * group whose regular simplification stalls is reduced piece by piece instead:
 *
 *  1. Each free piece (one that shares no vertex with another group) is simplified on its own, with its open edges
 *     free to move: a bent six-triangle leaflet becomes a flat quad. (Skipped when thinning alone is more accurate.)
 *  2. If that is not enough, neighbouring pieces are thinned in pairs: one piece of each pair is dropped and the
 *     other widened to cover both pieces' area (stochastic simplification of aggregate detail, Cook, Halstead,
 *     Planck and Ryu 2007; area preservation for foliage). A long thin piece widens across its length, so its outline
 *     stays the same; a compact one grows evenly in its plane.
 *
 * The merged piece sits between the two, so the error of a thinned pair is how far either piece's coverage moved (half
 * the distance between equal pieces), or the widening when that is larger. Pieces that share vertices with other groups are left alone, so the group still meets its
 * neighbours without cracks.
 */
import { MeshoptSimplifier } from 'meshoptimizer';

export interface AggregateResult {
  /** Triangles of the simplified group. Vertex ids >= `firstNewVertex` refer to `newVertices`. */
  indices: Uint32Array;
  error: number;
  /** Widened copies of thinned pieces: positions, and the source vertex each copy takes its other attributes from. */
  newPositions: Float32Array;
  newSources: Uint32Array;
  firstNewVertex: number;
}

/** Pieces looked at around each piece (in spatial order) when choosing the pair to thin. */
const PAIR_WINDOW = 24;

interface Piece {
  area: number;
  center: [number, number, number];
  /** Unit axes of the piece, longest first (the last one is its normal for flat pieces). */
  axes: [number, number, number][];
  /** Extent along the axes (max minus min). */
  extent: [number, number, number];
}

/**
 * Simplifies a group of aggregate geometry toward `targetTriangles`. `owner[weld[v]]` is the group using welded
 * vertex v (or -2 when several groups do); `group` is this group's id. Returns null when it cannot remove anything.
 */
export function simplifyAggregate(
  indices: Uint32Array,
  positions: Float32Array,
  weld: Uint32Array,
  owner: Int32Array,
  group: number,
  targetTriangles: number
): AggregateResult | null {
  const triangleCount = indices.length / 3;
  // ---- pieces: connected components over welded vertices
  const parent = new Map<number, number>();
  const find = (x: number): number => {
    let r = x;
    while (parent.get(r)! !== r) r = parent.get(r)!;
    while (parent.get(x)! !== r) {
      const next = parent.get(x)!;
      parent.set(x, r);
      x = next;
    }
    return r;
  };
  for (let i = 0; i < indices.length; i++) {
    const w = weld[indices[i]];
    if (!parent.has(w)) parent.set(w, w);
  }
  for (let t = 0; t < indices.length; t += 3) {
    const a = find(weld[indices[t]]);
    for (let k = 1; k < 3; k++) {
      const b = find(weld[indices[t + k]]);
      if (a !== b) parent.set(b, a);
    }
  }
  const byRoot = new Map<number, number[]>();
  for (let t = 0; t < triangleCount; t++) {
    const root = find(weld[indices[t * 3]]);
    let list = byRoot.get(root);
    if (!list) byRoot.set(root, (list = []));
    list.push(t);
  }
  if (byRoot.size < 2) return null;

  // Current triangles of each piece (global vertex ids), and which pieces may be changed.
  const initial: { indices: Uint32Array; free: boolean }[] = [];
  for (const triangles of byRoot.values()) {
    const own = new Uint32Array(triangles.length * 3);
    let free = true;
    triangles.forEach((t, i) => {
      for (let k = 0; k < 3; k++) {
        const v = indices[t * 3 + k];
        own[i * 3 + k] = v;
        if (owner[weld[v]] !== group) free = false;
      }
    });
    initial.push({ indices: own, free });
  }

  // Simplifying each piece first suits pieces with many triangles; thinning alone suits pieces already near their
  // fewest (a leaflet bent over three quads loses more shape when flattened than when its neighbour is merged into
  // it). Both are tried; the more accurate one that reaches the target wins.
  const candidates = [reduce(true), reduce(false)].filter((c): c is AggregateResult => c !== null);
  const reaches = candidates.filter((c) => c.indices.length / 3 <= targetTriangles * 1.25);
  const pool = reaches.length ? reaches : candidates;
  pool.sort((a, b) => (reaches.length ? a.error - b.error : a.indices.length - b.indices.length));
  return pool[0] ?? null;

  function reduce(simplifyFirst: boolean): AggregateResult | null {
    let error = 0;
    const pieces = initial.map((p) => ({ ...p }));

    // ---- 1. each free piece on its own, open edges free, by the group's reduction ratio
    const ratio = targetTriangles / triangleCount;
    let total = 0;
    for (const piece of pieces) {
      const target = Math.max(6, Math.floor((piece.indices.length * ratio) / 3) * 3);
      if (simplifyFirst && piece.free && piece.indices.length > target) {
        const local = compactLocal(piece.indices, positions);
        const [result, e] = MeshoptSimplifier.simplify(local.indices, local.positions, 3, target, 1e10, ['ErrorAbsolute']);
        if (result.length > 0 && result.length < piece.indices.length) {
          piece.indices = result.map((i) => local.toGlobal[i]);
          error = Math.max(error, e);
        }
      }
      total += piece.indices.length / 3;
    }

    // ---- 2. thin pairs of neighbouring free pieces
    const newPositions: number[] = [];
    const newSources: number[] = [];
    const firstNewVertex = positions.length / 3;
    if (total > targetTriangles) {
      const free = pieces.map((p, i) => ({ p, i })).filter(({ p }) => p.free);
      const info = new Map<number, Piece>();
      for (const { p, i } of free) info.set(i, describe(p.indices, positions));
      // Spatial order: pairs are looked for among nearby pieces only.
      const order = free.map(({ i }) => i);
      const bounds = boundsOf(order.map((i) => info.get(i)!.center));
      const span = Math.max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2], 1e-9);
      const key = (c: [number, number, number]) => morton([(c[0] - bounds[0]) / span, (c[1] - bounds[1]) / span, (c[2] - bounds[2]) / span]);
      order.sort((a, b) => key(info.get(a)!.center) - key(info.get(b)!.center));
      // Pairs only between near neighbours: at most twice the typical spacing of the pieces in this group.
      const nearest = order.map((a, n) => {
        let d = Infinity;
        for (let m = Math.max(0, n - PAIR_WINDOW); m < Math.min(order.length, n + 1 + PAIR_WINDOW); m++) {
          if (m !== n) d = Math.min(d, distance(info.get(a)!.center, info.get(order[m])!.center));
        }
        return d;
      });
      const spacing = [...nearest].sort((x, y) => x - y)[Math.floor(nearest.length / 2)];
      const taken = new Set<number>();
      for (let n = 0; n < order.length && total > targetTriangles; n++) {
        const a = order[n];
        if (taken.has(a)) continue;
        let best = -1;
        let bestDistance = spacing * 2;
        for (let m = n + 1; m < Math.min(order.length, n + 1 + PAIR_WINDOW); m++) {
          const b = order[m];
          if (taken.has(b)) continue;
          const d = distance(info.get(a)!.center, info.get(b)!.center);
          if (d <= bestDistance) {
            bestDistance = d;
            best = b;
          }
        }
        if (best < 0) continue;
        taken.add(a);
        taken.add(best);
        const [keep, drop] = info.get(a)!.area >= info.get(best)!.area ? [a, best] : [best, a];
        const k = info.get(keep)!;
        const dropped = info.get(drop)!;
        const pairArea = Math.max(k.area + dropped.area, 1e-12);
        const factor = pairArea / Math.max(k.area, 1e-12);
        // The merged piece sits at the pair's area-weighted centre, so neither piece's coverage moves farther than
        // its share of the gap (half of it for equal pieces).
        const w = dropped.area / pairArea;
        const shift: [number, number, number] = [0, 1, 2].map((i) => (dropped.center[i] - k.center[i]) * w) as [number, number, number];
        const widened = widen(pieces[keep].indices, positions, k, factor, shift, firstNewVertex + newSources.length, newPositions, newSources);
        error = Math.max(error, bestDistance * Math.max(w, 1 - w), widened.growth);
        total -= pieces[drop].indices.length / 3;
        pieces[keep].indices = widened.indices;
        pieces[drop].indices = new Uint32Array(0);
      }
    }

    const out = new Uint32Array(pieces.reduce((n, p) => n + p.indices.length, 0));
    let at = 0;
    for (const p of pieces) {
      out.set(p.indices, at);
      at += p.indices.length;
    }
    if (out.length >= indices.length) return null;
    return { indices: out, error, newPositions: Float32Array.from(newPositions), newSources: Uint32Array.from(newSources), firstNewVertex };
  }
}

/**
 * Copies a piece with its vertices moved apart across its width (or evenly in its plane) to scale its area, and
 * moved by `shift`.
 */
function widen(indices: Uint32Array, positions: Float32Array, piece: Piece, areaFactor: number, shift: [number, number, number], firstId: number, outPositions: number[], outSources: number[]) {
  const elongated = piece.extent[0] > piece.extent[1] * 2;
  // Long thin pieces widen across (axis 1); compact ones grow evenly along both in-plane axes.
  const scale = elongated ? [1, areaFactor] : [Math.sqrt(areaFactor), Math.sqrt(areaFactor)];
  const growth = Math.max((scale[0] - 1) * piece.extent[0], (scale[1] - 1) * piece.extent[1]) / 2;
  const map = new Map<number, number>();
  const out = new Uint32Array(indices.length);
  const c = piece.center;
  for (let i = 0; i < indices.length; i++) {
    const v = indices[i];
    let id = map.get(v);
    if (id === undefined) {
      id = firstId + map.size;
      map.set(v, id);
      const d = [positions[v * 3] - c[0], positions[v * 3 + 1] - c[1], positions[v * 3 + 2] - c[2]];
      const p = [positions[v * 3] + shift[0], positions[v * 3 + 1] + shift[1], positions[v * 3 + 2] + shift[2]];
      for (let a = 0; a < 2; a++) {
        const axis = piece.axes[a];
        const along = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
        for (let k = 0; k < 3; k++) p[k] += (scale[a] - 1) * along * axis[k];
      }
      outPositions.push(p[0], p[1], p[2]);
      outSources.push(v);
    }
    out[i] = id;
  }
  return { indices: out, growth };
}

/** Area, centroid, principal axes and extents of a piece. */
function describe(indices: Uint32Array, positions: Float32Array): Piece {
  const unique = [...new Set(indices)];
  const center: [number, number, number] = [0, 0, 0];
  for (const v of unique) for (let k = 0; k < 3; k++) center[k] += positions[v * 3 + k] / unique.length;
  const cov = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (const v of unique) {
    const d = [positions[v * 3] - center[0], positions[v * 3 + 1] - center[1], positions[v * 3 + 2] - center[2]];
    for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) cov[r * 3 + q] += d[r] * d[q];
  }
  const axes = eigenvectors(cov);
  const extent: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of unique) {
      const along = (positions[v * 3] - center[0]) * axes[a][0] + (positions[v * 3 + 1] - center[1]) * axes[a][1] + (positions[v * 3 + 2] - center[2]) * axes[a][2];
      lo = Math.min(lo, along);
      hi = Math.max(hi, along);
    }
    extent[a] = hi - lo;
  }
  let area = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3;
    const b = indices[t + 1] * 3;
    const c = indices[t + 2] * 3;
    const u = [positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]];
    const w = [positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2]];
    area += 0.5 * Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]);
  }
  return { area, center, axes, extent };
}

/** Eigenvectors of a symmetric 3x3 matrix (row-major), by decreasing eigenvalue (Jacobi rotations). */
function eigenvectors(m: number[]): [number, number, number][] {
  const a = [...m];
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 24; sweep++) {
    const off = Math.abs(a[1]) + Math.abs(a[2]) + Math.abs(a[5]);
    if (off < 1e-12 * (Math.abs(a[0]) + Math.abs(a[4]) + Math.abs(a[8]) + 1e-30)) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      const apq = a[p * 3 + q];
      if (Math.abs(apq) < 1e-30) continue;
      const theta = (a[q * 3 + q] - a[p * 3 + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k * 3 + p];
        const akq = a[k * 3 + q];
        a[k * 3 + p] = c * akp - s * akq;
        a[k * 3 + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p * 3 + k];
        const aqk = a[q * 3 + k];
        a[p * 3 + k] = c * apk - s * aqk;
        a[q * 3 + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k * 3 + p];
        const vkq = v[k * 3 + q];
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j * 3 + j] - a[i * 3 + i]);
  return order.map((i) => {
    const x = v[i];
    const y = v[3 + i];
    const z = v[6 + i];
    const l = Math.hypot(x, y, z) || 1;
    return [x / l, y / l, z / l] as [number, number, number];
  });
}

function compactLocal(indices: Uint32Array, positions: Float32Array) {
  const map = new Map<number, number>();
  const toGlobal: number[] = [];
  const local = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) {
    let l = map.get(indices[i]);
    if (l === undefined) {
      l = toGlobal.length;
      map.set(indices[i], l);
      toGlobal.push(indices[i]);
    }
    local[i] = l;
  }
  const localPositions = new Float32Array(toGlobal.length * 3);
  toGlobal.forEach((g, i) => localPositions.set(positions.subarray(g * 3, g * 3 + 3), i * 3));
  return { indices: local, positions: localPositions, toGlobal };
}

function boundsOf(points: [number, number, number][]) {
  const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let k = 0; k < 3; k++) {
      b[k] = Math.min(b[k], p[k]);
      b[k + 3] = Math.max(b[k + 3], p[k]);
    }
  }
  return b;
}

const distance = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** 30-bit Morton code of a point in [0, 1]^3 (clamped). */
function morton(p: number[]) {
  const spread = (x: number) => {
    let v = Math.min(1023, Math.max(0, Math.floor(x * 1024))) & 0x3ff;
    v = (v | (v << 16)) & 0x030000ff;
    v = (v | (v << 8)) & 0x0300f00f;
    v = (v | (v << 4)) & 0x030c30c3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
  };
  return (spread(p[0]) | (spread(p[1]) << 1) | (spread(p[2]) << 2)) >>> 0;
}
