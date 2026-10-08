/**
 * Groups meshlets into clusters of ~`groupSize` that share as many boundary
 * edges as possible. This stands in for METIS (the graph partitioner nanite-webgpu uses):
 * a greedy region-growing pass over the meshlet adjacency graph, seeded in
 * Morton order so groups stay spatially compact. Meshlets without enough
 * neighbours (e.g. disconnected leaves or grass blades) are then merged with
 * spatially close leftovers so they can still be simplified together.
 */

export interface PartitionInput {
  /** Boundary edge keys (position-welded vertex ids, see `edgeKey`). */
  boundaryEdges: number[];
  center: [number, number, number];
}

export function partitionMeshlets(meshlets: PartitionInput[], groupSize: number): number[][] {
  const count = meshlets.length;
  if (count <= groupSize) return [meshlets.map((_, i) => i)];

  // edge -> meshlets touching it
  const edgeOwners = new Map<number, number[]>();
  for (let i = 0; i < count; i++) {
    for (const e of meshlets[i].boundaryEdges) {
      const owners = edgeOwners.get(e);
      if (owners) {
        if (owners[owners.length - 1] !== i) owners.push(i);
      } else edgeOwners.set(e, [i]);
    }
  }

  // meshlet -> (neighbour -> shared edge count)
  const adjacency: Map<number, number>[] = Array.from({ length: count }, () => new Map());
  for (const owners of edgeOwners.values()) {
    for (let a = 0; a < owners.length; a++) {
      for (let b = a + 1; b < owners.length; b++) {
        const i = owners[a];
        const j = owners[b];
        adjacency[i].set(j, (adjacency[i].get(j) || 0) + 1);
        adjacency[j].set(i, (adjacency[j].get(i) || 0) + 1);
      }
    }
  }

  const order = mortonOrder(meshlets.map((m) => m.center));
  const assigned = new Uint8Array(count);
  const groups: number[][] = [];

  for (const seed of order) {
    if (assigned[seed]) continue;
    const group = [seed];
    assigned[seed] = 1;

    // frontier: candidate -> total shared edges with the group
    const frontier = new Map<number, number>();
    const addNeighbours = (m: number) => {
      for (const [n, shared] of adjacency[m]) {
        if (!assigned[n]) frontier.set(n, (frontier.get(n) || 0) + shared);
      }
    };
    addNeighbours(seed);

    while (group.length < groupSize && frontier.size > 0) {
      let best = -1;
      let bestScore = -1;
      let bestDist = Infinity;
      for (const [n, score] of frontier) {
        if (assigned[n]) continue;
        const d = dist2(meshlets[n].center, meshlets[seed].center);
        if (score > bestScore || (score === bestScore && d < bestDist)) {
          best = n;
          bestScore = score;
          bestDist = d;
        }
      }
      if (best < 0) break;
      frontier.delete(best);
      assigned[best] = 1;
      group.push(best);
      addNeighbours(best);
    }
    groups.push(group);
  }

  // Merge undersized groups with spatially close undersized groups.
  // `groups` is already in Morton order of their seeds.
  const result: number[][] = [];
  let pending: number[] = [];
  const minSize = Math.max(2, Math.floor(groupSize / 2));
  for (const group of groups) {
    if (group.length >= minSize) {
      result.push(group);
      continue;
    }
    if (pending.length + group.length > groupSize) {
      result.push(pending);
      pending = [];
    }
    pending.push(...group);
  }
  if (pending.length > 0) result.push(pending);
  return result;
}

function dist2(a: [number, number, number], b: [number, number, number]) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

/** Indices sorted along a 30-bit Morton curve of the given points. */
export function mortonOrder(points: [number, number, number][]): number[] {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], p[k]);
      max[k] = Math.max(max[k], p[k]);
    }
  }
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
  const codes = points.map((p) => {
    const q = (k: number) => Math.min(1023, Math.max(0, Math.floor(((p[k] - min[k]) / extent) * 1023)));
    return part1By2(q(0)) | (part1By2(q(1)) << 1) | (part1By2(q(2)) << 2);
  });
  return points.map((_, i) => i).sort((a, b) => codes[a] - codes[b]);
}

function part1By2(x: number) {
  x &= 0x3ff;
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}
