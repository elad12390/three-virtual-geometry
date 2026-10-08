/**
 * Procedural high-poly assets for the map demo. Everything is generated at
 * load time, so the demo needs no downloads. Meshes are deliberately dense:
 * the point is to let the meshlet LOD hierarchy do the reduction.
 */
import * as THREE from 'three/webgpu';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { VirtualMeshSource } from '../../src/index';
import { mergeSources } from '../../src/source';
import { Perlin, mulberry32, smoothstep } from './noise';

function weld(geometry: THREE.BufferGeometry) {
  for (const name of Object.keys(geometry.attributes)) if (name !== 'position') geometry.deleteAttribute(name);
  return mergeVertices(geometry, 1e-5);
}

function displace(geometry: THREE.BufferGeometry, fn: (p: THREE.Vector3) => void) {
  const pos = geometry.attributes.position;
  const p = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    fn(p);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
}

function toSource(geometry: THREE.BufferGeometry): VirtualMeshSource {
  geometry.computeVertexNormals();
  return {
    positions: new Float32Array(geometry.attributes.position.array),
    normals: new Float32Array(geometry.attributes.normal.array),
    indices: new Uint32Array(geometry.index!.array),
  };
}

function vertexColors(mesh: VirtualMeshSource, fn: (p: THREE.Vector3, n: THREE.Vector3, out: THREE.Color) => void) {
  const count = mesh.positions.length / 3;
  const colors = new Float32Array(count * 3);
  const p = new THREE.Vector3();
  const n = new THREE.Vector3();
  const c = new THREE.Color();
  for (let i = 0; i < count; i++) {
    p.fromArray(mesh.positions, i * 3);
    n.fromArray(mesh.normals, i * 3);
    fn(p, n, c);
    colors.set([c.r, c.g, c.b], i * 3);
  }
  mesh.colors = colors;
  return mesh;
}

// ---------------------------------------------------------------------------

/** Boulder, ~1 unit radius, flattened bottom. `detail` 80 -> ~131k triangles. */
export function makeRock(seed: number, detail = 80): VirtualMeshSource {
  const noise = new Perlin(seed);
  const g = weld(new THREE.IcosahedronGeometry(1, detail));
  const d = new THREE.Vector3();
  displace(g, (p) => {
    d.copy(p).normalize();
    const big = noise.fbm3(d.x * 1.1 + 3, d.y * 1.1, d.z * 1.1, 3) * 0.38;
    const ridges = (1 - Math.abs(noise.fbm3(d.x * 3.2, d.y * 3.2 + 7, d.z * 3.2, 4))) * 0.16;
    const strata = Math.sin(d.y * 38 + noise.noise3(d.x * 4, d.y * 4, d.z * 4) * 3) * 0.012;
    const fine = noise.fbm3(d.x * 16, d.y * 16, d.z * 16, 3) * 0.035;
    p.copy(d).multiplyScalar(1 + big + ridges + strata + fine);
    p.y *= 0.65;
    if (p.y < -0.18) p.y = -0.18 + (p.y + 0.18) * 0.12;
  });
  const mesh = toSource(g);
  const rockA = new THREE.Color(0x77736b);
  const rockB = new THREE.Color(0x8d8273);
  const moss = new THREE.Color(0x56692f);
  return vertexColors(mesh, (p, n, c) => {
    c.copy(rockA).lerp(rockB, 0.5 + 0.5 * noise.noise3(p.x * 3, p.y * 3, p.z * 3));
    const mossy = smoothstep(0.55, 0.9, n.y) * smoothstep(-0.2, 0.35, noise.fbm3(p.x * 2.5, p.y * 2.5, p.z * 2.5, 3));
    c.lerp(moss, mossy * 0.85);
  });
}

/** Spruce-like conifer, ~16 units tall, ~30k triangles. */
export function makeConifer(seed: number): VirtualMeshSource {
  const noise = new Perlin(seed);
  const height = 16;

  const trunkH = height * 0.8;
  const trunk = weld(new THREE.CylinderGeometry(0.18, 0.42, trunkH, 14, 12).translate(0, trunkH / 2, 0));
  displace(trunk, (p) => {
    const s = 1 + 0.12 * noise.noise3(p.x * 6, p.y * 1.5, p.z * 6);
    p.x *= s;
    p.z *= s;
  });

  const parts: { mesh: VirtualMeshSource; color?: number }[] = [{ mesh: toSource(trunk), color: 0x4a3426 }];
  const tiers = 8;
  const dark = new THREE.Color(0x1d3b22);
  const light = new THREE.Color(0x3f6b34);
  for (let i = 0; i < tiers; i++) {
    const t = i / (tiers - 1);
    const radius = THREE.MathUtils.lerp(4.4, 0.9, t);
    const coneH = THREE.MathUtils.lerp(4.6, 2.8, t);
    const baseY = 2.6 + t * (height - 4.4);
    const cone = weld(new THREE.ConeGeometry(radius, coneH, 80, 16, false).translate(0, baseY + coneH / 2, 0));
    displace(cone, (p) => {
      const dist = Math.hypot(p.x, p.z);
      const angle = Math.atan2(p.z, p.x);
      const rel = THREE.MathUtils.clamp((p.y - baseY) / coneH, 0, 1);
      // branch tips: angular noise, stronger towards the outer rim
      const branches = noise.noise3(Math.cos(angle) * 3.2, Math.sin(angle) * 3.2, i * 1.37 + rel * 1.8);
      const fine = noise.noise3(p.x * 2.5, p.y * 2.5, p.z * 2.5) * 0.12;
      const scale = 1 + (0.42 * branches + fine) * (dist / radius);
      p.x *= scale;
      p.z *= scale;
      p.y -= Math.pow(dist / radius, 2) * 0.7; // drooping branches
    });
    const mesh = toSource(cone);
    vertexColors(mesh, (p, _n, c) => {
      const rel = THREE.MathUtils.clamp(Math.hypot(p.x, p.z) / radius, 0, 1);
      c.copy(dark).lerp(light, rel * 0.8 + 0.2 * noise.noise3(p.x, p.y, p.z));
    });
    parts.push({ mesh });
  }
  return mergeSources(parts);
}

/** Broadleaf tree, ~12 units tall, ~55k triangles. */
export function makeBroadleaf(seed: number): VirtualMeshSource {
  const noise = new Perlin(seed);
  const rand = mulberry32(seed);
  const bark = 0x5a4632;
  const parts: { mesh: VirtualMeshSource; color?: number }[] = [];

  const trunk = weld(new THREE.CylinderGeometry(0.32, 0.55, 7, 16, 10).translate(0, 3.5, 0));
  displace(trunk, (p) => {
    const s = 1 + 0.15 * noise.noise3(p.x * 4, p.y * 0.8, p.z * 4);
    p.x = p.x * s + Math.sin(p.y * 0.5) * 0.15;
    p.z *= s;
  });
  parts.push({ mesh: toSource(trunk), color: bark });

  for (let i = 0; i < 4; i++) {
    const angle = (i / 4) * Math.PI * 2 + rand();
    const branch = new THREE.CylinderGeometry(0.1, 0.22, 4.2, 8, 4).translate(0, 2.1, 0);
    branch.applyMatrix4(
      new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0.75 * Math.sin(angle), 0, 0.75 * Math.cos(angle)))
    );
    branch.translate(0, 5.5, 0);
    parts.push({ mesh: toSource(weld(branch)), color: bark });
  }

  const leafA = new THREE.Color(0x355e24);
  const leafB = new THREE.Color(0x6b8a2c);
  for (let i = 0; i < 9; i++) {
    const angle = rand() * Math.PI * 2;
    const r = i === 0 ? 0 : 1.8 + rand() * 1.8;
    const center = new THREE.Vector3(Math.cos(angle) * r, 8.8 + rand() * 2.6 - (r > 2.6 ? 1 : 0), Math.sin(angle) * r);
    const radius = 2.2 + rand() * 1.3;
    const blob = weld(new THREE.IcosahedronGeometry(radius, 16));
    const d = new THREE.Vector3();
    displace(blob, (p) => {
      d.copy(p).normalize();
      const k = 1 + 0.22 * noise.fbm3(d.x * 2 + i, d.y * 2, d.z * 2, 3) + 0.07 * noise.noise3(d.x * 14, d.y * 14, d.z * 14);
      p.copy(d).multiplyScalar(radius * k);
      p.y *= 0.8;
      p.add(center);
    });
    const tint = rand();
    const mesh = toSource(blob);
    vertexColors(mesh, (p, n, c) => {
      c.copy(leafA).lerp(leafB, THREE.MathUtils.clamp(0.25 + 0.4 * tint + 0.35 * n.y + 0.2 * noise.noise3(p.x, p.y, p.z), 0, 1));
    });
    parts.push({ mesh });
  }
  return mergeSources(parts);
}

/** Low bush made of several leafy blobs, ~12k triangles. */
export function makeBush(seed: number): VirtualMeshSource {
  const noise = new Perlin(seed);
  const rand = mulberry32(seed);
  const parts: { mesh: VirtualMeshSource }[] = [];
  const leafA = new THREE.Color(0x2f4f1f);
  const leafB = new THREE.Color(0x5d7a2a);
  for (let i = 0; i < 5; i++) {
    const angle = rand() * Math.PI * 2;
    const r = i === 0 ? 0 : 0.5 + rand() * 0.4;
    const center = new THREE.Vector3(Math.cos(angle) * r, 0.45 + rand() * 0.35, Math.sin(angle) * r);
    const radius = 0.55 + rand() * 0.35;
    const blob = weld(new THREE.IcosahedronGeometry(radius, 9));
    const d = new THREE.Vector3();
    displace(blob, (p) => {
      d.copy(p).normalize();
      const k = 1 + 0.25 * noise.fbm3(d.x * 2.5 + i, d.y * 2.5, d.z * 2.5, 3) + 0.08 * noise.noise3(d.x * 12, d.y * 12, d.z * 12);
      p.copy(d).multiplyScalar(radius * k);
      p.y *= 0.75;
      p.add(center);
    });
    const mesh = toSource(blob);
    vertexColors(mesh, (p, n, c) => c.copy(leafA).lerp(leafB, THREE.MathUtils.clamp(0.3 + 0.4 * n.y + 0.3 * noise.noise3(p.x * 3, p.y * 3, p.z * 3), 0, 1)));
    parts.push({ mesh });
  }
  return mergeSources(parts);
}

/** Clump of closed, bent grass blades (optionally with flowers). ~1.5k triangles. Build with `prune: true`. */
export function makeGrassClump(seed: number, blades = 56, flowers = 0): VirtualMeshSource {
  const rand = mulberry32(seed);
  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const root = new THREE.Color(0x1f3d12);
  const tip = new THREE.Color(0x8fae45);
  const c = new THREE.Color();
  const segments = 4;

  for (let b = 0; b < blades; b++) {
    const angle = rand() * Math.PI * 2;
    const dist = Math.sqrt(rand()) * 0.35;
    const bx = Math.cos(angle) * dist;
    const bz = Math.sin(angle) * dist;
    const h = 0.35 + rand() * 0.45;
    const width = 0.025 + rand() * 0.015;
    const bendDir = rand() * Math.PI * 2;
    const bend = 0.1 + rand() * 0.25;
    const start = positions.length / 3;

    for (let s = 0; s < segments; s++) {
      const t = s / segments;
      const w = width * (1 - t * 0.8);
      const off = bend * t * t;
      const cx = bx + Math.cos(bendDir) * off;
      const cz = bz + Math.sin(bendDir) * off;
      const y = h * t;
      for (let k = 0; k < 3; k++) {
        const a = bendDir + (k / 3) * Math.PI * 2;
        positions.push(cx + Math.cos(a) * w, y, cz + Math.sin(a) * w);
        c.copy(root).lerp(tip, t);
        colors.push(c.r, c.g, c.b);
      }
    }
    positions.push(bx + Math.cos(bendDir) * bend, h, bz + Math.sin(bendDir) * bend);
    c.copy(tip);
    colors.push(c.r, c.g, c.b);
    const tipIndex = start + segments * 3;

    indices.push(start, start + 1, start + 2); // base cap (facing down)
    for (let s = 0; s < segments - 1; s++) {
      for (let k = 0; k < 3; k++) {
        const a = start + s * 3 + k;
        const b2 = start + s * 3 + ((k + 1) % 3);
        indices.push(a, a + 3, b2, b2, a + 3, b2 + 3);
      }
    }
    for (let k = 0; k < 3; k++) {
      const s = start + (segments - 1) * 3;
      indices.push(s + k, tipIndex, s + ((k + 1) % 3));
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setIndex(indices);
  const mesh = toSource(g);
  mesh.colors = new Float32Array(colors);

  if (flowers === 0) return mesh;
  const petals = [0xf2f2f2, 0xf5d142, 0xb07ad9, 0xe46a6a];
  const parts: { mesh: VirtualMeshSource; color?: number }[] = [{ mesh }];
  for (let f = 0; f < flowers; f++) {
    const angle = rand() * Math.PI * 2;
    const dist = Math.sqrt(rand()) * 0.3;
    const head = weld(new THREE.IcosahedronGeometry(0.045, 2).scale(1, 0.5, 1));
    head.translate(Math.cos(angle) * dist, 0.55 + rand() * 0.25, Math.sin(angle) * dist);
    parts.push({ mesh: toSource(head), color: petals[Math.floor(rand() * petals.length)] });
  }
  return mergeSources(parts);
}

/** Square heightfield centered at the origin. */
export function makeTerrain(size: number, segments: number, heightAt: (x: number, z: number) => number): VirtualMeshSource {
  const g = new THREE.PlaneGeometry(size, size, segments, segments);
  g.rotateX(-Math.PI / 2);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, heightAt(pos.getX(i), pos.getZ(i)));
  return toSource(weld(g));
}
