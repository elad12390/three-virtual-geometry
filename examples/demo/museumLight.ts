/** Visible sunbeams: soft additive prisms from every skylight, oculus and window down along the sun's direction. */
import * as THREE from 'three/webgpu';
import { color, dot, float, length, mix, mx_fractal_noise_float, mx_noise_float, normalize, normalView, positionLocal, positionView, positionWorld, smoothstep, time, uv, vec3 } from 'three/tsl';
import type { V3 } from './museumKit';

type N = any; // eslint-disable-line @typescript-eslint/no-explicit-any

/** A convex opening: its outline in world space (all at the same height). */
export type Opening = V3[];

/** A rectangle in a frame: corners from local centre/half sizes, transformed by `matrix`, at local height `y`. */
export function rectOpening(matrix: THREE.Matrix4, cx: number, cz: number, hx: number, hz: number, y: number): Opening {
  const corners: [number, number][] = [
    [cx - hx, cz - hz],
    [cx + hx, cz - hz],
    [cx + hx, cz + hz],
    [cx - hx, cz + hz],
  ];
  return corners.map(([x, z]) => {
    const p = new THREE.Vector3(x, y, z).applyMatrix4(matrix);
    return [p.x, p.y, p.z] as V3;
  });
}

/** A circular opening (n-gon), world centre and radius. */
export function circleOpening(cx: number, y: number, cz: number, radius: number, sides = 16): Opening {
  const out: Opening = [];
  for (let i = 0; i < sides; i++) {
    const a = (i / sides) * Math.PI * 2;
    out.push([cx + Math.cos(a) * radius, y, cz + Math.sin(a) * radius]);
  }
  return out;
}

/**
 * One mesh with a prism per opening: the outline at the top, the same outline projected along `toSun` (pointing at the
 * sun) down to y = `floorY`. Faces fade towards the bottom; edge-on faces are transparent, so the beams have soft sides.
 */
export function buildShafts(openings: Opening[], toSun: THREE.Vector3, floorY = 0) {
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const dir = toSun.clone().normalize().negate(); // the way the light travels
  for (const outline of openings) {
    const n = outline.length;
    const top = outline;
    const bottom = outline.map((p) => {
      const t = (p[1] - floorY) / -dir.y;
      return [p[0] + dir.x * t, floorY, p[2] + dir.z * t] as V3;
    });
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const base = positions.length / 3;
      positions.push(...top[i], ...top[j], ...bottom[j], ...bottom[i]);
      uvs.push(0, 0, 1, 0, 1, 1, 0, 1);
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(indices);
  g.computeVertexNormals();
  return g;
}

export function shaftMaterial(intensity = 0.1) {
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
  const along = (uv() as N).y;
  // Strong near the opening, thinning towards the floor; edge-on faces vanish, so the beam has soft sides.
  const facing = (normalView as N).z.abs().pow(1.4);
  const dust = (mx_noise_float((positionWorld as N).mul(0.22).add(vec3(0, (time as N).mul(0.04), 0))) as N).mul(0.5).add(0.75);
  m.colorNode = color(1.0, 0.82, 0.55).mul(intensity);
  // Fades out close to the camera, so the air is not milky wherever you stand.
  const near = smoothstep(4, 22, length(positionView as N) as N);
  m.opacityNode = float(1).sub(along).pow(0.9).mul(facing).mul(dust).mul(near);
  return m;
}

/**
 * A soft daylight sky for the openings and the glazed court: a warm haze at the horizon, a muted blue above, a few
 * high clouds and a gentle glow round the sun. Kept below pure white so the openings do not clip.
 */
export function createSky(sunDirection: THREE.Vector3) {
  const material = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false, fog: false });
  const dir = normalize(positionLocal as N) as N;
  const h = dir.y.max(0);
  const gradient = mix(color(0.92, 0.88, 0.8), color(0.4, 0.58, 0.8), h.pow(0.45));
  const cloudCoord = vec3(dir.x, 0, dir.z).div(dir.y.max(0.12).add(0.25)).mul(2.2);
  const cloud = smoothstep(0.1, 0.65, (mx_fractal_noise_float(cloudCoord.add(vec3(7, 0, 3)), 4, 2, 0.5) as N).mul(0.5).add(0.5));
  const withClouds = mix(gradient, color(1, 0.98, 0.94), cloud.mul(0.55).mul(smoothstep(0.03, 0.3, dir.y)));
  const glow = dot(dir, vec3(sunDirection.x, sunDirection.y, sunDirection.z)).max(0).pow(48).mul(0.35);
  material.colorNode = withClouds.mul(0.85).add(color(1, 0.9, 0.7).mul(glow));
  const sky = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), material);
  sky.scale.setScalar(2500); // inside the camera's far plane (3000)
  sky.frustumCulled = false;
  sky.renderOrder = -1;
  return sky;
}
